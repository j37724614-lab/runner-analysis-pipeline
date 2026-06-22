"""
core/pipeline.py

包含完整跑者動作分析 Pipeline 的排程與協調邏輯。
整合了：
  - Step 1 (track): YOLO 多相機追蹤與置中裁剪 (呼叫 core.tracking)
  - Step 2 (pose): HRNet / MotionAGFormer 姿態估計與角度計算 (動態載入 vis.py)
  - Step 3 (chart): 2D 追焦影片與角度折線圖合併 (呼叫 core.visualization)
  - Phase 3 (overlay): 原始未裁切影片之 2D 骨架與線條疊加 (呼叫 core.overlay)

提供一鍵分析介面 `run_analysis` 與完整排程介面 `run_pipeline`。
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

from core.utils import REPO_ROOT, convert_to_web_compatible_mp4
from core import tracking as tcr
from core.visualization import add_angle_overlay
from core.overlay import overlay_videos
from core.tracker_impl import compute_speed_from_bbox_map
from scripts.analysis.ankle_step_stride import (
    annotate_step_stride_video,
    run_step_stride_analysis,
)


def _copy_output_final_to_keypoints_archive(final_pose_dir: str, output_video: str):
    """Copy the web-compatible final video into the raw-keypoints archive folder."""
    pointer_path = Path(final_pose_dir) / "input_2D" / "keypoints_raw_archive_dir.txt"
    if not pointer_path.exists():
        return None

    archive_dir = Path(pointer_path.read_text(encoding="utf-8").strip())
    if not archive_dir.exists() or not os.path.exists(output_video):
        return None

    copied_video = archive_dir / "output_final.mp4"
    shutil.copy2(output_video, copied_video)

    final_videos_json = archive_dir / "final_videos.json"
    final_videos = []
    if final_videos_json.exists():
        try:
            with open(final_videos_json, encoding="utf-8") as f:
                final_videos = json.load(f).get("final_videos", [])
        except (OSError, json.JSONDecodeError):
            final_videos = []

    copied_video_str = str(copied_video)
    if copied_video_str not in final_videos:
        final_videos.append(copied_video_str)
    with open(final_videos_json, "w", encoding="utf-8") as f:
        json.dump({"final_videos": final_videos}, f, ensure_ascii=False, indent=2)

    return copied_video


def _export_video_frames(video_path: str, output_root: str, folder_name: str = "final_video_frames"):
    """Export every frame of a final video as PNG under output_root/folder_name/video_stem."""
    if not video_path or not os.path.exists(video_path):
        return None

    video_stem = Path(video_path).stem
    frames_dir = Path(output_root) / folder_name / video_stem
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ▶ 無法開啟最終影片輸出逐幀 PNG: {video_path}")
        return None

    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = frames_dir / f"{frame_idx:06d}.png"
        if cv2.imwrite(str(frame_path), frame):
            written += 1
        frame_idx += 1

    cap.release()
    print(f"  ▶ 最終影片逐幀 PNG: {frames_dir} ({written} frames)")
    return str(frames_dir)


def _add_time_to_angles_csv(angle_csv_path: str | None, video_path: str | None = None, fps: float | None = None):
    """Add final-video-relative time columns to an angle CSV."""
    if not angle_csv_path or not os.path.exists(angle_csv_path):
        return None

    resolved_fps = fps
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps and video_fps > 0:
                resolved_fps = float(video_fps)
        cap.release()

    if not resolved_fps or resolved_fps <= 0:
        resolved_fps = 60.0

    try:
        df = pd.read_csv(angle_csv_path)
        if "frame" not in df.columns:
            print(f"  ▶ 角度 CSV 缺少 frame 欄位，略過時間欄位補齊: {angle_csv_path}")
            return None

        time_sec = df["frame"].astype(float) / resolved_fps
        for col in ["time_s", "time_sec"]:
            if col in df.columns:
                df[col] = time_sec
            else:
                df.insert(1, col, time_sec)

        leading_cols = [col for col in ["frame", "time_sec", "time_s"] if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in leading_cols]
        df = df[leading_cols + remaining_cols]

        df.to_csv(angle_csv_path, index=False)
        print(f"  ▶ 已補齊角度時間欄位: {angle_csv_path} (fps={resolved_fps:.3f})")
        return angle_csv_path
    except Exception as e:
        print(f"  ▶ 補齊角度時間欄位失敗: {e}")
        return None


# -----------------------------------------------------------------------
# 延遲 import：只在真正需要時才載入 GPU-heavy 的 3D 重建函式庫
# -----------------------------------------------------------------------
def _import_vis(motion_ag_dir: Path):
    """
    動態且延遲載入 MotionAGFormer/demo/vis.py。
    此舉能避免在不執行 3D 姿態估計的機器上（例如純 CPU 環境）提早載入 GPU-heavy 的 PyTorch。
    """
    import importlib.util

    demo_dir = str(motion_ag_dir / "demo")
    mag_dir  = str(motion_ag_dir)

    # 確保路徑存在於 sys.path
    if mag_dir not in sys.path:
        sys.path.insert(0, mag_dir)

    for p in [mag_dir, demo_dir]:
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, mag_dir)   # position 0: MotionAGFormer/
    sys.path.insert(0, demo_dir)  # position 0: MotionAGFormer/demo/ (讓 lib/ pre-process 能被正確載入)

    spec = importlib.util.spec_from_file_location(
        "vis", str(motion_ag_dir / "demo" / "vis.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_pose_estimation


# -----------------------------------------------------------------------
# 各步驟實作函式
# -----------------------------------------------------------------------

def step1_track(cameras_cfg: list, extra_cfg: dict, gpu: str, output_dir: str) -> str:
    """
    Step 1：執行 YOLO 多相機追蹤與跑者置中裁剪。
    直接調用 core.tracking 模組，不啟用額外子行程。
    """
    print("=" * 60)
    print("Step 1 — 多相機追蹤 + 人物置中裁剪 (Core.Tracking)")
    print("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # 套用 extra_cfg 到模組全域常數
    if extra_cfg:
        if 'output_dir'          in extra_cfg: tcr.OUTPUT_DIR          = extra_cfg['output_dir']
        if 'crop_width'          in extra_cfg: tcr.CROP_WIDTH          = int(extra_cfg['crop_width'])
        if 'crop_height'         in extra_cfg: tcr.CROP_HEIGHT         = int(extra_cfg['crop_height'])
        if 'auto_crop'           in extra_cfg: tcr.AUTO_CROP           = bool(extra_cfg['auto_crop'])
        if 'show_overlay'        in extra_cfg: tcr.SHOW_OVERLAY        = bool(extra_cfg['show_overlay'])
        if 'draw_bbox_overlay'   in extra_cfg: tcr.DRAW_BBOX_OVERLAY   = bool(extra_cfg['draw_bbox_overlay'])
        if 'movement_threshold'  in extra_cfg: tcr.MOVEMENT_THRESHOLD  = int(extra_cfg['movement_threshold'])
        if 'min_movement_frames' in extra_cfg: tcr.MIN_MOVEMENT_FRAMES = int(extra_cfg['min_movement_frames'])
        if 'stationary_decay'    in extra_cfg: tcr.STATIONARY_DECAY    = int(extra_cfg['stationary_decay'])
        if 'max_person_memory'   in extra_cfg: tcr.MAX_PERSON_MEMORY   = int(extra_cfg['max_person_memory'])
        if 'tracking_mode'       in extra_cfg: tcr.TRACKING_MODE       = str(extra_cfg['tracking_mode'])
        if 'prescan_enabled'     in extra_cfg: tcr.PRESCAN_ENABLED     = bool(extra_cfg['prescan_enabled'])
        if 'prescan_engine_path' in extra_cfg: tcr.PRESCAN_ENGINE_PATH = str(extra_cfg['prescan_engine_path'])
        if 'prescan_stride'      in extra_cfg: tcr.PRESCAN_STRIDE      = int(extra_cfg['prescan_stride'])
        if 'prescan_imgsz'       in extra_cfg: tcr.PRESCAN_IMGSZ       = int(extra_cfg['prescan_imgsz'])
        if 'prescan_conf'        in extra_cfg: tcr.PRESCAN_CONF        = float(extra_cfg['prescan_conf'])
        if 'prescan_iou'         in extra_cfg: tcr.PRESCAN_IOU         = float(extra_cfg['prescan_iou'])
        if 'prescan_buffer_sec'  in extra_cfg: tcr.PRESCAN_BUFFER_SEC  = float(extra_cfg['prescan_buffer_sec'])
        if 'prescan_max_gap_sec' in extra_cfg: tcr.PRESCAN_MAX_GAP_SEC = float(extra_cfg['prescan_max_gap_sec'])
        if 'prescan_use_grab'    in extra_cfg: tcr.PRESCAN_USE_GRAB    = bool(extra_cfg['prescan_use_grab'])

    # 建立並過濾有效相機清單
    CAMERAS = [tcr._build_camera_from_entry(e) for e in cameras_cfg]
    CAMERAS = [c for c in CAMERAS if c['video_path'] is not None]
    if not CAMERAS:
        raise ValueError("所有相機的 video_path 均為 None，請至少設定一台。")

    tcr.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 開啟影像讀取器
    caps = []
    for cam_idx, cam in enumerate(CAMERAS):
        cap = cv2.VideoCapture(cam['video_path'])
        if not cap.isOpened():
            for c in caps: c.release()
            raise ValueError(f"無法開啟相機 {cam_idx+1}: {cam['video_path']}")
        caps.append(cap)

    # 決定輸出影片名稱
    if extra_cfg and 'output_name' in extra_cfg:
        output_name = extra_cfg['output_name'].replace('.mp4', '_cropped.mp4')
    else:
        first_cam_base = os.path.splitext(os.path.basename(CAMERAS[0]['video_path']))[0]
        output_name    = f"{first_cam_base}_tracked.mp4"
        
    output_path = os.path.join(output_dir, output_name)

    # 寫入最後一次產生的檔名作為 marker
    marker_path = os.path.join(output_dir, ".last_output_name")
    with open(marker_path, "w") as f:
        f.write(output_name)

    # 載入 YOLO 模型
    from ultralytics import YOLO
    model = YOLO(tcr.MODEL_PATH)
    # Warmup 模型
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), device=tcr.DEVICE, verbose=False)

    # auto_crop 第一遍掃描
    if tcr.AUTO_CROP:
        print("auto_crop：第一遍掃描（分析 bbox 大小）...")
        _, _, dry_bw, dry_bh, _, _ = tcr._process_cameras(caps, CAMERAS, model, None, dry_run=True)
        caps = [cv2.VideoCapture(cam['video_path']) for cam in CAMERAS]
        if dry_bw and dry_bh:
            tcr.CROP_WIDTH  = int(np.median(dry_bw)) * 2
            tcr.CROP_HEIGHT = int(np.median(dry_bh)) * 2
            print(f"  自動設定裁剪尺寸: {tcr.CROP_WIDTH} x {tcr.CROP_HEIGHT}")

    first_fps = caps[0].get(cv2.CAP_PROP_FPS) or 60.0
    fourcc    = cv2.VideoWriter_fourcc(*'mp4v')
    out       = cv2.VideoWriter(output_path, fourcc, first_fps,
                                (tcr.CROP_WIDTH, tcr.CROP_HEIGHT))

    frame_map_name = output_name.replace('.mp4', '_frame_map.csv')
    frame_map_path = os.path.join(output_dir, frame_map_name)

    if tcr.TRACKING_MODE == 'two_pass':
        frame_ranges_by_cam = tcr.run_temporal_prescan(CAMERAS, output_dir=output_dir) if tcr.PRESCAN_ENABLED else None
        print("two_pass 模式：第一遍收集所有候選人軌跡...")
        caps_pass1 = [cv2.VideoCapture(cam['video_path']) for cam in CAMERAS]
        all_detections, frame_cache = tcr._collect_all_detections(
            caps_pass1, CAMERAS, model, frame_ranges_by_cam=frame_ranges_by_cam
        )
        for c in caps_pass1:
            c.release()
        print(f"  收集完成：共 {len(all_detections)} 筆偵測")
        first_cam_base = os.path.splitext(os.path.basename(CAMERAS[0]['video_path']))[0]
        preset_ids, summaries = tcr._score_and_select_runners(
            all_detections, CAMERAS, frame_ranges_by_cam=frame_ranges_by_cam
        )
        tcr._stitch_target_id(frame_cache, preset_ids, CAMERAS, fps=first_fps)
        tcr._write_two_pass_debug(all_detections, summaries, preset_ids, first_cam_base)
        print("two_pass 模式：第二遍輸出追焦影片（快取模式，跳過 YOLO）...")
        tcr._process_cameras(caps, CAMERAS, model, out, frame_map_path=frame_map_path,
                             preset_target_ids=preset_ids, frame_cache=frame_cache,
                             frame_ranges_by_cam=frame_ranges_by_cam)
    else:
        tcr._process_cameras(caps, CAMERAS, model, out, frame_map_path=frame_map_path)
    out.release()

    print(f"\nStep 1 完成，置中裁剪影片儲存至：{output_path}\n")
    return output_path


def step2_pose(tracked_video_path: str, output_base_dir: str,
                only_2d: bool, gpu: str, motion_ag_dir: Path, skip_video: bool = False) -> str:
    """
    Step 2：進行 2D/3D 姿態估計與關節角度分析。
    """
    run_pose_estimation = _import_vis(motion_ag_dir)

    video_stem = Path(tracked_video_path).stem
    output_dir = os.path.join(output_base_dir, video_stem) + "/"
    os.makedirs(output_dir, exist_ok=True)
    
    abs_video_path = os.path.abspath(tracked_video_path)
    abs_output_dir = os.path.abspath(output_dir)
    
    bbox_csv_path = abs_video_path.replace('.mp4', '_bbox_map.csv')
    if not os.path.exists(bbox_csv_path):
        bbox_csv_path = None

    # 清理舊資料夾殘留的 PNG，以防新舊影格個數不一致污染影片生成
    import shutil
    for folder_name in ['pose2D', 'pose3D', 'pose']:
        folder_path = os.path.join(abs_output_dir, folder_name)
        if os.path.exists(folder_path):
            print(f"  [Step 2] 偵測到舊的 {folder_name} 資料夾，進行清理以避免新舊影格污染...")
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                print(f"  ⚠️  [Step 2] 清理 {folder_name} 失敗: {e}")

    print("=" * 60)
    print(f"Step 2 — 姿態估計（{'2D only' if only_2d else '2D + 3D + 角度'}）")
    print(f"  影片: {tracked_video_path}")
    print(f"  輸出: {output_dir}")
    print("=" * 60)

    # 暫時改變 sys.argv 與工作目錄，防止 HRNet parser 與相對路徑報錯
    _saved_argv = sys.argv[:]
    _saved_cwd  = os.getcwd()
    sys.argv = [sys.argv[0]]
    os.chdir(str(motion_ag_dir))
    try:
        run_pose_estimation(
            video_path=abs_video_path,
            output_dir=abs_output_dir,
            only_2d=only_2d,
            gpu=gpu,
            bbox_csv=bbox_csv_path,
            skip_video=skip_video,
        )
    finally:
        os.chdir(_saved_cwd)
        sys.argv = _saved_argv

    print(f"\nStep 2 完成，骨架與角度數據輸出至：{output_dir}\n")
    return output_dir


def step3_overlay(pose_output_dir: str, video_stem: str, gpu: str) -> str | None:
    """
    Step 3：將 2D 骨架影片與 4 個角度折線圖合併為單一影片。
    """
    video_2d = os.path.join(pose_output_dir, video_stem + "_2D.mp4")
    csv_path = os.path.join(pose_output_dir, "pred_3D", "angles",
                            video_stem + "_angles.csv")
    output   = os.path.join(pose_output_dir, video_stem + "_2D_angles.mp4")

    print("=" * 60)
    print("Step 3 — 2D 影片 + 角度折線圖合併")
    print("=" * 60)

    if not os.path.exists(csv_path):
        print(f"  ⚠️  [Step 3] 角度 CSV 不存在，略過 Step 3 合併 (可能是 only_2d=True)")
        return None
    if not os.path.exists(video_2d):
        print(f"  ⚠️  [Step 3] 2D 骨架影片不存在: {video_2d}，略過")
        return None

    # frame_map CSV 用於跨相機角度平滑
    frame_map_path = os.path.dirname(video_2d)
    frame_map_file = os.path.join(os.path.dirname(frame_map_path), video_stem + "_frame_map.csv")
    if not os.path.exists(frame_map_file):
        frame_map_file = None

    # 解析原始相機清單供圖表底圖使用
    config_marker = os.path.join(os.path.dirname(frame_map_path), ".config.json")
    main_video_paths = []
    if os.path.exists(config_marker):
        try:
            with open(config_marker, 'r', encoding='utf-8') as f:
                saved_cfg = json.load(f)
            main_video_paths = [
                c['video_path']
                for c in saved_cfg.get('cameras', [])
                if c.get('video_path')
            ]
        except Exception as e:
            print(f"  [Step 3] 無法讀取 config 暫存: {e}")

    add_angle_overlay(
        video_path=video_2d,
        csv_path=csv_path,
        output_path=output,
        main_video_paths=main_video_paths,
        frame_map_path=frame_map_file,
        chart_height=200,
        display_height=340,
        inset_height_ratio=0.45,
        inset_margin=10,
        smooth_camera_boundary=True,
        boundary_blend_frames=30,
        dpi=100
    )
    return output


# -----------------------------------------------------------------------
# CLI/Python Orchestration API
# -----------------------------------------------------------------------

def run_pipeline(cameras: list, extra_cfg: dict = None, output_dir: str = None,
                 gpu: str = "0", only_2d: bool = False, skip_track: bool = False,
                 skip_video: bool = False) -> dict:
    """
    生物力學與骨架分析完整排程。
    """
    if extra_cfg is None:
        extra_cfg = {}
    if output_dir is None:
        output_dir = extra_cfg.get('output_dir', str(REPO_ROOT / "output_cut"))

    motion_ag_dir = REPO_ROOT / "MotionAGFormer"

    # Step 1: YOLO 多相機追蹤置中裁剪
    if not skip_track:
        tracked_video = step1_track(cameras, extra_cfg, gpu, output_dir)
    else:
        print("略過 Step 1，讀取上一次的輸出結果...")
        marker_path = os.path.join(output_dir, ".last_output_name")
        if os.path.exists(marker_path):
            with open(marker_path, "r") as f:
                output_name = f.read().strip()
            tracked_video = os.path.join(output_dir, output_name)
        else:
            # 嘗試搜尋資料夾下的影片
            videos = [f for f in os.listdir(output_dir) if f.endswith(".mp4") and not f.endswith("_2D.mp4")]
            if videos:
                tracked_video = os.path.join(output_dir, sorted(videos)[0])
            else:
                raise FileNotFoundError(f"找不到已追蹤的影片，無法略過 Step 1")

    # 保存設定資訊供 Step 3 重構底圖使用
    config_dict = {"cameras": cameras}
    config_dict.update(extra_cfg)
    config_marker = os.path.join(output_dir, ".config.json")
    with open(config_marker, "w", encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    # Step 2: 2D/3D 姿態估計
    pose_dir = step2_pose(tracked_video, output_dir, only_2d, gpu, motion_ag_dir, skip_video=skip_video)

    # Step 3: 折線圖圖表合併
    overlay_video = None
    if not skip_video:
        video_stem = Path(tracked_video).stem
        overlay_video = step3_overlay(pose_dir, video_stem, gpu)

    return {
        "output_dir": pose_dir,
        "tracked_video": tracked_video,
        "overlay_video": overlay_video
    }


def run_analysis(config_dict, gpu="0", only_2d=False, skip_track=False, output_dest=None, progress_callback=None):
    """
    一鍵啟動分析管線：運動表現 (Phase 1) -> 生物力學 (Phase 2) -> 原始影片疊加 (Phase 3) -> 網頁格式轉檔。
    """
    cameras = config_dict.get("cameras", [])
    tracking_cameras = []
    for cam in cameras:
        tracking_cam = dict(cam)
        # Step-length analysis can use four-point homography, but the historical
        # speed/distance metrics are based on start/end line scaling.  Keep the
        # tracking path on that calibration so existing velocity outputs do not
        # change when homography is enabled for step length.
        tracking_cam.pop("homography_src_points", None)
        tracking_cam.pop("homography_dst_world", None)
        tracking_cameras.append(tracking_cam)
    if not output_dest:
        output_dest = config_dict.get("output_dest")
    
    # 預設輸出目錄為第一台相機影片所在的目錄
    if not output_dest and cameras:
        for cam in cameras:
            video_path = cam.get("video_path")
            if video_path:
                output_dest = os.path.dirname(os.path.abspath(video_path))
                break
    
    if output_dest:
        os.makedirs(output_dest, exist_ok=True)
        config_dict["output_dir"] = output_dest
        
    extra_cfg = {k: v for k, v in config_dict.items() if k != "cameras"}

    print("=" * 60)
    print("【階段一/二】骨架追蹤 + 2D/3D 姿態估計")
    print("=" * 60)

    if progress_callback: progress_callback(5)

    result = run_pipeline(
        cameras=tracking_cameras,
        extra_cfg=extra_cfg,
        output_dir=output_dest,
        gpu=gpu,
        only_2d=only_2d,
        skip_track=skip_track,
        skip_video=True,  # 中間裁剪骨架影片僅用於姿態估計，不輸出
    )

    if progress_callback: progress_callback(70)

    track_out_dir = config_dict.get("output_dir", "output_cut")
    track_out_name = config_dict.get("output_name", "sequential_tracked.mp4")
    metrics_csv_dest = os.path.join(output_dest if output_dest else track_out_dir, "metrics.csv")

    # 速度分析：從骨架追蹤產出的 bbox_map.csv 直接計算，無需重跑 YOLO
    tracked_video = result.get('tracked_video')
    if not skip_track and tracked_video:
        video_stem_for_bbox = Path(tracked_video).stem
        bbox_map_path = os.path.join(output_dest, f"{video_stem_for_bbox}_bbox_map.csv")
        # offsets.npz is named after the first camera's video, not the tracked output
        first_cam_stem = Path(cameras[0]['video_path']).stem if cameras else video_stem_for_bbox
        offsets_npz_path = os.path.join(output_dest, f"{first_cam_stem}_offsets.npz")
        if os.path.exists(bbox_map_path):
            print("\n" + "=" * 60)
            print("【速度分析】從 bbox_map.csv 計算速度與加速度（無需重跑 YOLO）")
            print("=" * 60)
            try:
                fps_val = 60.0
                if cameras and cameras[0].get('video_path'):
                    _cap = cv2.VideoCapture(cameras[0]['video_path'])
                    if _cap.isOpened():
                        fps_val = _cap.get(cv2.CAP_PROP_FPS) or 60.0
                        _cap.release()
                all_track_data = compute_speed_from_bbox_map(
                    bbox_map_path, tracking_cameras, fps_override=fps_val,
                    offsets_npz=offsets_npz_path)
                if all_track_data:
                    import csv as _csv_mod
                    with open(metrics_csv_dest, 'w', newline='', encoding='utf-8') as _f:
                        _w = _csv_mod.DictWriter(
                            _f, fieldnames=[
                                'cam', 'cam_frame', 'source_frame', 'absolute_frame',
                                'dist_m', 'dist_raw_m', 'dist_smooth_m',
                                'world_x', 'image_point_x', 'image_point_y',
                                'speed_mps', 'accel_mps2',
                                'is_interpolated', 'interp_gap_len', 'speed_confidence'])
                        _w.writeheader()
                        _w.writerows(all_track_data)
                    print(f"  ▶ 速度分析完成，{len(all_track_data)} 幀 → {metrics_csv_dest}")
                else:
                    print("  ▶ 速度計算未產出資料（無 calibration 資訊或 bbox 不足）")
            except Exception as _e:
                print(f"  ▶ 速度計算失敗: {_e}")
        else:
            print(f"  ▶ bbox_map.csv 不存在，速度分析略過: {bbox_map_path}")
    else:
        print("  使用者指定 skip_track，略過速度分析。")

    if progress_callback: progress_callback(80)

    print("\n" + "=" * 60)
    final_pose_dir = result.get('output_dir', '未定義')

    # MotionAGFormer 輸出資料夾統一改名為 sequential_tracked
    expected_pose_dir = os.path.join(output_dest, "sequential_tracked")
    if os.path.exists(final_pose_dir) and final_pose_dir != expected_pose_dir:
        if os.path.exists(expected_pose_dir):
            shutil.rmtree(expected_pose_dir)
        os.rename(final_pose_dir, expected_pose_dir)
        final_pose_dir = expected_pose_dir

    print(f"  ▶ 姿態分析資料夾: {final_pose_dir}")

    # 將角度 CSV 提取到 output 目錄下，命名為 angles.csv
    angle_csv_dest = None
    if tracked_video:
        video_stem = Path(tracked_video).stem
        angle_csv_orig = os.path.join(final_pose_dir, "pred_3D", "angles", f"{video_stem}_angles.csv")
        if os.path.exists(angle_csv_orig) and output_dest:
            angle_csv_dest = os.path.join(output_dest, "angles.csv")
            try:
                os.rename(angle_csv_orig, angle_csv_dest)
                print(f"  ▶ 關節角度資料 (CSV): {angle_csv_dest}")
                angle_fps = None
                if cameras and cameras[0].get('video_path'):
                    cap = cv2.VideoCapture(cameras[0]['video_path'])
                    if cap.isOpened():
                        angle_fps = cap.get(cv2.CAP_PROP_FPS) or None
                    cap.release()
                _add_time_to_angles_csv(angle_csv_dest, fps=angle_fps)
            except Exception as e:
                print(f"  ▶ 關節角度資料 (CSV): {angle_csv_orig} (重新命名失敗: {e})")
        elif os.path.exists(angle_csv_orig):
            angle_csv_dest = angle_csv_orig
            print(f"  ▶ 關節角度資料 (CSV): {angle_csv_orig}")
            angle_fps = None
            if cameras and cameras[0].get('video_path'):
                cap = cv2.VideoCapture(cameras[0]['video_path'])
                if cap.isOpened():
                    angle_fps = cap.get(cv2.CAP_PROP_FPS) or None
                cap.release()
            _add_time_to_angles_csv(angle_csv_dest, fps=angle_fps)

    avg_step_length = None
    step_analysis = None

    # 【階段三 + 四a】並行：原比例骨架影片疊加 + 步頻分析
    output_uncropped = None
    output_final_frames_dir = None
    if cameras:
        orig_stem = Path(cameras[0]['video_path']).stem
        offsets_npz = os.path.join(output_dest, f"{orig_stem}_offsets.npz")
        kps_npz = os.path.join(final_pose_dir, "input_2D", "keypoints.npz")
        output_uncropped = os.path.join(output_dest, "output_final.mp4")

        if os.path.exists(offsets_npz) and os.path.exists(kps_npz):
            if progress_callback: progress_callback(90)
            print("\n" + "=" * 60)
            print("【階段三/四a】並行：骨架影片疊加 + 步頻分析")
            print("=" * 60)
            try:
                with ThreadPoolExecutor(max_workers=2) as _pool:
                    _fut_overlay = _pool.submit(
                        overlay_videos,
                        cameras=cameras,
                        offsets_npz=offsets_npz,
                        kps_npz=kps_npz,
                        output_video=output_uncropped,
                        config=config_dict,
                    )
                    _fut_step = _pool.submit(
                        run_step_stride_analysis,
                        config=config_dict,
                        output_dir=output_dest,
                        make_video=False,
                    )

                _fut_overlay.result()
                step_analysis = _fut_step.result()
                avg_step_length = step_analysis.get("avg_step_length_m")
                print(f"  ▶ 腳踝位置資料 (CSV): {step_analysis['ankle_csv']}")
                print(f"  ▶ 步伐事件資料 (CSV): {step_analysis['steps_csv']}")
                print(f"  ▶ 偵測步數: {step_analysis['detected_steps']}")
                if step_analysis.get("avg_cadence_spm") is not None:
                    print(f"  ▶ 平均步頻: {step_analysis['avg_cadence_spm']:.2f} steps/min")
                if avg_step_length is not None:
                    print(f"  ▶ 平均步幅: {avg_step_length:.2f} m")

                # 【階段四b】步伐標注（需 overlay 影片 + 步頻資料都完成才能執行）
                print("\n" + "=" * 60)
                print("【階段四b】步伐標注影片合成")
                print("=" * 60)
                tmp_uncropped = output_uncropped.replace(".mp4", "_tmp_steps.mp4")
                annotate_step_stride_video(
                    input_video=output_uncropped,
                    output_video=tmp_uncropped,
                    ankle_rows=step_analysis["ankle_rows"],
                    step_events=step_analysis["step_events"],
                    avg_cadence_spm=step_analysis.get("avg_cadence_spm"),
                )
                os.replace(tmp_uncropped, output_uncropped)

                # 轉碼為 H.264 Web 相容格式
                print("\n  ▶ 正在將影片轉換為 Web 播放相容格式...")
                convert_to_web_compatible_mp4(output_uncropped)
                print(f"  ▶ [Core.Pipeline] 網頁串流格式轉檔成功: {output_uncropped}")
                _add_time_to_angles_csv(angle_csv_dest, video_path=output_uncropped)
                output_final_frames_dir = _export_video_frames(
                    output_uncropped,
                    output_dest if output_dest else os.path.dirname(output_uncropped),
                )
                archived_output_final = _copy_output_final_to_keypoints_archive(
                    final_pose_dir,
                    output_uncropped,
                )
                if archived_output_final:
                    print(f"  ▶ 已複製 Web 相容影片到 keypoints archive: {archived_output_final}")

                # 清理中間裁剪追蹤影片
                print("\n  ▶ 正在清理中間過程影片...")
                if tracked_video and os.path.exists(tracked_video):
                    os.remove(tracked_video)
                    print(f"    - 已移除置中裁剪追蹤影片: {tracked_video}")

            except Exception as e:
                print(f"匯出未裁切影片失敗: {e}")

    print("\n" + "=" * 60)

    if progress_callback: progress_callback(100)
    
    total_time = None
    avg_velocity = None
    avg_acceleration = None

    # 計算統計指標並回傳
    if os.path.exists(metrics_csv_dest):
        try:
            df = pd.read_csv(metrics_csv_dest)
            if not df.empty:
                fps_val = 60.0
                if cameras and cameras[0].get('video_path'):
                    cap = cv2.VideoCapture(cameras[0]['video_path'])
                    if cap.isOpened():
                        fps_val = cap.get(cv2.CAP_PROP_FPS) or 60.0
                        cap.release()

                total_time = float((df["absolute_frame"].max() + 1) / fps_val)
                avg_velocity = float(df["speed_mps"].mean())
                avg_acceleration = float(df["accel_mps2"].mean())
        except Exception as e:
            print(f"指標計算異常: {e}")

    # Fallback: metrics.csv 缺失時從 frame_map CSV 估算 total_time，避免 DB NOT NULL 失敗
    if total_time is None:
        try:
            fps_val = 60.0
            if cameras and cameras[0].get('video_path'):
                cap = cv2.VideoCapture(cameras[0]['video_path'])
                if cap.isOpened():
                    fps_val = cap.get(cv2.CAP_PROP_FPS) or 60.0
                    cap.release()
            frame_map_csv = os.path.join(
                track_out_dir,
                track_out_name.replace(".mp4", "_frame_map.csv")
            )
            if os.path.exists(frame_map_csv):
                df_fm = pd.read_csv(frame_map_csv)
                if not df_fm.empty and "orig_frame" in df_fm.columns:
                    total_time = float((df_fm["orig_frame"].max() + 1) / fps_val)
                    print(f"  [fallback] total_time 從 frame_map 估算: {total_time:.2f}s")
        except Exception as e:
            print(f"  [fallback] total_time 計算異常: {e}")

    if total_time is None:
        total_time = 0.0
        print("  [warning] total_time 無法取得，設為 0.0")

    if avg_velocity is None:
        avg_velocity = 0.0
    if avg_acceleration is None:
        avg_acceleration = 0.0

    return {
        "metrics_csv": metrics_csv_dest,
        "angles_csv": angle_csv_dest,
        "uncropped_video": output_uncropped,
        "output_final_frames_dir": output_final_frames_dir,
        "step_analysis": step_analysis,
        "total_time": total_time,
        "avg_velocity": avg_velocity,
        "avg_acceleration": avg_acceleration,
        "avg_step_length": avg_step_length
    }
