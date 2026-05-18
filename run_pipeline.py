"""
run_pipeline.py — 跑者分析完整流程（可 import 呼叫 / 可 CLI 執行）

流程：
  Step 1│ track_crop_roi   — YOLO 多相機追蹤，以最快跑者為中心裁剪，輸出單一合併影片
  Step 2│ vis.run_pose_estimation — 2D/3D 姿態估計 + 關節角度計算，輸出骨架影片與 CSV
  Step 3│ add_angle_overlay.add_angle_overlay — 將 2D 影片與 4 個角度折線圖合併為單一影片

直接呼叫範例（Python）：
    from run_pipeline import run_pipeline

    result = run_pipeline(
        cameras=[
            {"video_path": "/data/cam1.mp4", "crop": [0, 400, 1920, 800],
             "start_line": [[208, 715], [123, 725]],
             "end_line":   [[1760, 710], [1830, 718]]},
        ],
        output_dir="/data/results/",   # 最終輸出目錄（可選）
        gpu="0",
        only_2d=False,
        skip_track=False,
    )
    print(result["output_dir"])        # 最終輸出目錄
    print(result["tracked_video"])     # 追蹤後影片路徑

CLI 用法：
    python run_pipeline.py                       # 使用 track_crop_roi.py 內的硬編碼相機
    python run_pipeline.py --config cfg.yaml     # 從 YAML 讀取相機設定
    python run_pipeline.py --config-json '{...}' # 直接傳入 JSON 字串
    python run_pipeline.py --gpu 1 --2d_only
    python run_pipeline.py --skip-track          # 略過 Step 1（追蹤結果已存在）
"""

from __future__ import annotations

import sys
import os
import json
import shutil
import argparse
import resource
from pathlib import Path

# -----------------------------------------------------------------------
# 提升 RLIMIT_NPROC 軟限制，防止 PyTorch / YOLO 多執行緒 fork 超限
# -----------------------------------------------------------------------
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if _soft < _hard:
        resource.setrlimit(resource.RLIMIT_NPROC, (_hard, _hard))
except Exception:
    pass

# Configuration will be determined dynamically in run_pipeline()


# -----------------------------------------------------------------------
# 延遲 import：只在真正需要時才載入 GPU-heavy 的函式庫
# -----------------------------------------------------------------------
def _import_vis(motion_ag_dir: Path):
    """Import run_pose_estimation from vis.py（延遲載入）。

    vis.py 的 module-level import 需要兩個目錄同時在 sys.path：
      - MotionAGFormer/demo/  → from lib.preprocess / from lib.hrnet（lib/ 在 demo/ 下）
      - MotionAGFormer/       → from demo.lib.utils / from model.MotionAGFormer
    必須在 exec_module 之前設定，不能靠 vis.py 內部的 Path(__file__) 自行補。
    """
    import importlib.util

    demo_dir = str(motion_ag_dir / "demo")
    mag_dir  = str(motion_ag_dir)

    if mag_dir not in sys.path:
        sys.path.insert(0, mag_dir)

    # 確保兩個路徑都在 sys.path 最前面（移除舊的再重插）
    for p in [mag_dir, demo_dir]:
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, mag_dir)   # position 0: MotionAGFormer/
    sys.path.insert(0, demo_dir)  # position 0: MotionAGFormer/demo/ (優先，讓 lib/ 被找到)

    spec = importlib.util.spec_from_file_location(
        "vis", str(motion_ag_dir / "demo" / "vis.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_pose_estimation


def _import_overlay(overlay_script: Path):
    """Import add_angle_overlay function（延遲載入）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "add_angle_overlay", str(overlay_script)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.add_angle_overlay


# =======================================================================
# 各步驟 Python 函式
# =======================================================================

def step1_track(cameras_cfg: list, extra_cfg: dict, gpu: str, track_script: Path, output_dir: str) -> str:
    """
    Step 1：執行 YOLO 多相機追蹤與人物置中裁剪。

    直接在同一 Python 行程內呼叫 track_crop_roi 的邏輯，
    不再透過 subprocess。

    參數：
        cameras_cfg  list[dict]  相機設定列表（與 track_crop_roi.camera() 相容的 dict）
        extra_cfg    dict        其他全域設定（output_dir / auto_crop / crop_width 等）
        gpu          str         CUDA_VISIBLE_DEVICES 值

    回傳：
        output_video_path  str   追蹤輸出影片的完整路徑
    """
    import importlib.util
    import numpy as np
    import cv2
    import torch

    print("=" * 60)
    print("Step 1 — 多相機追蹤 + 人物置中裁剪")
    print("=" * 60)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # 動態載入 track_crop_roi 模組
    spec = importlib.util.spec_from_file_location(
        "track_crop_roi", str(track_script)
    )
    tcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcr)

    # 套用 extra_cfg 到模組全域常數
    if extra_cfg:
        if 'output_dir'          in extra_cfg: tcr.OUTPUT_DIR          = extra_cfg['output_dir']
        if 'crop_width'          in extra_cfg: tcr.CROP_WIDTH          = int(extra_cfg['crop_width'])
        if 'crop_height'         in extra_cfg: tcr.CROP_HEIGHT         = int(extra_cfg['crop_height'])
        if 'auto_crop'           in extra_cfg: tcr.AUTO_CROP           = bool(extra_cfg['auto_crop'])
        if 'show_overlay'        in extra_cfg: tcr.SHOW_OVERLAY        = bool(extra_cfg['show_overlay'])
        if 'movement_threshold'  in extra_cfg: tcr.MOVEMENT_THRESHOLD  = int(extra_cfg['movement_threshold'])
        if 'min_movement_frames' in extra_cfg: tcr.MIN_MOVEMENT_FRAMES = int(extra_cfg['min_movement_frames'])
        if 'stationary_decay'    in extra_cfg: tcr.STATIONARY_DECAY    = int(extra_cfg['stationary_decay'])
        if 'max_person_memory'   in extra_cfg: tcr.MAX_PERSON_MEMORY   = int(extra_cfg['max_person_memory'])

    # 建立相機清單
    CAMERAS = [tcr._build_camera_from_entry(e) for e in cameras_cfg]
    CAMERAS = [c for c in CAMERAS if c['video_path'] is not None]
    if not CAMERAS:
        raise ValueError("所有相機的 video_path 均為 None，請至少設定一台。")

    tcr.OUTPUT_DIR = output_dir
    OUTPUT_DIR = tcr.OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 開啟所有 VideoCapture
    caps = []
    for cam_idx, cam in enumerate(CAMERAS):
        cap = cv2.VideoCapture(cam['video_path'])
        if not cap.isOpened():
            for c in caps: c.release()
            raise ValueError(f"無法開啟相機 {cam_idx+1}: {cam['video_path']}")
        caps.append(cap)

    # 動態決定輸出檔名
    if extra_cfg and 'output_name' in extra_cfg:
        # 將 track_runners 的 .mp4 替換成 _cropped.mp4 以區分
        output_name = extra_cfg['output_name'].replace('.mp4', '_cropped.mp4')
    else:
        first_cam_base = os.path.splitext(os.path.basename(CAMERAS[0]['video_path']))[0]
        output_name    = f"{first_cam_base}_tracked.mp4"
        
    output_path    = os.path.join(OUTPUT_DIR, output_name)

    # 寫入 marker 檔
    marker_path = os.path.join(OUTPUT_DIR, ".last_output_name")
    with open(marker_path, "w") as f:
        f.write(output_name)

    # 載入 YOLO 模型
    from ultralytics import YOLO
    model = YOLO(tcr.MODEL_PATH)
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), device=tcr.DEVICE, verbose=False)

    # auto_crop 第一遍 dry-run
    if tcr.AUTO_CROP:
        print("auto_crop：第一遍掃描（分析 bbox）...")
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

    tcr._process_cameras(caps, CAMERAS, model, out)
    out.release()

    print(f"\nStep 1 完成，輸出影片：{output_path}\n")
    return output_path


def step2_pose(tracked_video_path: str, output_base_dir: str,
               only_2d: bool, gpu: str, motion_ag_dir: Path, skip_video: bool = False) -> str:
    """
    Step 2：執行 2D/3D 姿態估計 + 關節角度計算。

    直接 import vis.run_pose_estimation 呼叫，不再透過 subprocess。

    參數：
        tracked_video_path  str   Step 1 輸出的影片路徑
        output_base_dir     str   輸出根目錄；結果放在 {output_base_dir}/{video_stem}/
        only_2d             bool  True → 只跑 2D
        gpu                 str   CUDA device index

    回傳：
        output_dir  str   此次結果目錄
    """
    # 延遲載入，避免 import 時就吃 GPU 記憶體
    run_pose_estimation = _import_vis(motion_ag_dir)

    video_stem = Path(tracked_video_path).stem
    output_dir = os.path.join(output_base_dir, video_stem) + "/"
    os.makedirs(output_dir, exist_ok=True)
    
    abs_video_path = os.path.abspath(tracked_video_path)
    abs_output_dir = os.path.abspath(output_dir)

    print("=" * 60)
    print(f"Step 2 — 姿態估計（{'2D only' if only_2d else '2D + 3D + 角度'}）")
    print(f"  影片: {tracked_video_path}")
    print(f"  輸出: {output_dir}")
    print("=" * 60)

    # HRNet (lib/hrnet/gen_kpts.py) 有兩個問題：
    # 1. 內部用 argparse.parse_args() 直接讀 sys.argv，會看到 run_pipeline 的 --config 等參數
    # 2. 用相對路徑開啟 yaml（如 'demo/lib/hrnet/experiments/...'），相對於 MotionAGFormer/
    # 解法：呼叫前暫時清 sys.argv 並切換工作目錄至 MOTION_AG_DIR，完成後還原。
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
            skip_video=skip_video,
        )
    finally:
        os.chdir(_saved_cwd)
        sys.argv = _saved_argv

    print(f"\nStep 2 完成，輸出目錄：{output_dir}\n")
    return output_dir


def step3_overlay(pose_output_dir: str, video_stem: str, gpu: str, overlay_script: Path) -> str | None:
    """
    Step 3：將 2D 骨架影片與 4 個角度折線圖合併為單一影片。

    直接 import add_angle_overlay 呼叫，不再透過 subprocess。

    參數：
        pose_output_dir  str   Step 2 的輸出目錄（含 *_2D.mp4 與 pred_3D/angles/）
        video_stem       str   不含副檔名的影片名稱（用於定位檔案）
        gpu              str   CUDA device index（此步驟不使用 GPU，但保持介面一致）

    回傳：
        output_path  str | None   合併後影片路徑；被略過時回傳 None
    """
    video_2d = os.path.join(pose_output_dir, video_stem + "_2D.mp4")
    csv_path = os.path.join(pose_output_dir, "pred_3D", "angles",
                            video_stem + "_angles.csv")
    output   = os.path.join(pose_output_dir, video_stem + "_2D_angles.mp4")

    print("=" * 60)
    print("Step 3 — 2D 影片 + 角度折線圖合併")
    print("=" * 60)

    if not os.path.exists(csv_path):
        print(f"  ⚠  角度 CSV 不存在，略過 Step 3")
        print(f"     （若需角度圖請移除 only_2d=True 重新執行）")
        return None
    if not os.path.exists(video_2d):
        print(f"  ⚠  2D 影片不存在，略過 Step 3: {video_2d}")
        return None

    print(f"  2D 影片: {video_2d}")
    print(f"  角度 CSV: {csv_path}")
    print(f"  輸出: {output}")

    add_angle_overlay = _import_overlay(overlay_script)
    add_angle_overlay(video_path=video_2d, csv_path=csv_path, output_path=output)

    print(f"\nStep 3 完成：{output}\n")
    return output


# =======================================================================
# 主要公開介面
# =======================================================================

def run_pipeline(
    cameras: list = None,
    extra_cfg: dict = None,
    output_dir: str = None,
    gpu: str = "0",
    only_2d: bool = False,
    skip_track: bool = False,
    skip_video: bool = False,
    base_dir: str = None,
    track_script: str = None,
    motion_ag_dir: str = None,
    overlay_script: str = None,
    limit_threads: bool = True
) -> dict:
    """
    執行完整的跑者分析流程。

    參數：
        cameras      list[dict] | None
            相機設定列表。每個 dict 格式與 track_crop_roi.camera() 相容：
              {
                "video_path": "/data/cam1.mp4",
                "crop":       [0, 400, 1920, 800],   # 可省略
                "start_line": [[208, 715], [123, 725]],
                "end_line":   [[1760, 710], [1830, 718]],
              }
            None → 沿用 track_crop_roi.py 內的硬編碼 CAM1~CAM6。

        extra_cfg    dict | None
            覆蓋 track_crop_roi 全域常數，例如：
              {"auto_crop": True, "show_overlay": False}

        output_dir   str | None
            最終結果複製目的地；None → 結果留在 MotionAGFormer/demo/output/。

        gpu          str     CUDA_VISIBLE_DEVICES 值（預設 "0"）
        only_2d      bool    True → 只跑 2D，跳過 3D 與角度計算
        skip_track   bool    True → 略過 Step 1（追蹤結果已存在時使用）

    回傳 dict：
        {
            "output_dir":      str,        # 最終輸出目錄
            "tracked_video":   str,        # Step 1 追蹤影片路徑
            "pose_output_dir": str,        # Step 2 姿態估計輸出目錄
            "overlay_video":   str | None, # Step 3 合併影片路徑（only_2d 時為 None）
        }
    """
    if cameras is None:
        cameras = []   # track_crop_roi 模組將使用硬編碼設定
    if extra_cfg is None:
        extra_cfg = {}

    if limit_threads:
        _THREAD_ENV = {
            "OPENBLAS_NUM_THREADS":          "1",
            "OMP_NUM_THREADS":               "1",
            "MKL_NUM_THREADS":               "1",
            "NUMEXPR_NUM_THREADS":           "1",
            "GOMP_SPINCOUNT":                "0",
            "OPENCV_FFMPEG_CAPTURE_OPTIONS": "threads;1",
        }
        for _k, _v in _THREAD_ENV.items():
            os.environ.setdefault(_k, _v)

    if base_dir is None:
        base_dir_path = Path(__file__).resolve().parent
    else:
        base_dir_path = Path(base_dir)

    if track_script is None:
        track_script_path = base_dir_path / "scripts" / "tracking" / "track_crop_roi.py"
    else:
        track_script_path = Path(track_script)
        
    if motion_ag_dir is None:
        motion_ag_dir_path = base_dir_path / "MotionAGFormer"
    else:
        motion_ag_dir_path = Path(motion_ag_dir)
        
    if overlay_script is None:
        overlay_script_path = base_dir_path / "scripts" / "visualization" / "add_angle_overlay.py"
    else:
        overlay_script_path = Path(overlay_script)

    if not output_dir:
        if cameras and len(cameras) > 0 and cameras[0].get("video_path"):
            output_dir = str(Path(cameras[0]["video_path"]).parent)
        else:
            output_dir = str(base_dir_path / "output_cut")

    print("\n" + "=" * 60)
    print(f"run_pipeline — 跑者分析完整流程")
    print(f"  GPU: {gpu}")
    print(f"  模式: {'2D only' if only_2d else '2D + 3D + 角度'}")
    print(f"  Step 1: {'略過' if skip_track else '執行'}")
    if output_dir:
        print(f"  最終輸出: {output_dir}")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Step 1：追蹤（可略過）
    # ------------------------------------------------------------------
    if skip_track:
        # 從 marker 取得已存在的追蹤影片
        marker_dir = Path(output_dir)
        marker_path = marker_dir / ".last_output_name"
        if not marker_path.exists():
            raise FileNotFoundError(
                f"找不到 marker 檔：{marker_path}\n"
                "請先執行 Step 1（skip_track=False）。"
            )
        video_name = marker_path.read_text().strip()
        if not video_name:
            raise ValueError(f"Marker 檔內容為空：{marker_path}")
        tracked_video = str(marker_dir / video_name)
        if not os.path.exists(tracked_video):
            raise FileNotFoundError(f"找不到追蹤影片：{tracked_video}")
        print(f"Step 1 略過，使用既有追蹤影片：{tracked_video}\n")
    else:
        if cameras:
            # 從傳入的 cameras 清單執行
            tracked_video = step1_track(cameras, extra_cfg, gpu, track_script_path, output_dir)
        else:
            # 沿用硬編碼設定：以 subprocess 呼叫（保留 track_crop_roi 原有行為）
            import subprocess
            cmd = [sys.executable, str(track_script_path)]
            subprocess.run(
                cmd,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
                check=True,
            )
            marker_path = Path(output_dir) / ".last_output_name"
            video_name    = marker_path.read_text().strip()
            tracked_video = str(Path(output_dir) / video_name)

    video_stem = Path(tracked_video).stem

    # ------------------------------------------------------------------
    # Step 2：姿態估計
    # ------------------------------------------------------------------
    pose_base = output_dir
        
    pose_out_dir  = step2_pose(tracked_video, pose_base, only_2d, gpu, motion_ag_dir_path, skip_video=skip_video)

    # ------------------------------------------------------------------
    # Step 3：角度疊加
    # ------------------------------------------------------------------
    if not skip_video:
        overlay_video = step3_overlay(pose_out_dir, video_stem, gpu, overlay_script_path)
    else:
        overlay_video = None

    final_dir = pose_out_dir
    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"輸出目錄: {final_dir}")
    print(f"  {video_stem}_2D.mp4               ← 2D 骨架疊加影片")
    print(f"  {video_stem}.mp4                  ← 2D + 3D 並排影片")
    print(f"  {video_stem}_2D_angles.mp4        ← 2D + 4 角度折線圖")
    print(f"  pred_3D/angles/..._angles.csv     ← 各關節角度 CSV")
    print("=" * 60)
    # 機器可讀輸出（後端從 stdout 最後一行取得輸出目錄）
    print(f"PIPELINE_OUTPUT_DIR={final_dir}")

    return {
        "output_dir":      final_dir,
        "tracked_video":   tracked_video,
        "pose_output_dir": final_dir,
        "overlay_video":   overlay_video,
    }


# =======================================================================
# CLI 入口
# =======================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="一鍵跑完：追蹤 → 姿態估計 + 角度計算 → 角度疊加影片"
    )
    parser.add_argument("--gpu",         type=str, default="0",
                        help="CUDA GPU 編號（預設: 0）")
    parser.add_argument("--2d_only",     dest="two_d_only", action="store_true",
                        help="只跑 2D 骨架，跳過 3D 與角度計算")
    parser.add_argument("--skip-track",  dest="skip_track", action="store_true",
                        help="略過 Step 1（追蹤影片已存在時使用）")
    parser.add_argument("--config",      type=str, default=None,
                        help="相機設定 YAML 路徑")
    parser.add_argument("--config-json", dest="config_json", type=str, default=None,
                        help="相機設定 JSON 字串，直接傳入不需建立檔案")
    parser.add_argument("--output-dest", dest="output_dest", type=str, default=None,
                        help="最終輸出目錄（不指定則留在 demo/output/）")
    return parser.parse_args()


def main(default_cameras=None, default_extra_cfg=None):
    args = _parse_args()

    cameras   = default_cameras if default_cameras is not None else []
    extra_cfg = default_extra_cfg if default_extra_cfg is not None else {}

    # 解析相機設定
    if args.config_json:
        try:
            cfg = json.loads(args.config_json)
        except json.JSONDecodeError as e:
            print(f"錯誤：--config-json 格式錯誤：{e}")
            sys.exit(1)
        cameras   = cfg.get("cameras", [])
        extra_cfg = {k: v for k, v in cfg.items() if k != "cameras"}
        if not args.output_dest and "output_dest" in cfg:
            args.output_dest = cfg["output_dest"]

    elif args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cameras   = cfg.get("cameras", [])
        extra_cfg = {k: v for k, v in cfg.items() if k != "cameras"}
        if not args.output_dest and "output_dest" in cfg:
            args.output_dest = cfg["output_dest"]

    result = run_pipeline(
        cameras=cameras,
        extra_cfg=extra_cfg,
        output_dir=args.output_dest,
        gpu=args.gpu,
        only_2d=args.two_d_only,
        skip_track=args.skip_track,
    )


if __name__ == "__main__":
    # 您可以在這裡直接設定預設的相機參數供單獨執行時使用
    default_cameras = [
        {
            "video_path": "test/test/cam1.mov",
            "crop": [0, 400, 1920, 800],
            "start_line": [[208, 715], [123, 725]],
            "end_line": [[1760, 710], [1830, 718]],
            "distance_m": 20
        },
        # {
        #     "video_path": "test/test/cam2.mov",
        #     "crop": [0, 400, 1920, 800],
        #     "start_line": [[208, 715], [123, 725]],
        #     "end_line": [[1760, 710], [1830, 718]],
        #     "distance_m": 20
        # }
    ]
    main(default_cameras=default_cameras)
