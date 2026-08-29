"""
core/pipeline.py

包含完整跑者動作分析 Pipeline 的排程與協調邏輯。
整合了：
  - Step 1 (track): YOLO 多相機追蹤與置中裁剪 (呼叫 core.tracking)
  - Step 2 (pose): HRNet 2D 姿態估計；完整流程會在 2D 左右腿修正後才執行 MotionAGFormer 3D 與角度計算
  - Step 3 (chart): 2D 追焦影片與角度折線圖合併 (呼叫 core.visualization)
  - Phase 3 (overlay): 原始未裁切影片之 2D 骨架與線條疊加 (呼叫 core.overlay)

提供一鍵分析介面 `run_analysis` 與完整排程介面 `run_pipeline`。
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core import tracking as tcr
from core.overlay import overlay_videos, overlay_videos_per_camera
from core.tracker_impl import compute_speed_from_bbox_map
from core.utils import REPO_ROOT, convert_to_web_compatible_mp4
from core.visualization import add_angle_overlay
from scripts.analysis.ankle_step_stride import (
    align_foot_keypoints_to_body,
    annotate_step_stride_video,
    apply_anchor_leg_correction,
    apply_leg_swap_to_foot_keypoints,
    refresh_step_analysis_after_leg_correction,
    run_step_stride_analysis,
    update_leg_swap_metadata,
)


def _record_timing(timings: list | None, stage: str, started_at: float, **meta) -> float:
    """Record and print elapsed time for one pipeline stage."""
    elapsed = time.perf_counter() - started_at
    row = {
        "stage": stage,
        "elapsed_sec": round(elapsed, 4),
    }
    if meta:
        row.update(meta)
    if timings is not None:
        timings.append(row)
    print(f"  ⏱ [TIME] {stage}: {elapsed:.2f}s")
    return elapsed


def _write_timing_report(timings: list, output_dest: str | None) -> str | None:
    """Write timing_report.json under the analysis output directory."""
    if not output_dest:
        return None
    os.makedirs(output_dest, exist_ok=True)
    report_path = os.path.join(output_dest, "timing_report.json")
    total = sum(item.get("elapsed_sec", 0.0) for item in timings)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime_now_iso(),
                "note": "elapsed_sec is wall-clock time. Parallel stages overlap, so summed elapsed_sec can exceed total runtime.",
                "timings": timings,
                "summed_stage_elapsed_sec": round(total, 4),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  ⏱ [TIME] timing report: {report_path}")
    return report_path


def datetime_now_iso() -> str:
    """Return local timestamp without adding a hard dependency to timezone packages."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


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


def _export_homography_review_videos(
    cameras,
    steps_csv: str | None,
    output_dest: str,
    timeline_video: str | None = None,
    schematic_enabled: bool = False,
    schematic_px_per_meter: float = 75.0,
    schematic_padding_px: int = 60,
    full_trial_px_per_meter: float = 30.0,
) -> list[str]:
    """Export calibrated camera review plus a stitched full-trial schematic.

    The normal camera overlays remain the source of truth for image-space pose.
    This companion output is deliberately per camera: it shows the same step
    anchors in world coordinates with distance labels, without pretending that
    independently calibrated cameras are one continuous raster image. The
    final-camera warp remains available for detailed inspection. When all
    participating cameras have six ground controls, a second output joins them
    on the pipeline's global distance and frame axes. That full-trial output is
    schematic by design: only ground-plane coordinates are transformed, never
    the runner pixels.
    """
    if not steps_csv or not os.path.exists(steps_csv):
        return []

    tool_path = REPO_ROOT / "scripts" / "tools" / "rectify_video_from_cone_points.py"
    schematic_tool_path = REPO_ROOT / "scripts" / "tools" / "render_schematic_topdown_review.py"
    outputs = []
    if not cameras:
        return outputs

    cam_idx = len(cameras) - 1
    camera = cameras[cam_idx]
    src = camera.get("homography_src_points")
    dst = camera.get("homography_dst_world")
    video_path = camera.get("video_path")
    if not video_path or not os.path.exists(video_path) or not isinstance(src, list) or not isinstance(dst, list):
        return outputs
    # Flutter's calibrated runway workflow defines six control points.
    # Requiring the complete set keeps the displayed metre scale consistent
    # with the same calibration used by step/landing analysis.
    if len(src) != 6 or len(dst) != 6:
        return outputs

    points_path = Path(output_dest) / f"topdown_review_cam{cam_idx + 1}_controls.json"
    points = [
        {
            "id": index + 1,
            "x": float(image_pt[0]),
            "y": float(image_pt[1]),
            "world_x_m": float(world_pt[0]),
            "world_y_m": float(world_pt[1]),
        }
        for index, (image_pt, world_pt) in enumerate(zip(src, dst))
    ]
    points_path.write_text(json.dumps({"points": points}, ensure_ascii=False, indent=2), encoding="utf-8")

    generic_output = Path(output_dest) / "homography_rectified_preview.mp4"
    final_output = Path(output_dest) / f"cam{cam_idx + 1}_topdown_review.mp4"
    try:
        subprocess.run(
            [
                sys.executable, str(tool_path),
                "--video", str(video_path),
                "--points-json", str(points_path),
                "--output-dir", str(output_dest),
                "--control-count", str(len(points)),
                "--px-per-meter", "100",
                "--padding-px", "40",
                "--max-frames", "0",
                "--step-events-csv", str(steps_csv),
                "--camera-index", str(cam_idx),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if generic_output.exists():
            os.replace(generic_output, final_output)
            outputs.append(str(final_output))
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  ▶ Cam {cam_idx + 1} 俯視回顧影片輸出失敗: {exc}")

    # Optional schematic review: equal-scale runway, projected contacts, and a
    # fixed-size runner glyph. It is deliberately disabled by default while
    # the visual treatment is being evaluated in the front-end workflow.
    if schematic_enabled:
        metrics_path = Path(output_dest) / "metrics.csv"
        schematic_output = Path(output_dest) / f"cam{cam_idx + 1}_topdown_schematic_review.mp4"
        if metrics_path.exists():
            try:
                subprocess.run(
                    [
                        sys.executable, str(schematic_tool_path),
                        "--video", str(video_path),
                        "--points-json", str(points_path),
                        "--metrics-csv", str(metrics_path),
                        "--step-events-csv", str(steps_csv),
                        "--output", str(schematic_output),
                        "--camera-index", str(cam_idx),
                        "--px-per-meter", str(schematic_px_per_meter),
                        "--padding-px", str(schematic_padding_px),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if schematic_output.exists():
                    outputs.append(str(schematic_output))
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"  ▶ Cam {cam_idx + 1} 俯視示意影片輸出失敗: {exc}")
        else:
            print(f"  ▶ 略過 Cam {cam_idx + 1} 俯視示意影片：找不到 {metrics_path}")

    metrics_path = Path(output_dest) / "metrics.csv"
    timeline_path = Path(timeline_video) if timeline_video else None
    complete_calibrations = []
    for index, current_camera in enumerate(cameras):
        current_src = current_camera.get("homography_src_points")
        current_dst = current_camera.get("homography_dst_world")
        if not isinstance(current_src, list) or not isinstance(current_dst, list):
            complete_calibrations = []
            break
        if len(current_src) != 6 or len(current_dst) != 6:
            complete_calibrations = []
            break
        complete_calibrations.append(
            {
                "camera_index": index,
                "image_points": [[float(point[0]), float(point[1])] for point in current_src],
                "world_points": [[float(point[0]), float(point[1])] for point in current_dst],
            }
        )

    if (
        complete_calibrations
        and timeline_path is not None
        and timeline_path.exists()
        and metrics_path.exists()
    ):
        calibrations_path = Path(output_dest) / "trial_topdown_calibrations.json"
        calibrations_path.write_text(
            json.dumps({"cameras": complete_calibrations}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        full_trial_output = Path(output_dest) / "trial_topdown_review.mp4"
        try:
            subprocess.run(
                [
                    sys.executable, str(schematic_tool_path),
                    "--video", str(timeline_path),
                    "--calibrations-json", str(calibrations_path),
                    "--metrics-csv", str(metrics_path),
                    "--step-events-csv", str(steps_csv),
                    "--output", str(full_trial_output),
                    "--px-per-meter", str(full_trial_px_per_meter),
                    "--padding-px", str(schematic_padding_px),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            if full_trial_output.exists():
                outputs.append(str(full_trial_output))
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"  ▶ 完整賽事俯視路徑影片輸出失敗: {exc}")
    else:
        print("  ▶ 略過完整賽事俯視路徑：所有鏡頭皆需 6 點校正、metrics 與完整影片")
    return outputs


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
    except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError) as e:
        print(f"  ▶ 補齊角度時間欄位失敗: {e}")
        return None


def _write_dp_leg_swap_mask(swapped_mask, output_dir: str | None = None,
                            pre_dp_swapped_mask=None, anchor_dp_swapped_mask=None):
    """Persist final raw-to-output swaps and optional component masks."""
    if swapped_mask is None or not output_dir:
        return None

    try:
        swapped = np.asarray(swapped_mask, dtype=bool)
        swap_csv = os.path.join(output_dir, "dp_leg_identity_swaps.csv")
        columns = {
            "frame": np.arange(len(swapped), dtype=int),
            "dp_leg_swapped": swapped,
        }
        if pre_dp_swapped_mask is not None:
            pre = np.asarray(pre_dp_swapped_mask, dtype=bool).reshape(-1)
            columns["pre_dp_leg_swapped"] = np.pad(
                pre[:len(swapped)], (0, max(0, len(swapped) - len(pre))))
        if anchor_dp_swapped_mask is not None:
            anchor = np.asarray(anchor_dp_swapped_mask, dtype=bool).reshape(-1)
            columns["anchor_dp_leg_swapped"] = np.pad(
                anchor[:len(swapped)], (0, max(0, len(swapped) - len(anchor))))
        pd.DataFrame(columns).to_csv(swap_csv, index=False)
        print(f"  ▶ DP 左右腿交換紀錄: {swap_csv}")
        return {
            "swap_csv": swap_csv,
            "swapped_frames": int(swapped.sum()),
        }
    except (OSError, ValueError, TypeError) as e:
        print(f"  ▶ 寫出 DP 左右腿交換紀錄失敗: {e}")
        return None


def _align_angle_csv_to_leg_identity(angle_csv_path: str | None, swapped_mask, output_dir: str | None = None):
    """Fallback: swap left/right leg angle columns wherever DP swapped 2D identity.

    Preferred flow is to run MotionAGFormer 3D + angle computation only after
    apply_anchor_leg_correction() has already rewritten input_2D/keypoints.npz.
    In that flow this fallback is not needed because angles are computed from
    the corrected 2D identities. Keep it available for older outputs where only
    a pre-DP angles.csv exists.
    """
    if not angle_csv_path or swapped_mask is None or not os.path.exists(angle_csv_path):
        _write_dp_leg_swap_mask(swapped_mask, output_dir)
        return None

    try:
        df = pd.read_csv(angle_csv_path)
        swapped = np.asarray(swapped_mask, dtype=bool)
        n = min(len(df), len(swapped))
        if n == 0 or not np.any(swapped[:n]):
            _write_dp_leg_swap_mask(swapped_mask, output_dir)
            return None

        leg_angle_pairs = [
            ("left_knee_angle", "right_knee_angle"),
            ("left_hip_angle", "right_hip_angle"),
        ]
        applied_pairs = []
        for left_col, right_col in leg_angle_pairs:
            if left_col not in df.columns or right_col not in df.columns:
                continue
            mask = swapped[:n]
            left_values = df.loc[:n - 1, left_col].copy()
            df.loc[:n - 1, left_col] = np.where(mask, df.loc[:n - 1, right_col], df.loc[:n - 1, left_col])
            df.loc[:n - 1, right_col] = np.where(mask, left_values, df.loc[:n - 1, right_col])
            applied_pairs.append((left_col, right_col))

        if not applied_pairs:
            return None

        df.to_csv(angle_csv_path, index=False)

        swap_info = _write_dp_leg_swap_mask(swapped_mask, output_dir)
        swap_csv = swap_info.get("swap_csv") if swap_info else None

        swapped_count = int(swapped[:n].sum())
        print(
            "  ▶ 已依 DP 左右腿身份修正同步角度 CSV: "
            f"{angle_csv_path} (swapped_frames={swapped_count}, pairs={applied_pairs})"
        )
        return {
            "angle_csv": angle_csv_path,
            "swap_csv": swap_csv,
            "swapped_frames": swapped_count,
            "applied_pairs": applied_pairs,
        }
    except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError) as e:
        print(f"  ▶ 同步 DP 左右腿身份到角度 CSV 失敗: {e}")
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


def _import_vis_module(motion_ag_dir: Path):
    """Import MotionAGFormer/demo/vis.py and return the module object."""
    import importlib.util

    demo_dir = str(motion_ag_dir / "demo")
    mag_dir = str(motion_ag_dir)

    if mag_dir not in sys.path:
        sys.path.insert(0, mag_dir)

    for p in [mag_dir, demo_dir]:
        if p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, mag_dir)
    sys.path.insert(0, demo_dir)

    spec = importlib.util.spec_from_file_location(
        "vis", str(motion_ag_dir / "demo" / "vis.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rerun_3d_angles_from_corrected_2d(
    video_path: str,
    pose_output_dir: str,
    output_dest: str,
    gpu: str,
    motion_ag_dir: Path,
    timings: list | None = None,
) -> str | None:
    """Regenerate pred_3D and angles.csv from the DP-corrected 2D keypoints.npz."""
    if not video_path or not pose_output_dir or not output_dest:
        return None
    kps_npz = os.path.join(pose_output_dir, "input_2D", "keypoints.npz")
    if not os.path.exists(kps_npz):
        print(f"  ▶ 無法重算 3D 角度，找不到修正後 keypoints: {kps_npz}")
        return None

    def _video_size(path: str) -> tuple[float, float]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return 0.0, 0.0
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0
        cap.release()
        return float(width), float(height)

    def _archive_tracked_video() -> str | None:
        pointer_path = os.path.join(pose_output_dir, "input_2D", "keypoints_raw_archive_dir.txt")
        if not os.path.exists(pointer_path):
            return None
        try:
            archive_dir = Path(pointer_path).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeError):
            return None
        candidate = Path(archive_dir) / "cam1_tracked.mp4"
        return str(candidate) if candidate.exists() else None

    try:
        recon = np.load(kps_npz, allow_pickle=True)["reconstruction"]
        xy = recon[0, :, :, :2] if recon.ndim == 4 else recon[:, :, :2]
        kp_max_x = float(np.nanmax(xy[..., 0]))
        kp_max_y = float(np.nanmax(xy[..., 1]))
        video_w, video_h = _video_size(video_path)
        # keypoints.npz is stored in the tracked/cropped video coordinate system.
        # If a later manual rerun passes the overview/final 1280x720 video after
        # cam1_tracked.mp4 has been cleaned up, MotionAGFormer normalizes the 128px
        # keypoints with 1280px dimensions and produces near-straight 3D knees.
        if video_w > 0 and video_h > 0 and (kp_max_x < video_w * 0.35 or kp_max_y < video_h * 0.35):
            archived_video = _archive_tracked_video()
            if archived_video:
                archived_w, archived_h = _video_size(archived_video)
                if archived_w > 0 and archived_h > 0:
                    print(
                        "  ▶ 偵測到 3D 重算影片尺寸與 keypoints 座標系不一致，"
                        f"改用 archive tracked video: {archived_video} "
                        f"({video_w:.0f}x{video_h:.0f} -> {archived_w:.0f}x{archived_h:.0f})"
                    )
                    video_path = archived_video
            else:
                print(
                    "  ▶ 警告：3D 重算影片尺寸可能與 keypoints 座標系不一致，"
                    f"video={video_w:.0f}x{video_h:.0f}, keypoints max=({kp_max_x:.1f},{kp_max_y:.1f})"
                )
    except (OSError, ValueError, TypeError, KeyError, IndexError, cv2.error) as e:
        print(f"  ▶ 檢查 3D 重算影片尺寸失敗，繼續使用原影片: {e}")

    _saved_argv = sys.argv[:]
    _saved_cwd = os.getcwd()
    sys.argv = [sys.argv[0]]
    os.chdir(str(motion_ag_dir))
    try:
        started_at = time.perf_counter()
        vis_mod = _import_vis_module(motion_ag_dir)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        vis_mod.get_pose3D(video_path, pose_output_dir, skip_video=True)
        _record_timing(timings, "Analysis/rerun_3d_angles_after_leg_dp", started_at)
    finally:
        os.chdir(_saved_cwd)
        sys.argv = _saved_argv

    angle_csv_orig = os.path.join(
        pose_output_dir,
        "pred_3D",
        "angles",
        f"{Path(pose_output_dir).name}_angles.csv",
    )
    if not os.path.exists(angle_csv_orig):
        print(f"  ▶ 重算後角度 CSV 不存在: {angle_csv_orig}")
        return None

    angle_csv_dest = os.path.join(output_dest, "angles.csv")
    try:
        shutil.copy2(angle_csv_orig, angle_csv_dest)
        print(f"  ▶ 已用 DP 修正後 2D keypoints 重算 3D 角度: {angle_csv_dest}")
        return angle_csv_dest
    except OSError as e:
        print(f"  ▶ 複製重算後角度 CSV 失敗: {e}")
        return angle_csv_orig


# -----------------------------------------------------------------------
# 各步驟實作函式
# -----------------------------------------------------------------------

def step1_track(cameras_cfg: list, extra_cfg: dict, gpu: str, output_dir: str,
                timings: list | None = None) -> str:
    """
    Step 1：執行 YOLO 多相機追蹤與跑者置中裁剪。
    直接調用 core.tracking 模組，不啟用額外子行程。
    """
    print("=" * 60)
    print("Step 1 — 多相機追蹤 + 人物置中裁剪 (Core.Tracking)")
    print("=" * 60)
    step_started_at = time.perf_counter()

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
    model_started_at = time.perf_counter()
    model = YOLO(tcr.MODEL_PATH)
    # Warmup 模型
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), device=tcr.DEVICE, verbose=False)
    _record_timing(timings, "Step1/load_yolo_model_and_warmup", model_started_at,
                   model_path=str(tcr.MODEL_PATH))

    first_fps = caps[0].get(cv2.CAP_PROP_FPS) or 60.0

    frame_map_name = output_name.replace('.mp4', '_frame_map.csv')
    frame_map_path = os.path.join(output_dir, frame_map_name)

    # online 模式沒有 Pass 1 可先選定主跑者，因此仍用 dry-run 估計 crop size。
    # two_pass 模式會在 Pass 1 選出主跑者並 stitch 後，改用該主跑者 bbox 決定尺寸。
    if tcr.TRACKING_MODE != 'two_pass' and tcr.AUTO_CROP:
        print("auto_crop：第一遍掃描（分析 bbox 大小）...")
        dryrun_started_at = time.perf_counter()
        _, _, dry_bw, dry_bh, _, _ = tcr._process_cameras(caps, CAMERAS, model, None, dry_run=True)
        _record_timing(timings, "Step1/online_auto_crop_dry_run", dryrun_started_at,
                       bbox_samples=len(dry_bw))
        caps = [cv2.VideoCapture(cam['video_path']) for cam in CAMERAS]
        if dry_bw and dry_bh:
            crop_side = tcr._auto_crop_side_from_bbox_sizes(dry_bw, dry_bh)
            tcr.CROP_WIDTH = crop_side
            tcr.CROP_HEIGHT = crop_side
            print(f"  自動設定裁剪尺寸: {tcr.CROP_WIDTH} x {tcr.CROP_HEIGHT}（auto square）")

    if tcr.TRACKING_MODE == 'two_pass':
        prescan_started_at = time.perf_counter()
        frame_ranges_by_cam = tcr.run_temporal_prescan(CAMERAS, output_dir=output_dir) if tcr.PRESCAN_ENABLED else None
        _record_timing(timings, "Step1/prescan_person_frames", prescan_started_at,
                       enabled=bool(tcr.PRESCAN_ENABLED))
        print("two_pass 模式：第一遍收集所有候選人軌跡...")
        caps_pass1 = [cv2.VideoCapture(cam['video_path']) for cam in CAMERAS]
        pass1_started_at = time.perf_counter()
        all_detections, frame_cache = tcr._collect_all_detections(
            caps_pass1, CAMERAS, model, frame_ranges_by_cam=frame_ranges_by_cam
        )
        _record_timing(timings, "Step1/two_pass_pass1_collect_detections", pass1_started_at,
                       detections=len(all_detections), cached_frames=len(frame_cache))
        for c in caps_pass1:
            c.release()
        print(f"  收集完成：共 {len(all_detections)} 筆偵測")
        first_cam_base = os.path.splitext(os.path.basename(CAMERAS[0]['video_path']))[0]
        select_started_at = time.perf_counter()
        preset_ids, summaries = tcr._score_and_select_runners(
            all_detections, CAMERAS, frame_ranges_by_cam=frame_ranges_by_cam
        )
        _record_timing(timings, "Step1/two_pass_select_main_runner", select_started_at,
                       selected_ids={str(k): int(v) for k, v in preset_ids.items()})
        stitch_started_at = time.perf_counter()
        tcr._stitch_target_id(frame_cache, preset_ids, CAMERAS, fps=first_fps)
        _record_timing(timings, "Step1/two_pass_stitch_target_id", stitch_started_at)
        debug_started_at = time.perf_counter()
        tcr._write_two_pass_debug(all_detections, summaries, preset_ids, first_cam_base)
        _record_timing(timings, "Step1/two_pass_write_debug_csv", debug_started_at)
        if tcr.AUTO_CROP:
            crop_started_at = time.perf_counter()
            crop_side, sel_bw, _sel_bh = tcr._auto_crop_from_selected_cache(frame_cache, preset_ids)
            _record_timing(timings, "Step1/two_pass_auto_crop_from_selected_bbox", crop_started_at,
                           bbox_samples=len(sel_bw), crop_side=crop_side)
            if crop_side:
                tcr.CROP_WIDTH = crop_side
                tcr.CROP_HEIGHT = crop_side
                print(f"  two_pass auto_crop: 使用已選主跑者 bbox 設定裁剪尺寸 "
                      f"{tcr.CROP_WIDTH} x {tcr.CROP_HEIGHT}（samples={len(sel_bw)}）")
            else:
                print("  警告：two_pass auto_crop 未收集到已選主跑者 bbox，沿用目前尺寸")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, first_fps,
                              (tcr.CROP_WIDTH, tcr.CROP_HEIGHT))
        print("two_pass 模式：第二遍輸出追焦影片（快取模式，跳過 YOLO）...")
        pass2_started_at = time.perf_counter()
        tcr._process_cameras(caps, CAMERAS, model, out, frame_map_path=frame_map_path,
                             preset_target_ids=preset_ids, frame_cache=frame_cache,
                             frame_ranges_by_cam=frame_ranges_by_cam)
        _record_timing(timings, "Step1/two_pass_pass2_write_tracked_video", pass2_started_at,
                       output_path=output_path,
                       crop_width=int(tcr.CROP_WIDTH),
                       crop_height=int(tcr.CROP_HEIGHT))
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, first_fps,
                              (tcr.CROP_WIDTH, tcr.CROP_HEIGHT))
        online_started_at = time.perf_counter()
        tcr._process_cameras(caps, CAMERAS, model, out, frame_map_path=frame_map_path)
        _record_timing(timings, "Step1/online_tracking_write_tracked_video", online_started_at,
                       output_path=output_path,
                       crop_width=int(tcr.CROP_WIDTH),
                       crop_height=int(tcr.CROP_HEIGHT))
    out.release()

    print(f"\nStep 1 完成，置中裁剪影片儲存至：{output_path}\n")
    _record_timing(timings, "Step1/total_tracking", step_started_at,
                   output_path=output_path)
    return output_path


def step2_pose(tracked_video_path: str, output_base_dir: str,
                only_2d: bool, gpu: str, motion_ag_dir: Path, skip_video: bool = False,
                timings: list | None = None, pose_model_path: str | None = None) -> str:
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
    for folder_name in ['pose2D', 'pose3D', 'pose', 'pred_3D']:
        folder_path = os.path.join(abs_output_dir, folder_name)
        if os.path.exists(folder_path):
            print(f"  [Step 2] 偵測到舊的 {folder_name} 資料夾，進行清理以避免新舊影格污染...")
            try:
                shutil.rmtree(folder_path)
            except OSError as e:
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
        pose_started_at = time.perf_counter()
        run_pose_estimation(
            video_path=abs_video_path,
            output_dir=abs_output_dir,
            only_2d=only_2d,
            gpu=gpu,
            bbox_csv=bbox_csv_path,
            skip_video=skip_video,
            model_path=pose_model_path,
        )
        _record_timing(timings, "Step2/pose_estimation_total", pose_started_at,
                       video_path=abs_video_path,
                       output_dir=abs_output_dir,
                       only_2d=bool(only_2d),
                       skip_video=bool(skip_video))
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
        print("  ⚠️  [Step 3] 角度 CSV 不存在，略過 Step 3 合併 (可能是 only_2d=True)")
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
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError) as e:
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

def run_pipeline(cameras: list, extra_cfg: dict | None = None, output_dir: str | None = None,
                 gpu: str = "0", only_2d: bool = False, skip_track: bool = False,
                 skip_video: bool = False, timings: list | None = None) -> dict:
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
        tracked_video = step1_track(cameras, extra_cfg, gpu, output_dir, timings=timings)
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
                tracked_video = os.path.join(output_dir, min(videos))
            else:
                raise FileNotFoundError("找不到已追蹤的影片，無法略過 Step 1")

    # 保存設定資訊供 Step 3 重構底圖使用
    config_dict = {"cameras": cameras}
    config_dict.update(extra_cfg)
    config_marker = os.path.join(output_dir, ".config.json")
    with open(config_marker, "w", encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)

    # Step 2: 2D/3D 姿態估計
    pose_dir = step2_pose(
        tracked_video, output_dir, only_2d, gpu, motion_ag_dir,
        skip_video=skip_video, timings=timings,
        pose_model_path=extra_cfg.get('pose_model_path'))

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
    timings = []
    analysis_started_at = time.perf_counter()
    timing_report_path = None

    if (
        "auto_crop" not in config_dict
        and "crop_width" not in config_dict
        and "crop_height" not in config_dict
    ):
        config_dict["auto_crop"] = True

    cameras = config_dict.get("cameras", [])
    tracking_cameras = []
    for cam in cameras:
        tracking_cam = dict(cam)
        # Step-length analysis can use four-point homography, but the historical
        # speed/distance metrics are based on start/end line scaling.  Keep the
        # tracking path on that calibration so existing velocity outputs do not
        # change when homography is enabled for step length.
        dst_world = tracking_cam.pop("homography_dst_world", None)
        tracking_cam.pop("homography_src_points", None)
        # 6-point homography cameras only carry start_line/end_line as a
        # tracking-ROI approximation (synthesized from the two extreme
        # world_x_m calibration points, see _camera_config_from_homography_
        # anchors() in the backend); they never set distance_m, since that
        # would make _make_calibration() prefer this line-projection scale
        # over the real homography for step length. Without distance_m here
        # though, compute_speed_from_bbox_map() gets no calibration at all
        # and the trial's velocity/speed metrics come back empty. Derive an
        # approximate distance_m from the same world_x_m span, scoped to
        # this stripped copy only, so speed metrics behave like the 4-point
        # line-projection case without touching step-length calibration.
        if tracking_cam.get("distance_m") is None and dst_world and tracking_cam.get("start_line") and tracking_cam.get("end_line"):
            world_xs = [pt[0] for pt in dst_world]
            span = max(world_xs) - min(world_xs)
            if span > 0:
                tracking_cam["distance_m"] = span
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
        stale_angle_csv = os.path.join(output_dest, "angles.csv")
        if os.path.exists(stale_angle_csv):
            try:
                os.remove(stale_angle_csv)
                print(f"  ▶ 已清除舊角度 CSV，等待 DP 後重新產生: {stale_angle_csv}")
            except OSError as e:
                print(f"  ▶ 清除舊角度 CSV 失敗，後續將嘗試覆蓋: {e}")

    extra_cfg = {k: v for k, v in config_dict.items() if k != "cameras"}
    motion_ag_dir = REPO_ROOT / "MotionAGFormer"

    print("=" * 60)
    print("【階段一/二】骨架追蹤 + 2D 姿態估計")
    print("=" * 60)

    if progress_callback: progress_callback(5)

    result = run_pipeline(
        cameras=tracking_cameras,
        extra_cfg=extra_cfg,
        output_dir=output_dest,
        gpu=gpu,
        # Full analysis needs step/DP correction before 3D lifting. The first
        # pose pass therefore always stops at 2D; final 3D/angles are generated
        # once after apply_anchor_leg_correction() writes corrected keypoints.
        only_2d=True,
        skip_track=skip_track,
        skip_video=True,  # 中間裁剪骨架影片僅用於姿態估計，不輸出
        timings=timings,
    )

    if progress_callback: progress_callback(70)

    track_out_dir = config_dict.get("output_dir", "output_cut")
    track_out_name = config_dict.get("output_name", "sequential_tracked.mp4")
    metrics_csv_dest = os.path.join(output_dest if output_dest else track_out_dir, "metrics.csv")

    # 速度分析：從骨架追蹤產出的 bbox_map.csv 直接計算，無需重跑 YOLO
    tracked_video = result.get('tracked_video')
    if not skip_track and tracked_video:
        speed_started_at = time.perf_counter()
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
                requested_speed_mode = str(config_dict.get(
                    "speed_mode",
                    "homography" if any(c.get("homography_src_points") for c in cameras) else "pixel",
                )).lower()
                all_track_data = compute_speed_from_bbox_map(
                    bbox_map_path, cameras, fps_override=fps_val,
                    offsets_npz=offsets_npz_path,
                    pixel_cameras_cfg_list=tracking_cameras,
                    speed_mode=requested_speed_mode)
                if all_track_data:
                    import csv as _csv_mod
                    with open(metrics_csv_dest, 'w', newline='', encoding='utf-8') as _f:
                        _w = _csv_mod.DictWriter(
                            _f, fieldnames=[
                                'cam', 'cam_frame', 'source_frame', 'absolute_frame',
                                'dist_m', 'dist_raw_m', 'dist_smooth_m',
                                'world_x', 'image_point_x', 'image_point_y',
                                'speed_mps', 'accel_mps2',
                                'speed_mode_used',
                                'dist_pixel_m', 'speed_pixel_mps', 'accel_pixel_mps2',
                                'dist_homography_m', 'speed_homography_mps', 'accel_homography_mps2',
                                'is_interpolated', 'interp_gap_len', 'speed_confidence'])
                        _w.writeheader()
                        _w.writerows(all_track_data)
                    print(f"  ▶ 速度分析完成，{len(all_track_data)} 幀 → {metrics_csv_dest}")
                else:
                    print("  ▶ 速度計算未產出資料（無 calibration 資訊或 bbox 不足）")
            # Speed analysis is optional; isolate failures from third-party
            # numerical and video code so the primary pose result survives.
            except Exception as _e:  # noqa: BLE001
                print(f"  ▶ 速度計算失敗: {_e}")
        else:
            print(f"  ▶ bbox_map.csv 不存在，速度分析略過: {bbox_map_path}")
        _record_timing(timings, "Analysis/speed_metrics_from_bbox_map", speed_started_at,
                       bbox_map_path=bbox_map_path,
                       metrics_csv=metrics_csv_dest)
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

    # Full analysis intentionally has no pre-DP angle CSV. Final angles are
    # generated after apply_anchor_leg_correction() writes corrected 2D keypoints.
    angle_csv_dest = None

    avg_step_length = None
    step_analysis = None

    # 【階段三 + 四a】依序：步頻分析 -> DP 修正 -> 3D/角度 -> 原比例骨架影片疊加
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
            print("【階段三/四a】步頻分析 → 骨架左右腳修正 → 骨架影片疊加")
            print("=" * 60)
            try:
                block_started_at = time.perf_counter()

                step_started_at = time.perf_counter()
                step_analysis = run_step_stride_analysis(
                    config=config_dict,
                    output_dir=output_dest,
                    make_video=False,
                )
                _record_timing(timings, "Analysis/step_stride_analysis", step_started_at,
                               detected_steps=step_analysis.get("detected_steps"))

                # 用已確定交替的落地點當錨點，回頭修正骨架的左右腳身份，
                # 疊圖影片才會用到修正後的 keypoints。
                leg_fix_started_at = time.perf_counter()
                anchor_swapped_mask = apply_anchor_leg_correction(
                    kps_npz, step_analysis["step_events"])
                _record_timing(timings, "Analysis/apply_anchor_leg_correction", leg_fix_started_at)

                status_npz = os.path.join(
                    final_pose_dir, "input_2D", "keypoint_status.npz")
                pre_dp_swapped_mask = None
                if os.path.exists(status_npz):
                    with np.load(status_npz, allow_pickle=True) as status_data:
                        if "pre_dp_leg_swap_mask" in status_data.files:
                            pre_dp_swapped_mask = np.asarray(
                                status_data["pre_dp_leg_swap_mask"], dtype=bool)
                            if pre_dp_swapped_mask.ndim == 2:
                                pre_dp_swapped_mask = pre_dp_swapped_mask[0]

                swapped_mask = update_leg_swap_metadata(
                    kps_npz, pre_dp_swapped_mask, anchor_swapped_mask)
                foot_npz = os.path.join(final_pose_dir, "input_2D", "foot_keypoints.npz")
                apply_leg_swap_to_foot_keypoints(foot_npz, swapped_mask)
                raw_kps_npz = os.path.join(final_pose_dir, "input_2D", "keypoints_raw.npz")
                align_foot_keypoints_to_body(foot_npz, raw_kps_npz, kps_npz, swapped_mask)

                swap_info = _write_dp_leg_swap_mask(
                    swapped_mask,
                    output_dest,
                    pre_dp_swapped_mask=pre_dp_swapped_mask,
                    anchor_dp_swapped_mask=anchor_swapped_mask,
                )

                # 3D knee/hip angles must be computed from the same left/right
                # identity that the final 2D overlay uses. The first pose pass is
                # intentionally 2D-only; run 3D lift + angle computation once here,
                # after apply_anchor_leg_correction() has written the corrected
                # input_2D/keypoints.npz.
                if not only_2d:
                    recomputed_angle_csv = _rerun_3d_angles_from_corrected_2d(
                        video_path=tracked_video,
                        pose_output_dir=final_pose_dir,
                        output_dest=output_dest,
                        gpu=gpu,
                        motion_ag_dir=motion_ag_dir,
                        timings=timings,
                    )
                    if recomputed_angle_csv:
                        angle_csv_dest = recomputed_angle_csv
                        angle_fps = None
                        if cameras and cameras[0].get('video_path'):
                            cap = cv2.VideoCapture(cameras[0]['video_path'])
                            if cap.isOpened():
                                angle_fps = cap.get(cv2.CAP_PROP_FPS) or None
                            cap.release()
                        _add_time_to_angles_csv(angle_csv_dest, fps=angle_fps)
                    else:
                        angle_sync_started_at = time.perf_counter()
                        angle_sync = _align_angle_csv_to_leg_identity(
                            angle_csv_dest,
                            swapped_mask,
                            output_dir=output_dest,
                        )
                        _record_timing(timings, "Analysis/sync_angle_csv_to_leg_identity_fallback",
                                       angle_sync_started_at,
                                       swapped_frames=angle_sync.get("swapped_frames") if angle_sync else 0)
                elif swap_info:
                    _record_timing(timings, "Analysis/write_dp_leg_swap_mask", leg_fix_started_at,
                                   swapped_frames=swap_info.get("swapped_frames", 0))

                refresh_started_at = time.perf_counter()
                step_analysis = refresh_step_analysis_after_leg_correction(
                    step_analysis=step_analysis,
                    config=config_dict,
                    output_dir=output_dest,
                    keypoints_npz=kps_npz,
                    offsets_npz=offsets_npz,
                    foot_npz=foot_npz,
                )
                _record_timing(timings, "Analysis/refresh_step_analysis_after_leg_correction",
                               refresh_started_at)

                overlay_started_at = time.perf_counter()
                overlay_videos(
                    cameras=cameras,
                    offsets_npz=offsets_npz,
                    kps_npz=kps_npz,
                    output_video=output_uncropped,
                    config=config_dict,
                )
                _record_timing(timings, "Analysis/overlay_original_video", overlay_started_at,
                               output_video=output_uncropped)

                _record_timing(timings, "Analysis/step_and_overlay_block", block_started_at)
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
                annotate_started_at = time.perf_counter()
                annotate_step_stride_video(
                    input_video=output_uncropped,
                    output_video=tmp_uncropped,
                    ankle_rows=step_analysis["ankle_rows"],
                    step_events=step_analysis["step_events"],
                    avg_cadence_spm=step_analysis.get("avg_cadence_spm"),
                )
                os.replace(tmp_uncropped, output_uncropped)
                _record_timing(timings, "Analysis/annotate_step_stride_video", annotate_started_at,
                               output_video=output_uncropped)

                # 【階段四c】各相機獨立骨架+落地點疊圖影片（給跳遠三鏡頭回放頁面用，
                # 不是給前端合成影片用的 output_final.mp4，是三支各自獨立的檔案）
                per_camera_started_at = time.perf_counter()
                per_camera_output_paths = [
                    os.path.join(output_dest, f"cam{i + 1}_overlay.mp4")
                    for i in range(len(cameras))
                ]
                try:
                    overlay_videos_per_camera(
                        cameras=cameras,
                        offsets_npz=offsets_npz,
                        kps_npz=kps_npz,
                        ankle_rows=step_analysis["ankle_rows"],
                        step_events=step_analysis["step_events"],
                        output_paths=per_camera_output_paths,
                        config=config_dict,
                    )
                    for cam_output_path in per_camera_output_paths:
                        if os.path.exists(cam_output_path):
                            convert_to_web_compatible_mp4(cam_output_path)
                    _record_timing(timings, "Analysis/overlay_videos_per_camera", per_camera_started_at,
                                   output_videos=per_camera_output_paths)
                # Per-camera review videos are optional and call third-party
                # video code whose exception types are not part of its API.
                except Exception as e:  # noqa: BLE001
                    print(f"  ▶ 各相機獨立疊圖產生失敗（不影響主要分析結果）: {e}")

                # 【階段四d】Homography 場地俯視回顧。僅對完整地面控制點校正的
                # 相機輸出，避免沒有實際公尺座標的普通跑步 session 顯示
                # 看似精確、實際上不可靠的俯視圖。
                topdown_started_at = time.perf_counter()
                topdown_outputs = _export_homography_review_videos(
                    cameras=cameras,
                    steps_csv=step_analysis.get("steps_csv"),
                    output_dest=output_dest,
                    timeline_video=output_uncropped,
                    schematic_enabled=bool(config_dict.get("schematic_topdown_enabled", False)),
                    schematic_px_per_meter=float(
                        config_dict.get("schematic_topdown_px_per_meter", 75.0)
                    ),
                    schematic_padding_px=int(
                        config_dict.get("schematic_topdown_padding_px", 60)
                    ),
                    full_trial_px_per_meter=float(
                        config_dict.get("full_trial_topdown_px_per_meter", 30.0)
                    ),
                )
                _record_timing(
                    timings,
                    "Analysis/homography_topdown_review_videos",
                    topdown_started_at,
                    output_videos=topdown_outputs,
                )

                # 轉碼為 H.264 Web 相容格式
                print("\n  ▶ 正在將影片轉換為 Web 播放相容格式...")
                transcode_started_at = time.perf_counter()
                convert_to_web_compatible_mp4(output_uncropped)
                _record_timing(timings, "Analysis/transcode_web_compatible_mp4", transcode_started_at,
                               output_video=output_uncropped)
                print(f"  ▶ [Core.Pipeline] 網頁串流格式轉檔成功: {output_uncropped}")
                angle_time_started_at = time.perf_counter()
                _add_time_to_angles_csv(angle_csv_dest, video_path=output_uncropped)
                _record_timing(timings, "Analysis/update_angles_time_columns", angle_time_started_at,
                               angles_csv=angle_csv_dest)
                # Flutter front-end only reads output_final.mp4. Exporting every frame
                # to PNG is expensive and not needed for the current app flow.
                print("  ▶ 略過最終影片逐幀 PNG 輸出（前端未使用）")
                output_final_frames_dir = None
                archive_started_at = time.perf_counter()
                archived_output_final = _copy_output_final_to_keypoints_archive(
                    final_pose_dir,
                    output_uncropped,
                )
                _record_timing(timings, "Analysis/archive_output_final_video", archive_started_at,
                               archived_video=str(archived_output_final) if archived_output_final else None)
                if archived_output_final:
                    print(f"  ▶ 已複製 Web 相容影片到 keypoints archive: {archived_output_final}")

                # 清理中間裁剪追蹤影片
                print("\n  ▶ 正在清理中間過程影片...")
                if tracked_video and os.path.exists(tracked_video):
                    os.remove(tracked_video)
                    print(f"    - 已移除置中裁剪追蹤影片: {tracked_video}")

            # This optional export stage combines several external video and
            # analysis modules; its failure must not discard the main result.
            except Exception as e:  # noqa: BLE001
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
        except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError) as e:
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
        except (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError) as e:
            print(f"  [fallback] total_time 計算異常: {e}")

    if total_time is None:
        total_time = 0.0
        print("  [warning] total_time 無法取得，設為 0.0")

    if avg_velocity is None:
        avg_velocity = 0.0
    if avg_acceleration is None:
        avg_acceleration = 0.0

    _record_timing(timings, "Total/run_analysis", analysis_started_at,
                   output_dest=output_dest)
    timing_report_path = _write_timing_report(timings, output_dest)

    return {
        "metrics_csv": metrics_csv_dest,
        "angles_csv": angle_csv_dest,
        "uncropped_video": output_uncropped,
        "output_final_frames_dir": output_final_frames_dir,
        "timing_report": timing_report_path,
        "step_analysis": step_analysis,
        "total_time": total_time,
        "avg_velocity": avg_velocity,
        "avg_acceleration": avg_acceleration,
        "avg_step_length": avg_step_length
    }
