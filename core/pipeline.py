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

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core import angle_csv_store, tracking
from core.overlay import overlay_videos, overlay_videos_per_camera
from core.process_runtime import (
    PROCESS_STATE_LOCK as _PROCESS_STATE_LOCK,
)
from core.process_runtime import (
    temporary_environment_variable as _temporary_environment_variable,
)
from core.tracker_impl import compute_speed_from_bbox_map
from core.tracking_runtime import (
    TrackingRuntimeOptions,
)
from core.tracking_runtime import (
    temporary_tracking_runtime as _temporary_tracking_runtime,
)
from core.utils import REPO_ROOT, convert_to_web_compatible_mp4
from core.visualization import AngleOverlayConfig, add_angle_overlay
from scripts.analysis.ankle_step_stride import (
    annotate_step_stride_video,
    apply_anchor_leg_correction,
    apply_foot_leg_correction,
    refresh_step_analysis_after_leg_correction,
    run_step_stride_analysis,
    update_leg_swap_metadata,
)

DEFAULT_VIDEO_FPS = 60.0
HOMOGRAPHY_CONTROL_POINT_COUNT = 6
KEYPOINT_VIDEO_SIZE_RATIO_THRESHOLD = 0.35
YOLO_WARMUP_FRAME_SHAPE = (480, 640, 3)
OUTPUT_SEPARATOR = "=" * 60
PROGRESS_ANALYSIS_STARTED = 5
PROGRESS_POSE_COMPLETED = 70
PROGRESS_SPEED_ANALYSIS_COMPLETED = 80
PROGRESS_LEG_IDENTITY_STARTED = 90
PROGRESS_ANALYSIS_COMPLETED = 100
SPEED_METRIC_FIELD_NAMES = (
    "cam",
    "cam_frame",
    "source_frame",
    "absolute_frame",
    "dist_m",
    "dist_raw_m",
    "dist_smooth_m",
    "world_x",
    "image_point_x",
    "image_point_y",
    "speed_mps",
    "accel_mps2",
    "speed_mode_used",
    "dist_pixel_m",
    "speed_pixel_mps",
    "accel_pixel_mps2",
    "dist_homography_m",
    "speed_homography_mps",
    "accel_homography_mps2",
    "is_interpolated",
    "interp_gap_len",
    "speed_confidence",
)
STALE_POSE_OUTPUT_DIRECTORIES = ("pose2D", "pose3D", "pose", "pred_3D")
LEG_ANGLE_COLUMN_PAIRS = (
    ("left_knee_angle", "right_knee_angle"),
    ("left_hip_angle", "right_hip_angle"),
)
# 讀寫角度／指標 CSV 時預期會遇到的錯誤類型
CSV_IO_ERRORS = (OSError, ValueError, TypeError, KeyError, pd.errors.ParserError)


class PoseScope(Enum):
    """指定姿態分析產生 2D，或同時產生 3D 與角度。"""

    TWO_D_ONLY = "2d_only"
    TWO_D_AND_3D = "2d_and_3d"


class TrackedVideoSource(Enum):
    """指定追蹤影片由本次產生，或沿用既有輸出。"""

    GENERATE = "generate"
    EXISTING_OUTPUT = "existing_output"


class VideoOutput(Enum):
    """指定是否產生耗時的影片輸出。"""

    GENERATE = "generate"
    OMIT = "omit"


@dataclass(frozen=True)
class HomographyReviewOptions:
    """保存俯視回顧影片的輸出路徑與呈現設定。"""

    output_dest: str
    timeline_video: str | None = None
    camera_schematic_output: VideoOutput = VideoOutput.OMIT
    camera_pixels_per_meter: float = 75.0
    padding_pixels: int = 60
    trial_pixels_per_meter: float = 30.0


@dataclass(frozen=True)
class HomographyReviewRequest:
    """描述一組相機與完整賽事的俯視回顧輸出。"""

    cameras: list
    steps_csv: str | None
    options: HomographyReviewOptions


@dataclass(frozen=True)
class CameraHomographyInputs:
    """保存單鏡頭 Homography 回顧共用的輸入。"""

    video_path: str
    camera_index: int
    points_path: Path
    control_count: int
    steps_csv: str


@dataclass(frozen=True)
class TrialHomographyInputs:
    """保存完整賽事 Homography 回顧需要的輸入。"""

    calibrations: list[dict[str, object]]
    steps_csv: str


@dataclass(frozen=True)
class HomographyRenderJob:
    """描述一項外部 Homography 影片渲染工作。"""

    tool_path: Path
    arguments: tuple[str, ...]
    output_path: Path
    failure_message: str
    generated_path: Path | None = None


@contextmanager
def _temporary_motion_agformer_runtime(motion_ag_dir: Path, gpu: str):
    """暫時切換 MotionAGFormer 所需的程序狀態，完成後完整還原。"""
    with _PROCESS_STATE_LOCK:
        original_argv = sys.argv[:]
        original_cwd = os.getcwd()
        original_sys_path = sys.path[:]
        demo_dir = str(motion_ag_dir / "demo")
        motion_ag_path = str(motion_ag_dir)

        try:
            sys.argv = [sys.argv[0]]
            os.chdir(motion_ag_path)
            sys.path[:] = [
                demo_dir,
                motion_ag_path,
                *(
                    path
                    for path in original_sys_path
                    if path not in {demo_dir, motion_ag_path}
                ),
            ]
            with _temporary_environment_variable("CUDA_VISIBLE_DEVICES", gpu):
                yield
        finally:
            os.chdir(original_cwd)
            sys.argv = original_argv
            sys.path[:] = original_sys_path


def _record_timing(
    timings: list | None, stage: str, started_at: float, **meta
) -> float:
    """記錄單一分析階段的耗時與附加資訊，並輸出到終端。"""
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
    """將所有階段耗時寫入輸出目錄的 timing_report.json。"""
    if not output_dest:
        return None
    os.makedirs(output_dest, exist_ok=True)
    report_path = os.path.join(output_dest, "timing_report.json")
    total = sum(item.get("elapsed_sec", 0.0) for item in timings)
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(
            {
                "generated_at": datetime_now_iso(),
                "note": "elapsed_sec is wall-clock time. Parallel stages overlap, so summed elapsed_sec can exceed total runtime.",
                "timings": timings,
                "summed_stage_elapsed_sec": round(total, 4),
            },
            report_file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  ⏱ [TIME] timing report: {report_path}")
    return report_path


def datetime_now_iso() -> str:
    """回傳不依賴額外時區套件的本機 ISO 格式時間。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _remove_stale_angle_csv(output_dest: str) -> None:
    """在新姿態結果成功產生後，移除上一輪的角度 CSV。"""
    stale_angle_csv = os.path.join(output_dest, "angles.csv")
    if not angle_csv_store.exists(stale_angle_csv):
        return
    try:
        angle_csv_store.remove(stale_angle_csv)
        print(f"  ▶ 已清除舊角度 CSV，等待 DP 後重新產生: {stale_angle_csv}")
    except OSError as error:
        print(f"  ▶ 清除舊角度 CSV 失敗，後續將嘗試覆蓋: {error}")


def _remove_intermediate_tracked_video(tracked_video: str | None) -> None:
    """明確移除完成分析後不再需要的追蹤影片。"""
    print("\n  ▶ 正在清理中間過程影片...")
    if tracked_video and os.path.exists(tracked_video):
        os.remove(tracked_video)
        print(f"    - 已移除置中裁剪追蹤影片: {tracked_video}")


def _copy_output_final_to_keypoints_archive(final_pose_dir: str, output_video: str):
    """將 Web 相容的最終影片複製到原始關鍵點封存目錄。"""
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
            with open(final_videos_json, encoding="utf-8") as archive_metadata_file:
                final_videos = json.load(archive_metadata_file).get(
                    "final_videos",
                    [],
                )
        except (OSError, json.JSONDecodeError):
            final_videos = []

    copied_video_str = str(copied_video)
    if copied_video_str not in final_videos:
        final_videos.append(copied_video_str)
    with open(final_videos_json, "w", encoding="utf-8") as archive_metadata_file:
        json.dump(
            {"final_videos": final_videos},
            archive_metadata_file,
            ensure_ascii=False,
            indent=2,
        )

    return copied_video


def _complete_homography_calibration(camera: dict, camera_index: int):
    """將一台相機的完整六點校正轉成可序列化資料。"""
    image_points = camera.get("homography_src_points")
    world_points = camera.get("homography_dst_world")
    if not isinstance(image_points, list) or not isinstance(world_points, list):
        return None
    if (
        len(image_points) != HOMOGRAPHY_CONTROL_POINT_COUNT
        or len(world_points) != HOMOGRAPHY_CONTROL_POINT_COUNT
    ):
        return None
    return {
        "camera_index": camera_index,
        "image_points": [[float(point[0]), float(point[1])] for point in image_points],
        "world_points": [[float(point[0]), float(point[1])] for point in world_points],
    }


def _write_homography_control_points(
    calibration: dict,
    output_dest: str,
) -> tuple[Path, list[dict[str, float | int]]]:
    """寫出單鏡頭校正工具需要的控制點 JSON。"""
    camera_index = int(calibration["camera_index"])
    points = [
        {
            "id": index + 1,
            "x": image_point[0],
            "y": image_point[1],
            "world_x_m": world_point[0],
            "world_y_m": world_point[1],
        }
        for index, (image_point, world_point) in enumerate(
            zip(calibration["image_points"], calibration["world_points"])
        )
    ]

    points_path = (
        Path(output_dest) / f"topdown_review_cam{camera_index + 1}_controls.json"
    )
    points_path.write_text(
        json.dumps({"points": points}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return points_path, points


def _run_homography_render(job: HomographyRenderJob) -> str | None:
    """執行外部渲染工具，驗證並發佈產生的影片。"""
    try:
        subprocess.run(
            (
                sys.executable,
                str(job.tool_path),
                *job.arguments,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        generated_path = job.generated_path or job.output_path
        if not generated_path.exists():
            return None
        if generated_path != job.output_path:
            os.replace(generated_path, job.output_path)
        return str(job.output_path)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"  ▶ {job.failure_message}: {error}")
        return None


def _export_rectified_camera_review(
    inputs: CameraHomographyInputs,
    options: HomographyReviewOptions,
) -> str | None:
    """輸出最後一台相機的透視校正回顧影片。"""
    output_dir = Path(options.output_dest)
    return _run_homography_render(
        HomographyRenderJob(
            tool_path=(
                REPO_ROOT
                / "scripts"
                / "tools"
                / "rectify_video_from_cone_points.py"
            ),
            arguments=(
                "--video",
                inputs.video_path,
                "--points-json",
                str(inputs.points_path),
                "--output-dir",
                options.output_dest,
                "--control-count",
                str(inputs.control_count),
                "--px-per-meter",
                "100",
                "--padding-px",
                "40",
                "--max-frames",
                "0",
                "--step-events-csv",
                inputs.steps_csv,
                "--camera-index",
                str(inputs.camera_index),
            ),
            generated_path=(
                output_dir / "homography_rectified_preview.mp4"
            ),
            output_path=(
                output_dir
                / f"cam{inputs.camera_index + 1}_topdown_review.mp4"
            ),
            failure_message=(
                f"Cam {inputs.camera_index + 1} 俯視回顧影片輸出失敗"
            ),
        )
    )


def _export_schematic_camera_review(
    inputs: CameraHomographyInputs,
    options: HomographyReviewOptions,
) -> str | None:
    """輸出單鏡頭等比例跑道示意影片。"""
    metrics_path = Path(options.output_dest) / "metrics.csv"
    schematic_output = (
        Path(options.output_dest)
        / f"cam{inputs.camera_index + 1}_topdown_schematic_review.mp4"
    )
    if not metrics_path.exists():
        print(
            f"  ▶ 略過 Cam {inputs.camera_index + 1} "
            f"俯視示意影片：找不到 {metrics_path}"
        )
        return None
    return _run_homography_render(
        HomographyRenderJob(
            tool_path=(
                REPO_ROOT
                / "scripts"
                / "tools"
                / "render_schematic_topdown_review.py"
            ),
            arguments=(
                "--video",
                inputs.video_path,
                "--points-json",
                str(inputs.points_path),
                "--metrics-csv",
                str(metrics_path),
                "--step-events-csv",
                inputs.steps_csv,
                "--output",
                str(schematic_output),
                "--camera-index",
                str(inputs.camera_index),
                "--px-per-meter",
                str(options.camera_pixels_per_meter),
                "--padding-px",
                str(options.padding_pixels),
            ),
            output_path=schematic_output,
            failure_message=(
                f"Cam {inputs.camera_index + 1} 俯視示意影片輸出失敗"
            ),
        )
    )


def _collect_trial_calibrations(cameras: list) -> list[dict[str, object]]:
    """只有全部相機皆有完整六點校正時才回傳校正集合。"""
    complete_calibrations: list[dict[str, object]] = []
    for index, current_camera in enumerate(cameras):
        calibration = _complete_homography_calibration(current_camera, index)
        if calibration is None:
            return []
        complete_calibrations.append(calibration)
    return complete_calibrations


def _prepare_full_trial_render_job(
    inputs: TrialHomographyInputs,
    options: HomographyReviewOptions,
) -> HomographyRenderJob | None:
    """驗證完整賽事輸入、寫出校正設定並建立渲染工作。"""
    metrics_path = Path(options.output_dest) / "metrics.csv"
    timeline_path = Path(options.timeline_video) if options.timeline_video else None
    if (
        not inputs.calibrations
        or timeline_path is None
        or not timeline_path.exists()
        or not metrics_path.exists()
    ):
        return None

    calibrations_path = Path(options.output_dest) / "trial_topdown_calibrations.json"
    calibrations_path.write_text(
        json.dumps(
            {"cameras": inputs.calibrations},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output_path = Path(options.output_dest) / "trial_topdown_review.mp4"
    return HomographyRenderJob(
        tool_path=(
            REPO_ROOT
            / "scripts"
            / "tools"
            / "render_schematic_topdown_review.py"
        ),
        arguments=(
            "--video",
            str(timeline_path),
            "--calibrations-json",
            str(calibrations_path),
            "--metrics-csv",
            str(metrics_path),
            "--step-events-csv",
            inputs.steps_csv,
            "--output",
            str(output_path),
            "--px-per-meter",
            str(options.trial_pixels_per_meter),
            "--padding-px",
            str(options.padding_pixels),
        ),
        output_path=output_path,
        failure_message="完整賽事俯視路徑影片輸出失敗",
    )


def _export_full_trial_topdown_review(
    inputs: TrialHomographyInputs,
    options: HomographyReviewOptions,
) -> str | None:
    """將所有鏡頭校正投影到同一時間與距離軸的示意影片。"""
    render_job = _prepare_full_trial_render_job(inputs, options)
    if render_job is None:
        print("  ▶ 略過完整賽事俯視路徑：所有鏡頭皆需 6 點校正、metrics 與完整影片")
        return None
    return _run_homography_render(render_job)


def _export_homography_review_videos(
    request: HomographyReviewRequest,
) -> list[str]:
    """協調單鏡頭與完整賽事的 Homography 回顧影片輸出。"""
    if (
        not request.steps_csv
        or not os.path.exists(request.steps_csv)
        or not request.cameras
    ):
        return []

    camera_index = len(request.cameras) - 1
    camera = request.cameras[camera_index]
    video_path = camera.get("video_path")
    calibration = _complete_homography_calibration(camera, camera_index)
    if not video_path or not os.path.exists(video_path) or calibration is None:
        return []

    points_path, points = _write_homography_control_points(
        calibration,
        request.options.output_dest,
    )
    camera_inputs = CameraHomographyInputs(
        video_path=video_path,
        camera_index=camera_index,
        points_path=points_path,
        control_count=len(points),
        steps_csv=request.steps_csv,
    )
    outputs: list[str] = []
    rectified_output = _export_rectified_camera_review(
        camera_inputs,
        request.options,
    )
    if rectified_output:
        outputs.append(rectified_output)

    if request.options.camera_schematic_output is VideoOutput.GENERATE:
        schematic_output = _export_schematic_camera_review(
            camera_inputs,
            request.options,
        )
        if schematic_output:
            outputs.append(schematic_output)

    trial_output = _export_full_trial_topdown_review(
        TrialHomographyInputs(
            calibrations=_collect_trial_calibrations(request.cameras),
            steps_csv=request.steps_csv,
        ),
        request.options,
    )
    if trial_output:
        outputs.append(trial_output)
    return outputs


@dataclass(frozen=True)
class AngleTimeRequest:
    """描述角度 CSV 時間欄位補齊工作。"""

    angle_csv_path: str | None
    video_path: str | None = None
    frames_per_second: float | None = None


def _resolve_angle_frames_per_second(request: AngleTimeRequest) -> float:
    """依明確設定、影片資訊與預設值解析角度資料 FPS。"""
    if request.video_path and os.path.exists(request.video_path):
        video_fps = _read_video_frames_per_second(request.video_path, None)
        if video_fps:
            return video_fps
    if request.frames_per_second and request.frames_per_second > 0:
        return request.frames_per_second
    return DEFAULT_VIDEO_FPS


def _add_angle_time_columns(
    angle_dataframe: pd.DataFrame,
    frames_per_second: float,
) -> pd.DataFrame:
    """回傳補齊時間欄位的新角度資料，不修改輸入。"""
    timed_dataframe = angle_dataframe.copy(deep=True)
    time_seconds = timed_dataframe["frame"].astype(float) / frames_per_second
    for time_column in ("time_s", "time_sec"):
        if time_column in timed_dataframe.columns:
            timed_dataframe[time_column] = time_seconds
        else:
            timed_dataframe.insert(1, time_column, time_seconds)
    leading_columns = [
        column
        for column in ("frame", "time_sec", "time_s")
        if column in timed_dataframe.columns
    ]
    remaining_columns = [
        column
        for column in timed_dataframe.columns
        if column not in leading_columns
    ]
    return timed_dataframe[leading_columns + remaining_columns]


def _add_time_to_angles_csv(request: AngleTimeRequest) -> str | None:
    """依影片 FPS 為角度 CSV 補上相對時間欄位。"""
    if not angle_csv_store.exists(request.angle_csv_path):
        return None
    assert request.angle_csv_path is not None
    resolved_fps = _resolve_angle_frames_per_second(request)

    try:
        angle_dataframe = angle_csv_store.read(request.angle_csv_path)
        if "frame" not in angle_dataframe.columns:
            print(
                "  ▶ 角度 CSV 缺少 frame 欄位，略過時間欄位補齊: "
                f"{request.angle_csv_path}"
            )
            return None
        timed_dataframe = _add_angle_time_columns(angle_dataframe, resolved_fps)
        angle_csv_store.write(request.angle_csv_path, timed_dataframe)
        print(
            f"  ▶ 已補齊角度時間欄位: {request.angle_csv_path} "
            f"(fps={resolved_fps:.3f})"
        )
        return request.angle_csv_path
    except CSV_IO_ERRORS as error:
        print(f"  ▶ 補齊角度時間欄位失敗: {error}")
        return None


@dataclass(frozen=True)
class LegSwapMaskRequest:
    """描述 DP 左右腿交換紀錄的輸出內容。"""

    swapped_mask: np.ndarray | None
    output_dir: str | None = None
    pre_dp_swapped_mask: np.ndarray | None = None
    anchor_dp_swapped_mask: np.ndarray | None = None


def _padded_boolean_mask(mask, target_length: int) -> np.ndarray:
    """將布林遮罩裁切或補齊至指定長度。"""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return np.pad(
        values[:target_length],
        (0, max(0, target_length - len(values))),
    )


def _leg_swap_mask_dataframe(request: LegSwapMaskRequest) -> pd.DataFrame:
    """建立 DP 左右腿交換紀錄，不進行檔案寫入。"""
    assert request.swapped_mask is not None
    swapped = np.asarray(request.swapped_mask, dtype=bool).reshape(-1)
    columns = {
        "frame": np.arange(len(swapped), dtype=int),
        "dp_leg_swapped": swapped,
    }
    if request.pre_dp_swapped_mask is not None:
        columns["pre_dp_leg_swapped"] = _padded_boolean_mask(
            request.pre_dp_swapped_mask,
            len(swapped),
        )
    if request.anchor_dp_swapped_mask is not None:
        columns["anchor_dp_leg_swapped"] = _padded_boolean_mask(
            request.anchor_dp_swapped_mask,
            len(swapped),
        )
    return pd.DataFrame(columns)


def _write_dp_leg_swap_mask(request: LegSwapMaskRequest):
    """保存 DP 左右腿交換結果及可選的分階段遮罩。"""
    if request.swapped_mask is None or not request.output_dir:
        return None

    try:
        swapped = np.asarray(request.swapped_mask, dtype=bool).reshape(-1)
        swap_csv = os.path.join(
            request.output_dir,
            "dp_leg_identity_swaps.csv",
        )
        _leg_swap_mask_dataframe(request).to_csv(swap_csv, index=False)
        print(f"  ▶ DP 左右腿交換紀錄: {swap_csv}")
        return {
            "swap_csv": swap_csv,
            "swapped_frames": int(swapped.sum()),
        }
    except (OSError, ValueError, TypeError) as error:
        print(f"  ▶ 寫出 DP 左右腿交換紀錄失敗: {error}")
        return None


@dataclass(frozen=True)
class AngleAlignmentResult:
    """保存角度欄位交換後的資料與套用摘要。"""

    dataframe: pd.DataFrame
    swapped_frames: int
    applied_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class AngleCsvAlignmentRequest:
    """描述既有角度 CSV 與腿部身份遮罩的同步工作。"""

    angle_csv_path: str | None
    swapped_mask: np.ndarray | None
    output_dir: str | None = None


def _align_angle_dataframe_to_leg_identity(
    angle_dataframe: pd.DataFrame,
    swapped_mask,
) -> AngleAlignmentResult:
    """依左右腿交換遮罩轉換角度資料，不讀寫檔案或修改輸入。"""
    aligned_dataframe = angle_dataframe.copy(deep=True)
    swapped = np.asarray(swapped_mask, dtype=bool).reshape(-1)
    aligned_frame_count = min(len(aligned_dataframe), len(swapped))
    active_swap_mask = swapped[:aligned_frame_count]
    applied_pairs: list[tuple[str, str]] = []

    if aligned_frame_count and np.any(active_swap_mask):
        for left_column, right_column in LEG_ANGLE_COLUMN_PAIRS:
            if (
                left_column not in aligned_dataframe.columns
                or right_column not in aligned_dataframe.columns
            ):
                continue
            left_index = aligned_dataframe.columns.get_loc(left_column)
            right_index = aligned_dataframe.columns.get_loc(right_column)
            left_values = aligned_dataframe.iloc[
                :aligned_frame_count,
                left_index,
            ].copy()
            right_values = aligned_dataframe.iloc[
                :aligned_frame_count,
                right_index,
            ].copy()
            aligned_dataframe.iloc[:aligned_frame_count, left_index] = np.where(
                active_swap_mask,
                right_values,
                left_values,
            )
            aligned_dataframe.iloc[:aligned_frame_count, right_index] = np.where(
                active_swap_mask,
                left_values,
                right_values,
            )
            applied_pairs.append((left_column, right_column))

    return AngleAlignmentResult(
        dataframe=aligned_dataframe,
        swapped_frames=int(active_swap_mask.sum()),
        applied_pairs=tuple(applied_pairs),
    )


def _align_angle_csv_to_leg_identity(
    request: AngleCsvAlignmentRequest,
) -> dict[str, object] | None:
    """在 DP 交換 2D 腿部身份的影格同步交換左右腿角度欄位。

    Preferred flow is to run MotionAGFormer 3D + angle computation only after
    apply_anchor_leg_correction() has already rewritten input_2D/keypoints.npz.
    In that flow this fallback is not needed because angles are computed from
    the corrected 2D identities. Keep it available for older outputs where only
    a pre-DP angles.csv exists.
    """
    if (
        not request.angle_csv_path
        or request.swapped_mask is None
        or not angle_csv_store.exists(request.angle_csv_path)
    ):
        _write_dp_leg_swap_mask(
            LegSwapMaskRequest(request.swapped_mask, request.output_dir)
        )
        return None

    try:
        angle_dataframe = angle_csv_store.read(request.angle_csv_path)
        alignment = _align_angle_dataframe_to_leg_identity(
            angle_dataframe,
            request.swapped_mask,
        )
        if alignment.swapped_frames == 0:
            _write_dp_leg_swap_mask(
                LegSwapMaskRequest(request.swapped_mask, request.output_dir)
            )
            return None
        if not alignment.applied_pairs:
            return None

        angle_csv_store.write(request.angle_csv_path, alignment.dataframe)

        swap_info = _write_dp_leg_swap_mask(
            LegSwapMaskRequest(request.swapped_mask, request.output_dir)
        )
        swap_csv = swap_info.get("swap_csv") if swap_info else None

        print(
            "  ▶ 已依 DP 左右腿身份修正同步角度 CSV: "
            f"{request.angle_csv_path} "
            f"(swapped_frames={alignment.swapped_frames}, "
            f"pairs={list(alignment.applied_pairs)})"
        )
        return {
            "angle_csv": request.angle_csv_path,
            "swap_csv": swap_csv,
            "swapped_frames": alignment.swapped_frames,
            "applied_pairs": list(alignment.applied_pairs),
        }
    except CSV_IO_ERRORS as error:
        print(f"  ▶ 同步 DP 左右腿身份到角度 CSV 失敗: {error}")
        return None


# -----------------------------------------------------------------------
# 延遲 import：只在真正需要時才載入 GPU-heavy 的 3D 重建函式庫
# -----------------------------------------------------------------------
def _import_vis_module(motion_ag_dir: Path):
    """動態載入 MotionAGFormer 的 vis.py；呼叫端負責暫時程序環境。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vis", str(motion_ag_dir / "demo" / "vis.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"無法載入 MotionAGFormer vis.py: {motion_ag_dir}")
    vis_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vis_module)
    return vis_module


@dataclass(frozen=True)
class CorrectedPose3DRequest:
    """描述使用修正後 2D 關鍵點重新產生 3D 角度的工作。"""

    tracked_video_path: str | None
    pose_output_dir: str
    analysis_output_dir: str
    gpu: str
    motion_ag_dir: Path
    timings: list | None = None


def _video_dimensions(video_path: str) -> tuple[float, float]:
    """讀取影片寬高；影片無法開啟時回傳零值。"""
    video_capture = cv2.VideoCapture(video_path)
    try:
        if not video_capture.isOpened():
            return 0.0, 0.0
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0
        return float(width), float(height)
    finally:
        video_capture.release()


def _archived_tracked_video_path(pose_output_dir: str) -> str | None:
    """尋找關鍵點封存目錄中的第一相機追蹤影片。"""
    pointer_path = (
        Path(pose_output_dir) / "input_2D" / "keypoints_raw_archive_dir.txt"
    )
    if not pointer_path.exists():
        return None
    try:
        archive_dir = pointer_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    archived_video = Path(archive_dir) / "cam1_tracked.mp4"
    return str(archived_video) if archived_video.exists() else None


def _keypoint_coordinate_extent(keypoints_npz: str) -> tuple[float, float]:
    """讀取關鍵點資料並回傳 X、Y 座標最大值。"""
    reconstruction = np.load(keypoints_npz, allow_pickle=True)["reconstruction"]
    keypoint_coordinates = (
        reconstruction[0, :, :, :2]
        if reconstruction.ndim == 4
        else reconstruction[:, :, :2]
    )
    return (
        float(np.nanmax(keypoint_coordinates[..., 0])),
        float(np.nanmax(keypoint_coordinates[..., 1])),
    )


def _coordinate_system_may_differ(
    keypoint_extent: tuple[float, float],
    video_dimensions: tuple[float, float],
) -> bool:
    """判斷關鍵點與影片是否可能使用不同尺寸的座標系。"""
    max_keypoint_x, max_keypoint_y = keypoint_extent
    video_width, video_height = video_dimensions
    return (
        video_width > 0
        and video_height > 0
        and (
            max_keypoint_x < video_width * KEYPOINT_VIDEO_SIZE_RATIO_THRESHOLD
            or max_keypoint_y < video_height * KEYPOINT_VIDEO_SIZE_RATIO_THRESHOLD
        )
    )


def _select_compatible_3d_video(
    request: CorrectedPose3DRequest,
    keypoints_npz: str,
) -> str:
    """選擇與修正後關鍵點座標系相容的 3D 重算影片。"""
    assert request.tracked_video_path is not None
    selected_video = request.tracked_video_path
    try:
        keypoint_extent = _keypoint_coordinate_extent(keypoints_npz)
        selected_dimensions = _video_dimensions(selected_video)
        if not _coordinate_system_may_differ(keypoint_extent, selected_dimensions):
            return selected_video

        archived_video = _archived_tracked_video_path(request.pose_output_dir)
        if archived_video:
            archived_dimensions = _video_dimensions(archived_video)
            if all(dimension > 0 for dimension in archived_dimensions):
                print(
                    "  ▶ 偵測到 3D 重算影片尺寸與 keypoints 座標系不一致，"
                    f"改用 archive tracked video: {archived_video} "
                    f"({selected_dimensions[0]:.0f}x{selected_dimensions[1]:.0f} -> "
                    f"{archived_dimensions[0]:.0f}x{archived_dimensions[1]:.0f})"
                )
                return archived_video

        print(
            "  ▶ 警告：3D 重算影片尺寸可能與 keypoints 座標系不一致，"
            f"video={selected_dimensions[0]:.0f}x{selected_dimensions[1]:.0f}, "
            f"keypoints max=({keypoint_extent[0]:.1f},{keypoint_extent[1]:.1f})"
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        cv2.error,
    ) as error:
        print(f"  ▶ 檢查 3D 重算影片尺寸失敗，繼續使用原影片: {error}")
    return selected_video


def _generate_corrected_3d_angles(
    request: CorrectedPose3DRequest,
    video_path: str,
) -> None:
    """在隔離環境中執行 MotionAGFormer 3D 姿態與角度重算。"""
    with _temporary_motion_agformer_runtime(request.motion_ag_dir, request.gpu):
        started_at = time.perf_counter()
        visualization_module = _import_vis_module(request.motion_ag_dir)
        visualization_module.get_pose3D(
            video_path,
            request.pose_output_dir,
            skip_video=True,
        )
        _record_timing(
            request.timings,
            "Analysis/rerun_3d_angles_after_leg_dp",
            started_at,
        )


def _publish_corrected_angle_csv(request: CorrectedPose3DRequest) -> str | None:
    """將重算產生的角度 CSV 發佈到分析輸出目錄。"""
    source_angle_csv = os.path.join(
        request.pose_output_dir,
        "pred_3D",
        "angles",
        f"{Path(request.pose_output_dir).name}_angles.csv",
    )
    if not angle_csv_store.exists(source_angle_csv):
        print(f"  ▶ 重算後角度 CSV 不存在: {source_angle_csv}")
        return None

    output_angle_csv = os.path.join(request.analysis_output_dir, "angles.csv")
    try:
        angle_csv_store.publish(source_angle_csv, output_angle_csv)
        print(f"  ▶ 已用 DP 修正後 2D keypoints 重算 3D 角度: {output_angle_csv}")
        return output_angle_csv
    except OSError as error:
        print(f"  ▶ 複製重算後角度 CSV 失敗: {error}")
        return source_angle_csv


def _rerun_3d_angles_from_corrected_2d(
    request: CorrectedPose3DRequest,
) -> str | None:
    """協調修正後 2D 關鍵點的 3D 角度重算與結果發佈。"""
    if (
        not request.tracked_video_path
        or not request.pose_output_dir
        or not request.analysis_output_dir
    ):
        return None

    keypoints_npz = os.path.join(
        request.pose_output_dir,
        "input_2D",
        "keypoints.npz",
    )
    if not os.path.exists(keypoints_npz):
        print(f"  ▶ 無法重算 3D 角度，找不到修正後 keypoints: {keypoints_npz}")
        return None

    selected_video = _select_compatible_3d_video(request, keypoints_npz)
    _generate_corrected_3d_angles(request, selected_video)
    return _publish_corrected_angle_csv(request)


# -----------------------------------------------------------------------
# 各步驟實作函式
# -----------------------------------------------------------------------


def _active_tracking_cameras(camera_configs: list) -> list:
    """建立追蹤相機設定並排除沒有影片路徑的項目。"""
    active_cameras = [
        tracking._build_camera_from_entry(camera_entry)
        for camera_entry in camera_configs
    ]
    active_cameras = [
        camera for camera in active_cameras if camera["video_path"] is not None
    ]
    if not active_cameras:
        raise ValueError("所有相機的 video_path 均為 None，請至少設定一台。")
    return active_cameras


def _open_video_captures(active_cameras: list) -> list[cv2.VideoCapture]:
    """開啟所有相機影片；任一失敗時釋放已開啟資源。"""
    video_captures: list[cv2.VideoCapture] = []
    for camera_index, camera in enumerate(active_cameras):
        video_capture = cv2.VideoCapture(camera["video_path"])
        if not video_capture.isOpened():
            for opened_capture in video_captures:
                opened_capture.release()
            raise ValueError(f"無法開啟相機 {camera_index + 1}: {camera['video_path']}")
        video_captures.append(video_capture)
    return video_captures


def _tracking_output_name(active_cameras: list, extra_config: dict) -> str:
    """依明確設定或第一台相機檔名決定追蹤輸出名稱。"""
    if extra_config and "output_name" in extra_config:
        return extra_config["output_name"].replace(
            ".mp4",
            "_cropped.mp4",
        )
    first_camera_stem = Path(active_cameras[0]["video_path"]).stem
    return f"{first_camera_stem}_tracked.mp4"


def _write_tracking_output_marker(output_dir: str, output_name: str) -> None:
    """記錄最新追蹤輸出檔名，供略過 Step 1 時尋找結果。"""
    marker_path = os.path.join(output_dir, ".last_output_name")
    with open(marker_path, "w", encoding="utf-8") as marker_file:
        marker_file.write(output_name)


def _load_and_warm_up_tracking_model(timings: list | None):
    """載入 YOLO 並以空白影格完成第一次推論暖機。"""
    from ultralytics import YOLO

    model_started_at = time.perf_counter()
    model = YOLO(tracking.MODEL_PATH)
    model.predict(
        np.zeros(YOLO_WARMUP_FRAME_SHAPE, dtype=np.uint8),
        device=tracking.DEVICE,
        verbose=False,
    )
    _record_timing(
        timings,
        "Step1/load_yolo_model_and_warmup",
        model_started_at,
        model_path=str(tracking.MODEL_PATH),
    )
    return model


@dataclass(frozen=True)
class TrackingRunContext:
    """保存一次 tracking 執行所需的穩定輸入與相依物件。"""

    active_cameras: list
    video_captures: list[cv2.VideoCapture]
    model: object
    output_dir: str
    output_path: str
    frame_map_path: str
    frames_per_second: float
    timings: list | None = None


@dataclass(frozen=True)
class CandidateRunnerTracks:
    """保存 two-pass 第一遍掃描產生的候選軌跡資料。"""

    frame_ranges_by_camera: dict | None
    detections: list
    frame_cache: dict


@dataclass(frozen=True)
class SelectedRunnerTracks:
    """保存 two-pass 選定主跑者後的軌跡資料。"""

    candidates: CandidateRunnerTracks
    runner_ids: dict
    summaries: list


def _auto_size_online_crop(context: TrackingRunContext) -> TrackingRunContext:
    """以 dry run 的人物框尺寸設定 online 模式裁剪大小。"""
    if not tracking.AUTO_CROP:
        return context

    print("auto_crop：第一遍掃描（分析 bbox 大小）...")
    dry_run_started_at = time.perf_counter()
    (
        _,
        _,
        dry_run_bbox_widths,
        dry_run_bbox_heights,
        _,
        _,
    ) = tracking._process_cameras(
        context.video_captures,
        context.active_cameras,
        context.model,
        None,
        dry_run=True,
    )
    _record_timing(
        context.timings,
        "Step1/online_auto_crop_dry_run",
        dry_run_started_at,
        bbox_samples=len(dry_run_bbox_widths),
    )
    refreshed_captures = _open_video_captures(context.active_cameras)
    if dry_run_bbox_widths and dry_run_bbox_heights:
        crop_side = tracking._auto_crop_side_from_bbox_sizes(
            dry_run_bbox_widths,
            dry_run_bbox_heights,
        )
        tracking.CROP_WIDTH = crop_side
        tracking.CROP_HEIGHT = crop_side
        print(
            "  自動設定裁剪尺寸: "
            f"{tracking.CROP_WIDTH} x {tracking.CROP_HEIGHT}（auto square）"
        )
    return replace(context, video_captures=refreshed_captures)


def _create_tracking_video_writer(
    output_path: str,
    frames_per_second: float,
) -> cv2.VideoWriter:
    """建立使用目前裁剪尺寸的 MP4 writer。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    return cv2.VideoWriter(
        output_path,
        fourcc,
        frames_per_second,
        (tracking.CROP_WIDTH, tracking.CROP_HEIGHT),
    )


def _prescan_person_frame_ranges(context: TrackingRunContext) -> dict | None:
    """預先找出各相機包含人物的有效影格區間。"""
    prescan_started_at = time.perf_counter()
    frame_ranges_by_camera = (
        tracking.run_temporal_prescan(
            context.active_cameras,
            output_dir=context.output_dir,
        )
        if tracking.PRESCAN_ENABLED
        else None
    )
    _record_timing(
        context.timings,
        "Step1/prescan_person_frames",
        prescan_started_at,
        enabled=bool(tracking.PRESCAN_ENABLED),
    )
    return frame_ranges_by_camera


def _collect_candidate_runner_tracks(
    context: TrackingRunContext,
    frame_ranges_by_camera: dict | None,
) -> CandidateRunnerTracks:
    """執行 two-pass 第一遍並收集所有候選跑者軌跡。"""
    print("two_pass 模式：第一遍收集所有候選人軌跡...")
    first_pass_captures = _open_video_captures(context.active_cameras)
    try:
        pass1_started_at = time.perf_counter()
        detections, frame_cache = tracking._collect_all_detections(
            first_pass_captures,
            context.active_cameras,
            context.model,
            frame_ranges_by_cam=frame_ranges_by_camera,
        )
        _record_timing(
            context.timings,
            "Step1/two_pass_pass1_collect_detections",
            pass1_started_at,
            detections=len(detections),
            cached_frames=len(frame_cache),
        )
    finally:
        for first_pass_capture in first_pass_captures:
            first_pass_capture.release()
    print(f"  收集完成：共 {len(detections)} 筆偵測")
    return CandidateRunnerTracks(
        frame_ranges_by_camera=frame_ranges_by_camera,
        detections=detections,
        frame_cache=frame_cache,
    )


def _select_and_stitch_runner_tracks(
    context: TrackingRunContext,
    candidates: CandidateRunnerTracks,
) -> SelectedRunnerTracks:
    """選定主跑者並修補其短暫中斷的追蹤 ID。"""
    select_started_at = time.perf_counter()
    runner_ids, summaries = tracking._score_and_select_runners(
        candidates.detections,
        context.active_cameras,
        frame_ranges_by_cam=candidates.frame_ranges_by_camera,
    )
    _record_timing(
        context.timings,
        "Step1/two_pass_select_main_runner",
        select_started_at,
        selected_ids={
            str(camera_index): int(runner_id)
            for camera_index, runner_id in runner_ids.items()
        },
    )
    stitch_started_at = time.perf_counter()
    tracking._stitch_target_id(
        candidates.frame_cache,
        runner_ids,
        context.active_cameras,
        fps=context.frames_per_second,
    )
    _record_timing(
        context.timings,
        "Step1/two_pass_stitch_target_id",
        stitch_started_at,
    )
    return SelectedRunnerTracks(
        candidates=candidates,
        runner_ids=runner_ids,
        summaries=summaries,
    )


def _write_runner_selection_debug(
    context: TrackingRunContext,
    selected_tracks: SelectedRunnerTracks,
) -> None:
    """輸出 two-pass 主跑者選擇的診斷資料。"""
    debug_started_at = time.perf_counter()
    tracking._write_two_pass_debug(
        selected_tracks.candidates.detections,
        selected_tracks.summaries,
        selected_tracks.runner_ids,
        Path(context.active_cameras[0]["video_path"]).stem,
    )
    _record_timing(
        context.timings,
        "Step1/two_pass_write_debug_csv",
        debug_started_at,
    )


def _configure_two_pass_crop(
    context: TrackingRunContext,
    selected_tracks: SelectedRunnerTracks,
) -> None:
    """依已選主跑者的 bbox 樣本設定 two-pass 裁切尺寸。"""
    if not tracking.AUTO_CROP:
        return

    crop_started_at = time.perf_counter()
    crop_side, selected_bbox_widths, _ = tracking._auto_crop_from_selected_cache(
        selected_tracks.candidates.frame_cache,
        selected_tracks.runner_ids,
    )
    _record_timing(
        context.timings,
        "Step1/two_pass_auto_crop_from_selected_bbox",
        crop_started_at,
        bbox_samples=len(selected_bbox_widths),
        crop_side=crop_side,
    )
    if crop_side:
        tracking.CROP_WIDTH = crop_side
        tracking.CROP_HEIGHT = crop_side
        print(
            "  two_pass auto_crop: 使用已選主跑者 bbox 設定裁剪尺寸 "
            f"{tracking.CROP_WIDTH} x {tracking.CROP_HEIGHT}"
            f"（samples={len(selected_bbox_widths)}）"
        )
    else:
        print("  警告：two_pass auto_crop 未收集到已選主跑者 bbox，沿用目前尺寸")


def _render_two_pass_tracking(
    context: TrackingRunContext,
    selected_tracks: SelectedRunnerTracks,
) -> None:
    """使用已選定的軌跡快取輸出 two-pass 追焦影片。"""
    video_writer = _create_tracking_video_writer(
        context.output_path,
        context.frames_per_second,
    )
    try:
        print("two_pass 模式：第二遍輸出追焦影片（快取模式，跳過 YOLO）...")
        pass2_started_at = time.perf_counter()
        tracking._process_cameras(
            context.video_captures,
            context.active_cameras,
            context.model,
            video_writer,
            frame_map_path=context.frame_map_path,
            preset_target_ids=selected_tracks.runner_ids,
            frame_cache=selected_tracks.candidates.frame_cache,
            frame_ranges_by_cam=selected_tracks.candidates.frame_ranges_by_camera,
        )
        _record_timing(
            context.timings,
            "Step1/two_pass_pass2_write_tracked_video",
            pass2_started_at,
            output_path=context.output_path,
            crop_width=int(tracking.CROP_WIDTH),
            crop_height=int(tracking.CROP_HEIGHT),
        )
    finally:
        video_writer.release()


def _run_two_pass_tracking(context: TrackingRunContext) -> None:
    """協調 two-pass 候選掃描、主跑者選擇、裁切與影片輸出。"""
    frame_ranges = _prescan_person_frame_ranges(context)
    candidates = _collect_candidate_runner_tracks(context, frame_ranges)
    selected_tracks = _select_and_stitch_runner_tracks(context, candidates)
    _write_runner_selection_debug(context, selected_tracks)
    _configure_two_pass_crop(context, selected_tracks)
    _render_two_pass_tracking(context, selected_tracks)


def _run_online_tracking(context: TrackingRunContext) -> None:
    """以單次循序推論輸出追焦影片。"""
    video_writer = _create_tracking_video_writer(
        context.output_path,
        context.frames_per_second,
    )
    try:
        online_started_at = time.perf_counter()
        tracking._process_cameras(
            context.video_captures,
            context.active_cameras,
            context.model,
            video_writer,
            frame_map_path=context.frame_map_path,
        )
        _record_timing(
            context.timings,
            "Step1/online_tracking_write_tracked_video",
            online_started_at,
            output_path=context.output_path,
            crop_width=int(tracking.CROP_WIDTH),
            crop_height=int(tracking.CROP_HEIGHT),
        )
    finally:
        video_writer.release()


@dataclass(frozen=True)
class Step1TrackingRequest:
    """描述一次多相機追蹤與裁剪工作。"""

    camera_configs: list
    extra_config: dict
    gpu: str
    output_dir: str
    timings: list | None = None


def _step1_track_impl(request: Step1TrackingRequest) -> str:
    """協調 YOLO 多相機追蹤與跑者置中裁剪。"""
    print(OUTPUT_SEPARATOR)
    print("Step 1 — 多相機追蹤 + 人物置中裁剪 (Core.Tracking)")
    print(OUTPUT_SEPARATOR)
    step_started_at = time.perf_counter()

    active_cameras = _active_tracking_cameras(request.camera_configs)
    os.makedirs(request.output_dir, exist_ok=True)
    video_captures = _open_video_captures(active_cameras)
    output_name = _tracking_output_name(active_cameras, request.extra_config)
    output_path = os.path.join(request.output_dir, output_name)
    _write_tracking_output_marker(request.output_dir, output_name)
    model = _load_and_warm_up_tracking_model(request.timings)
    frames_per_second = video_captures[0].get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FPS
    frame_map_path = os.path.join(
        request.output_dir,
        output_name.replace(".mp4", "_frame_map.csv"),
    )
    tracking_context = TrackingRunContext(
        active_cameras=active_cameras,
        video_captures=video_captures,
        model=model,
        output_dir=request.output_dir,
        output_path=output_path,
        frame_map_path=frame_map_path,
        frames_per_second=frames_per_second,
        timings=request.timings,
    )

    if tracking.TRACKING_MODE == "two_pass":
        _run_two_pass_tracking(tracking_context)
    else:
        tracking_context = _auto_size_online_crop(tracking_context)
        _run_online_tracking(tracking_context)

    print(f"\nStep 1 完成，置中裁剪影片儲存至：{output_path}\n")
    _record_timing(
        request.timings,
        "Step1/total_tracking",
        step_started_at,
        output_path=output_path,
    )
    return output_path


def step1_track(request: Step1TrackingRequest) -> str:
    """在隔離的 tracking 設定與 GPU 環境中執行 Step 1。"""
    with _temporary_tracking_runtime(
        TrackingRuntimeOptions(
            config=request.extra_config,
            gpu=request.gpu,
            output_directory=request.output_dir,
        )
    ):
        return _step1_track_impl(request)


@dataclass(frozen=True)
class PoseEstimationRequest:
    """描述一次姿態估計工作及其輸出內容。"""

    tracked_video_path: str
    output_base_dir: str
    gpu: str
    motion_ag_dir: Path
    pose_scope: PoseScope = PoseScope.TWO_D_AND_3D
    video_output: VideoOutput = VideoOutput.GENERATE
    timings: list | None = None
    pose_model_path: str | None = None


@dataclass(frozen=True)
class PoseWorkspace:
    """保存姿態估計執行時使用的已解析路徑。"""

    video_path: str
    output_dir: str
    result_dir: str
    bbox_csv_path: str | None


def _prepare_pose_workspace(request: PoseEstimationRequest) -> PoseWorkspace:
    """建立輸出目錄，並解析姿態估計需要的輸入路徑。"""
    video_stem = Path(request.tracked_video_path).stem
    result_dir = os.path.join(request.output_base_dir, video_stem) + "/"
    os.makedirs(result_dir, exist_ok=True)

    video_path = os.path.abspath(request.tracked_video_path)
    bbox_csv_candidate = video_path.replace(
        ".mp4",
        "_bbox_map.csv",
    )
    return PoseWorkspace(
        video_path=video_path,
        output_dir=os.path.abspath(result_dir),
        result_dir=result_dir,
        bbox_csv_path=(
            bbox_csv_candidate if os.path.exists(bbox_csv_candidate) else None
        ),
    )


def _clear_stale_pose_outputs(workspace: PoseWorkspace) -> None:
    """清除舊影格，避免新舊輸出混合而污染生成影片。"""
    for folder_name in STALE_POSE_OUTPUT_DIRECTORIES:
        folder_path = os.path.join(workspace.output_dir, folder_name)
        if os.path.exists(folder_path):
            print(
                f"  [Step 2] 偵測到舊的 {folder_name} 資料夾，進行清理以避免新舊影格污染..."
            )
            try:
                shutil.rmtree(folder_path)
            except OSError as error:
                print(f"  ⚠️  [Step 2] 清理 {folder_name} 失敗: {error}")


def _print_pose_estimation_plan(
    request: PoseEstimationRequest,
    workspace: PoseWorkspace,
) -> None:
    """顯示本次姿態估計的範圍與輸出位置。"""
    includes_3d = request.pose_scope is PoseScope.TWO_D_AND_3D
    analysis_scope = "2D + 3D + 角度" if includes_3d else "2D only"
    print(OUTPUT_SEPARATOR)
    print(f"Step 2 — 姿態估計（{analysis_scope}）")
    print(f"  影片: {request.tracked_video_path}")
    print(f"  輸出: {workspace.result_dir}")
    print(OUTPUT_SEPARATOR)


def _execute_pose_estimation(
    request: PoseEstimationRequest,
    workspace: PoseWorkspace,
) -> None:
    """在隔離的 MotionAGFormer 執行環境中完成姿態估計。"""
    includes_3d = request.pose_scope is PoseScope.TWO_D_AND_3D
    generates_video = request.video_output is VideoOutput.GENERATE

    # 暫時改變程序環境，離開區塊後會完整還原。
    with _temporary_motion_agformer_runtime(request.motion_ag_dir, request.gpu):
        run_pose_estimation = _import_vis_module(
            request.motion_ag_dir
        ).run_pose_estimation
        pose_started_at = time.perf_counter()
        run_pose_estimation(
            video_path=workspace.video_path,
            output_dir=workspace.output_dir,
            only_2d=not includes_3d,
            gpu=request.gpu,
            bbox_csv=workspace.bbox_csv_path,
            skip_video=not generates_video,
            model_path=request.pose_model_path,
        )
        _record_timing(
            request.timings,
            "Step2/pose_estimation_total",
            pose_started_at,
            video_path=workspace.video_path,
            output_dir=workspace.output_dir,
            only_2d=not includes_3d,
            skip_video=not generates_video,
        )


def step2_pose(request: PoseEstimationRequest) -> str:
    """協調工作區準備、舊輸出清理與姿態估計。"""
    workspace = _prepare_pose_workspace(request)
    _clear_stale_pose_outputs(workspace)
    _print_pose_estimation_plan(request, workspace)
    _execute_pose_estimation(request, workspace)

    print(f"\nStep 2 完成，骨架與角度數據輸出至：{workspace.result_dir}\n")
    return workspace.result_dir


@dataclass(frozen=True)
class OverlayWorkspace:
    """保存 Step 3 角度疊圖使用的輸入與輸出路徑。"""

    pose_video: str
    angle_csv: str
    output_video: str
    frame_map: str | None
    config_marker: str


def _prepare_overlay_workspace(
    pose_output_dir: str,
    video_stem: str,
) -> OverlayWorkspace:
    """解析角度疊圖階段使用的所有路徑。"""
    pose_video = os.path.join(pose_output_dir, f"{video_stem}_2D.mp4")
    angle_csv = os.path.join(
        pose_output_dir,
        "pred_3D",
        "angles",
        f"{video_stem}_angles.csv",
    )
    pipeline_output_dir = os.path.dirname(
        os.path.normpath(pose_output_dir)
    )
    frame_map_candidate = os.path.join(
        pipeline_output_dir,
        f"{video_stem}_frame_map.csv",
    )
    return OverlayWorkspace(
        pose_video=pose_video,
        angle_csv=angle_csv,
        output_video=os.path.join(
            pose_output_dir,
            f"{video_stem}_2D_angles.mp4",
        ),
        frame_map=(
            frame_map_candidate if os.path.exists(frame_map_candidate) else None
        ),
        config_marker=os.path.join(pipeline_output_dir, ".config.json"),
    )


def _overlay_inputs_exist(workspace: OverlayWorkspace) -> bool:
    """確認角度疊圖必要輸入存在，並顯示略過原因。"""
    if not os.path.exists(workspace.angle_csv):
        print("  ⚠️  [Step 3] 角度 CSV 不存在，略過 Step 3 合併 (可能是 only_2d=True)")
        return False
    if not os.path.exists(workspace.pose_video):
        print(f"  ⚠️  [Step 3] 2D 骨架影片不存在: {workspace.pose_video}，略過")
        return False
    return True


def _load_overlay_main_video_paths(config_marker: str) -> list[str]:
    """從 Pipeline 暫存設定讀取原始相機影片路徑。"""
    if not os.path.exists(config_marker):
        return []
    try:
        with open(config_marker, encoding="utf-8") as config_file:
            saved_config = json.load(config_file)
        return [
            camera["video_path"]
            for camera in saved_config.get("cameras", [])
            if camera.get("video_path")
        ]
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        AttributeError,
        TypeError,
    ) as error:
        print(f"  [Step 3] 無法讀取 config 暫存: {error}")
        return []


def _render_angle_overlay(
    workspace: OverlayWorkspace,
    main_video_paths: list[str],
) -> None:
    """以固定顯示設定產生 2D 骨架與角度折線圖合併影片。"""
    add_angle_overlay(
        workspace.pose_video,
        workspace.angle_csv,
        workspace.output_video,
        AngleOverlayConfig(
            main_videos=main_video_paths,
            frame_map_path=workspace.frame_map,
        ),
    )


def step3_overlay(pose_output_dir: str, video_stem: str) -> str | None:
    """協調 2D 骨架影片與角度折線圖合併。"""
    workspace = _prepare_overlay_workspace(pose_output_dir, video_stem)

    print(OUTPUT_SEPARATOR)
    print("Step 3 — 2D 影片 + 角度折線圖合併")
    print(OUTPUT_SEPARATOR)

    if not _overlay_inputs_exist(workspace):
        return None
    _render_angle_overlay(
        workspace,
        _load_overlay_main_video_paths(workspace.config_marker),
    )
    return workspace.output_video


# -----------------------------------------------------------------------
# CLI/Python Orchestration API
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineOptions:
    """保存底層追蹤與姿態 Pipeline 的執行選項。"""

    output_dir: str | None = None
    gpu: str = "0"
    pose_scope: PoseScope = PoseScope.TWO_D_AND_3D
    tracked_video_source: TrackedVideoSource = TrackedVideoSource.GENERATE
    video_output: VideoOutput = VideoOutput.GENERATE
    timings: list | None = None


@dataclass(frozen=True)
class PipelineRequest:
    """描述一次底層追蹤與姿態 Pipeline 工作。"""

    cameras: list
    extra_config: dict = field(default_factory=dict)
    options: PipelineOptions = field(default_factory=PipelineOptions)


@dataclass(frozen=True)
class PipelineWorkspace:
    """保存 Pipeline 各階段依序產生的主要路徑。"""

    output_dir: str
    tracked_video: str
    pose_directory: str | None = None


def _pipeline_output_directory(request: PipelineRequest) -> str:
    """解析 Pipeline 所有階段共用的輸出目錄。"""
    return request.options.output_dir or request.extra_config.get(
        "output_dir",
        str(REPO_ROOT / "output_cut"),
    )


def _find_existing_tracked_video(output_dir: str) -> str:
    """依輸出標記或目錄內容尋找既有追蹤影片。"""
    marker_path = os.path.join(output_dir, ".last_output_name")
    if os.path.exists(marker_path):
        with open(marker_path, encoding="utf-8") as marker_file:
            output_name = marker_file.read().strip()
        return os.path.join(output_dir, output_name)

    videos = [
        filename
        for filename in (
            os.listdir(output_dir) if os.path.isdir(output_dir) else []
        )
        if filename.endswith(".mp4") and not filename.endswith("_2D.mp4")
    ]
    if not videos:
        raise FileNotFoundError("找不到已追蹤的影片，無法略過 Step 1")
    return os.path.join(output_dir, min(videos))


def _obtain_tracked_video(
    request: PipelineRequest,
    output_dir: str,
) -> str:
    """依設定產生追蹤影片，或沿用既有輸出。"""
    if request.options.tracked_video_source is TrackedVideoSource.GENERATE:
        return step1_track(
            Step1TrackingRequest(
                camera_configs=request.cameras,
                extra_config=request.extra_config,
                gpu=request.options.gpu,
                output_dir=output_dir,
                timings=request.options.timings,
            )
        )
    print("略過 Step 1，讀取上一次的輸出結果...")
    return _find_existing_tracked_video(output_dir)


def _write_pipeline_config(
    request: PipelineRequest,
    output_dir: str,
) -> None:
    """保存 Step 3 建立圖表底圖需要的相機設定。"""
    pipeline_config = {"cameras": request.cameras}
    pipeline_config.update(request.extra_config)
    config_marker = os.path.join(output_dir, ".config.json")
    with open(config_marker, "w", encoding="utf-8") as config_file:
        json.dump(pipeline_config, config_file, ensure_ascii=False, indent=2)


def _run_pose_pipeline_step(
    request: PipelineRequest,
    workspace: PipelineWorkspace,
) -> str:
    """執行 Pipeline 的姿態估計階段。"""
    return step2_pose(
        PoseEstimationRequest(
            tracked_video_path=workspace.tracked_video,
            output_base_dir=workspace.output_dir,
            gpu=request.options.gpu,
            motion_ag_dir=REPO_ROOT / "MotionAGFormer",
            pose_scope=request.options.pose_scope,
            video_output=request.options.video_output,
            timings=request.options.timings,
            pose_model_path=request.extra_config.get("pose_model_path"),
        )
    )


def _run_overlay_pipeline_step(
    request: PipelineRequest,
    workspace: PipelineWorkspace,
) -> str | None:
    """需要影片輸出時執行角度折線圖合併。"""
    if request.options.video_output is VideoOutput.OMIT:
        return None
    assert workspace.pose_directory is not None
    return step3_overlay(
        workspace.pose_directory,
        Path(workspace.tracked_video).stem,
    )


def run_pipeline(request: PipelineRequest) -> dict:
    """協調追蹤、姿態估計與角度影片輸出。"""
    output_dir = _pipeline_output_directory(request)
    tracked_video = _obtain_tracked_video(request, output_dir)
    _write_pipeline_config(request, output_dir)
    workspace = PipelineWorkspace(
        output_dir=output_dir,
        tracked_video=tracked_video,
    )
    workspace = replace(
        workspace,
        pose_directory=_run_pose_pipeline_step(request, workspace),
    )
    overlay_video = _run_overlay_pipeline_step(request, workspace)

    return {
        "output_dir": workspace.pose_directory,
        "tracked_video": workspace.tracked_video,
        "overlay_video": overlay_video,
    }


@dataclass(frozen=True)
class AnalysisOptions:
    """保存單次高階分析流程的執行選項。"""

    gpu: str = "0"
    pose_scope: PoseScope = PoseScope.TWO_D_AND_3D
    tracked_video_source: TrackedVideoSource = TrackedVideoSource.GENERATE
    output_dest: str | None = None
    progress_callback: Callable[[int], None] | None = None
    started_at: float = field(default_factory=time.perf_counter, repr=False)


@dataclass
class AnalysisContext:
    """保存所有內部分析階段共用且穩定的設定。"""

    config: dict
    cameras: list
    tracking_cameras: list
    output_dest: str
    options: AnalysisOptions
    started_at: float
    timings: list = field(default_factory=list)
    motion_ag_dir: Path = field(default_factory=lambda: REPO_ROOT / "MotionAGFormer")

    def report_progress(self, percentage: int) -> None:
        """呼叫使用者提供的 callback 回報目前進度。"""
        if self.options.progress_callback:
            self.options.progress_callback(percentage)


@dataclass
class AnalysisState:
    """保存各分析階段依序產生的路徑、資料與統計結果。"""

    tracked_video: str | None
    track_output_dir: str
    track_output_name: str
    metrics_csv: str
    final_pose_dir: str
    angles_csv: str | None = None
    output_video: str | None = None
    step_analysis: dict | None = None
    avg_step_length: float | None = None
    offsets_npz: str | None = None
    keypoints_npz: str | None = None
    foot_npz: str | None = None
    step_overlay_started_at: float | None = None


def prepare_analysis_context(
    analysis_config: dict,
    options: AnalysisOptions,
) -> AnalysisContext:
    """協調分析設定正規化、追蹤設定轉換與輸出目錄準備。"""
    normalized_config = _normalize_analysis_config(analysis_config)
    cameras = normalized_config.get("cameras", [])
    output_dir = _resolve_analysis_output_directory(
        normalized_config,
        options,
    )
    os.makedirs(output_dir, exist_ok=True)
    normalized_config["output_dir"] = output_dir

    return AnalysisContext(
        config=normalized_config,
        cameras=cameras,
        tracking_cameras=_tracking_camera_configs(cameras),
        output_dest=output_dir,
        options=options,
        started_at=options.started_at,
    )


def _normalize_analysis_config(analysis_config: dict) -> dict:
    """複製分析設定，並在未指定裁切尺寸時啟用自動裁切。"""
    normalized_config = dict(analysis_config)
    if (
        "auto_crop" not in normalized_config
        and "crop_width" not in normalized_config
        and "crop_height" not in normalized_config
    ):
        normalized_config["auto_crop"] = True
    return normalized_config


def _tracking_camera_configs(cameras: list) -> list:
    """建立不含 Homography 控制點的追蹤相機設定。"""
    return [_tracking_camera_config(camera) for camera in cameras]


def _tracking_camera_config(camera: dict) -> dict:
    """將完整分析相機設定轉成 tracking 所需設定。"""
    tracking_camera = dict(camera)
    destination_world = tracking_camera.pop("homography_dst_world", None)
    tracking_camera.pop("homography_src_points", None)
    if (
        tracking_camera.get("distance_m") is None
        and destination_world
        and tracking_camera.get("start_line")
        and tracking_camera.get("end_line")
    ):
        world_x_coordinates = [point[0] for point in destination_world]
        distance_span = max(world_x_coordinates) - min(world_x_coordinates)
        if distance_span > 0:
            tracking_camera["distance_m"] = distance_span
    return tracking_camera


def _resolve_analysis_output_directory(
    normalized_config: dict,
    options: AnalysisOptions,
) -> str:
    """依明確選項、相機位置或預設值解析分析輸出目錄。"""
    configured_output = options.output_dest or normalized_config.get("output_dest")
    if configured_output:
        return configured_output

    cameras = normalized_config.get("cameras", [])
    camera_output = next(
        (
            os.path.dirname(os.path.abspath(camera["video_path"]))
            for camera in cameras
            if camera.get("video_path")
        ),
        None,
    )
    if camera_output:
        return camera_output
    return normalized_config.get(
        "output_dir",
        str(REPO_ROOT / "output_cut"),
    )


@dataclass(frozen=True)
class SpeedAnalysisPaths:
    """保存速度分析所需的追蹤輸入與輸出路徑。"""

    bbox_map: str
    offsets_npz: str
    metrics_csv: str


def _first_camera_video_path(context: AnalysisContext) -> str | None:
    """回傳第一台相機影片路徑；未設定相機時回傳 None。"""
    if not context.cameras:
        return None
    return context.cameras[0].get("video_path")


def _read_video_frames_per_second(
    video_path: str | None,
    fallback: float | None,
) -> float | None:
    """讀取影片 FPS；路徑或影片無效時回傳指定 fallback。"""
    if not video_path:
        return fallback
    video_capture = cv2.VideoCapture(video_path)
    try:
        if not video_capture.isOpened():
            return fallback
        return video_capture.get(cv2.CAP_PROP_FPS) or fallback
    finally:
        video_capture.release()


def _speed_analysis_paths(
    context: AnalysisContext,
    state: AnalysisState,
) -> SpeedAnalysisPaths:
    """根據追蹤影片與第一台相機建立速度分析路徑。"""
    assert state.tracked_video is not None
    tracked_video_stem = Path(state.tracked_video).stem
    first_camera_path = _first_camera_video_path(context)
    first_camera_stem = (
        Path(first_camera_path).stem if first_camera_path else tracked_video_stem
    )
    return SpeedAnalysisPaths(
        bbox_map=os.path.join(
            context.output_dest,
            f"{tracked_video_stem}_bbox_map.csv",
        ),
        offsets_npz=os.path.join(
            context.output_dest,
            f"{first_camera_stem}_offsets.npz",
        ),
        metrics_csv=state.metrics_csv,
    )


def _requested_speed_mode(context: AnalysisContext) -> str:
    """讀取指定速度模式，未指定時依 Homography 校正自動選擇。"""
    default_mode = (
        "homography"
        if any(camera.get("homography_src_points") for camera in context.cameras)
        else "pixel"
    )
    return str(context.config.get("speed_mode", default_mode)).lower()


def _calculate_speed_metrics(
    context: AnalysisContext,
    paths: SpeedAnalysisPaths,
) -> list:
    """從 bbox map 計算逐幀距離、速度與加速度資料。"""
    return compute_speed_from_bbox_map(
        paths.bbox_map,
        context.cameras,
        fps_override=_first_camera_fps(context),
        offsets_npz=paths.offsets_npz,
        pixel_cameras_cfg_list=context.tracking_cameras,
        speed_mode=_requested_speed_mode(context),
    )


def _write_speed_metrics_csv(metrics_csv: str, tracking_rows: list) -> None:
    """將逐幀速度分析結果寫入固定欄位的 CSV。"""
    with open(metrics_csv, "w", newline="", encoding="utf-8") as metrics_file:
        writer = csv.DictWriter(
            metrics_file,
            fieldnames=SPEED_METRIC_FIELD_NAMES,
        )
        writer.writeheader()
        writer.writerows(tracking_rows)


def _execute_speed_analysis(
    context: AnalysisContext,
    paths: SpeedAnalysisPaths,
) -> None:
    """執行速度計算並在有結果時發佈 metrics CSV。"""
    print("\n" + OUTPUT_SEPARATOR)
    print("【速度分析】從 bbox_map.csv 計算速度與加速度（無需重跑 YOLO）")
    print(OUTPUT_SEPARATOR)
    tracking_rows = _calculate_speed_metrics(context, paths)
    if not tracking_rows:
        print("  ▶ 速度計算未產出資料（無 calibration 資訊或 bbox 不足）")
        return
    _write_speed_metrics_csv(paths.metrics_csv, tracking_rows)
    print(f"  ▶ 速度分析完成，{len(tracking_rows)} 幀 → {paths.metrics_csv}")


def run_speed_analysis(context: AnalysisContext, state: AnalysisState) -> None:
    """協調追蹤輸出的速度指標計算、發佈與計時。"""
    if (
        context.options.tracked_video_source is TrackedVideoSource.EXISTING_OUTPUT
        or not state.tracked_video
    ):
        print("  使用者指定沿用既有追蹤影片，略過速度分析。")
        return

    speed_started_at = time.perf_counter()
    paths = _speed_analysis_paths(context, state)
    if not os.path.exists(paths.bbox_map):
        print(f"  ▶ bbox_map.csv 不存在，速度分析略過: {paths.bbox_map}")
    else:
        try:
            _execute_speed_analysis(context, paths)
        # Speed analysis is optional; isolate failures from third-party
        # numerical and video code so the primary pose result survives.
        except Exception as error:  # noqa: BLE001
            print(f"  ▶ 速度計算失敗: {error}")

    _record_timing(
        context.timings,
        "Analysis/speed_metrics_from_bbox_map",
        speed_started_at,
        bbox_map_path=paths.bbox_map,
        metrics_csv=paths.metrics_csv,
    )


def _normalize_pose_output_dir(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """將姿態輸出移到後續分析固定使用的目錄。"""
    expected_pose_dir = os.path.join(context.output_dest, "sequential_tracked")
    if (
        os.path.exists(state.final_pose_dir)
        and state.final_pose_dir != expected_pose_dir
    ):
        if os.path.exists(expected_pose_dir):
            shutil.rmtree(expected_pose_dir)
        os.rename(state.final_pose_dir, expected_pose_dir)
        state.final_pose_dir = expected_pose_dir

    print(f"  ▶ 姿態分析資料夾: {state.final_pose_dir}")


def _prepare_leg_identity_paths(
    context: AnalysisContext,
    state: AnalysisState,
) -> bool:
    """設定腿部身份分析所需路徑，並確認必要輸入存在。"""
    if not context.cameras:
        return False

    original_stem = Path(context.cameras[0]["video_path"]).stem
    state.offsets_npz = os.path.join(
        context.output_dest,
        f"{original_stem}_offsets.npz",
    )
    state.keypoints_npz = os.path.join(
        state.final_pose_dir,
        "input_2D",
        "keypoints.npz",
    )
    state.output_video = os.path.join(context.output_dest, "output_final.mp4")
    return bool(
        os.path.exists(state.offsets_npz) and os.path.exists(state.keypoints_npz)
    )


def _run_initial_step_analysis(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """執行首次步頻分析，產生腳步事件供身份修正使用。"""
    state.step_analysis = run_step_stride_analysis(
        config=context.config,
        output_dir=context.output_dest,
    )


def _load_pre_dp_leg_swap_mask(state: AnalysisState):
    """讀取姿態模型在 DP 修正前記錄的左右腿交換遮罩。"""
    status_npz = os.path.join(
        state.final_pose_dir,
        "input_2D",
        "keypoint_status.npz",
    )
    if not os.path.exists(status_npz):
        return None

    with np.load(status_npz, allow_pickle=True) as status_data:
        if "pre_dp_leg_swap_mask" not in status_data.files:
            return None
        pre_dp_swapped_mask = np.asarray(
            status_data["pre_dp_leg_swap_mask"],
            dtype=bool,
        )
    if pre_dp_swapped_mask.ndim == 2:
        return pre_dp_swapped_mask[0]
    return pre_dp_swapped_mask


def _correct_leg_identity(
    context: AnalysisContext,
    state: AnalysisState,
):
    """套用錨點與 DP 腿部修正，並同步腳部關鍵點。"""
    assert state.keypoints_npz is not None
    assert state.step_analysis is not None

    anchor_swapped_mask = apply_anchor_leg_correction(
        state.keypoints_npz,
        state.step_analysis["step_events"],
    )
    pre_dp_swapped_mask = _load_pre_dp_leg_swap_mask(state)
    swapped_mask = update_leg_swap_metadata(
        state.keypoints_npz,
        pre_dp_swapped_mask,
        anchor_swapped_mask,
    )
    state.foot_npz = os.path.join(
        state.final_pose_dir,
        "input_2D",
        "foot_keypoints.npz",
    )
    raw_keypoints_npz = os.path.join(
        state.final_pose_dir,
        "input_2D",
        "keypoints_raw.npz",
    )
    apply_foot_leg_correction(
        state.foot_npz,
        raw_keypoints_npz,
        state.keypoints_npz,
        swapped_mask,
    )
    swap_info = _write_dp_leg_swap_mask(
        LegSwapMaskRequest(
            swapped_mask=swapped_mask,
            output_dir=context.output_dest,
            pre_dp_swapped_mask=pre_dp_swapped_mask,
            anchor_dp_swapped_mask=anchor_swapped_mask,
        )
    )
    return swapped_mask, swap_info


def _camera_video_fps(context: AnalysisContext) -> float | None:
    """讀取第一台相機的 FPS；影片無法開啟時回傳 None。"""
    return _read_video_frames_per_second(
        _first_camera_video_path(context),
        None,
    )


@dataclass(frozen=True)
class LegIdentityAngleUpdateRequest:
    """描述腿部身份修正後的角度資料更新工作。"""

    context: AnalysisContext
    state: AnalysisState
    swapped_mask: np.ndarray


def _update_angles_after_leg_correction(
    request: LegIdentityAngleUpdateRequest,
) -> None:
    """以修正後關鍵點重算角度；失敗時同步既有角度 CSV。"""
    context = request.context
    state = request.state
    recomputed_angle_csv = _rerun_3d_angles_from_corrected_2d(
        CorrectedPose3DRequest(
            tracked_video_path=state.tracked_video,
            pose_output_dir=state.final_pose_dir,
            analysis_output_dir=context.output_dest,
            gpu=context.options.gpu,
            motion_ag_dir=context.motion_ag_dir,
            timings=context.timings,
        )
    )
    if recomputed_angle_csv:
        state.angles_csv = recomputed_angle_csv
        _add_time_to_angles_csv(
            AngleTimeRequest(
                angle_csv_path=state.angles_csv,
                frames_per_second=_camera_video_fps(context),
            )
        )
        return

    angle_sync_started_at = time.perf_counter()
    angle_sync = _align_angle_csv_to_leg_identity(
        AngleCsvAlignmentRequest(
            angle_csv_path=state.angles_csv,
            swapped_mask=request.swapped_mask,
            output_dir=context.output_dest,
        )
    )
    _record_timing(
        context.timings,
        "Analysis/sync_angle_csv_to_leg_identity_fallback",
        angle_sync_started_at,
        swapped_frames=(angle_sync.get("swapped_frames") if angle_sync else 0),
    )


def _refresh_step_analysis(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """使用已修正的腿部身份重新計算步態分析結果。"""
    assert state.step_analysis is not None
    assert state.keypoints_npz is not None
    assert state.offsets_npz is not None
    assert state.foot_npz is not None

    state.step_analysis = refresh_step_analysis_after_leg_correction(
        step_analysis=state.step_analysis,
        config=context.config,
        output_dir=context.output_dest,
        keypoints_npz=state.keypoints_npz,
        offsets_npz=state.offsets_npz,
        foot_npz=state.foot_npz,
    )
    state.avg_step_length = state.step_analysis.get("avg_step_length_m")


def run_leg_identity_analysis(
    context: AnalysisContext,
    state: AnalysisState,
) -> bool:
    """依序協調步態分析、腿部身份修正及角度更新。"""
    _normalize_pose_output_dir(context, state)
    if not _prepare_leg_identity_paths(context, state):
        return False

    context.report_progress(PROGRESS_LEG_IDENTITY_STARTED)
    print("\n" + OUTPUT_SEPARATOR)
    print("【階段三/四a】步頻分析 → 骨架左右腳修正 → 骨架影片疊加")
    print(OUTPUT_SEPARATOR)
    state.step_overlay_started_at = time.perf_counter()

    step_started_at = time.perf_counter()
    _run_initial_step_analysis(context, state)
    assert state.step_analysis is not None
    _record_timing(
        context.timings,
        "Analysis/step_stride_analysis",
        step_started_at,
        detected_steps=state.step_analysis.get("detected_steps"),
    )

    leg_fix_started_at = time.perf_counter()
    swapped_mask, swap_info = _correct_leg_identity(context, state)
    _record_timing(
        context.timings,
        "Analysis/apply_anchor_leg_correction",
        leg_fix_started_at,
    )

    if context.options.pose_scope is PoseScope.TWO_D_AND_3D:
        _update_angles_after_leg_correction(
            LegIdentityAngleUpdateRequest(
                context=context,
                state=state,
                swapped_mask=swapped_mask,
            )
        )
    elif swap_info:
        _record_timing(
            context.timings,
            "Analysis/write_dp_leg_swap_mask",
            leg_fix_started_at,
            swapped_frames=swap_info.get("swapped_frames", 0),
        )

    refresh_started_at = time.perf_counter()
    _refresh_step_analysis(context, state)
    _record_timing(
        context.timings,
        "Analysis/refresh_step_analysis_after_leg_correction",
        refresh_started_at,
    )
    return True


def _create_main_overlay_video(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """建立原始影片骨架疊圖並記錄階段耗時。"""
    assert state.output_video is not None
    assert state.offsets_npz is not None
    assert state.keypoints_npz is not None
    overlay_started_at = time.perf_counter()
    overlay_videos(
        cameras=context.cameras,
        offsets_npz=state.offsets_npz,
        kps_npz=state.keypoints_npz,
        output_video=state.output_video,
        config=context.config,
    )
    _record_timing(
        context.timings,
        "Analysis/overlay_original_video",
        overlay_started_at,
        output_video=state.output_video,
    )
    if state.step_overlay_started_at is not None:
        _record_timing(
            context.timings,
            "Analysis/step_and_overlay_block",
            state.step_overlay_started_at,
        )


def _print_step_analysis_summary(state: AnalysisState) -> None:
    """輸出步態分析摘要。"""
    assert state.step_analysis is not None
    print(f"  ▶ 腳踝位置資料 (CSV): {state.step_analysis['ankle_csv']}")
    print(f"  ▶ 步伐事件資料 (CSV): {state.step_analysis['steps_csv']}")
    print(f"  ▶ 偵測步數: {state.step_analysis['detected_steps']}")
    if state.step_analysis.get("avg_cadence_spm") is not None:
        print(f"  ▶ 平均步頻: {state.step_analysis['avg_cadence_spm']:.2f} steps/min")
    if state.avg_step_length is not None:
        print(f"  ▶ 平均步幅: {state.avg_step_length:.2f} m")


def _annotate_main_overlay_video(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """把步伐事件標注合成到主要疊圖影片。"""
    assert state.output_video is not None
    assert state.step_analysis is not None
    print("\n" + OUTPUT_SEPARATOR)
    print("【階段四b】步伐標注影片合成")
    print(OUTPUT_SEPARATOR)
    temporary_output = state.output_video.replace(".mp4", "_tmp_steps.mp4")
    annotate_started_at = time.perf_counter()
    annotate_step_stride_video(
        input_video=state.output_video,
        output_video=temporary_output,
        ankle_rows=state.step_analysis["ankle_rows"],
        step_events=state.step_analysis["step_events"],
    )
    os.replace(temporary_output, state.output_video)
    _record_timing(
        context.timings,
        "Analysis/annotate_step_stride_video",
        annotate_started_at,
        output_video=state.output_video,
    )


def _export_per_camera_review_videos(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """建立並轉碼各相機獨立回顧影片。"""
    assert state.offsets_npz is not None
    assert state.keypoints_npz is not None
    assert state.step_analysis is not None
    per_camera_started_at = time.perf_counter()
    per_camera_output_paths = [
        os.path.join(context.output_dest, f"cam{index + 1}_overlay.mp4")
        for index in range(len(context.cameras))
    ]
    try:
        overlay_videos_per_camera(
            cameras=context.cameras,
            offsets_npz=state.offsets_npz,
            kps_npz=state.keypoints_npz,
            ankle_rows=state.step_analysis["ankle_rows"],
            step_events=state.step_analysis["step_events"],
            output_paths=per_camera_output_paths,
            config=context.config,
        )
        for camera_output_path in per_camera_output_paths:
            if os.path.exists(camera_output_path):
                convert_to_web_compatible_mp4(camera_output_path)
        _record_timing(
            context.timings,
            "Analysis/overlay_videos_per_camera",
            per_camera_started_at,
            output_videos=per_camera_output_paths,
        )
    # Per-camera review videos are optional and call third-party video code
    # whose exception types are not part of its interface.
    except Exception as error:  # noqa: BLE001
        print(f"  ▶ 各相機獨立疊圖產生失敗（不影響主要分析結果）: {error}")


def _export_trial_topdown_reviews(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """建立相機與完整賽事俯視回顧影片。"""
    assert state.output_video is not None
    assert state.step_analysis is not None
    topdown_started_at = time.perf_counter()
    topdown_outputs = _export_homography_review_videos(
        HomographyReviewRequest(
            cameras=context.cameras,
            steps_csv=state.step_analysis.get("steps_csv"),
            options=HomographyReviewOptions(
                output_dest=context.output_dest,
                timeline_video=state.output_video,
                camera_schematic_output=(
                    VideoOutput.GENERATE
                    if context.config.get("schematic_topdown_enabled", False)
                    else VideoOutput.OMIT
                ),
                camera_pixels_per_meter=float(
                    context.config.get("schematic_topdown_px_per_meter", 75.0)
                ),
                padding_pixels=int(
                    context.config.get("schematic_topdown_padding_px", 60)
                ),
                trial_pixels_per_meter=float(
                    context.config.get("full_trial_topdown_px_per_meter", 30.0)
                ),
            ),
        )
    )
    _record_timing(
        context.timings,
        "Analysis/homography_topdown_review_videos",
        topdown_started_at,
        output_videos=topdown_outputs,
    )


def _transcode_analysis_video(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """將主要分析影片轉為 Web 播放相容格式並記錄耗時。"""
    assert state.output_video is not None
    print("\n  ▶ 正在將影片轉換為 Web 播放相容格式...")
    transcode_started_at = time.perf_counter()
    convert_to_web_compatible_mp4(state.output_video)
    _record_timing(
        context.timings,
        "Analysis/transcode_web_compatible_mp4",
        transcode_started_at,
        output_video=state.output_video,
    )
    print(f"  ▶ [Core.Pipeline] 網頁串流格式轉檔成功: {state.output_video}")


def _update_final_angle_times(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """以最終影片 FPS 更新角度 CSV 時間欄位並記錄耗時。"""
    assert state.output_video is not None
    angle_time_started_at = time.perf_counter()
    _add_time_to_angles_csv(
        AngleTimeRequest(
            angle_csv_path=state.angles_csv,
            video_path=state.output_video,
        )
    )
    _record_timing(
        context.timings,
        "Analysis/update_angles_time_columns",
        angle_time_started_at,
        angles_csv=state.angles_csv,
    )


def _archive_final_analysis_video(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """將最終影片封存至 keypoints archive 並記錄耗時。"""
    assert state.output_video is not None
    print("  ▶ 略過最終影片逐幀 PNG 輸出（前端未使用）")
    archive_started_at = time.perf_counter()
    archived_output = _copy_output_final_to_keypoints_archive(
        state.final_pose_dir,
        state.output_video,
    )
    _record_timing(
        context.timings,
        "Analysis/archive_output_final_video",
        archive_started_at,
        archived_video=str(archived_output) if archived_output else None,
    )
    if archived_output:
        print(f"  ▶ 已複製 Web 相容影片到 keypoints archive: {archived_output}")


def _transcode_archive_and_cleanup_video(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """協調最終影片轉碼、角度更新、封存及中間檔清理。"""
    _transcode_analysis_video(context, state)
    _update_final_angle_times(context, state)
    _archive_final_analysis_video(context, state)
    _remove_intermediate_tracked_video(state.tracked_video)


def export_analysis_videos(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """依序建立、標注、轉碼並封存最終分析回顧影片。"""
    if (
        not state.output_video
        or not state.offsets_npz
        or not state.keypoints_npz
        or not state.step_analysis
    ):
        return

    _create_main_overlay_video(context, state)
    _print_step_analysis_summary(state)
    _annotate_main_overlay_video(context, state)
    _export_per_camera_review_videos(context, state)
    _export_trial_topdown_reviews(context, state)
    _transcode_archive_and_cleanup_video(context, state)


def _first_camera_fps(context: AnalysisContext) -> float:
    """讀取第一台相機 FPS，無法取得時使用 pipeline 預設值。"""
    frames_per_second = _read_video_frames_per_second(
        _first_camera_video_path(context),
        DEFAULT_VIDEO_FPS,
    )
    return frames_per_second or DEFAULT_VIDEO_FPS


def _read_primary_summary_metrics(
    context: AnalysisContext,
    state: AnalysisState,
) -> tuple[float | None, float | None, float | None]:
    """從主要 metrics.csv 讀取時間、平均速度與平均加速度。"""
    if not os.path.exists(state.metrics_csv):
        return None, None, None
    try:
        metrics = pd.read_csv(state.metrics_csv)
        if metrics.empty:
            return None, None, None
        total_time = float(
            (metrics["absolute_frame"].max() + 1) / _first_camera_fps(context)
        )
        return (
            total_time,
            float(metrics["speed_mps"].mean()),
            float(metrics["accel_mps2"].mean()),
        )
    except CSV_IO_ERRORS as error:
        print(f"指標計算異常: {error}")
        return None, None, None


def _estimate_total_time_from_frame_map(
    context: AnalysisContext,
    state: AnalysisState,
) -> float | None:
    """主要 metrics 缺失時，從 frame map 估算完整分析時間。"""
    frame_map_csv = os.path.join(
        state.track_output_dir,
        state.track_output_name.replace(".mp4", "_frame_map.csv"),
    )
    if not os.path.exists(frame_map_csv):
        return None
    try:
        frame_map = pd.read_csv(frame_map_csv)
        if frame_map.empty or "orig_frame" not in frame_map.columns:
            return None
        total_time = float(
            (frame_map["orig_frame"].max() + 1) / _first_camera_fps(context)
        )
        print(f"  [fallback] total_time 從 frame_map 估算: {total_time:.2f}s")
        return total_time
    except CSV_IO_ERRORS as error:
        print(f"  [fallback] total_time 計算異常: {error}")
        return None


def _finish_analysis_timing(context: AnalysisContext) -> str | None:
    """完成總耗時紀錄並明確寫出 timing report。"""
    _record_timing(
        context.timings,
        "Total/run_analysis",
        context.started_at,
        output_dest=context.output_dest,
    )
    return _write_timing_report(context.timings, context.output_dest)


def calculate_summary_metrics(
    context: AnalysisContext,
    state: AnalysisState,
) -> dict:
    """計算最終統計指標並組裝對外回傳的分析結果。"""
    total_time, average_velocity, average_acceleration = _read_primary_summary_metrics(
        context, state
    )
    if total_time is None:
        total_time = _estimate_total_time_from_frame_map(context, state)
    if total_time is None:
        total_time = 0.0
        print("  [warning] total_time 無法取得，設為 0.0")

    return {
        "metrics_csv": state.metrics_csv,
        "angles_csv": state.angles_csv,
        "uncropped_video": state.output_video,
        "timing_report": _finish_analysis_timing(context),
        "step_analysis": state.step_analysis,
        "total_time": total_time,
        "avg_velocity": average_velocity or 0.0,
        "avg_acceleration": average_acceleration or 0.0,
        "avg_step_length": state.avg_step_length,
    }


def _start_analysis(context: AnalysisContext) -> None:
    """顯示分析起始資訊並回報初始進度。"""
    print(OUTPUT_SEPARATOR)
    print("【階段一/二】骨架追蹤 + 2D 姿態估計")
    print(OUTPUT_SEPARATOR)
    context.report_progress(PROGRESS_ANALYSIS_STARTED)


def _run_base_analysis_pipeline(context: AnalysisContext) -> dict:
    """執行腿部身份修正前的追蹤與 2D 姿態階段。"""
    extra_config = {
        key: value for key, value in context.config.items() if key != "cameras"
    }
    return run_pipeline(
        PipelineRequest(
            cameras=context.tracking_cameras,
            extra_config=extra_config,
            options=PipelineOptions(
                output_dir=context.output_dest,
                gpu=context.options.gpu,
                # Full analysis corrects 2D leg identity before 3D lifting.
                pose_scope=PoseScope.TWO_D_ONLY,
                tracked_video_source=context.options.tracked_video_source,
                video_output=VideoOutput.OMIT,
                timings=context.timings,
            ),
        )
    )


def _analysis_state_from_pipeline_result(
    context: AnalysisContext,
    pipeline_result: dict,
) -> AnalysisState:
    """將底層 Pipeline 結果轉成後處理階段使用的狀態。"""
    return AnalysisState(
        tracked_video=pipeline_result.get("tracked_video"),
        track_output_dir=context.config.get("output_dir", "output_cut"),
        track_output_name=context.config.get(
            "output_name",
            "sequential_tracked.mp4",
        ),
        metrics_csv=os.path.join(context.output_dest, "metrics.csv"),
        final_pose_dir=pipeline_result.get("output_dir", "未定義"),
    )


def _run_analysis_post_processing(
    context: AnalysisContext,
    state: AnalysisState,
) -> None:
    """執行速度、腿部身份與影片輸出等可選後處理。"""
    run_speed_analysis(context, state)
    context.report_progress(PROGRESS_SPEED_ANALYSIS_COMPLETED)
    print("\n" + OUTPUT_SEPARATOR)

    try:
        if run_leg_identity_analysis(context, state):
            export_analysis_videos(context, state)
    # Optional post-processing must not discard the primary tracking result.
    except Exception as error:  # noqa: BLE001
        print(f"匯出未裁切影片失敗: {error}")


def run_analysis(
    analysis_config: dict,
    options: AnalysisOptions | None = None,
) -> dict:
    """依具名選項協調並執行完整跑者分析流程。"""
    context = prepare_analysis_context(
        analysis_config,
        options or AnalysisOptions(),
    )
    _start_analysis(context)
    pipeline_result = _run_base_analysis_pipeline(context)
    _remove_stale_angle_csv(context.output_dest)
    context.report_progress(PROGRESS_POSE_COMPLETED)
    state = _analysis_state_from_pipeline_result(context, pipeline_result)
    _run_analysis_post_processing(context, state)
    print("\n" + OUTPUT_SEPARATOR)
    context.report_progress(PROGRESS_ANALYSIS_COMPLETED)
    return calculate_summary_metrics(context, state)
