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
from core.overlay import (
    PerCameraOverlayRequest,
    overlay_videos,
    overlay_videos_per_camera,
)
from core.pose_stream import PoseStreamSession
from core.process_runtime import (
    PROCESS_STATE_LOCK as _PROCESS_STATE_LOCK,
)
from core.process_runtime import (
    temporary_environment_variable as _temporary_environment_variable,
)
from core.tracker_impl import SpeedComputationRequest, compute_speed_from_bbox_map
from core.tracking_runtime import (
    TrackingRuntimeOptions,
)
from core.tracking_runtime import (
    temporary_tracking_runtime as _temporary_tracking_runtime,
)
from core.utils import REPO_ROOT, convert_to_web_compatible_mp4
from core.visualization import AngleOverlayConfig, add_angle_overlay
from scripts.analysis.ankle_step_stride import (
    StepAnalysisRefreshRequest,
    StepStrideAnalysisRequest,
    StepStrideAnnotationRequest,
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




# =======================================================================
# 對外公開介面：從各子模組匯入，讓 `from core import pipeline` / `from core.pipeline
# import X` 的既有呼叫方（routes/upload.py、run_pipeline.py、analyze.py、
# core/pose_stream.py、各測試與原型腳本）完全不用修改。完整匯出每個子模組的所有
# 頂層名稱（含私有函式），對應原本單一檔案時全部攤平在同一個模組命名空間的狀態。
# =======================================================================
from core.pipeline.angle_post import (
    AngleAlignmentResult,
    AngleCsvAlignmentRequest,
    AngleTimeRequest,
    CorrectedPose3DRequest,
    LegSwapMaskRequest,
    _add_angle_time_columns,
    _add_time_to_angles_csv,
    _align_angle_csv_to_leg_identity,
    _align_angle_dataframe_to_leg_identity,
    _archived_tracked_video_path,
    _coordinate_system_may_differ,
    _generate_corrected_3d_angles,
    _keypoint_coordinate_extent,
    _leg_swap_mask_dataframe,
    _padded_boolean_mask,
    _persist_angle_alignment,
    _publish_corrected_angle_csv,
    _read_video_frames_per_second,
    _rerun_3d_angles_from_corrected_2d,
    _resolve_angle_frames_per_second,
    _select_compatible_3d_video,
    _video_dimensions,
    _write_dp_leg_swap_mask,
)
from core.pipeline.final_export import (
    _analysis_state_from_pipeline_result,
    _annotate_main_overlay_video,
    _archive_final_analysis_video,
    _create_main_overlay_video,
    _estimate_total_time_from_frame_map,
    _export_per_camera_review_videos,
    _export_trial_topdown_reviews,
    _finish_analysis_timing,
    _print_step_analysis_summary,
    _read_primary_summary_metrics,
    _run_analysis_post_processing,
    _run_base_analysis_pipeline,
    _start_analysis,
    _transcode_analysis_video,
    _transcode_archive_and_cleanup_video,
    _update_final_angle_times,
    calculate_summary_metrics,
    export_analysis_videos,
    run_analysis,
)
from core.pipeline.homography_review import (
    CameraHomographyInputs,
    HomographyRenderJob,
    HomographyReviewOptions,
    HomographyReviewRequest,
    TrialHomographyInputs,
    _collect_trial_calibrations,
    _complete_homography_calibration,
    _export_camera_homography_reviews,
    _export_full_trial_topdown_review,
    _export_homography_review_videos,
    _export_rectified_camera_review,
    _export_schematic_camera_review,
    _last_camera_homography_inputs,
    _prepare_full_trial_render_job,
    _run_homography_render,
    _write_homography_control_points,
)
from core.pipeline.motionag_runtime import (
    _import_vis_module,
    _temporary_motion_agformer_runtime,
)
from core.pipeline.orchestration import (
    AnalysisContext,
    AnalysisOptions,
    AnalysisState,
    PipelineOptions,
    PipelineRequest,
    PipelineWorkspace,
    _find_existing_tracked_video,
    _normalize_analysis_config,
    _obtain_tracked_video,
    _pipeline_output_directory,
    _resolve_analysis_output_directory,
    _run_overlay_pipeline_step,
    _run_pose_pipeline_step,
    _tracking_camera_config,
    _tracking_camera_configs,
    _write_pipeline_config,
    prepare_analysis_context,
    run_pipeline,
)
from core.pipeline.pose_step import (
    OverlayWorkspace,
    PoseEstimationRequest,
    PoseWorkspace,
    _clear_stale_pose_outputs,
    _execute_pose_estimation,
    _load_overlay_main_video_paths,
    _overlay_inputs_exist,
    _prepare_overlay_workspace,
    _prepare_pose_workspace,
    _print_pose_estimation_plan,
    _render_angle_overlay,
    step2_pose,
    step3_overlay,
)
from core.pipeline.speed_and_legs import (
    LegIdentityAngleUpdateRequest,
    SpeedAnalysisPaths,
    _calculate_speed_metrics,
    _camera_video_fps,
    _correct_leg_identity,
    _execute_speed_analysis,
    _first_camera_fps,
    _first_camera_video_path,
    _load_pre_dp_leg_swap_mask,
    _normalize_pose_output_dir,
    _prepare_leg_identity_paths,
    _refresh_step_analysis,
    _requested_speed_mode,
    _run_initial_step_analysis,
    _speed_analysis_paths,
    _sync_leg_identity_outputs,
    _update_angles_after_leg_correction,
    _write_speed_metrics_csv,
    run_leg_identity_analysis,
    run_speed_analysis,
)
from core.pipeline.tracking_step import (
    CandidateRunnerTracks,
    SelectedRunnerTracks,
    Step1TrackingRequest,
    TrackingRunContext,
    _active_tracking_cameras,
    _collect_candidate_runner_tracks,
    _configure_two_pass_crop,
    _create_tracking_video_writer,
    _DiscardVideoWriter,
    _load_and_warm_up_tracking_model,
    _open_video_captures,
    _prescan_person_frame_ranges,
    _render_two_pass_tracking,
    _run_two_pass_tracking,
    _select_and_stitch_runner_tracks,
    _step1_track_impl,
    _tracking_output_name,
    _write_runner_selection_debug,
    _write_tracking_output_marker,
    step1_track,
)

__all__ = [
    "AnalysisContext",
    "AnalysisOptions",
    "AnalysisState",
    "AngleAlignmentResult",
    "AngleCsvAlignmentRequest",
    "AngleOverlayConfig",
    "AngleTimeRequest",
    "Callable",
    "CameraHomographyInputs",
    "CandidateRunnerTracks",
    "CorrectedPose3DRequest",
    "Enum",
    "HomographyRenderJob",
    "HomographyReviewOptions",
    "HomographyReviewRequest",
    "LegIdentityAngleUpdateRequest",
    "LegSwapMaskRequest",
    "OverlayWorkspace",
    "Path",
    "PerCameraOverlayRequest",
    "PipelineOptions",
    "PipelineRequest",
    "PipelineWorkspace",
    "PoseEstimationRequest",
    "PoseStreamSession",
    "PoseWorkspace",
    "REPO_ROOT",
    "SelectedRunnerTracks",
    "SpeedAnalysisPaths",
    "SpeedComputationRequest",
    "Step1TrackingRequest",
    "StepAnalysisRefreshRequest",
    "StepStrideAnalysisRequest",
    "StepStrideAnnotationRequest",
    "TrackingRunContext",
    "TrackingRuntimeOptions",
    "TrialHomographyInputs",
    "_DiscardVideoWriter",
    "_PROCESS_STATE_LOCK",
    "_active_tracking_cameras",
    "_add_angle_time_columns",
    "_add_time_to_angles_csv",
    "_align_angle_csv_to_leg_identity",
    "_align_angle_dataframe_to_leg_identity",
    "_analysis_state_from_pipeline_result",
    "_annotate_main_overlay_video",
    "_archive_final_analysis_video",
    "_archived_tracked_video_path",
    "_calculate_speed_metrics",
    "_camera_video_fps",
    "_clear_stale_pose_outputs",
    "_collect_candidate_runner_tracks",
    "_collect_trial_calibrations",
    "_complete_homography_calibration",
    "_configure_two_pass_crop",
    "_coordinate_system_may_differ",
    "_correct_leg_identity",
    "_create_main_overlay_video",
    "_create_tracking_video_writer",
    "_estimate_total_time_from_frame_map",
    "_execute_pose_estimation",
    "_execute_speed_analysis",
    "_export_camera_homography_reviews",
    "_export_full_trial_topdown_review",
    "_export_homography_review_videos",
    "_export_per_camera_review_videos",
    "_export_rectified_camera_review",
    "_export_schematic_camera_review",
    "_export_trial_topdown_reviews",
    "_find_existing_tracked_video",
    "_finish_analysis_timing",
    "_first_camera_fps",
    "_first_camera_video_path",
    "_generate_corrected_3d_angles",
    "_import_vis_module",
    "_keypoint_coordinate_extent",
    "_last_camera_homography_inputs",
    "_leg_swap_mask_dataframe",
    "_load_and_warm_up_tracking_model",
    "_load_overlay_main_video_paths",
    "_load_pre_dp_leg_swap_mask",
    "_normalize_analysis_config",
    "_normalize_pose_output_dir",
    "_obtain_tracked_video",
    "_open_video_captures",
    "_overlay_inputs_exist",
    "_padded_boolean_mask",
    "_persist_angle_alignment",
    "_pipeline_output_directory",
    "_prepare_full_trial_render_job",
    "_prepare_leg_identity_paths",
    "_prepare_overlay_workspace",
    "_prepare_pose_workspace",
    "_prescan_person_frame_ranges",
    "_print_pose_estimation_plan",
    "_print_step_analysis_summary",
    "_publish_corrected_angle_csv",
    "_read_primary_summary_metrics",
    "_read_video_frames_per_second",
    "_refresh_step_analysis",
    "_render_angle_overlay",
    "_render_two_pass_tracking",
    "_requested_speed_mode",
    "_rerun_3d_angles_from_corrected_2d",
    "_resolve_analysis_output_directory",
    "_resolve_angle_frames_per_second",
    "_run_analysis_post_processing",
    "_run_base_analysis_pipeline",
    "_run_homography_render",
    "_run_initial_step_analysis",
    "_run_overlay_pipeline_step",
    "_run_pose_pipeline_step",
    "_run_two_pass_tracking",
    "_select_and_stitch_runner_tracks",
    "_select_compatible_3d_video",
    "_speed_analysis_paths",
    "_start_analysis",
    "_step1_track_impl",
    "_sync_leg_identity_outputs",
    "_temporary_environment_variable",
    "_temporary_motion_agformer_runtime",
    "_temporary_tracking_runtime",
    "_tracking_camera_config",
    "_tracking_camera_configs",
    "_tracking_output_name",
    "_transcode_analysis_video",
    "_transcode_archive_and_cleanup_video",
    "_update_angles_after_leg_correction",
    "_update_final_angle_times",
    "_video_dimensions",
    "_write_dp_leg_swap_mask",
    "_write_homography_control_points",
    "_write_pipeline_config",
    "_write_runner_selection_debug",
    "_write_speed_metrics_csv",
    "_write_tracking_output_marker",
    "add_angle_overlay",
    "angle_csv_store",
    "annotate_step_stride_video",
    "apply_anchor_leg_correction",
    "apply_foot_leg_correction",
    "calculate_summary_metrics",
    "compute_speed_from_bbox_map",
    "contextmanager",
    "convert_to_web_compatible_mp4",
    "csv",
    "cv2",
    "dataclass",
    "export_analysis_videos",
    "field",
    "json",
    "np",
    "os",
    "overlay_videos",
    "overlay_videos_per_camera",
    "pd",
    "prepare_analysis_context",
    "refresh_step_analysis_after_leg_correction",
    "replace",
    "run_analysis",
    "run_leg_identity_analysis",
    "run_pipeline",
    "run_speed_analysis",
    "run_step_stride_analysis",
    "shutil",
    "step1_track",
    "step2_pose",
    "step3_overlay",
    "subprocess",
    "sys",
    "time",
    "tracking",
    "update_leg_swap_metadata",
]
