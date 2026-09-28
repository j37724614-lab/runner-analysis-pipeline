"""速度計算，以及步頻分析／DP 左右腿身份修正／修正後角度更新的協調。"""
import csv
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.pipeline import (
    DEFAULT_VIDEO_FPS,
    OUTPUT_SEPARATOR,
    PROGRESS_LEG_IDENTITY_STARTED,
    SPEED_METRIC_FIELD_NAMES,
    PoseScope,
    TrackedVideoSource,
    _record_timing,
)
from core.tracker_impl import SpeedComputationRequest, compute_speed_from_bbox_map
from scripts.analysis.ankle_step_stride import (
    StepAnalysisRefreshRequest,
    StepStrideAnalysisRequest,
    apply_anchor_leg_correction,
    apply_foot_leg_correction,
    refresh_step_analysis_after_leg_correction,
    run_step_stride_analysis,
    update_leg_swap_metadata,
)

from .angle_post import (
    AngleCsvAlignmentRequest,
    AngleTimeRequest,
    CorrectedPose3DRequest,
    LegSwapMaskRequest,
    _add_time_to_angles_csv,
    _align_angle_csv_to_leg_identity,
    _read_video_frames_per_second,
    _rerun_3d_angles_from_corrected_2d,
    _write_dp_leg_swap_mask,
)
from .orchestration import AnalysisContext, AnalysisState


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


def _first_camera_fps(context: AnalysisContext) -> float:
    """讀取第一台相機 FPS，無法取得時使用 pipeline 預設值。

    跟 final_export.py 共用，放在這裡（而非它原本所在的 final_export 區塊）
    是為了避免 speed_and_legs.py 與 final_export.py 互相 import 造成循環。
    """
    frames_per_second = _read_video_frames_per_second(
        _first_camera_video_path(context),
        DEFAULT_VIDEO_FPS,
    )
    return frames_per_second or DEFAULT_VIDEO_FPS




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
    return compute_speed_from_bbox_map(SpeedComputationRequest(
        bbox_map_csv=paths.bbox_map,
        cameras=context.cameras,
        fps_override=_first_camera_fps(context),
        offsets_npz=paths.offsets_npz,
        pixel_cameras=context.tracking_cameras,
        speed_mode=_requested_speed_mode(context),
    ))


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
    """執行首次步頻分析，產生 touchdown 錨點供腿部身份 DP 使用。"""
    state.step_analysis = run_step_stride_analysis(StepStrideAnalysisRequest(
        config=context.config,
        output_dir=context.output_dest,
    ))


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
    return _sync_leg_identity_outputs(context, state, anchor_swapped_mask)


def _sync_leg_identity_outputs(
    context: AnalysisContext,
    state: AnalysisState,
    anchor_swapped_mask,
):
    """同步腿部交換遮罩、WholeBody 足部點與診斷輸出。"""
    assert state.keypoints_npz is not None
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
            video_metadata=state.tracked_video_metadata,
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
        StepAnalysisRefreshRequest(
            step_analysis=state.step_analysis,
            config=context.config,
            output_dir=context.output_dest,
            keypoints_npz=state.keypoints_npz,
            offsets_npz=state.offsets_npz,
            foot_npz=state.foot_npz,
        )
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

