"""最終回顧影片輸出（骨架疊圖、逐相機回顧、俯視回顧）、統計彙總，以及 run_analysis() 總入口。"""
import os
import time

import pandas as pd

from core.overlay import (
    PerCameraOverlayRequest,
    overlay_videos,
    overlay_videos_per_camera,
)
from core.pipeline import (
    CSV_IO_ERRORS,
    OUTPUT_SEPARATOR,
    PROGRESS_ANALYSIS_COMPLETED,
    PROGRESS_ANALYSIS_STARTED,
    PROGRESS_POSE_COMPLETED,
    PROGRESS_SPEED_ANALYSIS_COMPLETED,
    PoseScope,
    VideoOutput,
    _copy_output_final_to_keypoints_archive,
    _record_timing,
    _remove_intermediate_tracked_video,
    _remove_stale_angle_csv,
    _write_timing_report,
)
from core.utils import convert_to_web_compatible_mp4
from scripts.analysis.ankle_step_stride import (
    StepStrideAnnotationRequest,
    annotate_step_stride_video,
)

from .angle_post import AngleTimeRequest, _add_time_to_angles_csv
from .homography_review import (
    HomographyReviewOptions,
    HomographyReviewRequest,
    _export_homography_review_videos,
)
from .orchestration import (
    AnalysisContext,
    AnalysisOptions,
    AnalysisState,
    PipelineOptions,
    PipelineRequest,
    prepare_analysis_context,
    run_pipeline,
)
from .speed_and_legs import (
    _first_camera_fps,
    run_leg_identity_analysis,
    run_speed_analysis,
)


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
    annotate_step_stride_video(StepStrideAnnotationRequest(
        input_video=state.output_video,
        output_video=temporary_output,
        ankle_rows=state.step_analysis["ankle_rows"],
        step_events=state.step_analysis["step_events"],
    ))
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
        overlay_videos_per_camera(PerCameraOverlayRequest(
            cameras=context.cameras,
            offsets_npz=state.offsets_npz,
            keypoints_npz=state.keypoints_npz,
            ankle_rows=state.step_analysis["ankle_rows"],
            step_events=state.step_analysis["step_events"],
            output_paths=per_camera_output_paths,
            config=context.config,
        ))
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
        tracked_video_metadata=pipeline_result.get("tracked_video_metadata"),
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
