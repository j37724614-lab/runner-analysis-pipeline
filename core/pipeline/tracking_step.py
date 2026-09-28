"""Step 1：多相機 YOLO two-pass 追蹤與置中裁剪的排程（呼叫 core.tracking）。"""
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from core import tracking
from core.pipeline import (
    DEFAULT_VIDEO_FPS,
    OUTPUT_SEPARATOR,
    YOLO_WARMUP_FRAME_SHAPE,
    _record_timing,
)
from core.pose_stream import PoseStreamSession
from core.tracking_runtime import TrackingRuntimeOptions
from core.tracking_runtime import (
    temporary_tracking_runtime as _temporary_tracking_runtime,
)


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
    stream_pose: bool = False
    pose_model_path: str | None = None
    write_tracked_video: bool = True
    gpu: str = "0"


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
            tracker_config=tracking.TWO_PASS_TRACKER_CONFIG,
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
    video_writer = (
        _create_tracking_video_writer(context.output_path, context.frames_per_second)
        if context.write_tracked_video else _DiscardVideoWriter()
    )
    pose_stream = None
    try:
        if context.stream_pose:
            pose_stream = PoseStreamSession(
                video_path=context.output_path,
                output_dir=os.path.join(
                    context.output_dir, Path(context.output_path).stem,
                ),
                gpu=context.gpu,
                model_path=context.pose_model_path,
            )
        if pose_stream is not None:
            print("two_pass 模式：第二遍追焦影格送入 HRNet（快取模式，跳過 YOLO）...")
        else:
            print("two_pass 模式：第二遍輸出追焦影片（快取模式，跳過 YOLO）...")
        pass2_started_at = time.perf_counter()
        tracked_frame_count = tracking._process_cameras(tracking.CameraBatchRequest(
            captures=context.video_captures,
            cameras=context.active_cameras,
            output_writer=video_writer,
            frame_map_path=context.frame_map_path,
            preset_target_ids=selected_tracks.runner_ids,
            frame_cache=selected_tracks.candidates.frame_cache,
            frame_ranges_by_camera=selected_tracks.candidates.frame_ranges_by_camera,
            frame_sink=pose_stream,
        ))
        _record_timing(
            context.timings,
            (
                "Step1/two_pass_pass2_stream_frames" if pose_stream is not None
                else "Step1/two_pass_pass2_write_tracked_video"
            ),
            pass2_started_at,
            output_path=context.output_path,
            crop_width=int(tracking.CROP_WIDTH),
            crop_height=int(tracking.CROP_HEIGHT),
        )
        if pose_stream is not None:
            pose_started_at = time.perf_counter()
            pose_frames = pose_stream.finish()
            if tracked_frame_count != pose_frames:
                raise RuntimeError(
                    f"追蹤與 HRNet 幀數不符：{tracked_frame_count} != {pose_frames}"
                )
            _record_timing(
                context.timings, "Step2/pose_stream_finish_wait",
                pose_started_at, frames=pose_frames,
            )
            if context.timings is not None:
                context.timings.append({
                    "stage": "Step2/hrnet_stream_worker_total",
                    "elapsed_sec": pose_stream.pose_elapsed_sec,
                    "frames": pose_frames,
                    "overlaps_pass2": True,
                    "queue_capacity": pose_stream.capacity,
                    "producer_enqueue_sec": round(pose_stream.enqueue_sec, 4),
                    "producer_blocked_sec": round(pose_stream.blocked_sec, 4),
                    "queue_full_count": pose_stream.queue_full_count,
                    "queue_full_frames": pose_stream.queue_full_frames,
                    "producer_queue_full_retries": pose_stream.queue_full_retries,
                    "consumer_queue_wait_sec": pose_stream.worker_queue_wait_sec,
                    "consumer_idle_sec": pose_stream.consumer_idle_sec,
                    "consumer_frame_span_sec": pose_stream.worker_frame_span_sec,
                    **pose_stream.queue_depth_metrics,
                    **{f"hrnet_{key}": value for key, value in pose_stream.pose_metrics.items()},
                })
            print(f"  ⏱ [TIME] Step2/hrnet_stream_worker_total: {pose_stream.pose_elapsed_sec:.2f}s（與第二遍重疊）")
            metadata_path = context.output_path.replace(
                ".mp4", "_stream_metadata.json",
            )
            with open(metadata_path, "w", encoding="utf-8") as metadata_file:
                json.dump({
                    "width": int(tracking.CROP_WIDTH),
                    "height": int(tracking.CROP_HEIGHT),
                    "fps": context.frames_per_second,
                    "frames": pose_frames,
                }, metadata_file)
    except BaseException:
        if pose_stream is not None:
            pose_stream.abort()
        raise
    finally:
        video_writer.release()


class _DiscardVideoWriter:
    """Retain map/offset generation without an intermediate cropped MP4."""

    def write(self, _frame):
        pass

    def release(self):
        pass


def _run_two_pass_tracking(context: TrackingRunContext) -> None:
    """協調 two-pass 候選掃描、主跑者選擇、裁切與影片輸出。"""
    frame_ranges = _prescan_person_frame_ranges(context)
    candidates = _collect_candidate_runner_tracks(context, frame_ranges)
    selected_tracks = _select_and_stitch_runner_tracks(context, candidates)
    _write_runner_selection_debug(context, selected_tracks)
    _configure_two_pass_crop(context, selected_tracks)
    _render_two_pass_tracking(context, selected_tracks)


@dataclass(frozen=True)
class Step1TrackingRequest:
    """描述一次多相機追蹤與裁剪工作。"""

    camera_configs: list
    extra_config: dict
    gpu: str
    output_dir: str
    timings: list | None = None
    stream_pose: bool = False
    pose_model_path: str | None = None
    write_tracked_video: bool = True


def _step1_track_impl(request: Step1TrackingRequest) -> str:
    """協調 YOLO 多相機追蹤與跑者置中裁剪。"""
    if tracking.TRACKING_MODE != "two_pass":
        raise ValueError("僅支援 tracking_mode='two_pass'")
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
        stream_pose=request.stream_pose,
        pose_model_path=request.pose_model_path,
        write_tracked_video=request.write_tracked_video,
        gpu=request.gpu,
    )

    _run_two_pass_tracking(tracking_context)

    if request.write_tracked_video:
        print(f"\nStep 1 完成，置中裁剪影片儲存至：{output_path}\n")
    else:
        print("\nStep 1 完成，追焦影格已直接送入 HRNet（未產生中間 MP4）\n")
    _record_timing(
        request.timings,
        "Step1+Step2/tracking_and_stream_pose_total" if request.stream_pose else "Step1/total_tracking",
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

