"""Pass 2：逐相機輸出追焦影格、frame_map/bbox_map/offsets，並處理相機切換。"""

import csv
import os
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from core import tracking as _tracking
from core.tracking_debug import TrackingOverviewWriter
from core.tracking_geometry import _bbox_bottom_center, _project_onto_track

from .camera_switch import _CameraSwitchContext, _should_switch_camera
from .crop import _crop_from_bbox, _interpolate_bbox
from .pass1 import _advance_to_next_valid_range
from .pass2_frame import FrameProcessingConfig, process_frame
from .prescan import _frame_ranges_for_camera

_CameraResult = namedtuple('_CameraResult', [
    'written', 'skipped',
    'frame_map_rows', 'bbox_map_rows',
    'offsets', 'orig_frames', 'cam_indices',
    'output_frame_idx',
])


_CameraSetup = namedtuple('_CameraSetup', [
    'vid_w', 'vid_h', 'fps', 'total', 'valid_ranges',
    'overlay_start_pts', 'overlay_end_pts', 'track_roi',
])


@dataclass(frozen=True)
class _CameraProcessingRequest:
    """單次相機追蹤所需的穩定輸入。"""

    camera_index: int
    camera: dict
    capture: object
    output_writer: object | None
    frame_map_path: str | None
    preset_target_ids: dict | None
    frame_cache: dict | None
    frame_ranges_by_camera: dict | None
    camera_count: int
    output_frame_index: int
    frame_sink: object | None = None

    @property
    def is_last_camera(self) -> bool:
        """目前相機是否為本次分析的最後一台。"""
        return self.camera_index == self.camera_count - 1


def _camera_video_metadata(request):
    capture = request.capture
    if not capture.isOpened():
        raise ValueError(
            f"無法開啟相機 {request.camera_index + 1}: "
            f"{request.camera['video_path']}"
        )
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    ranges = _frame_ranges_for_camera(
        request.frame_ranges_by_camera, request.camera_index, total_frames,
    )
    return width, height, fps, total_frames, ranges


def _print_camera_setup(request, metadata, track_roi):
    width, height, fps, total_frames, valid_ranges = metadata
    camera = request.camera
    print(f"{'─' * 60}")
    print(
        f"相機 {request.camera_index + 1}/{request.camera_count}: "
        f"{camera['video_path']}"
    )
    print(f"  解析度: {width}x{height}，幀數: {total_frames}，FPS: {fps:.1f}")
    if valid_ranges:
        kept_frames = sum(end - start + 1 for start, end in valid_ranges)
        print(f"  pre-scan range: {valid_ranges}，處理 {kept_frames}/{total_frames} 幀")
    _print_camera_switch_setup(request, track_roi)


def _print_camera_switch_setup(request, track_roi):
    camera = request.camera
    if camera.get('H_matrix') is not None and camera.get('distance_m') is not None:
        print(f"  切換條件: Homography 距離 ≥ {float(camera['distance_m']):.2f}m")
    elif track_roi is not None:
        print(
            f"  ROI 模式: 斜線投影（pixel_span={camera['pixel_span']:.0f}px，"
            f"pre_roll={track_roi['pre_roll_px']}px）"
        )
        print("  切換條件: 投影距離 ≥ pixel_span（越過終點線）")
    elif camera.get('switch_x'):
        reference = 'bx2（右緣）' if request.is_last_camera else 'center_x'
        print(f"  切換條件: 最快人物 {reference}（原始座標）> {camera['switch_x']}px")
    else:
        print("  切換條件: 跑完整段影片")


def _overlay_line_points(camera):
    if not camera.get('start_line') or not camera.get('end_line'):
        return None, None

    def adjusted(points):
        return tuple(
            (int(point[0]), int(point[1]))
            for point in points
        )

    return adjusted(camera['start_line']), adjusted(camera['end_line'])


def _setup_camera_context(request: _CameraProcessingRequest):
    """驗證相機可用、印出相機資訊，並算出 ROI/疊加線設定。
    回傳 _CameraSetup。
    """
    metadata = _camera_video_metadata(request)
    vid_w, vid_h, fps, total, valid_ranges = metadata
    track_roi = request.camera.get('track_roi')
    _print_camera_setup(request, metadata, track_roi)
    overlay_start_pts, overlay_end_pts = _overlay_line_points(request.camera)

    return _CameraSetup(
        vid_w=vid_w, vid_h=vid_h, fps=fps, total=total, valid_ranges=valid_ranges,
        overlay_start_pts=overlay_start_pts, overlay_end_pts=overlay_end_pts,
        track_roi=track_roi,
    )


@dataclass
class _CameraRun:
    """Mutable accumulators for one camera's frame loop, so the loop body and
    the _write_output_frame / _flush_pending_missing closures share structured
    state instead of ~15 nonlocals."""

    output_frame_idx: int
    written: int = 0
    skipped: int = 0
    interpolated: int = 0
    frame_map_rows: list = field(default_factory=list)
    bbox_map_rows: list = field(default_factory=list)
    offsets: list = field(default_factory=list)
    orig_frames: list = field(default_factory=list)
    cam_indices: list = field(default_factory=list)
    last_valid_bbox: object = None
    pending_missing: list = field(default_factory=list)  # YOLO 暫時缺失的幀，等下一個有效 bbox 後補回


def _resolve_cached_detections(frame_cache, cam_idx, source_frame, preset_target_ids):
    """讀取相機及來源幀的 two-pass 快取，並只保留已選主跑者。"""
    if frame_cache is None:
        raise ValueError("two-pass 第二遍必須提供 frame_cache")
    if preset_target_ids is None or cam_idx not in preset_target_ids:
        raise ValueError(f"相機 {cam_idx + 1} 缺少已選主跑者 ID")
    cached = frame_cache.get((cam_idx, source_frame), [])
    target_id = preset_target_ids[cam_idx]
    return [d for d in cached if d['track_id'] == target_id]


class _FrameOutcome(Enum):
    """What the per-frame handler tells the camera loop to do next."""

    NEXT = "next"   # frame handled, continue with the next one
    SKIP = "skip"   # frame handled but must not be written (e.g. behind the
                     # start line); continue with the next one
    STOP = "stop"   # end this camera (past the end line / camera switch)


def _queue_or_skip_missing_frame(run, frame, source_frame):
    """主跑者暫時缺失時，有左側 bbox 才保留影格供稍後插值。"""
    if run.last_valid_bbox is not None:
        run.pending_missing.append({'frame': frame.copy(), 'source_frame': source_frame})
    else:
        run.skipped += 1


def _track_position_status(cam, entry):
    """For a slanted-ROI camera: is the runner still on the tracked stretch?
    Returns 'ok', 'behind_start' (before the start line -- skip this frame) or
    'past_end' (past the end line -- stop this camera)."""
    ground_x, ground_y = entry.get('ground_point', _bbox_bottom_center(entry['bbox']))
    proj_px = _project_onto_track(
        (ground_x, ground_y),
        cam['start_mid'], cam['track_dir'],
    )
    if proj_px < 0:
        return 'behind_start', proj_px
    if cam.get('pixel_span') and proj_px > cam['pixel_span']:
        return 'past_end', proj_px
    return 'ok', proj_px


@dataclass(frozen=True)
class _OutputFrame:
    """描述一幀即將寫入追蹤輸出的資料。"""

    image: object
    source_frame: int
    bbox: object = None
    track_id: object = None
    is_interpolated: bool = False
    interpolation_gap_length: int = 0
    offset_x: int = 0
    offset_y: int = 0


@dataclass(frozen=True)
class TrackedFramePacket:
    """One authoritative output frame and its pose/mapping coordinates."""

    output_frame: int
    camera_index: int
    source_frame: int
    image: object
    bbox: object
    offset_x: int
    offset_y: int
    track_id: object
    is_interpolated: bool
    interpolation_gap_length: int


@dataclass(frozen=True)
class _SourceFrame:
    """保存相機迴圈讀取的一個來源影格。"""

    image: object
    source_index: int
    next_frame_count: int


@dataclass(frozen=True)
class _TrackedFrame:
    """保存 `process_frame()` 對一個來源影格的追蹤結果。"""

    cropped_image: object
    runner_id: object
    runner_center_x: object
    runner_right_x: object
    bbox_in_crop: object
    offset_x: int
    offset_y: int


class _SingleCameraProcessor:
    """隱藏單相機逐幀追蹤、輸出與收尾細節。"""

    def __init__(self, request: _CameraProcessingRequest):
        self.request = request
        self.setup = _setup_camera_context(request)
        self.run = _CameraRun(output_frame_idx=request.output_frame_index)
        self.runner_states = {}
        self.range_cursor = 0
        self.frame_count = 0
        self.overview = TrackingOverviewWriter(
            enabled=_tracking.WRITE_OVERVIEW_VIDEO,
            frame_map_path=request.frame_map_path,
            camera_index=request.camera_index,
            fps=self.setup.fps,
            frame_size=(self.setup.vid_w, self.setup.vid_h),
        )

    def process(self):
        """處理整支相機影片並回傳該相機的累積結果。"""
        self._position_capture_at_first_valid_range()
        self._announce_processing_started()
        try:
            self._process_source_frames()
        finally:
            self._release_video_resources()
        self._discard_trailing_missing_frames()
        self._announce_processing_finished()
        return self._build_result()

    def _position_capture_at_first_valid_range(self):
        if not self.setup.valid_ranges:
            return
        self.frame_count = self.setup.valid_ranges[0][0]
        self.request.capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_count)

    def _announce_processing_started(self):
        print("  [處理中...]")

    def _process_source_frames(self):
        capture = self.request.capture
        while capture.isOpened():
            if not self._advance_to_valid_range():
                break
            read_succeeded, image = capture.read()
            if not read_succeeded:
                break
            source = _SourceFrame(
                image=image,
                source_index=self.frame_count,
                next_frame_count=self.frame_count + 1,
            )
            self.frame_count = source.next_frame_count
            if self._process_source_frame(source) is _FrameOutcome.STOP:
                break

    def _advance_to_valid_range(self) -> bool:
        previous_cursor = self.range_cursor
        self.range_cursor, self.frame_count, exhausted = (
            _advance_to_next_valid_range(
                self.request.capture,
                self.setup.valid_ranges,
                self.range_cursor,
                self.frame_count,
            )
        )
        if self.range_cursor != previous_cursor:
            self.run.pending_missing.clear()
        return not exhausted

    def _process_source_frame(self, source: _SourceFrame):
        tracked = self._track_source_frame(source)
        self.overview.write(
            source.image,
            tracks=self.runner_states,
            selected_id=tracked.runner_id,
            crop_offset=(0, 0),
            quadrilateral=self.request.camera.get('quad_roi'),
        )

        if tracked.cropped_image is None:
            _queue_or_skip_missing_frame(
                self.run,
                source.image,
                source.source_index,
            )
            return _FrameOutcome.NEXT

        current_bbox = self._record_current_bbox(tracked.runner_id)
        track_outcome = self._check_track_position(tracked.runner_id)
        if track_outcome is not _FrameOutcome.NEXT:
            return track_outcome

        if self.run.pending_missing and current_bbox is not None:
            self._flush_pending_missing(current_bbox, tracked)
        self._write_tracked_frame(source, tracked)
        self._print_periodic_progress(source.next_frame_count)

        if current_bbox is not None:
            self.run.last_valid_bbox = current_bbox
        return self._camera_switch_outcome(tracked)

    def _track_source_frame(self, source: _SourceFrame) -> _TrackedFrame:
        cached_detections = _resolve_cached_detections(
            self.request.frame_cache,
            self.request.camera_index,
            source.source_index,
            self.request.preset_target_ids,
        )
        config = self._frame_processing_config()
        result = process_frame(
            source.image,
            self.runner_states,
            config,
            cached_detections,
        )
        return _TrackedFrame(*result)

    def _frame_processing_config(self):
        camera = self.request.camera
        return FrameProcessingConfig(
            track_roi=self.setup.track_roi,
            quad_roi=camera.get('quad_roi'),
            homography_lane_margin_px=camera.get(
                'homography_lane_margin_px',
                80,
            ),
            overlay_start_pts=(
                self.setup.overlay_start_pts if _tracking.SHOW_OVERLAY else None
            ),
            overlay_end_pts=(
                self.setup.overlay_end_pts if _tracking.SHOW_OVERLAY else None
            ),
        )

    def _record_current_bbox(self, runner_id):
        entry = self.runner_states.get(runner_id)
        if entry is None:
            return None
        return tuple(entry['bbox'])

    def _check_track_position(self, runner_id):
        if (
            self.setup.track_roi is None
            or runner_id is None
        ):
            return _FrameOutcome.NEXT
        entry = self.runner_states.get(runner_id)
        if entry is None:
            return _FrameOutcome.NEXT
        status, projected_pixels = _track_position_status(
            self.request.camera,
            entry,
        )
        if status == 'behind_start':
            self.run.skipped += 1
            return _FrameOutcome.SKIP
        if status == 'past_end':
            print(
                f"  → 停止輸出：投影={projected_pixels:.0f}px > "
                f"end_line={self.request.camera['pixel_span']:.0f}px"
            )
            return _FrameOutcome.STOP
        return _FrameOutcome.NEXT

    def _flush_pending_missing(self, right_bbox, tracked):
        if self.run.last_valid_bbox is None or right_bbox is None:
            return
        gap_length = len(self.run.pending_missing)
        for index, item in enumerate(self.run.pending_missing, start=1):
            ratio = index / (gap_length + 1)
            interpolated_bbox = _interpolate_bbox(
                self.run.last_valid_bbox,
                right_bbox,
                ratio,
            )
            image, bbox, offset_x, offset_y = _crop_from_bbox(
                item['frame'],
                interpolated_bbox,
                label_interpolated=True,
                track_id=tracked.runner_id,
            )
            if image is None:
                continue
            self._write_output_frame(
                _OutputFrame(
                    image=image,
                    source_frame=item['source_frame'],
                    bbox=bbox,
                    track_id=tracked.runner_id,
                    is_interpolated=True,
                    interpolation_gap_length=gap_length,
                    offset_x=(
                        offset_x if offset_x is not None else tracked.offset_x
                    ),
                    offset_y=(
                        offset_y if offset_y is not None else tracked.offset_y
                    ),
                )
            )
            self.run.interpolated += 1
        self.run.pending_missing.clear()

    def _write_tracked_frame(self, source, tracked):
        self._write_output_frame(
            _OutputFrame(
                image=tracked.cropped_image,
                source_frame=source.source_index,
                bbox=tracked.bbox_in_crop,
                track_id=tracked.runner_id,
                offset_x=tracked.offset_x,
                offset_y=tracked.offset_y,
            )
        )

    def _write_output_frame(self, output: _OutputFrame):
        assert self.request.output_writer is not None
        self.request.output_writer.write(output.image)
        self._record_frame_mapping(output)
        if output.bbox is not None:
            self._record_bbox_mapping(output)
        self.run.offsets.append([output.offset_x, output.offset_y])
        self.run.orig_frames.append(output.source_frame)
        self.run.cam_indices.append(self.request.camera_index)
        if self.request.frame_sink is not None:
            self.request.frame_sink.emit(TrackedFramePacket(
                output_frame=self.run.output_frame_idx,
                camera_index=self.request.camera_index,
                source_frame=output.source_frame,
                image=output.image,
                bbox=output.bbox,
                offset_x=output.offset_x,
                offset_y=output.offset_y,
                track_id=output.track_id,
                is_interpolated=output.is_interpolated,
                interpolation_gap_length=output.interpolation_gap_length,
            ))
        self.run.written += 1
        self.run.output_frame_idx += 1

    def _record_frame_mapping(self, output):
        self.run.frame_map_rows.append({
            'output_frame': self.run.output_frame_idx,
            'cam': self.request.camera_index + 1,
            'cam_frame': output.source_frame,
            'source_frame': output.source_frame,
        })

    def _record_bbox_mapping(self, output):
        x1, y1, x2, y2 = output.bbox
        self.run.bbox_map_rows.append({
            'output_frame': self.run.output_frame_idx,
            'cam': self.request.camera_index + 1,
            'cam_frame': output.source_frame,
            'source_frame': output.source_frame,
            'track_id': output.track_id if output.track_id is not None else '',
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x2),
            'y2': int(y2),
            'is_interpolated': int(output.is_interpolated),
            'interp_gap_len': int(output.interpolation_gap_length),
        })

    def _print_periodic_progress(self, frame_count):
        if frame_count % 100 != 0:
            return
        print(
            f"  [幀 {frame_count}/{self.setup.total}] "
            f"主跑者狀態: {len(self.runner_states)} | "
            f"寫入: {self.run.written} | 捨棄: {self.run.skipped}"
        )

    def _camera_switch_outcome(self, tracked):
        should_switch, message = _should_switch_camera(
            _CameraSwitchContext(
                camera=self.request.camera,
                runner_states=self.runner_states,
                runner_id=tracked.runner_id,
                runner_center_x=tracked.runner_center_x,
                runner_right_x=tracked.runner_right_x,
                is_last_camera=self.request.is_last_camera,
            )
        )
        if not should_switch:
            return _FrameOutcome.NEXT
        if message:
            print(message)
        return _FrameOutcome.STOP

    def _release_video_resources(self):
        self.request.capture.release()
        self.overview.close()

    def _discard_trailing_missing_frames(self):
        self.run.skipped += len(self.run.pending_missing)
        self.run.pending_missing.clear()

    def _announce_processing_finished(self):
        print(
            f"  相機 {self.request.camera_index + 1} 完成："
            f"寫入 {self.run.written}，捨棄 {self.run.skipped}"
        )
        if self.run.interpolated:
            print(f"  補值: 已用前後有效 bbox 線性補回 {self.run.interpolated} 幀")

    def _build_result(self):
        return _CameraResult(
            written=self.run.written,
            skipped=self.run.skipped,
            frame_map_rows=self.run.frame_map_rows,
            bbox_map_rows=self.run.bbox_map_rows,
            offsets=self.run.offsets,
            orig_frames=self.run.orig_frames,
            cam_indices=self.run.cam_indices,
            output_frame_idx=self.run.output_frame_idx,
        )


def _process_single_camera(request: _CameraProcessingRequest):
    """透過單一 request 介面執行一台相機的完整追蹤流程。"""
    return _SingleCameraProcessor(request).process()


@dataclass(frozen=True)
class CameraBatchRequest:
    captures: list
    cameras: list
    output_writer: object | None
    frame_map_path: str | None = None
    preset_target_ids: dict | None = None
    frame_cache: dict | None = None
    frame_ranges_by_camera: dict | None = None
    frame_sink: object | None = None


@dataclass
class _CameraBatchResult:
    written: int = 0
    skipped: int = 0
    frame_map_rows: list = field(default_factory=list)
    bbox_map_rows: list = field(default_factory=list)
    offsets: list = field(default_factory=list)
    original_frames: list = field(default_factory=list)
    camera_indices: list = field(default_factory=list)
    output_frame_index: int = 0

    def include(self, camera_result):
        self.written += camera_result.written
        self.skipped += camera_result.skipped
        self.frame_map_rows.extend(camera_result.frame_map_rows)
        self.bbox_map_rows.extend(camera_result.bbox_map_rows)
        self.offsets.extend(camera_result.offsets)
        self.original_frames.extend(camera_result.orig_frames)
        self.camera_indices.extend(camera_result.cam_indices)
        self.output_frame_index = camera_result.output_frame_idx

def _write_camera_maps(request, result):
    if not request.frame_map_path:
        return
    directory = os.path.dirname(request.frame_map_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(request.frame_map_path, 'w', newline='') as file:
        writer = csv.DictWriter(
            file, fieldnames=['output_frame', 'cam', 'cam_frame', 'source_frame'],
        )
        writer.writeheader()
        writer.writerows(result.frame_map_rows)
    bbox_path = request.frame_map_path.replace('_frame_map.csv', '_bbox_map.csv')
    with open(bbox_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'output_frame', 'cam', 'cam_frame', 'source_frame', 'track_id',
            'x1', 'y1', 'x2', 'y2', 'is_interpolated', 'interp_gap_len',
        ])
        writer.writeheader()
        writer.writerows(result.bbox_map_rows)
    print(f"  Frame map: {request.frame_map_path}")
    print(f"  BBox map:  {bbox_path}")


def _write_camera_offsets(request, result):
    if not request.cameras or not request.cameras[0].get('video_path'):
        return
    base_name = os.path.splitext(os.path.basename(request.cameras[0]['video_path']))[0]
    path = os.path.join(_tracking.OUTPUT_DIR, f"{base_name}_offsets.npz")
    np.savez_compressed(
        path,
        offsets=np.array(result.offsets, dtype=np.int32),
        orig_frames=np.array(result.original_frames, dtype=np.int32),
        cam_indices=np.array(result.camera_indices, dtype=np.int32),
    )
    print(f"  Offsets: {path}")


def _process_cameras(request: CameraBatchRequest):
    """
    逐台相機以 two-pass 快取追焦，寫入映射資料並將影格交給輸出端。

    回傳實際輸出的全域影格數。
    """
    if request.frame_cache is None:
        raise ValueError("two-pass 第二遍必須提供 frame_cache")
    if request.preset_target_ids is None or any(
        index not in request.preset_target_ids for index in range(len(request.cameras))
    ):
        raise ValueError("two-pass 第二遍必須提供每台相機的已選主跑者 ID")
    result = _CameraBatchResult()
    for camera_index, (camera_config, capture) in enumerate(
        zip(request.cameras, request.captures),
    ):
        camera_result = _process_single_camera(
            _CameraProcessingRequest(
                camera_index=camera_index, camera=camera_config, capture=capture,
                output_writer=request.output_writer,
                frame_map_path=request.frame_map_path,
                preset_target_ids=request.preset_target_ids,
                frame_cache=request.frame_cache,
                frame_ranges_by_camera=request.frame_ranges_by_camera,
                camera_count=len(request.cameras),
                output_frame_index=result.output_frame_index,
                frame_sink=request.frame_sink,
            )
        )
        result.include(camera_result)
    _write_camera_maps(request, result)
    _write_camera_offsets(request, result)
    return result.written
