"""Two-pass 第一遍：以完整原始畫面收集所有候選跑者軌跡。"""

from dataclasses import dataclass

import cv2

from core import tracking as _tracking
from core.tracking_geometry import _bbox_bottom_center

from .camera_setup import _project_and_check_track_roi
from .prescan import _frame_ranges_for_camera


def _reset_yolo_trackers(model):
    """重置 model 底下所有 ByteTrack tracker 狀態，避免上一台相機的 track id 污染下一台。"""
    predictor = getattr(model, 'predictor', None)
    trackers = getattr(predictor, 'trackers', None) if predictor is not None else None
    if trackers:
        for tracker in trackers:
            reset = getattr(tracker, 'reset', None)
            if callable(reset):
                reset()


def _advance_to_next_valid_range(cap, valid_ranges, range_cursor, frame_count):
    """依 valid_ranges 跳過不在範圍內的區段（video seek）。

    回傳 (range_cursor, frame_count, exhausted)；exhausted=True 表示已無更多有效範圍，
    呼叫端應停止讀取。valid_ranges 為 None/空 時直接回傳原值、exhausted=False。
    """
    if not valid_ranges:
        return range_cursor, frame_count, False
    while range_cursor < len(valid_ranges) and frame_count > valid_ranges[range_cursor][1]:
        range_cursor += 1
        if range_cursor < len(valid_ranges):
            frame_count = valid_ranges[range_cursor][0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    return range_cursor, frame_count, range_cursor >= len(valid_ranges)


@dataclass(frozen=True)
class _PassOneRoi:
    track_roi: dict | None


@dataclass(frozen=True)
class _PassOneFrame:
    boxes: object
    track_ids: object
    camera_index: int
    source_frame: int
    roi: _PassOneRoi


def _pass1_detection_roi_filter(ground_pt, roi):
    """Pass 1 跑道投影篩選；使用原始畫面座標，回傳 (passes, proj_px)。
    """
    return _project_and_check_track_roi(ground_pt, roi.track_roi)


def _collect_frame_detections(request):
    """依高度與 ROI 過濾單幀 YOLO 偵測。回傳 (rows, cache_entries)：
      rows          list[dict]：供 all_rows 累加，供評分使用
      cache_entries list[dict]：供 frame_cache[(cam_idx, source_frame)] 累加，供 Pass 2 跳過 YOLO
    """
    rows = []
    cache_entries = []
    if request.track_ids is None:
        return rows, cache_entries

    for i in range(len(request.boxes)):
        bx1, by1, bx2, by2 = map(int, request.boxes[i])
        bbox_h = by2 - by1
        if bbox_h < _tracking.MIN_PERSON_HEIGHT:
            continue
        center_x = (bx1 + bx2) / 2.0
        center_y = (by1 + by2) / 2.0
        ground_pt = _bbox_bottom_center((bx1, by1, bx2, by2))

        passes, proj_px = _pass1_detection_roi_filter(
            ground_pt, request.roi,
        )
        if not passes:
            continue

        tid = int(request.track_ids[i])
        rows.append({
            'cam_idx':    request.camera_index,
            'frame_idx':  request.source_frame,
            'track_id':   tid,
            'bx1': bx1, 'by1': by1, 'bx2': bx2, 'by2': by2,
            'center_x':   center_x,
            'center_y':   center_y,
            'ground_x':   ground_pt[0],
            'ground_y':   ground_pt[1],
            'proj_px':    proj_px,
            'bbox_height': bbox_h,
        })
        cache_entries.append({
            'track_id': tid,
            'bx1': bx1, 'by1': by1, 'bx2': bx2, 'by2': by2,
        })

    return rows, cache_entries


class _PassOneDetectionCollector:
    def __init__(self, model, frame_ranges_by_camera):
        self.model = model
        self.frame_ranges_by_camera = frame_ranges_by_camera
        self.rows = []
        self.frame_cache = {}

    def collect(self, captures, cameras):
        for camera_index, (camera, capture) in enumerate(zip(cameras, captures)):
            _reset_yolo_trackers(self.model)
            self._collect_camera(camera_index, camera, capture)
        return self.rows, self.frame_cache

    def _collect_camera(self, camera_index, camera, capture):
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ranges = _frame_ranges_for_camera(
            self.frame_ranges_by_camera, camera_index, total_frames,
        )
        range_cursor, frame_count = 0, ranges[0][0] if ranges else 0
        if ranges:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        roi = _PassOneRoi(camera.get('track_roi'))
        while capture.isOpened():
            range_cursor, frame_count, exhausted = _advance_to_next_valid_range(
                capture, ranges, range_cursor, frame_count,
            )
            if exhausted:
                break
            success, frame = capture.read()
            if not success:
                break
            self._collect_frame(camera_index, frame_count, frame, roi)
            frame_count += 1

    def _collect_frame(self, camera_index, source_frame, frame, roi):
        result = self.model.track(
            frame, persist=True, classes=[0], show=False, device=_tracking.DEVICE,
            conf=0.25, iou=0.1, imgsz=1280, verbose=False, half=True,
            tracker=_tracking.TWO_PASS_TRACKER_CONFIG,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return
        rows, cache_entries = _collect_frame_detections(_PassOneFrame(
            boxes=result.boxes.xyxy.cpu().numpy(),
            track_ids=(
                result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
            ),
            camera_index=camera_index,
            source_frame=source_frame,
            roi=roi,
        ))
        self.rows.extend(rows)
        if cache_entries:
            self.frame_cache.setdefault((camera_index, source_frame), []).extend(
                cache_entries,
            )


def _collect_all_detections(caps, cameras, model, frame_ranges_by_cam=None):
    """
    Pass 1: 讀遍所有相機，以完整原始畫面收集每幀通過 ROI/高度過濾的所有偵測（不選最快）。
    回傳 (all_rows, frame_cache)：
      all_rows    list[dict]：每筆偵測（供評分使用）
      frame_cache dict[(cam_idx, frame_idx), list[dict]]：每幀的 bbox 快取（供 Pass 2 跳過 YOLO）
    ByteTrack 在相機切換時重置，與 _process_cameras 一致。
    """
    return _PassOneDetectionCollector(model, frame_ranges_by_cam).collect(
        caps, cameras,
    )


