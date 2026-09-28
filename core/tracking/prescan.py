"""Two-pass 追蹤前的可選 TensorRT INT8 時間軸預掃描。"""

import csv
import json
import os
import time
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

from core import tracking as _tracking
from core.tracking_geometry import _bbox_bottom_center

from .camera_setup import _project_and_check_track_roi


def _frame_ranges_for_camera(frame_ranges_by_cam, cam_idx, total_frames):
    if not frame_ranges_by_cam:
        return None
    ranges = frame_ranges_by_cam.get(cam_idx)
    if not ranges:
        return None
    cleaned = []
    last_start = None
    last_end = None
    for item in ranges:
        if isinstance(item, dict):
            start = int(item.get('start_frame', 0))
            end = int(item.get('end_frame', total_frames - 1))
        else:
            start, end = item
            start = int(start)
            end = int(end)
        start = max(0, min(start, max(total_frames - 1, 0)))
        end = max(0, min(end, max(total_frames - 1, 0)))
        if end < start:
            continue
        if last_start is None:
            last_start, last_end = start, end
        elif start <= last_end + 1:
            last_end = max(last_end, end)
        else:
            cleaned.append((last_start, last_end))
            last_start, last_end = start, end
    if last_start is not None:
        cleaned.append((last_start, last_end))
    return cleaned or None


def _merge_prescan_hit_frames(hit_frames, total_frames, stride, buffer_frames, max_gap_frames):
    if not hit_frames:
        return []
    ranges = []
    start = end = int(hit_frames[0])
    for frame_idx in hit_frames[1:]:
        frame_idx = int(frame_idx)
        if frame_idx - end <= max_gap_frames:
            end = frame_idx
        else:
            ranges.append((start, end))
            start = end = frame_idx
    ranges.append((start, end))

    expanded = []
    last_start = last_end = None
    for start, end in ranges:
        start = max(0, start - buffer_frames)
        end = min(max(total_frames - 1, 0), end + buffer_frames + stride - 1)
        if last_start is None:
            last_start, last_end = start, end
        elif start <= last_end + 1:
            last_end = max(last_end, end)
        else:
            expanded.append((last_start, last_end))
            last_start, last_end = start, end
    expanded.append((last_start, last_end))
    return expanded


def _prescan_detection_is_valid(result, cam, min_height):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return False, 0
    xyxy = boxes.xyxy.detach().cpu().numpy()
    track_roi = cam.get('track_roi')
    valid_count = 0
    for box in xyxy:
        bx1, by1, bx2, by2 = map(float, box[:4])
        if by2 - by1 < min_height:
            continue
        ground_pt = _bbox_bottom_center((bx1, by1, bx2, by2))
        passes, _proj_px = _project_and_check_track_roi(ground_pt, track_roi)
        if not passes:
            continue
        valid_count += 1
    return valid_count > 0, valid_count


@dataclass(frozen=True)
class _PrescanSamples:
    rows: list
    hit_frames: list
    valid_box_count: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _PrescanCameraReportInput:
    camera_index: int
    video_path: str
    fps: float
    total_frames: int
    samples: _PrescanSamples
    ranges: list


class _TemporalPrescanner:
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir

    def scan(self, cameras):
        ranges_by_camera = {}
        reports = []
        for camera_index, camera_config in enumerate(cameras):
            result = self._scan_camera(camera_index, camera_config)
            if result is None:
                continue
            ranges, report = result
            ranges_by_camera[camera_index] = ranges
            reports.append(report)
        self._write_report(cameras, reports)
        return ranges_by_camera

    def _scan_camera(self, camera_index, camera_config):
        video_path = camera_config.get('video_path')
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            print(f"  [prescan] 相機 {camera_index + 1}: 無法開啟 {video_path}")
            return None
        fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        samples = self._sample_frames(capture, camera_config, fps, total_frames)
        capture.release()
        ranges = _merge_prescan_hit_frames(
            samples.hit_frames, total_frames, _tracking.PRESCAN_STRIDE,
            max(0, round(_tracking.PRESCAN_BUFFER_SEC * fps)),
            max(0, round(_tracking.PRESCAN_MAX_GAP_SEC * fps)),
        )
        report = self._build_camera_report(_PrescanCameraReportInput(
            camera_index, video_path, fps, total_frames, samples, ranges,
        ))
        self._write_samples(video_path, samples.rows)
        return ranges, report

    def _sample_frames(self, capture, camera_config, fps, total_frames):
        rows, hits, valid_box_count = [], [], 0
        started_at = time.perf_counter()
        for frame_index in range(total_frames):
            if frame_index % _tracking.PRESCAN_STRIDE:
                ok = capture.grab() if _tracking.PRESCAN_USE_GRAB else capture.read()[0]
                if not ok:
                    break
                continue
            ok, frame = capture.read()
            if not ok:
                break
            result = self.model.predict(
                frame, imgsz=_tracking.PRESCAN_IMGSZ, conf=_tracking.PRESCAN_CONF,
                iou=_tracking.PRESCAN_IOU, device=_tracking.DEVICE, verbose=False,
            )[0]
            is_hit, valid_count = _prescan_detection_is_valid(
                result, camera_config, _tracking.MIN_PERSON_HEIGHT,
            )
            if is_hit:
                hits.append(frame_index)
                valid_box_count += valid_count
            rows.append({
                'frame': frame_index,
                'time_s': frame_index / fps if fps > 0 else None,
                'hit': int(is_hit),
                'valid_boxes': int(valid_count),
            })
        return _PrescanSamples(
            rows, hits, valid_box_count, time.perf_counter() - started_at,
        )

    def _build_camera_report(self, request):
        range_rows = [
            _prescan_range_row(start, end, request.fps)
            for start, end in request.ranges
        ]
        kept_frames = sum(row['num_frames'] for row in range_rows)
        print(
            f"  [prescan] 相機 {request.camera_index + 1}: "
            f"sampled={len(request.samples.rows)}, "
            f"hits={len(request.samples.hit_frames)}, ranges={len(request.ranges)}, "
            f"kept={kept_frames}/{request.total_frames}, "
            f"elapsed={request.samples.elapsed_seconds:.2f}s"
        )
        return {
            'cam_idx': request.camera_index, 'video_path': request.video_path,
            'fps': request.fps, 'total_frames': request.total_frames,
            'sampled_frames': len(request.samples.rows),
            'hit_sampled_frames': len(request.samples.hit_frames),
            'valid_boxes': request.samples.valid_box_count,
            'elapsed_sec': request.samples.elapsed_seconds, 'ranges': range_rows,
            'kept_frames': kept_frames,
            'kept_ratio': (
                kept_frames / request.total_frames if request.total_frames else 0.0
            ),
        }

    def _write_samples(self, video_path, rows):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        path = os.path.join(self.output_dir, f"{base_name}_prescan_samples.csv")
        with open(path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['frame', 'time_s', 'hit', 'valid_boxes'])
            writer.writeheader()
            writer.writerows(rows)

    def _write_report(self, cameras, reports):
        base_name = _first_camera_base_name(cameras)
        path = os.path.join(self.output_dir, f"{base_name}_prescan_ranges.json")
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(_prescan_report(reports), file, ensure_ascii=False, indent=2)
        print(f"  [prescan] report: {path}")


def _prescan_range_row(start, end, fps):
    return {
        'start_frame': int(start), 'end_frame': int(end),
        'num_frames': int(end - start + 1),
        'start_time_s': float(start / fps) if fps > 0 else None,
        'end_time_s': float(end / fps) if fps > 0 else None,
    }


def _first_camera_base_name(cameras):
    if cameras and cameras[0].get('video_path'):
        return os.path.splitext(os.path.basename(cameras[0]['video_path']))[0]
    return 'tracking'


def _prescan_report(reports):
    return {
        'enabled': True, 'engine_path': _tracking.PRESCAN_ENGINE_PATH,
        'params': {
            'stride': _tracking.PRESCAN_STRIDE, 'imgsz': _tracking.PRESCAN_IMGSZ,
            'conf': _tracking.PRESCAN_CONF, 'iou': _tracking.PRESCAN_IOU,
            'buffer_sec': _tracking.PRESCAN_BUFFER_SEC,
            'max_gap_sec': _tracking.PRESCAN_MAX_GAP_SEC, 'use_grab': _tracking.PRESCAN_USE_GRAB,
        },
        'cameras': reports,
    }


def run_temporal_prescan(cameras, output_dir=None):
    if not _tracking.PRESCAN_ENABLED:
        return None
    if not _tracking.PRESCAN_ENGINE_PATH or not os.path.exists(_tracking.PRESCAN_ENGINE_PATH):
        print(f"  [prescan] engine not found, skip: {_tracking.PRESCAN_ENGINE_PATH}")
        return None
    destination = output_dir or _tracking.OUTPUT_DIR
    os.makedirs(destination, exist_ok=True)
    print("temporal pre-scan：使用 INT8 engine 找有效 frame range...")
    ranges = _TemporalPrescanner(
        YOLO(_tracking.PRESCAN_ENGINE_PATH, task='detect'), destination,
    ).scan(cameras)
    if not any(ranges.values()):
        print("  [prescan] 未找到有效區間，two_pass 將退回掃完整影片")
        return None
    return ranges


