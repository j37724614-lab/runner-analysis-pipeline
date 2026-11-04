"""Two-pass 追蹤前的可選 TensorRT INT8 時間軸預掃描。"""

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from itertools import pairwise

import cv2
from ultralytics import YOLO

from core import tracking as _tracking


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


@dataclass(frozen=True)
class _PersonObservation:
    center_x: float
    center_y: float
    height: float


@dataclass(frozen=True)
class _MotionSample:
    frame_index: int
    person: _PersonObservation


@dataclass
class _MotionTrack:
    samples: list


@dataclass(frozen=True)
class _MotionMetrics:
    displacement_heights: float
    speed_heights_per_sec: float
    straightness: float


@dataclass(frozen=True)
class _MotionUpdate:
    hit_frames: tuple
    running_tracks: int
    max_displacement_heights: float
    max_speed_heights_per_sec: float
    max_straightness: float


def _person_observations(result, min_height):
    """Extract person boxes without applying track/lane ROI constraints."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else None
    observations = []
    for index, box in enumerate(xyxy):
        if classes is not None and int(classes[index]) != 0:
            continue
        bx1, by1, bx2, by2 = map(float, box[:4])
        height = by2 - by1
        if height < min_height:
            continue
        observations.append(_PersonObservation(
            center_x=(bx1 + bx2) / 2.0,
            center_y=(by1 + by2) / 2.0,
            height=height,
        ))
    return observations


def _sample_distance(first, second):
    return math.hypot(
        second.person.center_x - first.person.center_x,
        second.person.center_y - first.person.center_y,
    )


def _motion_metrics(samples, fps):
    heights = sorted(sample.person.height for sample in samples)
    median_height = heights[len(heights) // 2]
    if median_height <= 0 or fps <= 0 or len(samples) < 2:
        return _MotionMetrics(0.0, 0.0, 0.0)
    net_distance = _sample_distance(samples[0], samples[-1])
    path_length = sum(
        _sample_distance(previous, current)
        for previous, current in pairwise(samples)
    )
    elapsed_seconds = (
        samples[-1].frame_index - samples[0].frame_index
    ) / fps
    displacement = net_distance / median_height
    speed = displacement / elapsed_seconds if elapsed_seconds > 0 else 0.0
    straightness = net_distance / path_length if path_length > 0 else 0.0
    return _MotionMetrics(displacement, speed, straightness)


class _FastPersonMotionDetector:
    """Find fast, large person motion from sparse detector observations."""

    def __init__(self, fps, stride):
        self.fps = fps
        self.stride = max(1, int(stride))
        self.tracks = []

    def update(self, frame_index, people):
        self._drop_stale_tracks(frame_index)
        matched_track_indices = self._match_people(frame_index, people)
        matched_person_indices = set(matched_track_indices.values())
        for person_index, person in enumerate(people):
            if person_index not in matched_person_indices:
                self.tracks.append(_MotionTrack([
                    _MotionSample(frame_index, person),
                ]))
        return self._running_update(frame_index)

    def _drop_stale_tracks(self, frame_index):
        max_gap = self.stride * 2
        self.tracks = [
            track for track in self.tracks
            if frame_index - track.samples[-1].frame_index <= max_gap
        ]

    def _match_people(self, frame_index, people):
        candidates = []
        for track_index, track in enumerate(self.tracks):
            for person_index, person in enumerate(people):
                cost = self._match_cost(track, frame_index, person)
                if cost is not None:
                    candidates.append((cost, track_index, person_index))
        matched_tracks = {}
        used_people = set()
        for _cost, track_index, person_index in sorted(candidates):
            if track_index in matched_tracks or person_index in used_people:
                continue
            self.tracks[track_index].samples.append(
                _MotionSample(frame_index, people[person_index]),
            )
            history_limit = max(3, int(_tracking.PRESCAN_MOTION_MIN_SAMPLES))
            self.tracks[track_index].samples = (
                self.tracks[track_index].samples[-history_limit:]
            )
            matched_tracks[track_index] = person_index
            used_people.add(person_index)
        return matched_tracks

    def _match_cost(self, track, frame_index, person):
        latest = track.samples[-1]
        frame_gap = frame_index - latest.frame_index
        if frame_gap <= 0:
            return None
        predicted_x = latest.person.center_x
        predicted_y = latest.person.center_y
        if len(track.samples) >= 2:
            previous = track.samples[-2]
            previous_gap = latest.frame_index - previous.frame_index
            if previous_gap > 0:
                scale = frame_gap / previous_gap
                predicted_x += (
                    latest.person.center_x - previous.person.center_x
                ) * scale
                predicted_y += (
                    latest.person.center_y - previous.person.center_y
                ) * scale
        average_height = (latest.person.height + person.height) / 2.0
        if average_height <= 0:
            return None
        height_ratio = person.height / latest.person.height
        if not 0.6 <= height_ratio <= 1.7:
            return None
        normalized_distance = math.hypot(
            person.center_x - predicted_x,
            person.center_y - predicted_y,
        ) / average_height
        allowed_distance = (
            _tracking.PRESCAN_MOTION_MAX_MATCH_DISTANCE_HEIGHTS
            * max(1.0, frame_gap / self.stride)
        )
        if normalized_distance > allowed_distance:
            return None
        return normalized_distance + 0.5 * abs(math.log(height_ratio))

    def _running_update(self, frame_index):
        minimum_samples = max(2, int(_tracking.PRESCAN_MOTION_MIN_SAMPLES))
        running = []
        metrics = []
        for track in self.tracks:
            if track.samples[-1].frame_index != frame_index:
                continue
            if len(track.samples) < minimum_samples:
                continue
            window = track.samples[-minimum_samples:]
            current = _motion_metrics(window, self.fps)
            metrics.append(current)
            if (
                current.displacement_heights
                >= _tracking.PRESCAN_MOTION_MIN_DISPLACEMENT_HEIGHTS
                and current.speed_heights_per_sec
                >= _tracking.PRESCAN_MOTION_MIN_SPEED_HEIGHTS_PER_SEC
                and current.straightness
                >= _tracking.PRESCAN_MOTION_MIN_STRAIGHTNESS
            ):
                running.append(track)
        hit_frames = sorted({
            sample.frame_index
            for track in running
            for sample in track.samples[-minimum_samples:]
        })
        return _MotionUpdate(
            hit_frames=tuple(hit_frames),
            running_tracks=len(running),
            max_displacement_heights=max(
                (item.displacement_heights for item in metrics), default=0.0,
            ),
            max_speed_heights_per_sec=max(
                (item.speed_heights_per_sec for item in metrics), default=0.0,
            ),
            max_straightness=max(
                (item.straightness for item in metrics), default=0.0,
            ),
        )


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
        samples = self._sample_frames(capture, fps, total_frames)
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

    def _sample_frames(self, capture, fps, total_frames):
        rows, hit_frames = [], set()
        motion_detector = _FastPersonMotionDetector(
            fps=fps, stride=_tracking.PRESCAN_STRIDE,
        )
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
                classes=[0],
            )[0]
            people = _person_observations(
                result, _tracking.MIN_PERSON_HEIGHT,
            )
            motion = motion_detector.update(frame_index, people)
            hit_frames.update(motion.hit_frames)
            rows.append({
                'frame': frame_index,
                'time_s': frame_index / fps if fps > 0 else None,
                'hit': 0,
                'valid_boxes': int(motion.running_tracks),
                'person_boxes': len(people),
                'running_tracks': int(motion.running_tracks),
                'max_displacement_heights': round(
                    motion.max_displacement_heights, 4,
                ),
                'max_speed_heights_per_sec': round(
                    motion.max_speed_heights_per_sec, 4,
                ),
                'max_straightness': round(motion.max_straightness, 4),
            })
        for row in rows:
            if row['frame'] in hit_frames:
                row['hit'] = 1
                row['valid_boxes'] = max(1, row['valid_boxes'])
        return _PrescanSamples(
            rows, sorted(hit_frames),
            sum(row['valid_boxes'] for row in rows),
            time.perf_counter() - started_at,
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
            writer = csv.DictWriter(file, fieldnames=[
                'frame', 'time_s', 'hit', 'valid_boxes', 'person_boxes',
                'running_tracks', 'max_displacement_heights',
                'max_speed_heights_per_sec', 'max_straightness',
            ])
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
            'motion_min_samples': _tracking.PRESCAN_MOTION_MIN_SAMPLES,
            'motion_min_displacement_heights': (
                _tracking.PRESCAN_MOTION_MIN_DISPLACEMENT_HEIGHTS
            ),
            'motion_min_speed_heights_per_sec': (
                _tracking.PRESCAN_MOTION_MIN_SPEED_HEIGHTS_PER_SEC
            ),
            'motion_min_straightness': (
                _tracking.PRESCAN_MOTION_MIN_STRAIGHTNESS
            ),
            'motion_max_match_distance_heights': (
                _tracking.PRESCAN_MOTION_MAX_MATCH_DISTANCE_HEIGHTS
            ),
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


