import cv2

try:
    cv2.setLogLevel(3)  # type: ignore[attr-defined]  # 抑制 swscaler HDR 色彩轉換警告
except AttributeError:
    pass  # 舊版 OpenCV 無此 API，忽略
import csv
import os
from dataclasses import dataclass, field
from itertools import pairwise

import matplotlib
import numpy as np

matplotlib.use('Agg')
from filterpy.kalman import KalmanFilter  # type: ignore[import-untyped]
from scipy.signal import butter, filtfilt

from core.tracking_geometry import (
    LANE_WIDTH_M,
    _compute_homography,
    _project_onto_track,
    _project_point_to_track_line,
    _transform_point_homography,
)

# =======================================================================
# 預設參數與常數
# =======================================================================
DEFAULT_FLAT_INTERP_EPS_M = 0.001      # 距離變化小於此值視為 flat segment


@dataclass(frozen=True)
class CameraRequest:
    """建立一台追蹤相機所需的設定。"""

    video_path: object
    crop: object = None
    roi_x: tuple = (0, 9999)
    roi_y: tuple = (0, 9999)
    switch_x: object = None
    roi_zones: object = None
    distance_m: object = None
    start_line: object = None
    end_line: object = None
    pre_roll_px: int = 200
    end_roll_px: int = 120
    start_gate_px: int = 250
    start_roi_px: int = 100
    start_confirm_move_px: int = 8
    homography_matrix: object = None
    homography_src_points: object = None
    homography_dst_world: object = None


@dataclass(frozen=True)
class _LineGeometry:
    """由起終線推導的跑道影像幾何。"""

    start_mid: object = None
    end_mid: object = None
    track_direction: object = None
    pixel_span: object = None
    quadrilateral: object = None


def _line_geometry(request):
    if request.start_line is None or request.end_line is None:
        return _LineGeometry()
    start_mid = (
            (request.start_line[0][0] + request.start_line[1][0]) / 2.0,
            (request.start_line[0][1] + request.start_line[1][1]) / 2.0,
    )
    end_mid = (
            (request.end_line[0][0] + request.end_line[1][0]) / 2.0,
            (request.end_line[0][1] + request.end_line[1][1]) / 2.0,
    )
    delta_x = end_mid[0] - start_mid[0]
    delta_y = end_mid[1] - start_mid[1]
    pixel_span = (delta_x ** 2 + delta_y ** 2) ** 0.5
    track_direction = None
    if pixel_span > 0:
        track_direction = (delta_x / pixel_span, delta_y / pixel_span)
    return _LineGeometry(
        start_mid=start_mid,
        end_mid=end_mid,
        track_direction=track_direction,
        pixel_span=pixel_span,
        quadrilateral=np.array(
            [
                request.start_line[0],
                request.end_line[0],
                request.end_line[1],
                request.start_line[1],
            ],
            dtype=np.float32,
        ),
    )


def _homography_start_x(request):
    if request.homography_matrix is None:
        return None
    if request.start_line is not None:
        start_world = [
            _transform_point_homography(point, request.homography_matrix)
            for point in request.start_line
        ]
        if all(point is not None for point in start_world):
            return float(np.mean([point[0] for point in start_world]))
        return None
    if request.roi_x[0] != 0 or request.roi_y[0] != 0:
        start_world = _transform_point_homography(
            (request.roi_x[0], request.roi_y[0]),
            request.homography_matrix,
        )
        return None if start_world is None else float(start_world[0])
    return 0.0


def _meters_per_pixel(request, geometry):
    if request.homography_matrix is not None:
        return None
    if geometry.pixel_span and request.distance_m is not None:
        return request.distance_m / geometry.pixel_span
    if request.distance_m is not None and request.roi_x[1] > request.roi_x[0]:
        return request.distance_m / (
            request.roi_x[1] - request.roi_x[0]
        )
    return None


def _track_roi(request, geometry):
    if geometry.start_mid is None:
        return None
    return {
        'start_mid': geometry.start_mid,
        'track_dir': geometry.track_direction,
        'pixel_span': geometry.pixel_span,
        'pre_roll_px': request.pre_roll_px,
        'end_roll_px': request.end_roll_px,
        'start_roi_px': request.start_roi_px,
        'start_gate_px': request.start_gate_px,
        'start_confirm_move_px': request.start_confirm_move_px,
    }


def camera(request):
    """Derive the complete runtime camera mapping from one request."""
    geometry = _line_geometry(request)
    zones = (
        request.roi_zones
        if request.roi_zones is not None
        else [{'x': request.roi_x, 'y': request.roi_y}]
    )
    no_roi = (
        request.roi_zones is None
        and request.roi_x == (0, 9999)
        and request.roi_y == (0, 9999)
    )

    return {
        'video_path':  request.video_path,
        'crop_params': request.crop,
        'roi_enabled': not no_roi,
        'roi_zones':   zones,
        'switch_x': None if geometry.start_mid else request.switch_x,
        'start_x': (
            geometry.start_mid[0]
            if geometry.start_mid
            else request.roi_x[0]
        ),
        'm_per_pixel': _meters_per_pixel(request, geometry),
        'distance_m':  request.distance_m,
        'start_line':  request.start_line,
        'end_line':    request.end_line,
        'start_mid': geometry.start_mid,
        'end_mid': geometry.end_mid,
        'track_dir': geometry.track_direction,
        'pixel_span': geometry.pixel_span,
        'quad_roi': geometry.quadrilateral,
        'track_roi': _track_roi(request, geometry),
        'H_matrix':    request.homography_matrix,
        'homography_start_x': _homography_start_x(request),
        'homography_src_points': (
            np.asarray(request.homography_src_points, dtype=np.float32)
            if request.homography_src_points is not None else None
        ),
        'homography_dst_world': (
            np.asarray(request.homography_dst_world, dtype=np.float32)
            if request.homography_dst_world is not None else None
        ),
    }


def _ordered_line_corners(line):
    point_0, point_1 = ([float(value) for value in point] for point in line)
    return (
        (point_0, point_1)
        if point_0[1] <= point_1[1]
        else (point_1, point_0)
    )


def _automatic_line_homography(entry):
    start_far, start_near = _ordered_line_corners(entry['start_line'])
    end_far, end_near = _ordered_line_corners(entry['end_line'])
    distance = float(entry['distance_m'])
    source_points = [start_far, start_near, end_far, end_near]
    world_points = [
        [0.0, 0.0],
        [0.0, LANE_WIDTH_M],
        [distance, 0.0],
        [distance, LANE_WIDTH_M],
    ]
    try:
        matrix, _ = _compute_homography(
            np.float32(source_points),
            np.float32(world_points),
        )
        if np.linalg.cond(matrix) <= 5000:
            return matrix, source_points, world_points
    except (ValueError, np.linalg.LinAlgError):
        pass
    return None, None, None


def _json_homography(entry):
    source_points = entry.get('homography_src_points')
    world_points = entry.get('homography_dst_world')
    if source_points and world_points:
        matrix, _ = _compute_homography(
            np.float32(source_points),
            np.float32(world_points),
        )
        return matrix, source_points, world_points
    if (
        entry.get('start_line')
        and entry.get('end_line')
        and entry.get('distance_m') is not None
    ):
        return _automatic_line_homography(entry)
    return None, source_points, world_points


def _build_camera_from_json(entry):
    start_line = entry.get('start_line')
    end_line = entry.get('end_line')
    matrix, source_points, world_points = _json_homography(entry)

    return camera(CameraRequest(
        video_path=entry.get('video_path'),
        crop=tuple(entry['crop']) if entry.get('crop') else None,
        roi_x=tuple(entry['roi_x']) if 'roi_x' in entry else (0, 9999),
        roi_y=tuple(entry['roi_y']) if 'roi_y' in entry else (0, 9999),
        switch_x=entry.get('switch_x'),
        roi_zones=entry.get('roi_zones'),
        distance_m=entry.get('distance_m'),
        start_line=[tuple(point) for point in start_line] if start_line else None,
        end_line=[tuple(point) for point in end_line] if end_line else None,
        pre_roll_px=int(entry.get('pre_roll_px', 200)),
        end_roll_px=int(entry.get('end_roll_px', 120)),
        start_roi_px=int(entry.get('start_roi_px', 100)),
        homography_matrix=matrix,
        homography_src_points=source_points,
        homography_dst_world=world_points,
    ))


def _interpolate_flat_segments(d, eps=DEFAULT_FLAT_INTERP_EPS_M):
    """
    將中間的距離 flat segment 用前後變動點線性插值。
    只處理前後都有有效變動點的區段；開頭/結尾 flat 不外推。
    """
    d = np.asarray(d, dtype=float).copy()
    n = len(d)
    if n < 3:
        return d

    i = 1
    while i < n:
        if abs(d[i] - d[i - 1]) >= eps:
            i += 1
            continue

        flat_start = i - 1
        flat_val = d[flat_start]
        j = i
        while j < n and abs(d[j] - flat_val) < eps:
            j += 1

        # 需要前一個變動點與後一個變動點；避免對開頭/尾端憑空外推。
        prev_idx = flat_start - 1
        next_idx = j
        if prev_idx >= 0 and next_idx < n and d[next_idx] > d[prev_idx]:
            span = next_idx - prev_idx
            for k in range(flat_start, next_idx):
                ratio = (k - prev_idx) / span
                d[k] = d[prev_idx] + ratio * (d[next_idx] - d[prev_idx])

        i = max(j, i + 1)

    return d


def _interpolate_missing_numeric(values):
    """用前後有效值線性補齊 None；端點缺值用最近有效值延伸。"""
    out = list(values)
    valid = [i for i, v in enumerate(out) if v is not None]
    if not valid:
        return out

    first = valid[0]
    for i in range(first):
        out[i] = out[first]

    for left, right in pairwise(valid):
        if right == left + 1:
            continue
        start = float(out[left])
        end = float(out[right])
        span = right - left
        for i in range(left + 1, right):
            ratio = (i - left) / span
            out[i] = start + ratio * (end - start)

    last = valid[-1]
    for i in range(last + 1, len(out)):
        out[i] = out[last]
    return out


def _interpolation_metadata(interpolated_mask):
    """回傳每幀插值段長度與速度可信度。"""
    gap_len = [0] * len(interpolated_mask)
    confidence = [1.0] * len(interpolated_mask)

    i = 0
    while i < len(interpolated_mask):
        if not interpolated_mask[i]:
            i += 1
            continue

        start = i
        while i < len(interpolated_mask) and interpolated_mask[i]:
            i += 1
        length = i - start

        if length <= 3:
            conf = 0.7
        elif length <= 8:
            conf = 0.4
        else:
            conf = 0.2

        for j in range(start, i):
            gap_len[j] = length
            confidence[j] = conf

    return gap_len, confidence


def _normalized_measurement_confidence(measurement_confidence, n):
    """Per-frame Kalman measurement confidence, clipped to [0.05, 1.0];
    falls back to all-ones when absent or the wrong length."""
    if measurement_confidence is None:
        return np.ones(n, dtype=float)
    confidence = np.asarray(measurement_confidence, dtype=float)
    if len(confidence) != n:
        return np.ones(n, dtype=float)
    return np.clip(confidence, 0.05, 1.0)


def _monotonic_distance(d_raw, flat_interp_eps_m):
    """Force the distance series non-decreasing, then interpolate any flat
    segment so a 'stuck then jump' does not oscillate speed/acceleration."""
    d = np.array(d_raw, dtype=float)
    for k in range(1, len(d)):
        d[k] = max(d[k], d[k - 1])
    return _interpolate_flat_segments(d, eps=flat_interp_eps_m)


def _butterworth_smoothed_distance(d, fps):
    """Bidirectional 3.5 Hz low-pass (removes 30+ Hz bbox jitter, keeps the
    ~0-1 Hz real acceleration of a sprint). Needs n >= 15; re-clamps monotonic
    and guards the filtfilt boundary from dipping below the start."""
    if len(d) < 15:
        return d.copy()
    try:
        b_but, a_but = butter(2, 3.5 / (fps / 2.0), btype='low')
        d_smooth = filtfilt(b_but, a_but, d)
        for k in range(1, len(d_smooth)):
            d_smooth[k] = max(d_smooth[k], d_smooth[k - 1])
        return np.maximum(d_smooth, d[0])
    except (ValueError, np.linalg.LinAlgError):
        return d.copy()


def _kalman_velocity_acceleration(d_smooth, fps, init_v, init_a, measurement_confidence):
    """Constant-acceleration Kalman filter over the smoothed distance, returning
    (velocity, acceleration) arrays. init_v/init_a seed the state so a
    camera hand-off does not ramp speed back up from zero. Needs n >= 5;
    falls back to np.gradient on error."""
    n = len(d_smooth)
    dt = 1.0 / fps
    if n < 5:
        return np.zeros(n), np.zeros(n)
    try:
        kf = KalmanFilter(dim_x=3, dim_z=1)
        kf.F = np.array([[1, dt, 0.5 * dt ** 2],
                         [0,  1,            dt],
                         [0,  0,             1]])
        kf.H = np.array([[1, 0, 0]])
        # Q[2,2]: lower value → Kalman resists rapid velocity changes from noisy
        # measurements; 0.15 is tuned for 100m sprint (real accel ≤ 5 m/s²).
        kf.Q = np.diag([0.001, 0.01, 0.15])
        base_r = 0.15
        kf.R = np.array([[base_r]])
        kf.x = np.array([[d_smooth[0]], [float(init_v)], [float(init_a)]])
        p_v = 1.0 if init_v == 0.0 else 0.1
        p_a = 100.0 if init_a == 0.0 else 1.0
        kf.P = np.diag([1.0, p_v, p_a])
        velocities, accels = [], []
        for val, conf in zip(d_smooth, measurement_confidence):
            kf.predict()
            kf.R = np.array([[base_r / float(conf)]])
            kf.update([[val]])
            velocities.append(float(kf.x[1, 0]))
            accels.append(float(kf.x[2, 0]))
        return np.maximum(velocities, 0.0), np.array(accels)
    except (ValueError, np.linalg.LinAlgError):
        velocity = np.maximum(np.gradient(d_smooth, dt), 0.0)
        return velocity, np.gradient(velocity, dt)


def _compute_kf_series(d_raw, fps, init_v=0.0, init_a=0.0,
                       measurement_confidence=None,
                       flat_interp_eps_m=DEFAULT_FLAT_INTERP_EPS_M):
    """Smooth a per-frame distance series (metres) into (d_smooth, v_smooth,
    accel) numpy arrays of the same length.

    Pipeline: monotonic constraint + flat-segment interpolation → Butterworth
    low-pass → constant-acceleration Kalman filter. ``init_v`` / ``init_a`` carry
    the previous camera's state across a hand-off. Ported from
    smart_switch_tracker.py.
    """
    n = len(d_raw)
    confidence = _normalized_measurement_confidence(measurement_confidence, n)
    d = _monotonic_distance(d_raw, flat_interp_eps_m)
    d_smooth = _butterworth_smoothed_distance(d, fps)
    v_smooth, accel = _kalman_velocity_acceleration(
        d_smooth, fps, init_v, init_a, confidence
    )
    return d_smooth, v_smooth, accel


def _build_frame_offset_map(offsets_npz):
    """(source_frame, cam_0idx) -> (off_x, off_y) lookup from offsets.npz, so
    person-centred crop coordinates can be lifted back to original-image space."""
    offset_map = {}
    if offsets_npz and os.path.exists(offsets_npz):
        d = np.load(offsets_npz)
        offs, orig_frames, cam_indices = d['offsets'], d['orig_frames'], d['cam_indices']
        for i in range(len(orig_frames)):
            offset_map[(int(orig_frames[i]), int(cam_indices[i]))] = (
                int(offs[i, 0]), int(offs[i, 1])
            )
    return offset_map


def _read_bbox_rows_by_camera(bbox_map_csv):
    """bbox_map.csv rows grouped by 0-indexed camera, each list sorted by cam_frame."""
    rows_by_cam = {}
    with open(bbox_map_csv, 'r', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows_by_cam.setdefault(int(row['cam']) - 1, []).append(row)
    for cam_rows in rows_by_cam.values():
        cam_rows.sort(key=lambda r: int(r['cam_frame']))
    return rows_by_cam


def _resolve_camera_fps(cam, fps_override):
    """fps_override if given, else the camera video's FPS, else 60.0."""
    fps = fps_override
    if fps is None and cam.get('video_path'):
        cap = cv2.VideoCapture(cam['video_path'])
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
            cap.release()
    return fps or 60.0


def _pixel_distance_for_point(pixel_cam, cx_orig, cy_orig, dist_offset_m):
    """Legacy pixel calibration: bbox centre projected onto the start/end-line
    direction and scaled by that line's known metres-per-pixel. None when the
    camera has no linear calibration."""
    if pixel_cam.get('m_per_pixel') is None:
        return None
    if pixel_cam.get('track_dir') and pixel_cam.get('start_mid'):
        proj_px = _project_onto_track(
            (cx_orig, cy_orig), pixel_cam['start_mid'], pixel_cam['track_dir'])
        return dist_offset_m + max(0.0, proj_px * pixel_cam['m_per_pixel'])
    return dist_offset_m + max(
        0.0, (cx_orig - pixel_cam['start_x']) * pixel_cam['m_per_pixel'])


def _homography_distance_for_point(cam, cx_orig, y2_orig, dist_offset_m):
    """Six-point Homography calibration: bbox bottom centre, constrained to the
    image-space runway centreline, then mapped to metres. Returns
    (distance_m_or_None, world_point_or_None, image_point_or_None)."""
    if cam.get('H_matrix') is None:
        return None, None, None
    image_point = (cx_orig, y2_orig)
    if cam.get('start_mid') is not None and cam.get('track_dir') is not None:
        image_point = _project_point_to_track_line(
            image_point, cam['start_mid'], cam['track_dir'])
    else:
        sl, el = cam.get('start_line'), cam.get('end_line')
        if sl and el:
            track_y = (sl[0][1] + sl[1][1] + el[0][1] + el[1][1]) / 4.0
            image_point = (cx_orig, track_y)
    world = _transform_point_homography(image_point, cam['H_matrix'])
    if world is None:
        return None, None, image_point
    start_world_x = cam.get('homography_start_x') or 0.0
    local_dist = float(world[0]) - start_world_x
    return dist_offset_m + max(0.0, local_dist), world, image_point


@dataclass(frozen=True)
class SpeedComputationRequest:
    """從 bbox map 計算速度指標所需的完整輸入。"""

    bbox_map_csv: str
    cameras: list
    fps_override: object = None
    offsets_npz: object = None
    pixel_cameras: object = None
    speed_mode: str = 'pixel'


@dataclass
class _SpeedContinuity:
    """保存跨相機延續的距離、速度、加速度與絕對幀位置。"""

    pixel_distance: float = 0.0
    pixel_velocity: float = 0.0
    pixel_acceleration: float = 0.0
    homography_distance: float = 0.0
    homography_velocity: float = 0.0
    homography_acceleration: float = 0.0
    absolute_frame: int = 0


@dataclass
class _CameraDistanceSamples:
    """一台相機逐幀量測得到的原始距離與除錯座標。"""

    pixel: list = field(default_factory=list)
    homography: list = field(default_factory=list)
    world_points: list = field(default_factory=list)
    image_points: list = field(default_factory=list)
    interpolated_mask: list = field(default_factory=list)
    source_frames: list = field(default_factory=list)


@dataclass(frozen=True)
class _SmoothingRequest:
    """單一距離序列的 Kalman 平滑輸入。"""

    raw_values: list
    fps: float
    initial_velocity: float
    initial_acceleration: float
    confidence: list


@dataclass(frozen=True)
class _MotionSeries:
    """保留原始距離以及平滑後的距離、速度與加速度。"""

    raw: list
    distance: object
    velocity: object
    acceleration: object


@dataclass(frozen=True)
class _CameraSpeedSeries:
    """一台相機完成取樣與平滑後，組裝輸出列所需的資料。"""

    samples: _CameraDistanceSamples
    gap_lengths: list
    confidence: list
    pixel: object
    homography: object
    active_mode: str
    active: _MotionSeries


def _smooth_distance_samples(request):
    """平滑可用的距離量測；整段無量測時回傳 None。"""
    if not any(value is not None for value in request.raw_values):
        return None

    raw_for_csv = list(request.raw_values)
    values = request.raw_values
    if any(value is None for value in values):
        values = _interpolate_missing_numeric(values)
    distance, velocity, acceleration = _compute_kf_series(
        values,
        request.fps,
        init_v=request.initial_velocity,
        init_a=request.initial_acceleration,
        measurement_confidence=request.confidence,
    )
    return _MotionSeries(raw_for_csv, distance, velocity, acceleration)


class _BBoxSpeedCalculator:
    """協調 bbox 距離取樣、雙模式平滑與跨相機結果串接。"""

    def __init__(self, request):
        self.request = request
        self.offset_map = _build_frame_offset_map(request.offsets_npz)
        self.rows_by_camera = _read_bbox_rows_by_camera(request.bbox_map_csv)
        self.cameras = [_build_camera_from_json(item) for item in request.cameras]
        pixel_configs = request.pixel_cameras or request.cameras
        self.pixel_cameras = [_build_camera_from_json(item) for item in pixel_configs]
        self.prefer_homography = str(request.speed_mode).lower() == 'homography'
        self.continuity = _SpeedContinuity()

    def calculate(self):
        """依相機順序產生全程逐幀速度指標。"""
        all_rows = []
        for camera_index, camera in enumerate(self.cameras):
            camera_rows = self.rows_by_camera.get(camera_index, [])
            if camera_rows:
                all_rows.extend(
                    self._calculate_camera(camera_index, camera, camera_rows)
                )
        return all_rows

    def _calculate_camera(self, camera_index, camera, camera_rows):
        fps = _resolve_camera_fps(camera, self.request.fps_override)
        samples = self._sample_camera(camera_index, camera, camera_rows)
        if not self._has_distance_measurements(samples):
            self.continuity.absolute_frame += len(camera_rows)
            return []

        gap_lengths, confidence = _interpolation_metadata(samples.interpolated_mask)
        pixel_series = self._smooth_pixel_series(samples, fps, confidence)
        homography_series = self._smooth_homography_series(samples, fps, confidence)
        self._update_continuity(pixel_series, homography_series)
        active_mode, active_series = self._select_active_series(
            pixel_series,
            homography_series,
        )
        result = _CameraSpeedSeries(
            samples=samples,
            gap_lengths=gap_lengths,
            confidence=confidence,
            pixel=pixel_series,
            homography=homography_series,
            active_mode=active_mode,
            active=active_series,
        )
        rows = self._build_camera_rows(camera_index, result)
        self.continuity.absolute_frame += len(camera_rows)
        return rows

    def _sample_camera(self, camera_index, camera, camera_rows):
        samples = _CameraDistanceSamples()
        pixel_camera = (
            self.pixel_cameras[camera_index]
            if camera_index < len(self.pixel_cameras)
            else {}
        )
        crop_params = camera.get('crop_params')
        crop_x = crop_params[0] if crop_params else 0
        crop_y = crop_params[1] if crop_params else 0

        for row in camera_rows:
            source_frame = int(row['source_frame'])
            frame_x, frame_y = self.offset_map.get(
                (source_frame, camera_index),
                (0, 0),
            )
            x1, x2 = int(row['x1']), int(row['x2'])
            y1, y2 = int(row['y1']), int(row['y2'])
            center_x = (x1 + x2) / 2.0 + frame_x + crop_x
            center_y = (y1 + y2) / 2.0 + frame_y + crop_y
            bottom_y = y2 + frame_y + crop_y

            samples.pixel.append(_pixel_distance_for_point(
                pixel_camera,
                center_x,
                center_y,
                self.continuity.pixel_distance,
            ))
            distance, world_point, image_point = _homography_distance_for_point(
                camera,
                center_x,
                bottom_y,
                self.continuity.homography_distance,
            )
            samples.homography.append(distance)
            samples.world_points.append(world_point)
            samples.image_points.append(image_point)
            samples.interpolated_mask.append(
                bool(int(row.get('is_interpolated', 0)))
            )
            samples.source_frames.append(source_frame)
        return samples

    @staticmethod
    def _has_distance_measurements(samples):
        return (
            any(value is not None for value in samples.pixel)
            or any(value is not None for value in samples.homography)
        )

    def _smooth_pixel_series(self, samples, fps, confidence):
        return _smooth_distance_samples(_SmoothingRequest(
            raw_values=samples.pixel,
            fps=fps,
            initial_velocity=self.continuity.pixel_velocity,
            initial_acceleration=self.continuity.pixel_acceleration,
            confidence=confidence,
        ))

    def _smooth_homography_series(self, samples, fps, confidence):
        return _smooth_distance_samples(_SmoothingRequest(
            raw_values=samples.homography,
            fps=fps,
            initial_velocity=self.continuity.homography_velocity,
            initial_acceleration=self.continuity.homography_acceleration,
            confidence=confidence,
        ))

    def _update_continuity(self, pixel_series, homography_series):
        if pixel_series is not None:
            self.continuity.pixel_distance = float(pixel_series.distance[-1])
            self.continuity.pixel_velocity = float(pixel_series.velocity[-1])
            self.continuity.pixel_acceleration = float(pixel_series.acceleration[-1])
        if homography_series is not None:
            self.continuity.homography_distance = float(homography_series.distance[-1])
            self.continuity.homography_velocity = float(homography_series.velocity[-1])
            self.continuity.homography_acceleration = float(
                homography_series.acceleration[-1]
            )

    def _select_active_series(self, pixel_series, homography_series):
        use_homography = homography_series is not None and (
            self.prefer_homography or pixel_series is None
        )
        if use_homography:
            return 'homography', homography_series
        return 'pixel', pixel_series

    def _build_camera_rows(self, camera_index, result):
        return [
            self._build_frame_row(camera_index, frame_index, result)
            for frame_index in range(len(result.samples.source_frames))
        ]

    def _build_frame_row(self, camera_index, frame_index, result):
        samples = result.samples
        world = samples.world_points[frame_index]
        image_point = samples.image_points[frame_index]
        raw_value = result.active.raw[frame_index]
        return {
            'cam': camera_index + 1,
            'cam_frame': frame_index,
            'source_frame': samples.source_frames[frame_index],
            'absolute_frame': self.continuity.absolute_frame + frame_index,
            'dist_m': round(float(result.active.distance[frame_index]), 3),
            'dist_raw_m': round(float(raw_value), 3) if raw_value is not None else '',
            'dist_smooth_m': round(float(result.active.distance[frame_index]), 3),
            'world_x': round(float(world[0]), 6) if world is not None else '',
            'image_point_x': round(float(image_point[0]), 3) if image_point is not None else '',
            'image_point_y': round(float(image_point[1]), 3) if image_point is not None else '',
            'speed_mps': round(float(result.active.velocity[frame_index]), 3),
            'accel_mps2': round(float(result.active.acceleration[frame_index]), 3),
            'speed_mode_used': result.active_mode,
            'dist_pixel_m': self._series_value(result.pixel, 'distance', frame_index),
            'speed_pixel_mps': self._series_value(result.pixel, 'velocity', frame_index),
            'accel_pixel_mps2': self._series_value(result.pixel, 'acceleration', frame_index),
            'dist_homography_m': self._series_value(result.homography, 'distance', frame_index),
            'speed_homography_mps': self._series_value(result.homography, 'velocity', frame_index),
            'accel_homography_mps2': self._series_value(result.homography, 'acceleration', frame_index),
            'is_interpolated': int(samples.interpolated_mask[frame_index]),
            'interp_gap_len': result.gap_lengths[frame_index],
            'speed_confidence': round(float(result.confidence[frame_index]), 3),
        }

    @staticmethod
    def _series_value(series, attribute, frame_index):
        if series is None:
            return ''
        return round(float(getattr(series, attribute)[frame_index]), 3)


def compute_speed_from_bbox_map(request):
    """不重新執行 YOLO，從 bbox map 計算逐幀雙模式速度指標。"""
    return _BBoxSpeedCalculator(request).calculate()


