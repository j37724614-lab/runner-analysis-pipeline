"""Pass 2：讀取已選主跑者快取、ROI 驗證、更新狀態、疊加框線、裁切輸出。"""

from dataclasses import dataclass

import cv2
import numpy as np

from core import tracking as _tracking
from core.draw_utils import (
    draw_dashed_line as _draw_dashed_line,
)
from core.tracking_geometry import _bbox_bottom_center

from .camera_setup import _project_and_check_track_roi
from .crop import _fixed_size_crop


def _cached_detections_to_arrays(cached_detections):
    """將已篩選的主跑者快取轉為下游使用的 boxes 與 ids 陣列。"""
    if cached_detections is None:
        raise ValueError("two-pass 第二遍必須提供 frame_cache")
    if cached_detections:
        boxes = np.array([[d['bx1'], d['by1'], d['bx2'], d['by2']]
                           for d in cached_detections], dtype=np.float32)
        ids = np.array([d['track_id'] for d in cached_detections], dtype=np.float32)
    else:
        boxes, ids = None, None
    return boxes, ids


@dataclass(frozen=True)
class FrameProcessingConfig:
    """Everything process_frame() needs beyond the per-frame image and the
    (mutated) selected-runner state. Rebuilt cheaply per frame by
    `_SingleCameraProcessor._track_source_frame()`."""

    track_roi: object = None
    quad_roi: object = None
    homography_lane_margin_px: int = 80
    overlay_start_pts: object = None
    overlay_end_pts: object = None


def _detection_passes_roi(ground_orig, config):
    """判斷主跑者的 bbox 底部中心點是否落在跑道投影範圍內。"""
    passes, _proj_px = _project_and_check_track_roi(ground_orig, config.track_roi)
    return passes


def _filter_to_homography_lane(valid_detections, config):
    """Keep only detections whose ground point is inside (or within the margin
    of) the homography runway quad -- but only if that leaves at least one."""
    if config.quad_roi is None or not valid_detections:
        return valid_detections
    lane_filtered = []
    for det in valid_detections:
        (_bx1, _by1, _bx2, _by2, _, ground_pt) = det
        signed_dist = cv2.pointPolygonTest(
            config.quad_roi,
            (float(ground_pt[0]), float(ground_pt[1])),
            True,
        )
        if signed_dist >= -float(config.homography_lane_margin_px):
            lane_filtered.append(det)
    return lane_filtered or valid_detections


def _collect_valid_detections(boxes, ids, config):
    """依高度、跑道方向投影及四邊形範圍過濾本幀主跑者偵測。

    回傳 valid_detections：list of
      (bx1, by1, bx2, by2, runner_id, ground_point)
    """
    if boxes is None or ids is None or len(boxes) == 0:
        return []

    valid_detections = []
    for i in range(len(boxes)):
        bx1, by1, bx2, by2 = map(int, boxes[i])
        if (by2 - by1) < _tracking.MIN_PERSON_HEIGHT:
            continue
        ground_pt = _bbox_bottom_center((bx1, by1, bx2, by2))
        if not _detection_passes_roi(ground_pt, config):
            continue
        tid = int(ids[i])
        valid_detections.append((bx1, by1, bx2, by2, tid, ground_pt))

    valid_detections = _filter_to_homography_lane(valid_detections, config)

    return valid_detections


def _update_selected_runner_state(runner_states, valid_detections):
    """更新第一遍已選定主跑者的 bbox 與平滑地面點。"""
    if not valid_detections:
        for tid in list(runner_states):
            runner_states[tid]['frames_since_detected'] += 1
            if (
                runner_states[tid]['frames_since_detected']
                > _tracking.SELECTED_RUNNER_MEMORY_FRAMES
            ):
                del runner_states[tid]
        return None

    if len(valid_detections) != 1:
        raise ValueError(
            "two-pass 第二遍每幀應只有一個已選主跑者偵測"
        )

    bx1, by1, bx2, by2, runner_id, ground_pt = valid_detections[0]
    previous = runner_states.get(runner_id)
    previous_ground = (
        previous.get('smoothed_ground_point', ground_pt)
        if previous is not None else ground_pt
    )
    alpha = _tracking.GROUND_POINT_EMA_ALPHA
    smoothed_ground = (
        alpha * ground_pt[0] + (1.0 - alpha) * previous_ground[0],
        alpha * ground_pt[1] + (1.0 - alpha) * previous_ground[1],
    )

    runner_states.clear()
    runner_states[runner_id] = {
        'bbox': (bx1, by1, bx2, by2),
        'ground_point': ground_pt,
        'smoothed_ground_point': smoothed_ground,
        'frames_since_detected': 0,
    }
    return runner_id


def _draw_runner_overlay(img, runner_id, bbox):
    if not _tracking.DRAW_BBOX_OVERLAY:
        return
    bx1, by1, bx2, by2 = bbox
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    label = f"ID {runner_id}"
    label_y = max(by1 - 8, 20)
    for color, thickness in (((0, 0, 0), 4), ((0, 255, 0), 2)):
        cv2.putText(
            img, label, (bx1, label_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, color, thickness, cv2.LINE_AA,
        )


def _draw_track_boundary_overlays(img, start_points, end_points):
    if start_points and end_points:
        p0, p3 = start_points
        p1, p2 = end_points
        quad = np.array([p0, p1, p2, p3], dtype=np.int32)
        overlay_img = img.copy()
        cv2.fillPoly(overlay_img, [quad], (200, 220, 255))
        cv2.addWeighted(overlay_img, 0.15, img, 0.85, 0, img)
        for start, end in ((p0, p1), (p1, p2), (p2, p3), (p3, p0)):
            _draw_dashed_line(img, start, end, (255, 255, 255), thickness=2)
    if start_points:
        cv2.line(img, start_points[0], start_points[1], (0, 0, 0), 5)
        cv2.line(img, start_points[0], start_points[1], (180, 255, 255), 3)
    if end_points:
        cv2.line(img, end_points[0], end_points[1], (0, 0, 0), 5)
        cv2.line(img, end_points[0], end_points[1], (255, 200, 100), 3)


def _draw_frame_overlays(img, runner_id, bbox, config):
    """在前處理裁剪後的畫面上疊加跑者框與起終點線。"""
    _draw_runner_overlay(img, runner_id, bbox)
    _draw_track_boundary_overlays(
        img, config.overlay_start_pts, config.overlay_end_pts,
    )
    return img


def process_frame(img, runner_states, config, cached_detections):
    """
    對單幀執行：讀取第一遍已選主跑者快取 → ROI 驗證 →
                更新 bbox/地面點狀態 → 疊加框 → 固定大小裁剪。

    ``config`` is a FrameProcessingConfig; ``cached_detections`` is a list of
    ``{bx1,by1,bx2,by2,track_id}`` dicts from the two-pass first pass.

    回傳：(crop_frame, runner_id, runner_center_orig, runner_bx2_orig,
           bbox_in_crop, c1x_orig, c1y_orig)
      crop_frame=None          → ROI 內無有效人物，上層應跳過此幀
      runner_center_orig       → 主跑者 center_x（原始座標），非最後一機的切換基準
      runner_bx2_orig          → 主跑者 bbox 右緣（原始座標），最後一機的退出 ROI 基準
      bbox_in_crop             → 最終輸出裁切畫面內的 bbox，供 HRNet 限定偵測範圍
    """
    # Step 1: 讀取 two-pass 第一遍快取（原始影格座標）
    boxes, ids = _cached_detections_to_arrays(cached_detections)

    # Step 2: ROI 驗證 + 已選主跑者狀態更新
    valid_detections = _collect_valid_detections(boxes, ids, config)
    runner_id = _update_selected_runner_state(runner_states, valid_detections)
    if runner_id is None:
        return None, None, None, None, None, None, None

    d = runner_states[runner_id]
    bx1, by1, bx2, by2 = d['bbox']
    ground_x, _ground_y = d.get('ground_point', _bbox_bottom_center((bx1, by1, bx2, by2)))
    runner_center_orig = ground_x
    runner_bx2_orig = bx2

    # Step 3: 在原始影格上疊加框
    if _tracking.SHOW_OVERLAY:
        img = _draw_frame_overlays(img, runner_id, (bx1, by1, bx2, by2), config)

    # Step 4: 固定大小裁剪（以已選主跑者 bbox 中心為基準）
    crop_frame, bbox_in_crop, c1x, c1y = _fixed_size_crop(img, bx1, by1, bx2, by2)

    return (crop_frame, runner_id, runner_center_orig, runner_bx2_orig,
            bbox_in_crop, c1x, c1y)


