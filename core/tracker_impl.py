import cv2

try:
    cv2.setLogLevel(3)  # type: ignore[attr-defined]  # 抑制 swscaler HDR 色彩轉換警告
except AttributeError:
    pass  # 舊版 OpenCV 無此 API，忽略
import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field, replace
from enum import Enum

import matplotlib
import numpy as np
import torch
from ultralytics import YOLO

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter  # type: ignore[import-untyped]
from matplotlib import font_manager as fm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.signal import butter, filtfilt

from core.draw_utils import (
    draw_dashed_line as _draw_dashed_line,
    draw_text_bgr as _draw_text_bgr,
    get_font as _get_font,
)
from core.utils import DEFAULT_OUTPUT_DIR, get_font_path, get_model_path

# =======================================================================
# 預設參數與常數（供模組化 import，外部亦可傳參覆蓋）
# =======================================================================
DEFAULT_CUDA_VISIBLE_DEVICES = '0'
DEFAULT_DEVICE = 0
DEFAULT_MODEL_PATH = get_model_path("yolo26x.pt")
DEFAULT_FONT_PATH = get_font_path()
DEFAULT_OUTPUT_NAME = "sequential_tracked.mp4"
DEFAULT_TARGET_HEIGHT = 340
DEFAULT_CHART_HEIGHT = 200

DEFAULT_MOVEMENT_THRESHOLD  = 3   # 判定為移動的最小像素位移
DEFAULT_MIN_MOVEMENT_FRAMES = 3   # 需連續移動至少此幀數才視為「真正移動」
DEFAULT_STATIONARY_DECAY    = 2   # 靜止時每幀遞減 movement_count 的量
DEFAULT_MAX_PERSON_MEMORY   = 30  # 超過此幀數未偵測到則清除該人物的速度紀錄
DEFAULT_CAM_WARMUP_FRAMES   = 5   # 切換相機後前幾幀放寬選取條件
DEFAULT_MIN_PERSON_HEIGHT   = 40  # bbox 高度小於此值（裁切後像素）視為背景遠景人物，略過
DEFAULT_GROUND_POINT_EMA_ALPHA = 0.35  # bbox 底部中心點 EMA 平滑係數；越小越穩但延遲越大
DEFAULT_FLAT_INTERP_EPS_M = 0.001      # 距離變化小於此值視為 flat segment


def _configure_matplotlib_font(font_path=DEFAULT_FONT_PATH):
    """設定 Matplotlib 使用中文字型。"""
    if font_path and os.path.exists(font_path):
        try:
            if hasattr(fm.fontManager, 'addfont'):
                fm.fontManager.addfont(font_path)
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            plt.rcParams['font.family'] = [font_name]
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"已載入中文字型: {font_name} ({font_path})")
            return font_prop
        except Exception as e:
            print(f"中文字型載入失敗，改用預設字型: {e}")
    else:
        print(f"找不到中文字型檔: {font_path}")

    plt.rcParams['axes.unicode_minus'] = False
    return None


def _project_onto_track(point, start_mid, track_dir):
    """將 point 投影到 track_dir 方向，回傳從 start_mid 起的有號像素距離。"""
    dx = point[0] - start_mid[0]
    dy = point[1] - start_mid[1]
    return dx * track_dir[0] + dy * track_dir[1]


def _project_point_to_track_line(point, start_mid, track_dir):
    """將 point 投影到 start_mid + track_dir 定義的跑道中心線上。"""
    proj = _project_onto_track(point, start_mid, track_dir)
    return (
        start_mid[0] + track_dir[0] * proj,
        start_mid[1] + track_dir[1] * proj,
    )


def _compute_homography(src_points, dst_points_world):
    """根據對應點計算 Homography，失敗時拋出 ValueError。"""
    if src_points is None or dst_points_world is None:
        return None, None

    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points_world, dtype=np.float32)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(
            "homography_src_points / homography_dst_points_world 必須是 shape=(N,2) 且點數一致"
        )
    if len(src) < 4:
        raise ValueError("Homography 至少需要 4 組對應點")

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography 計算失敗，請檢查點位是否共線或順序是否對應")
    return H, mask


def _transform_point_homography(point, H):
    """把單一影像點轉到世界座標，回傳 (xw, yw)。"""
    if H is None:
        return None
    pts = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pts, H)
    return tuple(mapped[0, 0])


def _inverse_transform_points_homography(points_world, H):
    """把一批世界座標點反投影回影像座標。"""
    if H is None:
        return None
    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, np.linalg.inv(H))
    return mapped.reshape(-1, 2)


def _draw_cv_label(img, text, xy, color=(255, 255, 255), scale=0.55):
    """用 OpenCV 畫有黑色外框的短標籤。"""
    x, y = int(round(xy[0])), int(round(xy[1]))
    cv2.putText(img, str(text), (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, str(text), (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def _homography_point_color(raw_mps):
    """Homography 可視圖上的速度分類色彩：紅 <9、綠 9~11、藍 >11。"""
    if raw_mps is None:
        return (160, 160, 160)
    if raw_mps < 9.0:
        return (0, 0, 255)
    if raw_mps <= 11.0:
        return (0, 210, 0)
    return (255, 0, 0)


def _write_homography_visualizations(cameras, output_dir, track_data=None):
    """輸出每台相機的 Homography 線、控制點，以及可選的跑者落點速度分類。"""
    os.makedirs(output_dir, exist_ok=True)
    track_by_cam = {}
    if track_data:
        prev_dist_by_cam = {}
        for row in track_data:
            cam_no = row.get('cam')
            if cam_no is None:
                continue
            raw_mps = None
            dist_raw = row.get('dist_raw_m')
            prev_dist = prev_dist_by_cam.get(cam_no)
            if dist_raw not in ('', None):
                try:
                    dist_raw = float(dist_raw)
                    if prev_dist is not None:
                        raw_mps = (dist_raw - prev_dist) * 60.0
                    prev_dist_by_cam[cam_no] = dist_raw
                except (TypeError, ValueError):
                    pass
            enriched = dict(row)
            enriched['raw_mps_for_viz'] = raw_mps
            track_by_cam.setdefault(cam_no, []).append(enriched)

    written = []
    for cam_idx, cam in enumerate(cameras, start=1):
        H = cam.get('H_matrix')
        src_points = cam.get('homography_src_points')
        dst_world = cam.get('homography_dst_world')
        video_path = cam.get('video_path')
        if H is None or src_points is None or dst_world is None or not video_path:
            continue

        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"Homography 可視圖略過 cam{cam_idx}: 無法讀取第一幀")
            continue

        img = frame.copy()
        dst_world = np.asarray(dst_world, dtype=np.float32)
        src_points = np.asarray(src_points, dtype=np.float32)
        x_min = float(np.min(dst_world[:, 0]))
        x_max = float(np.max(dst_world[:, 0]))
        y_min = float(np.min(dst_world[:, 1]))
        y_max = float(np.max(dst_world[:, 1]))
        y_mid = (y_min + y_max) / 2.0

        lane_poly = _inverse_transform_points_homography(
            [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max]], H)
        if lane_poly is not None:
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.rint(lane_poly).astype(np.int32)], (230, 230, 180))
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

        # 每 2m 一條輔助線，每 10m 一條主線。
        for meter in np.arange(x_min, x_max + 0.001, 2.0):
            projected = _inverse_transform_points_homography(
                [[meter, y_min], [meter, y_max]], H)
            if projected is None:
                continue
            p0, p1 = np.rint(projected).astype(np.int32)
            major = abs((meter - x_min) % 10.0) < 1e-6 or abs(meter - x_max) < 1e-6
            color = (0, 255, 255) if major else (190, 190, 190)
            cv2.line(img, tuple(p0), tuple(p1), color, 3 if major else 1, cv2.LINE_AA)
            if major:
                _draw_cv_label(img, f"{int(round(meter))}m", (projected[0] + projected[1]) / 2,
                               color=(0, 255, 255), scale=0.7)

        # 遠側、中心線、近側。
        for y_val, color, thickness in [
                (y_min, (255, 120, 60), 2),
                (y_mid, (255, 255, 255), 1),
                (y_max, (80, 220, 80), 2)]:
            world_line = [[x, y_val] for x in np.linspace(x_min, x_max, 41)]
            projected = _inverse_transform_points_homography(world_line, H)
            if projected is None:
                continue
            projected = np.rint(projected).astype(np.int32)
            for p0, p1 in zip(projected[:-1], projected[1:]):
                cv2.line(img, tuple(p0), tuple(p1), color, thickness, cv2.LINE_AA)

        for idx, (src_pt, world_pt) in enumerate(zip(src_points, dst_world), start=1):
            p = tuple(np.rint(src_pt).astype(np.int32))
            cv2.circle(img, p, 7, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, p, 5, (0, 255, 255), -1, cv2.LINE_AA)
            _draw_cv_label(img, f"P{idx} {world_pt[0]:.0f}m,{world_pt[1]:.2f}",
                           p, color=(0, 255, 255), scale=0.5)

        for row in track_by_cam.get(cam_idx, []):
            x = row.get('image_point_x')
            y = row.get('image_point_y')
            if x in ('', None) or y in ('', None):
                continue
            try:
                pt = (int(round(float(x))), int(round(float(y))))
            except (TypeError, ValueError):
                continue
            cv2.circle(img, pt, 3, _homography_point_color(row.get('raw_mps_for_viz')),
                       -1, cv2.LINE_AA)

        cv2.rectangle(img, (15, 15), (760, 105), (0, 0, 0), -1)
        cv2.putText(img,
                    f"CAM{cam_idx} Homography visualization ({int(round(x_min))}-{int(round(x_max))}m, {len(src_points)} points)",
                    (28, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "landing points: red raw<9, green 9-11, blue >11 m/s",
                    (28, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)

        full_path = os.path.join(output_dir, f"cam{cam_idx}_homography_visualization.jpg")
        cv2.imwrite(full_path, img)
        written.append(full_path)

        crop = cam.get('crop_params')
        if crop:
            x1, y1, x2, y2 = [int(v) for v in crop]
            crop_img = img[y1:y2, x1:x2]
            crop_path = os.path.join(output_dir, f"cam{cam_idx}_homography_visualization_crop.jpg")
            cv2.imwrite(crop_path, crop_img)
            written.append(crop_path)

    if written:
        print("Homography 可視圖輸出：")
        for path in written:
            print(f"  {path}")
    return written


def _point_in_quad(point, quad_pts):
    """判斷 point 是否在四邊形內（含邊界）。quad_pts 為 np.float32 shape=(4,2)。"""
    return cv2.pointPolygonTest(quad_pts, (float(point[0]), float(point[1])), False) >= 0


def _bbox_bottom_center(bx1, by1, bx2, by2):
    """回傳 bbox 底邊中心點，作為地面位置的近似點。"""
    return ((float(bx1) + float(bx2)) / 2.0, float(by2))

# -----------------------------------------------------------------------
# camera() — 快速建立相機設定的 helper function
#
# 必填：
#   video_path  str or None   影片路徑；填 None 表示此台不使用（自動跳過）
#
# 選填（有預設值）：
#   crop        (x起,y起,x終,y終)  前處理裁剪範圍，None = 不裁剪
#   roi_x       (左, 右)           ROI x 範圍，以 bbox 右邊界判斷，預設不限制
#   roi_y       (上, 下)           ROI y 範圍，以 bbox 中心點判斷，預設不限制
#   switch_x    int or None        最快人物 center_x（原始座標）超過此值時切下一台
#                                  None = 跑完整段再切（最後有效台自動設為 None）
#   distance_m  float or None      roi_x 範圍對應的實際距離（公尺）
#                                  填入後程式自動計算 m_per_pixel 並啟用速度圖表
#                                  None = 不計算距離/速度（預設）
#
# 若需要多個 ROI zone，改用 roi_zones 參數直接傳入 list：
#   roi_zones=[{'x':(0,800),'y':(0,9999)}, {'x':(1000,1920),'y':(0,9999)}]
# -----------------------------------------------------------------------
def camera(video_path, crop=None,
           roi_x=(0, 9999), roi_y=(0, 9999),
           switch_x=None, roi_zones=None,
           distance_m=None,
           start_line=None, end_line=None,
           pre_roll_px=200,
           end_roll_px=120,
           start_gate_px=250,
           start_roi_px=100,
           start_confirm_move_px=8,
           H_matrix=None,
           homography_src_points=None,
           homography_dst_world=None):
    """
    start_line / end_line（可選）：各由兩個原始影像座標點組成的斜線，
      例如 start_line=[(150, 420), (150, 780)]。
    同時填入兩者時：
      - m_per_pixel 改由兩中點的像素距離換算（取代 roi_x 方式）
      - switch_x 自動設為 None（切換改由 end_line 越線事件觸發）
    """
    zones = roi_zones if roi_zones is not None else [{'x': roi_x, 'y': roi_y}]
    no_roi = (roi_zones is None and roi_x == (0, 9999) and roi_y == (0, 9999))

    # 斜線模式：兩條線同時填入才啟用
    start_mid = end_mid = track_dir = pixel_span = quad_roi = None
    if start_line is not None and end_line is not None:
        start_mid = ((start_line[0][0] + start_line[1][0]) / 2.0,
                     (start_line[0][1] + start_line[1][1]) / 2.0)
        end_mid   = ((end_line[0][0] + end_line[1][0]) / 2.0,
                     (end_line[0][1] + end_line[1][1]) / 2.0)
        dx = end_mid[0] - start_mid[0]
        dy = end_mid[1] - start_mid[1]
        pixel_span = (dx ** 2 + dy ** 2) ** 0.5
        if pixel_span > 0:
            track_dir = (dx / pixel_span, dy / pixel_span)
        switch_x = None  # end_line 取代 switch_x
        quad_roi = np.array(
            [start_line[0], end_line[0], end_line[1], start_line[1]],
            dtype=np.float32
        )

    # Homography 模式的世界座標起點（取最小 Xw 作為本段 0m）
    homography_start_x = None
    if H_matrix is not None:
        if start_line is not None:
            start_world = [_transform_point_homography(pt, H_matrix) for pt in start_line]
            if all(p is not None for p in start_world):
                homography_start_x = float(np.mean([p[0] for p in start_world]))
        elif roi_x[0] != 0 or roi_y[0] != 0:
            start_world = _transform_point_homography((roi_x[0], roi_y[0]), H_matrix)
            if start_world is not None:
                homography_start_x = float(start_world[0])
        else:
            homography_start_x = 0.0

    # 自動推算起跑點像素與公尺/像素換算比例
    start_x = start_mid[0] if start_mid else roi_x[0]
    if H_matrix is not None:
        m_per_pixel = None
    elif start_line is not None and end_line is not None and pixel_span and distance_m is not None:
        m_per_pixel = distance_m / pixel_span
    elif distance_m is not None and roi_x[1] > roi_x[0]:
        m_per_pixel = distance_m / (roi_x[1] - roi_x[0])
    else:
        m_per_pixel = None

    return {
        'video_path':  video_path,
        'crop_params': crop,
        'roi_enabled': not no_roi,
        'roi_zones':   zones,
        'switch_x':    switch_x,
        'start_x':     start_x,      # 原始影像座標（舊模式用 roi_x[0]，新模式用 start_mid.x）
        'm_per_pixel': m_per_pixel,  # 公尺/像素；None = 不計算距離
        'distance_m':  distance_m,   # 原始設定距離（公尺），供全程固定軸範圍用
        # 斜線模式額外欄位（舊模式均為 None）
        'start_line':  start_line,
        'end_line':    end_line,
        'start_mid':   start_mid,
        'end_mid':     end_mid,
        'track_dir':   track_dir,
        'pixel_span':  pixel_span,
        'quad_roi':    quad_roi,
        'track_roi':   {'start_mid': start_mid, 'track_dir': track_dir,
                        'pixel_span': pixel_span, 'pre_roll_px': pre_roll_px,
                        'end_roll_px': end_roll_px,
                        'start_roi_px': start_roi_px,
                        'start_gate_px': start_gate_px,
                        'start_confirm_move_px': start_confirm_move_px}
                       if start_mid is not None else None,
        'H_matrix':    H_matrix,
        'homography_start_x': homography_start_x,
        'homography_src_points': (
            np.asarray(homography_src_points, dtype=np.float32)
            if homography_src_points is not None else None
        ),
        'homography_dst_world': (
            np.asarray(homography_dst_world, dtype=np.float32)
            if homography_dst_world is not None else None
        ),
    }


LANE_WIDTH_M = 1.22


def _build_camera_from_json(entry):
    sl = entry.get('start_line')
    el = entry.get('end_line')
    crop_val = entry.get('crop')

    H_matrix = None
    src_pts = entry.get('homography_src_points')
    dst_pts = entry.get('homography_dst_world')
    if src_pts and dst_pts:
        H_matrix, _ = _compute_homography(
            np.float32(src_pts), np.float32(dst_pts))
    elif sl and el and entry.get('distance_m') is not None:
        # Auto-build homography from the 4 anchor corners (start_line × 2 + end_line × 2).
        # Near side = higher image-y (lower in frame); far side = lower image-y.
        s0, s1 = [float(v) for v in sl[0]], [float(v) for v in sl[1]]
        e0, e1 = [float(v) for v in el[0]], [float(v) for v in el[1]]
        start_far, start_near = (s0, s1) if s0[1] <= s1[1] else (s1, s0)
        end_far,   end_near   = (e0, e1) if e0[1] <= e1[1] else (e1, e0)
        dist = float(entry['distance_m'])
        src_pts = [start_far, start_near, end_far, end_near]
        dst_pts = [[0.0, 0.0], [0.0, LANE_WIDTH_M],
                   [dist, 0.0], [dist, LANE_WIDTH_M]]
        try:
            H_matrix, _ = _compute_homography(np.float32(src_pts), np.float32(dst_pts))
            # Reject near-singular H matrices: when the anchor quad has tiny vertical
            # extent (side-on camera), the H matrix is poorly conditioned and amplifies
            # bbox x-noise into large world_x errors, causing speed to trend upward.
            # Fall back to linear projection which is stable for side-on cameras.
            if np.linalg.cond(H_matrix) > 5000:
                H_matrix = None
                src_pts = dst_pts = None
        except Exception:
            H_matrix = None
            src_pts = dst_pts = None

    return camera(
        video_path=entry.get('video_path'),
        crop=tuple(crop_val) if crop_val else None,
        roi_x=tuple(entry['roi_x']) if 'roi_x' in entry else (0, 9999),
        roi_y=tuple(entry['roi_y']) if 'roi_y' in entry else (0, 9999),
        switch_x=entry.get('switch_x'),
        roi_zones=entry.get('roi_zones'),
        distance_m=entry.get('distance_m'),
        start_line=[tuple(p) for p in sl] if sl else None,
        end_line=[tuple(p) for p in el] if el else None,
        pre_roll_px=int(entry.get('pre_roll_px', 200)),
        end_roll_px=int(entry.get('end_roll_px', 120)),
        start_roi_px=int(entry.get('start_roi_px', 100)),
        H_matrix=H_matrix,
        homography_src_points=src_pts,
        homography_dst_world=dst_pts,
    )


def build_lane_world_points(start_meter, num_points=5, lane_width=LANE_WIDTH_M):
    """建立單一跑道區段的世界座標；支援 5 點或 6 點 Homography 標定。"""
    if num_points == 5:
        return np.float32([
            [start_meter + 0.0, lane_width],
            [start_meter + 0.0, 0.0],
            [start_meter + 10.0, 0.0],
            [start_meter + 20.0, 0.0],
            [start_meter + 20.0, lane_width],
        ])

    if num_points == 6:
        return np.float32([
            [start_meter + 0.0, lane_width],
            [start_meter + 0.0, 0.0],
            [start_meter + 10.0, 0.0],
            [start_meter + 10.0, lane_width],
            [start_meter + 20.0, 0.0],
            [start_meter + 20.0, lane_width],
        ])

    raise ValueError("src_points 目前只支援 5 點或 6 點 Homography 標定")


def build_camera_with_homography(video_path, crop, start_line, end_line,
                                 start_meter, src_points=None, distance_m=20,
                                 roi_x=(0, 9999), roi_y=(0, 9999),
                                 switch_x=None, roi_zones=None,
                                 pre_roll_px=200,
                                 end_roll_px=120,
                                 start_gate_px=250,
                                 start_roi_px=100,
                                 start_confirm_move_px=8):
    """
    簡化版相機設定：
      - 填入 start_meter 與 5/6 個 src_points 時，自動計算 Homography
      - src_points 未填滿時，自動退回既有 start_line/end_line 或 roi_x 量測模式
    src_points 固定順序：
      1. 0m 靠近側
      2. 0m 遠離側
      3. 10m 遠離側
      4. 20m 遠離側
      5. 20m 靠近側
    若使用 6 點，順序為：
      1. 0m 靠近側
      2. 0m 遠離側
      3. 10m 遠離側
      4. 10m 靠近側
      5. 20m 遠離側
      6. 20m 靠近側
    """
    H_matrix = None
    dst_points_world = None
    if src_points is not None and len(src_points) >= 4:
        dst_points_world = build_lane_world_points(start_meter, num_points=len(src_points))
        H_matrix, _ = _compute_homography(np.float32(src_points), dst_points_world)

    return camera(
        video_path=video_path,
        crop=crop,
        roi_x=roi_x,
        roi_y=roi_y,
        switch_x=switch_x,
        roi_zones=roi_zones,
        distance_m=distance_m,
        start_line=start_line,
        end_line=end_line,
        pre_roll_px=pre_roll_px,
        end_roll_px=end_roll_px,
        start_gate_px=start_gate_px,
        start_roi_px=start_roi_px,
        start_confirm_move_px=start_confirm_move_px,
        H_matrix=H_matrix,
        homography_src_points=src_points,
        homography_dst_world=dst_points_world,
    )


# -----------------------------------------------------------------------
# 相機設定（最多 6 台）
# -----------------------------------------------------------------------
def get_default_cameras():
    """建立並回傳預設的 6 台相機設定清單。"""
    cam1 = build_camera_with_homography(
        video_path="test/test/cam1.mov",
        crop=(0, 400, 1920, 800),
        start_line=[(222, 715), (148, 725)],
        end_line=[(1700, 710), (1790, 718)],
        start_meter=0.0,
        start_roi_px=100,
        src_points=[
            [148, 725],   # 0m 靠近側
            [222, 715],   # 0m 遠離側
            [961, 714],   # 10m 遠離側
            [966, 723],   # 10m 靠近側（由跑道透視與 1.22m 寬度推算）
            [1700, 710],  # 20m 遠離側
            [1790, 720],  # 20m 靠近側
        ],
    )

    cam2 = build_camera_with_homography(
        video_path="test/test/cam2.mov",
        crop=(0, 400, 1920, 800),
        start_line=[(220, 715), (135, 725)],
        end_line=[(1730, 710), (1825, 725)],
        start_meter=20.0,
        src_points=[
            [135, 725],   # 20m 靠近側
            [220, 715],   # 20m 遠離側
            [970, 714],   # 30m 遠離側
            [970, 721],   # 30m 靠近側（由跑道透視與 1.22m 寬度推算）
            [1730, 710],  # 40m 遠離側
            [1825, 725],  # 40m 靠近側
        ],
    )

    cam3 = build_camera_with_homography(
        video_path=None,
        crop=(0, 400, 1920, 800),
        start_line=[(212, 715), (127, 725)],
        end_line=[(1755, 710), (1835, 718)],
        start_meter=40.0,
        src_points=[
            [127, 725],   # 40m 靠近側
            [212, 715],   # 40m 遠離側
            [980, 714],   # 50m 遠離側
            [983, 721],   # 50m 靠近側（由跑道透視與 1.22m 寬度推算）
            [1755, 710],  # 60m 遠離側
            [1835, 718],  # 60m 靠近側
        ],
    )

    cam4 = build_camera_with_homography(
        video_path=None,
        crop=(0, 400, 1920, 800),
        start_line=[(227, 713), (140, 727)],
        end_line=[(1722, 718), (1825, 725)],
        start_meter=60.0,
        src_points=[
            [140, 727],   # 60m 靠近側
            [227, 713],   # 60m 遠離側
            [976, 716],   # 70m 遠離側
            [976, 726],   # 70m 靠近側（由跑道透視與 1.22m 寬度推算）
            [1722, 718],  # 80m 遠離側
            [1825, 725],  # 80m 靠近側
        ],
    )

    cam5 = build_camera_with_homography(
        video_path=None,
        crop=(0, 400, 1920, 800),
        start_line=[(150, 400), (150, 800)],
        end_line=[(1820, 400), (1820, 800)],
        start_meter=80.0,
        src_points=[],
    )

    cam6 = build_camera_with_homography(
        video_path=None,
        crop=(0, 400, 1920, 800),
        start_line=[(150, 400), (150, 800)],
        end_line=[(1820, 400), (1820, 800)],
        start_meter=100.0,
        src_points=[],
    )
    return [cam1, cam2, cam3, cam4, cam5, cam6]

# =======================================================================
# 以下為程式邏輯，一般不需修改
# =======================================================================

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

    for left, right in zip(valid, valid[1:]):
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


def _interpolate_missing_bboxes(values):
    """用前後有效 bbox 線性補齊 None；端點缺值用最近有效 bbox 延伸。"""
    out = list(values)
    valid = [i for i, v in enumerate(out) if v is not None]
    if not valid:
        return out

    first = valid[0]
    for i in range(first):
        out[i] = out[first]

    for left, right in zip(valid, valid[1:]):
        if right == left + 1:
            continue
        start = np.asarray(out[left], dtype=float)
        end = np.asarray(out[right], dtype=float)
        span = right - left
        for i in range(left + 1, right):
            ratio = (i - left) / span
            out[i] = tuple(np.rint(start + ratio * (end - start)).astype(int))

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
    except Exception:
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
    except Exception:
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


def compute_speed_from_bbox_map(bbox_map_csv, cameras_cfg_list, fps_override=None,
                                offsets_npz=None, pixel_cameras_cfg_list=None,
                                speed_mode='pixel'):
    """
    Compute dual pixel and homography speed/acceleration metrics from
    bbox_map.csv without re-running YOLO.

    ``cameras_cfg_list`` retains the original six-point Homography controls.
    ``pixel_cameras_cfg_list`` is the tracking-safe start/end-line copy used
    by the legacy pixel calibration. Both paths are always calculated when
    available; ``speed_mode`` only selects which result is exposed through the
    backwards-compatible ``speed_mps`` / ``accel_mps2`` columns.

    bbox_map.csv stores coordinates in per-frame person-centered crop space.
    offsets_npz (cam1_offsets.npz) contains the (off_x, off_y) per output frame
    needed to convert back to original image coordinates.
    """
    offset_map = _build_frame_offset_map(offsets_npz)
    rows_by_cam = _read_bbox_rows_by_camera(bbox_map_csv)

    cameras = [_build_camera_from_json(c) for c in cameras_cfg_list]
    pixel_cameras_cfg_list = pixel_cameras_cfg_list or cameras_cfg_list
    pixel_cameras = [_build_camera_from_json(c) for c in pixel_cameras_cfg_list]
    use_homography = str(speed_mode).lower() == 'homography'

    all_track_data = []
    pixel_cumulative_dist_offset = 0.0
    homography_cumulative_dist_offset = 0.0
    absolute_frame_offset = 0
    last_pixel_kf_v = last_pixel_kf_a = 0.0
    last_homography_kf_v = last_homography_kf_a = 0.0

    for cam_idx, cam in enumerate(cameras):
        cam_rows = rows_by_cam.get(cam_idx, [])
        if not cam_rows:
            continue

        fps = _resolve_camera_fps(cam, fps_override)

        cp = cam.get('crop_params')
        crop_x_offset = cp[0] if cp else 0
        crop_y_offset = cp[1] if cp else 0

        pixel_cam = pixel_cameras[cam_idx] if cam_idx < len(pixel_cameras) else {}
        pixel_raw = []
        homography_raw = []
        world_points = []
        homography_image_points = []
        interpolated_mask = []
        source_frames = []

        for row in cam_rows:
            x1, x2 = int(row['x1']), int(row['x2'])
            y2 = int(row['y2'])
            # bbox coords are in person-centered crop space; add frame-level offset
            # (off_x, off_y) from offsets.npz to recover original image coordinates.
            src_frame = int(row['source_frame'])
            frame_off_x, frame_off_y = offset_map.get((src_frame, cam_idx), (0, 0))
            cx_orig = (x1 + x2) / 2.0 + frame_off_x + crop_x_offset
            cy_orig = (int(row['y1']) + y2) / 2.0 + frame_off_y + crop_y_offset
            y2_orig = y2 + frame_off_y + crop_y_offset

            pixel_dist = _pixel_distance_for_point(
                pixel_cam, cx_orig, cy_orig, pixel_cumulative_dist_offset)
            homography_dist, world, image_point = _homography_distance_for_point(
                cam, cx_orig, y2_orig, homography_cumulative_dist_offset)

            pixel_raw.append(pixel_dist)
            homography_raw.append(homography_dist)
            world_points.append(world)
            homography_image_points.append(image_point)
            interpolated_mask.append(bool(int(row.get('is_interpolated', 0))))
            source_frames.append(src_frame)

        pixel_available = any(v is not None for v in pixel_raw)
        homography_available = any(v is not None for v in homography_raw)
        if not pixel_available and not homography_available:
            absolute_frame_offset += len(cam_rows)
            continue

        interp_gap_len, speed_confidence = _interpolation_metadata(interpolated_mask)
        def _smooth(raw, available, init_v, init_a):
            if not available:
                return None, None, None
            raw_for_csv = list(raw)
            values = _interpolate_missing_numeric(raw) if any(v is None for v in raw) else raw
            smooth, velocity, accel = _compute_kf_series(
                values, fps, init_v=init_v, init_a=init_a,
                measurement_confidence=speed_confidence)
            return raw_for_csv, smooth, (velocity, accel)

        pixel_for_csv, pixel_smooth, pixel_series = _smooth(
            pixel_raw, pixel_available, last_pixel_kf_v, last_pixel_kf_a)
        homo_for_csv, homo_smooth, homo_series = _smooth(
            homography_raw, homography_available,
            last_homography_kf_v, last_homography_kf_a)
        pixel_velocity, pixel_accel = pixel_series if pixel_series else (None, None)
        homo_velocity, homo_accel = homo_series if homo_series else (None, None)

        if pixel_smooth is not None:
            pixel_cumulative_dist_offset = float(pixel_smooth[-1])
            last_pixel_kf_v = float(pixel_velocity[-1])
            last_pixel_kf_a = float(pixel_accel[-1])
        if homo_smooth is not None:
            homography_cumulative_dist_offset = float(homo_smooth[-1])
            last_homography_kf_v = float(homo_velocity[-1])
            last_homography_kf_a = float(homo_accel[-1])

        # Explicit homography mode prefers the world-coordinate series, but a
        # partially calibrated legacy call must still fall back safely rather
        # than emitting an empty active metric.
        active_is_homography = homo_smooth is not None and (
            use_homography or pixel_smooth is None)
        active_raw = homo_for_csv if active_is_homography else pixel_for_csv
        active_smooth = homo_smooth if active_is_homography else pixel_smooth
        active_velocity = homo_velocity if active_is_homography else pixel_velocity
        active_accel = homo_accel if active_is_homography else pixel_accel
        active_mode = 'homography' if active_is_homography else 'pixel'

        for i in range(len(cam_rows)):
            raw_value = active_raw[i] if active_raw is not None else None
            world = world_points[i]
            image_point = homography_image_points[i]
            all_track_data.append({
                'cam':             cam_idx + 1,
                'cam_frame':       i,
                'source_frame':    source_frames[i] if i < len(source_frames) else '',
                'absolute_frame':  absolute_frame_offset + i,
                'dist_m':          round(float(active_smooth[i]), 3),
                'dist_raw_m':      round(float(raw_value), 3) if raw_value is not None else '',
                'dist_smooth_m':   round(float(active_smooth[i]), 3),
                'world_x':         round(float(world[0]), 6) if world is not None else '',
                'image_point_x':   round(float(image_point[0]), 3) if image_point is not None else '',
                'image_point_y':   round(float(image_point[1]), 3) if image_point is not None else '',
                'speed_mps':       round(float(active_velocity[i]), 3),
                'accel_mps2':      round(float(active_accel[i]), 3),
                'speed_mode_used': active_mode,
                'dist_pixel_m':    round(float(pixel_smooth[i]), 3) if pixel_smooth is not None else '',
                'speed_pixel_mps': round(float(pixel_velocity[i]), 3) if pixel_velocity is not None else '',
                'accel_pixel_mps2': round(float(pixel_accel[i]), 3) if pixel_accel is not None else '',
                'dist_homography_m': round(float(homo_smooth[i]), 3) if homo_smooth is not None else '',
                'speed_homography_mps': round(float(homo_velocity[i]), 3) if homo_velocity is not None else '',
                'accel_homography_mps2': round(float(homo_accel[i]), 3) if homo_accel is not None else '',
                'is_interpolated': int(interpolated_mask[i]) if i < len(interpolated_mask) else 0,
                'interp_gap_len':  interp_gap_len[i] if i < len(interp_gap_len) else 0,
                'speed_confidence': round(float(speed_confidence[i]), 3) if i < len(speed_confidence) else 1.0,
            })

        absolute_frame_offset += len(cam_rows)

    return all_track_data


def _draw_chart(fig, axes, canvas, d_smooth, v_smooth, a, fps, target_w, target_h,
                global_d_max, global_t_max, font_prop=None):
    """
    繪製 1×3 子圖（距離/時間、速度/距離、加速度/距離）。
    視覺風格與 smart_switch_tracker.py draw_plots() 一致：
      - cla() 後立即補 grid
      - 中文標題與標籤
      - 藍/綠/紅曲線，linewidth=2
      - tight_layout（無 pad）
    global_d_max / global_t_max 在相機迴圈前預算，整個執行期間不變，確保跨機軸固定。
    """
    t          = np.arange(len(d_smooth)) / fps
    t_max      = global_t_max
    d_max      = global_d_max
    total_dist = d_max / 1.1   # 實際設定總距離（去掉 10% headroom）

    # 距離 Y 軸刻度：每 step_d 公尺一格，確保 total_dist 出現
    if total_dist > 200:
        step_d = 50
    elif total_dist > 100:
        step_d = 25
    elif total_dist > 40:
        step_d = 10
    else:
        step_d = 5
    yticks_d = list(np.arange(0, d_max + step_d * 0.01, step_d))
    if not any(abs(v - total_dist) < step_d * 0.2 for v in yticks_d):
        yticks_d.append(total_dist)
        yticks_d.sort()

    # cla() + grid
    for ax in axes:
        ax.cla()
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[0].plot(t, d_smooth, color='b', linewidth=2)
    axes[0].set_title('距離 vs 時間', fontproperties=font_prop)
    axes[0].set_xlabel('時間 (秒)', fontproperties=font_prop)
    axes[0].set_ylabel('距離 (公尺)', fontproperties=font_prop)
    axes[0].set_xlim(0, t_max)
    axes[0].set_ylim(0, d_max)
    axes[0].set_yticks(yticks_d)
    axes[0].set_yticklabels([f'{v:.0f}' for v in yticks_d], fontsize=7)

    axes[1].plot(d_smooth, v_smooth, color='g', linewidth=2)
    axes[1].set_title('速度 vs 距離', fontproperties=font_prop)
    axes[1].set_xlabel('距離 (公尺)', fontproperties=font_prop)
    axes[1].set_ylabel('速度 (公尺/秒)', fontproperties=font_prop)
    axes[1].set_xlim(0, d_max)
    axes[1].set_ylim(0, 15)
    axes[1].set_yticks([0, 3, 6, 9, 12, 15])
    axes[1].set_yticklabels(['0', '3', '6', '9', '12', '15'], fontsize=7)

    axes[2].plot(d_smooth, a, color='r', linewidth=2)
    axes[2].set_title('加速度 vs 距離', fontproperties=font_prop)
    axes[2].set_xlabel('距離 (公尺)', fontproperties=font_prop)
    axes[2].set_ylabel('加速度 (公尺/秒^2)', fontproperties=font_prop)
    axes[2].set_xlim(0, d_max)
    axes[2].set_ylim(0, 25)
    axes[2].set_yticks([0, 5, 10, 15, 20, 25])
    axes[2].set_yticklabels(['0', '5', '10', '15', '20', '25'], fontsize=7)

    fig.tight_layout()
    canvas.draw()

    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    return cv2.resize(bgr, (target_w, target_h))


@dataclass(frozen=True)
class _FrameConfig:
    """process_frame 的單幀參數。前段是整台相機不變的常數，後 4 個（bbox_color /
    prefer_lead_runner / nearest_to_start / locked_target_id）依起跑確認狀態逐幀變動，
    由呼叫端用 dataclasses.replace 覆寫。"""
    device: int
    crop_params: object
    roi_enabled: bool
    roi_zones: object
    crop_x_offset: int
    crop_y_offset: int
    track_roi: object = None
    draw_bbox: bool = True
    bbox_color: tuple = (0, 255, 0)
    prefer_lead_runner: bool = False
    nearest_to_start: bool = False
    locked_target_id: object = None
    movement_threshold: int = DEFAULT_MOVEMENT_THRESHOLD
    stationary_decay: int = DEFAULT_STATIONARY_DECAY
    ground_point_ema_alpha: float = DEFAULT_GROUND_POINT_EMA_ALPHA
    min_person_height: int = DEFAULT_MIN_PERSON_HEIGHT
    min_movement_frames: int = DEFAULT_MIN_MOVEMENT_FRAMES
    max_person_memory: int = DEFAULT_MAX_PERSON_MEMORY


def _detection_passes_roi(ground_orig, orig_cx, orig_cy, config):
    """單一偵測是否通過 ROI 過濾。proj 為斜線模式的沿跑道投影距離，否則 None。"""
    if config.track_roi is not None:
        track_roi = config.track_roi
        proj = _project_onto_track(
            ground_orig, track_roi['start_mid'], track_roi['track_dir'])
        pre_roll = track_roi.get('pre_roll_px', 0)
        end_roll = track_roi.get('end_roll_px', 0)
        if not (-pre_roll <= proj <= track_roi['pixel_span'] + end_roll):
            return False, proj
        if config.nearest_to_start:
            start_roi = track_roi.get('start_roi_px', pre_roll)
            if abs(proj) > start_roi:
                return False, proj
        return True, proj
    if config.roi_enabled and config.roi_zones:
        inside = any(
            z['x'][0] <= orig_cx <= z['x'][1] and z['y'][0] <= orig_cy <= z['y'][1]
            for z in config.roi_zones
        )
        return inside, None
    return True, None


def _collect_valid_detections(boxes, ids, config):
    """依高度、ROI（斜線投影或矩形）、nearest_to_start 過濾出本幀候選偵測。
    回傳 list of (dist_to_start, cx, cy, bx1, by1, bx2, by2, tid, proj, ground_pt)。"""
    if boxes is None or ids is None or len(boxes) == 0:
        return []

    valid = []
    for i in range(len(boxes)):
        bx1, by1, bx2, by2 = map(int, boxes[i])
        if (by2 - by1) < config.min_person_height:
            continue
        center_x = (bx1 + bx2) / 2
        center_y = (by1 + by2) / 2
        ground_pt = _bbox_bottom_center(bx1, by1, bx2, by2)
        ground_orig = (ground_pt[0] + config.crop_x_offset,
                       ground_pt[1] + config.crop_y_offset)
        passes, proj = _detection_passes_roi(
            ground_orig,
            center_x + config.crop_x_offset,
            center_y + config.crop_y_offset,
            config,
        )
        if not passes:
            continue
        tid = int(ids[i])
        dist_to_start = abs(proj) if config.track_roi is not None else float('inf')
        valid.append((dist_to_start, center_x, center_y,
                      bx1, by1, bx2, by2, tid, proj, ground_pt))

    # nearest_to_start 模式：只保留距 start_line 最近的一人
    if config.nearest_to_start and valid:
        valid.sort(key=lambda x: x[0])
        valid = valid[:1]
    return valid


def _update_velocity_tracker(velocity_tracker, valid_detections, config):
    """依本幀候選偵測更新/建立 velocity_tracker 條目（in-place）。回傳 (seen_ids, lead_candidates)。"""
    seen_ids = set()
    lead_candidates = []
    for (_, center_x, center_y, bx1, by1, bx2, by2, tid, proj, ground_pt) in valid_detections:
        seen_ids.add(tid)
        if tid in velocity_tracker:
            d = velocity_tracker[tid]
            ox, oy = d['center']
            dist = np.sqrt((center_x - ox) ** 2 + (center_y - oy) ** 2)
            d['velocities'].append(dist)
            if dist > config.movement_threshold:
                d['movement_count'] += 1
                d['stationary_count'] = 0
            else:
                d['movement_count'] = max(0, d['movement_count'] - 1)
                d['stationary_count'] += config.stationary_decay
            d['center'] = (center_x, center_y)
            d['bbox']   = (bx1, by1, bx2, by2)
            d['ground_point'] = ground_pt
            prev_ground = d.get('smoothed_ground_point', ground_pt)
            alpha = config.ground_point_ema_alpha
            d['smoothed_ground_point'] = (
                alpha * ground_pt[0] + (1.0 - alpha) * prev_ground[0],
                alpha * ground_pt[1] + (1.0 - alpha) * prev_ground[1],
            )
            d['frames_since_detected'] = 0
        else:
            velocity_tracker[tid] = {
                'center': (center_x, center_y),
                'bbox':   (bx1, by1, bx2, by2),
                'ground_point': ground_pt,
                'smoothed_ground_point': ground_pt,
                'velocities': [0],
                'movement_count': 1,
                'stationary_count': 0,
                'frames_since_detected': 0,
            }
        if config.track_roi is not None and proj is not None:
            lead_candidates.append({'tid': tid, 'proj': proj})
    return seen_ids, lead_candidates


def _prune_stale_tracks(velocity_tracker, seen_ids, max_person_memory):
    """清除超過 max_person_memory 幀未偵測到的追蹤條目（in-place）。"""
    for tid in list(velocity_tracker):
        if tid not in seen_ids:
            velocity_tracker[tid]['frames_since_detected'] += 1
            if velocity_tracker[tid]['frames_since_detected'] > max_person_memory:
                del velocity_tracker[tid]


def _select_fastest_runner(velocity_tracker, config, lead_candidates):
    """選出本幀最快人物 ID；找不到（含鎖定目標本幀未偵測到）時回傳 None。"""
    if config.locked_target_id is not None:
        locked = int(config.locked_target_id)
        d = velocity_tracker.get(locked)
        return locked if (d is not None and d['frames_since_detected'] == 0) else None

    fastest_id = None
    max_vel = 0
    for tid, d in velocity_tracker.items():
        if (d['frames_since_detected'] == 0
                and d['movement_count'] >= config.min_movement_frames
                and d['stationary_count'] < 10):
            v = np.mean(d['velocities']) if d['velocities'] else 0
            if v > max_vel:
                max_vel = v
                fastest_id = tid

    if fastest_id is None and config.prefer_lead_runner and lead_candidates:
        fastest_id = max(lead_candidates, key=lambda c: c['proj'])['tid']
    return fastest_id


def _draw_tracked_overlays(img, velocity_tracker, fastest_id, config):
    """在裁剪後畫面上疊：最快跑者框 + 地面點、其他追蹤人物橘框、ROI 藍框。回傳 img。"""
    if fastest_id is not None:
        d = velocity_tracker[fastest_id]
        bx1, by1, bx2, by2 = d['bbox']
        if config.draw_bbox:
            cv2.rectangle(img, (bx1, by1), (bx2, by2), config.bbox_color, 2)
        ground_pt = d.get('ground_point')
        if ground_pt is not None:
            cv2.circle(img, (int(ground_pt[0]), int(ground_pt[1])), 4, (255, 255, 0), -1)

    # 其他被追蹤人物（橘色，含 ID）
    for tid, d in velocity_tracker.items():
        if tid == fastest_id or d['frames_since_detected'] != 0:
            continue
        bx1o, by1o, bx2o, by2o = d['bbox']
        cv2.rectangle(img, (bx1o, by1o), (bx2o, by2o), (0, 165, 255), 1)
        img = _draw_text_bgr(img, f"ID {tid}", (bx1o, max(by1o - 22, 5)),
                             font=_get_font(size=16), color=(0, 165, 255), thickness=1)

    # ROI 框（藍線）
    if config.roi_enabled and config.roi_zones:
        h_img, w_img = img.shape[:2]
        for i, z in enumerate(config.roi_zones):
            rx1 = int(np.clip(z['x'][0] - config.crop_x_offset, 0, w_img - 1))
            ry1 = int(np.clip(z['y'][0] - config.crop_y_offset, 0, h_img - 1))
            rx2 = int(np.clip(z['x'][1] - config.crop_x_offset, 0, w_img - 1))
            ry2 = int(np.clip(z['y'][1] - config.crop_y_offset, 0, h_img - 1))
            if rx1 < rx2 and ry1 < ry2:
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 100, 0), 2)
                img = _draw_text_bgr(img, f"ROI{i+1} X:{z['x']}", (rx1, max(ry1 - 22, 5)),
                                     font=_get_font(size=18), color=(255, 100, 0), thickness=1)
    return img


def process_frame(img, model, velocity_tracker, config):
    """對單幀執行：裁剪 → YOLO track → ROI 過濾 + 速度累積 → 選最快人物 → 疊框。
    config 為 _FrameConfig。回傳 (img, fastest_id, fastest_center_orig, fastest_bx2_orig)；
    fastest_center_orig / fastest_bx2_orig 是切換判斷用的原始座標基準，找不到人時為 None。
    """
    if config.crop_params:
        cx1, cy1, cx2, cy2 = config.crop_params
        h, w = img.shape[:2]
        cx1, cx2 = max(0, cx1), min(w, cx2)
        cy1, cy2 = max(0, cy1), min(h, cy2)
        if cx2 <= cx1 or cy2 <= cy1:
            return img, None, None, None
        img = img[cy1:cy2, cx1:cx2]

    results = model.track(img, persist=True, classes=[0], show=False, device=config.device,
                          conf=0.3, iou=0.1, imgsz=1280, verbose=False)
    r = results[0]
    boxes = ids = None
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None

    valid_detections = _collect_valid_detections(boxes, ids, config)
    seen_ids, lead_candidates = _update_velocity_tracker(
        velocity_tracker, valid_detections, config)
    _prune_stale_tracks(velocity_tracker, seen_ids, config.max_person_memory)

    fastest_id = _select_fastest_runner(velocity_tracker, config, lead_candidates)

    fastest_center_orig = fastest_bx2_orig = None
    if fastest_id is not None:
        bx1, _, bx2, _ = velocity_tracker[fastest_id]['bbox']
        fastest_center_orig = (bx1 + bx2) / 2.0 + config.crop_x_offset  # 非最後一機切換基準
        fastest_bx2_orig    = bx2 + config.crop_x_offset                # 最後一機退出 ROI 基準

    img = _draw_tracked_overlays(img, velocity_tracker, fastest_id, config)
    return img, fastest_id, fastest_center_orig, fastest_bx2_orig


def parse_args():
    parser = argparse.ArgumentParser(
        description="多相機跑者追蹤（支援 --config-json）"
    )
    parser.add_argument('--config-json', dest='config_json', type=str, default=None,
                        help="相機與執行參數 JSON 字串")
    return parser.parse_args()


def run_tracker(config_dict=None):
    return main(config_dict)


# cfg-key -> (config-key, converter). ``gpu`` is handled separately (it feeds
# two config keys). output_dir/output_name pass through unconverted, matching
# the original inline overrides.
_MAIN_CONFIG_FIELDS = {
    'output_dir':             ('output_dir',             lambda v: v),
    'output_name':            ('output_name',            lambda v: v),
    'target_height':          ('target_height',          int),
    'chart_height':           ('chart_height',           int),
    'movement_threshold':     ('movement_threshold',     int),
    'min_movement_frames':    ('min_movement_frames',    int),
    'stationary_decay':       ('stationary_decay',       int),
    'max_person_memory':      ('max_person_memory',      int),
    'cam_warmup_frames':      ('cam_warmup_frames',      int),
    'min_person_height':      ('min_person_height',      int),
    'ground_point_ema_alpha': ('ground_point_ema_alpha', float),
    'flat_interp_eps_m':      ('flat_interp_eps_m',      float),
}


def _build_config(cfg):
    """預設設定 + cfg dict 覆蓋，回傳 config dict。"""
    config = {
        "gpu": DEFAULT_CUDA_VISIBLE_DEVICES,
        "device": DEFAULT_DEVICE,
        "model_path": DEFAULT_MODEL_PATH,
        "font_path": DEFAULT_FONT_PATH,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "output_name": DEFAULT_OUTPUT_NAME,
        "target_height": DEFAULT_TARGET_HEIGHT,
        "chart_height": DEFAULT_CHART_HEIGHT,
        "movement_threshold": DEFAULT_MOVEMENT_THRESHOLD,
        "min_movement_frames": DEFAULT_MIN_MOVEMENT_FRAMES,
        "stationary_decay": DEFAULT_STATIONARY_DECAY,
        "max_person_memory": DEFAULT_MAX_PERSON_MEMORY,
        "cam_warmup_frames": DEFAULT_CAM_WARMUP_FRAMES,
        "min_person_height": DEFAULT_MIN_PERSON_HEIGHT,
        "ground_point_ema_alpha": DEFAULT_GROUND_POINT_EMA_ALPHA,
        "flat_interp_eps_m": DEFAULT_FLAT_INTERP_EPS_M,
    }
    if 'gpu' in cfg:
        config['gpu'] = str(cfg['gpu'])
        config['device'] = int(cfg['gpu'])
    for cfg_key, (config_key, converter) in _MAIN_CONFIG_FIELDS.items():
        if cfg_key in cfg:
            config[config_key] = converter(cfg[cfg_key])
    return config


def _compute_global_axis_limits(cameras):
    """全程固定圖表軸範圍：距離 = Σdistance_m × 1.1，時間 = Σ(frame_count / fps)。
    只在此算一次，跨相機畫布完全不跳動。"""
    total_dist_m = sum(c.get('distance_m') or 0.0 for c in cameras)
    global_d_max = max(total_dist_m * 1.1, 1.0)
    global_t_max = 0.0
    for c in cameras:
        if c['video_path']:
            cap = cv2.VideoCapture(c['video_path'])
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            f = cap.get(cv2.CAP_PROP_FPS) or 60.0
            global_t_max += n / f
            cap.release()
    return global_d_max, max(global_t_max, 1.0)


def _strip_line_points(cam, target_height):
    """起/終點斜線在 strip（縮圖）座標系的端點；無斜線設定時回傳 (None, None)。"""
    if not (cam.get('start_line') and cam.get('end_line') and cam.get('crop_params')):
        return None, None
    cx1, cy1, _cx2, cy2 = cam['crop_params']
    scale = target_height / (cy2 - cy1)

    def to_strip(pt):
        return (int((pt[0] - cx1) * scale), int((pt[1] - cy1) * scale))

    return (
        (to_strip(cam['start_line'][0]), to_strip(cam['start_line'][1])),
        (to_strip(cam['end_line'][0]), to_strip(cam['end_line'][1])),
    )


@dataclass(frozen=True)
class _StripStyle:
    """一台相機整段共用的 strip 疊圖樣式（標籤字型每台相機只載一次）。"""
    cam_label: str
    target_height: int
    label_font: object
    start_pts: object = None
    end_pts: object = None


def _render_strip(img, style, velocity_tracker=None, fastest_id=None):
    """把裁切畫面縮放到 strip 高度，加上相機標籤與起/終點線疊圖。"""
    h_img, w_img = img.shape[:2]
    new_w = int(w_img * style.target_height / h_img)
    strip = cv2.resize(img, (new_w, style.target_height))

    if (fastest_id is not None and velocity_tracker is not None
            and fastest_id in velocity_tracker):
        sc = style.target_height / h_img
        bbox = velocity_tracker[fastest_id]['bbox']
        bx, by = int(bbox[0] * sc), int(bbox[1] * sc)
        cv2.putText(strip, f"ID:{fastest_id}", (bx + 3, by + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    strip = _draw_text_bgr(strip, style.cam_label, (10, 10), font=style.label_font,
                           color=(255, 255, 255), thickness=2)

    start_pts, end_pts = style.start_pts, style.end_pts
    if start_pts and end_pts:
        p0, p3 = start_pts
        p1, p2 = end_pts
        quad = np.array([p0, p1, p2, p3], dtype=np.int32)
        overlay = strip.copy()
        cv2.fillPoly(overlay, [quad], (200, 220, 255))
        cv2.addWeighted(overlay, 0.15, strip, 0.85, 0, strip)
        for a, b in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
            _draw_dashed_line(strip, a, b, (255, 255, 255), thickness=2)
    if start_pts:
        cv2.line(strip, start_pts[0], start_pts[1], (0, 0, 0), 5)
        cv2.line(strip, start_pts[0], start_pts[1], (180, 255, 255), 3)
    if end_pts:
        cv2.line(strip, end_pts[0], end_pts[1], (0, 0, 0), 5)
        cv2.line(strip, end_pts[0], end_pts[1], (255, 200, 100), 3)
    return strip


def _warmup_fastest_id(velocity_tracker, cam, last_kf_v, fps):
    """切換相機後前幾幀 process_frame 沒選出主跑者時的退路：用瞬時速度挑最快人物，
    略過 movement_count 限制。無距離校準時回傳 None。"""
    m_px = cam.get('m_per_pixel')
    if m_px is None and cam.get('pixel_span') and cam.get('distance_m'):
        m_px = cam['distance_m'] / cam['pixel_span']
    if m_px is None:
        return None
    expected_px = last_kf_v / m_px / fps
    warmup_thresh = max(expected_px * 0.3, 3.0)
    best_v, best_id = 0.0, None
    for tid, d in velocity_tracker.items():
        if d['frames_since_detected'] == 0 and d['velocities']:
            inst_v = d['velocities'][-1]
            if inst_v > warmup_thresh and inst_v > best_v:
                best_v, best_id = inst_v, tid
    return best_id


class _FrameOutcome(Enum):
    NEXT = "next"
    STOP = "stop"


@dataclass
class _CameraPass:
    """第一段（YOLO 追蹤）對一台相機收集的所有結果。前段是逐幀 append 的緩衝，
    後半（interpolated_bbox_mask 之後）由 _finalize_pass 填。"""
    fps: float
    frame_buffer: list = field(default_factory=list)
    d_raw: list = field(default_factory=list)
    meta_buffer: list = field(default_factory=list)
    source_frame_buffer: list = field(default_factory=list)
    metric_debug_buffer: list = field(default_factory=list)
    cam_skipped: int = 0
    interpolated_bbox_mask: list = field(default_factory=list)
    has_any_metric: bool = False
    d_raw_for_csv: list = field(default_factory=list)
    interp_gap_len: list = field(default_factory=list)
    speed_confidence: list = field(default_factory=list)


@dataclass(frozen=True)
class _CameraContext:
    """一台相機整段第一段迴圈需要的常數（含前一台相機傳下來的跨機狀態）。"""
    cam: dict
    cam_idx: int
    is_last_cam: bool
    total_frames: int
    fps: float
    crop_offset: tuple           # (x, y)
    strip_style: _StripStyle
    config: dict
    cumulative_dist_offset: float
    last_kf_v: float


@dataclass
class _StartConfirmation:
    """track_roi 模式的起跑確認狀態。"""
    crossed: bool
    target_id: object = None
    candidates: list = field(default_factory=list)


_START_CONFIRM_FRAMES = 3   # 連續幾幀投影單調遞增才確認起跑


def _raw_distance(cam, track_entry, crop_offset, cumulative_dist_offset, target_height, img_h):
    """有校準時算這一幀主跑者的原始距離。
    回傳 (dist_raw, bbox_strip, homography_local_dist, metric_debug)；無校準時皆為 None。"""
    if cam['m_per_pixel'] is None and cam.get('H_matrix') is None:
        return None, None, None, None

    crop_x_offset, crop_y_offset = crop_offset
    bx1, by1, bx2, by2 = track_entry['bbox']
    cx_orig = (bx1 + bx2) / 2.0 + crop_x_offset
    cy_orig = (by1 + by2) / 2.0 + crop_y_offset
    homography_local_dist = None
    metric_debug = None

    if cam.get('H_matrix') is not None:
        ground_pt = track_entry.get('smoothed_ground_point') or track_entry.get('ground_point')
        if ground_pt is not None:
            image_point = (float(ground_pt[0]) + crop_x_offset,
                           float(ground_pt[1]) + crop_y_offset)
            if cam.get('start_mid') is not None and cam.get('track_dir') is not None:
                image_point = _project_point_to_track_line(
                    image_point, cam['start_mid'], cam['track_dir'])
            world_input = (float(image_point[0]), float(image_point[1]))
        else:
            world_input = (cx_orig, by2 + crop_y_offset)
        world = _transform_point_homography(world_input, cam['H_matrix'])
        world_x = float(world[0]) if world is not None else 0.0
        metric_debug = {
            'image_point_x': float(world_input[0]),
            'image_point_y': float(world_input[1]),
            'world_x': world_x,
        }
        start_world_x = cam.get('homography_start_x')
        if start_world_x is None:
            start_world_x = 0.0
        local_dist = world_x - start_world_x
        homography_local_dist = max(0.0, local_dist)
        if cam.get('distance_m') is not None:
            local_dist = min(local_dist, float(cam['distance_m']))
        dist_raw = cumulative_dist_offset + max(0.0, local_dist)
    elif cam.get('track_dir') and cam.get('start_mid'):
        # 斜線模式：將跑者中心點投影到跑道方向
        proj_px = _project_onto_track((cx_orig, cy_orig), cam['start_mid'], cam['track_dir'])
        dist_raw = cumulative_dist_offset + max(0.0, proj_px * cam['m_per_pixel'])
    else:
        # 舊模式：x 位移
        dist_raw = cumulative_dist_offset + max(0.0, (cx_orig - cam['start_x']) * cam['m_per_pixel'])

    scale = target_height / img_h
    bbox_strip = (int(bx1 * scale), int(by1 * scale), int(bx2 * scale), int(by2 * scale))
    return dist_raw, bbox_strip, homography_local_dist, metric_debug


def _switch_triggered(ctx, velocity_tracker, fastest_id, fastest_center_orig,
                      fastest_bx2_orig, homography_local_dist):
    """本幀是否觸發切換到下一台相機 / 退出最後一台的 ROI。"""
    cam = ctx.cam
    is_last_cam = ctx.is_last_cam
    crop_x_offset, crop_y_offset = ctx.crop_offset
    label = '退出ROI' if is_last_cam else '切換'

    if (cam.get('H_matrix') is not None and cam.get('distance_m') is not None
            and homography_local_dist is not None):
        # Homography 模式：距離計算與切換使用同一個世界座標基準。
        if homography_local_dist >= float(cam['distance_m']):
            print(f"  → 觸發{label}："
                  f"Homography距離={homography_local_dist:.2f}m >= {float(cam['distance_m']):.2f}m")
            return True
        return False

    if cam.get('track_dir') and cam.get('pixel_span'):
        # 斜線模式：投影距離 >= pixel_span 即越過終點線
        entry = velocity_tracker[fastest_id]
        bx1, by1, bx2, by2 = entry['bbox']
        cy_orig = (by1 + by2) / 2.0 + crop_y_offset
        ref_x = bx2 + crop_x_offset if is_last_cam else (bx1 + bx2) / 2.0 + crop_x_offset
        proj_px = _project_onto_track((ref_x, cy_orig), cam['start_mid'], cam['track_dir'])
        if proj_px >= cam['pixel_span']:
            print(f"  → 觸發{label}：投影={proj_px:.0f}px >= {cam['pixel_span']:.0f}px")
            return True
        return False

    # 舊模式：最後一機用 bx2（右緣退出 ROI），其餘機用 center_x
    switch_x = cam.get('switch_x')
    trigger_x = fastest_bx2_orig if is_last_cam else fastest_center_orig
    if switch_x is not None and trigger_x is not None and trigger_x > switch_x:
        ref_name = 'bx2' if is_last_cam else 'center_x'
        print(f"  → 觸發{label}：{ref_name}={trigger_x:.0f} > {switch_x}")
        return True
    return False


def _finalize_pass(run):
    """端點裁切（去掉前後沒有有效 bbox 的幀）+ 缺漏插值 + 插值中繼資料。原地修改 run。"""
    valid = [i for i, v in enumerate(run.d_raw) if v is not None]
    if valid and (valid[0] > 0 or valid[-1] < len(run.d_raw) - 1):
        lo, hi = valid[0], valid[-1]
        trimmed = lo + (len(run.d_raw) - 1 - hi)
        keep = slice(lo, hi + 1)
        run.frame_buffer = run.frame_buffer[keep]
        run.d_raw = run.d_raw[keep]
        run.meta_buffer = run.meta_buffer[keep]
        run.source_frame_buffer = run.source_frame_buffer[keep]
        run.metric_debug_buffer = run.metric_debug_buffer[keep]
        print(f"  補值: 裁掉 {trimmed} 幀沒有前後有效 bbox 的端點缺失")

    run.interpolated_bbox_mask = [False] * len(run.meta_buffer)
    missing = sum(v is None for v in run.d_raw)
    run.has_any_metric = any(v is not None for v in run.d_raw)
    run.d_raw_for_csv = list(run.d_raw)
    if missing and run.has_any_metric:
        run.interpolated_bbox_mask = [m is None for m in run.meta_buffer]
        run.d_raw = _interpolate_missing_numeric(run.d_raw)
        run.meta_buffer = _interpolate_missing_bboxes(run.meta_buffer)
        print(f"  補值: 已用前後有效偵測插值補 {missing} 幀距離/bbox")
    run.interp_gap_len, run.speed_confidence = _interpolation_metadata(run.interpolated_bbox_mask)


def _run_yolo_pass(cap, model, ctx):
    """第一段：逐幀 YOLO 追蹤，收集畫面 + 原始距離；回傳整理好的 _CameraPass。"""
    cam, config = ctx.cam, ctx.config
    crop_x_offset, crop_y_offset = ctx.crop_offset

    run = _CameraPass(fps=ctx.fps)
    velocity_tracker = {}
    if getattr(model, 'predictor', None) is not None and hasattr(model.predictor, 'trackers'):
        for t in model.predictor.trackers:
            t.reset()
    warmup_remaining = config['cam_warmup_frames']
    pre_roll = _StartConfirmation(crossed=cam.get('track_roi') is None)
    frame_cfg_base = _FrameConfig(
        device=config['device'],
        crop_params=cam['crop_params'],
        roi_enabled=cam['roi_enabled'],
        roi_zones=cam['roi_zones'],
        crop_x_offset=crop_x_offset,
        crop_y_offset=crop_y_offset,
        track_roi=cam.get('track_roi'),
        movement_threshold=config['movement_threshold'],
        stationary_decay=config['stationary_decay'],
        ground_point_ema_alpha=config['ground_point_ema_alpha'],
        min_person_height=config['min_person_height'],
        min_movement_frames=config['min_movement_frames'],
        max_person_memory=config['max_person_memory'],
    )

    def _confirm_start(fastest_id, strip, dist_val, proj_px, bbox_strip, frame_count, metric_debug):
        """越過起跑線後連續 K 幀投影單調遞增才鎖定主跑者、放行整批候選幀。"""
        if proj_px < 0:
            pre_roll.candidates.clear()
            return
        if pre_roll.candidates and proj_px <= pre_roll.candidates[-1][2]:
            pre_roll.candidates.clear()
        pre_roll.candidates.append(
            (strip, dist_val, proj_px, bbox_strip, frame_count, metric_debug))
        if len(pre_roll.candidates) < _START_CONFIRM_FRAMES:
            return
        pre_roll.crossed = True
        pre_roll.target_id = fastest_id
        for c_strip, c_dist, _proj, c_meta, c_src, c_debug in pre_roll.candidates:
            run.frame_buffer.append(c_strip)
            run.d_raw.append(c_dist)
            run.meta_buffer.append(c_meta)
            run.source_frame_buffer.append(c_src)
            run.metric_debug_buffer.append(c_debug)
        pre_roll.candidates.clear()
        print(f"  [debug cam{ctx.cam_idx+1}] confirmed start at frame {frame_count}, "
              f"locked_id={pre_roll.target_id}, "
              f"first_dist={run.d_raw[0] if run.d_raw else None}, "
              f"buffered={len(run.frame_buffer)}")

    def _process_one_frame(frame, frame_count):
        nonlocal warmup_remaining
        frame_cfg = replace(
            frame_cfg_base,
            bbox_color=(0, 255, 0) if pre_roll.crossed else (0, 215, 255),
            prefer_lead_runner=not pre_roll.crossed,
            nearest_to_start=not pre_roll.crossed,
            locked_target_id=pre_roll.target_id if pre_roll.crossed else None,
        )
        img, fastest_id, fastest_center_orig, fastest_bx2_orig = process_frame(
            frame, model, velocity_tracker, frame_cfg)

        # 診斷：前 5 幀 + 每 100 幀
        if frame_count <= 5 or frame_count % 100 == 0:
            print(f"  [幀 {frame_count}/{ctx.total_frames}] 最快ID:{fastest_id} "
                  f"追蹤中:{len(velocity_tracker)}人 "
                  f"center:{fastest_center_orig} bx2:{fastest_bx2_orig}")

        if fastest_id is None:
            if warmup_remaining > 0 and pre_roll.target_id is None:
                fastest_id = _warmup_fastest_id(velocity_tracker, cam, ctx.last_kf_v, ctx.fps)
            if fastest_id is None:
                if pre_roll.crossed and run.frame_buffer:
                    run.frame_buffer.append(_render_strip(img, ctx.strip_style))
                    run.d_raw.append(None)
                    run.meta_buffer.append(None)
                    run.source_frame_buffer.append(frame_count)
                    run.metric_debug_buffer.append(None)
                    return _FrameOutcome.NEXT
                run.cam_skipped += 1
                return _FrameOutcome.NEXT
        if warmup_remaining > 0:
            warmup_remaining -= 1

        dist_raw, bbox_strip, homography_local_dist, metric_debug = _raw_distance(
            cam, velocity_tracker[fastest_id], ctx.crop_offset,
            ctx.cumulative_dist_offset, config['target_height'], img.shape[0])

        # 縮放並加相機標籤（不加速度文字，留給第二段）
        strip = _render_strip(img, ctx.strip_style, velocity_tracker, fastest_id)

        # pre-roll / 起跑確認（track_roi 模式才有效）
        if cam.get('track_roi') is not None and not pre_roll.crossed:
            entry = velocity_tracker[fastest_id]
            bx1_c, by1_c, bx2_c, by2_c = entry['bbox']
            cx_orig = (bx1_c + bx2_c) / 2.0 + crop_x_offset
            cy_orig = (by1_c + by2_c) / 2.0 + crop_y_offset
            proj_px = _project_onto_track((cx_orig, cy_orig), cam['start_mid'], cam['track_dir'])
            dist_val = dist_raw if dist_raw is not None else (
                run.d_raw[-1] if run.d_raw else ctx.cumulative_dist_offset)
            if frame_count <= 30:
                print(
                    f"  [debug cam{ctx.cam_idx+1} f{frame_count}] "
                    f"fastest_id={fastest_id} proj_px={proj_px:.1f} "
                    f"crossed={pre_roll.crossed} dist_raw={dist_raw} "
                    f"cand={len(pre_roll.candidates)}"
                )
            _confirm_start(fastest_id, strip, dist_val, proj_px, bbox_strip,
                           frame_count, metric_debug)
            return _FrameOutcome.NEXT

        run.frame_buffer.append(strip)
        run.d_raw.append(dist_raw if dist_raw is not None else (
            run.d_raw[-1] if run.d_raw else ctx.cumulative_dist_offset))
        run.meta_buffer.append(bbox_strip)
        run.source_frame_buffer.append(frame_count)
        run.metric_debug_buffer.append(metric_debug)

        if _switch_triggered(ctx, velocity_tracker, fastest_id, fastest_center_orig,
                             fastest_bx2_orig, homography_local_dist):
            return _FrameOutcome.STOP
        return _FrameOutcome.NEXT

    print("  [第一段] YOLO 追蹤中...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if _process_one_frame(frame, frame_count) is _FrameOutcome.STOP:
            break
    cap.release()
    print(f"  [第一段完成] 收集 {len(run.frame_buffer)} 幀，捨棄 {run.cam_skipped} 幀")
    _finalize_pass(run)
    return run


@dataclass
class _CrossCameraState:
    """跨相機串接時逐台更新的狀態。"""
    cumulative_dist_offset: float = 0.0   # 前機最終距離 = 下一機距離起點（公尺）
    last_kf_v: float = 0.0                # 前機最終 Kalman 速度/加速度，供下一機初始化
    last_kf_a: float = 0.0
    accumulated_d: list = field(default_factory=list)   # 前機已完成的距離/速度/加速度序列
    accumulated_v: list = field(default_factory=list)   # （供圖表跨機連續顯示）
    accumulated_a: list = field(default_factory=list)
    absolute_frame_offset: int = 0        # 本機第 0 幀在全程的絕對幀號
    total_written: int = 0
    total_skipped: int = 0


@dataclass
class _RenderTargets:
    """所有相機共用、延遲初始化的輸出資源。"""
    out: object = None                    # VideoWriter（第一幀才知尺寸）
    chart_fig: object = None
    chart_axes: object = None
    chart_canvas: object = None


@dataclass(frozen=True)
class _RenderConfig:
    """第二段渲染的整場常數。"""
    skip_video: bool
    output_path: str
    chart_font_prop: object
    axis_limits: tuple                    # (global_d_max, global_t_max)


def _build_track_row(run, i, cam_idx, absolute_frame_offset, series):
    """組一列 metrics CSV 資料（第 i 個已寫入幀）。"""
    d_smooth, v_smooth, a_arr = series
    dist_i, speed_i, accel_i = float(d_smooth[i]), float(v_smooth[i]), float(a_arr[i])
    debug_i = run.metric_debug_buffer[i] if i < len(run.metric_debug_buffer) else None
    dist_raw_i = run.d_raw_for_csv[i] if i < len(run.d_raw_for_csv) else None
    world_x_i = debug_i.get('world_x') if debug_i else None
    image_point_x_i = debug_i.get('image_point_x') if debug_i else None
    image_point_y_i = debug_i.get('image_point_y') if debug_i else None
    return {
        'cam':            cam_idx + 1,
        'cam_frame':      i,
        'source_frame':   run.source_frame_buffer[i] if i < len(run.source_frame_buffer) else '',
        'absolute_frame': absolute_frame_offset + i,
        'dist_m':         round(dist_i, 3),
        'dist_raw_m':     round(float(dist_raw_i), 3) if dist_raw_i is not None else '',
        'dist_smooth_m':  round(dist_i, 3),
        'world_x':        round(float(world_x_i), 3) if world_x_i is not None else '',
        'image_point_x':  round(float(image_point_x_i), 3) if image_point_x_i is not None else '',
        'image_point_y':  round(float(image_point_y_i), 3) if image_point_y_i is not None else '',
        'speed_mps':      round(speed_i, 3),
        'accel_mps2':     round(accel_i, 3),
        'is_interpolated': int(run.interpolated_bbox_mask[i]) if i < len(run.interpolated_bbox_mask) else 0,
        'interp_gap_len':  run.interp_gap_len[i] if i < len(run.interp_gap_len) else 0,
        'speed_confidence': round(run.speed_confidence[i], 3) if i < len(run.speed_confidence) else 1.0,
    }


def _compose_metric_frame(strip, run, i, series, prev_series, ctx, rcfg, targets):
    """疊速度文字（含 interp 標記）+ 底部速度/距離/加速度圖表，回傳 vstack 後的輸出幀。"""
    d_smooth, v_smooth, a_arr = series
    d_prev, v_prev, a_prev = prev_series
    config = ctx.config
    dist_i, speed_i, accel_i = float(d_smooth[i]), float(v_smooth[i]), float(a_arr[i])

    bx1s, by1s = run.meta_buffer[i][0], run.meta_buffer[i][1]
    if i < len(run.interpolated_bbox_mask) and run.interpolated_bbox_mask[i]:
        bx1i, by1i, bx2i, by2i = run.meta_buffer[i]
        cv2.rectangle(strip, (bx1i, by1i), (bx2i, by2i), (255, 0, 255), 2)
        cv2.putText(strip, "interp", (bx1i + 3, by1i + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
    label = f"{dist_i:.1f}m  {speed_i:.2f}m/s  {accel_i:+.1f}m/s²"
    strip = _draw_text_bgr(strip, label, (bx1s, max(by1s - 32, 5)),
                           font=_get_font(size=22), color=(0, 255, 255), thickness=2)

    # 底部圖表：第一幀時建立畫布（之後跨相機/幀重複使用）
    if targets.chart_fig is None:
        cw = strip.shape[1]
        targets.chart_fig, targets.chart_axes = plt.subplots(
            1, 3, figsize=(cw / 100, config['chart_height'] / 100), dpi=100)
        targets.chart_canvas = FigureCanvas(targets.chart_fig)

    # 前機完整 + 本機到第 i 幀（absolute_frame 概念：跨機連續）
    if len(d_prev) > 0:
        d_cur = np.concatenate([d_prev, d_smooth[:i + 1]])
        v_cur = np.concatenate([v_prev, v_smooth[:i + 1]])
        a_cur = np.concatenate([a_prev, a_arr[:i + 1]])
    else:
        d_cur, v_cur, a_cur = d_smooth[:i + 1], v_smooth[:i + 1], a_arr[:i + 1]

    d_max, t_max = rcfg.axis_limits
    chart = _draw_chart(
        targets.chart_fig, targets.chart_axes, targets.chart_canvas,
        d_cur, v_cur, a_cur,
        ctx.fps, strip.shape[1], config['chart_height'],
        d_max, t_max, font_prop=rcfg.chart_font_prop)
    return np.vstack([strip, chart])


def _render_pass(run, ctx, rcfg, state, targets, all_track_data):
    """第二段：批次 Butterworth+Kalman、疊速度文字與圖表、寫影片、收 CSV 列。
    原地更新 state（跨機累積）與 targets（共用 writer / matplotlib 畫布）。"""
    cam, cam_idx, fps, config = ctx.cam, ctx.cam_idx, ctx.fps, ctx.config
    has_metrics = (
        (cam['m_per_pixel'] is not None or cam.get('H_matrix') is not None) and
        config['chart_height'] > 0 and len(run.d_raw) > 0 and run.has_any_metric
    )
    series = None
    prev_series = (np.array([]), np.array([]), np.array([]))
    if has_metrics:
        print(f"  [第二段] 計算 Butterworth + Kalman"
              f"（init_v={state.last_kf_v:.2f}, init_a={state.last_kf_a:.2f}）...")
        series = _compute_kf_series(
            run.d_raw, fps, init_v=state.last_kf_v, init_a=state.last_kf_a,
            measurement_confidence=run.speed_confidence,
            flat_interp_eps_m=config['flat_interp_eps_m'])
        d_smooth, v_smooth, a_arr = series
        # 更新跨機累計偏移 + 最終 Kalman 狀態（下一機起點）
        if len(d_smooth) > 0:
            state.cumulative_dist_offset = float(d_smooth[-1])
        state.last_kf_v = float(v_smooth[-1])
        state.last_kf_a = float(a_arr[-1])
        print(f"  本機最終距離: {state.cumulative_dist_offset:.2f}m，"
              f"速度: {state.last_kf_v:.2f}m/s（下一機起點）")
        prev_series = (
            np.array(state.accumulated_d) if state.accumulated_d else np.array([]),
            np.array(state.accumulated_v) if state.accumulated_v else np.array([]),
            np.array(state.accumulated_a) if state.accumulated_a else np.array([]),
        )

    print(f"  [第二段] {'CSV 計算中（skip_video）' if rcfg.skip_video else '渲染輸出中'}...")
    cam_written = 0
    for i, strip in enumerate(run.frame_buffer):
        frame_out = strip
        if has_metrics and run.meta_buffer[i] is not None:
            all_track_data.append(
                _build_track_row(run, i, cam_idx, state.absolute_frame_offset, series))
            if not rcfg.skip_video:
                frame_out = _compose_metric_frame(
                    strip, run, i, series, prev_series, ctx, rcfg, targets)
        elif has_metrics and run.meta_buffer[i] is None and not rcfg.skip_video:
            # pre-roll 幀：保留畫面但圖表區填黑
            empty = np.zeros((config['chart_height'], strip.shape[1], 3), dtype=np.uint8)
            frame_out = np.vstack([strip, empty])

        if not rcfg.skip_video:
            if targets.out is None:
                h_out, w_out = frame_out.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
                targets.out = cv2.VideoWriter(rcfg.output_path, fourcc, fps, (w_out, h_out))
                print(f"  VideoWriter 初始化：{w_out}x{h_out}")
            targets.out.write(frame_out)
            cam_written += 1
            state.total_written += 1

    if has_metrics:
        state.accumulated_d.extend(series[0].tolist())
        state.accumulated_v.extend(series[1].tolist())
        state.accumulated_a.extend(series[2].tolist())
    state.absolute_frame_offset += len(run.frame_buffer)
    print(f"  相機 {cam_idx+1} 完成：寫入 {cam_written} 幀")


def main(config_dict=None):
    args = parse_args()

    # 如果是從 CLI 傳入 --config-json，或是從函式傳入 config_dict，進行覆蓋
    cfg = {}
    if config_dict is not None:
        cfg = config_dict
    elif args.config_json:
        try:
            cfg = json.loads(args.config_json)
        except json.JSONDecodeError as e:
            print(f"\n錯誤：--config-json 格式錯誤：{e}")
            sys.exit(1)

    config = _build_config(cfg)
    skip_video = bool(cfg.get('skip_video', False))

    # 設定 GPU 環境變數
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']

    cameras_override = None
    if 'cameras' in cfg:
        cams = [_build_camera_from_json(e) for e in (cfg['cameras'] or [])[:6]]
        cameras_override = [c for c in cams if c['video_path'] is not None]

    chart_font_prop = _configure_matplotlib_font(config['font_path'])

    # 過濾 video_path=None 的槽位，組出有效相機清單
    default_cameras = get_default_cameras()
    CAMERAS = cameras_override if cameras_override is not None else \
              [c for c in default_cameras if c['video_path'] is not None]
    if not CAMERAS:
        raise ValueError("所有相機的 video_path 均為 None，請至少設定一台。")

    # CUDA 環境檢查
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"使用設備: cuda:{config['device']}")

    # 載入模型並預熱
    model = YOLO(config['model_path'])
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), device=config['device'], verbose=False)
    print(f"模型預熱完成，共 {len(CAMERAS)} 台相機（串接模式）\n")

    os.makedirs(config['output_dir'], exist_ok=True)
    output_path = os.path.join(config['output_dir'], config['output_name'])

    all_track_data = []    # 跨所有相機的速度/距離紀錄（CSV 用）
    state = _CrossCameraState()
    targets = _RenderTargets()

    # 預算全程固定圖表軸範圍（只在此處算一次，跨相機畫布完全不跳動）
    global_d_max, global_t_max = _compute_global_axis_limits(CAMERAS)
    print(f"全程固定軸：距離 0~{global_d_max:.1f}m，時間 0~{global_t_max:.1f}s")

    rcfg = _RenderConfig(
        skip_video=skip_video,
        output_path=output_path,
        chart_font_prop=chart_font_prop,
        axis_limits=(global_d_max, global_t_max),
    )
    plt.ioff()  # matplotlib：畫布在第二段第一幀建立、跨相機/幀重複使用

    # -----------------------------------------------------------------------
    # 逐台相機串接處理（兩段式）
    # -----------------------------------------------------------------------
    for cam_idx, cam in enumerate(CAMERAS):
        # 開啟影片並驗證 crop_params
        cap = cv2.VideoCapture(cam['video_path'])
        if not cap.isOpened():
            raise ValueError(f"無法開啟相機 {cam_idx+1}: {cam['video_path']}")

        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"{'─'*60}")
        print(f"相機 {cam_idx+1}/{len(CAMERAS)}: {cam['video_path']}")
        print(f"  解析度: {vid_w}x{vid_h}，幀數: {total}，FPS: {fps:.1f}")

        # crop 驗證
        cp = cam['crop_params']
        if cp:
            cx1c, cy1c = max(0, cp[0]), max(0, cp[1])
            cx2c, cy2c = min(vid_w, cp[2]), min(vid_h, cp[3])
            if cx2c <= cx1c or cy2c <= cy1c:
                raise ValueError(
                    f"相機 {cam_idx+1} crop_params 無效！\n"
                    f"  設定: x=({cp[0]},{cp[2]}) y=({cp[1]},{cp[3]})\n"
                    f"  影片範圍: x=(0,{vid_w}) y=(0,{vid_h})\n"
                    f"  請將座標調整在影片解析度範圍內。"
                )
            print(f"  CROP: ({cx1c},{cy1c}) → ({cx2c},{cy2c})，"
                  f"裁剪後: {cx2c-cx1c}x{cy2c-cy1c}")

        if cam['roi_enabled']:
            for j, z in enumerate(cam['roi_zones']):
                print(f"  ROI 區域 {j+1}: X={z['x']}, Y={z['y']}")

        switch_x    = cam.get('switch_x')
        is_last_cam = (cam_idx == len(CAMERAS) - 1)
        if switch_x:
            ref_label = 'bx2（右緣）' if is_last_cam else 'center_x'
            print(f"  切換條件: 最快人物 {ref_label}（原始座標）> {switch_x}px")
        else:
            print("  切換條件: 跑完整段影片")

        if cam.get('H_matrix') is not None:
            print(f"  距離校準: Homography 啟用, start_world_x={cam.get('homography_start_x'):.3f}, "
                  f"起始累計={state.cumulative_dist_offset:.1f}m")
        elif cam['m_per_pixel']:
            print(f"  距離校準: start_x={cam['start_x']}px, "
                  f"m/px={cam['m_per_pixel']:.5f}, "
                  f"起始累計={state.cumulative_dist_offset:.1f}m")

        # crop offset（bbox 轉回原始座標用）
        crop_x_offset = cp[0] if cp else 0
        crop_y_offset = cp[1] if cp else 0

        # 打包本台相機第一段所需常數（含前一台傳下來的跨機狀態），跑第一段
        strip_start_pts, strip_end_pts = _strip_line_points(cam, config['target_height'])
        ctx = _CameraContext(
            cam=cam,
            cam_idx=cam_idx,
            is_last_cam=is_last_cam,
            total_frames=total,
            fps=fps,
            crop_offset=(crop_x_offset, crop_y_offset),
            strip_style=_StripStyle(
                cam_label=f"相機 {cam_idx+1}",
                target_height=config['target_height'],
                label_font=_get_font(size=28, font_path=config['font_path']),
                start_pts=strip_start_pts,
                end_pts=strip_end_pts,
            ),
            config=config,
            cumulative_dist_offset=state.cumulative_dist_offset,
            last_kf_v=state.last_kf_v,
        )
        run = _run_yolo_pass(cap, model, ctx)
        state.total_skipped += run.cam_skipped
        _render_pass(run, ctx, rcfg, state, targets, all_track_data)

    # -----------------------------------------------------------------------
    # 收尾
    # -----------------------------------------------------------------------
    if targets.out:
        targets.out.release()
    if targets.chart_fig is not None:
        plt.close(targets.chart_fig)

    # CSV 輸出
    if all_track_data:
        csv_path = os.path.join(config['output_dir'], config['output_name'].replace('.mp4', '_metrics.csv'))
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=['cam', 'cam_frame', 'source_frame', 'absolute_frame',
                               'dist_m', 'dist_raw_m', 'dist_smooth_m',
                               'world_x', 'image_point_x', 'image_point_y',
                               'speed_mps', 'accel_mps2',
                               'is_interpolated', 'interp_gap_len',
                               'speed_confidence'])
            writer.writeheader()
            writer.writerows(all_track_data)
        print(f"CSV 輸出：{csv_path}")

    _write_homography_visualizations(CAMERAS, config['output_dir'], track_data=all_track_data)

    print(f"\n{'='*60}")
    print(f"全部完成：總寫入 {state.total_written} 幀，總捨棄 {state.total_skipped} 幀")
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"輸出: {output_path} ({size_mb:.2f} MB)")
    else:
        print("Critical Error: 輸出檔案不存在！")


if __name__ == "__main__":
    main()
