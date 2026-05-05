import cv2
try:
    cv2.setLogLevel(3)   # 抑制 swscaler HDR 色彩轉換警告（不影響運作）
except AttributeError:
    pass  # 舊版 OpenCV 無此 API，忽略
from ultralytics import YOLO
import os
import sys
import csv
import json
import argparse
import numpy as np
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from filterpy.kalman import KalmanFilter
from scipy.signal import butter, filtfilt
from PIL import Image, ImageDraw, ImageFont

# =======================================================================
# 設定區（每次修改只需改這裡）
# =======================================================================

_BASE = Path(__file__).resolve().parent   # repo 根目錄

# 使用哪張實體 GPU（'0' = 第 0 張，'1' = 第 1 張）
CUDA_VISIBLE_DEVICES = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
DEVICE = 0

# 模型權重路徑
MODEL_PATH = str(_BASE / "yolo26x.pt")

# 中文字型路徑（圖表與影片疊字使用）
FONT_PATH = str(_BASE / "MotionAGFormer" / "ChineseFont.ttf")

# 輸出目錄與檔名
OUTPUT_DIR  = str(_BASE / "output_cut")
OUTPUT_NAME = "sequential_tracked.mp4"

# 輸出影片每台相機的顯示高度（統一縮放後輸出）
TARGET_HEIGHT = 340

# 底部圖表高度（像素）；設為 0 = 不顯示圖表（distance_m 也需設定才會顯示）
CHART_HEIGHT = 200


def _get_font(size=28, font_path=FONT_PATH):
    """載入字型；失敗時回傳 None 以便安全降級。"""
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            return None
    return None


def _configure_matplotlib_font(font_path=FONT_PATH):
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


def _draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=12, gap_len=8):
    """在影像上畫虛線。"""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg = dash_len if drawing else gap_len
        end_pos = min(pos + seg, length)
        if drawing:
            x1 = int(pt1[0] + ux * pos)
            y1 = int(pt1[1] + uy * pos)
            x2 = int(pt1[0] + ux * end_pos)
            y2 = int(pt1[1] + uy * end_pos)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        pos = end_pos
        drawing = not drawing


def _point_in_quad(point, quad_pts):
    """判斷 point 是否在四邊形內（含邊界）。quad_pts 為 np.float32 shape=(4,2)。"""
    return cv2.pointPolygonTest(quad_pts, (float(point[0]), float(point[1])), False) >= 0


def _draw_text_bgr(img, text, org, font=None, color=(255, 255, 255), thickness=2,
                   outline_color=(0, 0, 0)):
    """在 BGR 影像上繪製可顯示中文的文字；org 為左上角座標。"""
    if not text:
        return img

    font = font or _get_font(size=28)
    if font is None:
        cv2.putText(img, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)
        return img

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    x, y = int(org[0]), int(org[1])

    if outline_color is not None:
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), str(text), font=font, fill=outline_color[::-1])

    draw.text((x, y), str(text), font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


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


LANE_WIDTH_M = 1.22


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
#
# 最省事的填法：改用 build_camera_with_homography(...)
# 你每台只要填：
#   1. video_path
#   2. start_line / end_line
#   3. start_meter（本台相機起始公尺）
#   4. src_points（固定 5 點；也可用 6 點補上中段近側點）
#
# src_points 固定順序：
#   1. 0m 靠近側
#   2. 0m 遠離側
#   3. 10m 遠離側
#   4. 20m 遠離側
#   5. 20m 靠近側
#
# src_points 若使用 6 點，固定順序：
#   1. 0m 靠近側
#   2. 0m 遠離側
#   3. 10m 遠離側
#   4. 10m 靠近側
#   5. 20m 遠離側
#   6. 20m 靠近側
#
# 若 src_points 尚未填滿，會自動退回既有 distance_m + start_line/end_line 模式，
# 方便你逐台補 Homography，不會影響腳本先跑。
# -----------------------------------------------------------------------
# 建議公尺區間：
#   CAM1: 0m~20m
#   CAM2: 20m~40m
#   CAM3: 40m~60m
#   CAM4: 60m~80m
#   CAM5: 80m~100m
#   CAM6: 100m~120m
# -----------------------------------------------------------------------
CAM1 = build_camera_with_homography(
    video_path="/home/jeter/pipeline_release/video/IMG_3564_1.mp4",
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

CAM2 = build_camera_with_homography(
    video_path= "/home/jeter/pipeline_release/video/IMG_2540_2.mp4",
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

CAM3 = build_camera_with_homography(
    video_path="/home/jeter/pipeline_release/video/0504_3.mp4",
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

CAM4 = build_camera_with_homography(
    video_path="/home/jeter/pipeline_release/video/0504_4.mp4",
    crop=(0, 400, 1920, 800),
    start_line=[(227, 713), (140, 727)],
    end_line=[(1722, 718), (1825, 725)],
    start_meter=60.0,
    src_points=[
        [140, 727],   # 60m 靠近側
        [227, 713],   # 60m 遠離側
        [976, 716],   # 70m 遠離側
        [976, 726],  # 70m 靠近側（由跑道透視與 1.22m 寬度推算）
        [1722, 718],  # 80m 遠離側
        [1825, 725],  # 80m 靠近側
    ],
)

CAM5 = build_camera_with_homography(
    video_path=None,
    crop=(0, 400, 1920, 800),
    start_line=[(150, 400), (150, 800)],
    end_line=[(1820, 400), (1820, 800)],
    start_meter=80.0,
    src_points=[
        # [x, y],  # 80m 靠近側
        # [x, y],  # 80m 遠離側
        # [x, y],  # 90m 遠離側
        # [x, y],  # 90m 靠近側
        # [x, y],  # 100m 遠離側
        # [x, y],  # 100m 靠近側
    ],
)

CAM6 = build_camera_with_homography(
    video_path=None,
    crop=(0, 400, 1920, 800),
    start_line=[(150, 400), (150, 800)],
    end_line=[(1820, 400), (1820, 800)],
    start_meter=100.0,
    src_points=[
        # [x, y],  # 100m 靠近側
        # [x, y],  # 100m 遠離側
        # [x, y],  # 110m 遠離側
        # [x, y],  # 110m 靠近側
        # [x, y],  # 120m 遠離側
        # [x, y],  # 120m 靠近側
    ],
)

# -----------------------------------------------------------------------
# 移動偵測參數
# -----------------------------------------------------------------------
MOVEMENT_THRESHOLD  = 3   # 判定為移動的最小像素位移
MIN_MOVEMENT_FRAMES = 3  # 需連續移動至少此幀數才視為「真正移動」
STATIONARY_DECAY    = 2   # 靜止時每幀遞減 movement_count 的量
MAX_PERSON_MEMORY   = 30  # 超過此幀數未偵測到則清除該人物的速度紀錄
CAM_WARMUP_FRAMES   = 5   # 切換相機後前幾幀放寬選取條件
MIN_PERSON_HEIGHT   = 80  # bbox 高度小於此值（裁切後像素）視為背景遠景人物，略過
GROUND_POINT_EMA_ALPHA = 0.35  # bbox 底部中心點 EMA 平滑係數；越小越穩但延遲越大
FLAT_INTERP_EPS_M = 0.001      # 距離變化小於此值視為 flat segment

# =======================================================================
# 以下為程式邏輯，一般不需修改
# =======================================================================

def _interpolate_flat_segments(d, eps=FLAT_INTERP_EPS_M):
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
    for i in range(0, first):
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
    for i in range(0, first):
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


def _compute_kf_series(d_raw, fps, init_v=0.0, init_a=0.0,
                       measurement_confidence=None):
    """
    移植自 smart_switch_tracker.py _compute_kf_series()。
    輸入：d_raw = list of float（每幀原始距離，公尺），fps = 幀率
          init_v / init_a：跨機傳遞的初始速度/加速度（預設 0，第一機使用）
          measurement_confidence：每幀量測可信度；插值幀可設較低，降低 Kalman 信任
    輸出：(d_smooth, v_smooth, a) 三個 numpy array，長度與 d_raw 相同

    流程（與 smart_switch 完全相同）：
      1. 單調約束（距離只能遞增）
      2. Butterworth 低通 filtfilt（6Hz，雙向，需完整序列）
      3. Kalman 濾波，狀態 [位置, 速度, 加速度]
    """
    n = len(d_raw)
    dt = 1.0 / fps
    d = np.array(d_raw, dtype=float)
    if measurement_confidence is None:
        measurement_confidence = np.ones(n, dtype=float)
    else:
        measurement_confidence = np.asarray(measurement_confidence, dtype=float)
        if len(measurement_confidence) != n:
            measurement_confidence = np.ones(n, dtype=float)
        measurement_confidence = np.clip(measurement_confidence, 0.05, 1.0)

    # 1. 單調約束
    for k in range(1, n):
        if d[k] < d[k - 1]:
            d[k] = d[k - 1]

    # 1b. 修正「距離卡住幾幀後突然跳動」造成的速度/加速度震盪
    d = _interpolate_flat_segments(d)

    # 2. Butterworth filtfilt（需 n >= 15）
    if n >= 15:
        try:
            b_but, a_but = butter(2, 6.0 / (fps / 2.0), btype='low')
            d_smooth = filtfilt(b_but, a_but, d)
            for k in range(1, n):
                if d_smooth[k] < d_smooth[k - 1]:
                    d_smooth[k] = d_smooth[k - 1]
            d_smooth = np.maximum(d_smooth, d[0])  # 防止 filtfilt 邊界效應把起點壓低
        except Exception:
            d_smooth = d.copy()
    else:
        d_smooth = d.copy()

    # 3. Kalman 濾波（需 n >= 5）
    if n >= 5:
        try:
            kf = KalmanFilter(dim_x=3, dim_z=1)
            kf.F = np.array([[1, dt, 0.5 * dt ** 2],
                             [0,  1,            dt],
                             [0,  0,             1]])
            kf.H = np.array([[1, 0, 0]])
            kf.Q = np.diag([0.001, 0.01, 0.5])
            base_r = 0.05
            kf.R = np.array([[base_r]])
            # 跨機傳遞初始速度與加速度，避免切換時從 0 重新爬升
            kf.x = np.array([[d_smooth[0]], [float(init_v)], [float(init_a)]])
            # 若有前機狀態，速度/加速度的初始不確定度可更小
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
            v_smooth = np.maximum(velocities, 0.0)
            a = np.array(accels)
        except Exception:
            v_smooth = np.maximum(np.gradient(d_smooth, dt), 0.0)
            a = np.gradient(v_smooth, dt)
    else:
        v_smooth = np.zeros(n)
        a = np.zeros(n)

    return d_smooth, v_smooth, a


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


def process_frame(img, model, velocity_tracker, device,
                  crop_params, roi_enabled, roi_zones,
                  crop_x_offset, crop_y_offset,
                  quad_roi=None, track_roi=None, draw_bbox=True,
                  bbox_color=(0, 255, 0), prefer_lead_runner=False,
                  nearest_to_start=False, locked_target_id=None):
    """
    對單幀執行裁剪 → YOLO track/pose → 速度累積 → 選目標人物 → 畫框。
    locked_target_id 有值時，只追蹤該 ByteTrack ID，避免切到裁判或其他人。
    回傳：(處理後畫面, fastest_id, fastest_center_orig, fastest_bx2_orig)
      fastest_center_orig：最快人物 center_x 在原始影片座標（非最後一機的切換基準）
      fastest_bx2_orig：最快人物 bbox 右緣 bx2 在原始影片座標（最後一機的退出 ROI 基準）
    """
    # Step 1: 裁剪
    if crop_params:
        cx1, cy1, cx2, cy2 = crop_params
        h, w = img.shape[:2]
        cx1, cx2 = max(0, cx1), min(w, cx2)
        cy1, cy2 = max(0, cy1), min(h, cy2)
        if cx2 <= cx1 or cy2 <= cy1:
            return img, None, None, None
        img = img[cy1:cy2, cx1:cx2]

    # Step 2: YOLO track()
    results = model.track(img, persist=True, classes=[0], show=False, device=device,
                          conf=0.3, iou=0.1, imgsz=1280, verbose=False)
    r = results[0]

    # Step 3: 速度累積 + ROI 過濾
    seen_ids = set()
    lead_candidates = []

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids   = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None

        # 第一輪：收集所有通過過濾的偵測
        valid_detections = []  # (dist_to_start_line, cx, cy, bx1, by1, bx2, by2, tid, proj, ground_pt)
        for i in range(len(boxes)):
            bx1, by1, bx2, by2 = map(int, boxes[i])
            if (by2 - by1) < MIN_PERSON_HEIGHT:
                continue
            center_x = (bx1 + bx2) / 2
            center_y = (by1 + by2) / 2
            ground_pt = _bbox_bottom_center(bx1, by1, bx2, by2)
            ground_orig = (
                ground_pt[0] + crop_x_offset,
                ground_pt[1] + crop_y_offset,
            )

            # ROI 過濾（原始影片座標）
            orig_cx = center_x + crop_x_offset
            orig_cy = center_y + crop_y_offset
            proj = None
            if track_roi is not None:
                proj = _project_onto_track(
                    ground_orig,
                    track_roi['start_mid'], track_roi['track_dir']
                )
                pre_roll = track_roi.get('pre_roll_px', 0)
                end_roll = track_roi.get('end_roll_px', 0)
                if not (-pre_roll <= proj <= track_roi['pixel_span'] + end_roll):
                    continue
                if nearest_to_start:
                    start_roi = track_roi.get('start_roi_px', pre_roll)
                    if abs(proj) > start_roi:
                        continue
            elif roi_enabled and roi_zones:
                if not any(z['x'][0] <= orig_cx <= z['x'][1] and
                           z['y'][0] <= orig_cy <= z['y'][1]
                           for z in roi_zones):
                    continue

            if ids is None:
                continue
            tid = int(ids[i])
            dist_to_start = (
                abs(proj)
                if track_roi is not None else float('inf')
            )
            valid_detections.append((dist_to_start, center_x, center_y,
                                     bx1, by1, bx2, by2, tid, proj, ground_pt))

        # nearest_to_start 模式：只保留距 start_line 最近的一人
        if nearest_to_start and valid_detections:
            valid_detections.sort(key=lambda x: x[0])
            valid_detections = valid_detections[:1]

        # 第二輪：更新 velocity_tracker
        for (_, center_x, center_y, bx1, by1, bx2, by2, tid, proj, ground_pt) in valid_detections:
            seen_ids.add(tid)
            if tid in velocity_tracker:
                d = velocity_tracker[tid]
                ox, oy = d['center']
                dist = np.sqrt((center_x - ox)**2 + (center_y - oy)**2)
                d['velocities'].append(dist)
                if dist > MOVEMENT_THRESHOLD:
                    d['movement_count'] += 1
                    d['stationary_count'] = 0
                else:
                    d['movement_count'] = max(0, d['movement_count'] - 1)
                    d['stationary_count'] += STATIONARY_DECAY
                d['center'] = (center_x, center_y)
                d['bbox']   = (bx1, by1, bx2, by2)
                d['ground_point'] = ground_pt
                prev_ground = d.get('smoothed_ground_point', ground_pt)
                alpha = GROUND_POINT_EMA_ALPHA
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
            if track_roi is not None and proj is not None:
                lead_candidates.append({'tid': tid, 'proj': proj})

    # 清除過期追蹤
    for tid in list(velocity_tracker):
        if tid not in seen_ids:
            velocity_tracker[tid]['frames_since_detected'] += 1
            if velocity_tracker[tid]['frames_since_detected'] > MAX_PERSON_MEMORY:
                del velocity_tracker[tid]

    # Step 4: 選出目標人物
    fastest_id = None
    if locked_target_id is not None:
        locked_target_id = int(locked_target_id)
        d = velocity_tracker.get(locked_target_id)
        if d is not None and d['frames_since_detected'] == 0:
            fastest_id = locked_target_id
    else:
        max_vel = 0
        for tid, d in velocity_tracker.items():
            if d['frames_since_detected'] == 0:
                if (d['movement_count'] >= MIN_MOVEMENT_FRAMES and
                        d['stationary_count'] < 10):
                    v = np.mean(d['velocities']) if d['velocities'] else 0
                    if v > max_vel:
                        max_vel = v
                        fastest_id = tid

        if fastest_id is None and prefer_lead_runner and lead_candidates:
            fastest_id = max(lead_candidates, key=lambda c: c['proj'])['tid']

    # Step 5: 畫框（最快人物）
    fastest_center_orig = None
    fastest_bx2_orig    = None
    if fastest_id is not None:
        d = velocity_tracker[fastest_id]
        bx1, by1, bx2, by2 = d['bbox']
        fastest_center_orig = (bx1 + bx2) / 2.0 + crop_x_offset  # 非最後一機的切換基準
        fastest_bx2_orig    = bx2 + crop_x_offset                 # 最後一機的退出 ROI 基準
        if draw_bbox:
            cv2.rectangle(img, (bx1, by1), (bx2, by2), bbox_color, 2)
        ground_pt = d.get('ground_point')
        if ground_pt is not None:
            gx, gy = int(ground_pt[0]), int(ground_pt[1])
            cv2.circle(img, (gx, gy), 4, (255, 255, 0), -1)

    # Step 5b: 畫其他被追蹤人物的框（橘色，含 ID）
    for tid, d in velocity_tracker.items():
        if tid == fastest_id or d['frames_since_detected'] != 0:
            continue
        bx1o, by1o, bx2o, by2o = d['bbox']
        cv2.rectangle(img, (bx1o, by1o), (bx2o, by2o), (0, 165, 255), 1)
        img = _draw_text_bgr(
            img,
            f"ID {tid}",
            (bx1o, max(by1o - 22, 5)),
            font=_get_font(size=16),
            color=(0, 165, 255),
            thickness=1,
        )

    # Step 6: 畫 ROI 框（藍線）
    if roi_enabled and roi_zones:
        h_img, w_img = img.shape[:2]
        for i, z in enumerate(roi_zones):
            rx1 = int(np.clip(z['x'][0] - crop_x_offset, 0, w_img - 1))
            ry1 = int(np.clip(z['y'][0] - crop_y_offset, 0, h_img - 1))
            rx2 = int(np.clip(z['x'][1] - crop_x_offset, 0, w_img - 1))
            ry2 = int(np.clip(z['y'][1] - crop_y_offset, 0, h_img - 1))
            if rx1 < rx2 and ry1 < ry2:
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (255, 100, 0), 2)
                img = _draw_text_bgr(
                    img,
                    f"ROI{i+1} X:{z['x']}",
                    (rx1, max(ry1 - 22, 5)),
                    font=_get_font(size=18),
                    color=(255, 100, 0),
                    thickness=1,
                )

    return img, fastest_id, fastest_center_orig, fastest_bx2_orig


def parse_args():
    parser = argparse.ArgumentParser(
        description="多相機跑者追蹤（支援 --config-json）"
    )
    parser.add_argument('--config-json', dest='config_json', type=str, default=None,
                        help="相機與執行參數 JSON 字串")
    return parser.parse_args()


def main():
    args = parse_args()

    global CUDA_VISIBLE_DEVICES, DEVICE, OUTPUT_DIR, OUTPUT_NAME
    global TARGET_HEIGHT, CHART_HEIGHT
    global MOVEMENT_THRESHOLD, MIN_MOVEMENT_FRAMES, STATIONARY_DECAY, MAX_PERSON_MEMORY

    cameras_override = None
    if args.config_json:
        try:
            cfg = json.loads(args.config_json)
        except json.JSONDecodeError as e:
            print(f"\n錯誤：--config-json 格式錯誤：{e}")
            sys.exit(1)
        if 'gpu' in cfg:
            CUDA_VISIBLE_DEVICES = str(cfg['gpu'])
            DEVICE = int(cfg['gpu'])
            os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
        if 'output_dir'          in cfg: OUTPUT_DIR          = cfg['output_dir']
        if 'output_name'         in cfg: OUTPUT_NAME         = cfg['output_name']
        if 'target_height'       in cfg: TARGET_HEIGHT       = int(cfg['target_height'])
        if 'chart_height'        in cfg: CHART_HEIGHT        = int(cfg['chart_height'])
        if 'movement_threshold'  in cfg: MOVEMENT_THRESHOLD  = int(cfg['movement_threshold'])
        if 'min_movement_frames' in cfg: MIN_MOVEMENT_FRAMES = int(cfg['min_movement_frames'])
        if 'stationary_decay'    in cfg: STATIONARY_DECAY    = int(cfg['stationary_decay'])
        if 'max_person_memory'   in cfg: MAX_PERSON_MEMORY   = int(cfg['max_person_memory'])
        if 'cameras' in cfg:
            cams = [_build_camera_from_json(e) for e in (cfg['cameras'] or [])[:6]]
            cameras_override = [c for c in cams if c['video_path'] is not None]

    chart_font_prop = _configure_matplotlib_font()

    # 過濾 video_path=None 的槽位，組出有效相機清單
    CAMERAS = cameras_override if cameras_override is not None else \
              [c for c in [CAM1, CAM2, CAM3, CAM4, CAM5, CAM6]
               if c['video_path'] is not None]
    if not CAMERAS:
        raise ValueError("所有相機的 video_path 均為 None，請至少設定一台。")
    # 切換策略：
    #   非最後一機 → 用 center_x，超過 switch_x 時切到下一台
    #   最後一機   → 用 bx2（右緣），超過 switch_x 時視為「退出 ROI」而停止收集
    #               switch_x=None 時最後一機仍跑完整支影片

    # CUDA 環境檢查
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"使用設備: cuda:{DEVICE}")

    # 載入模型並預熱
    model = YOLO(MODEL_PATH)
    model.predict(np.zeros((480, 640, 3), dtype=np.uint8), device=DEVICE, verbose=False)
    print(f"模型預熱完成，共 {len(CAMERAS)} 台相機（串接模式）\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)

    out           = None   # 所有相機共用同一個 VideoWriter
    total_written = 0
    total_skipped = 0
    all_track_data = []    # 跨所有相機的速度/距離紀錄（CSV 用）
    cumulative_dist_offset = 0.0  # 跨相機距離累積偏移（公尺）：前機最終距離

    # 跨相機連續序列（與 smart_switch absolute_frame / cam_frame 概念相同）
    accumulated_d = []   # 前機已完成的距離序列（供圖表連續顯示）
    accumulated_v = []   # 前機已完成的速度序列
    accumulated_a = []   # 前機已完成的加速度序列
    absolute_frame_offset = 0  # 本機第 0 幀在全程的絕對幀號

    # 跨機 Kalman 初始狀態：前機最終速度/加速度傳給下一機，避免切換時速度從 0 爬升
    last_kf_v = 0.0
    last_kf_a = 0.0

    # -----------------------------------------------------------------------
    # 預算全程固定軸範圍（只在此處算一次，跨相機畫布完全不跳動）
    #   global_d_max：所有有 distance_m 的相機距離總和 × 1.1
    #   global_t_max：掃描每支影片 frame_count / fps 加總
    # -----------------------------------------------------------------------
    total_dist_m = sum(c.get('distance_m') or 0.0 for c in CAMERAS)
    global_d_max = max(total_dist_m * 1.1, 1.0)
    global_t_max = 0.0
    for _c in CAMERAS:
        if _c['video_path']:
            _cap = cv2.VideoCapture(_c['video_path'])
            _n   = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            _f   = _cap.get(cv2.CAP_PROP_FPS) or 60.0
            global_t_max += _n / _f
            _cap.release()
    global_t_max = max(global_t_max, 1.0)
    print(f"全程固定軸：距離 0~{global_d_max:.1f}m，時間 0~{global_t_max:.1f}s")

    # matplotlib 畫布（smart_switch 策略：建立一次，跨相機/幀重複使用）
    plt.ioff()
    chart_fig = chart_axes = chart_canvas = None  # 延遲初始化（第一幀才知道寬度）

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
            print(f"  切換條件: 跑完整段影片")

        if cam.get('H_matrix') is not None:
            print(f"  距離校準: Homography 啟用, start_world_x={cam.get('homography_start_x'):.3f}, "
                  f"起始累計={cumulative_dist_offset:.1f}m")
        elif cam['m_per_pixel']:
            print(f"  距離校準: start_x={cam['start_x']}px, "
                  f"m/px={cam['m_per_pixel']:.5f}, "
                  f"起始累計={cumulative_dist_offset:.1f}m")

        # crop offset（bbox 轉回原始座標用）
        crop_x_offset = cp[0] if cp else 0
        crop_y_offset = cp[1] if cp else 0

        # 每台相機重置速度追蹤表 + ByteTrack 內部狀態
        velocity_tracker = {}
        if hasattr(model, 'predictor') and model.predictor is not None:
            if hasattr(model.predictor, 'trackers'):
                for t in model.predictor.trackers:
                    t.reset()
        cam_skipped          = 0
        frame_count          = 0
        cam_warmup_remaining = CAM_WARMUP_FRAMES

        # 第一段緩衝區
        frame_buffer = []   # 縮放後的畫面（含綠框、相機標籤）
        d_raw        = []   # 每幀原始距離（公尺）
        meta_buffer  = []   # 每幀 bbox 在 strip 座標（供第二段疊字）；None = pre-roll 幀
        source_frame_buffer = []  # 對應原始影片 frame_count，供 CSV 對照
        metric_debug_buffer = []  # 每幀原始量測 debug：world_x / image_point

        # pre-roll 狀態（track_roi 模式專用）
        runner_crossed_start = cam.get('track_roi') is None  # 舊模式直接視為已越線
        target_runner_id = None  # 起跑確認後固定追蹤同一個 ByteTrack ID
        candidate_buf = []   # 起跑候選幀 (strip, dist_val, proj_px, bbox_strip, source_frame, debug)
        K_CONFIRM     = 3    # 連續幾幀單調遞增才確認起跑

        # -----------------------------------------------------------------------
        # 預計算起終點線在 strip 座標系的位置
        # -----------------------------------------------------------------------
        strip_start_pts = strip_end_pts = None
        if cam.get('start_line') and cam.get('end_line') and cam.get('crop_params'):
            cp_l = cam['crop_params']
            ls = TARGET_HEIGHT / (cp_l[3] - cp_l[1])
            def _to_strip_pt(pt):
                return (int((pt[0] - cp_l[0]) * ls), int((pt[1] - cp_l[1]) * ls))
            strip_start_pts = (_to_strip_pt(cam['start_line'][0]),
                               _to_strip_pt(cam['start_line'][1]))
            strip_end_pts   = (_to_strip_pt(cam['end_line'][0]),
                               _to_strip_pt(cam['end_line'][1]))

        def _make_strip(img, fastest_id_for_label=None):
            h_img, w_img = img.shape[:2]
            new_w = int(w_img * TARGET_HEIGHT / h_img)
            strip = cv2.resize(img, (new_w, TARGET_HEIGHT))

            if fastest_id_for_label is not None and fastest_id_for_label in velocity_tracker:
                sc = TARGET_HEIGHT / h_img
                d_v = velocity_tracker[fastest_id_for_label]
                _bx = int(d_v['bbox'][0] * sc)
                _by = int(d_v['bbox'][1] * sc)
                cv2.putText(strip, f"ID:{fastest_id_for_label}", (_bx + 3, _by + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            strip = _draw_text_bgr(
                strip,
                f"相機 {cam_idx+1}",
                (10, 10),
                font=_get_font(size=28),
                color=(255, 255, 255),
                thickness=2,
            )

            if strip_start_pts and strip_end_pts:
                p0, p3 = strip_start_pts
                p1, p2 = strip_end_pts
                quad = np.array([p0, p1, p2, p3], dtype=np.int32)
                overlay = strip.copy()
                cv2.fillPoly(overlay, [quad], (200, 220, 255))
                cv2.addWeighted(overlay, 0.15, strip, 0.85, 0, strip)
                for a, b in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
                    _draw_dashed_line(strip, a, b, (255, 255, 255), thickness=2)

            if strip_start_pts:
                cv2.line(strip, strip_start_pts[0], strip_start_pts[1], (0, 0, 0), 5)
                cv2.line(strip, strip_start_pts[0], strip_start_pts[1], (180, 255, 255), 3)
            if strip_end_pts:
                cv2.line(strip, strip_end_pts[0], strip_end_pts[1], (0, 0, 0), 5)
                cv2.line(strip, strip_end_pts[0], strip_end_pts[1], (255, 200, 100), 3)
            return strip

        # -----------------------------------------------------------------------
        # 第一段：YOLO 追蹤，收集幀畫面與原始距離
        # -----------------------------------------------------------------------
        print(f"  [第一段] YOLO 追蹤中...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            img, fastest_id, fastest_center_orig, fastest_bx2_orig = process_frame(
                frame, model, velocity_tracker, DEVICE,
                cam['crop_params'], cam['roi_enabled'], cam['roi_zones'],
                crop_x_offset, crop_y_offset,
                quad_roi=cam.get('quad_roi'),
                track_roi=cam.get('track_roi'),
                draw_bbox=True,
                bbox_color=(0, 255, 0) if runner_crossed_start else (0, 215, 255),
                prefer_lead_runner=not runner_crossed_start,
                nearest_to_start=not runner_crossed_start,
                locked_target_id=target_runner_id if runner_crossed_start else None,
            )

            # 診斷：前 5 幀 + 每 100 幀
            if frame_count <= 5 or frame_count % 100 == 0:
                print(f"  [幀 {frame_count}/{total}] 最快ID:{fastest_id} "
                      f"追蹤中:{len(velocity_tracker)}人 "
                      f"center:{fastest_center_orig} bx2:{fastest_bx2_orig}")

            if fastest_id is None:
                if cam_warmup_remaining > 0 and target_runner_id is None:
                    m_px = cam.get('m_per_pixel')
                    if m_px is None and cam.get('pixel_span') and cam.get('distance_m'):
                        m_px = cam['distance_m'] / cam['pixel_span']
                    if m_px is not None:
                        # 暖機模式：用瞬時速度選最快人物，略過 movement_count 限制
                        expected_px = last_kf_v / m_px / fps
                        warmup_thresh = max(expected_px * 0.3, 3.0)
                        best_v, best_id = 0.0, None
                        for tid, d in velocity_tracker.items():
                            if d['frames_since_detected'] == 0 and d['velocities']:
                                inst_v = d['velocities'][-1]
                                if inst_v > warmup_thresh and inst_v > best_v:
                                    best_v, best_id = inst_v, tid
                        fastest_id = best_id
                if fastest_id is None:
                    if runner_crossed_start and frame_buffer:
                        strip = _make_strip(img)
                        frame_buffer.append(strip)
                        d_raw.append(None)
                        meta_buffer.append(None)
                        source_frame_buffer.append(frame_count)
                        metric_debug_buffer.append(None)
                        continue
                    cam_skipped += 1
                    continue
            if cam_warmup_remaining > 0:
                cam_warmup_remaining -= 1

            # 計算原始距離（僅在有校準時）
            dist_raw   = None
            bbox_strip = None
            homography_local_dist = None
            metric_debug = None
            if cam['m_per_pixel'] is not None or cam.get('H_matrix') is not None:
                d = velocity_tracker[fastest_id]
                bx1, by1, bx2, by2 = d['bbox']
                cx_orig = (bx1 + bx2) / 2.0 + crop_x_offset
                cy_orig = (by1 + by2) / 2.0 + crop_y_offset
                if cam.get('H_matrix') is not None:
                    ground_pt = d.get('smoothed_ground_point') or d.get('ground_point')
                    if ground_pt is not None:
                        image_point = (
                            float(ground_pt[0]) + crop_x_offset,
                            float(ground_pt[1]) + crop_y_offset,
                        )
                        if cam.get('start_mid') is not None and cam.get('track_dir') is not None:
                            image_point = _project_point_to_track_line(
                                image_point, cam['start_mid'], cam['track_dir'])
                        world_input = (
                            float(image_point[0]),
                            float(image_point[1]),
                        )
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
                    proj_px  = _project_onto_track((cx_orig, cy_orig),
                                                   cam['start_mid'], cam['track_dir'])
                    dist_raw = cumulative_dist_offset + max(0.0, proj_px * cam['m_per_pixel'])
                else:
                    # 舊模式：x 位移
                    dist_raw = cumulative_dist_offset + max(0.0, (cx_orig - cam['start_x']) * cam['m_per_pixel'])
                scale    = TARGET_HEIGHT / img.shape[0]
                bbox_strip = (int(bx1 * scale), int(by1 * scale),
                              int(bx2 * scale), int(by2 * scale))

            # 縮放並加相機標籤（不加速度文字，留給第二段）
            strip = _make_strip(img, fastest_id)
            # pre-roll / 起跑確認（track_roi 模式才有效）
            if cam.get('track_roi') is not None and not runner_crossed_start:
                d = velocity_tracker[fastest_id]
                bx1_c, by1_c, bx2_c, by2_c = d['bbox']
                cx_orig = (bx1_c + bx2_c) / 2.0 + crop_x_offset
                cy_orig = (by1_c + by2_c) / 2.0 + crop_y_offset
                proj_px = _project_onto_track((cx_orig, cy_orig),
                                              cam['start_mid'], cam['track_dir'])
                dist_val = dist_raw if dist_raw is not None else (d_raw[-1] if d_raw else cumulative_dist_offset)

                if frame_count <= 30:
                    print(
                        f"  [debug cam{cam_idx+1} f{frame_count}] "
                        f"fastest_id={fastest_id} proj_px={proj_px:.1f} "
                        f"crossed={runner_crossed_start} dist_raw={dist_raw} "
                        f"cand={len(candidate_buf)}"
                    )

                if proj_px < 0:
                    # 起跑線前：略過，並清空 candidate（如有後退）
                    candidate_buf.clear()
                    continue  # 不進 frame_buffer，不觸發切換

                else:  # proj_px >= 0：進入候選區
                    # 若不是單調遞增（後退或持平）→ 清空候選，從頭收集
                    if candidate_buf and proj_px <= candidate_buf[-1][2]:
                        candidate_buf.clear()

                    candidate_buf.append((strip, dist_val, proj_px, bbox_strip,
                                          frame_count, metric_debug))

                    if len(candidate_buf) >= K_CONFIRM:
                        # 確認起跑！
                        runner_crossed_start = True
                        target_runner_id = fastest_id
                        for c_strip, c_dist, _, c_meta, c_source_frame, c_debug in candidate_buf:
                            frame_buffer.append(c_strip)
                            d_raw.append(c_dist)
                            meta_buffer.append(c_meta)
                            source_frame_buffer.append(c_source_frame)
                            metric_debug_buffer.append(c_debug)
                        candidate_buf.clear()
                        print(
                            f"  [debug cam{cam_idx+1}] confirmed start at frame {frame_count}, "
                            f"locked_id={target_runner_id}, "
                            f"first_dist={d_raw[0] if d_raw else None}, "
                            f"buffered={len(frame_buffer)}"
                        )
                    continue  # 不論確認與否，本幀已透過 candidate_buf 處理

            frame_buffer.append(strip)
            d_raw.append(dist_raw if dist_raw is not None
                         else (d_raw[-1] if d_raw else cumulative_dist_offset))
            meta_buffer.append(bbox_strip)
            source_frame_buffer.append(frame_count)
            metric_debug_buffer.append(metric_debug)

            # 切換條件
            if (cam.get('H_matrix') is not None and
                    cam.get('distance_m') is not None and
                    homography_local_dist is not None):
                # Homography 模式：距離計算與切換使用同一個世界座標基準。
                if homography_local_dist >= float(cam['distance_m']):
                    print(f"  → 觸發{'退出ROI' if is_last_cam else '切換'}："
                          f"Homography距離={homography_local_dist:.2f}m >= {float(cam['distance_m']):.2f}m")
                    break
            elif cam.get('track_dir') and cam.get('pixel_span'):
                # 斜線模式：投影距離 >= pixel_span 即越過終點線
                d = velocity_tracker[fastest_id]
                bx1, by1, bx2, by2 = d['bbox']
                cy_orig = (by1 + by2) / 2.0 + crop_y_offset
                ref_x   = bx2 + crop_x_offset if is_last_cam else (bx1 + bx2) / 2.0 + crop_x_offset
                proj_px = _project_onto_track((ref_x, cy_orig),
                                              cam['start_mid'], cam['track_dir'])
                if proj_px >= cam['pixel_span']:
                    print(f"  → 觸發{'退出ROI' if is_last_cam else '切換'}："
                          f"投影={proj_px:.0f}px >= {cam['pixel_span']:.0f}px")
                    break
            else:
                # 舊模式：最後一機用 bx2（右緣退出 ROI），其餘機用 center_x
                trigger_x = fastest_bx2_orig if is_last_cam else fastest_center_orig
                if switch_x is not None and trigger_x is not None:
                    if trigger_x > switch_x:
                        ref_name = 'bx2' if is_last_cam else 'center_x'
                        print(f"  → 觸發{'退出ROI' if is_last_cam else '切換'}：{ref_name}={trigger_x:.0f} > {switch_x}")
                        break

        cap.release()
        total_skipped += cam_skipped
        print(f"  [第一段完成] 收集 {len(frame_buffer)} 幀，捨棄 {cam_skipped} 幀")

        valid_metric_indices = [i for i, v in enumerate(d_raw) if v is not None]
        if valid_metric_indices:
            first_valid = valid_metric_indices[0]
            last_valid = valid_metric_indices[-1]
            if first_valid > 0 or last_valid < len(d_raw) - 1:
                trimmed = first_valid + (len(d_raw) - 1 - last_valid)
                frame_buffer = frame_buffer[first_valid:last_valid + 1]
                d_raw = d_raw[first_valid:last_valid + 1]
                meta_buffer = meta_buffer[first_valid:last_valid + 1]
                source_frame_buffer = source_frame_buffer[first_valid:last_valid + 1]
                metric_debug_buffer = metric_debug_buffer[first_valid:last_valid + 1]
                print(f"  補值: 裁掉 {trimmed} 幀沒有前後有效 bbox 的端點缺失")

        interpolated_bbox_mask = [False] * len(meta_buffer)
        missing_metric_frames = sum(v is None for v in d_raw)
        has_any_metric = any(v is not None for v in d_raw)
        d_raw_for_csv = list(d_raw)
        if missing_metric_frames and has_any_metric:
            interpolated_bbox_mask = [m is None for m in meta_buffer]
            d_raw = _interpolate_missing_numeric(d_raw)
            meta_buffer = _interpolate_missing_bboxes(meta_buffer)
            print(f"  補值: 已用前後有效偵測插值補 {missing_metric_frames} 幀距離/bbox")
        interp_gap_len, speed_confidence = _interpolation_metadata(interpolated_bbox_mask)

        # -----------------------------------------------------------------------
        # 第二段：批次計算速度/加速度，疊加文字 + 圖表，寫入影片
        # -----------------------------------------------------------------------
        has_metrics = (
            (cam['m_per_pixel'] is not None or cam.get('H_matrix') is not None) and
            CHART_HEIGHT > 0 and len(d_raw) > 0 and has_any_metric
        )
        if has_metrics:
            print(f"  [第二段] 計算 Butterworth + Kalman（init_v={last_kf_v:.2f}, init_a={last_kf_a:.2f}）...")
            d_smooth, v_smooth, a_arr = _compute_kf_series(d_raw, fps,
                                                           init_v=last_kf_v,
                                                           init_a=last_kf_a,
                                                           measurement_confidence=speed_confidence)
            # 更新跨機累計偏移：本機最終距離即為下一機的起點
            cumulative_dist_offset = float(d_smooth[-1]) if len(d_smooth) > 0 else cumulative_dist_offset
            # 保存本機最終 Kalman 狀態，供下一機初始化用
            last_kf_v = float(v_smooth[-1])
            last_kf_a = float(a_arr[-1])
            print(f"  本機最終距離: {cumulative_dist_offset:.2f}m，速度: {last_kf_v:.2f}m/s（下一機起點）")

            # 前機已累積的完整序列（numpy array；第一機時為空）
            d_prev = np.array(accumulated_d) if accumulated_d else np.array([])
            v_prev = np.array(accumulated_v) if accumulated_v else np.array([])
            a_prev = np.array(accumulated_a) if accumulated_a else np.array([])

        print(f"  [第二段] 渲染輸出中...")
        cam_written = 0

        for i, strip in enumerate(frame_buffer):
            if has_metrics and meta_buffer[i] is not None:
                # 正式幀：文字 + 圖表 + CSV 資料
                dist_i  = float(d_smooth[i])
                speed_i = float(v_smooth[i])
                accel_i = float(a_arr[i])

                # 疊加速度文字
                bx1s, by1s = meta_buffer[i][0], meta_buffer[i][1]
                if i < len(interpolated_bbox_mask) and interpolated_bbox_mask[i]:
                    bx1i, by1i, bx2i, by2i = meta_buffer[i]
                    cv2.rectangle(strip, (bx1i, by1i), (bx2i, by2i), (255, 0, 255), 2)
                    cv2.putText(strip, "interp", (bx1i + 3, by1i + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2)
                label = (f"{dist_i:.1f}m  "
                         f"{speed_i:.2f}m/s  "
                         f"{accel_i:+.1f}m/s\u00b2")
                strip = _draw_text_bgr(
                    strip,
                    label,
                    (bx1s, max(by1s - 32, 5)),
                    font=_get_font(size=22),
                    color=(0, 255, 255),
                    thickness=2,
                )

                # 底部圖表：第一幀時建立畫布（之後重複使用）
                if chart_fig is None:
                    _cw = strip.shape[1]
                    chart_fig, chart_axes = plt.subplots(
                        1, 3, figsize=(_cw / 100, CHART_HEIGHT / 100), dpi=100)
                    chart_canvas = FigureCanvas(chart_fig)

                # 前機完整 + 本機到第 i 幀（absolute_frame 概念：跨機連續）
                if len(d_prev) > 0:
                    d_cur = np.concatenate([d_prev, d_smooth[:i + 1]])
                    v_cur = np.concatenate([v_prev, v_smooth[:i + 1]])
                    a_cur = np.concatenate([a_prev, a_arr[:i + 1]])
                else:
                    d_cur = d_smooth[:i + 1]
                    v_cur = v_smooth[:i + 1]
                    a_cur = a_arr[:i + 1]

                chart = _draw_chart(
                    chart_fig, chart_axes, chart_canvas,
                    d_cur, v_cur, a_cur,
                    fps, strip.shape[1], CHART_HEIGHT,
                    global_d_max, global_t_max,
                    font_prop=chart_font_prop)
                frame_out = np.vstack([strip, chart])

                debug_i = metric_debug_buffer[i] if i < len(metric_debug_buffer) else None
                dist_raw_i = d_raw_for_csv[i] if i < len(d_raw_for_csv) else None
                world_x_i = debug_i.get('world_x') if debug_i else None
                image_point_x_i = debug_i.get('image_point_x') if debug_i else None
                image_point_y_i = debug_i.get('image_point_y') if debug_i else None
                all_track_data.append({
                    'cam':            cam_idx + 1,
                    'cam_frame':      i,
                    'source_frame':   source_frame_buffer[i] if i < len(source_frame_buffer) else '',
                    'absolute_frame': absolute_frame_offset + i,
                    'dist_m':         round(dist_i, 3),
                    'dist_raw_m':     round(float(dist_raw_i), 3) if dist_raw_i is not None else '',
                    'dist_smooth_m':  round(dist_i, 3),
                    'world_x':        round(float(world_x_i), 3) if world_x_i is not None else '',
                    'image_point_x':  round(float(image_point_x_i), 3) if image_point_x_i is not None else '',
                    'image_point_y':  round(float(image_point_y_i), 3) if image_point_y_i is not None else '',
                    'speed_mps':      round(speed_i, 3),
                    'accel_mps2':     round(accel_i, 3),
                    'is_interpolated': int(interpolated_bbox_mask[i]) if i < len(interpolated_bbox_mask) else 0,
                    'interp_gap_len':  interp_gap_len[i] if i < len(interp_gap_len) else 0,
                    'speed_confidence': round(speed_confidence[i], 3) if i < len(speed_confidence) else 1.0,
                })
            elif has_metrics and meta_buffer[i] is None:
                # pre-roll 幀：保留畫面但圖表區填黑
                empty = np.zeros((CHART_HEIGHT, strip.shape[1], 3), dtype=np.uint8)
                frame_out = np.vstack([strip, empty])
            else:
                frame_out = strip

            # 初始化 VideoWriter（第一幀才知道最終尺寸）
            if out is None:
                h_out, w_out = frame_out.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (w_out, h_out))
                print(f"  VideoWriter 初始化：{w_out}x{h_out}")

            out.write(frame_out)
            cam_written  += 1
            total_written += 1

        # 本機序列加入跨機累積（供下一機圖表連續顯示）
        if has_metrics:
            accumulated_d.extend(d_smooth.tolist())
            accumulated_v.extend(v_smooth.tolist())
            accumulated_a.extend(a_arr.tolist())
        absolute_frame_offset += len(frame_buffer)  # 更新絕對幀號偏移

        print(f"  相機 {cam_idx+1} 完成：寫入 {cam_written} 幀")

    # -----------------------------------------------------------------------
    # 收尾
    # -----------------------------------------------------------------------
    if out:
        out.release()
    if chart_fig is not None:
        plt.close(chart_fig)

    # CSV 輸出
    if all_track_data:
        csv_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME.replace('.mp4', '_metrics.csv'))
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

    _write_homography_visualizations(CAMERAS, OUTPUT_DIR, track_data=all_track_data)

    print(f"\n{'='*60}")
    print(f"全部完成：總寫入 {total_written} 幀，總捨棄 {total_skipped} 幀")
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"輸出: {output_path} ({size_mb:.2f} MB)")
    else:
        print("Critical Error: 輸出檔案不存在！")


if __name__ == "__main__":
    main()
