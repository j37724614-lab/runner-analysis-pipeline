"""
track_crop_roi.py

功能：使用 YOLO 追蹤影片中的最快人物，以其為中心進行固定大小裁剪，
      輸出乾淨影片。支援最多 6 台相機串接，依 switch_x 自動切換。

處理流程：
  1. 前處理裁剪（crop_params）：縮小 YOLO 搜索範圍
  2. YOLO track()：持續追蹤人物，取得 bbox 與 track ID
  3. ROI 過濾：排除不在有效像素區域內的偵測結果
  4. 速度計算：累積移動距離，選出最快且正在移動的人物
  5. 疊加框（SHOW_OVERLAY）：綠色 bbox（最快跑者）+ 藍色 ROI 框
  6. 固定裁剪：以最快人物中心為基準，裁出固定尺寸並寫入輸出影片
  7. 相機切換：center_x / bx2 超過 switch_x 時切換至下一台

統計輸出：每幀的 bbox 尺寸（最大、平均、中位數），協助決定 CROP_WIDTH/CROP_HEIGHT 設定。
"""
# =======================================================================
# 負責系統資源控制與底層函式庫的初始化設定。在使用 Python 處理多鏡頭影片（OpenCV / FFMPEG）加上深度學習模型（PyTorch / YOLO）時
# 很容易因為「狂開執行緒（Threads）」而導致系統崩潰（報錯 Resource temporarily unavailable 或是直接死當）。
# 動作：透過環境變數，強制將這些底層運算的執行緒降到 1。FFMPEG 解碼影片也強制只用 1 個 thread。
# =======================================================================
import logging
import resource as _res

logger = logging.getLogger(__name__)

try:
    _soft, _hard = _res.getrlimit(_res.RLIMIT_NPROC)
    if _soft < _hard:
        _res.setrlimit(_res.RLIMIT_NPROC, (_hard, _hard))
except (OSError, ValueError):
    logger.warning(
        "無法調整 RLIMIT_NPROC，將使用原本的系統限制",
        exc_info=True,
    )

import os

# 限制執行緒數，避免 RLIMIT_NPROC 超限（在 import cv2 之前設定）
for _k, _v in {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "GOMP_SPINCOUNT": "0",
    "OPENCV_FFMPEG_CAPTURE_OPTIONS": "threads;1",  # FFMPEG option name is "threads"
}.items():
    os.environ[_k] = _v  # 強制覆蓋，不用 setdefault

import cv2

try:
    cv2.setLogLevel(3)  # type: ignore[attr-defined]  # 抑制 swscaler 色彩轉換警告
except AttributeError:
    pass
import argparse
import csv
import json
from collections import namedtuple
from itertools import pairwise

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from core.utils import DEFAULT_OUTPUT_DIR, get_font_path, get_model_path

# =======================================================================
# 設定區（每次修改只需改這裡）
# =======================================================================

# 使用哪張實體 GPU（'0' = 第 0 張，'1' = 第 1 張）
CUDA_VISIBLE_DEVICES = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
DEVICE = 0

# 模型權重路徑（下載方式見 README）
MODEL_PATH = get_model_path("yolo26x.pt")

# 中文字型路徑（ROI 標籤使用）
FONT_PATH = get_font_path()

# 輸出目錄（檔名依第一台有效相機自動命名：{輸入檔名}_tracked.mp4）
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

# fallback 裁剪尺寸（以最快人物中心為基準）
# 一般分析流程應使用 auto_crop，避免不同解析度或人物大小都被限制在固定尺寸。
# 建議值：先執行一次看底部「建議 CROP_WIDTH/CROP_HEIGHT」的統計輸出再調整
# 或在 config 加入 auto_crop: true，讓程式依 bbox 統計自動決定正方形尺寸
CROP_WIDTH  = 200
CROP_HEIGHT = 260
AUTO_CROP   = False  # True → 先 dry-run 收集 bbox 統計，自動設定裁剪尺寸
TRACKING_MODE = 'online'  # 'online' | 'two_pass'

# Optional temporal pre-scan. Only used when TRACKING_MODE == 'two_pass'.
# It scans sampled frames with a TensorRT INT8 YOLO engine, finds ranges where
# a valid person appears, expands the ranges with buffer, and limits both
# two-pass Pass 1 and Pass 2 to those original-frame ranges.
PRESCAN_ENABLED = False
PRESCAN_ENGINE_PATH = get_model_path("yolo26x_ultralytics_int8.engine")
PRESCAN_STRIDE = 8
PRESCAN_IMGSZ = 640
PRESCAN_CONF = 0.25
PRESCAN_IOU = 0.7
PRESCAN_BUFFER_SEC = 1.0
PRESCAN_MAX_GAP_SEC = 1.0
PRESCAN_USE_GRAB = True

# 是否在裁剪畫面上疊加框
#   True  = 綠色 bbox（最快跑者）+ 藍色 ROI 框
#   False = 輸出乾淨畫面（無任何疊加）
SHOW_OVERLAY = True
DRAW_BBOX_OVERLAY = True  # True → 輸出追焦影片畫 bbox 與 track ID；bbox_map.csv 仍會照常輸出給 HRNet

# -----------------------------------------------------------------------
# 移動偵測參數
# -----------------------------------------------------------------------
MOVEMENT_THRESHOLD  = 2   # 判定為移動的最小像素位移,連續兩幀中心位移 > 2px 才算「有移動」
MIN_MOVEMENT_FRAMES = 3   # 需連續移動至少此幀數才視為「真正移動」,必須連續移動 ≥ 3 幀，才被認定為「真的在跑」
STATIONARY_DECAY    = 2   # 靜止時每幀遞減 movement_count 的量, 靜止一幀就讓 movement_count 減 2（快速重置）
MAX_PERSON_MEMORY   = 30  # 超過此幀數未偵測到則清除該人物的速度紀錄, 30 幀（約 0.5 秒）沒出現就刪除
MIN_PERSON_HEIGHT   = 40  # bbox 高度小於此值（前處理裁剪後像素）視為背景遠景人物，略過
GROUND_POINT_EMA_ALPHA = 0.35  # 與 tracker_impl.py 對齊：bbox 底部中心點平滑係數
LANE_WIDTH_M = 1.22


# -----------------------------------------------------------------------
# camera() — 快速建立相機設定的 helper function
#
# 必填：
#   video_path  str or None   影片路徑；填 None 表示此台不使用（自動跳過）
#
# 選填（有預設值）：
#   crop        (x起,y起,x終,y終)  前處理裁剪範圍，None = 不裁剪
#   roi_x       (左, 右)           ROI x 範圍，以 bbox 中心點判斷，預設不限制
#   roi_y       (上, 下)           ROI y 範圍，以 bbox 中心點判斷，預設不限制
#   switch_x    int or None        最快人物 center_x（原始座標）超過此值時切下一台
#                                  None = 跑完整段再切（最後有效台自動忽略）
#
# 若需要多個 ROI zone，改用 roi_zones 參數直接傳入 list：
#   roi_zones=[{'x':(0,800),'y':(0,9999)}, {'x':(1000,1920),'y':(0,9999)}]
#
# ── 斜線起終點模式（選用，取代 roi_x / switch_x）──
# start_line  [(x1,y1), (x2,y2)]  起跑線兩端點（原始影像座標）
# end_line    [(x3,y3), (x4,y4)]  終點線兩端點（原始影像座標）
#             ← 同時填入時：switch_x 自動忽略，改由越線事件觸發切換
#             ← ROI 改由向量投影計算，不受相機視角偏斜影響
# pre_roll_px int  起跑線前的緩衝距離（投影像素），預設 200
# -----------------------------------------------------------------------
def camera(video_path, crop=None,
           roi_x=(0, 9999), roi_y=(0, 9999),
           switch_x=None, roi_zones=None,
           start_line=None, end_line=None,
           pre_roll_px=200,
           end_roll_px=120,
           start_roi_px=100,
           homography_lane_margin_px=80,
           distance_m=None,
           H_matrix=None,
           homography_src_points=None,
           homography_dst_world=None):
    """
    start_line / end_line（可選）：各由兩個原始影像座標點組成的斜線，
      例如 start_line=[(150, 420), (150, 780)]。
    同時填入兩者時：
      - ROI 改由向量投影取代矩形過濾
      - switch_x 自動設為 None（切換改由 end_line 越線事件觸發）
    """
    # switch_x 未指定且 roi_x 有設定時，自動取右邊界（語意：人物到達 ROI 右緣即切換）
    if switch_x is None and roi_x != (0, 9999):
        switch_x = roi_x[1]
    zones  = roi_zones if roi_zones is not None else [{'x': roi_x, 'y': roi_y}]
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

    if homography_src_points is not None:
        src_arr = np.asarray(homography_src_points, dtype=np.float32)
        if len(src_arr) >= 6:
            quad_roi = np.array([src_arr[0], src_arr[1], src_arr[4], src_arr[5]], dtype=np.float32)
        elif len(src_arr) == 5:
            quad_roi = np.array([src_arr[0], src_arr[1], src_arr[3], src_arr[4]], dtype=np.float32)

    return {
        'video_path':  video_path,
        'crop_params': crop,
        'roi_enabled': not no_roi,
        'roi_zones':   zones,
        'switch_x':    switch_x,
        'distance_m':  distance_m,
        # 斜線模式欄位（舊模式均為 None）
        'start_line':  start_line,
        'end_line':    end_line,
        'start_mid':   start_mid,
        'end_mid':     end_mid,
        'track_dir':   track_dir,
        'pixel_span':  pixel_span,
        'quad_roi':    quad_roi,
        'homography_lane_margin_px': homography_lane_margin_px,
        'track_roi':   {'start_mid':   start_mid,
                        'track_dir':   track_dir,
                        'pixel_span':  pixel_span,
                        'pre_roll_px': pre_roll_px,
                        'end_roll_px': end_roll_px,
                        'start_roi_px': start_roi_px}
                       if start_mid is not None and track_dir is not None else None,
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
# -----------------------------------------------------------------------
# 相機設定（最多 6 台）
#
# video_path  填影片路徑 = 啟用；填 None = 此台不使用
# crop        (x起, y起, x終, y終) 裁剪範圍，縮小 YOLO 搜索背景
# roi_x       (左, 右) ROI 有效範圍（原始影像像素）
# switch_x    跑者 center_x 超過此值時切換到下一台（最後一台用 bx2 右緣判斷）
# start_line  [(x1,y1),(x2,y2)] 起跑線兩端點；與 end_line 同時填入才啟用斜線模式
# end_line    [(x3,y3),(x4,y4)] 終點線兩端點
# -----------------------------------------------------------------------
CAM1 = camera("/home/jeter/pipeline_release/video/0506_1.mp4",
              crop=(0, 400, 1920, 800),
              start_line=[(208, 715), (123, 725)],
              end_line  =[(1760, 710), (1830, 718)])

CAM2 = camera("/home/jeter/pipeline_release/video/0506_2.mp4", #"/home/jeter/MotionAGFormer/IMG_5728.mp4",
              crop=(0, 400, 1920, 800),
              start_line=[(208, 715), (123, 725)],
              end_line  =[(1760, 710), (1830, 718)])

CAM3 = camera("/home/jeter/pipeline_release/video/0506_3.mp4", #"/home/jeter/MotionAGFormer/0331-2.mp4",
              crop=(0, 400, 1920, 800),
              start_line=[(215, 713), (130, 727)],
              end_line  =[(1735, 715), (1820, 722)])

CAM4 = camera(None,                  # ← 填路徑即可啟用
              crop=(0, 400, 1920, 800),
              start_line=[(150, 700), (150, 790)],
              end_line  =[(1820, 700), (1820, 790)])

CAM5 = camera(None,
              crop=(0, 400, 1920, 800),
              start_line=[(150, 700), (150, 790)],
              end_line  =[(1820, 700), (1820, 790)])

CAM6 = camera(None,
              crop=(0, 400, 1920, 800),
              start_line=[(150, 700), (150, 790)],
              end_line  =[(1820, 700), (1820, 790)])

# =======================================================================
# 以下為程式邏輯，一般不需修改
# =======================================================================

def _build_camera_from_entry(entry):
    """將 config dict 的單台相機 entry 轉換成 camera dict。"""
    crop_val = entry.get('crop')
    # 解析 start_line / end_line：YAML 格式為 [[x1,y1],[x2,y2]]
    sl = entry.get('start_line')
    el = entry.get('end_line')
    start_line = [tuple(p) for p in sl] if sl else None
    end_line   = [tuple(p) for p in el] if el else None
    H_matrix = None
    src_pts = entry.get('homography_src_points') or entry.get('src_points')
    dst_pts = (
        entry.get('homography_dst_world') or
        entry.get('homography_dst_points_world')
    )
    if src_pts is not None and dst_pts is None and entry.get('start_meter') is not None:
        dst_pts = build_lane_world_points(
            float(entry['start_meter']),
            num_points=len(src_pts),
        )
    if src_pts is not None and dst_pts is not None:
        H_matrix, _ = _compute_homography(src_pts, dst_pts)
    distance_m = entry.get('distance_m')
    if distance_m is None and src_pts and entry.get('start_meter') is not None:
        distance_m = 20.0

    # 若未手動設 crop 但有 start_line/end_line，自動以四個點的外接矩形建立 YOLO 前處理 ROI。
    # 上方保留較大的空間，避免近距離或高解析度影片中跑者上半身在 YOLO 前就被裁掉。
    if crop_val is None and start_line is not None and end_line is not None:
        padding = int(entry.get('pre_roll_px', 200))
        top_padding = int(entry.get('crop_top_padding_px', padding * 3))
        bottom_padding = int(entry.get('crop_bottom_padding_px', padding))
        all_pts = list(start_line) + list(end_line)
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        crop_val = (
            int(max(0, min(xs) - padding)),
            int(max(0, min(ys) - top_padding)),
            int(max(xs) + padding),
            int(max(ys) + bottom_padding),
        )

    return camera(
        video_path=entry.get('video_path'),
        crop=tuple(crop_val) if crop_val else None,
        roi_x=tuple(entry['roi_x']) if 'roi_x' in entry else (0, 9999),
        roi_y=tuple(entry['roi_y']) if 'roi_y' in entry else (0, 9999),
        switch_x=entry.get('switch_x'),
        roi_zones=entry.get('roi_zones'),
        start_line=start_line,
        end_line=end_line,
        pre_roll_px=int(entry.get('pre_roll_px', 200)),
        end_roll_px=int(entry.get('end_roll_px', 120)),
        start_roi_px=int(entry.get('start_roi_px', 100)),
        homography_lane_margin_px=int(entry.get('homography_lane_margin_px', 80)),
        distance_m=distance_m,
        H_matrix=H_matrix,
        homography_src_points=src_pts,
        homography_dst_world=dst_pts,
    )


def load_cameras_from_config(config_path):
    """
    從 YAML 設定檔載入相機清單，回傳 (cameras, cfg)。
    最多支援 6 台；video_path 填 null 或省略 → 該台停用。
    """
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cameras = [_build_camera_from_entry(e) for e in (cfg.get('cameras') or [])[:6]]
    return cameras, cfg


def _get_font(size=28, font_path=FONT_PATH):
    """載入字型；失敗時回傳 None 以便安全降級。"""
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            return None
    return None


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


def _point_track_area_proximity(point, quad_roi, margin_px=120):
    """
    評估一個點是否貼近起點線與終點線圍出的跑道四邊形。

    回傳值範圍 0..1：
      - 點在四邊形內：1
      - 點在四邊形外：依離邊界距離遞減
    """
    if point is None or quad_roi is None:
        return 1.0

    margin_px = max(float(margin_px), 1.0)
    signed_dist = cv2.pointPolygonTest(
        np.asarray(quad_roi, dtype=np.float32),
        (float(point[0]), float(point[1])),
        True,
    )
    if signed_dist >= 0:
        return 1.0
    return max(0.0, 1.0 + signed_dist / margin_px)


def _compute_homography(src_points, dst_points_world):
    """根據影像點與世界座標點計算 Homography。"""
    if src_points is None or dst_points_world is None:
        return None, None

    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points_world, dtype=np.float32)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(
            "homography_src_points / homography_dst_world 必須是 shape=(N,2) 且點數一致"
        )
    if len(src) < 4:
        raise ValueError("Homography 至少需要 4 組對應點")

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography 計算失敗，請檢查點位是否共線或順序是否對應")
    return H, mask


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


def _transform_point_homography(point, H):
    """把單一影像點轉到世界座標，回傳 (xw, yw)。"""
    if H is None:
        return None
    pts = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pts, H)
    return tuple(mapped[0, 0])


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


def _bbox_bottom_center(bbox):
    """回傳 bbox 的底部中心點，較接近跑者落地點。"""
    bx1, _by1, bx2, by2 = bbox
    return ((bx1 + bx2) / 2.0, float(by2))


def _interpolate_bbox(left_bbox, right_bbox, ratio):
    """用左右兩個有效 bbox 線性插值出中間 bbox。"""
    left = np.asarray(left_bbox, dtype=float)
    right = np.asarray(right_bbox, dtype=float)
    return tuple(np.rint(left + ratio * (right - left)).astype(int))


def _apply_preprocess_crop(img, crop_params):
    """套用前處理裁剪。回傳 (裁剪後 img 或 None, off_x, off_y)；crop_params 為 None 時 off_x=off_y=0。"""
    if not crop_params:
        return img, 0, 0
    cx1, cy1, cx2, cy2 = crop_params
    h, w = img.shape[:2]
    cx1, cx2 = max(0, cx1), min(w, cx2)
    cy1, cy2 = max(0, cy1), min(h, cy2)
    if cx2 <= cx1 or cy2 <= cy1:
        return None, None, None
    return img[cy1:cy2, cx1:cx2], cx1, cy1


def _fixed_size_crop(img, bx1, by1, bx2, by2):
    """以 bbox 中心為基準，裁出 CROP_WIDTH x CROP_HEIGHT 固定尺寸畫面。
    回傳 (crop_frame, bbox_in_crop, c1x, c1y)；c1x/c1y 為裁剪視窗左上角在 img 座標系中的位置。
    """
    h_img, w_img = img.shape[:2]
    cx = int((bx1 + bx2) / 2)
    cy = int((by1 + by2) / 2)
    c1x = cx - CROP_WIDTH // 2
    c1y = cy - CROP_HEIGHT // 2
    c2x = c1x + CROP_WIDTH
    c2y = c1y + CROP_HEIGHT

    if c1x < 0:
        c1x, c2x = 0, CROP_WIDTH
    elif c2x > w_img:
        c2x, c1x = w_img, w_img - CROP_WIDTH
    if c1y < 0:
        c1y, c2y = 0, CROP_HEIGHT
    elif c2y > h_img:
        c2y, c1y = h_img, h_img - CROP_HEIGHT
    c1x = max(0, c1x)
    c1y = max(0, c1y)
    c2x = min(c2x, w_img)
    c2y = min(c2y, h_img)

    crop_frame = img[c1y:c2y, c1x:c2x]
    bbox_in_crop = (
        int(np.clip(bx1 - c1x, 0, CROP_WIDTH - 1)),
        int(np.clip(by1 - c1y, 0, CROP_HEIGHT - 1)),
        int(np.clip(bx2 - c1x, 0, CROP_WIDTH - 1)),
        int(np.clip(by2 - c1y, 0, CROP_HEIGHT - 1)),
    )

    if crop_frame.shape[:2] != (CROP_HEIGHT, CROP_WIDTH):
        if crop_frame.size > 0:
            crop_frame = cv2.resize(crop_frame, (CROP_WIDTH, CROP_HEIGHT),
                                    interpolation=cv2.INTER_LINEAR)
        else:
            crop_frame = np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)

    return crop_frame, bbox_in_crop, c1x, c1y


def _crop_from_bbox(img, crop_params, bbox, label_interpolated=False, track_id=None):
    """
    使用指定 bbox 重新裁切一幀，供 YOLO 缺失幀的插值補幀使用。

    bbox 座標系與 process_frame() 相同：若有 crop_params，bbox 是前處理裁剪後的座標。
    回傳 (crop_frame, bbox_in_crop, off_x, off_y)。
    """
    img, cx1, cy1 = _apply_preprocess_crop(img, crop_params)
    if img is None:
        return None, None, None, None

    bx1, by1, bx2, by2 = map(int, bbox)
    h_img, w_img = img.shape[:2]
    bx1 = int(np.clip(bx1, 0, max(w_img - 1, 0)))
    bx2 = int(np.clip(bx2, 0, max(w_img - 1, 0)))
    by1 = int(np.clip(by1, 0, max(h_img - 1, 0)))
    by2 = int(np.clip(by2, 0, max(h_img - 1, 0)))
    if bx2 <= bx1 or by2 <= by1:
        return None, None, None, None

    if SHOW_OVERLAY and DRAW_BBOX_OVERLAY:
        color = (255, 0, 255) if label_interpolated else (0, 255, 0)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
        label_parts = []
        if track_id is not None:
            label_parts.append(f"ID {track_id}")
        if label_interpolated:
            label_parts.append("interp")
        if label_parts:
            label = " ".join(label_parts)
            label_y = max(by1 - 8, 20)
            cv2.putText(
                img,
                label,
                (bx1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                label,
                (bx1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

    crop_frame, bbox_in_crop, c1x_out, c1y_out = _fixed_size_crop(img, bx1, by1, bx2, by2)
    return crop_frame, bbox_in_crop, c1x_out + cx1, c1y_out + cy1


def _get_frame_detections(img, model, device, cached_detections, locked_target_id):
    """取得本幀偵測框。

    cached_detections 給定時走 two_pass 快取模式（跳過 YOLO inference），否則呼叫
    model.track()。若已鎖定 target ID（locked_target_id），丟棄所有其他偵測，讓非目標
    人物不進入 velocity_tracker。
    conf=0.3：只接受 YOLO 判定為人的分數 ≥ 0.3 的框（可調低以抓更模糊的）
    iou=0.1：框重疊超過 10% 就視為同一個（可調高以容許更多重疊）
    imgsz=1280：推論解析度（較高 → 偵測小人較準）
    回傳 (boxes, ids)，皆可能為 None。
    """
    if cached_detections is not None:
        # Pass 2 快取模式：跳過 YOLO inference，直接使用 Pass 1 的偵測結果
        if cached_detections:
            boxes = np.array([[d['bx1'], d['by1'], d['bx2'], d['by2']]
                               for d in cached_detections], dtype=np.float32)
            ids   = np.array([d['track_id'] for d in cached_detections], dtype=np.float32)
        else:
            boxes, ids = None, None
    else:
        results = model.track(img, persist=True, classes=[0], show=False, device=device,
                              conf=0.3, iou=0.1, imgsz=1280, verbose=False)
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids   = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None
        else:
            boxes, ids = None, None

    if locked_target_id is not None and ids is not None and len(ids) > 0:
        target_int = int(locked_target_id)
        mask = ids.astype(int) == target_int
        if mask.any():
            boxes = boxes[mask]
            ids   = ids[mask]
        else:
            boxes, ids = None, None

    return boxes, ids


def _collect_valid_detections(boxes, ids, track_roi, roi_enabled, roi_zones, quad_roi,
                              homography_lane_margin_px, nearest_to_start,
                              crop_x_offset, crop_y_offset):
    """依高度、ROI（斜線投影或矩形）、Homography 車道範圍、nearest_to_start 過濾出本幀候選偵測。

    回傳 valid_detections：list of
      (dist_to_start, center_x, center_y, bx1, by1, bx2, by2, tid, proj, ground_pt)
    """
    valid_detections = []
    if boxes is None or ids is None or len(boxes) == 0:
        return valid_detections

    # 第一輪：收集所有通過過濾的偵測，和 tracker_impl.py 保持一致。
    for i in range(len(boxes)):
        bx1, by1, bx2, by2 = map(int, boxes[i])
        if (by2 - by1) < MIN_PERSON_HEIGHT:
            continue
        center_x = (bx1 + bx2) / 2   # bbox 中心 x（裁剪座標）
        center_y = (by1 + by2) / 2   # bbox 中心 y（裁剪座標）
        ground_pt = _bbox_bottom_center((bx1, by1, bx2, by2))
        ground_orig = (
            ground_pt[0] + crop_x_offset,
            ground_pt[1] + crop_y_offset,
        )

        # ROI 過濾（換算回原始影片座標再比對）
        orig_cx = center_x + crop_x_offset
        orig_cy = center_y + crop_y_offset
        proj = None
        if track_roi is not None:
            # 斜線模式：投影到跑道方向，範圍 [-pre_roll_px, pixel_span]
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
            # 舊矩形模式
            if not any(z['x'][0] <= orig_cx <= z['x'][1] and
                       z['y'][0] <= orig_cy <= z['y'][1]
                       for z in roi_zones):
                continue

        if ids is None:
            continue
        tid = int(ids[i])
        dist_to_start = abs(proj) if track_roi is not None else float('inf')
        valid_detections.append((dist_to_start, center_x, center_y,
                                 bx1, by1, bx2, by2, tid, proj, ground_pt))

    # Homography 跑道區域優先：如果有候選人在跑道四邊形附近，就先排除遠離跑道的人。
    if quad_roi is not None and valid_detections:
        lane_filtered = []
        for det in valid_detections:
            (_, _, _, bx1, by1, bx2, by2, _, _, ground_pt) = det
            ground_orig = (
                ground_pt[0] + crop_x_offset,
                ground_pt[1] + crop_y_offset,
            )
            signed_dist = cv2.pointPolygonTest(
                quad_roi,
                (float(ground_orig[0]), float(ground_orig[1])),
                True,
            )
            if signed_dist >= -float(homography_lane_margin_px):
                lane_filtered.append(det)
        if lane_filtered:
            valid_detections = lane_filtered

    # nearest_to_start 模式：只保留距 start_line 最近的一人。
    if nearest_to_start and valid_detections:
        valid_detections.sort(key=lambda x: x[0])
        valid_detections = valid_detections[:1]

    return valid_detections


def _update_velocity_tracker(velocity_tracker, valid_detections, track_roi, instant_start):
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
            d['proj']   = proj
            d['frames_since_detected'] = 0
        else:
            velocity_tracker[tid] = {
                'center':                (center_x, center_y),
                'bbox':                  (bx1, by1, bx2, by2),
                'ground_point':          ground_pt,
                'smoothed_ground_point': ground_pt,
                'proj':                  proj,
                'velocities':            [0],
                'movement_count':        MIN_MOVEMENT_FRAMES if instant_start else 1,
                'stationary_count':      0,
                'frames_since_detected': 0,
            }
        if track_roi is not None and proj is not None:
            lead_candidates.append({'tid': tid, 'proj': proj})
    return seen_ids, lead_candidates


def _prune_stale_tracks(velocity_tracker, seen_ids):
    """清除超過 MAX_PERSON_MEMORY 幀未偵測到的追蹤條目（in-place）。"""
    for tid in list(velocity_tracker):
        if tid not in seen_ids:
            velocity_tracker[tid]['frames_since_detected'] += 1
            if velocity_tracker[tid]['frames_since_detected'] > MAX_PERSON_MEMORY:
                del velocity_tracker[tid]


def _select_fastest_runner(velocity_tracker, locked_target_id, prefer_lead_runner, lead_candidates):
    """選出本幀最快人物 ID；找不到（含鎖定目標本幀未偵測到）時回傳 None。"""
    if locked_target_id is not None:
        locked_target_id = int(locked_target_id)
        d = velocity_tracker.get(locked_target_id)
        if d is not None and d['frames_since_detected'] == 0:
            return locked_target_id
        return None

    fastest_id = None
    max_vel = -1.0
    for tid, d in velocity_tracker.items():
        if (
            d['frames_since_detected'] == 0  # 本幀有偵測到
            and d['movement_count'] >= MIN_MOVEMENT_FRAMES  # 連續移動 ≥ 3 幀
            and d['stationary_count'] < 10  # 靜止幀數 < 10
        ):
            v = np.mean(d['velocities']) if d['velocities'] else 0
            if v > max_vel:
                max_vel = v
                fastest_id = tid

    if fastest_id is None and prefer_lead_runner and lead_candidates:
        fastest_id = max(lead_candidates, key=lambda c: c['proj'])['tid']

    return fastest_id


def _draw_frame_overlays(img, fastest_id, bbox, roi_enabled, roi_zones,
                         crop_x_offset, crop_y_offset, track_roi,
                         overlay_start_pts, overlay_end_pts):
    """在前處理裁剪後的畫面上疊加最快跑者框、ROI 範圍框、起終點線。回傳更新後的 img。"""
    bx1, by1, bx2, by2 = bbox

    # 綠色框 — 最快跑者 bbox
    if DRAW_BBOX_OVERLAY:
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
        label = f"ID {fastest_id}"
        label_y = max(by1 - 8, 20)
        cv2.putText(
            img,
            label,
            (bx1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            label,
            (bx1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # 藍色框 — ROI 範圍（舊矩形模式）
    if roi_enabled and roi_zones and track_roi is None:
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

    # 斜線模式：四邊形區域 + 起跑線（奶黃）+ 終點線（天空藍）
    if overlay_start_pts and overlay_end_pts:
        p0, p3 = overlay_start_pts
        p1, p2 = overlay_end_pts
        quad = np.array([p0, p1, p2, p3], dtype=np.int32)
        overlay_img = img.copy()
        cv2.fillPoly(overlay_img, [quad], (200, 220, 255))
        cv2.addWeighted(overlay_img, 0.15, img, 0.85, 0, img)
        for a, b in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
            _draw_dashed_line(img, a, b, (255, 255, 255), thickness=2)
    if overlay_start_pts:
        cv2.line(img, overlay_start_pts[0], overlay_start_pts[1], (0, 0, 0), 5)
        cv2.line(img, overlay_start_pts[0], overlay_start_pts[1], (180, 255, 255), 3)
    if overlay_end_pts:
        cv2.line(img, overlay_end_pts[0], overlay_end_pts[1], (0, 0, 0), 5)
        cv2.line(img, overlay_end_pts[0], overlay_end_pts[1], (255, 200, 100), 3)

    return img


def process_frame(img, model, velocity_tracker, device,
                  crop_params, roi_enabled, roi_zones,
                  crop_x_offset, crop_y_offset,
                  instant_start=False,
                  track_roi=None,
                  quad_roi=None,
                  homography_lane_margin_px=80,
                  overlay_start_pts=None,
                  overlay_end_pts=None,
                  prefer_lead_runner=False,
                  nearest_to_start=False,
                  locked_target_id=None,
                  cached_detections=None):
    """
    對單幀執行：前處理裁剪 → YOLO track → ROI 過濾 + 速度累積 →
                選最快人物 → 疊加框 → 固定大小裁剪。

    track_roi          dict or None：斜線模式的投影 ROI 參數（含 start_mid/track_dir/pixel_span/pre_roll_px）
    quad_roi           np.ndarray or None：Homography/跑道四邊形，優先保留附近偵測
    overlay_start_pts  tuple[2] or None：起跑線兩端點（crop 後座標系），SHOW_OVERLAY 時繪製
    overlay_end_pts    tuple[2] or None：終點線兩端點（crop 後座標系），SHOW_OVERLAY 時繪製

    回傳：(crop_frame, fastest_id, fastest_center_orig, fastest_bx2_orig, bbox_in_crop)
      crop_frame=None          → ROI 內無有效人物，上層應跳過此幀
      fastest_center_orig      → 最快人物 center_x（原始座標），非最後一機的切換基準
      fastest_bx2_orig         → 最快人物 bbox 右緣（原始座標），最後一機的退出 ROI 基準
      bbox_in_crop             → 最終輸出裁切畫面內的 bbox，供 HRNet 限定偵測範圍
    """
    # Step 1: 前處理裁剪
    img, _, _ = _apply_preprocess_crop(img, crop_params)
    if img is None:
        return None, None, None, None, None, None, None

    # Step 2: YOLO track()（或 two_pass 快取）
    boxes, ids = _get_frame_detections(img, model, device, cached_detections, locked_target_id)

    # Step 3: ROI 過濾 + 速度累積
    valid_detections = _collect_valid_detections(
        boxes, ids, track_roi, roi_enabled, roi_zones, quad_roi,
        homography_lane_margin_px, nearest_to_start,
        crop_x_offset, crop_y_offset,
    )
    seen_ids, lead_candidates = _update_velocity_tracker(
        velocity_tracker, valid_detections, track_roi, instant_start
    )
    _prune_stale_tracks(velocity_tracker, seen_ids)

    # Step 4: 選出最快人物
    fastest_id = _select_fastest_runner(
        velocity_tracker, locked_target_id, prefer_lead_runner, lead_candidates
    )
    if fastest_id is None:
        return None, None, None, None, None, None, None

    d = velocity_tracker[fastest_id]
    bx1, by1, bx2, by2 = d['bbox']
    ground_x, _ground_y = d.get('ground_point', _bbox_bottom_center((bx1, by1, bx2, by2)))
    fastest_center_orig = ground_x + crop_x_offset
    fastest_bx2_orig    = bx2 + crop_x_offset

    # Step 5: 疊加框（在前處理裁剪後的畫面上）
    if SHOW_OVERLAY:
        img = _draw_frame_overlays(
            img, fastest_id, (bx1, by1, bx2, by2), roi_enabled, roi_zones,
            crop_x_offset, crop_y_offset, track_roi,
            overlay_start_pts, overlay_end_pts,
        )

    # Step 6: 固定大小裁剪（以最快人物 bbox 中心為基準）
    crop_frame, bbox_in_crop, c1x, c1y = _fixed_size_crop(img, bx1, by1, bx2, by2)

    return crop_frame, fastest_id, fastest_center_orig, fastest_bx2_orig, bbox_in_crop, c1x + crop_x_offset, c1y + crop_y_offset


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
    roi_enabled = cam.get('roi_enabled')
    roi_zones = cam.get('roi_zones') or []
    valid_count = 0
    for box in xyxy:
        bx1, by1, bx2, by2 = map(float, box[:4])
        if by2 - by1 < min_height:
            continue
        center_x = (bx1 + bx2) / 2.0
        center_y = (by1 + by2) / 2.0
        ground_pt = _bbox_bottom_center((bx1, by1, bx2, by2))
        if track_roi is not None:
            proj_px = _project_onto_track(
                ground_pt, track_roi['start_mid'], track_roi['track_dir']
            )
            pre_roll = track_roi.get('pre_roll_px', 0)
            end_roll = track_roi.get('end_roll_px', 0)
            if not (-pre_roll <= proj_px <= track_roi['pixel_span'] + end_roll):
                continue
        elif roi_enabled and roi_zones:
            if not any(z['x'][0] <= center_x <= z['x'][1] and
                       z['y'][0] <= center_y <= z['y'][1]
                       for z in roi_zones):
                continue
        valid_count += 1
    return valid_count > 0, valid_count


def run_temporal_prescan(cameras, output_dir=None):
    if not PRESCAN_ENABLED:
        return None
    if not PRESCAN_ENGINE_PATH or not os.path.exists(PRESCAN_ENGINE_PATH):
        print(f"  [prescan] engine not found, skip: {PRESCAN_ENGINE_PATH}")
        return None

    import time

    out_dir = output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(PRESCAN_ENGINE_PATH, task='detect')
    frame_ranges_by_cam = {}
    reports = []
    print("temporal pre-scan：使用 INT8 engine 找有效 frame range...")

    for cam_idx, cam in enumerate(cameras):
        video_path = cam.get('video_path')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [prescan] 相機 {cam_idx + 1}: 無法開啟 {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        buffer_frames = max(0, round(PRESCAN_BUFFER_SEC * fps))
        max_gap_frames = max(0, round(PRESCAN_MAX_GAP_SEC * fps))
        hit_frames = []
        sampled_rows = []
        sampled = 0
        valid_boxes_total = 0
        frame_idx = 0
        start_time = time.perf_counter()

        while frame_idx < total_frames:
            if frame_idx % PRESCAN_STRIDE == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                sampled += 1
                result = model.predict(
                    frame,
                    imgsz=PRESCAN_IMGSZ,
                    conf=PRESCAN_CONF,
                    iou=PRESCAN_IOU,
                    device=DEVICE,
                    verbose=False,
                )[0]
                is_hit, valid_count = _prescan_detection_is_valid(
                    result, cam, MIN_PERSON_HEIGHT
                )
                if is_hit:
                    hit_frames.append(frame_idx)
                    valid_boxes_total += valid_count
                sampled_rows.append({
                    'frame': frame_idx,
                    'time_s': frame_idx / fps if fps > 0 else None,
                    'hit': int(is_hit),
                    'valid_boxes': int(valid_count),
                })
            else:
                ok = cap.grab() if PRESCAN_USE_GRAB else cap.read()[0]
                if not ok:
                    break
            frame_idx += 1

        cap.release()
        elapsed = time.perf_counter() - start_time
        ranges = _merge_prescan_hit_frames(
            hit_frames, total_frames, PRESCAN_STRIDE, buffer_frames, max_gap_frames
        )
        frame_ranges_by_cam[cam_idx] = ranges
        range_rows = [
            {
                'start_frame': int(start),
                'end_frame': int(end),
                'num_frames': int(end - start + 1),
                'start_time_s': float(start / fps) if fps > 0 else None,
                'end_time_s': float(end / fps) if fps > 0 else None,
            }
            for start, end in ranges
        ]
        kept_frames = sum(r['num_frames'] for r in range_rows)
        reports.append({
            'cam_idx': cam_idx,
            'video_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'sampled_frames': sampled,
            'hit_sampled_frames': len(hit_frames),
            'valid_boxes': valid_boxes_total,
            'elapsed_sec': elapsed,
            'ranges': range_rows,
            'kept_frames': kept_frames,
            'kept_ratio': kept_frames / total_frames if total_frames else 0.0,
        })
        print(
            f"  [prescan] 相機 {cam_idx + 1}: sampled={sampled}, "
            f"hits={len(hit_frames)}, ranges={len(ranges)}, "
            f"kept={kept_frames}/{total_frames}, elapsed={elapsed:.2f}s"
        )

        first_cam_base = os.path.splitext(os.path.basename(video_path))[0]
        sample_path = os.path.join(out_dir, f"{first_cam_base}_prescan_samples.csv")
        with open(sample_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['frame', 'time_s', 'hit', 'valid_boxes']
            )
            writer.writeheader()
            writer.writerows(sampled_rows)

    first_cam_base = (
        os.path.splitext(os.path.basename(cameras[0]['video_path']))[0]
        if cameras and cameras[0].get('video_path') else 'tracking'
    )
    report_path = os.path.join(out_dir, f"{first_cam_base}_prescan_ranges.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'enabled': True,
            'engine_path': PRESCAN_ENGINE_PATH,
            'params': {
                'stride': PRESCAN_STRIDE,
                'imgsz': PRESCAN_IMGSZ,
                'conf': PRESCAN_CONF,
                'iou': PRESCAN_IOU,
                'buffer_sec': PRESCAN_BUFFER_SEC,
                'max_gap_sec': PRESCAN_MAX_GAP_SEC,
                'use_grab': PRESCAN_USE_GRAB,
            },
            'cameras': reports,
        }, f, ensure_ascii=False, indent=2)
    print(f"  [prescan] report: {report_path}")

    if not any(frame_ranges_by_cam.values()):
        print("  [prescan] 未找到有效區間，two_pass 將退回掃完整影片")
        return None
    return frame_ranges_by_cam


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


def _pass1_detection_roi_filter(ground_pt, center_x, center_y, track_roi, roi_enabled, roi_zones):
    """Pass 1 專用 ROI 過濾（斜線投影或矩形）。Pass 1 恆用完整原始畫面，crop_offset 恆為 0，
    故直接用 ground_pt/center_x/center_y，不需額外加 offset。回傳 (passes, proj_px)。
    """
    if track_roi is not None:
        proj_px = _project_onto_track(ground_pt, track_roi['start_mid'], track_roi['track_dir'])
        pre_roll = track_roi.get('pre_roll_px', 0)
        end_roll = track_roi.get('end_roll_px', 0)
        passes = -pre_roll <= proj_px <= track_roi['pixel_span'] + end_roll
        return passes, proj_px
    if roi_enabled and roi_zones:
        passes = any(z['x'][0] <= center_x <= z['x'][1] and
                     z['y'][0] <= center_y <= z['y'][1]
                     for z in roi_zones)
        return passes, None
    return True, None


def _collect_frame_detections(boxes, ids, cam_idx, source_frame, track_roi, roi_enabled, roi_zones):
    """依高度與 ROI 過濾單幀 YOLO 偵測。回傳 (rows, cache_entries)：
      rows          list[dict]：供 all_rows 累加，供評分使用
      cache_entries list[dict]：供 frame_cache[(cam_idx, source_frame)] 累加，供 Pass 2 跳過 YOLO
    """
    rows = []
    cache_entries = []
    if ids is None:
        return rows, cache_entries

    for i in range(len(boxes)):
        bx1, by1, bx2, by2 = map(int, boxes[i])
        bbox_h = by2 - by1
        if bbox_h < MIN_PERSON_HEIGHT:
            continue
        center_x = (bx1 + bx2) / 2.0
        center_y = (by1 + by2) / 2.0
        ground_pt = _bbox_bottom_center((bx1, by1, bx2, by2))

        passes, proj_px = _pass1_detection_roi_filter(
            ground_pt, center_x, center_y, track_roi, roi_enabled, roi_zones
        )
        if not passes:
            continue

        tid = int(ids[i])
        rows.append({
            'cam_idx':    cam_idx,
            'frame_idx':  source_frame,
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


def _collect_all_detections(caps, cameras, model, frame_ranges_by_cam=None):
    """
    Pass 1: 讀遍所有相機，以完整原始畫面收集每幀通過 ROI/高度過濾的所有偵測（不選最快）。
    回傳 (all_rows, frame_cache)：
      all_rows    list[dict]：每筆偵測（供評分使用）
      frame_cache dict[(cam_idx, frame_idx), list[dict]]：每幀的 bbox 快取（供 Pass 2 跳過 YOLO）
    ByteTrack 在相機切換時重置，與 _process_cameras 一致。
    """
    all_rows = []
    frame_cache = {}

    for cam_idx, (cam, cap) in enumerate(zip(cameras, caps)):
        _reset_yolo_trackers(model)

        # Pass 1 必須讓 YOLO 看完整原始 frame；bbox 因此也維持原始影像座標。
        track_roi = cam.get('track_roi')
        roi_enabled = cam['roi_enabled']
        roi_zones = cam['roi_zones']

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        valid_ranges = _frame_ranges_for_camera(frame_ranges_by_cam, cam_idx, total_frames)
        range_cursor = 0
        frame_count = 0
        if valid_ranges:
            frame_count = valid_ranges[0][0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        while cap.isOpened():
            range_cursor, frame_count, exhausted = _advance_to_next_valid_range(
                cap, valid_ranges, range_cursor, frame_count
            )
            if exhausted:
                break

            ret, frame = cap.read()
            if not ret:
                break

            source_frame = frame_count

            results = model.track(frame, persist=True, classes=[0], show=False,
                                  device=DEVICE, conf=0.25, iou=0.1,
                                  imgsz=1280, verbose=False, half=True)
            r = results[0]

            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None
                rows, cache_entries = _collect_frame_detections(
                    boxes, ids, cam_idx, source_frame, track_roi, roi_enabled, roi_zones
                )
                all_rows.extend(rows)
                if cache_entries:
                    frame_cache.setdefault((cam_idx, source_frame), []).extend(cache_entries)

            frame_count += 1

    return all_rows, frame_cache


def _compute_camera_total_frames(all_detections, frame_ranges_by_cam):
    """算出各相機的分母幀數，供 coverage 計算使用。

    frame_ranges_by_cam 有給時優先採用（prescan 有效區間總長）；否則用該相機所有偵測中
    最大的 frame_idx+1 推估。回傳 defaultdict(int)，缺項時安全回傳 0。
    """
    from collections import defaultdict

    cam_total_frames = defaultdict(int)
    if frame_ranges_by_cam:
        for cam_idx, ranges in frame_ranges_by_cam.items():
            cam_total_frames[cam_idx] = sum(int(end) - int(start) + 1 for start, end in ranges)
    else:
        for row in all_detections:
            ci = row['cam_idx']
            cam_total_frames[ci] = max(cam_total_frames[ci], row['frame_idx'] + 1)
    return cam_total_frames


def _score_track_candidate(rows, cam, total_frames):
    """依單一 (cam_idx, track_id) 的偵測序列，計算軌跡品質指標與綜合分數。

    total_frames 為該相機的分母幀數（用於 coverage）。回傳 dict：
    n_frames/coverage/progress/monotonic/start_proximity/roi_ratio/
    track_area_proximity/score（不含 cam_idx/track_id，由呼叫端補上）。
    """
    rows_s = sorted(rows, key=lambda r: r['frame_idx'])
    n = len(rows_s)
    coverage = n / (total_frames or 1)

    pixel_span = cam.get('pixel_span')
    quad_roi = cam.get('quad_roi')
    lane_margin_px = cam.get('homography_lane_margin_px', 120)
    projs = [r['proj_px'] for r in rows_s if r['proj_px'] is not None]

    if projs and pixel_span:
        progress = min(1.0, max(0.0, (max(projs) - min(projs)) / pixel_span))
    else:
        xs = [r['center_x'] for r in rows_s]
        progress = min(1.0, (max(xs) - min(xs)) / 1920.0) if len(xs) > 1 else 0.0

    ref_vals = projs if projs else [r['center_x'] for r in rows_s]
    if len(ref_vals) > 1:
        increases = sum(1 for a, b in pairwise(ref_vals) if b > a)
        monotonic = increases / (len(ref_vals) - 1)
    else:
        monotonic = 0.0

    first_proj = projs[0] if projs else 0.0
    start_proximity = 1.0 / (1.0 + max(0.0, float(first_proj)))

    if pixel_span and projs:
        roi_ratio = sum(1 for p in projs if 0 <= p <= pixel_span) / len(projs)
    else:
        roi_ratio = 1.0

    ground_points = [
        (r['ground_x'], r['ground_y'])
        for r in rows_s
        if r.get('ground_x') is not None and r.get('ground_y') is not None
    ]
    if quad_roi is not None and ground_points:
        track_area_proximity = sum(
            _point_track_area_proximity(p, quad_roi, lane_margin_px)
            for p in ground_points
        ) / len(ground_points)
    else:
        track_area_proximity = 1.0

    score = (0.10 * coverage + 0.10 * progress + 0.15 * monotonic +
             0.10 * start_proximity + 0.10 * roi_ratio +
             0.45 * track_area_proximity)

    return {
        'n_frames':        n,
        'coverage':        round(coverage, 4),
        'progress':        round(progress, 4),
        'monotonic':       round(monotonic, 4),
        'start_proximity': round(start_proximity, 4),
        'roi_ratio':       round(roi_ratio, 4),
        'track_area_proximity': round(track_area_proximity, 4),
        'score':           round(score, 4),
    }


def _select_best_candidate(summaries, cam_idx):
    """從 summaries 中選出指定相機分數最高的候選 track_id，並印出診斷訊息。

    回傳 track_id；找不到候選人時回傳 None。
    """
    candidates = [s for s in summaries if s['cam_idx'] == cam_idx
                  and s['n_frames'] >= 10
                  and s['progress'] >= 0.3
                  and s['monotonic'] >= 0.55]
    if not candidates:
        candidates = [s for s in summaries if s['cam_idx'] == cam_idx]
    if not candidates:
        print(f"  ⚠️  相機 {cam_idx+1}: 無候選人，two_pass 將退回 online 模式選人")
        return None

    candidates.sort(key=lambda s: s['score'], reverse=True)
    best = candidates[0]

    if len(candidates) > 1:
        second = candidates[1]
        if best['score'] > 0 and (best['score'] - second['score']) / best['score'] < 0.10:
            print(f"  ⚠️  相機 {cam_idx+1}: 主跑者選擇不穩定"
                  f"（第1={best['track_id']} {best['score']:.3f}，"
                  f"第2={second['track_id']} {second['score']:.3f}，差距<10%）")

    print(f"  [two_pass] 相機 {cam_idx+1}: 選中 ID={best['track_id']} "
          f"score={best['score']:.3f} "
          f"coverage={best['coverage']:.2f} progress={best['progress']:.2f} "
          f"monotonic={best['monotonic']:.2f} "
          f"track_area={best['track_area_proximity']:.2f}")

    return best['track_id']


def _score_and_select_runners(all_detections, cameras, frame_ranges_by_cam=None):
    """
    依軌跡品質評分，選出每台相機的主跑者。
    回傳 (preset_ids, summaries)：
      preset_ids  dict {cam_idx: track_id}
      summaries   list[dict] 所有 ID 的指標與分數
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for row in all_detections:
        groups[(row['cam_idx'], row['track_id'])].append(row)

    cam_total_frames = _compute_camera_total_frames(all_detections, frame_ranges_by_cam)

    summaries = []
    for (cam_idx, track_id), rows in groups.items():
        metrics = _score_track_candidate(rows, cameras[cam_idx], cam_total_frames[cam_idx])
        summaries.append({'cam_idx': cam_idx, 'track_id': track_id, **metrics})

    preset_ids = {}
    for cam_idx in range(len(cameras)):
        track_id = _select_best_candidate(summaries, cam_idx)
        if track_id is not None:
            preset_ids[cam_idx] = track_id

    return preset_ids, summaries


def _stitch_target_id(frame_cache, preset_ids, cameras, fps=30.0, max_dist_px=100):
    """
    In-place 修補 frame_cache：當 target_id 短暫消失，
    若空缺幀有其他 ID 的 bbox 位置與大小接近預期（線性插值），就把 track_id 改成 target_id。
    max_gap 由 fps 自動推算（約 0.2 秒），上限 15 幀。
    """
    max_gap = max(5, min(15, round(fps * 0.2)))
    for cam_idx, target_id in preset_ids.items():
        target_frames = sorted(
            fi for (ci, fi), dets in frame_cache.items()
            if ci == cam_idx and any(d['track_id'] == target_id for d in dets)
        )
        if len(target_frames) < 2:
            continue

        stitched = 0
        for i in range(len(target_frames) - 1):
            start_f = target_frames[i]
            end_f   = target_frames[i + 1]
            gap     = end_f - start_f - 1
            if gap < 1 or gap > max_gap:
                continue

            start_det = next(d for d in frame_cache[(cam_idx, start_f)] if d['track_id'] == target_id)
            end_det   = next(d for d in frame_cache[(cam_idx, end_f)]   if d['track_id'] == target_id)
            sx = (start_det['bx1'] + start_det['bx2']) / 2
            sy = (start_det['by1'] + start_det['by2']) / 2
            sh = start_det['by2'] - start_det['by1']
            ex = (end_det['bx1'] + end_det['bx2']) / 2
            ey = (end_det['by1'] + end_det['by2']) / 2
            eh = end_det['by2'] - end_det['by1']

            for t in range(start_f + 1, end_f):
                cached = frame_cache.get((cam_idx, t))
                if not cached:
                    continue
                if any(d['track_id'] == target_id for d in cached):
                    continue

                alpha  = (t - start_f) / (end_f - start_f)
                exp_cx = sx + alpha * (ex - sx)
                exp_cy = sy + alpha * (ey - sy)
                exp_h  = sh + alpha * (eh - sh)

                best_det   = None
                best_score = float('inf')
                for det in cached:
                    cx = (det['bx1'] + det['bx2']) / 2
                    cy = (det['by1'] + det['by2']) / 2
                    h  = det['by2'] - det['by1']
                    dist = ((cx - exp_cx) ** 2 + (cy - exp_cy) ** 2) ** 0.5
                    if dist > max_dist_px:
                        continue
                    size_ratio = max(h, exp_h) / max(min(h, exp_h), 1.0)
                    if size_ratio > 1.5:
                        continue
                    score = dist + 50.0 * (size_ratio - 1.0)
                    if score < best_score:
                        best_score = score
                        best_det   = det

                if best_det is not None:
                    best_det['track_id'] = target_id
                    stitched += 1

        if stitched:
            print(f"  [stitch] 相機 {cam_idx + 1}: 修補 {stitched} 幀 target_id={target_id}")


def _write_two_pass_debug(all_detections, summaries, preset_ids, base_name):
    """輸出 two_pass 的 debug CSV/JSON 供事後分析。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if all_detections:
        path = os.path.join(OUTPUT_DIR, f"{base_name}_all_tracks.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_detections[0].keys()))
            writer.writeheader()
            writer.writerows(all_detections)
        print(f"  All tracks:     {path}")

    if summaries:
        path = os.path.join(OUTPUT_DIR, f"{base_name}_track_summary.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"  Track summary:  {path}")

    selected_data = []
    for cam_idx, track_id in preset_ids.items():
        entry = {'cam_idx': cam_idx, 'track_id': track_id}
        for s in summaries:
            if s['cam_idx'] == cam_idx and s['track_id'] == track_id:
                entry.update({k: v for k, v in s.items()
                              if k not in ('cam_idx', 'track_id')})
                break
        selected_data.append(entry)
    path = os.path.join(OUTPUT_DIR, f"{base_name}_selected_runner.json")
    with open(path, 'w') as f:
        json.dump(selected_data, f, indent=2)
    print(f"  Selected runner: {path}")


def _auto_crop_side_from_bbox_sizes(widths, heights):
    """依 bbox 寬高樣本計算 auto square crop 邊長。"""
    if not widths or not heights:
        return None
    p90_side = max(np.percentile(widths, 90), np.percentile(heights, 90)) * 1.25
    max_side = max(max(widths), max(heights)) * 1.05
    return int(np.ceil(max(p90_side, max_side)))


def _auto_crop_from_selected_cache(frame_cache, preset_ids):
    """
    從 two_pass 已選主跑者 bbox 計算 auto crop。

    呼叫時機應在 _stitch_target_id() 之後，這樣短暫換 ID 或漏偵測後
    被修補回 target_id 的 bbox 也會納入尺寸統計。
    """
    widths = []
    heights = []
    for (cam_idx, _frame_idx), detections in frame_cache.items():
        target_id = preset_ids.get(cam_idx)
        if target_id is None:
            continue
        for det in detections:
            if int(det.get('track_id', -1)) != int(target_id):
                continue
            widths.append(int(det['bx2']) - int(det['bx1']))
            heights.append(int(det['by2']) - int(det['by1']))
    return _auto_crop_side_from_bbox_sizes(widths, heights), widths, heights


def _should_switch_camera(cam, velocity_tracker, fastest_id, fastest_center_orig,
                          fastest_bx2_orig, crop_x_offset, crop_y_offset, is_last_cam):
    """判斷是否觸發相機切換（或最後一台相機退出 ROI）。

    優先序：Homography 距離 → 斜線投影 pixel_span → switch_x，與 tracker_impl.py 對齊。
    純判斷、不印 log、不修改任何狀態。回傳 (should_switch, log_message)；
    log_message 在不切換時為 None。
    """
    track_roi = cam.get('track_roi')
    switch_x = cam.get('switch_x')
    action = '退出ROI' if is_last_cam else '切換'

    if (cam.get('H_matrix') is not None and
            cam.get('distance_m') is not None and
            fastest_id is not None):
        d = velocity_tracker.get(fastest_id)
        if d is not None:
            bx1, _by1, bx2, by2 = d['bbox']
            ground_pt = d.get('smoothed_ground_point') or d.get('ground_point')
            if ground_pt is not None:
                image_point = (
                    float(ground_pt[0]) + crop_x_offset,
                    float(ground_pt[1]) + crop_y_offset,
                )
                if cam.get('start_mid') is not None and cam.get('track_dir') is not None:
                    image_point = _project_point_to_track_line(
                        image_point, cam['start_mid'], cam['track_dir'])
            else:
                image_point = (
                    (bx1 + bx2) / 2.0 + crop_x_offset,
                    by2 + crop_y_offset,
                )
            world = _transform_point_homography(image_point, cam['H_matrix'])
            if world is not None:
                start_world_x = cam.get('homography_start_x')
                if start_world_x is None:
                    start_world_x = 0.0
                homography_local_dist = max(0.0, float(world[0]) - start_world_x)
                if homography_local_dist >= float(cam['distance_m']):
                    return True, (f"  → 觸發{action}："
                                  f"Homography距離={homography_local_dist:.2f}m "
                                  f">= {float(cam['distance_m']):.2f}m")
        return False, None

    if track_roi is not None and cam.get('pixel_span'):
        # 斜線模式：投影 >= pixel_span → 越過終點線
        d = velocity_tracker.get(fastest_id)
        if d is not None:
            bx1, _by1, bx2, by2 = d['bbox']
            ground_x, ground_y = d.get('ground_point', _bbox_bottom_center(d['bbox']))
            cy_orig = ground_y + crop_y_offset
            ref_x = bx2 + crop_x_offset if is_last_cam else ground_x + crop_x_offset
            proj_px = _project_onto_track(
                (ref_x, cy_orig), cam['start_mid'], cam['track_dir']
            )
            if proj_px >= cam['pixel_span']:
                return True, (f"  → 觸發{action}："
                              f"投影={proj_px:.0f}px >= {cam['pixel_span']:.0f}px")
        return False, None

    if switch_x is not None and fastest_id is not None:
        # 舊模式：x 座標比較
        trigger = fastest_bx2_orig if is_last_cam else fastest_center_orig
        if trigger is not None and trigger > switch_x:
            ref_name = 'bx2' if is_last_cam else 'center_x'
            return True, (f"  → 觸發{action}："
                          f"{ref_name}={trigger:.0f} > {switch_x}")
        return False, None

    return False, None


_CameraResult = namedtuple('_CameraResult', [
    'written', 'skipped', 'bbox_widths', 'bbox_heights',
    'max_bbox_width', 'max_bbox_height',
    'frame_map_rows', 'bbox_map_rows',
    'offsets', 'orig_frames', 'cam_indices',
    'output_frame_idx',
])


_CameraSetup = namedtuple('_CameraSetup', [
    'vid_w', 'vid_h', 'fps', 'total', 'valid_ranges',
    'cp', 'crop_x_offset', 'crop_y_offset',
    'overlay_start_pts', 'overlay_end_pts', 'track_roi',
])


def _setup_camera_context(cam_idx, cam, cap, model, dry_run, frame_cache,
                          is_last_cam, num_cameras, frame_ranges_by_cam):
    """驗證相機可用、重置 YOLO tracker、印出相機資訊，並算出本次處理所需的裁剪/ROI/疊加線設定。
    回傳 _CameraSetup。
    """
    if not cap.isOpened():
        raise ValueError(f"無法開啟相機 {cam_idx+1}: {cam['video_path']}")

    _reset_yolo_trackers(model)

    vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    valid_ranges = _frame_ranges_for_camera(frame_ranges_by_cam, cam_idx, total)

    if not dry_run:
        print(f"{'─'*60}")
        print(f"相機 {cam_idx+1}/{num_cameras}: {cam['video_path']}")
        print(f"  解析度: {vid_w}x{vid_h}，幀數: {total}，FPS: {fps:.1f}")
        if valid_ranges:
            kept_frames = sum(end - start + 1 for start, end in valid_ranges)
            print(f"  pre-scan range: {valid_ranges}，處理 {kept_frames}/{total} 幀")

    configured_cp = cam['crop_params']
    use_full_frame_tracking = frame_cache is not None
    cp = None if use_full_frame_tracking else configured_cp
    if configured_cp and not dry_run and not use_full_frame_tracking:
        cx1c, cy1c = max(0, configured_cp[0]), max(0, configured_cp[1])
        cx2c, cy2c = min(vid_w, configured_cp[2]), min(vid_h, configured_cp[3])
        print(f"  CROP: ({cx1c},{cy1c}) → ({cx2c},{cy2c})，"
              f"裁剪後: {cx2c-cx1c}x{cy2c-cy1c}")
    elif configured_cp and not dry_run:
        print("  two_pass 快取模式: Pass 1/Pass 2 的 YOLO bbox 使用完整原始畫面座標，不套用前處理 CROP")

    if not dry_run and cam['roi_enabled']:
        for j, z in enumerate(cam['roi_zones']):
            print(f"  ROI 區域 {j+1}: X={z['x']}, Y={z['y']}")

    switch_x  = cam.get('switch_x')
    track_roi = cam.get('track_roi')
    if not dry_run:
        if cam.get('H_matrix') is not None and cam.get('distance_m') is not None:
            print(f"  切換條件: Homography 距離 ≥ {float(cam['distance_m']):.2f}m")
        elif track_roi is not None:
            print(f"  ROI 模式: 斜線投影（pixel_span={cam['pixel_span']:.0f}px，"
                  f"pre_roll={track_roi['pre_roll_px']}px）")
            print("  切換條件: 投影距離 ≥ pixel_span（越過終點線）")
        elif switch_x:
            ref_label = 'bx2（右緣）' if is_last_cam else 'center_x'
            print(f"  切換條件: 最快人物 {ref_label}（原始座標）> {switch_x}px")
        else:
            print("  切換條件: 跑完整段影片")

    crop_x_offset = cp[0] if cp else 0
    crop_y_offset = cp[1] if cp else 0

    # 預計算起終點線在 crop 後影像座標系的位置（原始座標 − crop_offset）
    overlay_start_pts = overlay_end_pts = None
    if cam.get('start_line') and cam.get('end_line'):
        ox = crop_x_offset
        oy = crop_y_offset
        overlay_start_pts = (
            (int(cam['start_line'][0][0] - ox), int(cam['start_line'][0][1] - oy)),
            (int(cam['start_line'][1][0] - ox), int(cam['start_line'][1][1] - oy)),
        )
        overlay_end_pts = (
            (int(cam['end_line'][0][0] - ox), int(cam['end_line'][0][1] - oy)),
            (int(cam['end_line'][1][0] - ox), int(cam['end_line'][1][1] - oy)),
        )

    return _CameraSetup(
        vid_w=vid_w, vid_h=vid_h, fps=fps, total=total, valid_ranges=valid_ranges,
        cp=cp, crop_x_offset=crop_x_offset, crop_y_offset=crop_y_offset,
        overlay_start_pts=overlay_start_pts, overlay_end_pts=overlay_end_pts,
        track_roi=track_roi,
    )


def _call_process_frame_with_retry(frame, model, velocity_tracker, cp, cam,
                                   crop_x_offset, crop_y_offset, instant_start,
                                   track_roi, overlay_start_pts, overlay_end_pts,
                                   runner_crossed_start, target_runner_id, cached_detections):
    """呼叫 process_frame()；遇到 RuntimeError（多半是 CUDA OOM）時清 cache 後重試一次。"""
    kwargs = {
        'instant_start': instant_start,
        'track_roi': track_roi,
        'quad_roi': cam.get('quad_roi'),
        'homography_lane_margin_px': cam.get('homography_lane_margin_px', 80),
        'overlay_start_pts': overlay_start_pts,
        'overlay_end_pts': overlay_end_pts,
        'prefer_lead_runner': not runner_crossed_start,
        'nearest_to_start': not runner_crossed_start,
        'locked_target_id': target_runner_id,
        'cached_detections': cached_detections,
    }
    try:
        return process_frame(
            frame, model, velocity_tracker, DEVICE,
            cp, cam['roi_enabled'], cam['roi_zones'],
            crop_x_offset, crop_y_offset,
            **kwargs,
        )
    except RuntimeError:
        torch.cuda.empty_cache()
        return process_frame(
            frame, model, velocity_tracker, DEVICE,
            cp, cam['roi_enabled'], cam['roi_zones'],
            crop_x_offset, crop_y_offset,
            **kwargs,
        )


def _draw_overview_frame(overview_frame, velocity_tracker, fastest_id,
                         crop_x_offset, crop_y_offset, quad_roi):
    """在 overview 畫面上標記本幀所有偵測（主跑者綠色、其他橘色）與跑道四邊形（in-place）。"""
    if quad_roi is not None:
        cv2.polylines(overview_frame,
                      [quad_roi.reshape(-1, 1, 2).astype(np.int32)],
                      True, (255, 200, 50), 2)
    for tid, d in velocity_tracker.items():
        if d['frames_since_detected'] != 0:
            continue
        bx1, by1, bx2, by2 = d['bbox']
        ox1 = bx1 + crop_x_offset
        oy1 = by1 + crop_y_offset
        ox2 = bx2 + crop_x_offset
        oy2 = by2 + crop_y_offset
        color = (0, 255, 0) if tid == fastest_id else (0, 165, 255)
        cv2.rectangle(overview_frame, (ox1, oy1), (ox2, oy2), color, 2)
        label = f"ID {tid}"
        label_y = max(oy1 - 8, 20)
        cv2.putText(overview_frame, label, (ox1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(overview_frame, label, (ox1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


class _StartLineConfirmer:
    """管理斜線模式「起跑確認」狀態機。

    在跑者越過起跑線（proj_px >= 0）前，畫面進 pre-roll buffer 不輸出；越線後需連續
    k_confirm 幀投影單調遞增，才正式鎖定 target_id 並確認起跑（回補已緩衝的候選幀）。
    crossed=True 表示已確認越線（或非斜線模式/two_pass 模式下建構時直接視為已越線），
    此時呼叫端不應再呼叫 observe()，改用一般輸出路徑。
    """

    def __init__(self, crossed=False, target_id=None, k_confirm=3, pre_roll_max=5):
        self.crossed = crossed
        self.target_id = target_id
        self.k_confirm = k_confirm
        self.pre_roll_max = pre_roll_max
        self.pre_roll_buf = []        # (crop_frame, source_frame, bbox_in_crop, off_x, off_y)
        self.candidate_buf = []       # (crop_frame, proj_px, source_frame, bbox_in_crop, off_x, off_y)
        self.candidate_target_id = None

    def observe(self, cam, velocity_tracker, fastest_id, crop_frame, bbox_in_crop,
               source_frame, off_x, off_y, crop_x_offset, crop_y_offset,
               write_frame_fn, dry_run, frame_count):
        """處理一幀。回傳 'skip'（velocity_tracker 找不到 fastest_id，呼叫端應計入 skipped）、
        'buffered'（仍在 pre-roll 或候選累積中，不輸出）、'confirmed'（本次確認起跑，已透過
        write_frame_fn 補寫候選幀）。
        """
        d = velocity_tracker.get(fastest_id)
        if d is None:
            return 'skip'

        ground_x, ground_y = d.get('ground_point', _bbox_bottom_center(d['bbox']))
        cx_orig = ground_x + crop_x_offset
        cy_orig = ground_y + crop_y_offset
        proj_px = _project_onto_track(
            (cx_orig, cy_orig),
            cam['start_mid'], cam['track_dir']
        )

        if proj_px < 0:
            # 在起跑線前：放入 pre-roll buffer，不輸出
            self.candidate_buf.clear()
            self.candidate_target_id = None
            self.pre_roll_buf.append((crop_frame, source_frame, bbox_in_crop, off_x, off_y))
            if len(self.pre_roll_buf) > self.pre_roll_max:
                self.pre_roll_buf.pop(0)
            return 'buffered'

        # proj_px >= 0：進入候選區（需單調遞增才確認起跑）
        if self.candidate_target_id != fastest_id:
            self.candidate_target_id = fastest_id
            self.candidate_buf.clear()
        elif self.candidate_buf and proj_px <= self.candidate_buf[-1][1]:
            self.candidate_buf.clear()   # 後退或持平，重新收集

        self.candidate_buf.append((crop_frame, proj_px, source_frame, bbox_in_crop, off_x, off_y))

        if len(self.candidate_buf) >= self.k_confirm:
            self.crossed = True
            self.target_id = self.candidate_target_id
            # pre_roll 只用來提前鎖定追蹤 ID；輸出影片不寫入 start_line 前的幀。
            self.pre_roll_buf.clear()
            # 寫出 candidate 幀
            for c_frame, _, c_source, c_bbox, c_ox, c_oy in self.candidate_buf:
                write_frame_fn(c_frame, c_source, c_bbox, self.target_id,
                               off_x=c_ox, off_y=c_oy)
            self.candidate_buf.clear()
            if not dry_run:
                print(f"  [幀 {frame_count}] 確認起跑（投影={proj_px:.0f}px）")
            return 'confirmed'

        return 'buffered'


def _process_single_camera(cam_idx, cam, cap, model, out, dry_run, frame_map_path,
                           preset_target_ids, frame_cache, frame_ranges_by_cam,
                           is_last_cam, num_cameras, output_frame_idx):
    """處理單一相機的完整追蹤流程：setup → 逐幀迴圈（YOLO/cache → 起跑確認 → 輸出/插值 →
    相機切換判斷）→ teardown。

    output_frame_idx 為跨相機累計的輸出幀序號（frame_map/bbox_map 用），由呼叫端傳入、
    並在回傳值中續接給下一台相機。回傳 _CameraResult。
    """
    setup = _setup_camera_context(cam_idx, cam, cap, model, dry_run, frame_cache,
                                  is_last_cam, num_cameras, frame_ranges_by_cam)
    vid_w, vid_h, fps, total = setup.vid_w, setup.vid_h, setup.fps, setup.total
    valid_ranges = setup.valid_ranges
    cp = setup.cp
    crop_x_offset, crop_y_offset = setup.crop_x_offset, setup.crop_y_offset
    overlay_start_pts, overlay_end_pts = setup.overlay_start_pts, setup.overlay_end_pts
    track_roi = setup.track_roi
    range_cursor = 0

    velocity_tracker = {}
    _instant_start = (cam_idx > 0)
    cam_skipped = 0
    cam_written = 0
    total_written = 0
    total_skipped = 0
    all_bbox_widths  = []
    all_bbox_heights = []
    max_bbox_width   = 0
    max_bbox_height  = 0
    frame_map_rows = []
    bbox_map_rows = []
    all_offsets     = []
    all_orig_frames = []
    all_cam_indices = []
    frame_count = 0
    if valid_ranges:
        frame_count = valid_ranges[0][0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    cam_interpolated = 0
    pending_missing = []  # YOLO 暫時缺失的幀，等下一個有效 bbox 後線性補回
    last_valid_bbox = None

    # two_pass 模式：直接鎖定已選好的 ID，跳過起跑確認
    if preset_target_ids is not None and cam_idx in preset_target_ids:
        confirmer = _StartLineConfirmer(crossed=True, target_id=preset_target_ids[cam_idx])
    else:
        confirmer = _StartLineConfirmer(crossed=(track_roi is None))  # 舊模式直接視為已越線

    def _write_output_frame(frame_to_write, source_frame, bbox_for_hrnet=None,
                            track_id=None, is_interpolated=False, interp_gap_len=0,
                            off_x=0, off_y=0):
        nonlocal total_written, cam_written, output_frame_idx
        if not dry_run:
            out.write(frame_to_write)
            frame_map_rows.append({
                'output_frame': output_frame_idx,
                'cam': cam_idx + 1,
                'cam_frame': source_frame,
                'source_frame': source_frame,
            })
            if bbox_for_hrnet is not None:
                x1, y1, x2, y2 = bbox_for_hrnet
                bbox_map_rows.append({
                    'output_frame': output_frame_idx,
                    'cam': cam_idx + 1,
                    'cam_frame': source_frame,
                    'source_frame': source_frame,
                    'track_id': track_id if track_id is not None else '',
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'is_interpolated': int(is_interpolated),
                    'interp_gap_len': int(interp_gap_len),
                })
            all_offsets.append([off_x, off_y])
            all_orig_frames.append(source_frame)
            all_cam_indices.append(cam_idx)
        cam_written += 1
        total_written += 1
        output_frame_idx += 1

    def _flush_pending_missing(right_bbox, track_id=None, off_x=0, off_y=0):
        nonlocal cam_interpolated, last_valid_bbox
        if not pending_missing or last_valid_bbox is None or right_bbox is None:
            return
        gap_len = len(pending_missing)
        for idx, item in enumerate(pending_missing, start=1):
            ratio = idx / (gap_len + 1)
            interp_bbox = _interpolate_bbox(last_valid_bbox, right_bbox, ratio)
            interp_frame, interp_bbox_in_crop, interp_off_x, interp_off_y = _crop_from_bbox(
                item['frame'],
                cp,
                interp_bbox,
                label_interpolated=True,
                track_id=track_id,
            )
            if interp_frame is None:
                continue
            _write_output_frame(
                interp_frame,
                item['source_frame'],
                interp_bbox_in_crop,
                track_id,
                is_interpolated=True,
                interp_gap_len=gap_len,
                off_x=interp_off_x if interp_off_x is not None else off_x,
                off_y=interp_off_y if interp_off_y is not None else off_y,
            )
            cam_interpolated += 1
        pending_missing.clear()
    if not dry_run:
        print("  [處理中...]")

    # overview VideoWriter：輸出全幀（不追焦），標記所有偵測 ID
    overview_out = None
    overview_path = None
    if not dry_run and frame_map_path:
        _ov_fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
        overview_path = frame_map_path.replace(
            '_frame_map.csv', f'_cam{cam_idx+1}_overview.mp4'
        )
        overview_out = cv2.VideoWriter(overview_path, _ov_fourcc, fps, (vid_w, vid_h))

    while cap.isOpened():
        if valid_ranges:
            while range_cursor < len(valid_ranges) and frame_count > valid_ranges[range_cursor][1]:
                range_cursor += 1
                pending_missing.clear()
                if range_cursor < len(valid_ranges):
                    frame_count = valid_ranges[range_cursor][0]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            if range_cursor >= len(valid_ranges):
                break

        ret, frame = cap.read()
        if not ret:
            break
        source_frame = frame_count
        frame_count += 1

        # 在 process_frame 修改 img 之前先複製原始幀，供 overview 使用
        overview_frame = frame.copy() if overview_out is not None else None

        _ov_s = overlay_start_pts if SHOW_OVERLAY else None
        _ov_e = overlay_end_pts   if SHOW_OVERLAY else None
        _cached = frame_cache.get((cam_idx, source_frame), []) if frame_cache is not None else None
        if _cached is not None and preset_target_ids is not None:
            _tid = preset_target_ids.get(cam_idx)
            if _tid is not None:
                _cached = [d for d in _cached if d['track_id'] == _tid]

        crop_frame, fastest_id, fastest_center_orig, fastest_bx2_orig, bbox_in_crop, off_x, off_y = \
            _call_process_frame_with_retry(
                frame, model, velocity_tracker, cp, cam,
                crop_x_offset, crop_y_offset, _instant_start,
                track_roi, _ov_s, _ov_e,
                confirmer.crossed, confirmer.target_id, _cached,
            )

        # Overview：每幀都畫，主跑者綠色、其他偵測橘色
        if overview_frame is not None and overview_out is not None:
            _draw_overview_frame(overview_frame, velocity_tracker, fastest_id,
                                 crop_x_offset, crop_y_offset, cam.get('quad_roi'))
            overview_out.write(overview_frame)

        if crop_frame is None:
            if confirmer.crossed and last_valid_bbox is not None:
                pending_missing.append({
                    'frame': frame.copy(),
                    'source_frame': source_frame,
                })
            else:
                cam_skipped   += 1
                total_skipped += 1
            continue

        # ── 起跑確認邏輯（track_roi 模式且尚未確認越線）──
        if track_roi is not None and not confirmer.crossed:
            status = confirmer.observe(
                cam, velocity_tracker, fastest_id, crop_frame, bbox_in_crop,
                source_frame, off_x, off_y, crop_x_offset, crop_y_offset,
                _write_output_frame, dry_run, frame_count,
            )
            if status == 'skip':
                cam_skipped += 1
                total_skipped += 1
            elif status == 'confirmed':
                if fastest_id is not None and fastest_id in velocity_tracker:
                    last_valid_bbox = velocity_tracker[fastest_id]['bbox']
            # 無論結果為何，此幀已透過 confirmer 處理，不重複輸出
            continue

        # 正常幀（已越線或舊模式）：收集 bbox 統計
        current_valid_bbox = None
        if fastest_id is not None and fastest_id in velocity_tracker:
            bx1, by1, bx2, by2 = velocity_tracker[fastest_id]['bbox']
            current_valid_bbox = (bx1, by1, bx2, by2)
            bw = bx2 - bx1; bh = by2 - by1
            all_bbox_widths.append(bw); all_bbox_heights.append(bh)
            max_bbox_width = max(max_bbox_width, bw)
            max_bbox_height = max(max_bbox_height, bh)

        if not dry_run and track_roi is not None and fastest_id is not None:
            d = velocity_tracker.get(fastest_id)
            if d is not None:
                ground_x, ground_y = d.get('ground_point', _bbox_bottom_center(d['bbox']))
                proj_px = _project_onto_track(
                    (ground_x + crop_x_offset, ground_y + crop_y_offset),
                    cam['start_mid'],
                    cam['track_dir'],
                )
                if proj_px < 0:
                    cam_skipped += 1
                    total_skipped += 1
                    continue
                if cam.get('pixel_span') and proj_px > cam['pixel_span']:
                    print(f"  → 停止輸出：投影={proj_px:.0f}px > end_line={cam['pixel_span']:.0f}px")
                    break

        if pending_missing and current_valid_bbox is not None:
            _flush_pending_missing(current_valid_bbox, fastest_id, off_x=off_x, off_y=off_y)

        if not dry_run:
            _write_output_frame(crop_frame, source_frame, bbox_in_crop, fastest_id,
                                off_x=off_x, off_y=off_y)
            if frame_count % 100 == 0:
                print(f"  [幀 {frame_count}/{total}] 追蹤: {len(velocity_tracker)} | "
                      f"寫入: {cam_written} | 捨棄: {cam_skipped}")
        else:
            _write_output_frame(crop_frame, source_frame, bbox_in_crop, fastest_id,
                                off_x=off_x, off_y=off_y)

        if current_valid_bbox is not None:
            last_valid_bbox = current_valid_bbox

        # ── 切換條件：Homography → 斜線投影 → switch_x，與 tracker_impl.py 對齊 ──
        should_switch, switch_msg = _should_switch_camera(
            cam, velocity_tracker, fastest_id, fastest_center_orig, fastest_bx2_orig,
            crop_x_offset, crop_y_offset, is_last_cam,
        )
        if should_switch:
            if not dry_run and switch_msg:
                print(switch_msg)
            break

    cap.release()
    if overview_out is not None:
        overview_out.release()
        print(f"  Overview:  {overview_path}")
        overview_out = None
    if pending_missing:
        cam_skipped += len(pending_missing)
        total_skipped += len(pending_missing)
        pending_missing.clear()
    if not dry_run:
        print(f"  相機 {cam_idx+1} 完成：寫入 {cam_written}，捨棄 {cam_skipped}")
        if cam_interpolated:
            print(f"  補值: 已用前後有效 bbox 線性補回 {cam_interpolated} 幀")
        if track_roi is not None and not confirmer.crossed:
            print("  ⚠️  警告：整段影片跑者從未越過起跑線，本機輸出 0 幀")

    return _CameraResult(
        written=total_written,
        skipped=total_skipped,
        bbox_widths=all_bbox_widths,
        bbox_heights=all_bbox_heights,
        max_bbox_width=max_bbox_width,
        max_bbox_height=max_bbox_height,
        frame_map_rows=frame_map_rows,
        bbox_map_rows=bbox_map_rows,
        offsets=all_offsets,
        orig_frames=all_orig_frames,
        cam_indices=all_cam_indices,
        output_frame_idx=output_frame_idx,
    )


def _process_cameras(caps, cameras, model, out, dry_run=False, frame_map_path=None,
                     preset_target_ids=None, frame_cache=None, frame_ranges_by_cam=None):
    """
    逐台相機執行 YOLO 追蹤並（選擇性）寫入影片。

    dry_run=True  → 只收集 bbox 統計，不呼叫 out.write()，用於 auto_crop 第一遍掃描。
    dry_run=False → 正常模式，將裁剪後的幀寫入 out。

    回傳: (total_written, total_skipped, bbox_widths, bbox_heights, max_w, max_h)
    """
    total_written = 0
    total_skipped = 0
    all_bbox_widths  = []
    all_bbox_heights = []
    max_bbox_width   = 0
    max_bbox_height  = 0
    frame_map_rows = []
    bbox_map_rows = []
    # 跨所有相機累積，每幀記錄 (off_x, off_y)、原始幀號、所屬相機 index
    all_offsets     = []
    all_orig_frames = []
    all_cam_indices = []
    output_frame_idx = 0

    for cam_idx, (cam, cap) in enumerate(zip(cameras, caps)):
        is_last_cam = (cam_idx == len(cameras) - 1)
        result = _process_single_camera(
            cam_idx, cam, cap, model, out, dry_run, frame_map_path,
            preset_target_ids, frame_cache, frame_ranges_by_cam,
            is_last_cam, len(cameras), output_frame_idx,
        )
        total_written += result.written
        total_skipped += result.skipped
        all_bbox_widths.extend(result.bbox_widths)
        all_bbox_heights.extend(result.bbox_heights)
        max_bbox_width  = max(max_bbox_width, result.max_bbox_width)
        max_bbox_height = max(max_bbox_height, result.max_bbox_height)
        frame_map_rows.extend(result.frame_map_rows)
        bbox_map_rows.extend(result.bbox_map_rows)
        all_offsets.extend(result.offsets)
        all_orig_frames.extend(result.orig_frames)
        all_cam_indices.extend(result.cam_indices)
        output_frame_idx = result.output_frame_idx

    if frame_map_path and not dry_run:
        frame_dir = os.path.dirname(frame_map_path)
        if frame_dir:
            os.makedirs(frame_dir, exist_ok=True)
        with open(frame_map_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['output_frame', 'cam', 'cam_frame', 'source_frame']
            )
            writer.writeheader()
            writer.writerows(frame_map_rows)
        bbox_map_path = frame_map_path.replace('_frame_map.csv', '_bbox_map.csv')
        with open(bbox_map_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['output_frame', 'cam', 'cam_frame', 'source_frame',
                            'track_id', 'x1', 'y1', 'x2', 'y2',
                            'is_interpolated', 'interp_gap_len']
            )
            writer.writeheader()
            writer.writerows(bbox_map_rows)
        print(f"  Frame map: {frame_map_path}")
        print(f"  BBox map:  {bbox_map_path}")

    # 輸出 offsets.npz（每幀的裁剪視窗左上角在原始影片中的絕對像素座標）
    # 供 overlay_original.py 反查每幀對應的原始影片位置
    if not dry_run and cameras and cameras[0].get('video_path'):
        first_cam_base = os.path.splitext(os.path.basename(cameras[0]['video_path']))[0]
        offsets_path = os.path.join(OUTPUT_DIR, f"{first_cam_base}_offsets.npz")
        np.savez_compressed(
            offsets_path,
            offsets=np.array(all_offsets, dtype=np.int32),
            orig_frames=np.array(all_orig_frames, dtype=np.int32),
            cam_indices=np.array(all_cam_indices, dtype=np.int32),
        )
        print(f"  Offsets: {offsets_path}")

    return total_written, total_skipped, all_bbox_widths, all_bbox_heights, max_bbox_width, max_bbox_height



def _apply_config_overrides(cfg):
    """將 --config/--config-json 載入的 cfg dict 套用到模組層級設定常數。

    這是本模組唯一允許 mutate 這些全域常數的地方；main() 只呼叫這個函式，不直接碰 global。
    pipeline.py 會直接讀寫 tcr.CROP_WIDTH 等屬性做為執行期設定介面，因此這裡維持寫回模組
    全域變數的行為，而非回傳一個獨立的 config 物件（那需要同時改 pipeline.py 及本檔所有讀取
    這些常數的函式，屬於更大範圍的變更）。
    """
    global OUTPUT_DIR, CROP_WIDTH, CROP_HEIGHT, AUTO_CROP, SHOW_OVERLAY, DRAW_BBOX_OVERLAY, MOVEMENT_THRESHOLD, MIN_MOVEMENT_FRAMES, STATIONARY_DECAY, MAX_PERSON_MEMORY, MIN_PERSON_HEIGHT, TRACKING_MODE, PRESCAN_ENABLED, PRESCAN_ENGINE_PATH, PRESCAN_STRIDE, PRESCAN_IMGSZ, PRESCAN_CONF, PRESCAN_IOU, PRESCAN_BUFFER_SEC, PRESCAN_MAX_GAP_SEC, PRESCAN_USE_GRAB
    if 'output_dir'          in cfg: OUTPUT_DIR          = cfg['output_dir']
    if 'crop_width'          in cfg: CROP_WIDTH          = int(cfg['crop_width'])
    if 'crop_height'         in cfg: CROP_HEIGHT          = int(cfg['crop_height'])
    if 'auto_crop'           in cfg: AUTO_CROP           = bool(cfg['auto_crop'])
    if 'show_overlay'        in cfg: SHOW_OVERLAY        = bool(cfg['show_overlay'])
    if 'draw_bbox_overlay'   in cfg: DRAW_BBOX_OVERLAY   = bool(cfg['draw_bbox_overlay'])
    if 'movement_threshold'  in cfg: MOVEMENT_THRESHOLD  = int(cfg['movement_threshold'])
    if 'min_movement_frames' in cfg: MIN_MOVEMENT_FRAMES = int(cfg['min_movement_frames'])
    if 'stationary_decay'    in cfg: STATIONARY_DECAY    = int(cfg['stationary_decay'])
    if 'max_person_memory'   in cfg: MAX_PERSON_MEMORY   = int(cfg['max_person_memory'])
    if 'min_person_height'   in cfg: MIN_PERSON_HEIGHT   = int(cfg['min_person_height'])
    if 'tracking_mode'       in cfg: TRACKING_MODE       = str(cfg['tracking_mode'])
    if 'prescan_enabled'     in cfg: PRESCAN_ENABLED     = bool(cfg['prescan_enabled'])
    if 'prescan_engine_path' in cfg: PRESCAN_ENGINE_PATH = str(cfg['prescan_engine_path'])
    if 'prescan_stride'      in cfg: PRESCAN_STRIDE      = int(cfg['prescan_stride'])
    if 'prescan_imgsz'       in cfg: PRESCAN_IMGSZ       = int(cfg['prescan_imgsz'])
    if 'prescan_conf'        in cfg: PRESCAN_CONF        = float(cfg['prescan_conf'])
    if 'prescan_iou'         in cfg: PRESCAN_IOU         = float(cfg['prescan_iou'])
    if 'prescan_buffer_sec'  in cfg: PRESCAN_BUFFER_SEC  = float(cfg['prescan_buffer_sec'])
    if 'prescan_max_gap_sec' in cfg: PRESCAN_MAX_GAP_SEC = float(cfg['prescan_max_gap_sec'])
    if 'prescan_use_grab'    in cfg: PRESCAN_USE_GRAB    = bool(cfg['prescan_use_grab'])


def main():
    # --config 參數：若提供則從 YAML 動態載入相機設定，否則沿用上方硬編碼的 CAM1~CAM6
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument('--config',      type=str, default=None)
    _parser.add_argument('--config-json', type=str, default=None, dest='config_json')
    _args, _ = _parser.parse_known_args()

    if _args.config_json:
        try:
            _cfg = json.loads(_args.config_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"--config-json 格式錯誤：{e}") from e
        _cams = [_build_camera_from_entry(e) for e in (_cfg.get('cameras') or [])[:6]]
        CAMERAS = [c for c in _cams if c['video_path'] is not None]
    elif _args.config:
        _cams, _cfg = load_cameras_from_config(_args.config)
        CAMERAS = [c for c in _cams if c['video_path'] is not None]
    else:
        # 過濾 video_path=None 的槽位，組出有效相機清單
        CAMERAS = [c for c in [CAM1, CAM2, CAM3, CAM4, CAM5, CAM6]
                   if c['video_path'] is not None]
        _cfg = {}

    # 允許 config 覆蓋全域常數（--config 與 --config-json 共用）
    if _cfg:
        _apply_config_overrides(_cfg)

    if not CAMERAS:
        raise ValueError("所有相機的 video_path 均為 None，請至少設定一台。")

    # -----------------------------------------------------------------------
    # 先開啟所有 VideoCapture（在 YOLO 載入前），避免 FFMPEG 執行緒耗盡
    # -----------------------------------------------------------------------
    print("開啟影片檔案...")
    caps = []
    for cam_idx, cam in enumerate(CAMERAS):
        cap = cv2.VideoCapture(cam['video_path'])
        if not cap.isOpened():
            for c in caps:
                c.release()
            raise ValueError(f"無法開啟相機 {cam_idx+1}: {cam['video_path']}")
        caps.append(cap)
    print(f"  {len(caps)} 台相機影片開啟成功\n")

    # CUDA 環境檢查
    print("=" * 60)
    print("CUDA 環境檢查")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
    print(f"使用設備: cuda:{DEVICE}")
    print("=" * 60)

    # 載入模型並預熱
    print("\n加載 YOLO 模型...")
    try:
        model = YOLO(MODEL_PATH)
        model.predict(np.zeros((480, 640, 3), dtype=np.uint8), device=DEVICE, verbose=False)
        print(f"模型加載成功，共 {len(CAMERAS)} 台相機（串接模式）\n")
    except Exception as e:
        for c in caps:
            c.release()
        print(f"模型加載失敗: {e}")
        raise

    # 動態決定輸出檔名（依第一台有效相機的影片名）
    first_cam_base = os.path.splitext(os.path.basename(CAMERAS[0]['video_path']))[0]
    OUTPUT_NAME    = f"{first_cam_base}_tracked.mp4"
    output_path    = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    frame_map_path = os.path.join(OUTPUT_DIR, f"{first_cam_base}_tracked_frame_map.csv")

    # 寫入 marker 檔，供 run_pipeline.py 讀取實際輸出名稱
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, ".last_output_name"), "w") as _f:
        _f.write(OUTPUT_NAME)
    with open(os.path.join(OUTPUT_DIR, ".last_input_video"), "w") as _f:
        _f.write(CAMERAS[0]['video_path'])
    with open(os.path.join(OUTPUT_DIR, ".last_input_videos"), "w") as _f:
        _f.write("\n".join(cam['video_path'] for cam in CAMERAS))

    first_fps = caps[0].get(cv2.CAP_PROP_FPS) or 60.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]

    # crop 驗證（開始前確認所有相機的 crop 參數合法）
    for cam_idx, (cam, cap) in enumerate(zip(CAMERAS, caps)):
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    # -----------------------------------------------------------------------
    # auto_crop：online 模式用 dry-run 估計尺寸；two_pass 模式改在 Pass 1
    # 選出主跑者後，使用該主跑者 bbox 統計決定尺寸。
    # -----------------------------------------------------------------------
    if TRACKING_MODE != 'two_pass' and AUTO_CROP:
        print("auto_crop 模式：第一遍掃描（分析 bbox 尺寸，不寫影片）...")
        _, _, dry_bw, dry_bh, _, _ = _process_cameras(caps, CAMERAS, model, None, dry_run=True)
        # dry-run 已讀完所有影片，重新開啟
        caps = [cv2.VideoCapture(cam['video_path']) for cam in CAMERAS]
        if dry_bw and dry_bh:
            crop_side = _auto_crop_side_from_bbox_sizes(dry_bw, dry_bh)
            CROP_WIDTH = crop_side
            CROP_HEIGHT = crop_side
            print(f"  自動設定裁剪尺寸: {CROP_WIDTH} x {CROP_HEIGHT}（auto square）\n")
        else:
            print("  警告：未收集到 bbox 資料，沿用預設尺寸\n")

    print(f"輸出路徑: {output_path}")

    # -----------------------------------------------------------------------
    # 正式處理：逐台相機串接追蹤 + 寫入影片
    # -----------------------------------------------------------------------
    if TRACKING_MODE == 'two_pass':
        frame_ranges_by_cam = run_temporal_prescan(CAMERAS, output_dir=OUTPUT_DIR) if PRESCAN_ENABLED else None
        print("two_pass 模式：第一遍收集所有候選人軌跡...")
        caps_pass1 = [cv2.VideoCapture(cam['video_path']) for cam in CAMERAS]
        all_detections, frame_cache = _collect_all_detections(
            caps_pass1, CAMERAS, model, frame_ranges_by_cam=frame_ranges_by_cam
        )
        for c in caps_pass1:
            c.release()
        print(f"  收集完成：共 {len(all_detections)} 筆偵測")
        preset_ids, summaries = _score_and_select_runners(
            all_detections, CAMERAS, frame_ranges_by_cam=frame_ranges_by_cam
        )
        _stitch_target_id(frame_cache, preset_ids, CAMERAS, fps=first_fps)
        _write_two_pass_debug(all_detections, summaries, preset_ids, first_cam_base)
        if AUTO_CROP:
            crop_side, sel_bw, _sel_bh = _auto_crop_from_selected_cache(frame_cache, preset_ids)
            if crop_side:
                CROP_WIDTH = crop_side
                CROP_HEIGHT = crop_side
                print(f"  two_pass auto_crop: 使用已選主跑者 bbox 設定裁剪尺寸 "
                      f"{CROP_WIDTH} x {CROP_HEIGHT}（samples={len(sel_bw)}）")
            else:
                print("  警告：two_pass auto_crop 未收集到已選主跑者 bbox，沿用目前尺寸")
        out = cv2.VideoWriter(output_path, fourcc, first_fps, (CROP_WIDTH, CROP_HEIGHT))
        print("two_pass 模式：第二遍輸出追焦影片（快取模式，跳過 YOLO）...")
        total_written, total_skipped, all_bbox_widths, all_bbox_heights, max_bbox_width, max_bbox_height = \
            _process_cameras(caps, CAMERAS, model, out, frame_map_path=frame_map_path,
                             preset_target_ids=preset_ids, frame_cache=frame_cache,
                             frame_ranges_by_cam=frame_ranges_by_cam)
    else:
        out = cv2.VideoWriter(output_path, fourcc, first_fps, (CROP_WIDTH, CROP_HEIGHT))
        total_written, total_skipped, all_bbox_widths, all_bbox_heights, max_bbox_width, max_bbox_height = \
            _process_cameras(caps, CAMERAS, model, out, frame_map_path=frame_map_path)

    # -----------------------------------------------------------------------
    # 收尾
    # -----------------------------------------------------------------------
    out.release()

    avg_w = int(np.mean(all_bbox_widths))  if all_bbox_widths  else 0
    avg_h = int(np.mean(all_bbox_heights)) if all_bbox_heights else 0
    med_w = int(np.median(all_bbox_widths))  if all_bbox_widths  else 0
    med_h = int(np.median(all_bbox_heights)) if all_bbox_heights else 0

    print(f"\n{'='*60}")
    print(f"全部完成：總寫入 {total_written} 幀，總捨棄 {total_skipped} 幀")
    print(f"  輸出: {output_path}")
    print(f"  輸出尺寸: {CROP_WIDTH} x {CROP_HEIGHT}")
    print("\nBBox 尺寸統計:")
    print(f"  最大寬/高: {max_bbox_width} / {max_bbox_height} px")
    print(f"  平均寬/高: {avg_w} / {avg_h} px")
    print(f"  中位數寬/高: {med_w} / {med_h} px")
    if AUTO_CROP:
        print("  裁剪尺寸已自動套用（auto square）")
    else:
        if all_bbox_widths and all_bbox_heights:
            p90_side = max(np.percentile(all_bbox_widths, 90), np.percentile(all_bbox_heights, 90)) * 1.25
            max_side = max(max(all_bbox_widths), max(all_bbox_heights)) * 1.05
            suggested_side = int(np.ceil(max(p90_side, max_side)))
            print(f"  建議 CROP_WIDTH / CROP_HEIGHT: {suggested_side} / {suggested_side}  (auto square)")
        print("  （提示：config 加入 auto_crop: true 可下次自動套用）")


if __name__ == '__main__':
    main()
