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
import resource as _res
try:
    _soft, _hard = _res.getrlimit(_res.RLIMIT_NPROC)
    if _soft < _hard:
        _res.setrlimit(_res.RLIMIT_NPROC, (_hard, _hard))
except Exception:
    pass

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
    cv2.setLogLevel(3)   # 抑制 swscaler 色彩轉換警告
except AttributeError:
    pass
from ultralytics import YOLO
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import argparse
import json
import yaml
from pathlib import Path

# 此腳本位於 scripts/tracking/，往上三層為 repo 根目錄
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# =======================================================================
# 設定區（每次修改只需改這裡）
# =======================================================================

# 使用哪張實體 GPU（'0' = 第 0 張，'1' = 第 1 張）
CUDA_VISIBLE_DEVICES = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
DEVICE = 0

# 模型權重路徑（yolo11x.pt 放在 repo 根目錄；下載方式見 README）
MODEL_PATH = str(BASE_DIR / "yolo11x.pt")

# 中文字型路徑（ROI 標籤使用）
FONT_PATH = str(BASE_DIR / "MotionAGFormer" / "ChineseFont.ttf")

# 輸出目錄（檔名依第一台有效相機自動命名：{輸入檔名}_tracked.mp4）
OUTPUT_DIR = str(BASE_DIR / "output_cut")

# 固定裁剪尺寸（以最快人物中心為基準）
# 建議值：先執行一次看底部「建議 CROP_WIDTH/CROP_HEIGHT」的統計輸出再調整
# 或在 config 加入 auto_crop: true，讓程式自動以中位數 × 2 決定尺寸
CROP_WIDTH  = 200
CROP_HEIGHT = 260
AUTO_CROP   = False  # True → 先 dry-run 收集 bbox 統計，自動設定裁剪尺寸

# 是否在裁剪畫面上疊加框
#   True  = 綠色 bbox（最快跑者）+ 藍色 ROI 框
#   False = 輸出乾淨畫面（無任何疊加）
SHOW_OVERLAY = True

# -----------------------------------------------------------------------
# 移動偵測參數
# -----------------------------------------------------------------------
MOVEMENT_THRESHOLD  = 2   # 判定為移動的最小像素位移,連續兩幀中心位移 > 2px 才算「有移動」
MIN_MOVEMENT_FRAMES = 3   # 需連續移動至少此幀數才視為「真正移動」,必須連續移動 ≥ 3 幀，才被認定為「真的在跑」
STATIONARY_DECAY    = 2   # 靜止時每幀遞減 movement_count 的量, 靜止一幀就讓 movement_count 減 2（快速重置）
MAX_PERSON_MEMORY   = 30  # 超過此幀數未偵測到則清除該人物的速度紀錄, 30 幀（約 0.5 秒）沒出現就刪除


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
           pre_roll_px=200):
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

    return {
        'video_path':  video_path,
        'crop_params': crop,
        'roi_enabled': not no_roi,
        'roi_zones':   zones,
        'switch_x':    switch_x,
        # 斜線模式欄位（舊模式均為 None）
        'start_line':  start_line,
        'end_line':    end_line,
        'start_mid':   start_mid,
        'end_mid':     end_mid,
        'track_dir':   track_dir,
        'pixel_span':  pixel_span,
        'quad_roi':    quad_roi,
        'track_roi':   {'start_mid':   start_mid,
                        'track_dir':   track_dir,
                        'pixel_span':  pixel_span,
                        'pre_roll_px': pre_roll_px}
                       if start_mid is not None and track_dir is not None else None,
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
CAM1 = camera("/home/jeter/MotionAGFormer/0331-1.mp4",
              crop=(0, 400, 1920, 800),
              start_line=[(208, 715), (123, 725)],
              end_line  =[(1760, 710), (1830, 718)])

CAM2 = camera("/home/jeter/MotionAGFormer/IMG_5728.mp4",
              crop=(0, 400, 1920, 800),
              start_line=[(208, 715), (123, 725)],
              end_line  =[(1760, 710), (1830, 718)])

CAM3 = camera("/home/jeter/MotionAGFormer/0331-2.mp4",
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
        except Exception:
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


def process_frame(img, model, velocity_tracker, device,
                  crop_params, roi_enabled, roi_zones,
                  crop_x_offset, crop_y_offset,
                  instant_start=False,
                  track_roi=None,
                  overlay_start_pts=None,
                  overlay_end_pts=None):
    """
    對單幀執行：前處理裁剪 → YOLO track → ROI 過濾 + 速度累積 →
                選最快人物 → 疊加框 → 固定大小裁剪。

    track_roi          dict or None：斜線模式的投影 ROI 參數（含 start_mid/track_dir/pixel_span/pre_roll_px）
    overlay_start_pts  tuple[2] or None：起跑線兩端點（crop 後座標系），SHOW_OVERLAY 時繪製
    overlay_end_pts    tuple[2] or None：終點線兩端點（crop 後座標系），SHOW_OVERLAY 時繪製

    回傳：(crop_frame, fastest_id, fastest_center_orig, fastest_bx2_orig)
      crop_frame=None          → ROI 內無有效人物，上層應跳過此幀
      fastest_center_orig      → 最快人物 center_x（原始座標），非最後一機的切換基準
      fastest_bx2_orig         → 最快人物 bbox 右緣（原始座標），最後一機的退出 ROI 基準
    """
    # Step 1: 前處理裁剪
    if crop_params:
        cx1, cy1, cx2, cy2 = crop_params
        h, w = img.shape[:2]
        cx1, cx2 = max(0, cx1), min(w, cx2)
        cy1, cy2 = max(0, cy1), min(h, cy2)
        if cx2 <= cx1 or cy2 <= cy1:
            return None, None, None, None
        img = img[cy1:cy2, cx1:cx2]

# =======================================================================
#     # Step 2: YOLO track()
#     # conf=0.3：只接受 YOLO 判定為人的分數 ≥ 0.3 的框（可調低以抓更模糊的）
#     # iou=0.1：框重疊超過 10% 就視為同一個（可調高以容許更多重疊）
#     # imgsz=1280：推論解析度（較高 → 偵測小人較準）
#     # verbose=False：關閉 YOLO 的偵測進度 log
# =======================================================================

    results = model.track(img, persist=True, classes=[0], show=False, device=device,
                          conf=0.3, iou=0.1, imgsz=1280, verbose=False)
    r = results[0]


# =======================================================================
# Step 3: 速度累積 + ROI 過濾
# =======================================================================
    
    seen_ids = set()

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()       # shape (N, 4)，格式 x1 y1 x2 y2
        ids   = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None    # shape (N,)，YOLO 分配的 track ID

        for i in range(len(boxes)):
            bx1, by1, bx2, by2 = map(int, boxes[i])
            center_x = (bx1 + bx2) / 2   # bbox 中心 x（裁剪座標）
            center_y = (by1 + by2) / 2   # bbox 中心 y（裁剪座標）

            # ROI 過濾（換算回原始影片座標再比對）
            orig_cx = center_x + crop_x_offset
            orig_cy = center_y + crop_y_offset
            if track_roi is not None:
                # 斜線模式：投影到跑道方向，範圍 [-pre_roll_px, pixel_span]
                proj = _project_onto_track(
                    (orig_cx, orig_cy),
                    track_roi['start_mid'], track_roi['track_dir']
                )
                pre_roll = track_roi.get('pre_roll_px', 0)
                if not (-pre_roll <= proj <= track_roi['pixel_span']):
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
                d['frames_since_detected'] = 0
            else:
                velocity_tracker[tid] = {
                    'center':                (center_x, center_y),
                    'bbox':                  (bx1, by1, bx2, by2),
                    'velocities':            [0],
                    'movement_count':        MIN_MOVEMENT_FRAMES if instant_start else 1,
                    'stationary_count':      0,
                    'frames_since_detected': 0,
                }

    # 清除過期追蹤
    for tid in list(velocity_tracker):
        if tid not in seen_ids:
            velocity_tracker[tid]['frames_since_detected'] += 1
            if velocity_tracker[tid]['frames_since_detected'] > MAX_PERSON_MEMORY:
                del velocity_tracker[tid]

    # Step 4: 選出最快人物
    fastest_id = None
    max_vel = 0
    for tid, d in velocity_tracker.items():
        if d['frames_since_detected'] == 0:  # 本幀有偵測到
            if (d['movement_count'] >= MIN_MOVEMENT_FRAMES and  # 連續移動 ≥ 3 幀
                    d['stationary_count'] < 10):  # 靜止幀數 < 10
                v = np.mean(d['velocities']) if d['velocities'] else 0  # 平均移動距離
                if v > max_vel:
                    max_vel = v
                    fastest_id = tid

    if fastest_id is None:
        return None, None, None, None

    d = velocity_tracker[fastest_id]
    bx1, by1, bx2, by2 = d['bbox']
    fastest_center_orig = (bx1 + bx2) / 2.0 + crop_x_offset
    fastest_bx2_orig    = bx2 + crop_x_offset

    # Step 5: 疊加框（在前處理裁剪後的畫面上）
    if SHOW_OVERLAY:
        # 綠色框 — 最快跑者 bbox
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

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

    # Step 6: 固定大小裁剪（以最快人物 bbox 中心為基準）
    cx = int((bx1 + bx2) / 2)       # 最快人物 bbox 中心 x（裁剪座標系）
    cy = int((by1 + by2) / 2)       # 最快人物 bbox 中心 y（裁剪座標系）

    h_img, w_img = img.shape[:2]    # 裁剪後畫面的高與寬
    c1x = cx - CROP_WIDTH  // 2      # 最終裁剪視窗左上 x
    c1y = cy - CROP_HEIGHT // 2     # 最終裁剪視窗左上 y
    c2x = c1x + CROP_WIDTH          # 最終裁剪視窗右下 x
    c2y = c1y + CROP_HEIGHT         # 最終裁剪視窗右下 y

    # 邊界夾值：超出影片邊界時整體平移，保持裁剪尺寸不變
    if c1x < 0:       c1x, c2x = 0, CROP_WIDTH
    elif c2x > w_img: c2x, c1x = w_img, w_img - CROP_WIDTH
    if c1y < 0:       c1y, c2y = 0, CROP_HEIGHT
    elif c2y > h_img: c2y, c1y = h_img, h_img - CROP_HEIGHT
    c1x = max(0, c1x); c1y = max(0, c1y)
    c2x = min(c2x, w_img); c2y = min(c2y, h_img)

    crop_frame = img[c1y:c2y, c1x:c2x]

    # 若切出尺寸不符（影片本身比 crop 小），強制縮放補齊
    if crop_frame.shape[:2] != (CROP_HEIGHT, CROP_WIDTH):
        if crop_frame.size > 0:
            crop_frame = cv2.resize(crop_frame, (CROP_WIDTH, CROP_HEIGHT),
                                    interpolation=cv2.INTER_LINEAR)
        else:
            crop_frame = np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)

    return crop_frame, fastest_id, fastest_center_orig, fastest_bx2_orig


def _process_cameras(caps, cameras, model, out, dry_run=False):
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

    for cam_idx, (cam, cap) in enumerate(zip(cameras, caps)):
        if not cap.isOpened():
            raise ValueError(f"無法開啟相機 {cam_idx+1}: {cam['video_path']}")

        vid_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not dry_run:
            print(f"{'─'*60}")
            print(f"相機 {cam_idx+1}/{len(cameras)}: {cam['video_path']}")
            print(f"  解析度: {vid_w}x{vid_h}，幀數: {total}，FPS: {fps:.1f}")

        cp = cam['crop_params']
        if cp and not dry_run:
            cx1c, cy1c = max(0, cp[0]), max(0, cp[1])
            cx2c, cy2c = min(vid_w, cp[2]), min(vid_h, cp[3])
            print(f"  CROP: ({cx1c},{cy1c}) → ({cx2c},{cy2c})，"
                  f"裁剪後: {cx2c-cx1c}x{cy2c-cy1c}")

        if not dry_run:
            if cam['roi_enabled'] and not cam['roi_zones']:
                rx, ry = cam['roi_zones'][0]['x'], cam['roi_zones'][0]['y'] if cam['roi_zones'] else (None, None)
                for j, z in enumerate(cam['roi_zones']):
                    print(f"  ROI 區域 {j+1}: X={z['x']}, Y={z['y']}")
            elif cam['roi_enabled']:
                for j, z in enumerate(cam['roi_zones']):
                    print(f"  ROI 區域 {j+1}: X={z['x']}, Y={z['y']}")

        switch_x    = cam.get('switch_x')
        track_roi   = cam.get('track_roi')
        is_last_cam = (cam_idx == len(cameras) - 1)
        if not dry_run:
            if track_roi is not None:
                print(f"  ROI 模式: 斜線投影（pixel_span={cam['pixel_span']:.0f}px，"
                      f"pre_roll={track_roi['pre_roll_px']}px）")
                print(f"  切換條件: 投影距離 ≥ pixel_span（越過終點線）")
            elif switch_x:
                ref_label = 'bx2（右緣）' if is_last_cam else 'center_x'
                print(f"  切換條件: 最快人物 {ref_label}（原始座標）> {switch_x}px")
            else:
                print(f"  切換條件: 跑完整段影片")

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

        velocity_tracker = {}
        _instant_start = (cam_idx > 0)
        cam_skipped = 0
        cam_written = 0
        frame_count = 0

        # 起跑確認狀態（track_roi 模式專用）
        runner_crossed_start = (track_roi is None)  # 舊模式直接視為已越線
        pre_roll_buf  = []   # 已渲染的 crop_frame，最多 5 幀（起跑線前）
        candidate_buf = []   # (crop_frame, proj_px)，待確認起跑的候選幀
        K_CONFIRM     = 3    # 連續幾幀單調遞增才確認起跑

        if not dry_run:
            print(f"  [處理中...]")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            _ov_s = overlay_start_pts if SHOW_OVERLAY else None
            _ov_e = overlay_end_pts   if SHOW_OVERLAY else None
            try:
                crop_frame, fastest_id, fastest_center_orig, fastest_bx2_orig = process_frame(
                    frame, model, velocity_tracker, DEVICE,
                    cam['crop_params'], cam['roi_enabled'], cam['roi_zones'],
                    crop_x_offset, crop_y_offset,
                    instant_start=_instant_start,
                    track_roi=track_roi,
                    overlay_start_pts=_ov_s,
                    overlay_end_pts=_ov_e,
                )
            except RuntimeError:
                torch.cuda.empty_cache()
                crop_frame, fastest_id, fastest_center_orig, fastest_bx2_orig = process_frame(
                    frame, model, velocity_tracker, DEVICE,
                    cam['crop_params'], cam['roi_enabled'], cam['roi_zones'],
                    crop_x_offset, crop_y_offset,
                    instant_start=_instant_start,
                    track_roi=track_roi,
                    overlay_start_pts=_ov_s,
                    overlay_end_pts=_ov_e,
                )

            if crop_frame is None:
                cam_skipped   += 1
                total_skipped += 1
                continue

            # ── 起跑確認邏輯（track_roi 模式且尚未確認越線）──
            if track_roi is not None and not runner_crossed_start:
                d = velocity_tracker.get(fastest_id)
                if d is None:
                    cam_skipped += 1; total_skipped += 1
                    continue
                bx1_c, by1_c, bx2_c, by2_c = d['bbox']
                cx_orig = (bx1_c + bx2_c) / 2.0 + crop_x_offset
                cy_orig = (by1_c + by2_c) / 2.0 + crop_y_offset
                proj_px = _project_onto_track(
                    (cx_orig, cy_orig),
                    cam['start_mid'], cam['track_dir']
                )

                if proj_px < 0:
                    # 在起跑線前：放入 pre-roll buffer，不輸出
                    candidate_buf.clear()
                    pre_roll_buf.append(crop_frame)
                    if len(pre_roll_buf) > 5:
                        pre_roll_buf.pop(0)
                    continue

                else:
                    # proj_px >= 0：進入候選區（需單調遞增才確認起跑）
                    if candidate_buf and proj_px <= candidate_buf[-1][1]:
                        candidate_buf.clear()   # 後退或持平，重新收集

                    candidate_buf.append((crop_frame, proj_px))

                    if len(candidate_buf) >= K_CONFIRM:
                        runner_crossed_start = True
                        # 寫出 pre-roll 幀
                        for pre_f in pre_roll_buf:
                            if not dry_run:
                                out.write(pre_f)
                            cam_written += 1; total_written += 1
                        pre_roll_buf.clear()
                        # 寫出 candidate 幀
                        for c_frame, _ in candidate_buf:
                            if not dry_run:
                                out.write(c_frame)
                            cam_written += 1; total_written += 1
                        candidate_buf.clear()
                        if not dry_run:
                            print(f"  [幀 {frame_count}] 確認起跑（投影={proj_px:.0f}px）")
                    # 無論確認與否，此幀已透過 buffer 處理，不重複輸出
                    continue

            # 正常幀（已越線或舊模式）：收集 bbox 統計
            if fastest_id is not None and fastest_id in velocity_tracker:
                bx1, by1, bx2, by2 = velocity_tracker[fastest_id]['bbox']
                bw = bx2 - bx1; bh = by2 - by1
                all_bbox_widths.append(bw); all_bbox_heights.append(bh)
                if bw > max_bbox_width:  max_bbox_width  = bw
                if bh > max_bbox_height: max_bbox_height = bh

            if not dry_run:
                out.write(crop_frame)
                cam_written  += 1
                total_written += 1
                if frame_count % 100 == 0:
                    print(f"  [幀 {frame_count}/{total}] 追蹤: {len(velocity_tracker)} | "
                          f"寫入: {cam_written} | 捨棄: {cam_skipped}")
            else:
                cam_written  += 1
                total_written += 1

            # ── 切換條件（斜線模式優先）──
            if track_roi is not None and cam.get('pixel_span'):
                # 斜線模式：投影 >= pixel_span → 越過終點線
                d = velocity_tracker.get(fastest_id)
                if d is not None:
                    bx1, by1, bx2, by2 = d['bbox']
                    cy_orig = (by1 + by2) / 2.0 + crop_y_offset
                    ref_x = bx2 + crop_x_offset if is_last_cam else (bx1 + bx2) / 2.0 + crop_x_offset
                    proj_px = _project_onto_track(
                        (ref_x, cy_orig), cam['start_mid'], cam['track_dir']
                    )
                    if proj_px >= cam['pixel_span']:
                        if not dry_run:
                            print(f"  → 觸發{'退出ROI' if is_last_cam else '切換'}："
                                  f"投影={proj_px:.0f}px >= {cam['pixel_span']:.0f}px")
                        break
            elif switch_x is not None and fastest_id is not None:
                # 舊模式：x 座標比較
                trigger = fastest_bx2_orig if is_last_cam else fastest_center_orig
                if trigger is not None and trigger > switch_x:
                    if not dry_run:
                        ref_name = 'bx2' if is_last_cam else 'center_x'
                        print(f"  → 觸發{'退出ROI' if is_last_cam else '切換'}："
                              f"{ref_name}={trigger:.0f} > {switch_x}")
                    break

        cap.release()
        if not dry_run:
            print(f"  相機 {cam_idx+1} 完成：寫入 {cam_written}，捨棄 {cam_skipped}")
            if track_roi is not None and not runner_crossed_start:
                print(f"  ⚠️  警告：整段影片跑者從未越過起跑線，本機輸出 0 幀")

    return total_written, total_skipped, all_bbox_widths, all_bbox_heights, max_bbox_width, max_bbox_height


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
        global OUTPUT_DIR, CROP_WIDTH, CROP_HEIGHT, AUTO_CROP, SHOW_OVERLAY, MOVEMENT_THRESHOLD, MIN_MOVEMENT_FRAMES, STATIONARY_DECAY, MAX_PERSON_MEMORY
        if 'output_dir'          in _cfg: OUTPUT_DIR         = _cfg['output_dir']
        if 'crop_width'          in _cfg: CROP_WIDTH          = int(_cfg['crop_width'])
        if 'crop_height'         in _cfg: CROP_HEIGHT         = int(_cfg['crop_height'])
        if 'auto_crop'           in _cfg: AUTO_CROP           = bool(_cfg['auto_crop'])
        if 'show_overlay'        in _cfg: SHOW_OVERLAY        = bool(_cfg['show_overlay'])
        if 'movement_threshold'  in _cfg: MOVEMENT_THRESHOLD  = int(_cfg['movement_threshold'])
        if 'min_movement_frames' in _cfg: MIN_MOVEMENT_FRAMES = int(_cfg['min_movement_frames'])
        if 'stationary_decay'    in _cfg: STATIONARY_DECAY    = int(_cfg['stationary_decay'])
        if 'max_person_memory'   in _cfg: MAX_PERSON_MEMORY   = int(_cfg['max_person_memory'])

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

    # 寫入 marker 檔，供 run_pipeline.py 讀取實際輸出名稱
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, ".last_output_name"), "w") as _f:
        _f.write(OUTPUT_NAME)

    first_fps = caps[0].get(cv2.CAP_PROP_FPS) or 60.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

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
    # auto_crop：第一遍 dry-run 收集 bbox 統計，自動決定裁剪尺寸
    # -----------------------------------------------------------------------
    if AUTO_CROP:
        print("auto_crop 模式：第一遍掃描（分析 bbox 尺寸，不寫影片）...")
        _, _, dry_bw, dry_bh, _, _ = _process_cameras(caps, CAMERAS, model, None, dry_run=True)
        # dry-run 已讀完所有影片，重新開啟
        caps = [cv2.VideoCapture(cam['video_path']) for cam in CAMERAS]
        if dry_bw and dry_bh:
            CROP_WIDTH  = int(np.median(dry_bw)) * 2
            CROP_HEIGHT = int(np.median(dry_bh)) * 2
            print(f"  自動設定裁剪尺寸: {CROP_WIDTH} x {CROP_HEIGHT}（中位數 × 2）\n")
        else:
            print("  警告：未收集到 bbox 資料，沿用預設尺寸\n")

    print(f"輸出路徑: {output_path}")
    out = cv2.VideoWriter(output_path, fourcc, first_fps, (CROP_WIDTH, CROP_HEIGHT))

    # -----------------------------------------------------------------------
    # 正式處理：逐台相機串接追蹤 + 寫入影片
    # -----------------------------------------------------------------------
    total_written, total_skipped, all_bbox_widths, all_bbox_heights, max_bbox_width, max_bbox_height = \
        _process_cameras(caps, CAMERAS, model, out)

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
    print(f"\nBBox 尺寸統計:")
    print(f"  最大寬/高: {max_bbox_width} / {max_bbox_height} px")
    print(f"  平均寬/高: {avg_w} / {avg_h} px")
    print(f"  中位數寬/高: {med_w} / {med_h} px")
    if AUTO_CROP:
        print(f"  裁剪尺寸已自動套用（中位數 × 2）")
    else:
        print(f"  建議 CROP_WIDTH / CROP_HEIGHT: {med_w*2} / {med_h*2}  (中位數 × 2)")
        print(f"  （提示：config 加入 auto_crop: true 可下次自動套用）")


if __name__ == '__main__':
    main()
