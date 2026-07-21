"""
core/overlay.py

在原始（未裁切）影片上疊加 2D 骨架與起終點標線，支援多相機串接。
此模組封裝了原本在根目錄下 overlay_original.py 的核心運算邏輯。
"""

import sys
import os
import argparse
import numpy as np
import cv2
import yaml
import json
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 骨架繪製（H36M 17 關節格式）
# ---------------------------------------------------------------------------
def show2Dpose_original(kps, img, offset_x, offset_y, foot_kps=None, foot_scores=None):
    """
    在原始影格上繪製 H36M 17 關節格式的 2D 骨架。
    使用 offsets 修正回正確的相機原始空間座標。

    foot_kps/foot_scores（可選）：COCO-WholeBody 腳部 6 點，順序為
    L_big_toe, L_small_toe, L_heel, R_big_toe, R_small_toe, R_heel，
    連到 H36M 左(6)/右(3)腳踝，同樣套用 offset 修正回原始空間座標。
    """
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)   # Blue (左側)
    rcolor = (0, 0, 255)   # Red (右側)
    thickness = 3

    for j, c in enumerate(connections):
        start = kps[c[0]]
        end = kps[c[1]]
        sx, sy = int(start[0] + offset_x), int(start[1] + offset_y)
        ex, ey = int(end[0] + offset_x), int(end[1] + offset_y)
        cv2.line(img, (sx, sy), (ex, ey), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (sx, sy), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (ex, ey), thickness=-1, color=(0, 255, 0), radius=3)

    if foot_kps is not None:
        FOOT_COLOR = (0, 255, 255)
        RIGHT_ANKLE, LEFT_ANKLE = 3, 6
        foot_to_ankle = [(0, LEFT_ANKLE), (1, LEFT_ANKLE), (2, LEFT_ANKLE),
                          (3, RIGHT_ANKLE), (4, RIGHT_ANKLE), (5, RIGHT_ANKLE)]
        for fi, ankle_idx in foot_to_ankle:
            if foot_scores is not None and foot_scores[fi] < 0.3:
                continue
            fx, fy = int(foot_kps[fi, 0] + offset_x), int(foot_kps[fi, 1] + offset_y)
            ax, ay = int(kps[ankle_idx, 0] + offset_x), int(kps[ankle_idx, 1] + offset_y)
            cv2.line(img, (ax, ay), (fx, fy), FOOT_COLOR, 2)
            cv2.circle(img, (fx, fy), 4, FOOT_COLOR, -1)

    return img


# ---------------------------------------------------------------------------
# 輔助虛線繪製
# ---------------------------------------------------------------------------
def _draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=12, gap_len=8):
    """在影像上繪製指定長度間隔的虛線。"""
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


# ---------------------------------------------------------------------------
# 在影格上畫起終點線與跑道範圍
# ---------------------------------------------------------------------------
def _draw_lines(frame, start_line, end_line):
    """在原始影格上繪製起跑線、終點線以及中間包夾的虛線跑道區間。"""
    if start_line and end_line:
        p0 = (int(start_line[0][0]), int(start_line[0][1]))
        p3 = (int(start_line[1][0]), int(start_line[1][1]))
        p1 = (int(end_line[0][0]), int(end_line[0][1]))
        p2 = (int(end_line[1][0]), int(end_line[1][1]))
        for a, b in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
            _draw_dashed_line(frame, a, b, (255, 255, 255), thickness=2)
        cv2.line(frame, p0, p3, (0, 0, 0), 5)
        cv2.line(frame, p0, p3, (180, 255, 255), 3)  # 黃色起跑線
        cv2.line(frame, p1, p2, (0, 0, 0), 5)
        cv2.line(frame, p1, p2, (255, 200, 100), 3)  # 天藍色終點線
    elif start_line:
        pt1 = (int(start_line[0][0]), int(start_line[0][1]))
        pt2 = (int(start_line[1][0]), int(start_line[1][1]))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 5)
        cv2.line(frame, pt1, pt2, (180, 255, 255), 3)
    elif end_line:
        pt1 = (int(end_line[0][0]), int(end_line[0][1]))
        pt2 = (int(end_line[1][0]), int(end_line[1][1]))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 5)
        cv2.line(frame, pt1, pt2, (255, 200, 100), 3)


# ---------------------------------------------------------------------------
# 主要公開 API
# ---------------------------------------------------------------------------
def overlay_videos(cameras, offsets_npz, kps_npz, output_video, config=None):
    """
    在各台相機的原始影片上疊加 2D 骨架與起終點標線，輸出為單支合成影片。

    Parameters
    ----------
    cameras      : list[dict]  每個 dict 含 'video_path', 'start_line', 'end_line'
    offsets_npz  : str         npz 路徑（含 offsets / orig_frames / cam_indices）
    kps_npz      : str         keypoints.npz 路徑
    output_video : str         輸出影片路徑
    config       : dict|None   若提供，用 config['cameras'] 的 start/end_line 覆蓋
    """
    # ── 若有額外 config，以 config 的 start/end_line 為準 ──
    if config and 'cameras' in config:
        cfg_cams = config['cameras']
        merged = []
        for i, cam in enumerate(cameras):
            c = dict(cam)
            if i < len(cfg_cams):
                c.setdefault('start_line', cfg_cams[i].get('start_line'))
                c.setdefault('end_line',   cfg_cams[i].get('end_line'))
            merged.append(c)
        cameras = merged

    # ── 讀取 offsets npz ──
    offsets_data = np.load(offsets_npz)
    offsets     = offsets_data['offsets']       # (N, 2)
    orig_frames = offsets_data['orig_frames']   # (N,)   原始影片幀號
    # cam_indices: 向前相容舊版 npz（未含此欄位時預設全部為相機 0）
    if 'cam_indices' in offsets_data:
        cam_indices = offsets_data['cam_indices'].astype(int)  # (N,)
    else:
        print("  ⚠️  offsets.npz 未含 cam_indices，假設所有幀皆來自相機 0")
        cam_indices = np.zeros(len(orig_frames), dtype=int)

    # ── 讀取 keypoints npz ──
    kps_data   = np.load(kps_npz, allow_pickle=True)
    keypoints  = kps_data['reconstruction'][0]         # (valid_frames, 17, 4)
    valid_frames = np.asarray(kps_data['valid_frames']).flatten().astype(int)

    # valid_frames[i] 是裁切影片的第 i 個有效幀在「all_offsets 序列」中的索引
    # 建立 v_idx → keypoints 的對應
    kps_map = {}
    for i, v_idx in enumerate(valid_frames):
        if v_idx < len(orig_frames):
            kps_map[v_idx] = keypoints[i]

    # ── 讀取腳部 keypoints npz（若存在；舊的 17-only checkpoint 輸出不會有這個檔案）──
    foot_kps_map = {}
    foot_scores_map = {}
    foot_npz = os.path.join(os.path.dirname(kps_npz), 'foot_keypoints.npz')
    if os.path.exists(foot_npz):
        foot_data = np.load(foot_npz, allow_pickle=True)
        foot_keypoints = foot_data['keypoints'][0]   # (valid_frames, 6, 2)
        foot_scores = foot_data['scores'][0]         # (valid_frames, 6)
        for i, v_idx in enumerate(valid_frames):
            if v_idx < len(orig_frames) and i < len(foot_keypoints):
                foot_kps_map[v_idx] = foot_keypoints[i]
                foot_scores_map[v_idx] = foot_scores[i]

    # ── 依相機分組，找出每台相機需要哪些幀 ──
    # 每台相機只開啟一次 VideoCapture，依序讀取所需幀
    num_cams = max(cam_indices) + 1 if len(cam_indices) > 0 else len(cameras)

    # 先從第一台相機取得輸出尺寸與 FPS
    first_video = cameras[0].get('video_path')
    if not first_video or not os.path.exists(first_video):
        raise FileNotFoundError(f"找不到影片: {first_video}")

    cap0 = cv2.VideoCapture(first_video)
    width  = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap0.get(cv2.CAP_PROP_FPS)
    cap0.release()
    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    print(f"[Core.Overlay] 正在輸出原解析度骨架影片: {output_video} (幀數: {len(orig_frames)}, 相機數: {num_cams})")

    # ── 逐幀輸出：按 v_idx 順序，切換相機 VideoCapture ──
    current_cam_idx = -1   # 目前已開啟的相機
    cap = None
    current_frame_pos = 0  # 目前相機已讀到的幀號

    with tqdm(total=len(orig_frames), desc="Overlaying") as pbar:
        for v_idx in range(len(orig_frames)):
            c_idx     = int(cam_indices[v_idx])
            orig_idx  = int(orig_frames[v_idx])

            # 切換相機時開新 VideoCapture
            if c_idx != current_cam_idx:
                if cap is not None:
                    cap.release()
                video_path = cameras[c_idx].get('video_path') if c_idx < len(cameras) else None
                if not video_path or not os.path.exists(video_path):
                    print(f"  ⚠️  找不到相機 {c_idx} 的影片: {video_path}，跳過")
                    pbar.update(1)
                    continue
                cap = cv2.VideoCapture(video_path)
                current_cam_idx  = c_idx
                current_frame_pos = 0

            # 依序讀到目標幀（避免 cap.set() 在壓縮影片上定位不準）
            ret = False
            while current_frame_pos <= orig_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame_pos += 1

            if not ret:
                pbar.update(1)
                continue

            # ── 畫起終點線 ──
            cam_cfg = cameras[c_idx] if c_idx < len(cameras) else {}
            start_line = cam_cfg.get('start_line')
            end_line   = cam_cfg.get('end_line')
            _draw_lines(frame, start_line, end_line)

            # ── 畫骨架 ──
            if v_idx in kps_map:
                kps = kps_map[v_idx]
                off_x, off_y = offsets[v_idx]
                frame = show2Dpose_original(
                    kps, frame, off_x, off_y,
                    foot_kps=foot_kps_map.get(v_idx),
                    foot_scores=foot_scores_map.get(v_idx),
                )

            out.write(frame)
            pbar.update(1)

    if cap is not None:
        cap.release()
    out.release()
    print(f"✅ [Core.Overlay] 原影片骨架疊加完成！儲存至: {output_video}")


# ---------------------------------------------------------------------------
# CLI 參數解析器與入口（供 CLI Wrapper 直接導入呼叫）
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_video',   required=True,
                        help="第一台相機影片路徑（單相機模式，或作為 config 比對用）")
    parser.add_argument('--offsets_npz',  required=True)
    parser.add_argument('--kps_npz',      required=True)
    parser.add_argument('--config_yaml',  default=None)
    parser.add_argument('--config_json',  default=None)
    parser.add_argument('--output_video', required=True)
    return parser.parse_args()


def run_cli(args=None):
    if args is None:
        args = parse_args()

    config = None
    if args.config_yaml:
        with open(args.config_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif args.config_json:
        config = json.loads(args.config_json)
    else:
        print("Error: --config_yaml or --config_json is required.")
        sys.exit(1)

    # 從 config 取出相機清單（若沒有就用 orig_video 建一台）
    cfg_cams = config.get('cameras', [])
    if cfg_cams:
        cameras = [
            {
                'video_path': cam.get('video_path'),
                'start_line': cam.get('start_line'),
                'end_line':   cam.get('end_line'),
            }
            for cam in cfg_cams
        ]
    else:
        cameras = [{'video_path': args.orig_video}]

    overlay_videos(cameras, args.offsets_npz, args.kps_npz, args.output_video, config=config)
