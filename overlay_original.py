"""
overlay_original.py

在原始（未裁切）影片上疊加 2D 骨架與起終點標線，支援多相機串接。

公開 API:
    overlay_videos(cameras, offsets_npz, kps_npz, output_video, config=None)
        cameras     : list of dict，每個 dict 須含 'video_path' 與可選的 'start_line'/'end_line'
        offsets_npz : str，track_crop_roi 產出的 .npz 路徑
                      必須含 'offsets' (N,2)、'orig_frames' (N,)、'cam_indices' (N,) 三個 key
        kps_npz     : str，MotionAGFormer 產出的 keypoints.npz 路徑
        output_video: str，輸出影片路徑
        config      : dict or None，若含 'cameras' 欄位則以此為準（覆蓋 cameras 參數的 start/end_line）
"""

import sys
import os
import argparse
import numpy as np
import cv2
import yaml


# ---------------------------------------------------------------------------
# 骨架繪製（H36M 17 關節格式）
# ---------------------------------------------------------------------------
def show2Dpose_original(kps, img, offset_x, offset_y):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)   # Blue
    rcolor = (0, 0, 255)   # Red
    thickness = 3

    for j, c in enumerate(connections):
        start = kps[c[0]]
        end = kps[c[1]]
        sx, sy = int(start[0] + offset_x), int(start[1] + offset_y)
        ex, ey = int(end[0] + offset_x), int(end[1] + offset_y)
        cv2.line(img, (sx, sy), (ex, ey), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (sx, sy), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (ex, ey), thickness=-1, color=(0, 255, 0), radius=3)

    return img


# ---------------------------------------------------------------------------
# 虛線
# ---------------------------------------------------------------------------
def _draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=12, gap_len=8):
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
# 在影格上畫起終點線
# ---------------------------------------------------------------------------
def _draw_lines(frame, start_line, end_line):
    if start_line and end_line:
        p0 = (int(start_line[0][0]), int(start_line[0][1]))
        p3 = (int(start_line[1][0]), int(start_line[1][1]))
        p1 = (int(end_line[0][0]), int(end_line[0][1]))
        p2 = (int(end_line[1][0]), int(end_line[1][1]))
        for a, b in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
            _draw_dashed_line(frame, a, b, (255, 255, 255), thickness=2)
        cv2.line(frame, p0, p3, (0, 0, 0), 5)
        cv2.line(frame, p0, p3, (180, 255, 255), 3)  # yellow
        cv2.line(frame, p1, p2, (0, 0, 0), 5)
        cv2.line(frame, p1, p2, (255, 200, 100), 3)  # light blue
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

    print(f"Creating uncropped overlay video: {output_video} ({len(orig_frames)} frames, {num_cams} camera(s))")
    from tqdm import tqdm

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
                frame = show2Dpose_original(kps, frame, off_x, off_y)

            out.write(frame)
            pbar.update(1)

    if cap is not None:
        cap.release()
    out.release()
    print(f"✅ Uncropped overlay video saved to {output_video}")


# ---------------------------------------------------------------------------
# CLI 入口（保留，以便單獨測試）
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_video',   required=True,
                        help="第一台相機影片路徑（單相機模式，或作為 config 比對用）")
    parser.add_argument('--offsets_npz',  required=True)
    parser.add_argument('--kps_npz',      required=True)
    parser.add_argument('--config_yaml',  default=None)
    parser.add_argument('--config_json',  default=None)
    parser.add_argument('--output_video', required=True)
    args = parser.parse_args()

    if args.config_yaml:
        with open(args.config_yaml, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif args.config_json:
        import json
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


if __name__ == '__main__':
    main()
