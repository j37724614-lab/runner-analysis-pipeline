"""產生左右腿 DP 修正的診斷 sheet，畫法依 docs/developer/leg_swap_correction.md
的「診斷 sheet 固定畫法」規格，並額外疊上腳部（腳趾/腳跟）點。

輸入來源（皆為 core.pipeline.run_analysis() 的標準輸出檔案）：
  - 原始影片（未裁切，乾淨無疊圖）
  - <pose_dir>/input_2D/keypoints.npz          body 17 點（H36M 順序，裁切影片座標系）
  - <pose_dir>/input_2D/foot_keypoints.npz     foot 6 點（COCO-WholeBody 順序，同一座標系）
  - <pose_dir>/input_2D/post_dp_ankle_fill_mask.npz   dp_leg_swapped / post_dp_leg_fill_mask
  - <output_dest>/<stem>_offsets.npz           每幀 (offset_x, offset_y)：cropped + offset = original
  - <output_dest>/<stem>_step_events.csv       觸地事件（TD 標籤用）

輸出：<out_dir>/<prefix>_sheet_seq_000_019.jpg, ..._020_039.jpg, ...
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

# H36M 17 joints
CONNECTIONS = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
               [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
               [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
HEAD_JOINT = 10

BLUE_LEFT = (255, 0, 0)
RED_RIGHT = (0, 0, 255)
GREEN_TRUNK = (0, 255, 0)
BLACK = (0, 0, 0)

FOOT_COLOR = (0, 255, 255)
RIGHT_ANKLE, LEFT_ANKLE = 3, 6
FOOT_TO_ANKLE = [(0, LEFT_ANKLE), (1, LEFT_ANKLE), (2, LEFT_ANKLE),
                 (3, RIGHT_ANKLE), (4, RIGHT_ANKLE), (5, RIGHT_ANKLE)]

CELL_W, CELL_H = 240, 180
GRID_COLS, GRID_ROWS = 5, 4
FRAMES_PER_SHEET = GRID_COLS * GRID_ROWS


def draw_skeleton(img, kps, ox, oy, foot_kps=None):
    pts = kps[:, :2] + np.array([ox, oy])

    for j, c in enumerate(CONNECTIONS):
        sx, sy = int(pts[c[0], 0]), int(pts[c[0], 1])
        ex, ey = int(pts[c[1], 0]), int(pts[c[1], 1])
        color = GREEN_TRUNK if c[0] in (0, 7, 8, 9, 10) and c[1] in (0, 7, 8, 9, 10) else (
            BLUE_LEFT if LR[j] else RED_RIGHT)
        cv2.line(img, (sx, sy), (ex, ey), BLACK, 3)
        cv2.line(img, (sx, sy), (ex, ey), color, 2)

    trunk_head = {0, 7, 8, 9}
    for i in range(17):
        x, y = int(pts[i, 0]), int(pts[i, 1])
        color = GREEN_TRUNK if i in trunk_head or i == HEAD_JOINT else (BLUE_LEFT if i in (4, 5, 6, 11, 12, 13) else RED_RIGHT)
        if i == HEAD_JOINT:
            d = 6
            diamond = np.array([[x, y - d], [x + d, y], [x, y + d], [x - d, y]], dtype=np.int32)
            cv2.fillConvexPoly(img, diamond, BLACK)
            d2 = 4
            diamond2 = np.array([[x, y - d2], [x + d2, y], [x, y + d2], [x - d2, y]], dtype=np.int32)
            cv2.fillConvexPoly(img, diamond2, GREEN_TRUNK)
        else:
            cv2.circle(img, (x, y), 5, BLACK, -1)
            cv2.circle(img, (x, y), 3, color, -1)

    if foot_kps is not None:
        fpts = foot_kps + np.array([ox, oy])
        for fi, ankle_idx in FOOT_TO_ANKLE:
            fx, fy = int(fpts[fi, 0]), int(fpts[fi, 1])
            ax, ay = int(pts[ankle_idx, 0]), int(pts[ankle_idx, 1])
            cv2.line(img, (ax, ay), (fx, fy), FOOT_COLOR, 2)
            cv2.circle(img, (fx, fy), 4, FOOT_COLOR, -1)

    return img


def draw_label(img, text, x, y, bg_color, fg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.4, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x, y), (x + tw + 6, y + th + 6), bg_color, -1)
    cv2.putText(img, text, (x + 3, y + th + 2), font, scale, fg_color, thick, cv2.LINE_AA)
    return x + tw + 6 + 2


def draw_touchdown_marker(img, cx, cy):
    half = 6
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), BLACK, 3)
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (0, 255, 255), 1)


def make_cell(frame, kps, foot_kps, ox, oy, seq_idx, dp_swapped, fixed, td_label):
    pts = kps[:, :2] + np.array([ox, oy])
    all_pts = [pts]
    if foot_kps is not None:
        all_pts.append(foot_kps + np.array([ox, oy]))
    all_pts = np.concatenate(all_pts, axis=0)

    pad = 60
    x1 = int(max(0, all_pts[:, 0].min() - pad))
    y1 = int(max(0, all_pts[:, 1].min() - pad))
    x2 = int(min(frame.shape[1], all_pts[:, 0].max() + pad))
    y2 = int(min(frame.shape[0], all_pts[:, 1].max() + pad))
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]

    canvas = frame.copy()
    canvas = draw_skeleton(canvas, kps, ox, oy, foot_kps)
    if td_label is not None:
        tx, ty = int(pts[RIGHT_ANKLE if 'R' in td_label else LEFT_ANKLE, 0]), \
                  int(pts[RIGHT_ANKLE if 'R' in td_label else LEFT_ANKLE, 1])
        draw_touchdown_marker(canvas, tx, ty)

    crop = canvas[y1:y2, x1:x2]
    if crop.size == 0:
        crop = canvas
    cell = cv2.resize(crop, (CELL_W, CELL_H), interpolation=cv2.INTER_LINEAR)

    lx = 2
    ly = 2
    lx2 = draw_label(cell, f"seq {seq_idx}", lx, ly, (255, 255, 255), (0, 0, 0))
    if td_label is not None:
        lx2 = draw_label(cell, td_label, lx2, ly, (255, 180, 220), (0, 0, 0))
    if dp_swapped:
        lx2 = draw_label(cell, "DP", lx2, ly, (255, 120, 0), (255, 255, 255))
    if fixed:
        lx2 = draw_label(cell, "FIX", lx2, ly, (0, 220, 255), (0, 0, 0))

    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='原始未裁切影片路徑')
    ap.add_argument('--pose-dir', required=True, help='...sequential_tracked/ 目錄')
    ap.add_argument('--offsets-npz', required=True)
    ap.add_argument('--step-events-csv', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--prefix', default='newalgo_fullflow_with_feet')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    kps_data = np.load(os.path.join(args.pose_dir, 'input_2D', 'keypoints.npz'), allow_pickle=True)
    body = kps_data['reconstruction'][0]
    valid_frames = np.asarray(kps_data['valid_frames']).flatten().astype(int)

    foot_path = os.path.join(args.pose_dir, 'input_2D', 'foot_keypoints.npz')
    foot_all = np.load(foot_path, allow_pickle=True)['keypoints'][0] if os.path.exists(foot_path) else None

    fill_path = os.path.join(args.pose_dir, 'input_2D', 'post_dp_ankle_fill_mask.npz')
    if os.path.exists(fill_path):
        fill_data = np.load(fill_path, allow_pickle=True)
        dp_swapped_arr = fill_data['dp_leg_swapped']
        leg_fill_mask = fill_data['post_dp_leg_fill_mask']
    else:
        dp_swapped_arr = np.zeros(len(valid_frames), dtype=bool)
        leg_fill_mask = np.zeros((len(valid_frames), 17), dtype=bool)

    offsets_data = np.load(args.offsets_npz)
    offsets = offsets_data['offsets']
    orig_frames = offsets_data['orig_frames']

    step_events = pd.read_csv(args.step_events_csv)
    td_by_seq = {}
    for _, row in step_events.iterrows():
        foot_side = 'R' if str(row['foot']).lower().startswith('r') else 'L'
        td_by_seq[int(row['seq_frame'])] = f"TD S{int(row['step_index'])} {foot_side}"

    cap = cv2.VideoCapture(args.video)

    T = len(valid_frames)
    cells = []
    for i in range(T):
        v_idx = valid_frames[i]
        if v_idx >= len(orig_frames):
            continue
        orig_idx = int(orig_frames[v_idx])
        ox, oy = offsets[v_idx]

        cap.set(cv2.CAP_PROP_POS_FRAMES, orig_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        foot_kps = foot_all[i] if foot_all is not None else None
        dp_swapped = bool(dp_swapped_arr[i]) if i < len(dp_swapped_arr) else False
        fixed = bool(leg_fill_mask[i].any()) if i < len(leg_fill_mask) else False
        td_label = td_by_seq.get(i)

        cell = make_cell(frame, body[i], foot_kps, ox, oy, i, dp_swapped, fixed, td_label)
        cells.append(cell)

    cap.release()

    n_sheets = (len(cells) + FRAMES_PER_SHEET - 1) // FRAMES_PER_SHEET
    for s in range(n_sheets):
        start = s * FRAMES_PER_SHEET
        end = min(start + FRAMES_PER_SHEET, len(cells))
        sheet = np.full((CELL_H * GRID_ROWS, CELL_W * GRID_COLS, 3), 40, dtype=np.uint8)
        for k in range(start, end):
            local = k - start
            r, c = divmod(local, GRID_COLS)
            sheet[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = cells[k]
        out_path = os.path.join(
            args.out_dir, f"{args.prefix}_sheet_seq_{start:03d}_{end - 1:03d}.jpg")
        cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print('saved', out_path)

    print(f'total frames={len(cells)} sheets={n_sheets}')
    print('dp_swapped=' + ','.join(str(i) for i in range(len(cells)) if i < len(dp_swapped_arr) and dp_swapped_arr[i]))


if __name__ == '__main__':
    main()
