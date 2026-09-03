"""
core/overlay.py

在原始（未裁切）影片上疊加 2D 骨架與起終點標線，支援多相機串接。
此模組封裝了原本在根目錄下 overlay_original.py 的核心運算邏輯。
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from core.draw_utils import draw_dashed_line as _draw_dashed_line


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
    # [9, 10] (Neck/Nose -> Head) and [8, 9] (Thorax -> Neck/Nose) intentionally
    # omitted: both keypoints are unreliable for this checkpoint and jitter
    # badly, so they're left undrawn rather than rendering distracting
    # jumping points.
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)   # Blue (左側)
    rcolor = (0, 0, 255)   # Red (右側)
    thickness = 2

    for j, c in enumerate(connections):
        start = kps[c[0]]
        end = kps[c[1]]
        sx, sy = int(start[0] + offset_x), int(start[1] + offset_y)
        ex, ey = int(end[0] + offset_x), int(end[1] + offset_y)
        cv2.line(img, (sx, sy), (ex, ey), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (sx, sy), thickness=-1, color=(0, 255, 0), radius=2)
        cv2.circle(img, (ex, ey), thickness=-1, color=(0, 255, 0), radius=2)

    if foot_kps is not None:
        # Keep foot colors attached to the corrected logical L/R slots.  The
        # pipeline swaps each three-point foot group together with the body
        # legs before rendering, so these colors follow the DP-corrected
        # identity as well.
        LEFT_FOOT_COLOR = (0, 255, 255)    # yellow (BGR)
        RIGHT_FOOT_COLOR = (255, 0, 255)  # magenta (BGR)
        RIGHT_ANKLE, LEFT_ANKLE = 3, 6
        foot_to_ankle = [(0, LEFT_ANKLE), (1, LEFT_ANKLE), (2, LEFT_ANKLE),
                          (3, RIGHT_ANKLE), (4, RIGHT_ANKLE), (5, RIGHT_ANKLE)]
        for fi, ankle_idx in foot_to_ankle:
            if foot_scores is not None and foot_scores[fi] < 0.3:
                continue
            foot_color = LEFT_FOOT_COLOR if fi < 3 else RIGHT_FOOT_COLOR
            fx, fy = int(foot_kps[fi, 0] + offset_x), int(foot_kps[fi, 1] + offset_y)
            ax, ay = int(kps[ankle_idx, 0] + offset_x), int(kps[ankle_idx, 1] + offset_y)
            cv2.line(img, (ax, ay), (fx, fy), foot_color, 2)
            cv2.circle(img, (fx, fy), 4, foot_color, -1)

    return img


# ---------------------------------------------------------------------------
# 在影格上畫起終點線與跑道範圍
# ---------------------------------------------------------------------------
def _draw_lines(frame, start_line, end_line, homography_points=None):
    """在原始影格上繪製起跑線、終點線以及中間包夾的虛線跑道區間。

    4 點線性投影校正的相機有 start_line/end_line，用兩條實線 + 四邊虛線框住
    中間的跑道範圍。6 點 homography 校正的相機沒有這兩條線（沒有對應的兩線
    語意），改成把 homography_points（該相機的 6 個校正點，pixel 座標）依序
    連成一圈虛線多邊形，畫出同樣的「框住跑道範圍」效果，讓兩種校正模式在疊圖
    影片上的視覺呈現一致。
    """
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
    elif homography_points and len(homography_points) >= 2:
        pts = [(int(p[0]), int(p[1])) for p in homography_points]
        for a, b in zip(pts, pts[1:] + [pts[0]]):
            _draw_dashed_line(frame, a, b, (255, 255, 255), thickness=2)


# ---------------------------------------------------------------------------
# 共用：載入 offsets / keypoints / foot npz 與 config 合併
# ---------------------------------------------------------------------------
@dataclass
class _OverlaySources:
    """overlay_videos() 與 overlay_videos_per_camera() 共用的輸入資料。"""
    cameras: list
    offsets: "np.ndarray"
    orig_frames: "np.ndarray"
    cam_indices: "np.ndarray"
    kps_map: dict
    foot_kps_map: dict
    foot_scores_map: dict
    num_cams: int


def _merge_line_config(cameras, config):
    """以 config['cameras'] 的 start_line/end_line 補上各相機（不覆蓋已有值）。"""
    if not (config and 'cameras' in config):
        return cameras
    cfg_cams = config['cameras']
    merged = []
    for i, cam in enumerate(cameras):
        c = dict(cam)
        if i < len(cfg_cams):
            c.setdefault('start_line', cfg_cams[i].get('start_line'))
            c.setdefault('end_line', cfg_cams[i].get('end_line'))
        merged.append(c)
    return merged


def _load_overlay_sources(cameras, offsets_npz, kps_npz, config):
    """讀 offsets / keypoints /（可選）foot npz，建立 v_idx → keypoints 對應表。"""
    cameras = _merge_line_config(cameras, config)

    offsets_data = np.load(offsets_npz)
    offsets = offsets_data['offsets']
    orig_frames = offsets_data['orig_frames']
    if 'cam_indices' in offsets_data:
        cam_indices = offsets_data['cam_indices'].astype(int)
    else:
        print("  ⚠️  offsets.npz 未含 cam_indices，假設所有幀皆來自相機 0")
        cam_indices = np.zeros(len(orig_frames), dtype=int)

    kps_data = np.load(kps_npz, allow_pickle=True)
    keypoints = kps_data['reconstruction'][0]
    valid_frames = np.asarray(kps_data['valid_frames']).flatten().astype(int)
    kps_map = {v: keypoints[i] for i, v in enumerate(valid_frames)
               if v < len(orig_frames)}

    foot_kps_map, foot_scores_map = {}, {}
    foot_npz = os.path.join(os.path.dirname(kps_npz), 'foot_keypoints.npz')
    if os.path.exists(foot_npz):
        foot_data = np.load(foot_npz, allow_pickle=True)
        foot_keypoints = foot_data['keypoints'][0]
        foot_scores = foot_data['scores'][0]
        for i, v in enumerate(valid_frames):
            if v < len(orig_frames) and i < len(foot_keypoints):
                foot_kps_map[v] = foot_keypoints[i]
                foot_scores_map[v] = foot_scores[i]

    num_cams = int(max(cam_indices)) + 1 if len(cam_indices) > 0 else len(cameras)
    return _OverlaySources(cameras, offsets, orig_frames, cam_indices,
                           kps_map, foot_kps_map, foot_scores_map, num_cams)


def _iter_original_frames(sources, wanted_cams=None):
    """依 orig_frames 順序產出 (v_idx, c_idx, frame)。每台相機只開一次
    VideoCapture 並循序讀取（cap.set() 在壓縮影片上定位不準）。
    不在 wanted_cams、或無法開啟/讀取的幀 → frame 為 None。"""
    current_cam_idx = -1
    cap = None
    current_frame_pos = 0
    try:
        for v_idx in range(len(sources.orig_frames)):
            c_idx = int(sources.cam_indices[v_idx])
            orig_idx = int(sources.orig_frames[v_idx])

            if wanted_cams is not None and c_idx not in wanted_cams:
                yield v_idx, c_idx, None
                continue

            if c_idx != current_cam_idx:
                if cap is not None:
                    cap.release()
                    cap = None
                video_path = (sources.cameras[c_idx].get('video_path')
                              if c_idx < len(sources.cameras) else None)
                if not video_path or not os.path.exists(video_path):
                    print(f"  ⚠️  找不到相機 {c_idx} 的影片: {video_path}，跳過")
                    yield v_idx, c_idx, None
                    continue
                cap = cv2.VideoCapture(video_path)
                current_cam_idx = c_idx
                current_frame_pos = 0

            ret = False
            frame = None
            while current_frame_pos <= orig_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame_pos += 1
            yield v_idx, c_idx, (frame if ret else None)
    finally:
        if cap is not None:
            cap.release()


def _draw_skeleton_on_frame(frame, sources, v_idx):
    """在原始幀上畫起終點線與（若有）骨架，回傳更新後的 frame。"""
    if v_idx not in sources.kps_map:
        return frame
    off_x, off_y = sources.offsets[v_idx]
    return show2Dpose_original(
        sources.kps_map[v_idx], frame, off_x, off_y,
        foot_kps=sources.foot_kps_map.get(v_idx),
        foot_scores=sources.foot_scores_map.get(v_idx),
    )


# ---------------------------------------------------------------------------
# 主要公開 API
# ---------------------------------------------------------------------------
def overlay_videos(cameras, offsets_npz, kps_npz, output_video, config=None):
    """
    在各台相機的原始影片上疊加 2D 骨架與起終點標線，輸出為單支合成影片。

    cameras / offsets_npz / kps_npz / output_video 同舊版；config 提供時以其
    start/end_line 覆寫。
    """
    sources = _load_overlay_sources(cameras, offsets_npz, kps_npz, config)

    first_video = sources.cameras[0].get('video_path')
    if not first_video or not os.path.exists(first_video):
        raise FileNotFoundError(f"找不到影片: {first_video}")
    cap0 = cv2.VideoCapture(first_video)
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap0.get(cv2.CAP_PROP_FPS)
    cap0.release()
    if fps <= 0:
        fps = 30.0

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    print(f"[Core.Overlay] 正在輸出原解析度骨架影片: {output_video} "
          f"(幀數: {len(sources.orig_frames)}, 相機數: {sources.num_cams})")

    with tqdm(total=len(sources.orig_frames), desc="Overlaying") as pbar:
        for v_idx, c_idx, frame in _iter_original_frames(sources):
            if frame is not None:
                cam_cfg = sources.cameras[c_idx] if c_idx < len(sources.cameras) else {}
                _draw_lines(frame, cam_cfg.get('start_line'), cam_cfg.get('end_line'),
                            cam_cfg.get('homography_src_points'))
                frame = _draw_skeleton_on_frame(frame, sources, v_idx)
                out.write(frame)
            pbar.update(1)

    out.release()
    print(f"✅ [Core.Overlay] 原影片骨架疊加完成！儲存至: {output_video}")


def _open_per_camera_writers(sources, output_paths):
    """依 output_paths 為每台有效相機建立 VideoWriter（以該相機影片尺寸）。"""
    writers = {}
    for c_idx in range(sources.num_cams):
        out_path = output_paths[c_idx] if c_idx < len(output_paths) else None
        if not out_path:
            continue
        video_path = (sources.cameras[c_idx].get('video_path')
                      if c_idx < len(sources.cameras) else None)
        if not video_path or not os.path.exists(video_path):
            print(f"  ⚠️  找不到相機 {c_idx} 的影片，略過該相機的疊圖輸出: {video_path}")
            continue
        probe = cv2.VideoCapture(video_path)
        w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        probe.release()
        writers[c_idx] = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return writers


def _annotate_landing(frame, row, cam_history, total_steps, contact_display):
    """在一幀上畫腳踝點、過去 20 個落地事件標記、時間/步數文字。"""
    from scripts.analysis.ankle_step_stride import TEXT_COLOR

    if row:
        cv2.circle(frame, (int(row["right_ankle_x"]), int(row["right_ankle_y"])), 3, (0, 0, 255), -1)
        cv2.circle(frame, (int(row["left_ankle_x"]), int(row["left_ankle_y"])), 3, (255, 0, 0), -1)

    for past in cam_history[-20:]:
        # homography_lateral_valid 只在 homography 相機上設定；False = 該點世界座標
        # 被標為離群值 → 不畫。
        if past.get("homography_lateral_valid") is False:
            continue
        px, py, colour, joint_tag = contact_display(past)
        cv2.circle(frame, (px, py), 6, TEXT_COLOR, 2)
        cv2.circle(frame, (px, py), 3, colour, -1)
        label = f"S{past['step_index']} {joint_tag}"
        if past["step_length_m"] is not None:
            label += f" L={past['step_length_m']:.2f}m"
        elif past["step_length_px"] is not None:
            label += f" L={past['step_length_px']:.0f}px"
        label_y = py + 43 if (past['step_index'] % 2 == 1) else py + 70
        cv2.putText(frame, label, (px + 8, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)

    if row:
        cv2.putText(frame, f"Time: {row['seq_time_s']:.2f}s", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Steps: {total_steps}", (30, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)


def overlay_videos_per_camera(cameras, offsets_npz, kps_npz, ankle_rows, step_events,
                              output_paths, config=None):
    """
    同 overlay_videos()（骨架疊圖），但額外疊上落地點標註，且每台相機
    各自輸出一支影片（不拼接）。output_paths 長度需與相機數一致，
    某相機不輸出時該項傳 None。
    """
    from scripts.analysis.ankle_step_stride import _event_contact_display

    sources = _load_overlay_sources(cameras, offsets_npz, kps_npz, config)
    rows_by_seq = {int(r["seq_frame"]): r for r in ankle_rows}
    events_by_seq = {int(e["seq_frame"]): e for e in step_events}
    event_history_by_cam = {}

    writers = _open_per_camera_writers(sources, output_paths)
    print(f"[Core.Overlay] 正在輸出各相機獨立骨架+落地點疊圖影片（相機數: {len(writers)}）")

    with tqdm(total=len(sources.orig_frames), desc="Per-camera overlaying") as pbar:
        for v_idx, c_idx, frame in _iter_original_frames(sources, wanted_cams=set(writers)):
            if frame is not None:
                cam_cfg = sources.cameras[c_idx] if c_idx < len(sources.cameras) else {}
                _draw_lines(frame, cam_cfg.get('start_line'), cam_cfg.get('end_line'),
                            cam_cfg.get('homography_src_points'))
                frame = _draw_skeleton_on_frame(frame, sources, v_idx)

                event = events_by_seq.get(v_idx)
                if event:
                    event_history_by_cam.setdefault(c_idx, []).append(event)
                total_steps = sum(len(h) for h in event_history_by_cam.values())
                _annotate_landing(frame, rows_by_seq.get(v_idx),
                                  event_history_by_cam.get(c_idx, []),
                                  total_steps, _event_contact_display)
                writers[c_idx].write(frame)
            pbar.update(1)

    for w in writers.values():
        w.release()
    print(f"✅ [Core.Overlay] 各相機獨立疊圖完成，共 {len(writers)} 支影片")



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
