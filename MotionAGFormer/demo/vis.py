"""
vis.py — 2D/3D 姿態估計 + 關節角度計算

可作為模組 import（推薦），也可從命令列直接執行：
  import 用法：
    from demo.vis import run_pose_estimation
    run_pose_estimation(video_path="/path/to/video.mp4",
                        output_dir="/path/to/output/",
                        only_2d=False,
                        gpu="0")

  CLI 用法（向下相容）：
    python vis.py --video sample_video.mp4 --gpu 0
    python vis.py --video sample_video.mp4 --2d_only

注意：此腳本以 MotionAGFormer/ 為 repo 根目錄，
      所有路徑（模型、config）均從 __file__ 推算，不依賴工作目錄。
"""

import sys
import argparse
import json
from pathlib import Path
import os
import time
from datetime import datetime

# vis.py 位於 MotionAGFormer/demo/vis.py
# 需要兩個目錄同時在 sys.path：
#   demo/  → from lib.preprocess / from lib.hrnet（lib/ 在 demo/ 下）
#   MotionAGFormer/  → from demo.lib.utils / from model.MotionAGFormer
_DEMO_DIR = Path(__file__).resolve().parent          # MotionAGFormer/demo/
_REPO_ROOT = _DEMO_DIR.parent                        # MotionAGFormer/

for _p in [str(_REPO_ROOT), str(_DEMO_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import glob
import shutil
from tqdm import tqdm
import copy
import yaml

from demo.lib.utils import normalize_screen_coordinates, camera_to_world
from model.MotionAGFormer import MotionAGFormer

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


H36M_JOINT_NAMES = [
    'Hip',
    'RHip', 'RKnee', 'RAnkle',
    'LHip', 'LKnee', 'LAnkle',
    'Spine', 'Thorax',
    'Neck_Nose', 'Head',
    'LShoulder', 'LElbow', 'LWrist',
    'RShoulder', 'RElbow', 'RWrist',
]

# Per-joint motion allowance for H36M 17-joint order. Core joints are stricter;
# ankles and wrists can move faster but are still bounded to suppress drift.
_JOINT_SPEED_H36M = np.array([
    0.7,  # 0: Pelvis
    0.8,  # 1: R.Hip
    1.0,  # 2: R.Knee
    1.2,  # 3: R.Ankle
    0.8,  # 4: L.Hip
    1.0,  # 5: L.Knee
    1.2,  # 6: L.Ankle
    0.7,  # 7: Spine
    0.7,  # 8: Thorax
    0.9,  # 9: Nose
    0.9,  # 10: Head
    0.8,  # 11: L.Shoulder
    1.0,  # 12: L.Elbow
    1.2,  # 13: L.Wrist
    0.8,  # 14: R.Shoulder
    1.0,  # 15: R.Elbow
    1.2,  # 16: R.Wrist
], dtype=np.float32)

# Per-joint displacement threshold (px) — from joint_motion_stats_from_raw_archive.csv.
# Most joints use p95; knees use p90 to reject large visual misdetections earlier.
_JOINT_MAX_PX = np.array([
    10.35, 11.76, 19.62, 63.90,  # Hip, RHip, RKnee, RAnkle
    15.35, 25.38, 63.49,          # LHip, LKnee, LAnkle
    12.49, 14.23, 12.10, 12.92,  # Spine, Thorax, Neck_Nose, Head
    17.77, 36.43, 50.29,          # LShoulder, LElbow, LWrist
    15.14, 35.21, 50.69,          # RShoulder, RElbow, RWrist
], dtype=np.float32)

# Per-joint p99 displacement (px) — kept as a looser reference threshold.
_JOINT_HARD_PX = np.array([
    16.75, 17.47, 55.29, 80.52,  # Hip, RHip, RKnee, RAnkle
    23.64, 55.38, 80.70,          # LHip, LKnee, LAnkle
    20.80, 26.58, 20.26, 24.67,  # Spine, Thorax, Neck_Nose, Head
    29.21, 57.39, 72.75,          # LShoulder, LElbow, LWrist
    25.34, 53.99, 72.80,          # RShoulder, RElbow, RWrist
], dtype=np.float32)

# Per-joint Savitzky-Golay window. Torso and knees get stronger smoothing
# because they are the most visible drift sources in the current data.
_SG_WINDOW_BY_JOINT = np.array([
    15,        # 0  Hip
    11, 7, 3,  # 1  RHip, RKnee, RAnkle
    11, 7, 3,  # 4  LHip, LKnee, LAnkle
    15, 15, 15, 15,  # 7  Spine, Thorax, Neck_Nose, Head
    13, 15, 13,  # 11 LShoulder, LElbow, LWrist
    13, 15, 13,  # 14 RShoulder, RElbow, RWrist
], dtype=np.int32)

# H36M kinematic tree root-to-leaf (parent, child) — for bone-length normalization.
_BONES_H36M = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

# (proximal, joint, distal, min_deg) — prevents hyperextension in 2D projection.
_JOINT_ANGLE_LIMITS_2D = [
    (1, 2, 3, 20),    # RKnee
    (4, 5, 6, 20),    # LKnee
    (11, 12, 13, 15), # LElbow
    (14, 15, 16, 15), # RElbow
]

_ARCHIVE_POINTER_FILENAME = "keypoints_raw_archive_dir.txt"


def _reset_dir(path):
    """清掉同名輸出資料夾，避免混到上一次不同尺寸或不同幀數的 PNG。"""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _archive_raw_keypoints(video_path, output_dir, raw_keypoints_with_conf, valid_frames):
    """Save a non-overwriting copy of raw HRNet keypoints and its source video."""
    archive_root = Path(
        os.getenv(
            "KEYPOINT_RAW_ARCHIVE_DIR",
            str(_REPO_ROOT.parent / "keypoints_raw_archive"),
        )
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    video_stem = Path(video_path).stem
    archive_dir = archive_root / f"{timestamp}_{video_stem}"
    archive_dir.mkdir(parents=True, exist_ok=False)

    archive_npz = archive_dir / "keypoints_raw.npz"
    np.savez_compressed(
        archive_npz,
        reconstruction=raw_keypoints_with_conf,
        valid_frames=valid_frames,
    )

    source_video_copy = None
    source_video_original_name_copy = None
    source_video_error = None
    video_path_obj = Path(video_path)
    if video_path_obj.exists():
        source_video_copy = archive_dir / f"source_video{video_path_obj.suffix}"
        source_video_original_name_copy = archive_dir / video_path_obj.name
        try:
            shutil.copy2(video_path_obj, source_video_copy)
            if source_video_original_name_copy != source_video_copy:
                shutil.copy2(video_path_obj, source_video_original_name_copy)
        except OSError as exc:
            source_video_error = str(exc)
            source_video_copy = None
            source_video_original_name_copy = None
    else:
        source_video_error = f"source video not found: {video_path}"

    metadata = {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "created_at": timestamp,
        "keypoints_shape": list(raw_keypoints_with_conf.shape),
        "source_video_copy": str(source_video_copy) if source_video_copy else None,
        "source_video_original_name_copy": (
            str(source_video_original_name_copy)
            if source_video_original_name_copy
            else None
        ),
        "source_video_error": source_video_error,
    }
    with open(archive_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return archive_npz


def _remember_keypoints_archive_dir(output_dir, archive_npz):
    """Record the archive folder so final mp4 outputs can be copied there later."""
    pointer_path = Path(output_dir) / "input_2D" / _ARCHIVE_POINTER_FILENAME
    pointer_path.write_text(str(Path(archive_npz).parent), encoding="utf-8")


def _final_video_candidates(video_path, output_dir):
    video_name = Path(video_path).stem
    return [
        Path(output_dir) / f"{video_name}.mp4",
        Path(output_dir) / f"{video_name}_2D.mp4",
    ]


def _export_final_video_frames(video_path, output_dir):
    """Export each generated final mp4 frame as PNG under output_dir/final_video_frames."""
    frame_root = Path(output_dir) / "final_video_frames"
    exported_dirs = []

    for src in _final_video_candidates(video_path, output_dir):
        if not src.exists():
            continue

        target_dir = frame_root / src.stem
        _reset_dir(target_dir)

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            print(f"Warning: cannot open final video for PNG export: {src}")
            continue

        frame_idx = 0
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out_path = target_dir / f"{frame_idx:06d}.png"
            if cv2.imwrite(str(out_path), frame):
                written += 1
            frame_idx += 1

        cap.release()
        exported_dirs.append(target_dir)
        print(f"Final video PNG frames saved to {target_dir} ({written} frames)")

    return exported_dirs


def _copy_final_videos_to_keypoints_archive(video_path, output_dir):
    """Copy generated pose videos into the same folder as keypoints_raw.npz."""
    pointer_path = Path(output_dir) / "input_2D" / _ARCHIVE_POINTER_FILENAME
    if not pointer_path.exists():
        return []

    archive_dir = Path(pointer_path.read_text(encoding="utf-8").strip())
    if not archive_dir.exists():
        return []

    copied = []
    for src in _final_video_candidates(video_path, output_dir):
        if not src.exists():
            continue
        dst = archive_dir / f"final_{src.name}"
        shutil.copy2(src, dst)
        copied.append(dst)

    if copied:
        final_video_metadata = {
            "final_videos": [str(path) for path in copied],
        }
        with open(archive_dir / "final_videos.json", "w", encoding="utf-8") as f:
            json.dump(final_video_metadata, f, ensure_ascii=False, indent=2)

    return copied


def _interpolate_low_conf_keypoints(keypoints, scores, conf_threshold=0.50):
    """
    用前後高信心幀線性插值補低信心 keypoints。

    這是離線影片後處理，所以可以看見未來幀。只有在低信心片段前後
    都有高信心錨點時才補，避免影片開頭/結尾缺少一側錨點時硬猜位置。
    回傳的 interpolated_mask 會告訴 EMA：這些點雖然原始 HRNet confidence 低，
    但位置已由前後可信點補過，可以跟著更新而不是停在上一幀。
    """
    if keypoints.size == 0 or scores.size == 0:
        return keypoints, np.zeros(scores.shape, dtype=bool)

    filled = keypoints.astype(np.float32, copy=True)
    interpolated_mask = np.zeros(scores.shape, dtype=bool)
    num_person, num_frames, num_joints = scores.shape

    for person_idx in range(num_person):
        for joint_idx in range(num_joints):
            good = scores[person_idx, :, joint_idx] >= conf_threshold
            good_idx = np.flatnonzero(good)
            if good_idx.size < 2:
                continue

            first_good = int(good_idx[0])
            last_good = int(good_idx[-1])
            fill_range = np.arange(first_good, last_good + 1)
            missing = ~good[first_good:last_good + 1]
            if not np.any(missing):
                continue

            for coord_idx in range(2):
                filled[person_idx, fill_range, joint_idx, coord_idx] = np.interp(
                    fill_range,
                    good_idx,
                    keypoints[person_idx, good_idx, joint_idx, coord_idx],
                )
            interpolated_mask[person_idx, fill_range[missing], joint_idx] = True

    return filled, interpolated_mask


def _load_bbox_heights(bbox_csv, num_frames, num_persons=1):
    """
    Read per-frame bbox heights from track_crop_roi.py bbox_map.csv.

    Missing frames are forward-filled then backward-filled. The reference height
    is the median valid bbox height, so jump limits scale with the runner size.
    """
    import csv as _csv

    heights = np.zeros((num_persons, num_frames), dtype=np.float32)
    if not bbox_csv or not os.path.exists(bbox_csv):
        return None, None

    with open(bbox_csv, newline='') as f:
        for row in _csv.DictReader(f):
            fi = int(row['output_frame'])
            if 0 <= fi < num_frames:
                heights[0, fi] = max(float(row['y2']) - float(row['y1']), 0.0)

    valid = heights[heights > 0]
    if valid.size == 0:
        return None, None
    ref_height = float(np.median(valid))

    for person_idx in range(num_persons):
        h = heights[person_idx]
        last = 0.0
        for frame_idx in range(num_frames):
            if h[frame_idx] > 0:
                last = h[frame_idx]
            elif last > 0:
                h[frame_idx] = last
        last = 0.0
        for frame_idx in range(num_frames - 1, -1, -1):
            if h[frame_idx] > 0:
                last = h[frame_idx]
            elif last > 0:
                h[frame_idx] = last

    return heights, ref_height


def _smooth_keypoints_ema(keypoints, scores, alpha=0.45, conf_threshold=0.35,
                          max_jump_px=45.0, hard_jump_px=90.0,
                          interpolated_mask=None, bbox_heights=None,
                          bbox_ref_height=None, trust_low_conf=0.35,
                          trust_full_conf=0.60, return_debug=False):
    """
    對 HRNet 2D keypoints 做自適應時間平滑。

    keypoints: shape (M, T, J, 2)
    scores:    shape (M, T, J)

    - bbox height 將像素跳點限制正規化到人體當前大小。
    - H36M per-joint speed factor 讓核心關節更穩、末梢關節允許較快移動。
    - velocity prediction 用上一幀速度預測合理位置，避免連續快動作被誤判。
    - confidence trust 使用連續權重，避免 0.49/0.51 造成硬切。
    """
    if keypoints.size == 0 or scores.size == 0:
        return (keypoints, []) if return_debug else keypoints

    smoothed = keypoints.astype(np.float32, copy=True)
    num_person, num_frames, num_joints = scores.shape
    if interpolated_mask is None:
        interpolated_mask = np.zeros(scores.shape, dtype=bool)

    if num_joints <= len(_JOINT_SPEED_H36M):
        joint_speed = _JOINT_SPEED_H36M[:num_joints].astype(np.float32)
    else:
        extra = np.ones(num_joints - len(_JOINT_SPEED_H36M), dtype=np.float32)
        joint_speed = np.concatenate([_JOINT_SPEED_H36M, extra])

    debug_rows = []
    trust_span = max(trust_full_conf - trust_low_conf, 1e-6)

    for person_idx in range(num_person):
        prev = None
        velocity = None
        for frame_idx in range(num_frames):
            current = smoothed[person_idx, frame_idx]
            current_scores = scores[person_idx, frame_idx]

            if prev is None:
                prev = current.copy()
                velocity = np.zeros_like(prev, dtype=np.float32)
                smoothed[person_idx, frame_idx] = prev
                if return_debug:
                    for joint_idx in range(num_joints):
                        debug_rows.append({
                            'person': person_idx,
                            'frame': frame_idx,
                            'joint': H36M_JOINT_NAMES[joint_idx] if joint_idx < len(H36M_JOINT_NAMES) else str(joint_idx),
                            'confidence': float(current_scores[joint_idx]),
                            'raw_x': float(current[joint_idx, 0]),
                            'raw_y': float(current[joint_idx, 1]),
                            'smoothed_x': float(prev[joint_idx, 0]),
                            'smoothed_y': float(prev[joint_idx, 1]),
                            'trust': 1.0,
                            'jump': 0.0,
                            'joint_max': 0.0,
                            'joint_hard': 0.0,
                            'bbox_height': 0.0,
                            'bbox_scale': 1.0,
                            'update_type': 'init',
                        })
                continue

            if bbox_heights is not None and bbox_ref_height and bbox_ref_height > 0:
                bbox_height = max(float(bbox_heights[person_idx, frame_idx]), 1.0)
                scale = np.clip(bbox_height / bbox_ref_height, 0.6, 1.8)
            else:
                bbox_height = 0.0
                scale = 1.0

            joint_max = max_jump_px * scale * joint_speed
            joint_hard = hard_jump_px * scale * joint_speed

            vel_mag = np.linalg.norm(velocity, axis=1, keepdims=True)
            velocity = np.where(
                vel_mag > joint_max[:, None],
                velocity / np.maximum(vel_mag, 1e-6) * joint_max[:, None],
                velocity,
            )
            predicted = prev + velocity

            diff = current - predicted
            jump = np.linalg.norm(diff, axis=1)

            trust = np.clip((current_scores - trust_low_conf) / trust_span, 0.0, 1.0)
            is_interp = interpolated_mask[person_idx, frame_idx]
            trust = np.where(is_interp, np.maximum(trust, 0.5), trust)
            w_cur = trust * (1.0 - alpha)

            safe_dist = np.maximum(jump, 1e-6)
            capped = predicted + diff / safe_dist[:, None] * joint_max[:, None]

            normal_update = jump <= joint_max
            limited_update = (jump > joint_max) & (jump <= joint_hard)
            outlier_update = jump > joint_hard

            target = np.where(
                normal_update[:, None],
                current,
                np.where(limited_update[:, None], capped, prev),
            )

            low_trust = w_cur <= 1e-6
            new_prev = (1.0 - w_cur[:, None]) * prev + w_cur[:, None] * target
            final_step = new_prev - prev
            final_step_mag = np.linalg.norm(final_step, axis=1, keepdims=True)
            final_step_cap = joint_max[:, None] * 0.80
            final_step_limited = final_step_mag > final_step_cap
            new_prev = np.where(
                final_step_limited,
                prev + final_step / np.maximum(final_step_mag, 1e-6) * final_step_cap,
                new_prev,
            )

            displacement = new_prev - prev
            new_velocity = 0.5 * velocity + 0.5 * displacement
            new_velocity[outlier_update] *= 0.25
            new_velocity[low_trust] *= 0.5

            if return_debug:
                update_type = np.full(num_joints, 'normal', dtype=object)
                update_type[limited_update] = 'limited'
                update_type[outlier_update] = 'outlier_hold'
                update_type[low_trust] = 'low_trust_hold'
                update_type[is_interp & ~outlier_update & ~limited_update & ~low_trust] = 'interpolated'
                update_type[final_step_limited[:, 0] & ~outlier_update & ~low_trust] = 'final_step_limited'
                for joint_idx in range(num_joints):
                    debug_rows.append({
                        'person': person_idx,
                        'frame': frame_idx,
                        'joint': H36M_JOINT_NAMES[joint_idx] if joint_idx < len(H36M_JOINT_NAMES) else str(joint_idx),
                        'confidence': float(current_scores[joint_idx]),
                        'raw_x': float(current[joint_idx, 0]),
                        'raw_y': float(current[joint_idx, 1]),
                        'smoothed_x': float(new_prev[joint_idx, 0]),
                        'smoothed_y': float(new_prev[joint_idx, 1]),
                        'trust': float(trust[joint_idx]),
                        'jump': float(jump[joint_idx]),
                        'joint_max': float(joint_max[joint_idx]),
                        'joint_hard': float(joint_hard[joint_idx]),
                        'bbox_height': float(bbox_height),
                        'bbox_scale': float(scale),
                        'update_type': update_type[joint_idx],
                    })

            prev = new_prev
            velocity = new_velocity
            smoothed[person_idx, frame_idx] = prev

    return (smoothed, debug_rows) if return_debug else smoothed



def _correct_arm_swaps(kp, lookback=7, threshold=0.75):
    """
    Detect frames where HRNet swapped anatomical L/R arm labels and correct them.

    For each frame, compare current arm vectors (shoulder→elbow/wrist) against a
    causal reference built from the last `lookback` already-corrected frames.
    If swapping labels reduces the mismatch by more than (1-threshold), apply the swap.

    This preserves actual detected motion — only the label assignment is fixed,
    so the skeleton follows real movement without any interpolation lag.

    threshold=0.75 means: swap only when swap_cost < 75% of no_swap_cost.
    """
    L_SH, L_EL, L_WR = 11, 12, 13
    R_SH, R_EL, R_WR = 14, 15, 16

    T = kp.shape[0]
    result = kp.copy()

    for t in range(lookback, T):
        ref = result[t - lookback : t]   # corrected recent history

        l_el_ref = np.median(ref[:, L_EL] - ref[:, L_SH], axis=0)
        l_wr_ref = np.median(ref[:, L_WR] - ref[:, L_SH], axis=0)
        r_el_ref = np.median(ref[:, R_EL] - ref[:, R_SH], axis=0)
        r_wr_ref = np.median(ref[:, R_WR] - ref[:, R_SH], axis=0)

        l_el = kp[t, L_EL] - kp[t, L_SH]
        l_wr = kp[t, L_WR] - kp[t, L_SH]
        r_el = kp[t, R_EL] - kp[t, R_SH]
        r_wr = kp[t, R_WR] - kp[t, R_SH]

        cost_no_swap = (np.linalg.norm(l_el - l_el_ref) + np.linalg.norm(l_wr - l_wr_ref) +
                        np.linalg.norm(r_el - r_el_ref) + np.linalg.norm(r_wr - r_wr_ref))
        cost_swap    = (np.linalg.norm(r_el - l_el_ref) + np.linalg.norm(r_wr - l_wr_ref) +
                        np.linalg.norm(l_el - r_el_ref) + np.linalg.norm(l_wr - r_wr_ref))

        if cost_swap < cost_no_swap * threshold:
            result[t, L_SH] = kp[t, R_SH]
            result[t, L_EL] = kp[t, R_EL]
            result[t, L_WR] = kp[t, R_WR]
            result[t, R_SH] = kp[t, L_SH]
            result[t, R_EL] = kp[t, L_EL]
            result[t, R_WR] = kp[t, L_WR]

    return result


def _fill_left_arm_by_mirror(kp, final_bad_all):
    """
    For bad left-arm frames, estimate position from mirrored right arm.

    In side-view running, arm swing is anti-phase: the LShoulder→LElbow vector
    is the horizontal mirror of RShoulder→RElbow (flip x, keep y).
    Only fills when LShoulder, RShoulder, and the right counterpart are all good.

    Modifies kp in-place; clears bad flag for filled frames.
    Returns (kp, final_bad_all, mirror_filled) where mirror_filled is (T, J) bool.
    """
    ARM_PAIRS = [(12, 15), (13, 16)]  # (LElbow, RElbow), (LWrist, RWrist)
    L_SHOULDER, R_SHOULDER = 11, 14

    T, J = final_bad_all.shape
    mirror_filled = np.zeros((T, J), dtype=bool)

    for l_jnt, r_jnt in ARM_PAIRS:
        for t in np.where(final_bad_all[:, l_jnt])[0]:
            if (final_bad_all[t, r_jnt] or
                    final_bad_all[t, L_SHOULDER] or
                    final_bad_all[t, R_SHOULDER]):
                continue
            ls = kp[t, L_SHOULDER]
            rs = kp[t, R_SHOULDER]
            rv = kp[t, r_jnt] - rs       # right arm vector
            kp[t, l_jnt, 0] = ls[0] - rv[0]  # flip x (anti-phase)
            kp[t, l_jnt, 1] = ls[1] + rv[1]  # keep y
            final_bad_all[t, l_jnt] = False
            mirror_filled[t, l_jnt] = True

    return kp, final_bad_all, mirror_filled


def _smooth_keypoints_sg(keypoints, scores,
                          conf_threshold=0.50,
                          sg_window=7,
                          sg_polyorder=2,
                          frame_bad_joint_threshold=8,
                          bbox_heights=None,
                          bbox_ref_height=None,
                          return_debug=False):
    """
    離線平滑：confidence 過濾 → 雙側孤立點偵測 → 線性插值 → Savitzky-Golay。

    Phase 1: 低信心幀標記（conf < conf_threshold）
    Phase 2: 雙側孤立點偵測 — frame t 若與前後幀距離均超過 _JOINT_MAX_PX（p95），
             且前後幀彼此距離較小，則視為孤立跳點
    Phase 3: 若同一幀壞點過多，整幀視為不可靠，避免半套錯骨架留下來
    Phase 4: np.interp 線性插值補壞點
    Phase 5: Savitzky-Golay 平滑整段軌跡

    keypoints: (M, T, J, 2)
    scores:    (M, T, J)
    """
    from scipy.signal import savgol_filter

    if keypoints.size == 0 or scores.size == 0:
        return (keypoints, []) if return_debug else keypoints

    smoothed = keypoints.astype(np.float32, copy=True)
    M, T, J = scores.shape
    debug_rows = []

    for person_idx in range(M):
        smoothed[person_idx] = _correct_arm_swaps(smoothed[person_idx])
        kp = smoothed[person_idx]   # (T, J, 2)
        sc = scores[person_idx]     # (T, J)

        # Per-frame bbox scale (T,)
        if bbox_heights is not None and bbox_ref_height and bbox_ref_height > 0:
            bh    = np.maximum(bbox_heights[person_idx], 1.0)
            scale = np.clip(bh / bbox_ref_height, 0.6, 1.8)
        else:
            scale = np.ones(T, dtype=np.float32)

        low_conf_all = sc < conf_threshold
        bilateral_all = np.zeros((T, J), dtype=bool)
        thresh_all = np.zeros((T, J), dtype=np.float32)
        jumps_all = np.zeros((T, J), dtype=np.float32)

        for joint_idx in range(J):
            traj_x = kp[:, joint_idx, 0]
            traj_y = kp[:, joint_idx, 1]

            max_px = float(_JOINT_MAX_PX[joint_idx]) if joint_idx < len(_JOINT_MAX_PX) else 45.0
            thresh = max_px * scale
            thresh_all[:, joint_idx] = thresh

            x_prev = np.concatenate([[traj_x[0]], traj_x[:-1]])
            x_next = np.concatenate([traj_x[1:], [traj_x[-1]]])
            y_prev = np.concatenate([[traj_y[0]], traj_y[:-1]])
            y_next = np.concatenate([traj_y[1:], [traj_y[-1]]])

            d_prev = np.hypot(traj_x - x_prev, traj_y - y_prev)
            d_next = np.hypot(traj_x - x_next, traj_y - y_next)
            d_span = np.hypot(x_next - x_prev, y_next - y_prev)

            bilateral = (d_prev > thresh) & (d_next > thresh) & (d_span < d_prev) & (d_span < d_next)
            bilateral[0] = False
            bilateral[-1] = False
            bilateral_all[:, joint_idx] = bilateral
            jumps_all[:, joint_idx] = d_prev

        point_bad_all = low_conf_all | bilateral_all
        frame_bad_count = np.sum(point_bad_all, axis=1)
        frame_bad = frame_bad_count >= int(frame_bad_joint_threshold)

        final_bad_all = low_conf_all | bilateral_all | frame_bad[:, None]
        kp, final_bad_all, mirror_filled_all = _fill_left_arm_by_mirror(kp, final_bad_all)

        for joint_idx in range(J):
            traj_x = kp[:, joint_idx, 0].copy()
            traj_y = kp[:, joint_idx, 1].copy()
            conf   = sc[:, joint_idx]

            # Phase 1/2/3: point-level bad mask plus frame-level fallback.
            bad_mask = low_conf_all[:, joint_idx]
            bilateral = bilateral_all[:, joint_idx]
            thresh = thresh_all[:, joint_idx]
            final_bad = final_bad_all[:, joint_idx]

            # Phase 4: interpolate over bad spans
            good_idx = np.where(~final_bad)[0]
            if len(good_idx) == 0:
                if return_debug:
                    for t in range(T):
                        jname = H36M_JOINT_NAMES[joint_idx] if joint_idx < len(H36M_JOINT_NAMES) else str(joint_idx)
                        debug_rows.append({'person': person_idx, 'frame': t, 'joint': jname,
                                           'confidence': float(conf[t]),
                                           'raw_x': float(traj_x[t]), 'raw_y': float(traj_y[t]),
                                           'smoothed_x': float(traj_x[t]), 'smoothed_y': float(traj_y[t]),
                                           'jump': 0.0, 'threshold': float(thresh[t]),
                                           'frame_bad_count': int(frame_bad_count[t]),
                                           'update_type': 'all_bad'})
                continue

            all_idx      = np.arange(T)
            traj_x_fixed = np.interp(all_idx, good_idx, traj_x[good_idx])
            traj_y_fixed = np.interp(all_idx, good_idx, traj_y[good_idx])

            # Phase 5: Savitzky-Golay smoothing
            win = int(_SG_WINDOW_BY_JOINT[joint_idx]) if joint_idx < len(_SG_WINDOW_BY_JOINT) else sg_window
            win = min(win, T)
            if win % 2 == 0:
                win -= 1
            if win >= sg_polyorder + 2 and T >= win:
                traj_x_smooth = savgol_filter(traj_x_fixed, win, sg_polyorder, mode='mirror')
                traj_y_smooth = savgol_filter(traj_y_fixed, win, sg_polyorder, mode='mirror')
            else:
                traj_x_smooth = traj_x_fixed
                traj_y_smooth = traj_y_fixed

            smoothed[person_idx, :, joint_idx, 0] = traj_x_smooth
            smoothed[person_idx, :, joint_idx, 1] = traj_y_smooth

            if return_debug:
                mirror_filled = mirror_filled_all[:, joint_idx]
                update_types = np.where(
                    mirror_filled, 'mirror_arm',
                    np.where(frame_bad, 'frame_bad',
                             np.where(bad_mask, 'low_conf',
                                      np.where(bilateral, 'bilateral_outlier', 'good')))
                )
                jumps = jumps_all[:, joint_idx]
                for t in range(T):
                    jname = H36M_JOINT_NAMES[joint_idx] if joint_idx < len(H36M_JOINT_NAMES) else str(joint_idx)
                    debug_rows.append({
                        'person':      person_idx,
                        'frame':       t,
                        'joint':       jname,
                        'confidence':  float(conf[t]),
                        'raw_x':       float(traj_x[t]),
                        'raw_y':       float(traj_y[t]),
                        'smoothed_x':  float(traj_x_smooth[t]),
                        'smoothed_y':  float(traj_y_smooth[t]),
                        'jump':        float(jumps[t]),
                        'threshold':   float(thresh[t]),
                        'frame_bad_count': int(frame_bad_count[t]),
                        'update_type': str(update_types[t]),
                    })

    return (smoothed, debug_rows) if return_debug else smoothed


def _save_hrnet_confidence_csv(scores, valid_frames, output_csv):
    """將 HRNet/H36M 17 個關節的 confidence 另存成 CSV，方便檢查飄點來源。"""
    if scores.size == 0:
        return

    rows = []
    for person_idx in range(scores.shape[0]):
        frames = valid_frames[person_idx] if person_idx < len(valid_frames) else None
        for frame_idx in range(scores.shape[1]):
            source_frame = int(frames[frame_idx]) if frames is not None and frame_idx < len(frames) else frame_idx
            row = {
                'person': person_idx,
                'frame': frame_idx,
                'source_frame': source_frame,
            }
            for joint_idx, joint_name in enumerate(H36M_JOINT_NAMES):
                row[joint_name] = float(scores[person_idx, frame_idx, joint_idx])
            rows.append(row)

    pd.DataFrame(rows).to_csv(output_csv, index=False)


def _normalize_bone_lengths_2d(keypoints, blend=0.8):
    """Soft-constrain each bone to its per-video median length (root-to-leaf).
    Only adjusts frames deviating >10%. blend=0.8 means 80% pull toward target.
    """
    M, T, J, C = keypoints.shape
    result = keypoints.copy()
    for person_idx in range(M):
        kp = result[person_idx]
        for parent_idx, child_idx in _BONES_H36M:
            diff = kp[:, child_idx, :2] - kp[:, parent_idx, :2]
            lengths = np.linalg.norm(diff, axis=1)
            valid = lengths > 1.0
            if valid.sum() < 2:
                continue
            ref_len = float(np.median(lengths[valid]))
            ratios = np.where(lengths > 1.0, ref_len / lengths, 1.0)
            adjust = np.abs(ratios - 1.0) > 0.10
            if not np.any(adjust):
                continue
            normalized = kp[:, parent_idx, :2] + diff * ratios[:, None]
            kp[adjust, child_idx, :2] = (
                (1.0 - blend) * kp[adjust, child_idx, :2] + blend * normalized[adjust]
            )
    return result


def _apply_anatomical_limits_2d(keypoints):
    """Prevent impossible 2D joint angles (below anatomical minimum).
    Rotates the distal joint around the joint centre to enforce the minimum angle.
    """
    M, T, J, _ = keypoints.shape
    result = keypoints.copy()
    for person_idx in range(M):
        kp = result[person_idx]
        for p_idx, j_idx, d_idx, min_deg in _JOINT_ANGLE_LIMITS_2D:
            p = kp[:, p_idx, :2]
            j = kp[:, j_idx, :2]
            d = kp[:, d_idx, :2]
            v1 = p - j
            v2 = d - j
            len1 = np.linalg.norm(v1, axis=1)
            len2 = np.linalg.norm(v2, axis=1)
            valid = (len1 > 1.0) & (len2 > 1.0)
            cos_a = np.einsum('ti,ti->t', v1, v2) / np.maximum(len1 * len2, 1e-8)
            angles = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
            needs_fix = valid & (angles < min_deg)
            if not np.any(needs_fix):
                continue
            for t in np.where(needs_fix)[0]:
                delta = np.radians(float(min_deg) - float(angles[t]))
                cross = v1[t, 0] * v2[t, 1] - v1[t, 1] * v2[t, 0]
                if cross < 0:
                    delta = -delta
                cos_d, sin_d = np.cos(delta), np.sin(delta)
                v2r = np.array([
                    v2[t, 0] * cos_d - v2[t, 1] * sin_d,
                    v2[t, 0] * sin_d + v2[t, 1] * cos_d,
                ])
                kp[t, d_idx, :2] = j[t] + v2r
    return result


def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


def get_pose2D(video_path, output_dir, bbox_csv=None):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    print('\nGenerating 2D pose...')
    if bbox_csv:
        print(f'Using external bbox CSV: {bbox_csv}')
    keypoints, scores = hrnet_pose(
        video_path,
        det_dim=416,
        num_peroson=1,
        gen_output=True,
        bbox_csv=bbox_csv,
    )
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    raw_keypoints = keypoints.copy()
    raw_scores = scores.copy()
    bbox_heights = None
    bbox_ref_height = None
    if bbox_csv and keypoints.shape[0] > 0:
        bbox_heights, bbox_ref_height = _load_bbox_heights(
            bbox_csv,
            keypoints.shape[1],
            keypoints.shape[0],
        )
        if bbox_ref_height:
            print(f'Adaptive bbox reference height: {bbox_ref_height:.1f}px')

    keypoints, smoothing_debug_rows = _smooth_keypoints_sg(
        keypoints,
        scores,
        conf_threshold=0.50,
        sg_window=7,
        sg_polyorder=2,
        bbox_heights=bbox_heights,
        bbox_ref_height=bbox_ref_height,
        return_debug=True,
    )

    keypoints = _normalize_bone_lengths_2d(keypoints)
    keypoints = _apply_anatomical_limits_2d(keypoints)

    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_2d_dir = os.path.join(output_dir, 'input_2D')
    _reset_dir(output_2d_dir)

    output_npz = os.path.join(output_2d_dir, 'keypoints.npz')
    np.savez_compressed(output_npz, reconstruction=keypoints, valid_frames=valid_frames)
    raw_output_npz = os.path.join(output_2d_dir, 'keypoints_raw.npz')
    raw_keypoints_with_conf = np.concatenate((raw_keypoints, raw_scores[..., None]), axis=-1)
    np.savez_compressed(
        raw_output_npz,
        reconstruction=raw_keypoints_with_conf,
        valid_frames=valid_frames,
    )
    print(f'Raw HRNet keypoints saved to {raw_output_npz}')
    archived_raw_npz = _archive_raw_keypoints(
        video_path,
        output_dir,
        raw_keypoints_with_conf,
        valid_frames,
    )
    _remember_keypoints_archive_dir(output_dir, archived_raw_npz)
    print(f'Raw HRNet keypoints archived to {archived_raw_npz}')
    confidence_csv = os.path.join(output_dir, 'hrnet_confidence.csv')
    _save_hrnet_confidence_csv(scores, valid_frames, confidence_csv)
    print(f'HRNet confidence CSV saved to {confidence_csv}')
    if smoothing_debug_rows:
        debug_csv = os.path.join(output_dir, 'keypoint_smoothing_debug.csv')
        pd.DataFrame(smoothing_debug_rows).to_csv(debug_csv, index=False)
        print(f'Keypoint smoothing debug CSV saved to {debug_csv}')


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    pose_dir   = os.path.join(output_dir, 'pose')
    pose2d_dir = os.path.join(output_dir, 'pose2D')
    names_pose   = sorted(glob.glob(os.path.join(pose_dir,   '*.png')))
    names_pose2d = sorted(glob.glob(os.path.join(pose2d_dir, '*.png')))

    video_name = os.path.basename(video_path).split('.')[0]

    # 2D-only 影片
    if names_pose2d:
        img = cv2.imread(names_pose2d[0])
        h, w = img.shape[:2]
        size = (w // 2 * 2, h // 2 * 2)
        out_path = os.path.join(output_dir, video_name + '_2D.mp4')
        videoWrite = cv2.VideoWriter(out_path, fourcc, fps, size)
        for name in names_pose2d:
            img = cv2.imread(name)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != size:
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            videoWrite.write(img)
        videoWrite.release()
        print(f"2D Video saved to {out_path}")

    # 2D + 3D 並排影片
    if names_pose:
        img = cv2.imread(names_pose[0])
        h, w = img.shape[:2]
        size = (w // 2 * 2, h // 2 * 2)
        out_path = os.path.join(output_dir, video_name + '.mp4')
        videoWrite = cv2.VideoWriter(out_path, fourcc, fps, size)
        for name in names_pose:
            img = cv2.imread(name)
            if img is None:
                continue
            if (img.shape[1], img.shape[0]) != size:
                img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            videoWrite.write(img)
        videoWrite.release()
        print(f"Pose video saved to {out_path}")


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    ax.imshow(img)


def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    downsample_indices = []

    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        _, unique_idx = np.unique(new_indices, return_index=True)
        downsample_indices.append(unique_idx)
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length < 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                _, unique_idx = np.unique(new_indices, return_index=True)
                downsample_indices.append(unique_idx)
            else:
                clips.append(keypoints_clip)
                downsample_indices.append(np.arange(243))

    return clips, downsample_indices


def turn_into_h36m(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 11, :]
    new_keypoints[..., 2, :] = keypoints[..., 13, :]
    new_keypoints[..., 3, :] = keypoints[..., 15, :]
    new_keypoints[..., 4, :] = keypoints[..., 12, :]
    new_keypoints[..., 5, :] = keypoints[..., 14, :]
    new_keypoints[..., 6, :] = keypoints[..., 16, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 6, :]
    new_keypoints[..., 12, :] = keypoints[..., 8, :]
    new_keypoints[..., 13, :] = keypoints[..., 10, :]
    new_keypoints[..., 14, :] = keypoints[..., 5, :]
    new_keypoints[..., 15, :] = keypoints[..., 7, :]
    new_keypoints[..., 16, :] = keypoints[..., 9, :]
    return new_keypoints


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data


# ===============================
# 角度計算工具
# ===============================
def _angle_between(v1, v2):
    """計算兩個向量之間的夾角（度數）"""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    cos = np.dot(v1, v2) / (norm1 * norm2)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _fix_supplementary_angle_flips(angles, max_delta=45.0, improvement_margin=15.0):
    """
    Correct isolated 180-angle flips while preserving normal knee motion.

    MotionAGFormer sometimes puts the knee on the opposite 3D side for a few
    frames. The unsigned arccos angle then becomes the supplementary angle.
    We only flip when 180-angle is much closer to the local temporal trend.
    """
    arr = np.asarray(angles, dtype=np.float32).copy()
    if arr.size < 3:
        return arr, []

    corrected = arr.copy()
    corrections = []

    for idx in range(arr.size):
        neighbor_values = []
        for nidx in (idx - 2, idx - 1, idx + 1, idx + 2):
            if 0 <= nidx < arr.size and np.isfinite(corrected[nidx]):
                neighbor_values.append(float(corrected[nidx]))
        if len(neighbor_values) < 2 or not np.isfinite(arr[idx]):
            continue

        local_ref = float(np.median(neighbor_values))
        raw = float(arr[idx])
        supplementary = 180.0 - raw
        raw_error = abs(raw - local_ref)
        supp_error = abs(supplementary - local_ref)

        if raw_error > max_delta and supp_error + improvement_margin < raw_error:
            corrected[idx] = supplementary
            corrections.append({
                'frame': int(idx),
                'raw_angle': raw,
                'corrected_angle': supplementary,
                'local_reference': local_ref,
                'raw_error': raw_error,
                'corrected_error': supp_error,
            })

    return corrected, corrections


def _smooth_angles_butterworth(df, fps, cutoff_hz=10.0, order=4):
    """Zero-phase Butterworth low-pass filter applied to all angle columns in df."""
    from scipy.signal import butter, filtfilt
    nyq = fps / 2.0
    if cutoff_hz >= nyq:
        return
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    min_samples = 3 * max(len(a), len(b))
    angle_cols = [c for c in df.columns if c != 'frame']
    for col in angle_cols:
        arr = df[col].to_numpy(dtype=np.float64)
        if len(arr) >= min_samples:
            df[col] = filtfilt(b, a, arr)


def _smooth_angles_sg(df, fps, polyorder=3):
    """FPS-adaptive Savitzky-Golay smoothing on angle columns.
    Fits a local polynomial over ~0.27s window. Preserves peaks better than
    Butterworth at equivalent bandwidth because the passband is flat.
    Effective BW ≈ 6 Hz at both 30fps and 60fps.
    """
    from scipy.signal import savgol_filter
    win = max(polyorder + 2, int(fps * 0.22))
    if win % 2 == 0:
        win += 1
    angle_cols = [c for c in df.columns if c != 'frame']
    for col in angle_cols:
        arr = df[col].to_numpy(dtype=np.float64)
        if len(arr) >= win:
            df[col] = savgol_filter(arr, win, polyorder, mode='mirror')


def compute_angles(npz_path, output_dir, fps=30.0):
    """
    讀取 3D keypoints npz，逐幀計算各關節角度，並存成 CSV。

    H36M 17-joint index:
      0: Pelvis, 1: RHip, 2: RKnee, 3: RAnkle,
      4: LHip, 5: LKnee, 6: LAnkle,
      7: Spine, 8: Thorax, 9: Neck/Nose, 10: Head,
      11: LShoulder, 12: LElbow, 13: LWrist,
      14: RShoulder, 15: RElbow, 16: RWrist
    """
    data = np.load(npz_path)
    poses = data['pred_3d']   # shape: (num_frames, 17, 3)
    num_frames = poses.shape[0]
    results = []

    for i in range(num_frames):
        pose = poses[i]

        left_knee           = _angle_between(pose[4] - pose[5], pose[6] - pose[5])
        left_hip            = _angle_between(pose[11] - pose[4], pose[5] - pose[4])
        right_knee          = _angle_between(pose[1] - pose[2], pose[3] - pose[2])
        right_hip           = _angle_between(pose[14] - pose[1], pose[2] - pose[1])
        left_arm_torso      = _angle_between(pose[4] - pose[11], pose[12] - pose[11])
        left_elbow_flexion  = _angle_between(pose[13] - pose[12], pose[11] - pose[12])
        right_arm_torso     = _angle_between(pose[1] - pose[14], pose[15] - pose[14])
        right_elbow_flexion = _angle_between(pose[16] - pose[15], pose[14] - pose[15])
        left_shoulder_flex  = _angle_between(pose[0] - pose[8], pose[12] - pose[8])
        right_shoulder_flex = _angle_between(pose[0] - pose[8], pose[15] - pose[8])
        vertical            = np.array([0, -1, 0])
        torso_vec           = pose[8] - pose[0]
        pelvis_torso_angle  = _angle_between(vertical, torso_vec)

        results.append([
            i,
            left_knee, left_hip,
            right_knee, right_hip,
            left_arm_torso, left_elbow_flexion,
            right_arm_torso, right_elbow_flexion,
            left_shoulder_flex, right_shoulder_flex,
            pelvis_torso_angle,
        ])

    columns = [
        'frame',
        'left_knee_angle', 'left_hip_angle',
        'right_knee_angle', 'right_hip_angle',
        'left_arm_torso_angle', 'left_elbow_flexion_angle',
        'right_arm_torso_angle', 'right_elbow_flexion_angle',
        'left_shoulder_flexion', 'right_shoulder_flexion',
        'pelvis_torso_angle',
    ]

    df = pd.DataFrame(results, columns=columns)
    knee_corrections = []
    for col in ('left_knee_angle', 'right_knee_angle'):
        fixed, corrections = _fix_supplementary_angle_flips(df[col].to_numpy())
        df[col] = fixed
        for correction in corrections:
            correction['joint'] = col
            knee_corrections.append(correction)

    _smooth_angles_sg(df, fps)

    out_dir = os.path.join(output_dir, 'pred_3D', 'angles')
    os.makedirs(out_dir, exist_ok=True)
    video_name = os.path.basename(os.path.normpath(output_dir))
    out_csv = os.path.join(out_dir, f'{video_name}_angles.csv')
    df.to_csv(out_csv, index=False)
    if knee_corrections:
        debug_csv = os.path.join(out_dir, f'{video_name}_knee_angle_corrections.csv')
        pd.DataFrame(knee_corrections).to_csv(debug_csv, index=False)
        print(f'Knee supplementary angle corrections saved to {debug_csv}')
    print(f'\n✅ Angle computation done. Saved to {out_csv}')
    return out_csv


@torch.no_grad()
def get_pose3D(video_path, output_dir, skip_video=False):
    # 從此腳本所在位置推算 config 絕對路徑，不受工作目錄影響
    config_path = str(_REPO_ROOT / "configs" / "h36m" / "MotionAGFormer-large.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 處理 config 參數型態
    if 'act_layer' in config and isinstance(config['act_layer'], str):
        if config['act_layer'].lower() == 'gelu':
            config['act_layer'] = nn.GELU
        elif config['act_layer'].lower() == 'relu':
            config['act_layer'] = nn.ReLU

    for k, v in config.items():
        if isinstance(v, str):
            if v.lower() == 'true':
                config[k] = True
            elif v.lower() == 'false':
                config[k] = False

    # 過濾掉非模型 __init__ 參數
    import inspect
    model_keys = set(inspect.signature(MotionAGFormer.__init__).parameters.keys()) - {'self'}
    model_config = {k: v for k, v in config.items() if k in model_keys}

    model = nn.DataParallel(MotionAGFormer(**model_config)).cuda()

    # 從此腳本所在位置推算 checkpoint 絕對路徑
    checkpoint_dir = str(_REPO_ROOT / "checkpoint")
    model_path = sorted(glob.glob(os.path.join(checkpoint_dir, 'motionagformer-l-h36m.pth.tr')))[0]

    pre_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()

    ## input
    input_2d_dir = os.path.join(output_dir, 'input_2D')
    data = np.load(os.path.join(input_2d_dir, 'keypoints.npz'), allow_pickle=True)
    keypoints   = data['reconstruction']
    valid_frames = np.asarray(data['valid_frames']).flatten().astype(int)

    clips, downsample_indices = turn_into_clips(keypoints)

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0
    ret, first_img = cap.read()
    if not ret:
        cap.release()
        print("Error: Could not read video file.")
        return
    img_size = first_img.shape

    valid_frame_count = keypoints.shape[1]

    ## 2D pose image
    if not skip_video:
        print('\nGenerating 2D pose image...')
        output_dir_2D = os.path.join(output_dir, 'pose2D')
        os.makedirs(output_dir_2D, exist_ok=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for i in tqdm(range(valid_frame_count)):
            frame_idx = valid_frames[i] if i < len(valid_frames) else i
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, img = cap.read()
            if img is None:
                continue
            input_2D_raw = keypoints[0][i]
            image = show2Dpose(input_2D_raw, copy.deepcopy(img))
            cv2.imwrite(os.path.join(output_dir_2D, str(('%04d' % i)) + '_2D.png'), image)

    cap.release()

    ## 3D pose
    print('\nGenerating 3D pose...')
    all_poses_all_clips = []

    for idx, (clip, d_idx) in enumerate(zip(clips, downsample_indices)):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0])
        input_2D_aug = flip_data(input_2D)

        input_2D     = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        output_3D_non_flip = model(input_2D)
        output_3D_flip     = flip_data(model(input_2D_aug))
        output_3D          = (output_3D_non_flip + output_3D_flip) / 2
        output_3D          = output_3D[:, d_idx]

        post_out_clip = output_3D[0].cpu().detach().numpy()
        all_poses_all_clips.append(post_out_clip)

    post_out_all = np.concatenate(all_poses_all_clips, axis=0)
    print(f"Total processed frames: {post_out_all.shape[0]}")

    output_dir_3D_npz = os.path.join(output_dir, 'pred_3D')
    os.makedirs(output_dir_3D_npz, exist_ok=True)
    npz_out_path = os.path.join(output_dir_3D_npz, '3Dkeypoints.npz')
    np.savez_compressed(npz_out_path, pred_3d=post_out_all)

    # 角度計算（整合在此，不需外部再呼叫）
    compute_angles(npz_out_path, output_dir, fps=video_fps)

    ## 3D pose images
    if not skip_video:
        output_dir_3D = os.path.join(output_dir, 'pose3D')
        os.makedirs(output_dir_3D, exist_ok=True)

        for j, post_out in enumerate(tqdm(post_out_all, desc="Saving 3D images")):
            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            post_out /= max_value

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, ax)

            plt.savefig(os.path.join(output_dir_3D, str(('%04d' % j)) + '_3D.png'),
                        dpi=200, format='png', bbox_inches='tight')
            plt.close(fig)

    print('Generating 3D pose successful!')

    if not skip_video:
        ## 2D + 3D 並排
        image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
        image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

        print('\nGenerating demo...')
        output_dir_pose = os.path.join(output_dir, 'pose')
        os.makedirs(output_dir_pose, exist_ok=True)

        for i in tqdm(range(len(image_2d_dir))):
            image_2d = plt.imread(image_2d_dir[i])
            image_3d = plt.imread(image_3d_dir[i])

            edge_2d = 10
            if image_2d.shape[1] > edge_2d * 2:
                image_2d = image_2d[:, edge_2d:image_2d.shape[1] - edge_2d]

            edge = 60
            image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

            font_size = 12
            fig = plt.figure(figsize=(15.0, 5.4))
            ax = plt.subplot(121)
            showimage(ax, image_2d)
            ax.set_title("Input", fontsize=font_size)

            ax = plt.subplot(122)
            showimage(ax, image_3d)
            ax.set_title("Reconstruction", fontsize=font_size)

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(os.path.join(output_dir_pose, str(('%04d' % i)) + '_pose.png'),
                        dpi=200, bbox_inches='tight')
            plt.close(fig)


def run_pose_estimation(video_path: str, output_dir: str,
                        only_2d: bool = False, gpu: str = "0",
                        bbox_csv: str = None,
                        skip_video: bool = False) -> str:
    """
    執行完整的 2D/3D 姿態估計流程。

    參數：
        video_path  str        輸入影片的絕對路徑
        output_dir  str        輸出目錄（若不存在將自動建立）
        only_2d     bool       True → 只執行 2D 姿態，跳過 3D 與角度計算
        gpu         str        CUDA_VISIBLE_DEVICES 值，例如 "0" 或 "1"
        bbox_csv    str | None track_crop_roi 輸出的 bbox_map.csv 路徑，
                               限定 HRNet 只在跑者 bbox 範圍內偵測
        skip_video  bool       True → 跳過 PNG 渲染與 mp4 合成（只輸出 CSV）

    回傳：
        output_dir  str   結果存放目錄（與傳入一致）
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.makedirs(output_dir, exist_ok=True)

    pose_timings = []

    def _record_pose_timing(stage, started_at):
        elapsed = time.perf_counter() - started_at
        pose_timings.append({"stage": stage, "elapsed_sec": round(elapsed, 4)})
        print(f"  ⏱ [TIME] {stage}: {elapsed:.2f}s")
        return elapsed

    pose_total_started_at = time.perf_counter()
    pose2d_started_at = time.perf_counter()
    get_pose2D(video_path, output_dir, bbox_csv=bbox_csv)
    _record_pose_timing("Step2/hrnet_2d_pose", pose2d_started_at)
    if not only_2d:
        pose3d_started_at = time.perf_counter()
        get_pose3D(video_path, output_dir, skip_video=skip_video)
        _record_pose_timing("Step2/motionagformer_3d_and_angles", pose3d_started_at)

    if not skip_video:
        video_started_at = time.perf_counter()
        img2video(video_path, output_dir)
        _export_final_video_frames(video_path, output_dir)
        archived_final_videos = _copy_final_videos_to_keypoints_archive(video_path, output_dir)
        for archived_video in archived_final_videos:
            print(f'Final pose video archived to {archived_video}')
        print('Generating demo successful!')
        _record_pose_timing("Step2/pose_video_render_and_archive", video_started_at)

    _record_pose_timing("Step2/vis_run_pose_estimation_total", pose_total_started_at)
    with open(os.path.join(output_dir, "pose_timing_report.json"), "w", encoding="utf-8") as f:
        json.dump({"timings": pose_timings}, f, ensure_ascii=False, indent=2)
    return output_dir


# -----------------------------------------------------------------------
# CLI 入口（向下相容）
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video filename')
    parser.add_argument('--gpu',   type=str, default='0', help='CUDA device index')
    parser.add_argument('--2d_only', action='store_true', help='Only generate 2D pose video')
    parser.add_argument('--bbox-csv', type=str, default=None,
                        help='External bbox CSV generated by track_crop_roi.py')
    args = parser.parse_args()

    # CLI 模式：影片放在 demo/video/，輸出到 demo/output/{video_name}/
    _video_path  = str(_REPO_ROOT / "demo" / "video" / args.video)
    _video_name  = Path(args.video).stem
    _output_dir  = str(_REPO_ROOT / "demo" / "output" / _video_name) + "/"

    run_pose_estimation(
        video_path=_video_path,
        output_dir=_output_dir,
        only_2d=args.__dict__.get('2d_only', False),
        gpu=args.gpu,
        bbox_csv=args.bbox_csv,
    )
