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
from pathlib import Path
import os

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


def _reset_dir(path):
    """清掉同名輸出資料夾，避免混到上一次不同尺寸或不同幀數的 PNG。"""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


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


def _smooth_keypoints_ema(keypoints, scores, alpha=0.45, conf_threshold=0.35,
                          max_jump_px=45.0, hard_jump_px=90.0,
                          interpolated_mask=None):
    """
    對 HRNet 2D keypoints 做時間 EMA 平滑。

    keypoints: shape (M, T, J, 2)
    scores:    shape (M, T, J)

    高信心或已插值補過、且位移合理的點用 EMA 更新。
    高信心但位移偏大的點會限制最大步長後更新，避免快速擺動關節卡在上一幀。
    未插值的低信心或極端跳點才沿用上一幀，避免錯點把骨架拉飄。
    """
    if keypoints.size == 0 or scores.size == 0:
        return keypoints

    smoothed = keypoints.astype(np.float32, copy=True)
    num_person, num_frames, num_joints = scores.shape
    if interpolated_mask is None:
        interpolated_mask = np.zeros(scores.shape, dtype=bool)

    for person_idx in range(num_person):
        prev = None
        for frame_idx in range(num_frames):
            current = smoothed[person_idx, frame_idx]
            current_scores = scores[person_idx, frame_idx]

            if prev is None:
                prev = current.copy()
                continue

            high_conf = (current_scores >= conf_threshold) | interpolated_mask[person_idx, frame_idx]
            jump = np.linalg.norm(current - prev, axis=1)
            normal_update = high_conf & (jump <= max_jump_px)

            if np.any(normal_update):
                prev[normal_update] = (
                    alpha * prev[normal_update] +
                    (1.0 - alpha) * current[normal_update]
                )

            limited_update = high_conf & (jump > max_jump_px) & (jump <= hard_jump_px)
            if np.any(limited_update):
                direction = current[limited_update] - prev[limited_update]
                distance = jump[limited_update][:, None]
                capped_current = prev[limited_update] + direction / distance * max_jump_px
                prev[limited_update] = (
                    alpha * prev[limited_update] +
                    (1.0 - alpha) * capped_current
                )

            smoothed[person_idx, frame_idx] = prev

    return smoothed


def _save_hrnet_confidence_csv(scores, valid_frames, output_csv):
    """將 HRNet/H36M 17 個關節的 confidence 另存成 CSV，方便檢查飄點來源。"""
    if scores.size == 0:
        return

    joint_names = [
        'Hip',
        'RHip', 'RKnee', 'RAnkle',
        'LHip', 'LKnee', 'LAnkle',
        'Spine', 'Thorax',
        'Neck_Nose', 'Head',
        'LShoulder', 'LElbow', 'LWrist',
        'RShoulder', 'RElbow', 'RWrist',
    ]
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
            for joint_idx, joint_name in enumerate(joint_names):
                row[joint_name] = float(scores[person_idx, frame_idx, joint_idx])
            rows.append(row)

    pd.DataFrame(rows).to_csv(output_csv, index=False)


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
    keypoints, interpolated_mask = _interpolate_low_conf_keypoints(
        keypoints,
        scores,
        conf_threshold=0.50,
    )
    keypoints = _smooth_keypoints_ema(
        keypoints,
        scores,
        alpha=0.20,
        conf_threshold=0.50,
        max_jump_px=45.0,
        hard_jump_px=90.0,
        interpolated_mask=interpolated_mask,
    )

    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_2d_dir = os.path.join(output_dir, 'input_2D')
    _reset_dir(output_2d_dir)

    output_npz = os.path.join(output_2d_dir, 'keypoints.npz')
    np.savez_compressed(output_npz, reconstruction=keypoints, valid_frames=valid_frames)
    confidence_csv = os.path.join(output_dir, 'hrnet_confidence.csv')
    _save_hrnet_confidence_csv(scores, valid_frames, confidence_csv)
    print(f'HRNet confidence CSV saved to {confidence_csv}')


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


def compute_angles(npz_path, output_dir):
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

    out_dir = os.path.join(output_dir, 'pred_3D', 'angles')
    os.makedirs(out_dir, exist_ok=True)
    video_name = os.path.basename(os.path.normpath(output_dir))
    out_csv = os.path.join(out_dir, f'{video_name}_angles.csv')
    df.to_csv(out_csv, index=False)
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
    compute_angles(npz_out_path, output_dir)

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

    get_pose2D(video_path, output_dir, bbox_csv=bbox_csv)
    if not only_2d:
        get_pose3D(video_path, output_dir, skip_video=skip_video)

    if not skip_video:
        img2video(video_path, output_dir)
        print('Generating demo successful!')

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
