
import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
import yaml

sys.path.append(os.getcwd())
from demo.lib.utils import normalize_screen_coordinates, camera_to_world
from model.MotionAGFormer import MotionAGFormer

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
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

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    
    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints, valid_frames=valid_frames)


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Support both pose and pose2D
    pose_dir = 'pose/'
    pose2d_dir = 'pose2D/'
    names_pose = sorted(glob.glob(os.path.join(output_dir + pose_dir, '*.png')))
    names_pose2d = sorted(glob.glob(os.path.join(output_dir + pose2d_dir, '*.png')))

    # If pose2D images exist, export 2D-only video
    if names_pose2d:
        img = cv2.imread(names_pose2d[0])
        size = (img.shape[1], img.shape[0])
        video_name = os.path.basename(video_path).split('.')[0]
        videoWrite = cv2.VideoWriter(output_dir + video_name + '_2D.mp4', fourcc, fps, size)
        for name in names_pose2d:
            img = cv2.imread(name)
            videoWrite.write(img)
        videoWrite.release()
        print(f"2D Video saved to {output_dir + video_name + '_2D.mp4'}")

    # Standard pose video
    if names_pose:
        img = cv2.imread(names_pose[0])
        size = (img.shape[1], img.shape[0])
        video_name = os.path.basename(video_path).split('.')[0]
        videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size)
        for name in names_pose:
            img = cv2.imread(name)
            videoWrite.write(img)
        videoWrite.release()
        print(f"Pose video saved to {output_dir + video_name + '.mp4'}")


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
        # 找出哪些 index 是唯一的（解採樣回到原始長度）
        _, unique_idx = np.unique(new_indices, return_index=True)
        downsample_indices.append(unique_idx)
    else:
        # 分段處理，每段 243 幀
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length < 243:
                # 最後一段不足 243 幀，需要補幀
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
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
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

        # Left knee
        left_knee = _angle_between(pose[4] - pose[5], pose[6] - pose[5])

        # Left hip
        left_hip = _angle_between(pose[11] - pose[4], pose[5] - pose[4])

        # Right knee
        right_knee = _angle_between(pose[1] - pose[2], pose[3] - pose[2])

        # Right hip
        right_hip = _angle_between(pose[14] - pose[1], pose[2] - pose[1])

        # Left arm-torso
        left_arm_torso = _angle_between(pose[4] - pose[11], pose[12] - pose[11])

        # Left elbow flexion
        left_elbow_flexion = _angle_between(pose[13] - pose[12], pose[11] - pose[12])

        # Right arm-torso
        right_arm_torso = _angle_between(pose[1] - pose[14], pose[15] - pose[14])

        # Right elbow flexion
        right_elbow_flexion = _angle_between(pose[16] - pose[15], pose[14] - pose[15])

        # Left shoulder flexion
        left_shoulder_flexion = _angle_between(pose[0] - pose[8], pose[12] - pose[8])

        # Right shoulder flexion
        right_shoulder_flexion = _angle_between(pose[0] - pose[8], pose[15] - pose[8])

        # Pelvis-torso angle (vs vertical, Y axis points down)
        vertical = np.array([0, -1, 0])
        torso_vec = pose[8] - pose[0]
        pelvis_torso_angle = _angle_between(vertical, torso_vec)

        results.append([
            i,
            left_knee, left_hip,
            right_knee, right_hip,
            left_arm_torso, left_elbow_flexion,
            right_arm_torso, right_elbow_flexion,
            left_shoulder_flexion, right_shoulder_flexion,
            pelvis_torso_angle
        ])

    columns = [
        'frame',
        'left_knee_angle', 'left_hip_angle',
        'right_knee_angle', 'right_hip_angle',
        'left_arm_torso_angle', 'left_elbow_flexion_angle',
        'right_arm_torso_angle', 'right_elbow_flexion_angle',
        'left_shoulder_flexion', 'right_shoulder_flexion',
        'pelvis_torso_angle'
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
def get_pose3D(video_path, output_dir):
    # 讀取 L 版 YAML config
    config_path = '/home/jeter/MotionAGFormer/MotionAGFormer/configs/h36m/MotionAGFormer-large.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    # 處理 config 參數型態（如 GELU 激活函數）
    if 'act_layer' in config and isinstance(config['act_layer'], str):
        if config['act_layer'].lower() == 'gelu':
            config['act_layer'] = nn.GELU
        elif config['act_layer'].lower() == 'relu':
            config['act_layer'] = nn.ReLU
        # 可擴充其他激活函數

    # 其餘參數型態處理（如 True/False 字串）
    for k, v in config.items():
        if isinstance(v, str):
            if v.lower() == 'true':
                config[k] = True
            elif v.lower() == 'false':
                config[k] = False

    # 過濾掉非模型 __init__ 參數
    import inspect
    model_keys = inspect.signature(MotionAGFormer.__init__).parameters.keys()
    model_keys = set(model_keys) - {'self'}
    model_config = {k: v for k, v in config.items() if k in model_keys}

    model = nn.DataParallel(MotionAGFormer(**model_config)).cuda()

    # Put the pretrained model of MotionAGFormer in 'checkpoint/'
    model_path = sorted(glob.glob(os.path.join('checkpoint', 'motionagformer-l-h36m.pth.tr')))[0]

    pre_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(pre_dict['model'], strict=True)

    model.eval()

    ## input
    data = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)
    keypoints = data['reconstruction']
    valid_frames = np.asarray(data['valid_frames']).flatten().astype(int)  # 轉換為一維整數數組
    # keypoints = np.load('demo/lakeside3.npy')
    # keypoints = keypoints[:240]
    # keypoints = keypoints[None, ...]
    # keypoints = turn_into_h36m(keypoints)
    

    clips, downsample_indices = turn_into_clips(keypoints)

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_img = cap.read()
    if not ret:
        print("Error: Could not read video file.")
        return
    img_size = first_img.shape
    
    # 使用實際檢測到的關鍵點幀數，避免HRNet無法檢測所有幀的問題
    valid_frame_count = keypoints.shape[1]

    ## 3D
    print('\nGenerating 2D pose image...')
    output_dir_2D = os.path.join(output_dir, 'pose2D/')
    os.makedirs(output_dir_2D, exist_ok=True)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到開頭
    for i in tqdm(range(valid_frame_count)):
        # 根據 valid_frames 跳到正確的視頻幀位置
        frame_idx = valid_frames[i] if i < len(valid_frames) else i
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        ret, img = cap.read()
        if img is None:
            continue
        # image = show2Dpose(input_2D, copy.deepcopy(img)) # 這裡原本有錯，修正一下
        input_2D_raw = keypoints[0][i]
        image = show2Dpose(input_2D_raw, copy.deepcopy(img))

        cv2.imwrite(os.path.join(output_dir_2D, str(('%04d'% i)) + '_2D.png'), image)

    
    print('\nGenerating 3D pose...')
    all_poses_all_clips = []
    
    for idx, (clip, d_idx) in enumerate(zip(clips, downsample_indices)):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
        input_2D_aug = flip_data(input_2D)
        
        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        output_3D_non_flip = model(input_2D) 
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        # 根據保存的索引進行解採樣，恢復原始長度（例如最後一段從 243 變回 84）
        output_3D = output_3D[:, d_idx]

        post_out_clip = output_3D[0].cpu().detach().numpy()
        all_poses_all_clips.append(post_out_clip)

    # 合併所有片段
    post_out_all = np.concatenate(all_poses_all_clips, axis=0)
    
    print(f"Total processed frames: {post_out_all.shape[0]}")
    
    output_dir_3D_npz = output_dir + 'pred_3D/'
    os.makedirs(output_dir_3D_npz, exist_ok=True)
    npz_out_path = os.path.join(output_dir_3D_npz, '3Dkeypoints.npz')
    np.savez_compressed(npz_out_path, pred_3d=post_out_all)

    # 角度計算
    compute_angles(npz_out_path, output_dir)

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

        output_dir_3D = output_dir +'pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)
        plt.savefig(output_dir_3D + str(('%04d'% j)) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
        plt.close(fig)
        

        
    print('Generating 3D pose successful!')

    ## all
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        # 2D影像：完全不裁剪或只裁剪很小的邊界
        edge_2d = 10  # 只裁剪10像素（白邊）
        if image_2d.shape[1] > edge_2d * 2:
            image_2d = image_2d[:, edge_2d:image_2d.shape[1] - edge_2d]
        
        # 3D影像：裁剪較多（去除matplotlib邊框）
        edge = 60
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--2d_only', action='store_true', help='Only generate 2D pose video')
    parser.add_argument('--no_angles', action='store_true', help='Skip angle computation (angles are saved by default)')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'

    get_pose2D(video_path, output_dir)
    if not args.__dict__.get('2d_only', False):
        get_pose3D(video_path, output_dir)   # ← 角度計算已整合在內，自動執行
    img2video(video_path, output_dir)
    print('Generating demo successful!')


# 檢查 NPZ 文件
output_dir = "./demo/output/second_original_IMG_2534_runner_tracked/"  # 改成你的影片名稱
npz_path = output_dir + "input_2D/keypoints.npz"

if os.path.exists(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    keypoints = data['reconstruction']
    valid_frames = data['valid_frames']
    
    print(f"關鍵點檢測數量: {keypoints.shape[1]}")  # 第1維是幀數
    print(f"有效幀索引數量: {len(valid_frames)}")
    print(f"有效幀索引: {valid_frames.flatten()}")
    
    # 檢查生成的 2D 圖片
