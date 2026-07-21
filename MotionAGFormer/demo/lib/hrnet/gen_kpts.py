from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import argparse
import time
import numpy as np
from tqdm import tqdm
import json
import csv
import torch
import torch.backends.cudnn as cudnn
import cv2
import copy

from lib.hrnet.lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
from lib.hrnet.lib.config import cfg, update_config
from lib.hrnet.lib.utils.transforms import *
from lib.hrnet.lib.utils.inference import get_final_preds_dark
from lib.hrnet.lib.models import pose_hrnet

cfg_dir = 'demo/lib/hrnet/experiments/'
model_dir = 'demo/lib/checkpoint/'

# Loading human detector model
from lib.yolov3.human_detector import load_model as yolo_model
from lib.yolov3.human_detector import yolo_human_det as yolo_det
from lib.sort.sort import Sort


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=cfg_dir + 'w48_384x288_wholebody23_dark.yaml',
                        help='experiment configure file name')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help="Modify config options using the command-line")
    parser.add_argument('--modelDir', type=str, default=model_dir + 'pose_hrnet_w48_wholebody23_384x288_dark.pth',
                        help='The model directory')
    parser.add_argument('--det-dim', type=int, default=416,
                        help='The input dimension of the detected image')
    parser.add_argument('--thred-score', type=float, default=0.10,
                        help='The threshold of object Confidence')
    parser.add_argument('-a', '--animation', action='store_true',
                        help='output animation')
    parser.add_argument('-np', '--num-person', type=int, default=1,
                        help='The maximum number of estimated poses')
    parser.add_argument("-v", "--video", type=str, default='camera',
                        help="input video file name")
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args, _ = parser.parse_known_args(argv)

    return args


def reset_config(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


# load model
def model_load(config):
    model = pose_hrnet.get_pose_net(config, is_train=False)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(config.OUTPUT_DIR, weights_only=False)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    # print('HRNet network successfully loaded')
    
    return model


def _load_bbox_map(bbox_csv):
    if not bbox_csv:
        return None
    bbox_map = {}
    with open(bbox_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            bbox_map[int(row['output_frame'])] = [[
                float(row['x1']),
                float(row['y1']),
                float(row['x2']),
                float(row['y2']),
            ]]
    return bbox_map


def gen_video_kpts(video, det_dim=416, num_peroson=1, gen_output=False, bbox_csv=None):
    # Updating configuration
    args = parse_args([])
    args.det_dim = det_dim
    args.num_person = num_peroson
    reset_config(args)

    cap = cv2.VideoCapture(video)

    bbox_map = _load_bbox_map(bbox_csv)

    # Loading detector and pose model, initialize sort for track
    human_model = None if bbox_map is not None else yolo_model(inp_dim=det_dim)
    pose_model = model_load(cfg)
    people_sort = Sort(min_hits=0)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    kpts_result = []
    scores_result = []
    bboxs_pre = None
    scores_pre = None

    for ii in tqdm(range(video_length)):
        ret, frame = cap.read()

        if not ret:
            continue

        if bbox_map is not None:
            track_bboxs = bbox_map.get(ii)
            if not track_bboxs:
                if bboxs_pre is None:
                    continue
                track_bboxs = bboxs_pre
            else:
                bboxs_pre = copy.deepcopy(track_bboxs)
        else:
            bboxs, scores = yolo_det(frame, human_model, reso=det_dim, confidence=args.thred_score)

            if bboxs is None or not bboxs.any():
                if bboxs_pre is None:
                    # No person detected in the first frame(s)
                    continue
                bboxs = np.array(bboxs_pre)
                scores = scores_pre
            else:
                bboxs_pre = copy.deepcopy(bboxs) 
                scores_pre = copy.deepcopy(scores) 

            # Using Sort to track people
            people_track = people_sort.update(bboxs)

            # Track the first two people in the video and remove the ID
            if people_track.shape[0] == 1:
                people_track_ = people_track[-1, :-1].reshape(1, 4)
            elif people_track.shape[0] >= 2:
                people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
                people_track_ = people_track_[::-1]
            else:
                continue

            track_bboxs = []
            for bbox in people_track_:
                bbox = [round(i, 2) for i in list(bbox)]
                track_bboxs.append(bbox)
            bboxs_pre = copy.deepcopy(track_bboxs)

        with torch.no_grad():
            # bbox is coordinate location
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)

            inputs = inputs[:, [2, 1, 0]]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output = pose_model(inputs)

            # compute coordinate — DarkPose unbiased decode for all joints (body + foot)
            preds, maxvals = get_final_preds_dark(
                cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

        kpts = np.zeros((num_peroson, cfg.MODEL.NUM_JOINTS, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, cfg.MODEL.NUM_JOINTS), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt

        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        kpts_result.append(kpts)
        scores_result.append(scores)

    if not kpts_result:
        print("Warning: No keypoints generated for any frame.")
        # Return dummy data or handle gracefully?
        # For now, let's return zeros matching shape to avoid crash downstream, or let caller handle.
        # But caller expects valid data. 
        # Better to raise specific error or return empty arrays that check checks.
        return np.zeros((num_peroson, 0, cfg.MODEL.NUM_JOINTS, 2)), np.zeros((num_peroson, 0, cfg.MODEL.NUM_JOINTS))

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores
