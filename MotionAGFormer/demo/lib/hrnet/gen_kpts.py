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
# The deployment default intentionally points to the best runner-domain
# candidate selected from the jump-broadcast / pilot300 experiments.  Keep
# this absolute path derived from this file so pipeline callers are not
# affected by their current working directory.
DEFAULT_WHOLEBODY23_MODEL = osp.abspath(osp.join(
    osp.dirname(__file__), '..', '..', '..', '..',
    'data', 'runner_wholebody23', 'exports',
    'pose_hrnet_w48_wholebody23_384x288_dark_jump_broadcast_long_triple_pilot300_headonly_epoch3.pth'))

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
    parser.add_argument('--modelDir', type=str, default=DEFAULT_WHOLEBODY23_MODEL,
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


def gen_video_kpts(video, det_dim=416, num_peroson=1, gen_output=False,
                   bbox_csv=None, model_path=None, frame_source=None,
                   profiling=None):
    # Updating configuration
    args = parse_args([])
    args.det_dim = det_dim
    args.num_person = num_peroson
    # Optional experiment override.  The default remains the production
    # WholeBody23 checkpoint so existing callers are unaffected.
    if model_path:
        args.modelDir = os.path.abspath(model_path)
    reset_config(args)

    cap = cv2.VideoCapture(video) if frame_source is None else None
    bbox_map = _load_bbox_map(bbox_csv) if frame_source is None else None

    # Loading detector and pose model, initialize sort for track
    human_model = None if bbox_map is not None or frame_source is not None else yolo_model(inp_dim=det_dim)
    model_load_started_at = time.perf_counter()
    pose_model = model_load(cfg)
    if profiling is not None:
        profiling.update({
            'model_load_sec': round(time.perf_counter() - model_load_started_at, 4),
            'preprocess_sec': 0.0,
            'decode_sec': 0.0,
            'frame_processing_wall_sec': 0.0,
            'inference_device_sec': 0.0,
            'inference_frames': 0,
        })
    cuda_events = []
    people_sort = Sort(min_hits=0)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap is not None else None
    frames = (cap.read() for _ in range(video_length)) if cap is not None else frame_source

    kpts_result = []
    scores_result = []
    bboxs_pre = None
    scores_pre = None

    for ii, item in enumerate(tqdm(frames, total=video_length)):
        frame_started_at = time.perf_counter() if profiling is not None else None
        if frame_source is None:
            ret, frame = item
            supplied_bbox = None
        else:
            frame, supplied_bbox = item
            ret = frame is not None

        if not ret:
            continue

        if frame_source is not None:
            track_bboxs = [supplied_bbox] if supplied_bbox is not None else bboxs_pre
            if track_bboxs is None:
                continue
            bboxs_pre = copy.deepcopy(track_bboxs)
        elif bbox_map is not None:
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
            preprocess_started_at = time.perf_counter() if profiling is not None else None
            inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)

            inputs = inputs[:, [2, 1, 0]]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            if profiling is not None:
                profiling['preprocess_sec'] += time.perf_counter() - preprocess_started_at
                if inputs.is_cuda:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    inference_started_at = time.perf_counter()
            output = pose_model(inputs)
            if profiling is not None:
                if inputs.is_cuda:
                    end_event.record()
                    cuda_events.append((start_event, end_event))
                else:
                    profiling['inference_device_sec'] += time.perf_counter() - inference_started_at
                profiling['inference_frames'] += 1

            # compute coordinate — DarkPose unbiased decode for all joints (body + foot)
            decode_started_at = time.perf_counter() if profiling is not None else None
            preds, maxvals = get_final_preds_dark(
                cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))
            if profiling is not None:
                profiling['decode_sec'] += time.perf_counter() - decode_started_at

        kpts = np.zeros((num_peroson, cfg.MODEL.NUM_JOINTS, 2), dtype=np.float32)
        scores = np.zeros((num_peroson, cfg.MODEL.NUM_JOINTS), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt

        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        kpts_result.append(kpts)
        scores_result.append(scores)
        if profiling is not None:
            profiling['frame_processing_wall_sec'] += time.perf_counter() - frame_started_at

    if profiling is not None:
        if cuda_events:
            torch.cuda.synchronize()
            profiling['inference_device_sec'] = sum(
                start.elapsed_time(end) for start, end in cuda_events
            ) / 1000.0
        profiling['inference_device_kind'] = 'cuda' if cuda_events else 'cpu'
        for key in ('preprocess_sec', 'decode_sec', 'frame_processing_wall_sec',
                    'inference_device_sec'):
            profiling[key] = round(profiling[key], 4)

    if cap is not None:
        cap.release()

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
