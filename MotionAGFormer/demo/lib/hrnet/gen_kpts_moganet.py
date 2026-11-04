"""MogaNet（AthletePose3D 微調版）2D 關鍵點產生器。

跟 gen_kpts.py 的 gen_video_kpts() 共用完全一樣的呼叫介面與回傳 shape，
讓 vis.py 的 get_pose2D() 可以用 backend 參數在兩者之間切換，不動到既有
HRNet 路徑。

跟 HRNet 版本的差異：
  - 只輸出 17 個標準 COCO body 點，沒有腳趾/腳跟（MogaNet 的 AthletePose3D
    微調資料集本身就沒有腳部標註）。呼叫端（get_pose2D）靠 `shape[2] > 17`
    判斷要不要拆腳部關鍵點，17 點時自然跳過，不需要額外處理。
  - 前處理/後處理不是照抄 gen_kpts.py 的 PreProcess（那是配合你們自己
    HRNet 訓練時的 BGR/RGB 怪癖寫的），而是用
    scripts/pose2d/moganet_ap2d_infer.py 裡已經驗證過、照 MMPose v0.29.0
    官方演算法實作的版本。
  - 目前只支援外部提供 bbox（frame_source 或 bbox_csv），不內建 YOLO 偵測
    器——production pipeline（core/pose_stream.py）本來就是透過 frame_source
    餵 bbox 進來，不需要自己偵測。
"""

import csv
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

_THIS_DIR = osp.dirname(osp.abspath(__file__))
_REPO_ROOT = osp.abspath(osp.join(_THIS_DIR, "..", "..", "..", ".."))
sys.path.insert(0, osp.join(_REPO_ROOT, "scripts", "pose2d"))

from moganet_ap2d_infer import MogaNetAP2D, infer_keypoints  # noqa: E402

NUM_JOINTS = 17

# 僅供速度對照用的暫時開關：HRNet 那條路徑的 config 雖然寫 FLIP_TEST=true，
# 但 gen_kpts.py 從未真的實作，每幀只跑一次 forward；這裡預設忠於論文做
# flip test（準確度較好，forward 次數變兩倍）。設 MOGANET_FLIP_TEST=0
# 可以關掉，用來跟 HRNet 的實際推論量做公平的速度比較。
_FLIP_TEST = os.environ.get("MOGANET_FLIP_TEST", "1") != "0"

# 僅供速度對照用的暫時開關：torch.compile(mode="reduce-overhead") 用 CUDA Graph
# 把 forward 的上千個小 kernel 包成一次性提交，實測單幀耗時可從 ~8.2ms 降到 ~3.9ms。
# 設 MOGANET_TORCH_COMPILE=1 開啟，預設關閉（還沒在正式環境的 bounded_stream
# multiprocessing worker 裡跑過完整影片驗證過關鍵點數值一致）。
_TORCH_COMPILE = os.environ.get("MOGANET_TORCH_COMPILE", "0") == "1"

_model_cache = {}


def _get_model(model_path, device):
    key = (model_path, device)
    if key not in _model_cache:
        if model_path:
            _model_cache[key] = MogaNetAP2D.from_checkpoint(
                ckpt_path=model_path, device=device, compile_model=_TORCH_COMPILE)
        else:
            _model_cache[key] = MogaNetAP2D.from_checkpoint(
                device=device, compile_model=_TORCH_COMPILE)
    return _model_cache[key]


def _xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox[:4]
    return (x1, y1, x2 - x1, y2 - y1)


def _load_bbox_map(bbox_csv):
    if not bbox_csv:
        return None
    bbox_map = {}
    with open(bbox_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bbox_map[int(row["output_frame"])] = [
                float(row["x1"]), float(row["y1"]),
                float(row["x2"]), float(row["y2"]),
            ]
    return bbox_map


def gen_video_kpts_moganet(video, det_dim=416, num_peroson=1, gen_output=False,
                           bbox_csv=None, model_path=None, frame_source=None,
                           profiling=None):
    if num_peroson != 1:
        raise ValueError("MogaNet backend 目前只支援 num_peroson=1（單一已選跑者）")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_load_started_at = time.perf_counter()
    model = _get_model(model_path, device)
    if profiling is not None:
        profiling.update({
            "model_load_sec": round(time.perf_counter() - model_load_started_at, 4),
            "preprocess_sec": 0.0,
            "decode_sec": 0.0,
            "frame_processing_wall_sec": 0.0,
            "inference_device_sec": 0.0,
            "inference_frames": 0,
            "inference_device_kind": "cuda" if device == "cuda" else "cpu",
        })

    if frame_source is None:
        cap = cv2.VideoCapture(video)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        bbox_map = _load_bbox_map(bbox_csv)
        frames = (cap.read() for _ in range(video_length))
    else:
        cap = None
        video_length = None
        bbox_map = None
        frames = frame_source

    kpts_result = []
    scores_result = []
    bbox_pre = None

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
            bbox = supplied_bbox if supplied_bbox is not None else bbox_pre
        elif bbox_map is not None:
            bbox = bbox_map.get(ii, bbox_pre)
        else:
            raise ValueError(
                "MogaNet backend 需要外部提供 bbox（frame_source 或 bbox_csv），"
                "不支援自動偵測"
            )
        if bbox is None:
            continue
        bbox_pre = bbox

        if profiling is not None:
            pre_started_at = time.perf_counter()
        kpts, scores = infer_keypoints(model, frame, [_xyxy_to_xywh(bbox)], device=device,
                                       flip_test=_FLIP_TEST)
        if profiling is not None:
            elapsed = time.perf_counter() - pre_started_at
            profiling["inference_device_sec"] += elapsed
            profiling["inference_frames"] += 1

        kpts_result.append(kpts)   # (1, 17, 2)
        scores_result.append(scores)  # (1, 17)
        if profiling is not None:
            profiling["frame_processing_wall_sec"] += time.perf_counter() - frame_started_at

    if profiling is not None:
        for key in ("preprocess_sec", "decode_sec", "frame_processing_wall_sec",
                    "inference_device_sec"):
            profiling[key] = round(profiling[key], 4)

    if cap is not None:
        cap.release()

    if not kpts_result:
        print("Warning: No keypoints generated for any frame.")
        return (np.zeros((num_peroson, 0, NUM_JOINTS, 2)),
                np.zeros((num_peroson, 0, NUM_JOINTS)))

    keypoints = np.array(kpts_result)  # (T, 1, 17, 2)
    scores = np.array(scores_result)   # (T, 1, 17)
    keypoints = keypoints.transpose(1, 0, 2, 3)  # (1, T, 17, 2)
    scores = scores.transpose(1, 0, 2)           # (1, T, 17)

    return keypoints, scores
