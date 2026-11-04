"""
MogaNet-base 2D 姿態推論（AthletePose3D 微調版），不依賴 MMPose。

背景：model_params-20260930T101153Z-1-001/model_params/moganet_b_ap2d_384x288.pth
是用舊版 MMPose（0.x，TopDown + keypoint_head 命名）訓練的，跟本專案的新版 MMPose
（wholebody23 用的 1.x，head 命名）互不相容。這支腳本直接重建等價的 backbone + head
（結構、key 名稱、前處理/後處理演算法均照 MMPose v0.29.0 官方原始碼實作，
 https://github.com/open-mmlab/mmpose/tree/v0.29.0），跳過對 MMPose 套件本身的依賴。

輸出關鍵點為標準 COCO-17 格式（無腳趾/腳跟）。

用法（單張圖片 + 一個 bbox）：
    python moganet_ap2d_infer.py --image path/to/img.jpg --bbox x y w h
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

# HRNet 那條路徑的 config 有開這個（cfg.CUDNN.BENCHMARK=True），讓 cuDNN
# 針對固定輸入尺寸（384x288，每幀都一樣）自動挑選最快的卷積演算法；
# 這裡原本沒開，等於一直用預設、未調校的演算法。
torch.backends.cudnn.benchmark = True

_REPO_ROOT = Path(__file__).resolve().parents[2]  # runner-analysis-pipeline/
_AP3D_ROOT = _REPO_ROOT / "athletepose3d"
sys.path.insert(0, str(_AP3D_ROOT / "MogaNet"))

from models.moganet import MogaNet  # noqa: E402  (vendored, gitignored)

CHECKPOINT_PATH = _AP3D_ROOT / "model_params" / "moganet_b_ap2d_384x288.pth"

IMAGE_SIZE = (288, 384)   # (W, H) — 對應 config 的 image_size
HEATMAP_SIZE = (72, 96)   # (W, H) — 對應 config 的 heatmap_size
NUM_JOINTS = 17
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PIXEL_STD = 200.0
BBOX_PADDING = 1.25

# COCO-17 左右對稱點（AthletePose3D pose_2d/ap2d.py 的 dataset_info 與此一致）
COCO_FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


# ---------------------------------------------------------------------------
# 模型：MogaNet-base backbone + TopdownHeatmapSimpleHead 等價結構
# ---------------------------------------------------------------------------

class SimpleHead(nn.Module):
    """等價於 mmpose.models.heads.TopdownHeatmapSimpleHead
    (num_deconv_layers=3, num_deconv_filters=(256,256,256), num_deconv_kernels=(4,4,4))。
    """

    def __init__(self, in_channels=512, out_channels=NUM_JOINTS):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(3):
            layers += [
                nn.ConvTranspose2d(ch, 256, kernel_size=4, stride=2, padding=1,
                                    output_padding=0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ]
            ch = 256
        self.deconv_layers = nn.Sequential(*layers)
        self.final_layer = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.deconv_layers(x)
        return self.final_layer(x)


class MogaNetAP2D(nn.Module):
    """backbone(MogaNet-base, fork_feat=True) + SimpleHead，對應 checkpoint 的
    `backbone.*` / `keypoint_head.*` 兩組權重。
    """

    def __init__(self):
        super().__init__()
        self.backbone = MogaNet(arch="base", fork_feat=True, drop_path_rate=0.4, init_value=1e-5)
        self.keypoint_head = SimpleHead(in_channels=512, out_channels=NUM_JOINTS)

    def forward(self, x):
        feats = self.backbone(x)   # list of 4 stage feature maps (fork_feat=True)
        return self.keypoint_head(feats[3])  # in_index=3 -> 最後一個 stage

    @classmethod
    def from_checkpoint(cls, ckpt_path=CHECKPOINT_PATH, device="cpu", compile_model=False):
        """compile_model=True：用 torch.compile(mode="reduce-overhead")（CUDA Graph）
        包住 forward。單幀耗時從量測的 ~8.2ms 降到 ~3.9ms（RTX PRO 6000 Blackwell 實測），
        因為這個 backbone 單次 forward 要發起 1000+ 個小 kernel（depthwise conv 切分 +
        gating 的逐元素運算），CUDA Graph 把整串 kernel 包成一次性提交，省掉大部分
        launch overhead。輸入尺寸固定為 384x288、batch 固定為 1，符合 CUDA Graph
        需要靜態 shape 的前提。第一次呼叫會觸發編譯，所以这里先跑兩次暖機，
        把編譯成本放在模型載入階段、不要算進逐幀推論的計時裡。
        """
        model = cls()
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        model.to(device).eval()
        if compile_model:
            model = torch.compile(model, mode="reduce-overhead")
            with torch.no_grad():
                dummy = torch.zeros(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0], device=device)
                model(dummy)
                model(dummy)
        return model


# ---------------------------------------------------------------------------
# 前處理 / 後處理（照抄 mmpose v0.29.0 的演算法，不依賴 mmcv/mmpose 套件）
# ---------------------------------------------------------------------------

def bbox_xywh2cs(bbox, aspect_ratio, padding=BBOX_PADDING, pixel_std=PIXEL_STD):
    x, y, w, h = bbox[:4]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32) / pixel_std * padding
    return center, scale


def _get_3rd_point(a, b):
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def _rotate_point(pt, angle_rad):
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    return np.array([pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs], dtype=np.float32)


def get_affine_transform(center, scale, rot, output_size, inv=False):
    scale_tmp = scale * PIXEL_STD
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size

    rot_rad = np.pi * rot / 180
    src_dir = _rotate_point(np.array([0.0, src_w * -0.5], dtype=np.float32), rot_rad)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _get_3rd_point(src[0], src[1])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def transform_preds(coords, center, scale, output_size):
    scale = scale * PIXEL_STD
    scale_x = scale[0] / output_size[0]
    scale_y = scale[1] / output_size[1]
    target = coords.copy()
    target[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return target


def get_max_preds(heatmaps):
    """heatmaps: (N, K, H, W) -> preds (N,K,2), maxvals (N,K,1)"""
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = np.argmax(flat, axis=2)
    maxvals = np.amax(flat, axis=2)
    preds = np.zeros((N, K, 2), dtype=np.float32)
    preds[..., 0] = idx % W
    preds[..., 1] = idx // W
    preds[maxvals <= 0.0] = -1
    return preds, maxvals[..., None]


def decode_heatmaps_default(heatmaps, center, scale, output_size):
    """post_process='default'：argmax + 1/4-pixel 修正，再映射回原圖座標。"""
    N, K, H, W = heatmaps.shape
    preds, maxvals = get_max_preds(heatmaps)
    for n in range(N):
        for k in range(K):
            hm = heatmaps[n, k]
            px, py = int(preds[n, k, 0]), int(preds[n, k, 1])
            if 1 < px < W - 1 and 1 < py < H - 1:
                diff = np.array([hm[py, px + 1] - hm[py, px - 1],
                                  hm[py + 1, px] - hm[py - 1, px]])
                preds[n, k] += np.sign(diff) * 0.25
    for n in range(N):
        preds[n] = transform_preds(preds[n], center[n], scale[n], output_size)
    return preds, maxvals


def flip_back(heatmaps_flipped, flip_pairs):
    """heatmaps_flipped: (N,K,H,W) numpy，水平翻轉圖片跑出來的熱力圖 -> 還原成正常朝向。"""
    out = heatmaps_flipped.copy()
    for left, right in flip_pairs:
        out[:, left] = heatmaps_flipped[:, right]
        out[:, right] = heatmaps_flipped[:, left]
    return out[..., ::-1].copy()


# ---------------------------------------------------------------------------
# 推論主流程
# ---------------------------------------------------------------------------

def preprocess(img_rgb, bbox_xywh):
    aspect_ratio = IMAGE_SIZE[0] / IMAGE_SIZE[1]
    center, scale = bbox_xywh2cs(bbox_xywh, aspect_ratio)
    trans = get_affine_transform(center, scale, 0, IMAGE_SIZE)
    warped = cv2.warpAffine(img_rgb, trans, IMAGE_SIZE, flags=cv2.INTER_LINEAR)
    tensor = warped.astype(np.float32) / 255.0
    tensor = (tensor - MEAN) / STD
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).float()
    return tensor, center, scale


@torch.no_grad()
def infer_keypoints(model, img_bgr, bboxes_xywh, device="cpu", flip_test=True):
    """
    Args:
        model: MogaNetAP2D.from_checkpoint(...) 的結果
        img_bgr: cv2.imread 讀出來的 BGR 圖片
        bboxes_xywh: list of (x, y, w, h)，來自你自己的偵測器（例如現有的 YOLO/HRNet 流程）
        flip_test: 論文 config 的 test_cfg.flip_test=True 做法（多跑一次翻轉圖取平均，
            會讓 forward 次數變兩倍）。HRNet 那條路徑的 config 雖然寫 FLIP_TEST=true，
            但 gen_kpts.py 實際上沒有實作、從來沒真的用到，所以要跟 HRNet 的實際推論量
            做速度比較時，這裡要關掉才公平。
    Returns:
        keypoints: (N, 17, 2) 原圖座標
        scores:    (N, 17) 置信度
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    batch, centers, scales = [], [], []
    for bbox in bboxes_xywh:
        tensor, center, scale = preprocess(img_rgb, bbox)
        batch.append(tensor)
        centers.append(center)
        scales.append(scale)
    batch = torch.stack(batch).to(device)

    heatmaps = model(batch).cpu().numpy()

    if flip_test:
        flipped_input = batch.flip(3)
        heatmaps_flipped = model(flipped_input).cpu().numpy()
        heatmaps_flipped_back = flip_back(heatmaps_flipped, COCO_FLIP_PAIRS)
        # shift_heatmap=True：偶數尺寸翻轉造成的半像素偏移修正
        heatmaps_flipped_back[..., 1:] = heatmaps_flipped_back[..., :-1]
        heatmaps = (heatmaps + heatmaps_flipped_back) / 2.0

    preds, maxvals = decode_heatmaps_default(heatmaps, centers, scales, HEATMAP_SIZE)
    return preds, maxvals[..., 0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    model = MogaNetAP2D.from_checkpoint(device=device)

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    keypoints, scores = infer_keypoints(model, img, [tuple(args.bbox)], device=device)
    for i, (xy, s) in enumerate(zip(keypoints[0], scores[0])):
        print(f"kpt{i:2d}: ({xy[0]:7.2f}, {xy[1]:7.2f})  score={s:.3f}")
