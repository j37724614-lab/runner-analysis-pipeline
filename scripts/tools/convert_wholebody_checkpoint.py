"""將 MMPose HRNet-W48 DarkPose+ COCO-WholeBody checkpoint 轉成本地 pose_hrnet.py 可載入的格式。

來源 checkpoint 結構：{'meta': ..., 'state_dict': {'backbone.xxx': ..., 'keypoint_head.final_layer.xxx': ...}}
輸出：攤平的 state_dict，key 與本地 PoseHighResolutionNet 對齊，final_layer 只保留前 NUM_KEEP_JOINTS 個 channel
（COCO-WholeBody 順序：0-16 身體 17 點，17-22 為 L/R 大腳趾、小腳趾、腳跟）。
"""
import argparse
from collections import OrderedDict

import torch

NUM_KEEP_JOINTS = 23  # 17 body + 6 foot (big_toe, small_toe, heel x L/R)


def convert(src_path, dst_path):
    ckpt = torch.load(src_path, map_location='cpu', weights_only=False)
    src_state = ckpt['state_dict']

    new_state = OrderedDict()
    for k, v in src_state.items():
        if k.startswith('backbone.'):
            new_state[k[len('backbone.'):]] = v
        elif k == 'keypoint_head.final_layer.weight':
            new_state['final_layer.weight'] = v[:NUM_KEEP_JOINTS].clone()
        elif k == 'keypoint_head.final_layer.bias':
            new_state['final_layer.bias'] = v[:NUM_KEEP_JOINTS].clone()
        else:
            raise ValueError(f'unexpected key in source checkpoint: {k}')

    torch.save(new_state, dst_path)
    print(f'saved {len(new_state)} tensors to {dst_path}')
    print(f"final_layer.weight shape: {new_state['final_layer.weight'].shape}")
    print(f"final_layer.bias shape: {new_state['final_layer.bias'].shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    args = parser.parse_args()
    convert(args.src, args.dst)
