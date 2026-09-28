"""two-pass 診斷輸出（CSV/JSON）與 auto crop 尺寸估計。"""

import csv
import json
import os

import numpy as np

from core import tracking as _tracking


def _write_two_pass_debug(all_detections, summaries, preset_ids, base_name):
    """輸出 two_pass 的 debug CSV/JSON 供事後分析。"""
    os.makedirs(_tracking.OUTPUT_DIR, exist_ok=True)

    if all_detections:
        path = os.path.join(_tracking.OUTPUT_DIR, f"{base_name}_all_tracks.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_detections[0].keys()))
            writer.writeheader()
            writer.writerows(all_detections)
        print(f"  All tracks:     {path}")

    if summaries:
        path = os.path.join(_tracking.OUTPUT_DIR, f"{base_name}_track_summary.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"  Track summary:  {path}")

    selected_data = []
    for cam_idx, track_id in preset_ids.items():
        entry = {'cam_idx': cam_idx, 'track_id': track_id}
        for s in summaries:
            if s['cam_idx'] == cam_idx and s['track_id'] == track_id:
                entry.update({k: v for k, v in s.items()
                              if k not in ('cam_idx', 'track_id')})
                break
        selected_data.append(entry)
    path = os.path.join(_tracking.OUTPUT_DIR, f"{base_name}_selected_runner.json")
    with open(path, 'w') as f:
        json.dump(selected_data, f, indent=2)
    print(f"  Selected runner: {path}")


def _auto_crop_side_from_bbox_sizes(widths, heights):
    """依 bbox 寬高樣本計算 auto square crop 邊長。"""
    if not widths or not heights:
        return None
    p90_side = max(np.percentile(widths, 90), np.percentile(heights, 90)) * 1.25
    max_side = max(max(widths), max(heights)) * 1.05
    return int(np.ceil(max(p90_side, max_side)))


def _auto_crop_from_selected_cache(frame_cache, preset_ids):
    """
    從 two_pass 已選主跑者 bbox 計算 auto crop。

    呼叫時機應在 _stitch_target_id() 之後，這樣短暫換 ID 或漏偵測後
    被修補回 target_id 的 bbox 也會納入尺寸統計。
    """
    widths = []
    heights = []
    for (cam_idx, _frame_idx), detections in frame_cache.items():
        target_id = preset_ids.get(cam_idx)
        if target_id is None:
            continue
        for det in detections:
            if int(det.get('track_id', -1)) != int(target_id):
                continue
            widths.append(int(det['bx2']) - int(det['bx1']))
            heights.append(int(det['by2']) - int(det['by1']))
    return _auto_crop_side_from_bbox_sizes(widths, heights), widths, heights


