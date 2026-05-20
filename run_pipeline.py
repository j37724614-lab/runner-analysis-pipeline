"""
run_pipeline.py — 跑者分析完整流程（極簡 CLI 入口，底層由 core.pipeline 驅動）

此檔案已重構為輕量包裝器，其核心排程引擎已移入 core/ 目錄。
"""

import sys
import os
import json
import argparse
from pathlib import Path

from core.pipeline import run_pipeline

# =======================================================================
# CLI 參數解析器
# =======================================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="一鍵跑完：追蹤 → 姿態估計 + 角度計算 → 角度疊加影片"
    )
    parser.add_argument("--gpu",         type=str, default="0",
                        help="CUDA GPU 編號（預設: 0）")
    parser.add_argument("--2d_only",     dest="two_d_only", action="store_true",
                        help="只跑 2D 骨架，跳過 3D 與角度計算")
    parser.add_argument("--skip-track",  dest="skip_track", action="store_true",
                        help="略過 Step 1（追蹤影片已存在時使用）")
    parser.add_argument("--config",      type=str, default=None,
                        help="相機設定 YAML 路徑")
    parser.add_argument("--config-json", dest="config_json", type=str, default=None,
                        help="相機設定 JSON 字串，直接傳入不需建立檔案")
    parser.add_argument("--output-dest", dest="output_dest", type=str, default=None,
                        help="最終輸出目錄（不指定則留在 demo/output/）")
    return parser.parse_args()


def main(default_cameras=None, default_extra_cfg=None):
    args = _parse_args()

    cameras   = default_cameras if default_cameras is not None else []
    extra_cfg = default_extra_cfg if default_extra_cfg is not None else {}

    # 解析相機設定
    if args.config_json:
        try:
            cfg = json.loads(args.config_json)
        except json.JSONDecodeError as e:
            print(f"錯誤：--config-json 格式錯誤：{e}")
            sys.exit(1)
        cameras   = cfg.get("cameras", [])
        extra_cfg = {k: v for k, v in cfg.items() if k != "cameras"}
        if not args.output_dest and "output_dest" in cfg:
            args.output_dest = cfg["output_dest"]

    elif args.config:
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cameras   = cfg.get("cameras", [])
        extra_cfg = {k: v for k, v in cfg.items() if k != "cameras"}
        if not args.output_dest and "output_dest" in cfg:
            args.output_dest = cfg["output_dest"]

    # 執行 core/pipeline.py 中的核心實作
    run_pipeline(
        cameras=cameras,
        extra_cfg=extra_cfg,
        output_dir=args.output_dest,
        gpu=args.gpu,
        only_2d=args.two_d_only,
        skip_track=args.skip_track,
    )


if __name__ == "__main__":
    # 預設的相機設定參數，供單獨執行測試使用
    default_cameras = [
        {
            "video_path": "test/test/cam1.mov",
            "crop": [0, 400, 1920, 800],
            "start_line": [[208, 715], [123, 725]],
            "end_line": [[1760, 710], [1830, 718]],
            "distance_m": 20
        },
        {
            "video_path": "test/test/cam2.mov",
            "crop": [0, 400, 1920, 800],
            "start_line": [[208, 715], [123, 725]],
            "end_line": [[1760, 710], [1830, 718]],
            "distance_m": 20
        }
    ]
    main(default_cameras=default_cameras)
