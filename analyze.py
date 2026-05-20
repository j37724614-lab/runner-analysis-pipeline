"""
analyze.py — 綜合分析入口

整合了速度與加速度追蹤 (track_runners.py) 以及姿態與關節角度分析 (run_pipeline.py)。
透過此腳本，可以一鍵完成所有分析流程。
"""

from core.pipeline import run_analysis

if __name__ == "__main__":
    # 在此處直接定義設定參數 (config_dict)，取代原本的 --config / --config-json 命令列輸入
    config_dict = {
        "cameras": [
            {
                "video_path": "test/test/cam1.mov",
                "crop": [0, 400, 1920, 800],
                "start_line": [[222, 715], [148, 725]],
                "end_line": [[1700, 710], [1790, 718]],
                "distance_m": 20.0
            },
            {
                "video_path": "test/test/cam2.mov",
                "crop": [0, 400, 1920, 800],
                "start_line": [[220, 715], [135, 725]],
                "end_line": [[1730, 710], [1825, 725]],
                "distance_m": 20.0
            }
        ]
    } 

    run_analysis(
        config_dict=config_dict,
        gpu="0",              # CUDA GPU 編號
        only_2d=False,        # 是否只跑 2D 骨架
        skip_track=False,     # 是否略過 Step 1 追蹤
        output_dest=None      # 最終輸出目錄 (設為 None 則預設為第一支影片所在目錄)
    )
