"""
analyze.py — 綜合分析入口

整合了速度與加速度追蹤 (track_runners.py) 以及姿態與關節角度分析 (run_pipeline.py)。
透過此腳本，可以一鍵完成所有分析流程。
"""

import sys
import os
import json
import argparse
import subprocess
import yaml
from pathlib import Path

# 載入 run_pipeline 的主要函數
try:
    from run_pipeline import run_pipeline
except ImportError:
    print("找不到 run_pipeline.py，請確保 analyze.py 與其在同一目錄下。")
    sys.exit(1)

def run_analysis(config_dict, gpu="0", only_2d=False, skip_track=False, output_dest=None):
    """
    執行完整分析流程：運動表現分析 (Phase 1) + 生物力學分析 (Phase 2) + 原影片疊加 (Phase 3)。
    
    參數：
        config_dict (dict): 包含相機設定與全域參數的字典
        gpu (str): 使用的 GPU 編號
        only_2d (bool): 是否只執行 2D 姿態估計
        skip_track (bool): 是否跳過 Phase 1 追蹤
        output_dest (str): 輸出目錄
    """
    cameras = config_dict.get("cameras", [])
    if not output_dest:
        output_dest = config_dict.get("output_dest")
    
    # 如果未指定輸出目錄，預設為第一支影片所在的資料夾
    if not output_dest and cameras:
        for cam in cameras:
            video_path = cam.get("video_path")
            if video_path:
                output_dest = os.path.dirname(os.path.abspath(video_path))
                break
    
    if output_dest:
        os.makedirs(output_dest, exist_ok=True)
        config_dict["output_dir"] = output_dest
        
    extra_cfg = {k: v for k, v in config_dict.items() if k != "cameras"}

    print("=" * 60)
    print("【階段一】運動表現分析 (track_runners.py) — 速度與加速度")
    print("=" * 60)
    
    if not skip_track:
        track_cmd = [sys.executable, "track_runners.py"]
        track_config = config_dict.copy()
        track_config["gpu"] = gpu
        track_config["skip_video"] = True  # 階段一不產生影片
        track_cmd.extend(["--config-json", json.dumps(track_config)])
            
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            subprocess.run(track_cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"track_runners.py 執行失敗: {e}")
            return None
    else:
        print("使用者指定 skip_track，略過運動表現分析影片生成。")

    print("\n" + "=" * 60)
    print("【階段二】生物力學分析 (run_pipeline.py) — 姿態與關節角度")
    print("=" * 60)
    
    result = run_pipeline(
        cameras=cameras,
        extra_cfg=extra_cfg,
        output_dir=output_dest,
        gpu=gpu,
        only_2d=only_2d,
        skip_track=skip_track,
        skip_video=True, # 階段二不產生中間過程影片
    )
    
    import shutil
    
    # 輸出統整結果
    print("\n" + "★" * 60)
    print("🎉 所有分析流程已順利完成！ 🎉")
    print("★" * 60)
    
    track_out_dir = config_dict.get("output_dir", "output_cut")
    track_out_name = config_dict.get("output_name", "sequential_tracked.mp4")
    
    # 強制將 track_runners 的 CSV 改名為 metrics.csv
    track_csv_orig = os.path.join(track_out_dir, track_out_name.replace(".mp4", "_metrics.csv"))
    metrics_csv_dest = os.path.join(output_dest if output_dest else track_out_dir, "metrics.csv")
    
    if os.path.exists(track_csv_orig) and track_csv_orig != metrics_csv_dest:
        os.rename(track_csv_orig, metrics_csv_dest)
        track_csv_display = metrics_csv_dest
    else:
        track_csv_display = track_csv_orig if os.path.exists(track_csv_orig) else "未生成"

    print("\n[分析結果]")
    print(f"  ▶ 運動表現分析資料 (CSV): {track_csv_display}")
    
    print("\n[生物力學分析結果]")
    final_pose_dir = result.get('output_dir', '未定義')
    
    # 將 run_pipeline 產生的目錄重新命名為 sequential_tracked
    expected_pose_dir = os.path.join(output_dest, "sequential_tracked")
    if os.path.exists(final_pose_dir) and final_pose_dir != expected_pose_dir:
        if os.path.exists(expected_pose_dir):
            shutil.rmtree(expected_pose_dir)
        os.rename(final_pose_dir, expected_pose_dir)
        final_pose_dir = expected_pose_dir

    print(f"  ▶ 姿態分析資料夾: {final_pose_dir}")
         
    # 將 run_pipeline 產生的 CSV 提取出來，並重新命名為 angles.csv
    tracked_video = result.get('tracked_video')
    angle_csv_dest = None
    if tracked_video:
        video_stem = Path(tracked_video).stem
        angle_csv_orig = os.path.join(final_pose_dir, "pred_3D", "angles", f"{video_stem}_angles.csv")
        
        if os.path.exists(angle_csv_orig) and output_dest:
            angle_csv_dest = os.path.join(output_dest, "angles.csv")
            try:
                os.rename(angle_csv_orig, angle_csv_dest)
                print(f"  ▶ 關節角度資料 (CSV): {angle_csv_dest}")
            except Exception as e:
                print(f"  ▶ 關節角度資料 (CSV): {angle_csv_orig} (重新命名失敗: {e})")
        elif os.path.exists(angle_csv_orig):
            angle_csv_dest = angle_csv_orig
            print(f"  ▶ 關節角度資料 (CSV): {angle_csv_orig}")
            
    # 【階段三】產出原比例骨架與標線影片
    output_uncropped = None
    if cameras:
        orig_video = cameras[0].get('video_path')
        if orig_video:
            orig_stem = Path(orig_video).stem
            offsets_npz = os.path.join(output_dest, f"{orig_stem}_offsets.npz")
            kps_npz = os.path.join(final_pose_dir, "input_2D", "keypoints.npz")
            output_uncropped = os.path.join(output_dest, f"{orig_stem}_uncropped_2D.mp4")
            
            if os.path.exists(offsets_npz) and os.path.exists(kps_npz):
                print("\n" + "=" * 60)
                print("【階段三】匯出未裁切之原比例骨架影片")
                print("=" * 60)
                overlay_cmd = [
                    sys.executable, "overlay_original.py",
                    "--orig_video", orig_video,
                    "--offsets_npz", offsets_npz,
                    "--kps_npz", kps_npz,
                    "--config_json", json.dumps(config_dict),
                    "--output_video", output_uncropped
                ]
                try:
                    subprocess.run(overlay_cmd, check=True)
                    print(f"\n  ▶ 原尺寸未裁切骨架疊加影片: {output_uncropped}")
                    
                    # 清理中間過程產生的影片檔案
                    print("\n  ▶ 正在清理中間過程影片...")
                    if os.path.exists(tracked_video):
                        os.remove(tracked_video)
                        print(f"    - 已移除追蹤影片: {tracked_video}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"匯出未裁切影片失敗: {e}")

    print("\n" + "=" * 60)
    return {
        "metrics_csv": metrics_csv_dest,
        "angles_csv": angle_csv_dest,
        "uncropped_video": output_uncropped
    }

def _parse_args():
    parser = argparse.ArgumentParser(
        description="一鍵執行速度分析與姿態角度分析"
    )
    parser.add_argument("--gpu",         type=str, default="0",
                        help="CUDA GPU 編號（預設: 0）")
    parser.add_argument("--2d_only",     dest="two_d_only", action="store_true",
                        help="只跑 2D 骨架，跳過 3D 與角度計算")
    parser.add_argument("--skip-track",  dest="skip_track", action="store_true",
                        help="略過 Step 1 追蹤（當追蹤影片已存在時使用）")
    parser.add_argument("--config",      type=str, default=None,
                        help="相機設定 YAML 路徑")
    parser.add_argument("--config-json", dest="config_json", type=str, default=None,
                        help="相機設定 JSON 字串")
    parser.add_argument("--output-dest", dest="output_dest", type=str, default=None,
                        help="最終姿態分析輸出目錄")
    return parser.parse_args()

def main():
    args = _parse_args()
    
    config_dict = {}
    if args.config_json:
        try:
            config_dict = json.loads(args.config_json)
        except json.JSONDecodeError as e:
            print(f"錯誤：--config-json 格式錯誤：{e}")
            sys.exit(1)
    elif args.config:
        with open(args.config, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    
    if not config_dict and not args.skip_track:
        print("請提供 --config 或 --config-json")
        sys.exit(1)

    run_analysis(
        config_dict=config_dict,
        gpu=args.gpu,
        only_2d=args.two_d_only,
        skip_track=args.skip_track,
        output_dest=args.output_dest
    )

if __name__ == "__main__":
    main()
