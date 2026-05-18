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

# 載入 overlay_original 的主要函數
try:
    from overlay_original import overlay_videos
except ImportError:
    print("找不到 overlay_original.py，請確保 analyze.py 與其在同一目錄下。")
    sys.exit(1)

def run_analysis(config_dict, gpu="0", only_2d=False, skip_track=False, output_dest=None, progress_callback=None):
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
    
    if progress_callback: progress_callback(5)
    
    if not skip_track:
        track_script = os.path.join(os.path.dirname(__file__), "track_runners.py")
        track_cmd = [sys.executable, track_script]
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

    if progress_callback: progress_callback(30)

    print("\n" + "=" * 60)
    print("【階段二】生物力學分析 (run_pipeline.py) — 姿態與關節角度")
    print("=" * 60)
    
    if progress_callback: progress_callback(40)
    
    result = run_pipeline(
        cameras=cameras,
        extra_cfg=extra_cfg,
        output_dir=output_dest,
        gpu=gpu,
        only_2d=only_2d,
        skip_track=skip_track,
        skip_video=True, # 階段二不產生中間過程影片
    )
    
    if progress_callback: progress_callback(80)
    
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
        orig_stem = Path(cameras[0]['video_path']).stem
        offsets_npz = os.path.join(output_dest, f"{orig_stem}_offsets.npz")
        kps_npz = os.path.join(final_pose_dir, "input_2D", "keypoints.npz")
        output_uncropped = os.path.join(output_dest, f"{orig_stem}_uncropped_2D.mp4")

        if os.path.exists(offsets_npz) and os.path.exists(kps_npz):
            if progress_callback: progress_callback(90)
            print("\n" + "=" * 60)
            print("【階段三】匯出未裁切之原比例骨架影片")
            print("=" * 60)
            try:
                overlay_videos(
                    cameras=cameras,
                    offsets_npz=offsets_npz,
                    kps_npz=kps_npz,
                    output_video=output_uncropped,
                    config=config_dict,
                )
                print(f"\n  ▶ 原尺寸未裁切骨架疊加影片: {output_uncropped}")

                # 轉換為網頁相容格式 (加入 metadata / faststart)
                print("\n  ▶ 正在轉換影片格式以支援網頁播放...")
                web_output = os.path.join(output_dest, f"{orig_stem}_uncropped_2D_web.mp4")
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", output_uncropped,
                    "-c:v", "libx264", "-preset", "fast",
                    "-movflags", "+faststart",
                    "-pix_fmt", "yuv420p",
                    web_output
                ]
                try:
                    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    os.replace(web_output, output_uncropped)
                    print(f"  ▶ 影片已成功轉換為網頁相容格式")
                except subprocess.CalledProcessError as e:
                    print(f"  ▶ 影片轉換失敗: {e}")

                # 清理中間過程產生的影片檔案
                print("\n  ▶ 正在清理中間過程影片...")
                if tracked_video and os.path.exists(tracked_video):
                    os.remove(tracked_video)
                    print(f"    - 已移除追蹤影片: {tracked_video}")

            except Exception as e:
                print(f"匯出未裁切影片失敗: {e}")


    print("\n" + "=" * 60)
    
    if progress_callback: progress_callback(100)
    
    total_time = None
    avg_velocity = None
    avg_acceleration = None
    avg_step_length = None ## TODO: add step length

    if os.path.exists(metrics_csv_dest):
        import pandas as pd
        import cv2
        try:
            df = pd.read_csv(metrics_csv_dest)
            if not df.empty:
                fps_val = 60.0
                if cameras and cameras[0].get('video_path'):
                    cap = cv2.VideoCapture(cameras[0]['video_path'])
                    if cap.isOpened():
                        fps_val = cap.get(cv2.CAP_PROP_FPS) or 60.0
                        cap.release()
                
                total_time = float((df["absolute_frame"].max() + 1) / fps_val)
                avg_velocity = float(df["speed_mps"].mean())
                avg_acceleration = float(df["accel_mps2"].mean())
        except Exception as e:
            print(f"Failed to calculate metrics: {e}")

    return {
        "metrics_csv": metrics_csv_dest,
        "angles_csv": angle_csv_dest,
        "uncropped_video": output_uncropped,
        "total_time": total_time,
        "avg_velocity": avg_velocity,
        "avg_acceleration": avg_acceleration,
        "avg_step_length": avg_step_length
    }

if __name__ == "__main__":
    # 在此處直接定義設定參數 (config_dict)，取代原本的 --config / --config-json 命令列輸入
    config_dict = {
        "cameras": [
            {
                "video_path": "test/test/cam1.mov", # 請替換為實際的影片路徑
            },
            {
                "video_path": "test/test/cam2.mov", # 請替換為實際的影片路徑
            }
        ]
        # 若有其他全域設定參數可加在此處
    } 

    # config_dict = {'cameras': [{'video_path': '/home/hsuanya/workspace/running_analysis/backend/data/run_sessions/00399f18-1d1b-40b6-a247-5b44847fa579/c43dcfd1-01fa-4bbc-b11b-71c5a65d696c/cam1.mov',
    #  'start_line': [[177, 710], [77, 725]],
    #   'end_line': [[1752, 698], [1866, 717]],
    #    'distance_m': 20.0}]}

    run_analysis(
        config_dict=config_dict,
        gpu="0",              # CUDA GPU 編號
        only_2d=False,        # 是否只跑 2D 骨架
        skip_track=False,     # 是否略過 Step 1 追蹤
        output_dest=None      # 最終輸出目錄 (設為 None 則預設為第一支影片所在目錄)
    )
