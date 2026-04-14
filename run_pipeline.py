"""
run_pipeline.py — 一鍵執行完整跑者分析流程

整體流程分四步：
  Step 1│ track_crop_roi.py — YOLO 多相機追蹤，以最快跑者為中心裁剪，
        │                     輸出單一合併影片，並將實際檔名寫入 .last_output_name
  Step 2│ 讀取 .last_output_name 取得動態檔名，複製影片至 vis.py 的輸入目錄
  Step 3│ vis.py — 2D/3D 姿態估計 + 關節角度計算，輸出骨架影片與 CSV
  Step 4│ add_angle_overlay.py — 將 2D 影片與 4 個角度折線圖合併為單一影片

用法：
  python run_pipeline.py                # 完整流程（GPU 0）
  python run_pipeline.py --gpu 1        # 使用 GPU 1
  python run_pipeline.py --2d_only      # 只跑 2D，跳過 3D 與角度計算（Step 4 自動略過）
  python run_pipeline.py --skip-track   # 略過 Step 1，直接從 Step 2 開始
                                        # （追蹤影片已存在時使用，節省時間）
"""

import sys
import os
import shutil
import argparse
import subprocess
import resource

# -----------------------------------------------------------------------
# 啟動時將 RLIMIT_NPROC 軟限制提升至硬限制上限。
# 此設定會被所有子程序繼承，可防止 PyTorch / YOLO 多執行緒
# 在 fork 時因行程數配額不足而拋出 "Resource temporarily unavailable"。
# -----------------------------------------------------------------------
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if _soft < _hard:
        resource.setrlimit(resource.RLIMIT_NPROC, (_hard, _hard))
except Exception:
    pass  # 部分系統不支援或無權限，靜默略過

# =======================================================================
# 路徑設定（只需修改此區即可搬移整個 pipeline）
# =======================================================================

# Step 1：追蹤腳本路徑與輸出目錄
TRACK_SCRIPT  = "/home/jeter/MotionAGFormer/scripts/tracking/track_crop_roi.py"
TRACK_OUT_DIR = "/home/jeter/MotionAGFormer/output_cut"
# track_crop_roi.py 執行完畢後，會將實際輸出的影片檔名（不含路徑）
# 寫入此 marker 檔；step2_copy() 讀取它以取得動態命名的結果。
TRACK_MARKER  = os.path.join(TRACK_OUT_DIR, ".last_output_name")

# Step 3：vis.py 的相關路徑
# vis.py 使用相對路徑載入模型權重，因此必須以 VIS_WORKDIR 作為工作目錄執行。
VIS_WORKDIR   = "/home/jeter/MotionAGFormer/MotionAGFormer"
VIS_SCRIPT    = "demo/vis.py"    # 相對於 VIS_WORKDIR
VIS_VIDEO_DIR = "demo/video"     # vis.py 讀取輸入影片的目錄（相對於 VIS_WORKDIR）

# Step 4：角度疊加腳本
OVERLAY_SCRIPT = "/home/jeter/MotionAGFormer/scripts/visualization/add_angle_overlay.py"

# =======================================================================

# -----------------------------------------------------------------------
# 子程序共用的執行緒限制環境變數。
# NumPy / PyTorch / OpenCV 底層的 BLAS、OpenMP、MKL 預設會啟動多執行緒，
# 在行程數受限（RLIMIT_NPROC = 512）的環境下容易觸發 segfault。
# 統一限制為單執行緒可避免此問題，對 GPU 推論效能影響極小。
# -----------------------------------------------------------------------
THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS":       "1",
    "MKL_NUM_THREADS":       "1",
    "NUMEXPR_NUM_THREADS":   "1",
    "GOMP_SPINCOUNT":        "0",                    # GCC OpenMP 自旋等待關閉
    "OPENCV_FFMPEG_CAPTURE_OPTIONS": "threads;1",    # FFmpeg H.264 解碼器執行緒數
}


def step1_track(gpu: str, config: str = None, config_json: str = None):
    """
    Step 1：執行 track_crop_roi.py，進行 YOLO 多相機追蹤與人物置中裁剪。

    - config      不為 None → 傳 --config <path>（YAML 檔案路徑）
    - config_json 不為 None → 傳 --config-json <str>（JSON 字串，無需建檔）
    - 兩者皆為 None → 沿用腳本內的硬編碼預設值
    - check=True：若子程序返回非零結束碼，自動拋出 CalledProcessError。
    """
    print("=" * 60)
    print("Step 1 / 3 — 多相機追蹤 + 人物置中裁剪")
    print(f"  腳本: {TRACK_SCRIPT}")
    if config:
        print(f"  設定檔: {config}")
    elif config_json:
        print(f"  設定來源: --config-json（JSON 字串）")
    print("=" * 60)
    cmd = [sys.executable, TRACK_SCRIPT]
    if config:
        cmd += ["--config", config]
    elif config_json:
        cmd += ["--config-json", config_json]
    subprocess.run(
        cmd,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, **THREAD_ENV},
        check=True,
    )
    print(f"\nStep 1 完成，影片輸出至: {TRACK_OUT_DIR}/\n")


def step2_copy():
    """
    Step 2：讀取 TRACK_MARKER，找到 Step 1 實際產出的影片，複製到 vis.py 的輸入目錄。

    動態命名機制說明：
      track_crop_roi.py 依第一台有效相機的檔名決定輸出名稱（如 0331-1_tracked.mp4），
      並將檔名字串寫入 .last_output_name。此函式讀取該 marker 後才決定來源路徑，
      避免兩個腳本之間硬編碼同一個字串。

    回傳：
      (video_name, video_name_base)
        video_name      含副檔名的影片檔名，例如 "0331-1_tracked.mp4"
        video_name_base 去除副檔名，例如 "0331-1_tracked"，用作輸出目錄名與前綴
    """
    print("=" * 60)
    print("Step 2 / 3 — 複製影片到 vis.py 輸入目錄")
    print("=" * 60)

    # 讀取 marker 檔（由 track_crop_roi.py 在完成時寫入）
    if not os.path.exists(TRACK_MARKER):
        raise FileNotFoundError(
            f"找不到 marker 檔：{TRACK_MARKER}\n"
            "請先執行 Step 1（不要加 --skip-track）。"
        )
    with open(TRACK_MARKER) as f:
        video_name = f.read().strip()
    if not video_name:
        raise ValueError(f"Marker 檔內容為空：{TRACK_MARKER}")

    # 確認追蹤輸出影片確實存在
    track_output = os.path.join(TRACK_OUT_DIR, video_name)
    if not os.path.exists(track_output):
        raise FileNotFoundError(
            f"找不到追蹤輸出影片：{track_output}\n"
            "請確認 track_crop_roi.py 是否已成功執行。"
        )

    # 複製到 vis.py 的 demo/video/ 目錄
    dest_dir  = os.path.join(VIS_WORKDIR, VIS_VIDEO_DIR)
    dest_path = os.path.join(dest_dir, video_name)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(track_output, dest_path)   # copy2 保留時間戳記等 metadata
    print(f"  來源: {track_output}")
    print(f"  目的: {dest_path}")
    print(f"\nStep 2 完成\n")

    return video_name, os.path.splitext(video_name)[0]


def step3_vis(gpu: str, only_2d: bool, video_name: str, video_name_base: str):
    """
    Step 3：在 VIS_WORKDIR 下執行 vis.py，進行 2D/3D 姿態估計與關節角度計算。
    video_name_base: 不包含 .mp4 副檔名的名字
    參數：
      gpu            CUDA_VISIBLE_DEVICES 的值，決定使用哪張 GPU
      only_2d        True → 只跑 2D 骨架偵測（較快）；False → 2D + 3D + 角度 CSV
      video_name     影片檔名（含副檔名），傳給 vis.py 的 --video 參數
      video_name_base 去除副檔名的名稱，用於印出輸出目錄路徑

    注意：必須以 cwd=VIS_WORKDIR 執行，vis.py 使用相對路徑載入模型與設定。
    """
    print("=" * 60)
    mode = "2D only" if only_2d else "2D + 3D + 角度計算"
    print(f"Step 3 / 3 — 姿態估計（{mode}）")
    print(f"  影片: {video_name}")
    print(f"  工作目錄: {VIS_WORKDIR}")
    print("=" * 60)

    cmd = [sys.executable, VIS_SCRIPT, "--video", video_name, "--gpu", gpu]
    if only_2d:
        cmd.append("--2d_only")

    subprocess.run(
        cmd, cwd=VIS_WORKDIR,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, **THREAD_ENV},
        check=True,
    )

    output_dir = os.path.join(VIS_WORKDIR, "demo", "output", video_name_base)
    print(f"\nStep 3 完成，輸出目錄: {output_dir}")


def step4_overlay(gpu: str, video_name_base: str):
    """
    Step 4：將 2D 骨架影片與 4 個角度折線圖合併為單一影片。

    輸入：
      {output_dir}/{video_name_base}_2D.mp4          ← Step 3 的 2D 影片
      {output_dir}/pred_3D/angles/..._angles.csv     ← Step 3 的角度 CSV

    輸出：
      {output_dir}/{video_name_base}_2D_angles.mp4   ← 合併後影片

    若 CSV 不存在（--2d_only 情境），自動略過並印出提示。
    """
    print("=" * 60)
    print("Step 4 / 4 — 2D 影片 + 角度折線圖合併")
    print("=" * 60)

    out_dir  = os.path.join(VIS_WORKDIR, "demo", "output", video_name_base)
    video_2d = os.path.join(out_dir, video_name_base + "_2D.mp4")
    csv_path = os.path.join(out_dir, "pred_3D", "angles",
                            video_name_base + "_angles.csv")
    output   = os.path.join(out_dir, video_name_base + "_2D_angles.mp4")

    if not os.path.exists(csv_path):
        print(f"  ⚠  角度 CSV 不存在，略過 Step 4")
        print(f"     （若需角度圖請移除 --2d_only 重新執行）")
        return
    if not os.path.exists(video_2d):
        print(f"  ⚠  2D 影片不存在，略過 Step 4: {video_2d}")
        return

    print(f"  2D 影片: {video_2d}")
    print(f"  角度 CSV: {csv_path}")
    print(f"  輸出: {output}")

    subprocess.run(
        [sys.executable, OVERLAY_SCRIPT,
         "--video",  video_2d,
         "--csv",    csv_path,
         "--output", output],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, **THREAD_ENV},
        check=True,
    )
    print(f"\nStep 4 完成：{output}")


def parse_args():
    """
    解析命令列參數。

    --gpu        str   指定 CUDA GPU 編號（預設 "0"）
    --2d_only    flag  只跑 2D 骨架，跳過 3D 與角度計算（速度較快）
    --skip-track flag  略過 Step 1；marker 檔與追蹤影片須已存在，
                       適用於重跑 vis.py 但不需重新追蹤的情況
    """
    parser = argparse.ArgumentParser(
        description="一鍵跑完：track_crop_roi → vis.py 姿態估計 + 角度計算"
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="指定 GPU 編號（預設: 0）"
    )
    parser.add_argument(
        "--2d_only", dest="two_d_only", action="store_true",
        help="只跑 2D 骨架，跳過 3D 與角度計算（速度較快）"
    )
    parser.add_argument(
        "--skip-track", dest="skip_track", action="store_true",
        help="略過 Step 1（追蹤影片已存在時使用）"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="相機設定 YAML 路徑（傳給 track_crop_roi.py）；"
             "未指定時沿用 track_crop_roi.py 內的硬編碼預設值"
    )
    parser.add_argument(
        "--config-json", dest="config_json", type=str, default=None,
        help="相機設定 JSON 字串，直接傳入不需建立檔案；"
             "例：'{\"cameras\":[{\"video_path\":\"/data/cam1.mp4\"}]}'"
    )
    parser.add_argument(
        "--output-dest", dest="output_dest", type=str, default=None,
        help="最終輸出目錄；指定後會將結果複製至此路徑，"
             "PIPELINE_OUTPUT_DIR 也會指向此處"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 從 config 讀取 output_dest（CLI 未指定時才套用）
    if not args.output_dest:
        if args.config_json:
            import json as _json
            _cfg = _json.loads(args.config_json)
            if 'output_dest' in _cfg:
                args.output_dest = _cfg['output_dest']
        elif args.config:
            import yaml as _yaml
            with open(args.config, encoding='utf-8') as _f:
                _cfg = _yaml.safe_load(_f)
            if 'output_dest' in _cfg:
                args.output_dest = _cfg['output_dest']

    print("\n" + "=" * 60)
    print(f"目前使用的 Python 環境：{sys.executable}")
    print("run_pipeline.py — 跑者分析完整流程")
    print(f"  GPU: {args.gpu}")
    print(f"  模式: {'2D only' if args.two_d_only else '2D + 3D + 角度'}")
    print(f"  Step 1: {'略過' if args.skip_track else '執行'}")
    if args.config:
        print(f"  設定檔: {args.config}")
    elif args.config_json:
        print(f"  設定來源: --config-json")
    if args.output_dest:
        print(f"  最終輸出目的地: {args.output_dest}")
    print("=" * 60 + "\n")

    # 預先初始化，確保即使在 try 區塊外使用也不會出現 NameError
    video_name = video_name_base = ""

    try:
        # Step 1：可選略過（--skip-track），適合影片已追蹤完的重跑情境
        if not args.skip_track:
            step1_track(args.gpu, args.config, args.config_json)

        # Step 2：從 marker 取得實際檔名後複製；回傳值供 Step 3 使用
        video_name, video_name_base = step2_copy()

        # Step 3：姿態估計，結果寫入 demo/output/{video_name_base}/
        step3_vis(args.gpu, args.two_d_only, video_name, video_name_base)

        # Step 4：2D 影片 + 角度折線圖合併（若 --2d_only 則自動略過）
        step4_overlay(args.gpu, video_name_base)

    except subprocess.CalledProcessError as e:
        # 子程序（track_crop_roi.py 或 vis.py）返回非零結束碼
        print(f"\n錯誤：子程序執行失敗（回傳碼 {e.returncode})")
        print(f"  指令: {' '.join(str(x) for x in e.cmd)}")
        sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        # marker 檔不存在、內容為空、或追蹤影片遺失
        print(f"\n錯誤：{e}")
        sys.exit(1)

    # 全部完成，印出輸出摘要
    output_dir = os.path.join(VIS_WORKDIR, "demo", "output", video_name_base)

    # --output-dest：將結果複製至後端指定的目的地
    if args.output_dest and video_name_base:
        dest = os.path.join(args.output_dest, video_name_base)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(output_dir, dest)
        print(f"\n結果已複製至: {dest}")
        output_dir = dest

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"輸出目錄: {output_dir}")
    print(f"  {video_name_base}_2D.mp4               ← 2D 骨架疊加影片")
    print(f"  {video_name_base}.mp4                  ← 2D + 3D 並排影片")
    print(f"  {video_name_base}_2D_angles.mp4        ← 2D + 4 角度折線圖")
    print(f"  pred_3D/angles/..._angles.csv          ← 各關節角度 CSV")
    print("=" * 60)
    # 機器可讀輸出：後端從 stdout 最後一行取得輸出目錄
    print(f"PIPELINE_OUTPUT_DIR={output_dir}")


if __name__ == "__main__":
    main()
