# run_pipeline.py — 開發者筆記

## 一句話說明

`run_pipeline.py` 是整個跑者分析流程的入口腳本：先做多相機 YOLO 追蹤與裁切，再把追蹤影片送進 MotionAGFormer 做 2D/3D 姿態估計與角度計算，最後把 2D 骨架影片和角度折線圖合成一支輸出影片。

---

## 整體流程

```
run_pipeline.py
  │
  ├─ Step 1: scripts/tracking/track_crop_roi.py
  │    YOLO 多相機追蹤、以最快跑者為中心裁剪，輸出合併影片
  │    同時寫入 output_cut/.last_output_name 與 .last_input_video
  │
  ├─ Step 2: step2_copy()
  │    讀取 .last_output_name，找到實際輸出的 tracked mp4
  │    複製到 MotionAGFormer/demo/video/
  │
  ├─ Step 3: MotionAGFormer/demo/vis.py
  │    執行 2D 骨架、3D 姿態估計與角度 CSV 產生
  │    輸出到 MotionAGFormer/demo/output/{video_name_base}/
  │
  └─ Step 4: scripts/visualization/add_angle_overlay.py
       把 2D 骨架影片、角度 CSV、可選的原始主畫面與 frame map 合成
       {video_name_base}_2D_angles.mp4
```

`video_name_base` 是追蹤輸出影片去掉副檔名後的名稱，例如：

```
output_cut/0331-1_tracked.mp4
  ├─ video_name      = 0331-1_tracked.mp4
  └─ video_name_base = 0331-1_tracked
```

---

## 主要路徑

| 變數 | 路徑 / 內容 |
|------|-------------|
| `BASE_DIR` | `run_pipeline.py` 所在目錄，也就是 repo 根目錄 |
| `TRACK_SCRIPT` | `scripts/tracking/track_crop_roi.py` |
| `TRACK_OUT_DIR` | `output_cut/` |
| `TRACK_MARKER` | `output_cut/.last_output_name` |
| `TRACK_INPUT_MARKER` | `output_cut/.last_input_video` |
| `TRACK_INPUTS_MARKER` | `output_cut/.last_input_videos` |
| `VIS_WORKDIR` | `MotionAGFormer/` |
| `VIS_SCRIPT` | `demo/vis.py`，相對於 `VIS_WORKDIR` |
| `VIS_VIDEO_DIR` | `demo/video`，相對於 `VIS_WORKDIR` |
| `OVERLAY_SCRIPT` | `scripts/visualization/add_angle_overlay.py` |

重點：`vis.py` 依賴相對路徑載入模型和設定，所以 Step 3 必須用 `cwd=MotionAGFormer` 執行。

---

## 啟動時的資源限制處理

腳本一開始會嘗試把 `RLIMIT_NPROC` 的 soft limit 提升到 hard limit：

```python
_soft, _hard = resource.getrlimit(resource.RLIMIT_NPROC)
resource.setrlimit(resource.RLIMIT_NPROC, (_hard, _hard))
```

用途是降低 PyTorch / YOLO / OpenCV 在 fork 或啟動多執行緒時遇到 `Resource temporarily unavailable` 的機率。若系統不支援或權限不足，會靜默略過。

子程序也會統一套用 `THREAD_ENV`，把 BLAS、OpenMP、MKL、NumExpr、OpenCV FFmpeg 解碼執行緒限制為 1，避免在行程數受限的機器上 segfault。

---

## Step 1：追蹤與裁切

函式：`step1_track(gpu, config=None, config_json=None)`

執行：

```bash
python scripts/tracking/track_crop_roi.py
```

會依參數選擇傳入：

| 條件 | 傳給 `track_crop_roi.py` |
|------|--------------------------|
| 有 `--config` | `--config <path>` |
| 有 `--config-json` | `--config-json <json-string>` |
| 都沒有 | 使用 `track_crop_roi.py` 內部硬編碼預設值 |

環境變數會包含：

```text
CUDA_VISIBLE_DEVICES=<gpu>
THREAD_ENV...
```

Step 1 成功後，`track_crop_roi.py` 預期會在 `output_cut/` 產生：

| 檔案 | 用途 |
|------|------|
| `{video_name_base}.mp4` | 追蹤裁切後的影片 |
| `.last_output_name` | 記錄實際輸出的影片檔名，供 Step 2 使用 |
| `.last_input_video` | 記錄第一台有效相機的原始影片，保留舊版相容 |
| `.last_input_videos` | 記錄所有有效相機的原始影片，每行一支，供 Step 4 依 `cam` 欄位切換主畫面 |
| `{video_name_base}_bbox_map.csv` | 若存在，Step 3 會傳給 `vis.py --bbox-csv` |
| `{video_name_base}_frame_map.csv` | 若存在，Step 4 會傳給 overlay 腳本；欄位含 `output_frame,cam,cam_frame,source_frame` |

---

## Step 2：複製影片到 MotionAGFormer

函式：`step2_copy()`

流程：

1. 檢查 `output_cut/.last_output_name` 是否存在。
2. 讀出 `video_name`。
3. 確認 `output_cut/{video_name}` 存在。
4. 複製到 `MotionAGFormer/demo/video/{video_name}`。
5. 回傳 `(video_name, video_name_base)`。

這一步的設計是為了避免硬編碼輸出檔名。追蹤腳本可以依第一台有效相機動態命名，pipeline 只透過 marker 取得實際檔名。

常見錯誤：

| 錯誤 | 原因 |
|------|------|
| 找不到 `.last_output_name` | 沒跑 Step 1，或錯誤使用 `--skip-track` |
| marker 內容為空 | Step 1 沒正確寫入輸出名稱 |
| 找不到 `output_cut/{video_name}` | 追蹤輸出影片不存在或被移動 |

---

## Step 3：姿態估計與角度計算

函式：`step3_vis(gpu, only_2d, video_name, video_name_base)`

在 `MotionAGFormer/` 目錄下執行：

```bash
python demo/vis.py --video <video_name> --gpu <gpu>
```

若 `output_cut/{video_name_base}_bbox_map.csv` 存在，額外加入：

```bash
--bbox-csv output_cut/{video_name_base}_bbox_map.csv
```

若使用 `--2d_only`，額外加入：

```bash
--2d_only
```

輸出目錄：

```text
MotionAGFormer/demo/output/{video_name_base}/
```

完整模式預期會產生：

| 檔案 | 說明 |
|------|------|
| `{video_name_base}_2D.mp4` | 2D 骨架疊加影片 |
| `{video_name_base}.mp4` | 2D + 3D 並排影片 |
| `pred_3D/angles/{video_name_base}_angles.csv` | 關節角度 CSV |

`--2d_only` 模式下不會產生 3D 與角度 CSV，因此 Step 4 會自動略過。

---

## Step 4：合併 2D 影片與角度折線圖

函式：`step4_overlay(gpu, video_name_base, main_video_path=None)`

輸入：

| 來源 | 路徑 |
|------|------|
| 2D 影片 | `MotionAGFormer/demo/output/{video_name_base}/{video_name_base}_2D.mp4` |
| 角度 CSV | `MotionAGFormer/demo/output/{video_name_base}/pred_3D/angles/{video_name_base}_angles.csv` |
| frame map | `output_cut/{video_name_base}_frame_map.csv`，存在才傳入 |
| 原始主畫面 | config 內所有有效相機的 `video_path`，或 `.last_input_videos` |

輸出：

```text
MotionAGFormer/demo/output/{video_name_base}/{video_name_base}_2D_angles.mp4
```

執行指令形式：

```bash
python scripts/visualization/add_angle_overlay.py \
  --video <2d-video> \
  --csv <angles-csv> \
  --output <output-video> \
  [--main-videos <cam1-video> <cam2-video> ...] \
  [--frame-map <frame-map-csv>]
```

多相機時，overlay 會讀 `frame_map` 的 `cam` 欄位，使用 `cam=1` 對應 `--main-videos` 的第一支影片、`cam=2` 對應第二支，以此類推。若只傳舊版 `--main-video`，所有輸出幀都只會從同一支原始影片取畫面。

overlay 預設會對跨相機接縫做顯示層平滑：目前只處理 `pelvis_torso_angle`。切到新相機時，第一幀會用 offset 接上上一台相機的最後角度，然後在 `--boundary_blend_frames` 指定的幀數內線性回到新相機原本估計值。這只影響圖表顯示，不會改寫原始 angles CSV。若要關閉：

```bash
python scripts/visualization/add_angle_overlay.py ... --no-boundary-smooth
```

略過條件：

| 條件 | 行為 |
|------|------|
| 角度 CSV 不存在 | 印出提示並略過，通常是 `--2d_only` |
| 2D 影片不存在 | 印出提示並略過 |

---

## `main_video_paths` 的解析規則

函式：`_resolve_main_video_paths(config=None, config_json=None)`

用途是找出 Step 4 要疊加或對齊的所有原始主畫面。

解析順序：

1. 若有 `--config-json`，解析 JSON，取 `cameras` 中所有有 `video_path` 的相機。
2. 若有 `--config`，讀 YAML，取 `cameras` 中所有有 `video_path` 的相機。
3. 若沒有 config，嘗試讀 `output_cut/.last_input_videos`。
4. 若沒有 `.last_input_videos`，退回讀 `output_cut/.last_input_video`。
5. 都沒有就回傳空陣列，Step 4 不傳 `--main-videos`。

---

## CLI 參數

| 參數 | 預設 | 說明 |
|------|------|------|
| `--gpu` | `"0"` | 指定 CUDA GPU 編號，也會設為 `CUDA_VISIBLE_DEVICES` |
| `--2d_only` | `False` | 只跑 2D，跳過 3D 與角度計算；Step 4 通常會略過 |
| `--skip-track` | `False` | 略過 Step 1，直接使用既有 `output_cut/.last_output_name` |
| `--config` | `None` | YAML 相機設定檔，會傳給追蹤腳本 |
| `--config-json` | `None` | JSON 字串形式的相機設定，適合後端直接呼叫 |
| `--output-dest` | `None` | 最終輸出目錄；完成後會把結果資料夾複製到這裡 |

`--config` 與 `--config-json` 同時存在時，Step 1 會優先使用 `--config`。`output_dest` 也是 CLI 優先；CLI 沒指定時才從 config 裡讀 `output_dest`。

---

## `output_dest` 行為

`main()` 在所有 step 成功後才處理 `output_dest`。

來源優先序：

1. CLI `--output-dest`
2. `--config-json` 內的 `output_dest`
3. `--config` YAML 內的 `output_dest`

複製規則：

```text
source = MotionAGFormer/demo/output/{video_name_base}
dest   = {output_dest}/{video_name_base}
```

如果 `dest` 已存在，程式會先 `shutil.rmtree(dest)` 再 `shutil.copytree(source, dest)`。這是覆蓋式複製，使用時要注意目的地同名資料夾會被刪掉。

最後 stdout 的最後一行會印：

```text
PIPELINE_OUTPUT_DIR=<output_dir>
```

後端可以用這行取得最終結果路徑；若有 `output_dest`，這裡會指向複製後的位置。

---

## 常用指令

完整流程：

```bash
python run_pipeline.py --config my_config.yaml
```

指定 GPU：

```bash
python run_pipeline.py --config my_config.yaml --gpu 1
```

只跑 2D：

```bash
python run_pipeline.py --config my_config.yaml --2d_only
```

追蹤已完成，只重跑姿態估計與 overlay：

```bash
python run_pipeline.py --config my_config.yaml --skip-track
```

後端用 JSON 字串傳設定：

```bash
python run_pipeline.py --config-json '{"cameras":[{"video_path":"/data/cam1.mp4"}]}'
```

指定最終輸出目的地：

```bash
python run_pipeline.py --config my_config.yaml --output-dest /tmp/pipeline_outputs
```

---

## 錯誤處理

`main()` 只明確捕捉兩類錯誤：

| 例外 | 處理 |
|------|------|
| `subprocess.CalledProcessError` | 子程序非零退出，印出 return code 與指令，`sys.exit(1)` |
| `FileNotFoundError` / `ValueError` | marker、追蹤影片或 marker 內容錯誤，印出訊息，`sys.exit(1)` |

其他例外，例如 JSON/YAML 解析錯、`copytree` 權限問題、`rmtree` 失敗，會直接往外拋出 traceback。

---

## 下次修改時要注意

- Step 顯示文字目前有一點不一致：Step 1/2/3 印的是 `/ 3`，但實際流程有 Step 4。
- `--skip-track` 依賴 `output_cut/.last_output_name` 和追蹤影片仍存在；如果要給後端穩定重跑，需要保留這些檔案。
- `--output-dest` 會刪除目的地同名資料夾再複製，這是有破壞性的覆蓋行為。
- `vis.py` 必須以 `MotionAGFormer/` 作為工作目錄執行，不能任意改成 repo root。
- `--config` 與 `--config-json` 都傳入時，目前優先使用 `--config`。
- `THREAD_ENV` 是為了穩定性設計，若拿掉可能在受限環境出現 fork 或多執行緒相關錯誤。
