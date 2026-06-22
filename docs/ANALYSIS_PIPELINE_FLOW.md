# Runner Analysis Pipeline Flow

本文整理目前系統從前端送出影片到後端輸出分析結果的完整流程。內容以目前程式行為為準，重點包含資料進出、主要處理階段、產物檔案、失敗條件，以及前端為什麼可能顯示 `N/A`。

## 1. 整體入口

分析流程由後端 API 觸發。

主要入口：

- 後端：`/home/jeter/running-analysis-backend/running-analysis-backend/routes/upload.py`
- Pipeline：`/home/jeter/runner-analysis-pipeline/core/pipeline.py`
- 步頻/步幅分析：`/home/jeter/runner-analysis-pipeline/scripts/analysis/ankle_step_stride.py`

前端送出影片與錨點後，後端會：

1. 建立 `run_session`
2. 儲存影片到該 session 目錄
3. 把前端錨點轉成 pipeline config
4. 背景執行 `run_analysis()`
5. 分析完成後寫入 `analysis_meta`
6. 前端讀取 session、影片、圖表與 CSV 結果

## 2. 輸入資料

每次分析主要需要：

- 原始影片：`cam1.mov`, `cam2.mov` 等
- 前端標定錨點：每台 camera 4 個點
- 距離校正：`top_distance_m`, `bottom_distance_m`
- runner/session metadata：跑者、日期、fps、camera 數量

目前錨點會被轉成：

```text
start_line = [左側線上點, 左側線下點]
end_line   = [右側線上點, 右側線下點]
distance_m = top/bottom distance 平均值
```

目前步幅公尺換算不使用 homography/H matrix，而是使用：

```text
meters_per_pixel = distance_m / start_line_midpoint 到 end_line_midpoint 的 pixel 距離
```

再用腳踝點沿跑道方向的 pixel 投影距離換算成公尺。

## 3. Session 目錄結構

一次分析會輸出到：

```text
/home/jeter/running-analysis-backend/data/run_sessions/{runner_id}/{run_session_id}/
```

常見輸出：

```text
metadata.json
.config.json
cam1.mov
cam1_tracked_cam1_overview.mp4
cam1_tracked_frame_map.csv
cam1_tracked_bbox_map.csv
cam1_offsets.npz
sequential_tracked/
metrics.csv
angles.csv
cam1_ankle_positions.csv
cam1_step_events.csv
output_final.mp4
final_video_frames/output_final/
```

如果分析失敗，可能只會有前半段檔案，例如 `.config.json`、`metadata.json`、overview，但沒有 `metrics.csv`、`cam1_step_events.csv`、`output_final.mp4`。

## 4. 後端流程

後端入口在 `routes/upload.py`。

### 4.1 建立 Session

收到上傳資料後，後端會建立：

- `run_session`
- `video`
- `metadata.json`
- `.config.json`

`run_session` 初始狀態：

```text
status = pending
progress = 0
```

開始背景分析後：

```text
status = processing
```

完成後：

```text
status = done
progress = 100
```

失敗後：

```text
status = failed
```

### 4.2 產生 Config

後端會把前端 normalized anchors 轉成影片 pixel 座標。

目前 config 主要欄位：

```json
{
  "cameras": [
    {
      "video_path": ".../cam1.mov",
      "start_line": [[x1, y1], [x2, y2]],
      "end_line": [[x3, y3], [x4, y4]],
      "distance_m": 20.0
    }
  ],
  "output_dir": "..."
}
```

注意：目前新 config 不再寫入 `homography_src_points` / `homography_dst_world`。即使舊 config 有 homography，步幅分析也會優先使用 `start_line/end_line/distance_m` 的 pixel line calibration。

### 4.3 寫入 DB 結果

分析成功後，後端會把 summary 寫入 `analysis_meta`：

- `total_time`
- `avg_velocity`
- `avg_acceleration`
- `avg_step_length`
- `summary`

如果沒有 `analysis_meta`，前端多數總結欄位會顯示 `N/A`。

## 5. Pipeline 總流程

核心入口：

```python
run_analysis(config_dict, ...)
```

內部主要呼叫：

```python
run_pipeline(...)
```

目前大流程：

```text
前端上傳
  ↓
後端建立 run_session + config
  ↓
run_analysis()
  ↓
Step 1: YOLO tracking + 置中裁剪
  ↓
Step 2: HRNet / MotionAGFormer 2D/3D pose + angles
  ↓
速度/加速度分析
  ↓
原圖骨架 overlay + 步頻/步幅分析並行
  ↓
步頻步幅標注影片
  ↓
轉 Web 播放格式 + 匯出逐幀圖
  ↓
寫入 DB analysis_meta
  ↓
前端顯示結果
```

## 6. Step 1：YOLO Tracking + 置中裁剪

程式位置：

```text
core/pipeline.py -> step1_track()
core/tracking.py
core/tracker_impl.py
scripts/tracking/track_crop_roi.py
```

目的：

- 在原始影片中偵測/追蹤跑者
- 選出目標 runner
- 依 bbox 產生置中裁剪影片
- 建立 frame 對應表與 bbox 對應表

主要輸出：

```text
cam1_tracked.mp4
cam1_tracked_cam1_overview.mp4
cam1_tracked_frame_map.csv
cam1_tracked_bbox_map.csv
cam1_offsets.npz
cam1_selected_runner.json
cam1_track_summary.csv
cam1_all_tracks.csv
```

關鍵用途：

- `cam1_tracked.mp4`：給 MotionAGFormer / HRNet 做姿態估計
- `cam1_offsets.npz`：把裁剪影片中的骨架點貼回原圖座標
- `cam1_tracked_bbox_map.csv`：速度、距離、bbox-based metrics 來源
- `cam1_tracked_frame_map.csv`：裁剪輸出 frame 對應原影片 frame

常見失敗條件：

- YOLO 沒有穩定偵測到跑者
- 錨點順序錯誤，導致起終點線或跑道方向錯
- 追蹤判斷跑者從未越過起跑線，輸出 0 幀
- `bbox_map.csv` 只有 header，後續 pose 無法產生 keypoints

如果 Step 1 輸出 0 幀，後面通常會失敗，前端結果會是 `N/A`。

## 7. Step 2：2D/3D Pose + Angles

程式位置：

```text
core/pipeline.py -> step2_pose()
MotionAGFormer/demo/vis.py
```

目的：

- 對 Step 1 的裁剪追蹤影片做 2D pose
- 產生 3D pose
- 計算膝蓋、髖、踝等角度
- 產生 pose 影片與角度 CSV

主要輸出：

```text
sequential_tracked/input_2D/keypoints.npz
sequential_tracked/input_2D/keypoints_raw.npz
sequential_tracked/hrnet_confidence.csv
sequential_tracked/keypoint_smoothing_debug.csv
sequential_tracked/pred_3D/3Dkeypoints.npz
angles.csv
knee_angle_corrections.csv
```

補充：

- `keypoints_raw.npz` 是 HRNet 原始 keypoints
- `keypoints.npz` 是後續平滑/補幀後的 keypoints
- `valid_frames`、`offsets` 會用來對回原影片座標

常見失敗條件：

- Step 1 追蹤影片 0 frame
- bbox map 沒有有效資料
- HRNet 沒有產生任何 keypoints
- keypoints shape 不符合後續流程預期

## 8. 速度與加速度分析

程式位置：

```text
core/pipeline.py
core/tracker_impl.py -> compute_speed_from_bbox_map()
```

速度分析來源是：

```text
cam1_tracked_bbox_map.csv
```

目前後端 web flow 的速度/距離來源是 `bbox_map.csv` 與 `start_line/end_line/distance_m` 的 pixel line calibration，不會為速度再重跑 YOLO，也不會使用前端四錨點 homography。

補充：`core/tracker_impl.py` 本身仍保留 homography 距離換算能力；若 standalone config 明確提供有效 H matrix，速度計算可使用 homography。只是目前 web/API 送入 `run_analysis()` 的 tracking config 會移除 homography 欄位，避免速度與步幅被不穩定 H matrix 影響。

主要輸出：

```text
metrics.csv
```

常見欄位：

```text
frame
time_s
position_m
speed_mps
accel_mps2
is_interpolated
interp_gap_len
speed_confidence
```

DB summary 會從 `metrics.csv` 算：

- `total_time`
- `avg_velocity`
- `avg_acceleration`

如果 `metrics.csv` 不存在或空，相關欄位可能變成 `N/A`。

## 9. 原圖骨架 Overlay

程式位置：

```text
core/pipeline.py
core/overlay.py
```

目的：

- 把 Step 2 產生的骨架，依 `offsets.npz` 貼回原始影片座標
- 合成原解析度骨架影片

主要輸出：

```text
output_final.mp4
final_video_frames/output_final/
```

注意：

- 如果 Step 1 的 frame map / offsets 不正確，骨架貼回原圖會錯位
- 如果 keypoints 使用了補幀/interp，但 frame 對應沒有一致，overlay 也會錯位

## 10. 步頻與步幅分析

程式位置：

```text
scripts/analysis/ankle_step_stride.py
```

入口：

```python
run_step_stride_analysis(...)
```

主要輸入：

```text
sequential_tracked/input_2D/keypoints.npz
cam1_offsets.npz
.config.json
```

主要輸出：

```text
cam1_ankle_positions.csv
cam1_step_events.csv
```

### 10.1 腳踝座標產生

系統會從 keypoints 讀：

```text
RIGHT_ANKLE = 3
LEFT_ANKLE = 6
```

每一幀會輸出：

```text
right_ankle_x/y/conf
left_ankle_x/y/conf
lower_foot
lower_ankle_x/y/conf
```

`lower_ankle_y` 是左右腳踝中畫面 y 座標較低者，用來降低左右腳 label swap 的影響。

### 10.2 Smooth touchdown 主偵測

主訊號：

```text
lower_ankle_y
```

先做 6Hz Butterworth low-pass smoothing，再用 `find_peaks()` 找 local max。

主要參數：

```text
peak_distance = max(3, int(fps / 8.0))
prominence = max(IQR(y_smooth) * 0.20, 1.5)
```

這一步找的是主要 touchdown 候選。

### 10.3 RAW Rescue 補點

RAW rescue 的目的：

```text
smooth 訊號如果疑似漏掉 touchdown，才從 raw lower_ankle_y 裡找補點。
```

raw peak 設定：

```text
raw_prominence = max(IQR(raw_y) * 0.20, 1.5)
raw_peaks = find_peaks(raw_y, distance=peak_distance, prominence=raw_prominence)
```

RAW rescue 觸發條件有兩種。

時間 gap 過大：

```text
min_step_frames = max(4, int(fps / 4.5))
peak_distance = max(3, int(fps / 8.0))
rescue_gap_frames = max(min_step_frames + peak_distance, round(min_step_frames * 1.8))
```

約等於：

```text
30fps: 11 frames，約 0.37 秒
60fps: 23 frames，約 0.38 秒
```

步幅/位移 gap 過大：

```text
large_step_threshold = min(
  median_smooth_step_px * 1.6,
  2.6m / meters_per_pixel
)
```

目前 `2.6m` 會先換成 pixel，再跟實際 pixel 投影位移比較。

RAW 候選還要通過：

- 必須在前後兩個 smooth touchdown 中間
- 距離前後 touchdown 至少 `min_touchdown_gap_frames`
- `ankle_conf >= 0.50`
- `raw prominence >= raw_prominence`
- 位置在前後 touchdown 的跑道位置中間附近
- ankle y 高度合理
- 位置切分合理，不能太靠近前一點或後一點

如果同一段有多個 raw 候選，只取分數最高者：

```text
(prominence, ankle_conf, ankle_y)
```

### 10.4 短步幅去重

去重門檻：

```text
min_step_length_m = 0.60
```

功能：

```text
如果兩個連續 touchdown 的跑道方向距離 < 0.60m，
視為同一次落地被重複偵測，只保留分數較高的一個。
```

目前也是用 pixel line calibration：

```text
0.60m / meters_per_pixel = 最小有效步長 pixel 門檻
```

### 10.5 步幅換算

目前步幅換算不用 homography。

流程：

```text
腳踝點 -> 投影到 start/end line 方向的一維 pixel 位置
相鄰 touchdown 的 pixel 差 -> step_length_px
step_length_px * meters_per_pixel -> step_length_m
```

輸出到：

```text
cam1_step_events.csv
```

常見欄位：

```text
step_index
seq_frame
orig_frame
time_s
seq_time_s
cam
foot
ankle_x
ankle_y
ankle_conf
track_position_px
step_length_px
step_length_m
cadence_spm
avg_cadence_spm
```

## 11. 步頻步幅標注影片

程式位置：

```text
scripts/analysis/ankle_step_stride.py -> annotate_step_stride_video()
```

目的：

- 把 `cam1_step_events.csv` 的 touchdown 點標到 `output_final.mp4`
- 顯示 `S1`, `S2`, `L=...m`
- 顯示 cumulative steps

這個階段需要：

- 原圖骨架 overlay 影片完成
- step analysis 完成

## 12. 影片轉檔與逐幀輸出

分析最後會：

1. 把 `output_final.mp4` 轉成 Web 播放相容格式
2. 匯出逐幀 PNG
3. 複製 Web 影片到 keypoints archive
4. 清理中間的 `cam1_tracked.mp4`

常見輸出：

```text
output_final.mp4
final_video_frames/output_final/*.png
keypoints_raw_archive/{timestamp}_cam1_tracked/output_final.mp4
```

## 13. 前端讀取與 N/A 條件

前端主要透過後端 API 讀：

- run session 狀態
- analysis meta
- output video
- graphs
- CSV 結果

如果 session 是：

```text
status = failed
```

或沒有：

```text
analysis_meta
metrics.csv
cam1_step_events.csv
output_final.mp4
```

前端就可能顯示：

```text
N/A
```

典型原因：

- Step 1 tracking 產生 0 幀
- 錨點順序錯誤，跑者被判定沒有越過起跑線
- bbox map 只有 header
- pose 階段沒有產生 keypoints
- DB 寫入 `analysis_meta` 失敗

## 14. 主要輸出對應表

| 輸出 | 來源階段 | 用途 |
|---|---|---|
| `.config.json` | 後端 config | pipeline camera 設定與距離校正 |
| `metadata.json` | 後端 | session/camera/anchor metadata |
| `cam1_tracked.mp4` | Step 1 | 裁剪追蹤影片，供 pose 使用 |
| `cam1_tracked_bbox_map.csv` | Step 1 | bbox 位置、速度距離分析 |
| `cam1_tracked_frame_map.csv` | Step 1 | 裁剪 frame 對原影片 frame |
| `cam1_offsets.npz` | Step 1 | 骨架貼回原圖座標 |
| `keypoints_raw.npz` | Step 2 | HRNet 原始 keypoints |
| `keypoints.npz` | Step 2 | 平滑/補幀後 keypoints |
| `angles.csv` | Step 2 | 關節角度 |
| `metrics.csv` | 速度分析 | 速度、加速度、距離 |
| `cam1_ankle_positions.csv` | 步頻步幅 | 每幀腳踝座標 |
| `cam1_step_events.csv` | 步頻步幅 | touchdown、步幅、步頻 |
| `output_final.mp4` | overlay + 標注 | 前端播放的最終影片 |
| `analysis_meta` | 後端 DB | 前端摘要數值 |

## 15. 快速 Debug Checklist

如果前端顯示 `N/A`：

1. 查 DB：

```bash
sqlite3 /home/jeter/running-analysis-backend/running-analysis-backend/running.db \
  "select id,status,progress from run_session where id='{run_session_id_without_dash_or_with_uuid_format}';"
```

2. 查 session 目錄是否有：

```text
metrics.csv
cam1_step_events.csv
output_final.mp4
```

3. 查 Step 1 是否輸出 0 幀：

```bash
wc -l cam1_tracked_bbox_map.csv
wc -l cam1_tracked_frame_map.csv
ffprobe cam1_tracked.mp4
```

4. 查 log：

```bash
rg "{run_session_id}" /home/jeter/running-analysis-backend/backend_18000.log
```

5. 如果看到：

```text
寫入 0，捨棄 XXX
跑者從未越過起跑線
No keypoints generated
```

優先檢查：

- 前端四個錨點順序
- start/end line 是否交叉
- ROI 是否裁錯
- YOLO 是否偵測到 runner
