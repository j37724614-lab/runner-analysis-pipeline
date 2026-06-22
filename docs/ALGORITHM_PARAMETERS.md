# 跑者分析流程演算法參數與約束

更新日期：2026-06-03

本文整理目前 pipeline 實際使用的「演算法參數」、「閾值參數」、「過濾條件」與「追蹤/偵測約束」。內容以程式碼現況為準，主要來源：

- `core/pipeline.py`
- `core/tracking.py`
- `MotionAGFormer/demo/vis.py`
- `scripts/analysis/ankle_step_stride.py`
- `core/overlay.py`

## 流程總覽

目前完整流程分為五個階段：

1. **Step 1: YOLO / ByteTrack 多相機追蹤**
   - 偵測跑者 bbox。
   - 選定主跑者。
   - 產生追焦裁切影片 `sequential_tracked.mp4`。
   - 產生 `bbox_map.csv`、`frame_map.csv`、`cam1_offsets.npz`。
   - 中間漏偵測 bbox 時，若前後都有有效 bbox，會做線性補幀。

2. **Step 2: HRNet 2D keypoints + MotionAGFormer 3D pose**
   - HRNet 在補完後的追焦影片上偵測 2D keypoints。
   - MotionAGFormer 將 2D keypoints 轉為 3D pose。
   - 2D keypoints 會經過信心值過濾、跳點偵測、線性插值與 Savitzky-Golay 平滑。

3. **Step 3: 原影片 overlay**
   - 使用 `cam1_offsets.npz` 將追焦影片座標轉回原影片座標。
   - 原影片座標計算：

     ```text
     original_x = cropped_x + off_x
     original_y = cropped_y + off_y
     ```

4. **Step 4: 步頻 / 落地點 / 步幅分析**
   - 使用 HRNet 2D ankle keypoints，不直接使用 YOLO bbox 底部作為落地點。
   - ankle keypoints 透過 offsets 貼回原影片座標。
   - 用較低腳踝的 y 座標序列找 peak，作為落地事件。

5. **Step 5: Web 相容轉檔與輸出歸檔**
   - `output_final.mp4` 轉為 H.264 / yuv420p / faststart。
   - 複製到 keypoints archive 目錄。

## Step 1: YOLO Tracking 參數

來源：`core/tracking.py`

### 全域模型與輸出參數

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `CUDA_VISIBLE_DEVICES` | `'0'` | 預設使用第 0 張 GPU。實際可由 API/CLI 傳入 gpu 覆蓋。 |
| `DEVICE` | `0` | YOLO 推論 device。 |
| `MODEL_PATH` | `yolo26x.pt` | YOLO 偵測權重。 |
| `CROP_WIDTH` | `200` | 追焦裁切輸出寬度。 |
| `CROP_HEIGHT` | `260` | 追焦裁切輸出高度。 |
| `AUTO_CROP` | `False` | 若為 true，先 dry-run 統計 bbox 尺寸，再設為中位 bbox 寬高的 2 倍。 |
| `TRACKING_MODE` | `'online'` | 支援 `'online'` 與 `'two_pass'`。 |
| `SHOW_OVERLAY` | `True` | 追焦影片是否顯示視覺輔助線/框。 |
| `DRAW_BBOX_OVERLAY` | `True` | 追焦影片是否畫 bbox 與 track ID。`bbox_map.csv` 不受此開關影響。 |

`core/pipeline.py` 會從 `extra_cfg` 覆蓋下列參數：

```text
output_dir
crop_width
crop_height
auto_crop
show_overlay
draw_bbox_overlay
movement_threshold
min_movement_frames
stationary_decay
max_person_memory
tracking_mode
```

### YOLO 推論參數

Online 模式或 Pass 2 非快取時：

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `classes` | `[0]` | 只偵測 COCO person。 |
| `conf` | `0.3` | YOLO person confidence 閾值。 |
| `iou` | `0.1` | tracking / NMS IoU 相關閾值。 |
| `imgsz` | `1280` | YOLO 推論解析度。 |
| `persist` | `True` | 啟用 ByteTrack 連續追蹤。 |
| `verbose` | `False` | 關閉 YOLO 推論 log。 |

Two-pass Pass 1 收集候選人時：

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `conf` | `0.25` | Pass 1 放寬偵測信心值，避免漏掉候選主跑者。 |
| `iou` | `0.1` | 同 online。 |
| `imgsz` | `1280` | 同 online。 |
| `half` | `True` | Pass 1 使用 half precision。 |
| `classes` | `[0]` | 只偵測 person。 |

### 移動與主跑者選擇參數

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `MOVEMENT_THRESHOLD` | `2 px` | 連續幀 bbox 中心位移大於 2 px 才視為移動。 |
| `MIN_MOVEMENT_FRAMES` | `3` | 連續移動至少 3 幀才視為有效跑者。 |
| `STATIONARY_DECAY` | `2` | 靜止時每幀遞減 movement count 的量。 |
| `MAX_PERSON_MEMORY` | `30` | 某 track ID 超過 30 幀沒偵測到即從 tracker 記憶移除。 |
| `MIN_PERSON_HEIGHT` | `40 px` | bbox 高度低於 40 px 視為背景/遠景人物並略過。 |
| `GROUND_POINT_EMA_ALPHA` | `0.35` | bbox 底部中心點 EMA 平滑係數。 |
| `LANE_WIDTH_M` | `1.22 m` | 跑道寬度參考值。 |

主跑者選擇條件：

- 本幀必須有偵測到該 track ID。
- `movement_count >= MIN_MOVEMENT_FRAMES`。
- `stationary_count < 10`。
- 在候選者中選平均 bbox 中心位移最大的 track ID。
- 起跑前若 `nearest_to_start=True`，優先保留投影距離起跑線最近者。
- 若指定 `locked_target_id`，只保留該 track ID，其他偵測全部丟棄。

### ROI / 起終點過濾條件

相機可使用矩形 ROI 或斜線起終點模式。

矩形 ROI：

```text
roi_x[0] <= orig_center_x <= roi_x[1]
roi_y[0] <= orig_center_y <= roi_y[1]
```

斜線起終點模式：

- 由 `start_line` 和 `end_line` 計算：
  - `start_mid`
  - `end_mid`
  - `track_dir`
  - `pixel_span`
- bbox 底部中心點投影到跑道方向：

```text
proj_px = dot(ground_point - start_mid, track_dir)
```

斜線模式過濾：

| 參數 | 目前預設 | 說明 |
|---|---:|---|
| `pre_roll_px` | `200` | 起跑線前保留候選區間。 |
| `end_roll_px` | `120` | 終點線後容許緩衝。 |
| `start_roi_px` | `100` | 起跑確認前，只保留距起跑線較近的候選者。 |
| `homography_lane_margin_px` | `80` | 若有跑道四邊形，允許 bbox 底部中心點在四邊形外 80 px 內。 |

有效投影區間：

```text
-pre_roll_px <= proj_px <= pixel_span + end_roll_px
```

起跑確認條件：

- `proj_px >= 0` 後進入 candidate buffer。
- 連續 `K_CONFIRM = 3` 幀投影單調遞增才確認起跑。
- 起跑線前的 `pre_roll_buf` 最多保留 5 幀，但目前只用於提前鎖定 ID，不輸出 start line 前的幀。

停止 / 切換條件：

- 若有 Homography 與 `distance_m`：

```text
homography_local_dist >= distance_m
```

- 否則若有斜線起終點：

```text
proj_px >= pixel_span
```

- 否則舊模式使用 `switch_x`：
  - 最後一機用 bbox 右緣 `bx2`。
  - 非最後一機用 ground point / center reference。

### Two-pass 主跑者評分參數

Two-pass 模式包含：

1. Pass 1：完整原始畫面收集所有候選 bbox。
2. 依軌跡品質選主跑者。
3. 修補短暫 ID 斷裂。
4. Pass 2：使用快取 bbox 輸出追焦影片，不再跑 YOLO inference。

候選主跑者評分：

```text
score =
  0.25 * coverage
+ 0.25 * progress
+ 0.20 * monotonic
+ 0.15 * start_proximity
+ 0.10 * roi_ratio
```

Two-pass 候選過濾：

| 條件 | 目前值 |
|---|---:|
| 最少出現幀數 `n_frames` | `>= 10` |
| 進度 `progress` | `>= 0.3` |
| 單調性 `monotonic` | `>= 0.55` |

若最佳與第二名分數差距小於最佳分數的 10%，會輸出不穩定警告。

ID stitch 修補條件：

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `max_gap` | `5` | 只修補長度不超過 5 幀的 ID 短缺口。 |
| `max_dist_px` | `100` | 候選 bbox 中心距離線性預測位置不得超過 100 px。 |
| `size_ratio` | `<= 1.5` | 候選 bbox 高度與預測高度比例不得超過 1.5。 |

### BBox 補幀條件

當主跑者已確認且中途 YOLO 暫時漏偵測：

- 若前面有 `last_valid_bbox`。
- 後面重新出現有效 bbox。
- 中間 pending missing frames 會用左右 bbox 做線性插值。
- 補值幀會重新裁切並寫入追焦影片。
- 補值資訊寫入 `bbox_map.csv`：

```text
is_interpolated = 1
interp_gap_len = 缺口長度
```

不補的情況：

- 開頭還沒有前一個有效 bbox。
- 結尾沒有後一個有效 bbox。
- 起跑確認前尚未鎖定主跑者。
- 已達終點/切換條件後。

每個輸出幀同時記錄：

- `bbox_map.csv`：追焦畫面內 bbox。
- `frame_map.csv`：輸出幀對應原始影片幀號。
- `cam1_offsets.npz`：追焦裁切視窗左上角在原影片中的 `(off_x, off_y)`。

## Step 2: HRNet / MotionAGFormer 姿態估計參數

來源：`MotionAGFormer/demo/vis.py`

### HRNet 偵測設定

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `det_dim` | `416` | HRNet 前置偵測解析度。 |
| `num_peroson` | `1` | 每幀只保留 1 個人。 |
| `bbox_csv` | tracking 產生的 `*_bbox_map.csv` | 若存在，HRNet 直接使用 tracking bbox，不再自行找全畫面人物。 |

Step 2 會尋找：

```text
tracked_video_path.replace(".mp4", "_bbox_map.csv")
```

若存在，傳入 HRNet 作為外部 bbox 約束。

### 骨架格式約束

HRNet 原始 COCO 17 點會轉成 H36M 17 點：

```text
0  Hip / Pelvis
1  RHip
2  RKnee
3  RAnkle
4  LHip
5  LKnee
6  LAnkle
7  Spine
8  Thorax
9  Neck_Nose
10 Head
11 LShoulder
12 LElbow
13 LWrist
14 RShoulder
15 RElbow
16 RWrist
```

H36M 補點：

- `Hip`：左右 hip 平均。
- `Thorax`：左右 shoulder 與 nose 推估。
- `Spine`：hip / thorax 相關點推估。
- `Head`：臉部點推估。

### 2D Keypoint 平滑與過濾

目前實際呼叫的是 `_smooth_keypoints_sg()`：

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `conf_threshold` | `0.50` | HRNet confidence 低於 0.50 視為低信心點。 |
| `sg_window` | `7` | fallback Savitzky-Golay window。實際多數 joint 使用 per-joint window。 |
| `sg_polyorder` | `2` | Savitzky-Golay 多項式階數。 |
| `frame_bad_joint_threshold` | `8` | 同一幀壞點數 >= 8 時，整幀視為不可靠。 |
| bbox scale clip | `[0.6, 1.8]` | 若有 bbox height，跳點閾值依人體大小縮放。 |
| bbox reference height | median bbox height | 使用 `bbox_map.csv` 的 bbox 高度中位數。 |

2D keypoints 處理階段：

1. 低信心點標記：

```text
score < 0.50
```

2. 雙側孤立跳點偵測：

對 frame `t`，若同時符合：

```text
distance(point[t], point[t-1]) > per_joint_threshold
distance(point[t], point[t+1]) > per_joint_threshold
distance(point[t-1], point[t+1]) < distance(point[t], point[t-1])
distance(point[t-1], point[t+1]) < distance(point[t], point[t+1])
```

則視為 `bilateral_outlier`。

3. 整幀壞點 fallback：

```text
bad_joint_count >= 8
```

4. 壞點以 `np.interp` 線性插值補上。
5. 對每個 joint 使用 Savitzky-Golay 平滑。

### Per-joint 跳點閾值

`_JOINT_MAX_PX`：

| Joint | 閾值 px |
|---|---:|
| Hip | 10.35 |
| RHip | 11.76 |
| RKnee | 19.62 |
| RAnkle | 63.90 |
| LHip | 15.35 |
| LKnee | 25.38 |
| LAnkle | 63.49 |
| Spine | 12.49 |
| Thorax | 14.23 |
| Neck_Nose | 12.10 |
| Head | 12.92 |
| LShoulder | 17.77 |
| LElbow | 36.43 |
| LWrist | 50.29 |
| RShoulder | 15.14 |
| RElbow | 35.21 |
| RWrist | 50.69 |

`_JOINT_HARD_PX` 目前保留為較寬鬆參考閾值，但主要 SG 平滑流程未直接使用它：

| Joint | hard px |
|---|---:|
| Hip | 16.75 |
| RHip | 17.47 |
| RKnee | 55.29 |
| RAnkle | 80.52 |
| LHip | 23.64 |
| LKnee | 55.38 |
| LAnkle | 80.70 |
| Spine | 20.80 |
| Thorax | 26.58 |
| Neck_Nose | 20.26 |
| Head | 24.67 |
| LShoulder | 29.21 |
| LElbow | 57.39 |
| LWrist | 72.75 |
| RShoulder | 25.34 |
| RElbow | 53.99 |
| RWrist | 72.80 |

### Per-joint Savitzky-Golay window

| Joint | window |
|---|---:|
| Hip | 15 |
| RHip | 11 |
| RKnee | 7 |
| RAnkle | 3 |
| LHip | 11 |
| LKnee | 7 |
| LAnkle | 3 |
| Spine | 15 |
| Thorax | 15 |
| Neck_Nose | 15 |
| Head | 15 |
| LShoulder | 11 |
| LElbow | 9 |
| LWrist | 7 |
| RShoulder | 11 |
| RElbow | 9 |
| RWrist | 7 |

### 3D Pose 模型

| 項目 | 目前值 |
|---|---|
| 模型 | `MotionAGFormer-large` |
| config | `MotionAGFormer/configs/h36m/MotionAGFormer-large.yaml` |
| checkpoint | `checkpoint/motionagformer-l-h36m.pth.tr` |
| input clip length | `243` frames |
| 短片補幀 | 若影片短於 243 幀，以均勻 index resample 到 243。 |
| 長片切段 | 每 243 幀一段，最後不足 243 幀做 resample。 |
| test-time augmentation | 左右翻轉推論後平均。 |

3D 輸出保存：

```text
pred_3D/3Dkeypoints.npz
```

## 關節角度計算與平滑

來源：`MotionAGFormer/demo/vis.py`

### 角度欄位

目前輸出：

- `left_knee_angle`
- `left_hip_angle`
- `right_knee_angle`
- `right_hip_angle`
- `left_arm_torso_angle`
- `left_elbow_flexion_angle`
- `right_arm_torso_angle`
- `right_elbow_flexion_angle`
- `left_shoulder_flexion`
- `right_shoulder_flexion`
- `pelvis_torso_angle`

角度由 3D joint vector 的 arccos 計算，cos 會 clamp 到 `[-1, 1]`。

### 膝角 supplementary flip 修正

只套用於：

- `left_knee_angle`
- `right_knee_angle`

參數：

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `max_delta` | `45.0 deg` | 若原始角度距局部趨勢超過 45 度才考慮修正。 |
| `improvement_margin` | `15.0 deg` | `180 - angle` 必須比原角度更接近局部趨勢至少 15 度。 |
| local reference | 前後 2 幀中位數 | 使用鄰近有效角度估計局部趨勢。 |

### 角度平滑

目前使用 `_smooth_angles_sg()`：

| 參數 | 目前值 |
|---|---:|
| `polyorder` | `3` |
| window | `max(polyorder + 2, int(fps * 0.22))`，若為偶數則 +1 |
| mode | `mirror` |

註：`_smooth_angles_butterworth(cutoff_hz=10.0, order=4)` 仍保留在程式中，但目前 `compute_angles()` 實際呼叫的是 Savitzky-Golay。

## 原影片 Overlay 座標約束

來源：`core/overlay.py`、`scripts/analysis/ankle_step_stride.py`

所有追焦影片座標貼回原影片時，使用 `cam1_offsets.npz`：

```text
original_x = cropped_x + off_x
original_y = cropped_y + off_y
```

`cam1_offsets.npz` 欄位：

| 欄位 | 說明 |
|---|---|
| `offsets` | 每個追焦輸出幀的 crop 左上角原影片座標 `(off_x, off_y)`。 |
| `orig_frames` | 每個追焦輸出幀對應原始影片第幾幀。 |
| `cam_indices` | 每個追焦輸出幀來自第幾台相機，0-based。 |

注意：

- 所有後續分析應使用補完後的追焦影片、補完後的 `keypoints.npz`、補完後的 `cam1_offsets.npz`。
- `cam*_overview.mp4` 只是觀察原始 YOLO 偵測的輔助影片，不應作為後續分析輸入。

## 步頻 / 落地點 / 步幅參數

來源：`scripts/analysis/ankle_step_stride.py`

### 資料來源

步頻與落地點不是直接使用 YOLO bbox 底部中心，而是使用 HRNet / MotionAGFormer 2D ankle keypoints：

| Joint | H36M index |
|---|---:|
| Right ankle | `3` |
| Left ankle | `6` |

每幀將追焦座標轉回原圖：

```text
right_ankle_original = right_ankle_cropped + offset
left_ankle_original  = left_ankle_cropped + offset
```

每幀取 y 較大的腳踝作為較低腳：

```text
lower_foot = right if right_ankle_y >= left_ankle_y else left
```

### 落地事件偵測

使用 `lower_ankle_y` 時序找 peak。

| 參數 | 目前值 | 說明 |
|---|---:|---|
| FPS | 優先用 ankle rows 的時間戳推估；若缺失則 30。`run_step_stride_analysis()` 從影片讀不到 FPS 時會先用 60。 |
| smoothing | 4 階 Butterworth low-pass |
| cutoff | `6.0 Hz` |
| smoothing minimum samples | `n >= 13` 才套用 filter。 |
| `min_step_frames` | 若未指定，`max(4, int(fps / 4.5))` |
| `min_touchdown_gap_frames` | `max(3, min_step_frames // 2)` |
| `peak_distance` | `max(3, int(fps / 8.0))` |
| peak prominence | 若未指定，`max((q75 - q25) * 0.20, 1.5)` |
| ankle confidence | `< 0.30` 的候選落地事件丟棄 |

Peak 偵測：

```text
find_peaks(
  y_smooth,
  distance=peak_distance,
  prominence=lower_prominence
)
```

`min_step_frames` 不是 `find_peaks()` 的 distance。它代表後續物理約束使用的最小步距時間尺度；`peak_distance = fps / 8` 才是 local max 偵測時避免同一個尖峰被重複偵測的距離。這樣 60fps 下左右腳交替約 12 幀的落地間隔，不會因為 `fps / 4.5` 太大而直接被 peak detector 擋掉。

若兩個 touchdown candidates 太近：

```text
frame_gap < min_touchdown_gap_frames
```

則保留分數較高者：

```text
(prominence, ankle_conf, ankle_y)
```

### 步幅與步頻

落地點位置：

```text
point = (ankle_x, ankle_y)
```

目前步幅優先使用 start/end line 與 `distance_m` 做 pixel line calibration：

- 將 ankle point 投影到跑道方向。
- 使用 `meters_per_pixel = distance_m / pixel_span`。
- `step_length_m = step_length_px * meters_per_pixel`。

若沒有 line calibration，才會嘗試 homography fallback：

- 轉成世界座標。
- `step_length_m = abs(current_world_x - previous_world_x)`。

若沒有距離標定：

- 只輸出 `step_length_px`。
- `step_length_m` 為空。

步頻：

- instantaneous cadence：

```text
cadence_spm = 60 / delta_time_between_step_events
```

- global average cadence：

```text
avg_cadence_spm = detected_steps / total_duration * 60
```

## 目前輸出檔案與用途

| 檔案 | 用途 |
|---|---|
| `sequential_tracked.mp4` | 補完後的追焦裁切影片，HRNet / MotionAGFormer 的輸入。 |
| `*_bbox_map.csv` | 每個追焦輸出幀對應的 bbox，含 `is_interpolated`。 |
| `*_frame_map.csv` | 追焦輸出幀對應原始影片幀號。 |
| `cam1_offsets.npz` | 追焦座標貼回原影片座標的 offset。 |
| `sequential_tracked/input_2D/keypoints.npz` | 平滑後 2D keypoints，後續 overlay / 步頻使用。 |
| `sequential_tracked/input_2D/keypoints_raw.npz` | 原始 HRNet keypoints 備查。 |
| `sequential_tracked/keypoint_smoothing_debug.csv` | 2D keypoint 平滑與壞點判定 debug。 |
| `sequential_tracked/hrnet_confidence.csv` | 每幀每 joint 的 HRNet confidence。 |
| `sequential_tracked/pred_3D/3Dkeypoints.npz` | MotionAGFormer 3D pose。 |
| `angles.csv` | 關節角度輸出。 |
| `cam1_ankle_positions.csv` | 每幀左右腳踝與 lower ankle 原影片座標。 |
| `cam1_step_events.csv` | 偵測到的落地事件、步頻、步幅。 |
| `output_final.mp4` | 原影片比例骨架 overlay + 步頻/步幅標註後輸出。 |

## 關鍵約束摘要

- YOLO 只偵測 `person` 類別。
- 主跑者必須通過 ROI / 起終點投影 / 高度 / 移動條件。
- 補幀只補中間缺失，不補端點缺失。
- HRNet 只分析補完後追焦影片。
- HRNet 若拿到 `bbox_map.csv`，會使用 tracking bbox 限定偵測範圍。
- 2D keypoints 低信心、孤立跳點與壞幀會被插值和平滑。
- 原影片 overlay 與步頻分析都必須使用與補完影片對齊的 `cam1_offsets.npz`。
- 步頻落地點使用 ankle keypoints，不使用 YOLO bbox 底部中心直接判定。

## 四類參數詳細說明

本節將目前系統中的設定拆成四類：

- **演算法參數**：決定使用哪一種演算法、模型、流程策略與資料流。
- **閾值參數**：決定數值到達多少才觸發某個判斷。
- **過濾條件**：決定哪些偵測、幀、關節或事件會被保留或丟棄。
- **追蹤/偵測約束**：限制偵測與追蹤只能在合理的人、區域、時間與骨架結構內發生。

這四類不是完全互斥。例如 `MIN_PERSON_HEIGHT=40` 既是閾值參數，也是 YOLO 偵測過濾條件；`bbox_map.csv` 既是追蹤輸出，也是 HRNet 的偵測約束。

### 1. 演算法參數

演算法參數描述「系統用什麼方法做事」。它們通常不只是單一數值，而是決定資料如何流過 pipeline。

#### 1.1 YOLO / ByteTrack 追蹤策略

目前 Step 1 使用 YOLO 偵測 person，再用 YOLO 內建 tracking 保持 track ID。程式呼叫：

```python
model.track(
    img,
    persist=True,
    classes=[0],
    conf=0.3,
    iou=0.1,
    imgsz=1280,
)
```

演算法意義：

- `classes=[0]`：只偵測人，不偵測其他 COCO 類別。
- `persist=True`：讓 tracker 在連續幀之間保留 ID。
- YOLO 只負責 bbox 與 track ID，不負責最後的腳踝落地點。
- 主跑者由 bbox 的移動、ROI、起終點投影與 two-pass score 決定。

目前支援兩種 tracking 模式：

| 模式 | 作用 |
|---|---|
| `online` | 邊讀影片邊偵測、邊選主跑者、邊輸出追焦影片。 |
| `two_pass` | 第一遍收集所有候選軌跡，先決定主跑者 ID；第二遍用快取 bbox 輸出追焦影片。 |

`two_pass` 的演算法重點是減少即時選錯人的風險。它會用 coverage、progress、monotonic、start proximity、roi ratio 評分，選出最像主跑者的 ID。

調整風險：

- `online` 反應快，但多人交錯時更容易選錯 ID。
- `two_pass` 較穩，但需要多跑一遍偵測，耗時較高。
- 如果場景中有多個跑者軌跡都很像，two-pass 仍可能不穩，程式會在前兩名 score 差距小於 10% 時警告。

#### 1.2 追焦裁切演算法

追焦影片不是原影片縮放，而是以主跑者 bbox 中心為基準裁出固定大小畫面：

```text
CROP_WIDTH  = 200
CROP_HEIGHT = 260
```

裁切流程：

1. 取得主跑者 bbox。
2. 計算 bbox 中心 `(cx, cy)`。
3. 以 `(cx, cy)` 為中心裁出 `200x260`。
4. 若裁切視窗超出原圖邊界，整個視窗往回平移，保持尺寸不變。
5. 記錄裁切視窗左上角 `(off_x, off_y)` 到 `cam1_offsets.npz`。

這個演算法參數會直接影響：

- HRNet 看到的人體大小。
- keypoints 是否穩定。
- 貼回原影片時的座標正確性。
- 步頻/落地點使用的 ankle 原圖位置。

調整風險：

- crop 太小：手腳容易被裁掉，HRNet 腳踝/手腕會失真。
- crop 太大：跑者變小，HRNet 精度下降，背景人物干擾變多。
- crop 尺寸改變後，既有 keypoint 跳點閾值可能需要重新統計。

#### 1.3 BBox 補幀演算法

當 tracking 中間幾幀漏掉主跑者，但前後都有有效 bbox，系統會做線性補值：

```text
interp_bbox = left_bbox + ratio * (right_bbox - left_bbox)
```

演算法作用：

- 補回追焦影片中間缺失幀。
- 補回 `bbox_map.csv`。
- 補回 `cam1_offsets.npz`。
- 讓後續 HRNet / MotionAGFormer / overlay / 步頻分析都使用補完後資料。

補幀不是重新偵測，而是根據前後 bbox 推估中間位置。補值幀會標記：

```text
is_interpolated = 1
interp_gap_len = 缺口長度
```

調整風險：

- 短缺口通常合理。
- 長缺口若跑者有加速、急停、遮擋或換道，線性補值可能偏離真實位置。
- 補值幀必須保存自己的 offset；若 offset 沿用錯誤幀，骨架貼回原圖會整段偏移。

#### 1.4 HRNet + MotionAGFormer 姿態估計演算法

姿態估計分兩層：

1. HRNet 偵測 2D keypoints。
2. MotionAGFormer 將 2D keypoints 轉為 3D pose。

HRNet 的輸入不是原影片，而是 Step 1 補完後的追焦影片。若存在 `bbox_map.csv`，HRNet 使用 tracking bbox 約束偵測範圍。

MotionAGFormer 參數：

| 項目 | 目前值 |
|---|---|
| 模型 | `MotionAGFormer-large` |
| 訓練資料格式 | H36M 17 joints |
| checkpoint | `motionagformer-l-h36m.pth.tr` |
| clip length | 243 frames |
| augmentation | horizontal flip + average |

短片會 resample 到 243 幀後推論，再用保存的 index 還原原始長度。

調整風險：

- 若 2D keypoints 已錯，3D pose 通常會跟著錯。
- MotionAGFormer 不會保證固定骨長或合法關節角，只是根據訓練分布輸出合理 3D pose。
- 影片很短時，resample 會重複幀，可能讓 3D temporal pattern 較不自然。

#### 1.5 步頻與落地點演算法

步頻不是從 YOLO bbox 算，而是從 2D ankle keypoints 算。

流程：

1. 讀 `keypoints.npz`。
2. 讀 `cam1_offsets.npz`。
3. 取 H36M ankle：
   - right ankle = index `3`
   - left ankle = index `6`
4. 將 ankle 從追焦座標貼回原圖座標：

```text
ankle_original = ankle_cropped + offset
```

5. 每幀取 y 較大的腳踝作為 `lower_ankle`。
6. 對 `lower_ankle_y` 時序找 peak，peak 視為落地事件。

演算法假設：

- 影像座標 y 越大代表越靠近畫面下方。
- 跑步落地時，當下接觸地面的腳踝通常是左右腳踝中 y 較大的那一個。
- 落地事件會在 `lower_ankle_y` 序列形成局部高點。

調整風險：

- 若相機角度很斜、腳踝被遮擋或 HRNet 腳踝飄移，落地 peak 會不穩。
- 若 offset 錯，落地點原圖座標會錯，但 peak 時序仍可能看似合理。
- 若跑者步頻很高，`min_step_frames` 太大會漏步；太小會把抖動當成步。

### 2. 閾值參數

閾值參數決定「什麼程度才算有效、異常、移動或事件」。以下列出目前會明顯影響結果的閾值。

#### 2.1 YOLO 偵測閾值

| 閾值 | 目前值 | 作用 | 太低的風險 | 太高的風險 |
|---|---:|---|---|---|
| online `conf` | `0.3` | 人物偵測最低信心值 | 背景誤判、非主跑者混入 | 遠距/模糊跑者漏偵測 |
| two-pass `conf` | `0.25` | Pass 1 收集候選軌跡 | 候選雜訊變多 | 主跑者短暫漏收集 |
| `iou` | `0.1` | YOLO tracking / NMS 相關重疊門檻 | 太多相近框混入 | 擁擠時可能合併/丟框 |
| `MIN_PERSON_HEIGHT` | `40 px` | 過濾太小的人 | 遠景人物混入 | 遠距主跑者被丟棄 |

#### 2.2 移動判定閾值

| 閾值 | 目前值 | 作用 |
|---|---:|---|
| `MOVEMENT_THRESHOLD` | `2 px` | bbox 中心位移 > 2 px 才算移動。 |
| `MIN_MOVEMENT_FRAMES` | `3` | 至少連續移動 3 幀才視為有效跑者。 |
| `STATIONARY_DECAY` | `2` | 靜止時 movement count 每幀快速下降。 |
| `stationary_count` gate | `< 10` | 靜止累積太多時不選為最快跑者。 |
| `MAX_PERSON_MEMORY` | `30 frames` | 超過 30 幀未偵測到就刪除該 track。 |

這些參數主要避免把站立的人、觀眾或短暫雜訊選成主跑者。

#### 2.3 起終點 / ROI 閾值

| 閾值 | 目前值 | 作用 |
|---|---:|---|
| `pre_roll_px` | `200` | 起跑線前候選區間。 |
| `start_roi_px` | `100` | 起跑確認前，只看靠近起跑線的候選。 |
| `end_roll_px` | `120` | 終點後容許保留的投影距離。 |
| `K_CONFIRM` | `3 frames` | 起跑後需連續 3 幀投影單調遞增才確認。 |
| `homography_lane_margin_px` | `80` | 跑道四邊形外 80 px 內仍視為可接受。 |

這些閾值影響影片開始、結束與切相機時機。

#### 2.4 Two-pass 評分閾值

| 閾值 | 目前值 | 作用 |
|---|---:|---|
| `n_frames` | `>= 10` | 候選 track 至少出現 10 幀。 |
| `progress` | `>= 0.3` | 候選 track 必須有足夠前進量。 |
| `monotonic` | `>= 0.55` | 軌跡大致往終點方向前進。 |
| score gap warning | `< 10%` | 第一名與第二名太接近時警告。 |
| stitch `max_gap` | `5 frames` | 只修補不超過 5 幀的 ID 缺口。 |
| stitch `max_dist_px` | `100 px` | 可修補候選需接近預測位置。 |
| stitch `size_ratio` | `<= 1.5` | bbox 高度差距不能太大。 |

#### 2.5 HRNet / keypoint 閾值

| 閾值 | 目前值 | 作用 |
|---|---:|---|
| HRNet bbox `det_dim` | `416` | HRNet 前處理解析度。 |
| `num_peroson` | `1` | 每幀只保留一人。 |
| keypoint `conf_threshold` | `0.50` | 低於 0.50 的 2D keypoint 視為低信心。 |
| `frame_bad_joint_threshold` | `8` | 同一幀壞 joint >= 8，整幀視為不可靠。 |
| bbox scale clip | `[0.6, 1.8]` | keypoint 跳點閾值依人體大小縮放。 |

Per-joint 跳點閾值已在前文列出。它們是 keypoint temporal outlier 判定的核心。

#### 2.6 角度與步頻閾值

角度：

| 閾值 | 目前值 | 作用 |
|---|---:|---|
| knee flip `max_delta` | `45 deg` | 膝角偏離局部趨勢超過 45 度才考慮修正。 |
| knee flip `improvement_margin` | `15 deg` | supplementary angle 必須至少改善 15 度。 |
| angle SG window | `int(fps * 0.22)` | 約 0.22 秒局部平滑。 |
| angle SG `polyorder` | `3` | 三階 Savitzky-Golay。 |

步頻：

| 閾值 | 目前值 | 作用 |
|---|---:|---|
| ankle peak low-pass cutoff | `6 Hz` | 落地 peak 偵測前的腳踝 y 序列低通。 |
| Butterworth order | `4` | 四階低通。 |
| minimum samples | `13` | 少於 13 點不做低通。 |
| `min_step_frames` | `max(4, int(fps / 4.5))` | 兩步之間最小幀距。 |
| `min_touchdown_gap_frames` | `max(3, min_step_frames // 2)` | 太近的候選只留一個。 |
| `peak_distance` | `max(3, int(fps / 8.0))` | `find_peaks()` 的 local max 最小距離。 |
| peak prominence | `max((q75-q25)*0.20, 1.5)` | 落地 peak 最小突出程度。 |
| ankle confidence | `>= 0.30` | 低於 0.30 的落地候選丟棄。 |
| RAW rescue gap | `max(min_step_frames + peak_distance, round(min_step_frames * 1.8))` | Smooth touchdown 間隔過大時，才回 raw signal 找補點。 |
| RAW rescue large step | `min(median_smooth_step_px * 1.6, 2.6m / meters_per_pixel)` | Smooth 兩點間投影位移過大時，回 raw signal 找中間漏步。 |
| min valid step length | `0.60m` | 兩個 touchdown 距離小於此值時視為重複偵測候選。 |

### 3. 過濾條件

過濾條件描述「資料何時被保留、何時被丟掉」。目前主要有以下幾層。

#### 3.1 偵測層過濾

YOLO bbox 必須符合：

- 類別是 person。
- confidence 達標。
- bbox 高度 >= `MIN_PERSON_HEIGHT`。
- 通過 ROI 或起終點投影範圍。
- 若有 `locked_target_id`，track ID 必須等於鎖定 ID。

不符合者不會進入主跑者候選，也不會進入追焦輸出。

#### 3.2 ROI / 跑道層過濾

矩形 ROI 使用 bbox center。

斜線模式使用 bbox bottom center，也就是較接近跑者落地位置的點。

若有跑道四邊形 `quad_roi`：

- 使用 `cv2.pointPolygonTest()` 測 bbox bottom center 到四邊形的 signed distance。
- 若距離 >= `-homography_lane_margin_px`，保留。
- 如果有至少一個人在跑道附近，遠離跑道的人會被排除。

這層過濾避免觀眾、旁邊跑道、背景人物進入候選。

#### 3.3 主跑者層過濾

候選 track 必須：

- 本幀有出現。
- 移動累積足夠。
- 沒有太久沒偵測到。
- 靜止計數未過高。

Two-pass 模式還會額外要求：

- 覆蓋率足夠。
- 進度足夠。
- 方向大致單調。
- 起點位置合理。

#### 3.4 補幀層過濾

只有「中間缺失」補幀：

```text
有效 bbox -> 缺失幀 -> 有效 bbox
```

端點缺失不補：

```text
缺失幀 -> 有效 bbox
有效 bbox -> 缺失幀
```

補幀後仍會輸出到後續流程，但會被標記為 interpolated。這讓後續分析可以保留時間連續性，也能回頭檢查哪些幀是推估出來的。

#### 3.5 Keypoint 層過濾

HRNet keypoint 會被標為 bad point 的情況：

- confidence < `0.50`。
- 雙側孤立跳點。
- 所在 frame 壞點數 >= 8。

bad point 不會直接刪掉該幀，而是用鄰近 good points 線性插值補回，再做平滑。

#### 3.6 Step event 層過濾

落地候選會被丟棄的情況：

- lower ankle confidence < `0.30`。
- peak prominence 不足。
- 與前一個落地候選太近，且分數較低。

這層過濾的目的不是讓步數最多，而是避免骨架抖動造成假落地。

### 4. 追蹤/偵測約束

追蹤/偵測約束描述「系統被限制在什麼假設內運作」。這些約束比單一閾值更高層。

#### 4.1 人物類別約束

YOLO 只追蹤 COCO `person`：

```text
classes = [0]
```

所以其他物件不會進入 tracking。但如果背景人物也是 person，仍需要靠 ROI、起終點線、移動條件與 two-pass score 排除。

#### 4.2 主跑者唯一性約束

HRNet 與後續分析假設每個追焦輸出幀只有一個主要跑者：

```text
num_peroson = 1
```

因此 Step 1 的 tracking 必須先選對主跑者。若 Step 1 選錯人，後續 HRNet 會很穩定地分析錯的人。

#### 4.3 時間連續性約束

系統假設主跑者軌跡在短時間內連續：

- tracking ID 可短暫斷裂。
- bbox 可短暫漏偵測。
- 短缺口可用線性補值。
- 長缺口或端點缺失不硬補。

這個約束讓後續 HRNet、3D pose、步頻分析能保持固定幀序列。

#### 4.4 空間區域約束

系統假設主跑者位於指定跑道或 ROI：

- 矩形 ROI：用 bbox center。
- 斜線跑道：用 bbox bottom center 投影。
- Homography / quad ROI：用跑道四邊形與世界座標距離。

這些約束是排除背景人的主要手段。

#### 4.5 骨架格式約束

所有 pose 都轉成 H36M 17-joint 格式。後續：

- 2D overlay 用 H36M 連線。
- MotionAGFormer 以 H36M 17-joint 作為輸入。
- 角度計算使用固定 H36M joint index。
- 步頻使用 H36M ankle index `3` 與 `6`。

如果輸入不是 H36M order，所有角度、腳踝與 overlay 都會錯。

#### 4.6 原影片座標約束

任何要貼回原影片的點都必須滿足：

```text
point_original = point_cropped + offset_for_same_output_frame
```

其中 `offset_for_same_output_frame` 必須來自同一個補完後輸出幀。

這是目前最重要的資料一致性約束：

- `sequential_tracked.mp4`
- `keypoints.npz`
- `bbox_map.csv`
- `frame_map.csv`
- `cam1_offsets.npz`

必須全部對齊同一組補完後幀序列。

如果拿 raw overview、舊 offsets、未補完 keypoints 或不同 run 的檔案混用，骨架和落地點會貼錯。

#### 4.7 落地點偵測約束

步頻/落地點目前有三個核心假設：

1. 腳踝 keypoints 足夠可靠。
2. 影像座標 y 較大代表腳更接近地面。
3. lower ankle y 的 local peak 對應一次落地。

因此它不適合直接解讀成壓力板等級的真實接觸瞬間；它是基於 2D 視覺 keypoints 的 touchdown candidate。

#### 4.8 可調參數的調整建議

如果遇到不同問題，優先調整順序如下：

| 問題 | 優先檢查 / 調整 |
|---|---|
| 主跑者選錯 | ROI、start/end line、`tracking_mode=two_pass`、two-pass score debug。 |
| 漏 bbox 太多 | YOLO `conf`、`MIN_PERSON_HEIGHT`、ROI 是否太窄、影片解析度/模糊。 |
| 補幀貼回原圖偏掉 | `cam1_offsets.npz` 是否和補完後 `keypoints.npz` 對齊。 |
| HRNet 腳踝飄 | crop 是否裁到腳、`bbox_map.csv` 是否正確、keypoint smoothing debug。 |
| 步數太少 | `min_step_frames`、peak prominence、ankle confidence。 |
| 步數太多 | 增加 `min_step_frames` 或 prominence，檢查 ankle jitter。 |
| 角度跳動 | 檢查 2D keypoints confidence、3D knee correction debug、角度 smoothing window。 |
