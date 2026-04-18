# track_runners.py — 開發者筆記

## 一句話說明

多台相機影片依序串接，以 YOLO 追蹤最快跑者，並在畫面下方即時疊加「距離 / 速度 / 加速度」折線圖，最後輸出單一合併影片與逐幀 CSV。

---

## 整體流程

```
輸入：CAM1–CAM6（或 --config-json 的 cameras 陣列）
         │
         ▼ 逐台相機（串接，非並行）
    ┌────────────────────────────────┐
    │ 第一段：YOLO 追蹤              │
    │  每幀 crop → model.track()    │
    │  velocity_tracker 累積速度    │
    │  選出 fastest_id              │
    │  收集 frame_buffer / d_raw    │
    │  偵測越終點線 → break         │
    └────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────┐
    │ 第二段：批次平滑 + 渲染        │
    │  Butterworth filtfilt          │
    │  Kalman 濾波                   │
    │  疊加速度文字 + matplotlib 圖  │
    │  寫入 VideoWriter              │
    └────────────────────────────────┘
         │
         ▼
輸出：sequential_tracked.mp4 + _metrics.csv
```

兩段式設計的原因：Butterworth `filtfilt` 是雙向濾波，**需要完整序列**才能執行，所以必須先把整台相機的幀全收完（第一段），再一次性計算平滑結果（第二段）。

---

## 關鍵資料結構

### `velocity_tracker`

每台相機都重置，dict 以 YOLO track ID 為 key：

```python
{
    tid: {
        'center': (cx, cy),          # 裁剪後座標
        'bbox':   (bx1, by1, bx2, by2),
        'velocities': [float, ...],  # 每幀像素位移歷史
        'movement_count':   int,     # 連續移動幀計數
        'stationary_count': int,     # 靜止衰減計數
        'frames_since_detected': int,
    }
}
```

`fastest_id` = `velocities` 平均值最大、且 `movement_count >= MIN_MOVEMENT_FRAMES` 的 track。

### `frame_buffer / d_raw / meta_buffer`

第一段收集的三個等長陣列：

| 陣列 | 內容 |
|------|------|
| `frame_buffer[i]` | 縮放後的 strip 畫面（含綠框） |
| `d_raw[i]` | 原始距離（公尺，未平滑） |
| `meta_buffer[i]` | bbox 在 strip 座標，或 `None`（pre-roll 幀） |

`meta_buffer[i] is None` 表示跑者尚未越過起跑線，這些幀不疊加圖表。

---

## 兩種 ROI / 切換模式

### 舊模式（`roi_x` + `switch_x`）

- ROI 過濾：`roi_x[0] <= center_x_orig <= roi_x[1]`
- 切換：`center_x_orig > switch_x`（非最後一機）或 `bx2_orig > switch_x`（最後一機）

### 斜線投影模式（`start_line` + `end_line`）

填入兩端點後自動啟用，`switch_x` 被忽略：

```
start_mid = 起跑線中點
end_mid   = 終點線中點
track_dir = (end_mid - start_mid) 的單位向量
pixel_span = 兩中點距離（像素）

proj_px = dot(runner_pos - start_mid, track_dir)
```

- `proj_px < 0`：pre-roll 區，不記錄距離
- `proj_px >= pixel_span`：越終點線，break 切下一台
- 距離 `= cumulative_offset + proj_px * m_per_pixel`

這個模式對斜向跑道或相機偏角的場景更準確。

---

## 距離 / 速度平滑流程（`_compute_kf_series`）

```
d_raw（原始像素投影 → 公尺）
  │
  ▼ 單調約束（距離只能遞增，不能後退）
  │
  ▼ Butterworth 低通 filtfilt（6 Hz cutoff，order=2，需 n >= 15）
  │
  ▼ Kalman 濾波（狀態：[位置, 速度, 加速度]）
    ├─ 輸出 d_smooth（平滑距離）
    ├─ 輸出 v_smooth（速度，已 clamp >= 0）
    └─ 輸出 a_arr（加速度）
```

跨相機時，前機最終的 Kalman 速度 / 加速度會作為下一機的初始狀態（`init_v`, `init_a`），避免切換時速度從 0 重新爬升。

---

## 圖表軸固定機制

在主迴圈**開始前**預算：

```python
global_d_max = sum(distance_m for all cams) * 1.1
global_t_max = sum(frame_count / fps for all cams)
```

整個執行期間 Y/X 軸範圍不變，跨相機畫面不跳動。matplotlib 畫布也只建立一次，每幀 `cla()` 後重繪。

---

## JSON 介面（`--config-json`）

所有全域變數與 CAM1–CAM6 都可被 JSON 覆蓋，解析在 `main()` 最前面：

```
parse_args()
  └─ args.config_json
       └─ json.loads()
            ├─ gpu, output_dir, output_name, target_height, chart_height
            ├─ movement_threshold, min_movement_frames, stationary_decay, max_person_memory
            └─ cameras[] → _build_camera_from_json() → cameras_override
```

`cameras_override` 不為 `None` 時，完全取代 `[CAM1, ..., CAM6]`。

---

## 檔案位置速查

| 符號 | 位置 |
|------|------|
| 設定區 | [track_runners.py:24–49](track_runners.py#L24-L49) |
| `camera()` | [track_runners.py:164](track_runners.py#L164) |
| `_build_camera_from_json()` | [track_runners.py:228](track_runners.py#L228) |
| CAM1–CAM6 定義 | [track_runners.py:270–330](track_runners.py#L270-L330) |
| `process_frame()` | [track_runners.py:480](track_runners.py#L480) |
| `_compute_kf_series()` | [track_runners.py:342](track_runners.py#L342) |
| `_draw_chart()` | [track_runners.py:408](track_runners.py#L408) |
| `main()` — JSON 解析 | [track_runners.py:635–665](track_runners.py#L635-L665) |
| `main()` — 第一段 | [track_runners.py:760](track_runners.py#L760) |
| `main()` — 第二段 | [track_runners.py:920](track_runners.py#L920) |
