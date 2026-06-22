# Two-pass 兩階段追蹤：實作說明

更新版本：`3b367aebb4ab23986691f52c8793ee93fa6de647`

Two-pass 已實作於 `core/tracking.py`。後端 `analyze.py` 預設強制使用 `two_pass` 模式（`analyze.py` 第 261 行），`core/tracking.py` 的 module-level 預設為 `'online'`。

Two-pass 的目的不是取代 YOLO，而是把「主跑者 ID 選擇」從逐幀即時決策改成離線全片評分。它會先收集所有候選軌跡，再選出最像主跑者的 ID，最後用快取 bbox 輸出追焦影片與 `bbox_map.csv`。

## 模式差異

```text
online   ：逐幀即時選 ID
two_pass ：全片收集候選軌跡，評分後再輸出主跑者（後端預設）
```

若要比較穩定性，可以用同一批影片切換 `tracking_mode`，觀察 `*_all_tracks.csv`、`*_track_summary.csv`、`*_selected_runner.json` 與 final output。

## 第一階段：收集候選軌跡

每一幀照常跑：

```text
YOLO → ByteTrack → crop / ROI / start_line / end_line 過濾
```

但第一階段不立刻決定主跑者，而是把所有通過基本條件的人記錄下來。候選資料包含：

- `frame_idx`
- `track_id`
- `bbox`
- `bbox_center`
- `bbox_bottom_center`
- `bbox_height`
- `proj_px`
- `world_x_m`
- `in_roi`
- `confidence`

輸出：

```text
*_all_tracks.csv
```

## 第二階段：軌跡摘要

程式把同一個 `track_id` 的資料串成軌跡，並計算摘要：

- 出現幀數
- 出現比例
- 起始位置
- 結束位置
- 前進距離
- 單調前進比例
- ROI 合法比例
- 起跑線接近程度

輸出：

```text
*_track_summary.csv
```

## 第三階段：主跑者評分

目前評分公式：

```text
score =
  0.25 * coverage
+ 0.25 * progress
+ 0.20 * monotonic
+ 0.15 * start_proximity
+ 0.10 * roi_ratio
```

候選過濾：

| 條件 | 目前值 |
|---|---:|
| 最少出現幀數 | `>= 10` |
| 進度 | `>= 0.3` |
| 單調性 | `>= 0.55` |

如果最佳與第二名分數差距小於最佳分數的 10%，會輸出不穩定警告，方便檢查是否有多人競爭或主跑者被誤選。

輸出：

```text
*_selected_runner.json
```

## 第四階段：ID Stitch

如果 ByteTrack 中途短暫換 ID 或漏 ID，`_stitch_target_id()` 會嘗試把接近預測位置的 bbox 補回主跑者 ID。

| 參數 | 目前值 | 作用 |
|---|---:|---|
| `max_gap` | `5 frames` | 只修補短缺口。 |
| `max_dist_px` | `100 px` | 候選 bbox 中心不得離線性預測位置太遠。 |
| `size_ratio` | `<= 1.5` | 候選 bbox 高度與預測高度比例不能差太多。 |

## 第五階段：用快取 bbox 輸出

選定主跑者後，第二遍不再重跑 YOLO，而是使用第一遍快取的 bbox 輸出：

```text
sequential_tracked.mp4
*_bbox_map.csv
*_frame_map.csv
*_offsets.npz
```

若中間仍有短暫漏 bbox，追焦輸出階段會用前後有效 bbox 做線性插值。`bbox_map.csv` 會寫入：

```text
is_interpolated
interp_gap_len
```

## 優點與限制

優點：

- 比逐幀 online 模式更不容易中途選錯人。
- 適合離線分析影片。
- 會留下 `all_tracks`、`track_summary`、`selected_runner` 方便 debug。
- 已包含短 ID gap 修補。

限制：

- 需要至少掃描影片兩次，耗時比 online 長。
- 如果 YOLO 完全漏掉主跑者，two-pass 無法從不存在的候選中救回。
- 如果 crop/ROI/start_line 設錯，把主跑者排除，two-pass 也會選不到正確跑者。
- 多人軌跡高度重疊時，仍可能需要調整 ROI 或起終點線。
