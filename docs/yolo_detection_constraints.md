# YOLO 偵測與追蹤約束整理（Two-pass 模式）

更新版本：`3b367aebb4ab23986691f52c8793ee93fa6de647`

後端 `analyze.py:261` 強制使用 `two_pass` 模式。本文只說明 two-pass 實際執行的流程。

---

## 整體流程

```text
Pass 1  → YOLO 掃全片，收集所有候選（完整原始畫面，conf=0.25）
評分    → 依軌跡品質選出每台相機的主跑者 ID
Stitch  → 修補 frame_cache 裡短暫換 ID 的幀
Pass 2  → 不重跑 YOLO，用 frame_cache 輸出追焦影片
補幀    → frame_cache 為空的幀用線性插值填補
```

---

## Pass 1：YOLO 怎麼偵測

```python
model.track(
    frame,            # 完整原始畫面（不裁切）
    classes=[0],      # 只偵測 person
    conf=0.25,        # 較寬鬆，避免起跑時主跑者被漏掉
    iou=0.1,          # 低關聯門檻，高速位移時不易斷 ID
    imgsz=1280,
    half=True,        # FP16，加快掃描速度
    persist=True,
)
```

偵測後過濾：

- **高度**：bbox 高度 < `MIN_PERSON_HEIGHT = 40px` → 略過（排除遠景小人）
- **跑道投影**：把 bbox 底部中心（EMA 平滑，α=0.35）投影到 start/end line 方向，只保留投影值落在 `[-pre_roll_px, pixel_span + end_roll_px]` 區間內的人

```text
proj_px = dot(ground_pt - start_mid, track_dir)
有效：-200 <= proj_px <= pixel_span + 120
```

通過過濾的每筆偵測存入 `frame_cache[(cam_idx, frame_idx)]`，以及 `all_detections`（供後續評分使用）。

---

## 評分：選出主跑者

對每個 `(cam_idx, track_id)` 計算：

```text
score =
  0.25 × coverage        （出現幀數 / 總幀數）
+ 0.25 × progress        （(max_proj - min_proj) / pixel_span）
+ 0.20 × monotonic       （相鄰幀投影遞增比例）
+ 0.15 × start_proximity （1 / (1 + first_proj_px)）
+ 0.10 × roi_ratio       （0 ≤ proj_px ≤ pixel_span 的幀比例）
```

候選過濾門檻（不過則不進評分）：

| 條件 | 值 |
|---|---:|
| 最少出現幀數 | `>= 10` |
| progress | `>= 0.3` |
| monotonic | `>= 0.55` |

- 過濾後無人 → 退回全部候選保底
- 完全無偵測 → 印警告，退回 online 模式
- 前兩名分數差 < 10% → 印不穩定警告

Debug 輸出：`*_all_tracks.csv`、`*_track_summary.csv`、`*_selected_runner.json`

---

## Pass 2：不重跑 YOLO

```python
_cached = [d for d in frame_cache[(cam_idx, frame)] if d['track_id'] == target_id]
process_frame(..., cached_detections=_cached)
# cached_detections 不為 None → 跳過 YOLO，直接用快取 bbox
```

Pass 2 對每幀以 bbox 中心為圓心，切出 `CROP_WIDTH=200 × CROP_HEIGHT=260` px 的追焦方塊。offset 存入 `*_offsets.npz`，overlay 階段用來把骨架貼回原始影片座標。

輸出：

```text
sequential_tracked.mp4
*_bbox_map.csv     ← 含 is_interpolated / interp_gap_len
*_frame_map.csv
*_offsets.npz
```

---

## ID Stitch：修補短暫換 ID

### 問題背景

ByteTrack 在追蹤時，偶爾會把**同一個人**在幾幀內誤判成不同的 ID。`max_gap` 由影片 FPS 自動推算（約 0.2 秒），上限 15 幀：

| FPS | max_gap | 實際時間 |
|-----|---------|---------|
| 30  | 6 幀    | 0.20 s  |
| 60  | 12 幀   | 0.20 s  |


例如：

```
Frame 10: 主跑者 ID=3 ✓
Frame 11: 主跑者 ID=3 ✓
Frame 12: 主跑者 ID=7  ← ByteTrack 誤換了 ID
Frame 13: 主跑者 ID=7  ← 還是誤換
Frame 14: 主跑者 ID=3 ✓（恢復正確）
```

此時 `frame_cache` 裡 frame 12-13 找不到 `target_id=3`，Pass 2 只能用線性插值補，骨架位置可能不準。Stitch 的目的是**把那幾幀其實是主跑者的偵測，改回 `target_id`**，讓 Pass 2 可以直接使用真實 bbox。

### 演算法步驟

**Step 1：找空缺**

掃描 `frame_cache`，找出 `target_id` 連續出現幀之間的空缺：

```
有 target_id=3：[frame 10, 11, 14, ...]
空缺：frame 12, 13（gap = 2，≤ 5 → 進行修補）
```

**Step 2：算出空缺幀的「預期位置」**

用空缺前一幀（frame 11）和後一幀（frame 14）的 bbox，線性插值算出空缺幀的預期中心座標與 bbox 高度：

```
frame 12 預期中心 = frame11_center * (2/3) + frame14_center * (1/3)
frame 13 預期中心 = frame11_center * (1/3) + frame14_center * (2/3)
```

**Step 3：對空缺幀裡其他 ID 評分**

在 frame 12 裡找所有被偵測到的人（例如 ID=7），計算分數：

```python
dist = hypot(actual_cx - expected_cx, actual_cy - expected_cy)
# dist：實際中心到預期中心的像素距離

size_ratio = max(actual_h, expected_h) / min(actual_h, expected_h)
# size_ratio：高度差異比例，1.0 表示完全一樣高

score = dist + 50.0 * (size_ratio - 1.0)
# 距離越遠 or 大小差越多 → 分數越高（越差）
```

過濾條件（太離譜的直接排除）：
- `dist > 100px`：位置差太遠，不可能是同一人
- `size_ratio > 1.5`：大小差太多，不可能是同一人

**Step 4：取最低分者，改寫 ID**

分數最低的偵測（例如 ID=7）→ in-place 改為 `target_id=3`，完成修補。

### 分數公式說明

`50 × (size_ratio - 1)` 的係數 50 讓大小異常的懲罰很重，避免把旁邊體型差很多的路人誤接回主跑者：

| 情況 | dist | size_ratio | score |
|------|------|-----------|-------|
| 完美匹配（位置準、大小一樣） | 5px | 1.0 | **5** |
| 位置稍偏、大小正常 | 30px | 1.1 | **35** |
| 位置接近、但高度差 40% | 10px | 1.4 | **30** |

---

## BBox 補幀

Pass 2 過程中 `frame_cache` 為空（stitch 也無法填補）但前後有有效 bbox 時，線性插值：

```python
ratio = idx / (gap_len + 1)
interp_bbox = _interpolate_bbox(last_valid_bbox, right_bbox, ratio)
```

`bbox_map.csv` 標記 `is_interpolated=1`，追焦影片以紫色框顯示。

---

## 距離換算優先順序

`core/tracker_impl.py` 的 `_make_calibration()`：

1. **Line calibration**（web flow 走這條）：有 `start_line`/`end_line` + `distance_m` → 直接用 pixel_span 換算
2. Homography：有四錨點 → 計算單應矩陣，condition number > 5000 退回線性投影
3. Pixel-only：僅 ROI 座標，粗估
