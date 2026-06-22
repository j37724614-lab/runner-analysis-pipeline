# 距離與速度計算說明

來源：`core/tracker_impl.py`

---

## 整體流程

```text
影像像素座標
  → 校正（m_per_pixel 或 Homography）
  → proj_px（跑道方向投影像素）
  → world_x_m（公尺距離）
  → 單調約束 + Butterworth 低通濾波
  → Kalman 濾波
  → d_smooth / v_smooth / a（輸出）
```

---

## Step 1：像素距離換算（校正）

### 方式 A：線段校正（web flow 使用這條）

前端傳入 `start_line`、`end_line`、`distance_m`，系統計算兩條線中點的像素距離：

```python
start_mid = 兩點平均（start_line）
end_mid   = 兩點平均（end_line）
pixel_span = hypot(end_mid.x - start_mid.x, end_mid.y - start_mid.y)

m_per_pixel = distance_m / pixel_span
```

`track_dir`（跑道方向單位向量）也在此步驟計算：

```python
dx = end_mid.x - start_mid.x
dy = end_mid.y - start_mid.y
track_dir = (dx / pixel_span, dy / pixel_span)
```

### 方式 B：Homography（四錨點，斜視相機）

用 `start_line`（2 點）＋ `end_line`（2 點）共 4 個像素座標，對應真實世界座標：

```
像素 → 世界座標（公尺）
start_line[near] → (0,    1.22)
start_line[far]  → (0,    0   )
end_line[near]   → (20m,  1.22)
end_line[far]    → (20m,  0   )
```

計算 3×3 Homography 矩陣 `H`，直接將任何像素點換算成世界座標（x_m, y_m）。

若 H 矩陣條件數 > 5000（側拍相機近乎奇異），退回方式 A。

---

## Step 2：每幀距離計算

### 方式 A（線段校正）

把 bbox 底部中心投影到跑道方向：

```python
proj_px = dot(ground_pt - start_mid, track_dir)
world_x_m = proj_px * m_per_pixel
```

`ground_pt` 是 bbox 底部中心，經 EMA 平滑（α=0.35）降低 bbox 抖動影響。

### 方式 B（Homography）

**重要：輸入不是 bbox 底部，而是 bbox 中心 X + 固定 track_y**

```
bbox 底部 Y 會隨跑者起腳而上下抖動，
這個 Y 噪聲餵進 H 矩陣後會被放大成 world_x 誤差。
因此 Homography 模式固定用跑道中線 Y，只讓 X 軸影響結果。
```

實際計算：

```python
# 1. bbox 座標（在追焦裁切空間）→ 還原到原始影像座標
cx_orig = (x1 + x2) / 2.0 + off_x + crop_x_offset   # bbox 中心 X
y2_orig = y2 + off_y + crop_y_offset                   # bbox 底部 Y（不使用）

# 2. 固定 track_y = 四個錨點 Y 的平均（跑道中線高度）
track_y = (sl[0][1] + sl[1][1] + el[0][1] + el[1][1]) / 4.0

# 3. 餵進 H 矩陣的點是 (cx_orig, track_y)，而不是 bbox 底部
world = H @ [cx_orig, track_y, 1]   # 齊次座標
world_x_m = world[0] / world[2]     # 正規化後取 X（公尺）

# 4. 扣掉起跑線世界座標，得到本段距離
local_dist = world_x_m - homography_start_x
```

**座標還原流程（bbox_map.csv → 原始影像）**：

```
bbox_map.csv 的 x1,y1,x2,y2
  ↓ + offsets.npz 的 (off_x, off_y)    ← 追焦裁切的左上角偏移
  ↓ + crop_x_offset, crop_y_offset      ← 前處理裁切的偏移（two-pass 為 0）
  = 原始影像座標
```

`offsets.npz` 結構：

| 欄位 | 說明 |
|------|------|
| `offsets` | shape (N, 2)，每個輸出幀的 (off_x, off_y) |
| `orig_frames` | 對應的原始影片幀編號 |
| `cam_indices` | 相機編號 |

---

## Step 3：速度與加速度計算

來源函式：`_smooth_distance_velocity(d_raw, fps, ...)`

### 3a. 單調約束

距離只能遞增，不能倒退：

```python
for k in range(1, n):
    if d[k] < d[k-1]:
        d[k] = d[k-1]
```

### 3b. Flat segment 插值

距離連續幾幀卡在同一個值，代表 bbox 沒動（YOLO 漏偵測），用前後有效幀線性插值補齊，避免之後的速度計算突然跳動。

### 3c. Butterworth 低通濾波（雙向 filtfilt）

截止頻率 3.5 Hz，去除 bbox 抖動（30+ Hz 高頻噪聲），保留真實加速度變化（0~1 Hz）：

```python
b, a = butter(2, 3.5 / (fps / 2.0), btype='low')
d_smooth = filtfilt(b, a, d)   # 雙向，不造成相位延遲
```

需至少 15 幀才執行，否則直接用原始距離。

### 3d. Kalman 濾波

狀態向量：`[位置, 速度, 加速度]`

```
狀態轉移（等加速運動）：
  x(t+1) = F · x(t)
  F = [[1, dt, 0.5*dt²],
       [0,  1,      dt ],
       [0,  0,       1 ]]

觀測：z = [距離]（只觀測位置）
```

噪聲矩陣：

| 參數 | 值 | 說明 |
|------|-----|------|
| `Q[0,0]` | 0.001 | 位置過程噪聲（低） |
| `Q[1,1]` | 0.01  | 速度過程噪聲 |
| `Q[2,2]` | 0.15  | 加速度過程噪聲（較高，允許 ≤5 m/s² 真實加速） |
| `R`      | 0.15 / conf | 觀測噪聲；插值幀 conf 較低 → R 較大 → Kalman 較不信任 |

**跨相機連續性**：`init_v`、`init_a` 從前一台相機末幀傳入，避免切換時速度從 0 重新爬升。

輸出：

```python
d_smooth  # 平滑後距離（公尺）
v_smooth  # 速度（公尺/秒），強制 >= 0
a         # 加速度（公尺/秒²）
```

---

## 輸出欄位（metrics.csv）

| 欄位 | 說明 |
|------|------|
| `world_x_m` | 每幀跑者位置（公尺，從起跑線算起） |
| `speed_ms` | 瞬時速度（公尺/秒） |
| `accel_ms2` | 瞬時加速度（公尺/秒²） |
| `absolute_frame` | 跨相機連續幀編號 |

---

## 參數總覽

| 參數 | 預設值 | 說明 |
|------|--------|------|
| Butterworth 截止頻率 | 3.5 Hz | 去除 bbox 抖動 |
| Kalman Q[2,2] | 0.15 | 加速度允許變化量 |
| Kalman base_R | 0.15 | 觀測噪聲基準值 |
| Homography 條件數上限 | 5000 | 超過退回線段校正 |
| flat_interp_eps_m | 0.001 m | 距離變化小於此值視為卡住 |
