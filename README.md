# Runner Analysis Pipeline

多相機跑者動作分析 Pipeline：自動追蹤跑者、估計 2D/3D 姿態、計算關節角度並輸出疊加影片。

---

## 流程概覽

```
輸入影片（多台相機）
    │
    ▼ Step 1 track_crop_roi.py
  YOLO 追蹤最快跑者，裁剪並合併成單一影片
    │
    ▼ Step 2
  自動複製影片到 vis.py 輸入目錄
    │
    ▼ Step 3 vis.py
  HRNet 2D 姿態估計 + MotionAGFormer 3D 重建 + 關節角度計算
    │
    ▼ Step 4 add_angle_overlay.py
  2D 骨架影片 + 4 個角度折線圖合併輸出
    │
    ▼ 輸出
  _2D.mp4 / .mp4 / _2D_angles.mp4 / _angles.csv
```

---

## 系統需求

- Python 3.11
- CUDA GPU（建議 8GB VRAM 以上）
- ffmpeg（用於 MOV → MP4 轉換，需另行安裝）

```bash
# 安裝 Python 套件
pip install -r requirements.txt

# 安裝 ffmpeg（Ubuntu/Debian）
sudo apt install ffmpeg
```

---

## 目錄結構

```
runner-analysis-pipeline/
├── run_pipeline.py                  ← 主程式（從這裡執行）
├── example_config.yaml              ← 相機設定範例
├── requirements.txt                 ← Python 套件清單
├── convert_mov_to_mp4.py            ← iPhone MOV 轉 MP4 工具
├── yolo11x.pt                       ← YOLO 權重（見下方下載說明）
├── output_cut/                      ← Step 1 輸出目錄（自動建立）
├── scripts/
│   ├── tracking/
│   │   └── track_crop_roi.py
│   └── visualization/
│       └── add_angle_overlay.py
└── MotionAGFormer/
    ├── checkpoint/                  ← AGFormer 權重（見下方下載說明）
    │   └── motionagformer-l-h36m.pth.tr
    ├── demo/
    │   ├── vis.py
    │   ├── video/                   ← Step 2 自動放置輸入影片
    │   └── lib/
    │       └── checkpoint/          ← HRNet + YOLOv3 權重（見下方下載說明）
    │           ├── pose_hrnet_w48_384x288.pth
    │           ├── h36m_sh_conf_cam_source_final.pkl
    │           └── yolov3.weights
    ├── model/
    └── configs/
```

---

## 模型權重下載

執行前必須手動下載以下權重檔並放置於對應目錄。

### 1. `yolo11x.pt` → 放在根目錄

從本 repo 的 [Releases](https://github.com/j37724614-lab/runner-analysis-pipeline/releases) 頁面下載：

```
runner-analysis-pipeline/
└── yolo11x.pt   ← 放這裡
```

---

### 2. MotionAGFormer 權重 → 放在 `MotionAGFormer/checkpoint/`

從 Google Drive 下載（選 L 版本，vis.py 預設使用）：

| 版本 | 下載連結 |
|------|----------|
| **MotionAGFormer-L（推薦）** | [下載](https://drive.google.com/file/d/1WI8QSsD84wlXIdK1dLp6hPZq4FPozmVZ/view?usp=sharing) |
| MotionAGFormer-B | [下載](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view?usp=drive_link) |
| MotionAGFormer-S | [下載](https://drive.google.com/file/d/1DrF7WZdDvRPsH12gQm5DPXbviZ4waYFf/view?usp=sharing) |

下載後放置：
```
MotionAGFormer/checkpoint/
└── motionagformer-l-h36m.pth.tr   ← 放這裡
```

---

### 3. HRNet + YOLOv3 權重 → 放在 `MotionAGFormer/demo/lib/checkpoint/`

從 Google Drive 打包下載（來源：MotionAGFormer 官方 Demo）：

[下載 YOLOv3 + HRNet 權重包](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing)

包含三個檔案，下載後放置：
```
MotionAGFormer/demo/lib/checkpoint/
├── pose_hrnet_w48_384x288.pth         (244 MB)
├── h36m_sh_conf_cam_source_final.pkl  (1.1 GB)
└── yolov3.weights                     (237 MB)
```

---

## 使用方式

### 第一步：（選用）iPhone MOV 轉 MP4

若影片為 iPhone 拍攝的 `.MOV` 格式，先用工具轉換：

```bash
# 編輯 convert_mov_to_mp4.py，在 video_list 填入你的 MOV 路徑，然後執行：
python convert_mov_to_mp4.py
```

轉換後會在同目錄產生同名的 `.mp4` 檔案。

---

### 第二步：建立相機設定檔

複製範例設定並修改影片路徑：

```bash
cp example_config.yaml my_config.yaml
```

編輯 `my_config.yaml`，填入你的影片路徑：

```yaml
cameras:
  - video_path: /path/to/camera1.mp4
    crop: [0, 450, 1920, 800]   # [x起, y起, x終, y終]
    roi_x: [150, 1820]
    switch_x: 1800

  - video_path: /path/to/camera2.mp4
    crop: [0, 450, 1920, 800]
    roi_x: [150, 1820]
    switch_x: 1800

# 自動決定裁剪尺寸（建議第一次使用時開啟）
auto_crop: true
```

> 所有路徑設定均以 repo 根目錄為基準自動推算，**clone 後不需修改任何程式碼**。

---

### 第三步：執行 Pipeline

```bash
# 完整流程（GPU 0）
python run_pipeline.py --config my_config.yaml

# 指定 GPU
python run_pipeline.py --config my_config.yaml --gpu 1

# 只跑 2D（跳過 3D 與角度，速度較快）
python run_pipeline.py --config my_config.yaml --2d_only

# 略過追蹤（追蹤影片已存在時使用）
python run_pipeline.py --config my_config.yaml --skip-track
```

---

## 輸出檔案

輸出位於 `MotionAGFormer/demo/output/{影片名稱}/`：

| 檔案 | 說明 |
|------|------|
| `{name}_2D.mp4` | 2D 骨架疊加影片 |
| `{name}.mp4` | 2D + 3D 並排影片 |
| `{name}_2D_angles.mp4` | 2D 影片 + 4 個角度折線圖 |
| `pred_3D/angles/{name}_angles.csv` | 逐幀關節角度數據 |

角度 CSV 包含以下欄位：

```
frame, left_knee_angle, right_knee_angle,
left_hip_angle, right_hip_angle,
left_elbow_flexion_angle, right_elbow_flexion_angle,
left_shoulder_flexion, right_shoulder_flexion,
pelvis_torso_angle
```

---

## 參考來源

- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer) — 3D 姿態估計模型
- [MHFormer Demo](https://github.com/Vegetebird/MHFormer) — HRNet + YOLOv3 2D 姿態估計框架
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — YOLO 多相機追蹤
