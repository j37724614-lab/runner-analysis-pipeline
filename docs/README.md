# Runner Analysis Pipeline

多相機跑者動作分析 Pipeline：自動追蹤跑者、估計 2D/3D 姿態、計算關節角度並輸出疊加影片。

---

## 流程概覽

```
輸入影片（多台相機）
    │
    ▼ Step 1 [Tracking] (core.tracking)
  YOLO + ByteTrack 追蹤主跑者，輸出追焦裁切影片、bbox_map、frame_map、offsets
    │
    ▼ Step 2 [Pose] (MotionAGFormer/demo/vis.py)
  HRNet 2D keypoints + MotionAGFormer 3D pose + 關節角度 CSV
    │
    ▼ Step 3 [Metrics] (core.tracker_impl)
  直接讀 bbox_map 計算距離、速度、加速度，不重跑 YOLO
    │
    ▼ Step 4 [Overlay + Step/Stride] (core.overlay + scripts.analysis.ankle_step_stride)
  骨架貼回原影片，偵測落地點、步頻、步幅，輸出 final video
    │
    ▼ 輸出
  output_final.mp4 / metrics.csv / angles.csv / cam1_step_events.csv
```

目前後端一鍵分析入口是 `core.pipeline.run_analysis()`。它會先跑 tracking 與 pose，接著平行執行原影片 overlay 與步頻/步幅分析，最後把 `cam1_step_events.csv` 的落地點標回 `output_final.mp4`，並轉成 Web 播放相容格式。

---

## 系統需求

- Python 3.10 / 3.11
- CUDA GPU（建議 8GB VRAM 以上）
- ffmpeg（用於影片網頁轉碼與 MOV 轉檔，需另行安裝）

```bash
# 安裝 Python 套件
pip install -r requirements.txt

# 安裝 ffmpeg（Ubuntu/Debian）
sudo apt install ffmpeg
```

---

## 新目錄結構 (重構後)

```
runner-analysis-pipeline/
├── analyze.py                       ← [入口] 綜合一鍵分析 CLI (調用 core.pipeline)
├── run_pipeline.py                  ← [入口] 跑者分析生物力學 CLI (調用 core.pipeline)
├── track_runners.py                 ← [入口] 速度與加速度追蹤 CLI (調用 core.tracker)
├── requirements.txt                 ← Python 套件清單
│
├── core/                            ← [新增] 核心實作模組套件
│   ├── __init__.py                  ← 暴露主要 Package API 介面
│   ├── pipeline.py                  ← 整合生物力學分析與一鍵分析之核心排程引擎
│   ├── tracker.py                   ← 提取自 track_runners 的 Kalman 濾波與速度追蹤
│   ├── overlay.py                   ← 提取自 overlay_original 的原比例骨架疊加核心
│   ├── visualization.py             ← 提取自 add_angle_overlay 的角度折線圖合併模組
│   └── utils.py                     ← 共享基礎函式 (動態資源定位、中文字型配置、FFmpeg 網頁轉碼)
│
├── assets/                          ← [新增] 靜態資源目錄
│   └── fonts/
│       └── ChineseFont.ttf          ← 移入此處 (中文字型檔)
│
├── models/                          ← [新增] 模型權重目錄
│   ├── yolo11x.pt                   ← YOLOv11 權重
│   └── yolo26x.pt                   ← YOLO26x 權重 (由根目錄移入)
│
├── scripts/                         ← 輔助腳本目錄
│   ├── tracking/
│   │   └── track_crop_roi.py        ← YOLO 追蹤與裁剪
│   ├── visualization/
│   │   └── add_angle_overlay.py     ← 角度圖表合併
│   └── utilities/                   ← [新增] 獨立實用工具目錄
│       ├── convert_mov_to_mp4.py    ← iPhone MOV 轉 MP4 工具 (由根目錄移入)
│       └── extract_frames.py        ← 影片提取影格工具 (由根目錄移入)
│
└── MotionAGFormer/                  ← 第三方 3D 重建套件目錄 (完全保留未修改)
    ├── checkpoint/                  ← MotionAGFormer-L 權重 (motionagformer-l-h36m.pth.tr)
    └── demo/
        ├── vis.py                   ← 姿態預測入口
        └── lib/
            └── checkpoint/          ← HRNet + YOLOv3 權重 (pose_hrnet_w48_384x288.pth 等)
```

---

## 模型權重下載與放置

執行前必須手動下載以下權重檔並放置於對應目錄。

### 1. YOLO 權重 → 放在 `models/` 目錄下

從本 repo 的 [Releases](https://github.com/j37724614-lab/runner-analysis-pipeline/releases) 頁面下載：
- `yolo11x.pt`
- `yolo26x.pt`

下載後放置：
```
runner-analysis-pipeline/
└── models/
    ├── yolo11x.pt   ← 放這裡
    └── yolo26x.pt   ← 放這裡
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

若影片為 iPhone 拍攝的 `.MOV` 格式，可利用 `scripts/utilities/` 中的工具轉換：

```bash
# 執行轉換指令 (請先編輯 convert_mov_to_mp4.py 修改輸入檔案清單)
python scripts/utilities/convert_mov_to_mp4.py
```

---

### 第二步：建立相機設定檔

建立 `my_config.yaml`，填入你的影片路徑：

```yaml
cameras:
  - video_path: test/test/cam1.mov
    crop: [0, 400, 1920, 800]
    start_line: [[222, 715], [148, 725]]
    end_line: [[1700, 710], [1790, 718]]
    distance_m: 20.0

  - video_path: test/test/cam2.mov
    crop: [0, 400, 1920, 800]
    start_line: [[220, 715], [135, 725]]
    end_line: [[1730, 710], [1825, 725]]
    distance_m: 20.0
```

> **注意：** 所有相機與模型資源路徑現在皆已在 `core/utils.py` 中進行「動態根目錄定位 (PathManager)」，因此在任何路徑執行程式皆不會發生路徑找不到的 `FileNotFoundError`。

---

### 第三步：執行一鍵分析 (Preferred)

一鍵分析會執行：【運動表現追蹤 (Phase 1)】 + 【姿態與角度估計 (Phase 2)】 + 【原比例原影片骨架疊加 (Phase 3)】，並轉碼為**瀏覽器直接預覽播放之 Web 相容 MP4 影片**：

```bash
python analyze.py
```

您也可以單獨執行姿態與生物力學管線：

```bash
# 完整姿態流程（GPU 0）
python run_pipeline.py --config my_config.yaml

# 只跑 2D（跳過 3D 與角度以求加速）
python run_pipeline.py --config my_config.yaml --2d_only

# 略過追蹤（若追蹤影片已存在）
python run_pipeline.py --config my_config.yaml --skip-track
```

或是單獨執行跑者速度追蹤：

```bash
python track_runners.py
```

---

## 輸出檔案說明

執行一鍵分析完成後，結果會整合輸出在 run session 的分析目錄下：

| 檔案路徑 | 說明 |
|------|------|
| `metrics.csv` | 運動表現數據，含逐幀距離、速度、加速度。來源是 `*_bbox_map.csv`，不再重跑 YOLO。 |
| `angles.csv` | 生物力學關節角度數據，含 `frame`、對應 `output_final.mp4` 的 `time_sec`、膝、髖、肘、肩及骨盆軀幹角度。 |
| `cam1_step_events.csv` | 落地事件、步頻、步幅、腳踝位置與 confidence。 |
| `output_final.mp4` | Web 相容的最終影片，包含原影片骨架 overlay 與步頻/步幅標註。 |
| `final_video_frames/output_final/` | `output_final.mp4` 匯出的逐幀 PNG。 |
| `sequential_tracked/` | MotionAGFormer 2D/3D pose、HRNet confidence、keypoint smoothing debug 等中間資料。 |
| `cam1_tracked_bbox_map.csv` | 追焦 bbox 對應表，含原始 frame、crop 內 bbox、原圖 bbox、補幀標記。 |
| `cam1_tracked_frame_map.csv` | 追焦輸出 frame 對應原影片 frame。 |
| `cam1_offsets.npz` | 每個追焦輸出 frame 的 crop offset，用來把骨架貼回原影片座標。 |

---

## 參考與感謝

- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer) — 3D 姿態估計模型
- [MHFormer Demo](https://github.com/Vegetebird/MHFormer) — HRNet + YOLOv3 2D 姿態估計框架
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — YOLO 多相機追蹤
