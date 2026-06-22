# Project Structure

本專案以「程式碼、文件、模型、輸入資料、輸出結果」分區收納。根目錄保留主要入口檔與必要設定，實驗輸出和暫存狀態不要直接散放在根目錄。

## Top-Level Layout

```text
runner-analysis-pipeline/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── analyze.py
├── track_runners.py
├── core/
├── scripts/
├── MotionAGFormer/
├── assets/
├── models/
├── configs/
├── docs/
├── output_cut/
├── outputs/
└── runtime_state/
```

## Directory Guide

| Path | Purpose | Git policy |
| --- | --- | --- |
| `core/` | 跑者追蹤、pipeline、overlay、視覺化與共用工具。 | Commit |
| `run_pipeline.py` | 主要 pipeline 入口。 | Commit |
| `analyze.py` | 分析流程入口或輔助分析程式。 | Commit |
| `track_runners.py` | 跑者追蹤入口。 | Commit |
| `scripts/` | 可獨立執行的分析、追蹤、視覺化與工具腳本。 | Commit |
| `scripts/legacy/` | 舊版或一次性測試腳本，保留但不放在根目錄。 | Commit if still useful |
| `MotionAGFormer/` | 3D pose estimation 相關模型程式、設定與 demo 程式。 | Commit code/config only |
| `assets/` | 字型與其他小型靜態資源。 | Commit |
| `models/` | YOLO、MotionAGFormer 等模型權重。大型權重由 `.gitignore` 排除。 | Do not commit weights |
| `configs/` | 相機、homography 或實驗設定檔。 | Commit reusable configs |
| `docs/` | 專案文件、流程說明、使用手冊與參數說明。 | Commit |
| `docs/references/` | 課程 PDF、論文、外部參考資料。 | Do not commit large references |
| `output_cut/` | 預設 pipeline 輸出位置，部分腳本仍直接使用此路徑。 | Keep generated files ignored |
| `outputs/` | 手動整理後的實驗輸出、舊輸出與雜項結果。 | Do not commit generated outputs |
| `runtime_state/` | `.config.json`、`.last_input_video` 等本機執行狀態。 | Do not commit |
| `keypoints_raw_archive/` | MotionAGFormer / HRNet 產生的 keypoint raw archive。 | Do not commit generated archive |

## Current Cleanup

本次整理將根目錄散落檔案移到下列位置：

| Original files | New location |
| --- | --- |
| `0506_4_*` analysis CSV/JSON/NPZ/MP4 files | `outputs/0506_4/` |
| `angles.csv`, `output_final.mp4`, `sequential_tracked.mp4` | `outputs/misc/` |
| `.config.json`, `.last_input_video`, `.last_input_videos`, `.last_output_name` | `runtime_state/` |
| `EAI2025fall_*.pdf` | `docs/references/` |
| `100mTracking_demo_20251127.py`, `process.py` | `scripts/legacy/` |
| `step_homography_0331.yaml` | `configs/homography/` |
| `yolo26x.pt` | `models/` |

## Storage Rules

1. Keep source code in `core/`, root entry files, or a specific subfolder under `scripts/`.
2. Keep reusable configuration in `configs/`; avoid leaving YAML/JSON config files in the root unless they are project-level config.
3. Keep generated analysis results in `output_cut/` or `outputs/<experiment_name>/`.
4. Keep runtime state in `runtime_state/`; do not commit local last-run files.
5. Keep model weights in `models/`; large weights are ignored and should be downloaded or copied locally.
6. Keep project documentation in `docs/`; external PDFs and large references go under `docs/references/`.
7. Before pushing, run `git status -sb` and verify that generated output files are not staged.
