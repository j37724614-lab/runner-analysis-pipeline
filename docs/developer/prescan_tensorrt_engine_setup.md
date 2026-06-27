# Pipeline 開發者說明

## 近期更新（6/27–6/28）

### Auto Crop 升級（`core/tracking.py`）

追焦裁切從舊的矩形 median×2，改為**正方形 p90×1.25**。

- `_auto_crop_side_from_bbox_sizes(widths, heights)`：取 bbox 寬/高的 p90，取較大值 ×1.25 作為正方形邊長。
- `_auto_crop_from_selected_cache(frame_cache, preset_ids)`：two_pass 模式在 Pass 1 選完主跑者後，直接用主跑者 bbox 計算 crop，不需額外 dry-run。
- `online` 模式：仍用 dry-run 估算。
- `two_pass` 模式（預設）：Pass 1 選完主跑者後立即計算，比 dry-run 更準確。

### 分析計時（`core/pipeline.py`）

每個 pipeline 階段印出耗時，並在 session 輸出目錄寫入 `timing_report.json`。

```
⏱ [TIME] Step1/load_yolo_model_and_warmup: 2.31s
⏱ [TIME] Step1/prescan_person_frames: 4.82s
```

### HRNet 骨架後處理（`MotionAGFormer/demo/vis.py`）

以下四個函式依序作用於 2D keypoints，在送進 MotionAGFormer 3D lift 之前執行：

**`_correct_arm_swaps(kp, lookback=7, threshold=0.75)`**
側面跑步時左右手臂交叉，HRNet 偶爾把左肘/腕標成右邊。用過去 7 幀已修正歷史建立參考向量，比較換與不換標籤的 cost，只在換 cost < 75% 閾值時才執行。

**`_fill_left_arm_by_mirror(kp, final_bad_all)`**
左臂低信心幀的補值。側面跑步時左右臂擺動反相，因此用右臂向量水平翻轉來估計左臂位置（flip x、保留 y）。只在 LShoulder、RShoulder、右側對應關節都是好幀時才補。

**`_normalize_bone_lengths_2d(keypoints, blend=0.8)`**
Soft-constrain 每根骨骼長度。以全片該骨骼長度的中位數為參考，偏差超過 10% 的幀，以 80% blend 比例拉回目標長度。用 root-to-leaf 順序逐骨處理，保留正常運動的自然變形。

**`_apply_anatomical_limits_2d(keypoints)`**
防止不可能的 2D 關節角度。對膝、肘等關節設定最小角度限制，違反時旋轉遠端關節（distal joint）到最小角度邊界，不動近端關節。

---

## TensorRT INT8 Prescan Engine

**檔案**：`models/yolo26x_ultralytics_int8.engine`（82 MB）

**用途**：two_pass tracking 前的快速預掃描。用 INT8 YOLO 找出原始影片哪些 frame range 有人，讓 two_pass 只處理有效區段，節省時間。

**程式進入點**：

```
routes/upload.py → core/pipeline.py → core/tracking.py::run_temporal_prescan()
```

---

## 環境需求

已驗證版本（conda env: `yolo_new`）：

```
tensorrt    11.0.0.114
ultralytics 8.4.21
torch       2.7.0+cu128
opencv      4.13.0
numpy       1.26.3
```

安裝：

```bash
conda create -n yolo_new python=3.11 -y
conda activate yolo_new
pip install ultralytics opencv-python numpy tensorrt
```

---

## 啟動後端

> Placeholder 說明：`<REPO_ROOT>` = runner-analysis-pipeline 根目錄；`<BACKEND_ROOT>` = running-analysis-backend FastAPI 目錄；`<TRT_LIB_DIR>` = TensorRT runtime library 目錄（含 `libnvinfer.so`）；`<PYTHON_BIN>` = Python 執行檔；`<PORT>` = 服務 port。

啟動前必須設定 TensorRT runtime library path，否則 Ultralytics 載入 `.engine` 時會失敗：

```bash
export LD_LIBRARY_PATH=<TRT_LIB_DIR>:${LD_LIBRARY_PATH:-}
```

前景啟動：

```bash
cd <BACKEND_ROOT>
export LD_LIBRARY_PATH=<TRT_LIB_DIR>:${LD_LIBRARY_PATH:-}
<PYTHON_BIN> -m uvicorn main:app --host 0.0.0.0 --port <PORT>
```

背景執行：

```bash
setsid bash -c 'cd <BACKEND_ROOT> && \
export LD_LIBRARY_PATH=<TRT_LIB_DIR>:${LD_LIBRARY_PATH:-} && \
exec <PYTHON_BIN> -m uvicorn main:app --host 0.0.0.0 --port <PORT> \
> <BACKEND_LOG> 2>&1' < /dev/null &
```

---

## Config 設定

```python
{
    "auto_crop": True,
    "tracking_mode": "two_pass",
    "prescan_enabled": True,
    "prescan_engine_path": "<REPO_ROOT>/models/yolo26x_ultralytics_int8.engine",
}
```

`prescan_enabled=False` 時 two_pass tracking 仍執行，但跳過 prescan 掃描步驟。

---

## 驗證

**確認環境正常：**

```bash
LD_LIBRARY_PATH=<TRT_LIB_DIR>:${LD_LIBRARY_PATH:-} <PYTHON_BIN> -c "
import tensorrt as trt, ultralytics, cv2, torch
print('TRT:', trt.__version__, '| Ultralytics:', ultralytics.__version__, '| Torch:', torch.__version__)
"
```

**確認 engine 存在：**

```bash
ls -lh <REPO_ROOT>/models/yolo26x_ultralytics_int8.engine
```

**確認 prescan 有被執行（分析 log）：**

```
[prescan] ...
⏱ [TIME] Step1/prescan_person_frames: ...
```

Session 輸出目錄下也會有：

```
*_prescan_samples.csv   prescan 抽樣偵測結果
*_prescan_ranges.json   prescan 判定的人物有效 frame ranges
timing_report.json      各階段耗時記錄
```

---

## 常見錯誤

| 錯誤 | 原因 | 修正 |
|------|------|------|
| `libnvonnxparser.so.11: cannot open shared object file` | `LD_LIBRARY_PATH` 未設定 | `export LD_LIBRARY_PATH=<TRT_LIB_DIR>:...` 後重啟後端 |
| `[prescan] engine not found, skip: ...` | `prescan_engine_path` 路徑錯誤或檔案不存在 | 確認 `models/yolo26x_ultralytics_int8.engine` 存在 |
| `import tensorrt` crash / Bus error | TensorRT runtime `.so` 版本不符 | 確認 `LD_LIBRARY_PATH` 指到正確 TRT 11 runtime 目錄 |

---

## Git 注意事項

`.gitignore` 設定：全部 `*.engine` 忽略，只允許這一份追蹤：

```gitignore
*.engine
!models/yolo26x_ultralytics_int8.engine
```

目前 82MB，低於 GitHub 100MB 限制。若未來超過，需改用 Git LFS。
