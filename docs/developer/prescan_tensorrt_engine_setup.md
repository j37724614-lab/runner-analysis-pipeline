# TensorRT INT8 Prescan Engine 啟用說明

本文說明如何正確啟用：

```text
models/yolo26x_ultralytics_int8.engine
```

這份 engine 目前用在 tracking 前的 temporal prescan。目的不是取代正式 two-pass YOLO tracking，而是先快速掃描原始影片，找出有人物出現的 frame range，讓後面的 two-pass tracking 只處理有效區段。

---

## 1. 會使用這份 engine 的程式位置

後端分析流程會依序經過：

```text
backend routes/upload.py
→ core/pipeline.py
→ core/tracking.py::run_temporal_prescan()
```

真正載入 TensorRT engine 的位置在：

```text
core/tracking.py::run_temporal_prescan()
```

關鍵程式邏輯：

```python
model = YOLO(PRESCAN_ENGINE_PATH, task="detect")
```

`PRESCAN_ENGINE_PATH` 由後端 config 傳入，預設會指到：

```text
models/yolo26x_ultralytics_int8.engine
```

---

## 2. 需要的 Python 套件

已驗證可用的環境格式如下；實際路徑請依部署機器調整：

```text
conda env: yolo_new
python: <PYTHON_BIN>
```

已驗證版本：

```text
tensorrt    11.0.0.114
ultralytics 8.4.21
opencv      4.13.0
torch       2.7.0+cu128
numpy       1.26.3
```

必要套件至少包含：

```bash
pip install ultralytics opencv-python numpy
pip install tensorrt
```

如果使用 conda，也可以先建立環境後再安裝：

```bash
conda create -n yolo_new python=3.11 -y
conda activate yolo_new
pip install ultralytics opencv-python numpy tensorrt
```

注意：`tensorrt` Python package 只提供 Python binding，實際執行還需要 TensorRT runtime `.so` library。

---

## 3. TensorRT runtime library

這份 engine 需要 TensorRT 11 runtime library。若 runtime library path 沒有設定正確，常見錯誤會是：

```text
libnvonnxparser.so.11: cannot open shared object file
```

TensorRT runtime library 目錄以 `<TRT_LIB_DIR>` 表示。部署時請將它替換成實際 TensorRT library 目錄，例如包含 `libnvinfer.so` 與 `libnvonnxparser.so` 的資料夾：

```text
<TRT_LIB_DIR>
```

啟動後端前需要設定：

```bash
export LD_LIBRARY_PATH=<TRT_LIB_DIR>:${LD_LIBRARY_PATH:-}
```

如果沒有這段，`import tensorrt` 可能成功也可能失敗；即使 Python package 存在，Ultralytics 載入 `.engine` 時仍可能因為找不到 `.so` 而失敗。

---

## 4. 啟動後端的正確方式

後端應使用 `yolo_new` 環境，並帶入 TensorRT runtime library path：

本文使用以下 placeholder：

```text
<REPO_ROOT>     runner-analysis-pipeline repository 根目錄
<BACKEND_ROOT>  running-analysis-backend 的 FastAPI app 目錄
<TRT_LIB_DIR>   TensorRT runtime library 目錄
<PYTHON_BIN>    要啟動後端的 Python executable
<PORT>          後端服務 port
<BACKEND_LOG>   後端 log 輸出檔
```

```bash
cd <BACKEND_ROOT>

export LD_LIBRARY_PATH=<TRT_LIB_DIR>:${LD_LIBRARY_PATH:-}

<PYTHON_BIN> -m uvicorn main:app \
  --host 0.0.0.0 \
  --port <PORT>
```

背景執行範例：

```bash
setsid bash -c 'cd <BACKEND_ROOT> && \
export LD_LIBRARY_PATH=<TRT_LIB_DIR>:${LD_LIBRARY_PATH:-} && \
exec <PYTHON_BIN> -m uvicorn main:app --host 0.0.0.0 --port <PORT> \
> <BACKEND_LOG> 2>&1' < /dev/null &
```

---

## 5. 後端 config 需要包含的設定

Flutter 上傳影片後，後端建立的 `config_dict` 需要包含：

```python
{
    "auto_crop": True,
    "tracking_mode": "two_pass",
    "prescan_enabled": True,
    "prescan_engine_path": "<REPO_ROOT>/models/yolo26x_ultralytics_int8.engine",
}
```

關鍵欄位：

```text
tracking_mode = two_pass
prescan_enabled = True
prescan_engine_path = engine 檔案路徑
```

如果 `prescan_enabled=False`，two-pass tracking 仍會執行，但不會使用這份 TensorRT INT8 engine 先掃有效 frame range。

---

## 6. 快速驗證 TensorRT 環境

使用正確 `LD_LIBRARY_PATH` 後，執行：

```bash
cd <REPO_ROOT>

LD_LIBRARY_PATH=<TRT_LIB_DIR>:${LD_LIBRARY_PATH:-} \
<PYTHON_BIN> - <<'PY'
import tensorrt as trt
import ultralytics
import cv2
import torch

print("TensorRT:", trt.__version__)
print("Ultralytics:", ultralytics.__version__)
print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__)
PY
```

正常輸出應類似：

```text
TensorRT: 11.0.0.114
Ultralytics: 8.4.21
OpenCV: 4.13.0
Torch: 2.7.0+cu128
```

確認 engine 檔案存在：

```bash
ls -lh <REPO_ROOT>/models/yolo26x_ultralytics_int8.engine
```

目前檔案大小約：

```text
82M
```

---

## 7. 如何確認 prescan 有被執行

跑完整分析後，log 應該會出現：

```text
[prescan] ...
⏱ [TIME] Step1/prescan_person_frames: ...
```

session 輸出資料夾也會產生：

```text
*_prescan_samples.csv
*_prescan_ranges.json
```

其中：

```text
*_prescan_samples.csv  記錄 prescan 抽樣偵測結果
*_prescan_ranges.json  記錄 prescan 判定的人物有效 frame ranges
```

---

## 8. 常見錯誤

### 8.1 找不到 TensorRT runtime library

錯誤：

```text
libnvonnxparser.so.11: cannot open shared object file
```

原因：

```text
LD_LIBRARY_PATH 沒有包含 TensorRT runtime library 目錄。
```

修正：

```bash
export LD_LIBRARY_PATH=<TRT_LIB_DIR>:${LD_LIBRARY_PATH:-}
```

然後重新啟動後端。

### 8.2 engine 找不到

log：

```text
[prescan] engine not found, skip: ...
```

原因：

```text
prescan_engine_path 指到不存在的檔案，或 models/yolo26x_ultralytics_int8.engine 沒有被下載/放好。
```

修正：

```bash
ls -lh <REPO_ROOT>/models/yolo26x_ultralytics_int8.engine
```

確認檔案存在後重新執行分析。

### 8.3 直接 import tensorrt 發生 crash 或 Bus error

原因通常是：

```text
Python binding 載到了，但底層 TensorRT runtime library 沒有正確對應。
```

處理方式：

```text
不要只看 pip package 是否存在，要用正確 LD_LIBRARY_PATH 啟動 Python / backend。
```

---

## 9. Git 注意事項

`models/yolo26x_ultralytics_int8.engine` 約 82MB，已低於 GitHub 100MB 單檔限制，但高於 GitHub 建議的 50MB。

目前 `.gitignore` 保留：

```gitignore
*.engine
```

並只允許這一份 engine 被追蹤：

```gitignore
!models/yolo26x_ultralytics_int8.engine
```

因此：

```text
其他 .engine 檔仍會被忽略
只有 models/yolo26x_ultralytics_int8.engine 會進 Git
```

如果未來 engine 超過 100MB，應改用 Git LFS。
