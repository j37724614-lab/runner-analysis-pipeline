# 前後端連接說明

## 服務端口總覽

| 服務 | Port | URL |
|------|------|-----|
| 後端 API | 18000 | `http://catslab.ee.ncku.edu.tw:18000` |
| 前端 App | 18001 | `http://catslab.ee.ncku.edu.tw:18001/running_analysis` |
| 後端 API 文件 | 18000 | `http://catslab.ee.ncku.edu.tw:18000/running_analysis/api/docs` |

---

## 後端啟動

**目錄**：`/home/jeter/running-analysis-backend/running-analysis-backend/`

**指令**：

```bash
cd /home/jeter/running-analysis-backend/running-analysis-backend
conda run -n yolo_new uvicorn main:app --host 0.0.0.0 --port 18000
```

- Conda 環境：`yolo_new`
- API prefix：`/running_analysis/api`
- 資料庫：`running.db`（SQLite，同目錄下）
- Log 輸出：`/home/jeter/running-analysis-backend/backend_18000.log`

---

## 前端啟動

**目錄**：`/home/jeter/running-analysis-frontend/`

**指令**：

```bash
cd /home/jeter/running-analysis-frontend
/home/jeter/flutter/bin/flutter run -d web-server \
  --web-hostname=0.0.0.0 \
  --web-port=18001 \
  --release \
  --dart-define=API_BASE_URL=http://catslab.ee.ncku.edu.tw:18000/running_analysis/api
```

- Flutter 版本：3.44.0（`/home/jeter/flutter/bin/flutter`）
- Flutter App 路徑：`/running_analysis`（需要完整輸入）
- `--dart-define=API_BASE_URL` **必須指定**，否則前端會連到正式後端（`https://catslab.ee.ncku.edu.tw/running_analysis/api`），看不到本地資料

---

## API_BASE_URL 說明

前端 `lib/utils/api.dart` 的 base URL 由環境變數決定：

```dart
static const baseUrl = String.fromEnvironment(
  "API_BASE_URL",
  defaultValue: "https://catslab.ee.ncku.edu.tw/running_analysis/api",
);
```

| 情境 | API_BASE_URL |
|------|-------------|
| 本地開發（連本地後端） | `http://catslab.ee.ncku.edu.tw:18000/running_analysis/api` |
| 正式環境（預設值） | `https://catslab.ee.ncku.edu.tw/running_analysis/api` |

若未指定 `--dart-define`，前端連的是正式後端，本地資料庫（`running.db`）的資料**不會出現**在前端。

---

## API 端點一覽

所有端點前綴為 `/running_analysis/api`。

### 跑者（Runner）

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/runner` | 取得所有跑者列表 |
| POST | `/runner` | 新增跑者 |
| GET | `/runner/{runner_id}/run_sessions` | 取得該跑者的所有分析紀錄 |
| GET | `/runner/{runner_id}/run_sessions/unanalyzed` | 取得該跑者尚未分析的紀錄 |

### 分析結果（Run Session）

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/run_session/{run_session_id}` | 取得 session 詳細資訊 |
| GET | `/run_session/{run_session_id}/graphs` | 取得速度/加速度/角度圖表資料 |
| GET | `/run_session/{run_session_id}/angles` | 取得逐幀時間對應角度表格資料 |
| GET | `/run_session/{run_session_id}/video` | 取得分析輸出影片 |

### 角度時間資料

Pipeline 會在 `angles.csv` 加上：

| 欄位 | 說明 |
|------|------|
| `frame` | 最終輸出影片的 frame index，從 0 開始。 |
| `time_sec` | 使用 `output_final.mp4` 實際 FPS 換算出的秒數。 |
| `time_s` | 與 `time_sec` 相同，保留給既有圖表相容。 |

`GET /run_session/{run_session_id}/angles` 回傳格式：

```json
{
  "columns": ["left_knee_angle", "right_knee_angle"],
  "samples": [
    {
      "frame": 0,
      "timeSec": 0.0,
      "values": {
        "left_knee_angle": 106.2,
        "right_knee_angle": 114.1
      }
    }
  ]
}
```

Flutter playback 頁面會在圖表下方顯示 `Angle Data` 可展開表格，欄位包含時間、frame 與各關節角度。

### 上傳影片（Temp Video）

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/temp_video/{temp_video_id}/thumbnail` | 取得暫存影片縮圖 |
| POST | `/temp_video/{index}` | 上傳單支影片（依相機編號） |
| POST | `/upload_all_info` | 一次性上傳全部影片與設定 |
| POST | `/upload_seperately_new` | 分批上傳（建立新 session） |
| POST | `/upload_seperately_select` | 分批上傳（接回既有 session） |

### WebSocket

| 路徑 | 說明 |
|------|------|
| `/ws` | 多相機錄影同步與即時狀態推送 |
