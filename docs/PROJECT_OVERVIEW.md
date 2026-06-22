# Running Analysis Frontend 專案完整說明

本文是 `/home/jeter/running-analysis-frontend` 的專案導讀。這個 repository 是一個 Flutter 前端，產品名稱在 App 內顯示為「百米分析」。它不是 Flutter 預設範例，而是一套面向短跑或百米跑步分析的前端系統，用來把跑者、多相機錄影、影片上傳、分析狀態追蹤、結果回放與圖表呈現串在一起。

前端目前直接連到正式後端：

```text
https://catslab.ee.ncku.edu.tw/running_analysis/api
```

整體可以理解成三個主要工作流：

1. `上傳`: 使用者選擇跑者、上傳 1 到 5 台相機拍到的影片，設定日期、FPS、備註與相機錨點，送後端分析。
2. `錄影`: 多台裝置加入同一個 WebSocket 房間，由主控端同步控制開始與停止錄影，錄完自動上傳。
3. `回放`: 使用者選擇跑者與 Run Session，觀看分析後影片、基本數據、歷史紀錄與同步圖表。

## 技術摘要

專案基礎：

- Flutter App，支援 Web、Android、iOS、macOS、Linux、Windows 的平台目錄。
- Dart SDK constraint 是 `^3.8.1`。
- 主要程式碼集中在 `lib/`。
- `android/`、`ios/`、`macos/`、`linux/`、`windows/` 大多是 Flutter 產生的平台殼與 plugin glue code。
- `web/` 放 Web 入口、manifest、icons 與 splash 檔案。
- `assets/` 放 `icon.png` 與 Lottie splash 動畫 `splash.json`。

主要套件：

- `flutter_riverpod`: 狀態管理。
- `go_router`: 路由。
- `dio`: REST API 呼叫。
- `web_socket_channel`: 同步錄影房間的 WebSocket。
- `camera`: 裝置相機與錄影。
- `video_player`、`video_player_web`: 影片播放。
- `file_picker`: 選取本機影片檔。
- `fl_chart`: 分析圖表。
- `flutter_staggered_grid_view`: 回放頁的 masonry grid layout。
- `shimmer`、`flutter_spinkit`: loading UI。
- `toastification`: toast 錯誤與成功提示。
- `dropdown_button2`、`custom_sliding_segmented_control`、`sidebarx`、`convex_bottom_bar`: UI 控制元件。
- `intl`: 日期格式化。
- `mime`、`http_parser`: 判斷上傳影片 mime type 與 multipart content type。
- `lottie`: splash 動畫。
- `image`: 目前主要出現在註解掉的預覽壓縮邏輯中。

## App 啟動與路由

入口檔案是 `lib/main.dart`。

啟動流程：

1. 呼叫 `usePathUrlStrategy()`，讓 Flutter Web 使用 path URL，而不是 `/#/` hash URL。
2. 用 `ProviderScope` 包住整個 App，啟用 Riverpod。
3. `MyApp` 讀取 `goRouterProvider`。
4. 用 `ToastificationWrapper` 包住 `MaterialApp.router`，讓整個 App 都能顯示 toast。
5. 設定 App title 為 `百米分析`。
6. 設定主要色系為淺藍色，並關閉多數 button hover/splash overlay。

路由定義在 `lib/utils/router.dart`。

主要路由：

- `/splash`: 啟動畫面。非 Web 平台的 initial route。
- `/playback`: 回放頁。Web 平台的 initial route。
- `/upload`: 上傳頁。
- `/record`: 同步錄影頁。

`/playback` 支援 query parameters：

```text
/playback?runnerId=<runnerId>&videoId=<runSessionId>
```

這讓上傳完成後可以直接導到剛建立或剛補齊的 Run Session。

路由層級：

- `/splash` 是獨立頁面。
- `/playback`、`/upload`、`/record` 放在 `ShellRoute` 裡。
- `ShellRoute` 的 builder 是 `HomePage(child: child)`，所以這三頁共用同一個外框。

`HomePage` 在 `lib/feature/home_page.dart`，負責主導覽：

- 橫向寬螢幕時使用 `SidebarX` 側邊欄。
- 直向或窄螢幕時使用 `ConvexAppBar` bottom navigation。
- 三個 tab 對應 `回放`、`上傳`、`錄影`。
- 會根據目前 URL 同步選中的 tab。

## Splash

`lib/feature/splash/splash_page.dart` 是啟動畫面。

行為：

- 背景色是 `#A5C3ED`。
- 中央播放 `assets/splash.json` Lottie 動畫。
- 2 秒後自動導到 `/playback`。
- Web 不從 splash 開始，Web initial location 是 `/playback`。

## 後端與 API 層

### API endpoint 定義

`lib/utils/api.dart` 定義所有 REST endpoint。

base URL：

```text
https://catslab.ee.ncku.edu.tw/running_analysis/api
```

endpoint 對應：

| 前端名稱 | HTTP | URL | 用途 |
|---|---:|---|---|
| `getRunner` | GET | `/runner` | 取得跑者清單 |
| `addRunner` | POST | `/runner` | 新增跑者 |
| `getRunnerHistory(runnerId)` | GET | `/runner/{runnerId}/run_sessions` | 取得跑者所有 Run Session |
| `getRunnerUnanalyzedHistory(runnerId)` | GET | `/runner/{runnerId}/run_sessions/unanalyzed` | 取得尚未完整或尚未分析的紀錄 |
| `getRunSessionInfo(runSessionId)` | GET | `/run_session/{runSessionId}` | 取得單筆 Run Session metadata |
| `getGraphData(runSessionId)` | GET | `/run_session/{runSessionId}/graphs` | 取得分析圖表資料 |
| `getRunSessionVideo(runSessionId)` | GET | `/run_session/{runSessionId}/video` | 取得可播放影片 URL |
| `getTempVideoThumbnail(tempVideoId)` | GET | `/temp_video/{tempVideoId}/thumbnail` | 取得暫存影片縮圖 |
| `uploadVideo(index)` | POST | `/temp_video/{index}` | 上傳單支相機影片到暫存區 |
| `uploadAllInfo` | POST | `/upload_all_info` | 一次提交完整多相機資料 |
| `uploadSeperatelyNew` | POST | `/upload_seperately_new` | 分別上傳時建立新紀錄 |
| `uploadSeperatelySelect` | POST | `/upload_seperately_select` | 分別上傳時補齊既有紀錄 |

注意：`seperately` 是程式中的既有拼法，英文正確拼法是 `separately`。目前前端類別、方法與 API 名稱都採用既有拼法，若要修正需要同步改引用與後端契約。

### NetUtils

`lib/utils/net_utils.dart` 包裝 Dio。

功能：

- 建立 singleton `NetUtils`。
- 設定 `connectTimeout` 與 `receiveTimeout` 都是 180 秒，符合影片處理或大檔上傳可能耗時較久的情境。
- `reqeustData<T>()` 依 `DioMethod` 呼叫 GET、POST、PATCH、PUT、DELETE。
- 若傳入 token，會放入 `Authorization: Token <token>`。
- 每次 request 都會加上 `ngrok-skip-browser-warning` header。
- 錯誤時會把 Dio timeout 轉成人類可讀訊息。
- `requestStream()` 可讀取 stream response，會解析 `data:` 開頭的 server-sent event-like 文字資料，但目前主要業務流程沒有明顯使用它。

注意：`reqeustData` 方法名稱拼錯，但全專案都照這個名字呼叫，因此目前不影響執行。

### BackendInterface

`lib/backend/backend_interface.dart` 定義前端需要的後端能力，讓正式 REST 實作與假資料實作共用同一套介面。

方法：

- `getRunners()`: 取得跑者清單。
- `getGraphData(runSessionId)`: 取得圖表資料。
- `getRunSessionInfo(runSessionId)`: 取得單筆 Run Session 資訊。
- `getRunnerHistory(runnerId)`: 取得跑者歷史紀錄。
- `getRunnerUnanalyzedHistory(runnerId)`: 取得尚未完整或尚未分析的紀錄。
- `addRunner(name)`: 新增跑者並回傳 runner id。
- `uploadAllInfo(...)`: 一次提交完整多相機資料並回傳 run session id。
- `uploadSeperatelyNew(...)`: 分別上傳時建立新紀錄。
- `uploadSeperatelySelect(...)`: 分別上傳時補齊既有紀錄。
- `uploadVideo(index, file)`: 上傳單支影片並回傳 `tempVideoId`。

### RestBackendRepo

`lib/backend/rest_backend_repo.dart` 是正式後端實作。

重要行為：

- `getRunners()` 把 API 回傳 list map 成 `RunnerInfo`。
- `getGraphData()` 把 API 回傳 list map 成 `GraphData`。
- `getRunSessionInfo()` map 成 `RunSessionInfo`。
- `getRunnerHistory()` map 成 `RunSessionInfo` list。
- `getRunnerUnanalyzedHistory()` map 成 `UnanalyzedRunSessionInfo` list。
- `uploadVideo()` 用 `FormData` 與 `MultipartFile.fromBytes` 上傳影片 bytes，content type 由 `UploadVideoFile.mimeType` 解析。
- `uploadAllInfo()` 送出 runner、日期、相機數量、fps、note、videos 陣列。
- `uploadSeperatelyNew()` 送出新 Run Session 所需資訊與單支 temp video。
- `uploadSeperatelySelect()` 送出既有 Run Session id 與補傳影片。
- 日期格式統一用 `yyyy-MM-dd HH:mm:ss`。

`uploadAllInfo` 的 videos 結構大致是：

```dart
[
  {
    "tempVideoId": "...",
    "anchors": {
      "points": [
        {"x": 0.1, "y": 0.2},
        {"x": 0.9, "y": 0.2},
        {"x": 0.9, "y": 0.8},
        {"x": 0.1, "y": 0.8}
      ],
      "topDistanceM": 1.22,
      "bottomDistanceM": 1.22
    }
  }
]
```

### FakeBackendRepo

`lib/backend/fake_backend_repo.dart` 是本地假資料實作。

啟用方式在 `lib/utils/config.dart`：

```dart
const kUseFakeRepos = false;
```

改成 `true` 後，`backendProvider` 會用 `FakeBackendRepo()`。假資料來自 `lib/utils/test_data.dart`。

用途：

- 不接正式 API 時測 UI。
- 模擬跑者清單、影片歷史、圖表資料與上傳結果。

注意：

- 某些 fake method delay 設成 100 秒，像 `getGraphData()`、`getRunSessionInfo()`、`getRunnerHistory()`，這會讓 UI 長時間停在 loading 狀態。
- fake repo 比較像開發過程的暫存工具，不是完整 mock server。

### backendProvider 與資料 Providers

`lib/backend/backend_provider.dart` 是資料讀取的核心 Riverpod provider。

Provider 列表：

- `backendProvider`: 依 `kUseFakeRepos` 回傳 `FakeBackendRepo` 或 `RestBackendRepo`。
- `runnerProvider`: 取得跑者清單。
- `graphDataProvider(runSessionId)`: 取得圖表資料。
- `videoInfoProvider(runSessionId)`: 取得單筆 Run Session 資訊。如果 status 是 `processing`，5 秒後 `invalidateSelf()` 重新拉資料。
- `runnerHistoryProvider(runnerId)`: 取得跑者歷史紀錄。如果其中任何紀錄 status 是 `processing`，5 秒後自動刷新。
- `runnerUnanalyzedHistoryProvider(runnerId)`: 取得未分析或未完整紀錄。

這套設計讓 UI 不需要自己寫 polling，直接 watch provider 即可。

## 資料模型 Entities

所有 API 資料模型在 `lib/entities/`。

### RunnerInfo

檔案：`lib/entities/runner_info.dart`

欄位：

- `name`: 跑者姓名。
- `id`: runner id。
- `lastVideoId`: 該跑者最近一筆影片/Run Session id，沒有則是空字串。

用途：

- 回放頁選跑者後，自動切到該跑者最後一筆紀錄。
- 上傳頁與錄影頁選擇跑者。

### RunSessionInfo

檔案：`lib/entities/run_session_info.dart`

欄位：

- `runSessionId`: 一次跑步紀錄 id。
- `runnerId`: 跑者 id。
- `runnerName`: 跑者姓名。
- `date`: 拍攝或紀錄時間。
- `cameraCount`: 相機數量。
- `fps`: 影片 fps。
- `avgVelocity`: 平均速度。
- `avgAcceleration`: 平均加速度。
- `avgStepLength`: 平均步幅。
- `totalTime`: 總時間。
- `note`: 備註。
- `status`: 後端分析狀態，例如 `processing`、`done`、`failed`。
- `progress`: 分析進度百分比。

用途：

- 回放頁影片資訊表。
- 歷史紀錄表。
- 判斷是否顯示 processing 進度。

### UnanalyzedRunSessionInfo

檔案：`lib/entities/unanalyzed_run_session_info.dart`

欄位：

- `runSessionId`
- `runnerId`
- `runnerName`
- `date`
- `cameraCount`
- `fps`
- `note`
- `unuploadedCameraIndexes`: 尚未上傳的相機 index。
- `videoPaths`: 已上傳或既有影片路徑。

用途：

- 分別上傳模式下，讓使用者選擇還沒補齊的紀錄。
- 顯示目前缺哪些相機影片。

### GraphData

檔案：`lib/entities/graph_data.dart`

欄位：

- `title`: 圖表標題，例如 Distance、Velocity、Acceleration。
- `yLabel`: y 軸標籤。
- `yMin`: y 軸最小值。
- `yMax`: y 軸最大值。
- `x`: x 軸資料點，一般代表時間。
- `y`: y 軸資料點，與 x 一一對應。

用途：

- `GraphListView` 用它建立 `FlSpot` list，畫出分析曲線。

### UploadVideoFile

檔案：`lib/entities/upload_video_file.dart`

欄位：

- `bytes`: 檔案 bytes。
- `filename`: 檔名。
- `mimeType`: MIME type。

用途：

- `file_picker` 選到影片後，轉成這個物件交給 `backend.uploadVideo()`。
- 錄影完成後，`XFile.readAsBytes()` 也會轉成這個物件上傳。

### UploadSeperatelyStatus

檔案：`lib/entities/upload_seperately_status.dart`

欄位：

- `runnerId`
- `runSessionId`
- `isAllUploaded`: 是否已補齊所有相機影片。
- `unuploadedCameraIndexes`: 尚缺哪些相機。

用途：

- 分別上傳後判斷要導到回放頁，或提示繼續補傳。

### VideoPlayback

檔案：`lib/entities/video_playback.dart`

欄位：

- `position`: 影片目前位置，毫秒。
- `duration`: 影片總長，毫秒。
- `currentFrame`: 根據 detection 計算出的目前 frame，目前欄位有保留但核心圖表同步是用 position/duration。
- `isDragging`: 使用者是否正在拖拉 slider。

用途：

- 影片播放與圖表同步。
- 拖拉 slider 時暫停自動更新 position，避免 UI 跳動。

## 回放功能詳解

核心目錄：`lib/feature/playback/`

### PlaybackPage

檔案：`lib/feature/playback/playback_page.dart`

職責：

- 顯示跑者 dropdown。
- 保存或讀取目前選中的 runner id 與 run session id。
- 如果 URL 帶 `runnerId` 與 `videoId`，初始化時寫入 playback provider。
- 根據畫面寬度決定 masonry grid 欄數，小於 800px 用 1 欄，大於等於 800px 用 2 欄。
- 組合四個主要內容：影片播放器、圖表、影片資訊、歷史紀錄。

重要 provider：

- `playbackSelectedRunnerIdProvider`
- `playbackSelectedRunSessionIdProvider`
- `runnerProvider`

選跑者邏輯：

1. `runnerProvider` 載入所有跑者。
2. dropdown 選到某個 runner。
3. `playbackSelectedRunnerIdProvider` 更新。
4. `playbackSelectedRunSessionIdProvider` 設成該跑者的 `lastVideoId`。
5. 如果 `lastVideoId` 是空字串，顯示「此跑者尚無歷史紀錄，請先上傳」。

### VideoPlayerView

檔案：

- `lib/feature/playback/widget/video_player_view.dart`
- `lib/feature/playback/widget/video_player_controller.dart`
- `lib/feature/playback/widget/video_slider_item.dart`

流程：

1. 從 `playbackSelectedRunSessionIdProvider` 取得目前影片 id。
2. 如果沒有選影片，顯示影片 placeholder。
3. 用 `videoInfoProvider(videoId)` 取得狀態。
4. 如果 status 是 `processing`，顯示影片 shimmer 與分析進度。
5. 如果已可播放，建立 `_VideoContentPlayer`。
6. `_VideoContentPlayer` 透過 `videoManagerProvider(videoId)` 初始化 `VideoPlayerController.networkUrl()`。
7. controller 初始化後 seek 到 0。
8. 點影片可切換 play/pause。
9. controller listener 每次 tick 更新 `videoPlaybackStateProvider` 的 position 與 duration。
10. `VideoSliderView` 讀取 playback state，拖動時呼叫 `seekTo()`。

影片 URL：

```dart
API.getRunSessionVideo(id)[1]
```

也就是：

```text
GET /run_session/{id}/video
```

目前 `VideoPlayerController.networkUrl()` 直接使用這個 endpoint 當 URL，而不是先透過 Dio 取真正影片位置。

### GraphListView

檔案：`lib/feature/playback/widget/graph_list_view.dart`

職責：

- 顯示多張分析圖表。
- 在後端 processing 時顯示進度。
- 已完成時讀 `graphDataProvider(videoId)`。
- 使用 `fl_chart` 的 `LineChart`。
- 根據影片播放進度，把已播放區段畫成紅線，未播放區段畫成半透明白線。

同步邏輯：

```dart
final progress = videoPlayback.position / videoPlayback.duration;
final currentIndex = (progress * (spots.length - 1)).floor();
final playedSpots = spots.sublist(0, currentIndex + 1);
```

視覺設定：

- 圖表標題是白色粗體。
- bottom axis 是 `Time`。
- left axis 使用 `GraphData.yLabel`。
- y 軸範圍使用 `GraphData.yMin` 與 `GraphData.yMax`。
- 每秒畫一條 dashed vertical line。
- tooltip 會顯示點的 y 值。

### VideoInfoView

檔案：`lib/feature/playback/widget/video_info_view.dart`

顯示資訊：

- 選手姓名
- 日期時間
- 相機數量
- fps
- 平均速度
- 平均加速度
- 平均步幅
- 總時間
- 備註

如果 status 是 `processing`，顯示 `VideoInfoShimmer` 與 `ProcessingProgressWidget`。

### RunnerHistoryView

檔案：`lib/feature/playback/widget/runner_history_view.dart`

職責：

- 顯示目前跑者的歷史 Run Session。
- 欄位：日期時間、相機數量、總時間、備註。
- 點某一列可切換目前播放的 Run Session。
- 目前選中的列會用 `primaryColorDark` 高亮。
- 如果點到 status 是 `failed` 的紀錄，會顯示錯誤 toast，不切換。
- 如果 status 是 `processing`，總時間欄顯示 `分析中...`。

### Placeholder 與 Shimmer

placeholder 目錄：

- `graph_list_placeholder.dart`
- `one_graph_placeholder_item.dart`
- `runner_history_placeholder.dart`
- `video_info_placeholder.dart`

shimmer 目錄：

- `graph_list_shimmer.dart`
- `runner_history_shimmer.dart`
- `video_info_shimmer.dart`
- `video_player_shimmer.dart`

用途：

- 沒選資料時顯示 placeholder。
- API loading 時顯示 shimmer。
- processing 時 shimmer 疊上進度框。

## 上傳功能詳解

核心目錄：`lib/feature/upload/`

上傳功能分兩層：

- `UploadPage`: 選跑者、新增跑者、切換上傳模式。
- `UploadAllView` / `UploadSeperatelyView`: 實際選檔、設定錨點、送出。

### UploadPage

檔案：`lib/feature/upload/upload_page.dart`

主要 UI：

- `選擇選手` / `新增選手` segmented control。
- 選擇既有跑者的 dropdown。
- 新增跑者的文字欄與新增按鈕。
- `一起上傳` / `分別上傳` segmented control。
- 依上傳模式顯示 `UploadAllView` 或 `UploadSeperatelyView`。
- 全頁 loading overlay 由 `uploadControllerProvider` 的 `AsyncLoading` 控制。

上傳頁相關 provider：

- `uploadSelectedRunnerIdProvider`
- `uploadSelectedRunSessionIdProvider`
- `uploadTypeProvider`
- `runnerSourceProvider`
- `runnerNameInputProvider`
- `uploadRunnerListProvider`
- `uploadControllerProvider`

新增跑者流程：

1. 使用者切到 `新增選手`。
2. 輸入名稱。
3. 點新增。
4. 呼叫 `uploadRunnerListProvider.notifier.addRunner(name)`。
5. 後端回傳新 runner id。
6. 前端把新跑者加入現有清單。
7. 自動切回 `選擇選手` 並選中新 runner。

### UploadController

檔案：`lib/feature/upload/upload_controller.dart`

它是一個 `StateNotifier<AsyncValue<void>>`，負責送出最終上傳資訊，不負責單支影片暫存上傳。

方法：

- `uploadAllInfo(...)`: 組合日期與時間後呼叫 backend.uploadAllInfo。
- `uploadSeperatelyNew(...)`: 建立新的分別上傳紀錄。
- `uploadSeperatelySelect(...)`: 補傳既有分別上傳紀錄。

錯誤處理：

- 開始時把 state 設成 `AsyncLoading()`。
- 成功後設回 `AsyncValue.data(null)`。
- 失敗時設成 `AsyncValue.error(...)`。
- `UploadPage` 用 `ref.listen` 監聽錯誤並顯示 toast。

### UploadAllView

檔案：

- `lib/feature/upload/widget/upload_all_view.dart`
- `lib/feature/upload/widget/upload_all_controller.dart`

用途：

- 一次上傳完整的多相機影片。
- 適合所有相機影片都已經準備好的情境。

流程：

1. 使用者先在 `UploadPage` 選跑者。
2. `UploadAllView` 顯示日期、時間、相機數量、FPS、備註設定。
3. 依相機數量顯示 1 到 5 個影片格。
4. 每個格子點擊後用 `FilePicker.platform.pickFiles(type: FileType.video, withData: true)` 選影片。
5. 選到影片後建立 `UploadVideoFile`。
6. 呼叫 `UploadAllController.uploadVideo(index, uploadFile)`。
7. controller 呼叫 backend.uploadVideo，拿到 `tempVideoId`。
8. 用 `API.getTempVideoThumbnail(tempVideoId)` 產生縮圖 URL。
9. 顯示縮圖。
10. 上傳成功後立即彈出錨點設定 dialog。
11. 每支影片都上傳完成後，點「上傳」。
12. 呼叫 `UploadController.uploadAllInfo(...)`。
13. 成功後 invalidate `runnerHistoryProvider(runnerId)`。
14. 導到 `/playback?runnerId=...&videoId=...`。

`UploadAllState` 欄位：

- `tempVideoStates`: 每台相機的暫存狀態。
- `cameraCount`: 相機數量，預設 5。
- `isUploadingAll`: 給 uploadAllInfo 用，但目前主要 loading 是由 `uploadControllerProvider` 控制。
- `error`: 錯誤訊息。

`UploadThumbnailState` 欄位：

- `thumbnailUrl`: 縮圖 URL。
- `tempVideoId`: 後端暫存影片 id。
- `isUploading`: 這支影片是否正在上傳。
- `error`: 單支影片錯誤。
- `anchorResult`: 這支影片的錨點設定。

送出前檢查：

- 沒選跑者：alert `請選擇跑者`。
- 任一影片沒有縮圖：alert `請上傳所有視頻`。

### UploadSeperatelyView

檔案：

- `lib/feature/upload/widget/upload_seperately_view.dart`
- `lib/feature/upload/widget/upload_seperately_controller.dart`
- `lib/feature/upload/widget/unanalyzed_history_view.dart`

用途：

- 分批上傳多相機影片。
- 可以先上傳某台相機，之後再補其他相機。
- 適合影片不在同一台電腦，或各台相機影片要分開處理的情境。

分別上傳有兩種子模式：

- `新增紀錄`: 建立新的 Run Session，但可能只先上傳其中一台相機。
- `選擇紀錄`: 從未完整紀錄清單中選一筆，補上缺少的相機影片。

新增紀錄流程：

1. 選跑者。
2. 切到 `分別上傳`。
3. 子模式選 `新增紀錄`。
4. 設定日期、時間、相機數量、FPS、備註。
5. 選擇第幾個相機。
6. 選影片並暫存上傳。
7. 設定錨點。
8. 點「上傳」。
9. 呼叫 `uploadSeperatelyNew(...)`。
10. 後端回傳 `UploadSeperatelyStatus`。
11. 如果 `isAllUploaded == true`，導到回放頁。
12. 如果還沒補齊，提示尚缺的相機，例如 `相機2, 相機4`。

選擇紀錄補傳流程：

1. 選跑者。
2. 切到 `分別上傳`。
3. 子模式選 `選擇紀錄`。
4. 前端用 `runnerUnanalyzedHistoryProvider(runnerId)` 取得未完整紀錄。
5. `UnanalyzedHistoryView` 顯示清單。
6. 點某一筆紀錄，設定 `uploadSelectedRunSessionIdProvider`。
7. 讀取該紀錄的 `unuploadedCameraIndexes`，只允許選尚缺的相機。
8. 選影片、設定錨點、點上傳。
9. 呼叫 `uploadSeperatelySelect(...)`。
10. 如果補齊所有相機，導到回放頁；否則提示繼續補傳。

`UploadSeperatelyState` 欄位：

- `thumbnail`: 本次選到影片的縮圖。
- `tempVideoId`: 暫存影片 id。
- `isUploading`: 是否正在上傳單支影片。
- `error`: 錯誤訊息。
- `anchorResult`: 本支影片錨點。

### DateTimeSelectionWidget

檔案：`lib/feature/upload/widget/date_time_selection_widget.dart`

共用於一起上傳與分別上傳。

可設定：

- 日期：`showDatePicker`，範圍 2000 到 2100。
- 時間：`showTimePicker`。
- 相機數量：1 到 5。
- FPS：30 或 60。
- 備註：文字輸入。

注意：

- 備註欄目前只有在 `value.isNotEmpty` 時才呼叫 `onNoteSelected(value)`，所以輸入後再清空不會把 note 設回空字串。

### 錨點設定 Dialog

檔案：`lib/feature/upload/widget/anchor_point_dialog.dart`

用途：

- 針對影片縮圖設定四個校正點。
- 輸入上邊與下邊實際距離。
- 回傳 `AnchorResult` 給上傳流程。

資料結構：

```dart
class AnchorPoint {
  final double x;
  final double y;
}

class AnchorResult {
  final List<AnchorPoint> points;
  final double topDistanceM;
  final double bottomDistanceM;
}
```

座標系統：

- 點位是 normalized 座標。
- `x` 與 `y` 都在 `0.0` 到 `1.0`。
- 因此後端可依實際影片寬高還原像素位置。

操作邏輯：

- 使用者依序點四個點：左上、右上、右下、左下。
- 點位可以拖曳微調。
- 拖曳時會顯示放大鏡。
- 可以復原、重設、取消。
- 四點都設定完，且上下邊距離都是大於 0 的數字，才可確認。

## 同步錄影功能詳解

核心目錄：`lib/feature/record/`

這一塊是專案中比較複雜的功能。它讓多台裝置進入同一個「錄影房間」，由主控裝置統一下達開始與停止錄影。錄影完成後，各裝置自動把自己的影片上傳到後端並形成同一筆 Run Session。

### RecordController

檔案：`lib/feature/record/record_controller.dart`

`RecordController` 是 `StateNotifier<RecordState>`。

核心責任：

- 建立 WebSocket 連線。
- 建立房間。
- 加入房間。
- 接收房間狀態。
- 發送 ready 狀態。
- 發送開始錄影與停止錄影指令。
- 同步 upload complete 後的 `runSessionId`。
- 保存主控端設定的 runner、fps、note。

WebSocket URL：

```dart
API.baseUrl
  .replaceFirst('https://', 'wss://')
  .replaceFirst('http://', 'ws://') + '/ws'
```

目前實際會是：

```text
wss://catslab.ee.ncku.edu.tw/running_analysis/api/ws
```

### RecordState

檔案：`lib/feature/record/record_state.dart`

主要欄位：

- `role`: `master`、`slave`、`none`。
- `status`: `idle`、`connecting`、`ready`、`recording`、`uploading`、`finished`。
- `roomId`: 房間號碼。
- `members`: 已連線成員清單。
- `error`: 錯誤訊息。
- `myDeviceInfoIndex`: 註解表示 slave 用，但目前核心流程多用 `myCameraIndex`。
- `myCameraIndex`: 本裝置負責的相機 index。
- `sharedRunSessionId`: 第一台成功建立 Run Session 後共享給其他裝置的 id。
- `expectedCameraCount`: 預期相機數。
- `isRecordingEnabled`: Master 本機是否也參與錄影。
- `isPhysicallyReady`: 本機是否已橫放。
- `runnerSource`: 選擇既有跑者或新增跑者。
- `runnerId`
- `runnerName`
- `fps`
- `note`
- `anchorResult`: 本裝置相機錨點。

derived getter：

- `anchorIsSet`: 是否已有錨點。

### RecordMessage 與 WebSocket 協議

檔案：`lib/feature/record/record_enums.dart`

訊息格式：

```json
{
  "type": "messageTypeName",
  "data": {}
}
```

支援訊息型別：

- `createRoom`: Master 建立房間。
- `joinRoom`: Slave 加入房間，或 Master 切換本機是否作為相機。
- `roomStatus`: 後端廣播房間狀態。
- `startRecording`: Master 發開始錄影，後端轉發給房間成員。
- `stopRecording`: Master 發停止錄影。
- `uploadComplete`: 某台裝置通知 Run Session 建立完成。
- `updateReady`: 裝置回報是否 ready。
- `cameraPreview`: Slave 傳相機預覽圖；目前相關 timer 邏輯被註解掉。
- `error`: 後端回報錯誤。

`roomStatus` data 目前前端預期：

```json
{
  "roomId": "...",
  "members": [
    {
      "id": "...",
      "cameraIndex": 0,
      "isReady": true
    }
  ],
  "expectedCameraCount": 5
}
```

### RecordPage

檔案：`lib/feature/record/record_page.dart`

畫面分成兩種狀態：

1. 初始畫面：尚未進房或正在連線。
2. 房間畫面：已建立或加入房間。

初始畫面提供兩個入口：

- 主控裝置：選預計連線裝置數，點「建立錄影房間」。
- 錄影手機：輸入房間號碼、選相機位置，點「加入錄影房間」。

房間畫面會顯示：

- 房間號碼。
- 目前身份：Master 或 Slave。
- Master 的錄影參數設定。
- Master 是否本機也參與錄影。
- Slave 可更改相機位置。
- 已連線設備清單。
- 開始同步錄影或停止錄影按鈕。
- 參與錄影時的 `RecordCameraView`。
- 離開房間按鈕。

Master 開始錄影前的檢查：

- `runnerId` 不能是 null。
- 所有預期相機 index 都要出現在連線成員中。
- 所有參與錄影的 member 都要 `isReady == true`。

ready 狀態來源：

- 裝置方向不是 portrait，才算物理上 ready。
- 本機有分配相機編號時，還需要錨點已設定。
- `RecordController.updatePhysicallyReady()` 會根據 `MediaQuery.orientation` 更新。
- `RecordController.setAnchor()` 會在設定錨點後重新同步 ready 狀態。

### RecordCameraView

檔案：`lib/feature/record/widget/record_camera_view.dart`

這是同步錄影最核心的相機元件。

相機初始化：

- 使用 `availableCameras()` 取得所有相機。
- 優先挑後置鏡頭。
- 如果後置鏡頭名稱包含 `triple` 或 `dual`，優先選這種邏輯鏡頭，因為通常支援較廣縮放範圍。
- 使用 `CameraController`，`ResolutionPreset.veryHigh`，`fps: 60`，`enableAudio: true`。
- 讀取 min/max zoom，最多限制到 10x。

UI 功能：

- 相機預覽。
- 縮放 slider。
- 多後鏡頭時可更換鏡頭。
- 可進入全螢幕。
- portrait 時顯示「請橫放裝置錄製」遮罩。
- 非全螢幕時左上顯示錨點狀態 badge。
- 上傳中顯示自動上傳 overlay。

錄影監聽：

- 監聽 `recordControllerProvider.select((s) => s.status)`。
- 狀態變成 `recording` 時呼叫 `startVideoRecording()`。
- 狀態從 `recording` 變成 `uploading` 時呼叫 `stopVideoRecording()`。
- 錄完把 `XFile` 存到 `_recordedFile`。
- 接著呼叫 `_processUpload()`。

自動上傳邏輯：

1. 取得所有有 `cameraIndex` 的 members。
2. 找出最小 `cameraIndex`。
3. 最小 index 的裝置是 leader。
4. leader 先上傳自己的影片到 temp video。
5. leader 呼叫 `uploadSeperatelyNew()` 建立 Run Session。
6. leader 透過 WebSocket `uploadComplete` 廣播 `runSessionId`。
7. 非 leader 如果還沒有 `sharedRunSessionId`，先等待。
8. 非 leader 拿到 shared run session 後，呼叫 `uploadSeperatelySelect()` 補傳自己的影片。

檔名與 MIME 處理：

- 從 `XFile.name` 與 `lookupMimeType()` 判斷。
- 如果檔名沒有副檔名，會根據 MIME type 補副檔名。
- 無法判斷時 fallback 到 mp4 或 webm。

全螢幕錨點設定：

- 進入全螢幕時會重新初始化 camera，避免 Flutter Web 的 `HtmlElementView` video element detach 後串流斷掉。
- 全螢幕中可進入 anchor mode。
- anchor mode 會先拍一張 snapshot 供放大鏡使用。
- 設定四點與上下距離後，寫入 `recordControllerProvider.notifier.setAnchor(result)`。
- 錨點完成後會重新評估 ready。

## 錄影與上傳兩條路的關係

手動上傳與同步錄影最後都會進到類似的後端流程。

手動一起上傳：

```text
選跑者 -> 選多支影片 -> 每支影片 uploadVideo 得到 tempVideoId
-> 設錨點 -> uploadAllInfo -> 得到 runSessionId -> playback
```

手動分別上傳：

```text
選跑者 -> 選一支影片 -> uploadVideo 得到 tempVideoId
-> 設錨點 -> uploadSeperatelyNew 或 uploadSeperatelySelect
-> 若補齊則 playback，否則提示缺少相機
```

同步錄影：

```text
Master 建房 -> Slave 加房 -> 所有相機 ready
-> Master startRecording -> 各裝置本地錄影
-> Master stopRecording -> 各裝置停止並上傳 temp video
-> leader uploadSeperatelyNew 建 session
-> 其他裝置 uploadSeperatelySelect 補 session
-> 後端分析 -> playback 可看結果
```

所以同步錄影本質上是自動化的「分別上傳」。

## 分析進度與狀態顯示

分析進度由 `RunSessionInfo.status` 與 `RunSessionInfo.progress` 控制。

前端已知狀態：

- `processing`: 後端分析中，前端顯示 shimmer 與進度，並定期 polling。
- `done`: 可顯示影片、資訊與圖表。
- `failed`: 歷史紀錄中點擊會顯示錯誤 toast。

`ProcessingProgressWidget` 根據 progress 顯示文字：

| progress | 顯示文字 |
|---:|---|
| `< 5` | 準備中... |
| `>= 5` | 準備轉檔 |
| `>= 15` | 影片轉檔完成 |
| `>= 40` | 影片追蹤 (Tracking) 完成 |
| `>= 80` | 姿勢估計 (Pose Estimation) 完成 |
| `>= 90` | 資料後處理完成 |
| `>= 100` | 全部結束並存檔 |

## 共用 Widget

目錄：`lib/widget/`

- `AsyncValueWidget<T>`: 包裝 Riverpod `AsyncValue`，統一處理 data/loading/error。
- `AsyncValueUI`: extension，當 AsyncValue error 時用 toast 顯示錯誤。
- `LoadingIcon`: 中央 loading spinner，可帶文字。
- `LoadingOverlay`: 全頁 modal loading overlay。
- `ProcessingProgressWidget`: 分析中進度卡片。
- `RoundedBoxWidget`: 用 primary color 與圓角包住 child。

## 工具函式

目錄：`lib/utils/`

- `api.dart`: REST endpoint。
- `config.dart`: 是否使用 fake repo。
- `net_utils.dart`: Dio wrapper 與 stream request。
- `router.dart`: GoRouter 設定。
- `combine_date_and_time.dart`: 把 `DateTime` 日期與 `TimeOfDay` 時間合成完整 `DateTime`。
- `test_data.dart`: fake repo 用的 runner 與 run session 假資料。

## 目錄與檔案職責

完整核心程式碼職責如下：

```text
lib/main.dart
  App 入口、Theme、ToastificationWrapper、MaterialApp.router。

lib/backend/
  backend_interface.dart
    後端能力抽象介面。
  backend_provider.dart
    Riverpod provider，依 config 選 REST 或 fake，並提供資料 FutureProvider。
  rest_backend_repo.dart
    真實 REST API 實作。
  fake_backend_repo.dart
    假資料實作。
  video_playback_state_provider.dart
    影片播放位置、長度、拖曳狀態。

lib/entities/
  graph_data.dart
  run_session_info.dart
  runner_info.dart
  unanalyzed_run_session_info.dart
  upload_seperately_status.dart
  upload_video_file.dart
  video_playback.dart

lib/feature/home_page.dart
  共用 App shell，桌面 sidebar 與手機 bottom navigation。

lib/feature/splash/
  splash_page.dart
    Lottie 啟動畫面。

lib/feature/playback/
  playback_page.dart
    回放頁組合與跑者選擇。
  playback_provider.dart
    目前選擇的 runnerId 與 runSessionId。
  widget/
    video_player_view.dart
    video_player_controller.dart
    video_slider_item.dart
    graph_list_view.dart
    video_info_view.dart
    runner_history_view.dart
  placeholder/
    尚未選擇或預設展示用 placeholder。
  shimmer/
    loading skeleton UI。

lib/feature/upload/
  upload_page.dart
    上傳首頁、跑者選擇、新增跑者、模式切換。
  upload_controller.dart
    最終送出 uploadAllInfo / uploadSeperatelyNew / uploadSeperatelySelect。
  upload_provider.dart
    上傳頁選中的 runnerId / runSessionId。
  widget/
    upload_form_provider.dart
      上傳表單狀態、模式狀態。
    upload_enums.dart
      UploadType、RunnerSource、SperatedType。
    upload_all_view.dart
      一次完整上傳 UI。
    upload_all_controller.dart
      多相機暫存影片狀態。
    upload_seperately_view.dart
      分別上傳 UI。
    upload_seperately_controller.dart
      單支暫存影片狀態。
    unanalyzed_history_view.dart
      未分析或未完整紀錄清單。
    date_time_selection_widget.dart
      日期、時間、相機數量、FPS、備註共用表單。
    anchor_point_dialog.dart
      手動上傳時的錨點設定 dialog。

lib/feature/record/
  record_page.dart
    錄影房間 UI。
  record_controller.dart
    WebSocket、房間、錄影控制、上傳協調。
  record_state.dart
    錄影狀態資料。
  record_enums.dart
    角色、狀態、WebSocket message type、member model。
  widget/record_camera_view.dart
    相機初始化、預覽、錄影、上傳、全螢幕錨點。

lib/utils/
  API、router、network、config、fake test data 等。

lib/widget/
  共用 loading、async、progress、rounded box 元件。
```

## 平台目錄說明

這些目錄大多不是產品邏輯，但負責讓 Flutter 在各平台可 build：

```text
android/
  Gradle 設定、AndroidManifest、MainActivity、launcher icon、splash background。

ios/
  Xcode project、Podfile、Info.plist、AppDelegate、AppIcon、LaunchScreen。

macos/
  macOS Runner、Podfile、entitlements、AppIcon、MainFlutterWindow。

linux/
  Linux Runner、CMake、generated plugin registrant。

windows/
  Windows Runner、CMake、Win32 window、resource、manifest、icon。

web/
  index.html、manifest.json、favicon、web icons、splash.json。
```

## 資源檔

`assets/`：

- `assets/icon.png`: launcher icon 來源。
- `assets/splash.json`: Lottie splash 動畫。

`pubspec.yaml` 中有設定：

```yaml
flutter:
  uses-material-design: true
  assets:
    - assets/splash.json
    - assets/icon.png

flutter_launcher_icons:
  android: true
  ios: true
  image_path: "assets/icon.png"

flutter_native_splash:
  color: "#A5C3ED"
  android: true
  ios: true
  web: false
```

## 使用者視角的完整操作情境

### 情境 A：已經有多台相機影片，要一次上傳分析

1. 打開 App。
2. 進入「上傳」。
3. 選擇既有跑者，或新增跑者。
4. 選擇「一起上傳」。
5. 設定日期、時間、相機數量、FPS、備註。
6. 點每個相機格子選影片。
7. 每支影片上傳後設定四個錨點與實際距離。
8. 所有相機都有縮圖後，點「上傳」。
9. 後端建立 Run Session 並開始分析。
10. 前端導到回放頁。
11. 若分析中，看到 progress；完成後看到影片與圖表。

### 情境 B：只先上傳一部分相機影片

1. 進入「上傳」。
2. 選跑者。
3. 選擇「分別上傳」。
4. 選「新增紀錄」。
5. 設定日期、時間、相機數量、FPS、備註。
6. 選相機編號。
7. 選影片、設定錨點。
8. 點「上傳」。
9. 如果還缺其他相機，前端提示缺哪些相機。
10. 下次再進入「分別上傳」的「選擇紀錄」補齊。

### 情境 C：補上未完整紀錄

1. 進入「上傳」。
2. 選跑者。
3. 選「分別上傳」。
4. 選「選擇紀錄」。
5. 從未分析紀錄表選一筆。
6. 下拉選單只會顯示缺少的相機 index。
7. 選影片、設定錨點、上傳。
8. 若補齊，導到回放頁；否則繼續提示缺少相機。

### 情境 D：多裝置現場同步錄影

1. 主控端進入「錄影」。
2. 選預計連線裝置數。
3. 建立錄影房間。
4. 其他手機進入「錄影」，輸入房間號碼並選相機位置。
5. 主控端設定跑者、FPS、備註。
6. 如果主控端也要錄影，開啟本機參與錄影並選相機編號。
7. 每台參與錄影裝置橫放，進入相機畫面設定錨點。
8. 所有裝置 ready 後，主控端點「開始同步錄影」。
9. 所有裝置收到 WebSocket 指令並開始本地錄影。
10. 主控端點「停止錄影並上傳」。
11. 所有裝置停止錄影並上傳。
12. leader 建立 Run Session，其他裝置補傳。
13. 完成後後端分析，回放頁可看結果。

## 目前看到的技術債與風險

這些不是一定要立即修，但對維護者很重要。

### 文件與測試

- `README.md` 仍是 Flutter 預設模板，沒有描述實際產品。
- `PROJECT_OVERVIEW.md` 現在補了較完整導讀，但正式 README 還沒更新。
- `test/widget_test.dart` 仍是 counter smoke test，和目前 App 不符合。它期待畫面有 `0`、`1` 與 `Icons.add`，現有 App 沒有這些東西。
- 沒有看到針對上傳流程、provider、API mapping、錄影 WebSocket 狀態機的測試。

### 命名

- `reqeustData` 拼字錯。
- `seperately`、`SperatedType` 拼字錯或不一致。
- 因為這些名字可能已與後端 API 契約或大量引用綁定，重命名需要整體處理。

### 設定管理

- API base URL 寫死在 `lib/utils/api.dart`。
- 沒有 dev/staging/prod flavor 或 dart define 設定。
- `kUseFakeRepos` 也寫死在 config 檔，需要改程式碼才能切 mock。

### API 型別安全

- `API` endpoint 目前用 `List` 包 `[DioMethod, url]`，型別不夠明確。
- `NetUtils.reqeustData<T>()` 直接 `response.data as T`，如果後端格式改變會在 runtime 才爆。
- entity `fromJson` 沒有 defensive parsing，例如 number 有可能是 int 或 double 時，某些欄位直接指派給 `double?` 可能有型別風險。

### UI 狀態

- `DateTimeSelectionWidget` 的 note 清空不會更新成空字串。
- `UploadAllState.isUploadingAll` 目前看起來沒有完整被使用。
- `RecordCameraView.build()` 內呼叫 `_listenToRecordingStatus()`，每次 build 都會註冊 `ref.listen`。這在 Riverpod Consumer build 中可以運作，但要小心重複 listener 或生命週期問題；較穩定做法通常是在 `initState` 中註冊，或使用 `ref.listen` 時確認不會重複副作用。
- `RecordStatus.finished` enum 存在，但目前流程沒有明顯設定到 `finished`。

### 錄影與 Web 平台

- Flutter Web 的 camera preview 是 platform view，程式中特別為全螢幕 detach/reinitialize 做了處理，表示這塊曾遇過 Web 相容性問題。
- 錄影 MIME fallback 使用 `video/webm` 或 mp4，實際不同瀏覽器產出的格式可能不同。
- 同步錄影高度依賴後端 WebSocket 協議，目前 repo 中沒有正式協議文件。

### 安全與錯誤處理

- 沒看到登入、權限、token 管理；所有 API 看起來是直接打。
- CORS 相關 header 放在 request header 中，但 CORS 通常主要由 server response header 控制。
- 上傳影片是 `withData: true` 一次讀進記憶體，大影片可能造成 Web 或低階裝置記憶體壓力。

## 建議後續整理方向

優先順序可以這樣排：

1. 更新 `README.md`，把本文件精簡成開發者入口。
2. 修掉或替換 `test/widget_test.dart`，至少改成能 pump app shell 的 smoke test。
3. 把 API base URL 改成 `--dart-define` 或 flavor。
4. 把 API endpoint 的 `[method, url]` List 改成明確 class，例如 `ApiEndpoint(method, url)`。
5. 補 WebSocket 協議文件，記錄 message type 與 data schema。
6. 補上 upload state 與 record state 的單元測試。
7. 評估影片上傳是否能改成 stream 或限制大小，避免一次載入完整 bytes。
8. 若要重命名 `seperately` 等 typo，先確認後端 endpoint 是否也能同步改。

## 如何執行

安裝依賴：

```bash
flutter pub get
```

跑 Flutter：

```bash
flutter run
```

跑 Web：

```bash
flutter run -d chrome
```

靜態分析：

```bash
flutter analyze
```

測試：

```bash
flutter test
```

注意：目前 `test/widget_test.dart` 仍是 Flutter 預設 counter 測試，和 App 現況不符合，因此 `flutter test` 可能不具代表性，甚至可能失敗。

如果團隊使用 FVM，專案根目錄有 `.fvmrc`，可用對應 Flutter 版本執行：

```bash
fvm flutter pub get
fvm flutter run
```

## 一句話總結

這個專案是一個 Flutter 寫的「百米跑步多相機分析前端」。它讓使用者建立跑者資料、上傳或同步錄製多台相機影片，將影片與錨點校正資訊送到後端分析，並在分析完成後用影片播放器、歷史紀錄表與同步圖表呈現跑步表現。
