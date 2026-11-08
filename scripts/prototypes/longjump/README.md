# 跳遠事件分析原型

演算法說明見 [docs/developer/longjump_board_sand_detection.md](../../../docs/developer/longjump_board_sand_detection.md)。

所有功能都在 `longjump_analysis.py`，還沒有整合進正式 pipeline。路徑都相對於 repo 根目錄，資料放在 `data/videos/`（已被 .gitignore 排除）。

## 指令

在這個資料夾內執行：

```bash
cd scripts/prototypes/longjump
PY="/home/jeter/.conda/envs/yolo_new/bin/python -W ignore"

$PY longjump_analysis.py board ../../../1_3IMG_0117.MOV /tmp/board_0117 --pit-side left
$PY longjump_analysis.py full-video /path/to/complete_jump.MOV /path/to/output --name jump_01
$PY longjump_analysis.py events
$PY longjump_analysis.py takeoff-frames
$PY longjump_analysis.py marks
$PY longjump_analysis.py videos
```

| 指令 | 內容 | 輸出 |
|---|---|---|
| `full-video` | 完整影片一鍵執行動作 prescan、跳遠者追蹤、WholeBody23 姿態、事件分析、人物 segmentation 與左上角事件面板 | `<output_dir>/<name>_event_review.mp4`、`<name>_event_results.json`、prescan／tracking／pose 中間產物 |
| `board` | 用第一幀偵測起跳板與起跳線 | 印出 JSON、`<前綴>_det.jpg` |
| `events` | 以單腳低速區間找起跳支撐期，再用起跳板分類 `BEFORE_BOARD` / `ON_BOARD` / `OVER_LINE` | `jump_events/board_aware_results.csv` |
| `takeoff-frames` | 輸出每一跳最小腳尖淨距發生幀，標示起跳板、起跳線、腳趾與越線結果 | `jump_events/takeoff_line_frames/*_full.jpg`、`*_zoom.jpg`、`takeoff_clearance_sheet.jpg` |
| `marks` | 人物分割找最靠近起跳線的沙坑痕跡 | `data/videos/2026-10-19跳遠最後一機畫面/sand_mark/mark_results.csv`、`mark_sheet.jpg` |
| `videos` | 每一跳一支檢查影片（裁切後的片段），每幀畫上追蹤 bbox 與 W、L、W/L（W = 寬、L = 高，像素），以及起跳線、踩板、離板、落沙、裁切點、碰沙點、沙坑痕跡；分割輪廓只從騰空最高點畫到裁切點，落沙之後的分割直接沿用 `marks` 的結果 | `<資料集>/jump_review/<影片>_review.mp4`，以及同內容的 4 倍慢動作 `<影片>_review_slow4x.mp4` |

## 程式結構（`longjump_analysis.py`，由上到下）

| 區塊 | 內容 |
|---|---|
| rules | 每組門檻一個 frozen dataclass：`FlightRule`、`SandLandingRule`、`BoardContactRule`、`TrimRule`、`BoardRule`、`SandMarkRule` |
| data | `RunSpec`、`RUNS`（7 支影片）、`PoseTrack`（逐幀訊號）、讀取 GT 與人工紀錄 |
| trim | `find_trim_end`：高速段 → 錨點 → 坐在沙裡 → 開始起身 |
| events | `find_flight`、`find_sand_landing`、`find_board_contact_with_board`、`analyse_jump`（先裁切，再融合起跳板位置與腳部速度） |
| board | `detect_board`、`fit_takeoff_line`、`find_takeoff_line` |
| sand mark | `RunnerSegmenter`、`board_side_contact`、`find_sand_mark` |
| drawing / videos | `Mp4Writer`、`sand_mark_tile`、`JumpReviewVideo` |
| commands | 三個子指令 |

`yolo26x-seg.pt` 是 Ultralytics 官方權重（`YOLO("yolo26x-seg.pt")` 會自動下載），放在 `models/`，不進 git。

## 目前寫死的部分

- 影片清單 `RUNS`：10/19 的 5 跳和 0117、0118。
- 跑動方向：由右往左，沙坑在畫面左側。
