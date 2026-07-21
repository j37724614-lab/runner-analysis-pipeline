# 腳部關鍵點整合與修正紀錄（2026-07-21）

本文件記錄今天針對「腳尖/腳跟關鍵點偵測」功能所做的整合與後續修正，依時間順序列出每一項更正的問題、原因與解法。

## 背景

原本的 HRNet-W48 只輸出 COCO 17 個身體關節，沒有腳尖/腳跟點。MotionAGFormer 的 3D 模型架構固定吃 17 個關節（`nn.Parameter` 寫死維度），無法直接改成 23 點重新訓練。因此採用「側路（side-channel）」設計：

- 身體 17 點：完全不動，照原本流程走（HRNet → `h36m_coco_format` → MotionAGFormer 3D 抬升 → DP 左右腿修正 → 平滑）。
- 腳部 6 點（左右腳：大腳趾、小腳趾、腳跟）：從 HRNet-W48 COCO-WholeBody 模型（`hrnet_w48_coco_wholebody_384x288_dark`）額外輸出，另存成 `input_2D/foot_keypoints.npz`，不進 3D 模型。

## 更正 1：checkpoint 格式轉換

**問題**：MMPose 釋出的 wholebody checkpoint 格式（`{'meta':..., 'state_dict': {'backbone.x', 'keypoint_head.x'}}`）與專案本地 vendored 的 `pose_hrnet.py` 所需的扁平 state_dict 不相容。

**解法**：新增 `scripts/tools/convert_wholebody_checkpoint.py`，去除 `backbone.`/`keypoint_head.` 前綴、將 `final_layer.weight/bias` 裁切到前 23 channel，輸出成扁平 dict。以 strict load 驗證，0 個 missing/unexpected key。

## 更正 2：腳部點解碼精度（DarkPose unbiased decoding）

**問題**：原本 `get_final_preds` 只做 argmax + 四分之一像素偏移的簡易解碼，精度不足以支撐腳尖這種小型關節的偵測。

**解法**：在 `MotionAGFormer/demo/lib/hrnet/lib/utils/inference.py` 新增 `get_final_preds_dark()`，實作 DarkPose 的「高斯模糊（kernel=11）+ log-heatmap + 二階泰勒展開」次像素精修解碼，對齊 checkpoint 內嵌 MMPose config 的 `modulate_kernel=11, unbiased_decoding=True` 設定。

**更正 2.1（範圍擴大）**：使用者要求「身體 17 點也要走 DarkPose」，因此把 `gen_kpts.py` 的解碼邏輯統一成全 23 點都走 `get_final_preds_dark`，移除原本區分身體/腳部兩套解碼的中間版本（`NUM_BODY_JOINTS` 相關的過渡程式碼）。

## 更正 3：骨架繪圖路徑遺漏腳部點

**問題**：新增腳部資料後，專案內實際會畫骨架疊圖的兩個地方（`MotionAGFormer/demo/vis.py` 的 `show2Dpose`、`core/overlay.py` 的 `show2Dpose_original`，後者是正式產出 `output_final.mp4` 使用的函式）都還沒有畫腳部點的邏輯。

**解法**：兩處都加上可選參數 `foot_kps`/`foot_scores`，用黃色線段從 H36M 腳踝（`LEFT_ANKLE=6`, `RIGHT_ANKLE=3`）連到對應腳部點，`score < 0.3` 時跳過不畫。`core/overlay.py` 版本另外加上 `offset_x/offset_y` 轉換，因為它畫在「原始未裁切影片」座標系上。

## 更正 4：測試入口不對

**問題**：一開始的測試都是呼叫較低層的 `run_pipeline()`，但這個入口沒有接左右腿 DP 修正機制。

**解法**：改用正式的 `run_analysis()`（"一鍵啟動" 高層入口，含 DP 修正 + 最終疊圖 + 網頁轉檔）重新測試，才能驗證腳部點在真實產線流程下的行為。

## 更正 5：左右腳交叉偵測（跟身體 DP 修正機制脫節）

**問題**：跑步影片中身體 17 點的左右腿常因遮擋/交叉而誤判左右（DP 修正機制 `apply_anchor_leg_correction` 會偵測並修正這種左右互換）。但腳部 6 點是獨立存的，並未套用同一套修正，導致腳部點的左右腳歸屬跟身體不同步。

**解法**：新增 `apply_leg_swap_to_foot_keypoints()`（`scripts/analysis/ankle_step_stride.py`），重用 `apply_anchor_leg_correction()` 算出的 `swapped_mask`（每幀布林值），在被判定為「交換」的幀，把左右腳的 6 個點對調。並在 `core/pipeline.py` 的 `run_analysis()` 中，於 `apply_anchor_leg_correction()` 之後立即呼叫。

## 更正 6：診斷 sheet 顯示腳部點交叉連線（本次最新修正）

**問題**：使用者在診斷 sheet（`diagnostic_sheets_IMG_2533_with_feet/..._sheet_seq_020_039.jpg`）中發現 seq31~39 的黃色腳部連線出現明顯 X 型交叉。

**排查過程**：
1. 寫單元測試驗證 `apply_leg_swap_to_foot_keypoints()` 本身的交換邏輯數學上正確。
2. 進一步發現真正原因：診斷腳本把「腳部點（原始未平滑）」跟「身體骨架的平滑後腳踝（`keypoints.npz`，經過 `_post_dp_smooth_and_limit_legs()`）」放在一起畫。這個平滑步驟在 DP 交換邊界後會做**最多約 50px 的修正性跳動**，用來消除交換造成的位置不連續。腳部點因為從未被平滑，仍停留在原始（僅左右對調）位置，於是跟平滑後大幅位移的腳踝之間出現視覺上的交叉——**並非左右腳身分判斷錯誤**（用 raw-vs-raw 比對驗證全程穩定在 3-7px）。

**解法**：新增 `align_foot_keypoints_to_body()`（`scripts/analysis/ankle_step_stride.py`），把每個腳部點表示成「相對於原始（僅左右對調後）腳踝的偏移量」，再把這個偏移量套用到**平滑後**的腳踝位置上，重新算出腳部點座標。這樣腳部點就會跟著身體骨架的平滑/DP 修正結果一起移動，而不是停在未平滑的位置。同樣接在 `core/pipeline.py` 的 `run_analysis()` 中呼叫，緊接在 `apply_leg_swap_to_foot_keypoints()` 之後：

```python
swapped_mask = apply_anchor_leg_correction(kps_npz, step_analysis["step_events"])
foot_npz = os.path.join(final_pose_dir, "input_2D", "foot_keypoints.npz")
apply_leg_swap_to_foot_keypoints(foot_npz, swapped_mask)
raw_kps_npz = os.path.join(final_pose_dir, "input_2D", "keypoints_raw.npz")
align_foot_keypoints_to_body(foot_npz, raw_kps_npz, kps_npz, swapped_mask)
```

**驗證結果**：
- 數值：t=34~39 的「腳跟-腳踝」距離從修正前的 35~53px，收斂到修正後穩定的 3.4~7.1px（跟其他無 DP 交換的幀一致）。
- 視覺：重新產生診斷 sheet 後，seq31~39 的 X 型交叉線完全消失，腳部點乾淨地連在各自對應的腳上。

## 目前資料流總覽

```
HRNet-W48 wholebody (23 點, DarkPose 解碼)
  → 前 17 點：h36m_coco_format → MotionAGFormer 3D → apply_anchor_leg_correction（DP 左右腿修正 + 平滑）
  → 後 6 點：foot_keypoints.npz
       → apply_leg_swap_to_foot_keypoints()   跟身體同步做左右交換
       → align_foot_keypoints_to_body()       跟身體同步做平滑後重新錨定
  → 疊圖：vis.py show2Dpose / core/overlay.py show2Dpose_original 畫黃色腳部連線
```

三個階段（HRNet 原始輸出 → 左右交換 → 平滑錨定）腳部點的處理都跟身體骨架保持一致。

## 相關檔案

| 檔案 | 變更內容 |
|---|---|
| `scripts/tools/convert_wholebody_checkpoint.py` | 新增，checkpoint 格式轉換 |
| `MotionAGFormer/demo/lib/hrnet/experiments/w48_384x288_wholebody23_dark.yaml` | 新增，23 關節 config |
| `MotionAGFormer/demo/lib/hrnet/lib/utils/inference.py` | 新增 `get_final_preds_dark()` |
| `MotionAGFormer/demo/lib/hrnet/gen_kpts.py` | 全 23 點改走 DarkPose 解碼 |
| `MotionAGFormer/demo/vis.py` | 切分身體/腳部點、存 `foot_keypoints.npz`、`show2Dpose` 加腳部繪圖 |
| `core/overlay.py` | `show2Dpose_original` 加腳部繪圖、`overlay_videos` 讀取 foot_keypoints.npz |
| `scripts/analysis/ankle_step_stride.py` | 新增 `apply_leg_swap_to_foot_keypoints()`、`align_foot_keypoints_to_body()` |
| `core/pipeline.py` | `run_analysis()` 接上述兩個函式 |
| `scripts/tools/generate_leg_swap_diagnostic_sheets.py` | 新增，診斷 sheet 產生工具（含腳部點繪圖） |
