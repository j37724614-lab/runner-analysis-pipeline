import cv2
import os

def video_to_images(video_path, output_dir):
    # 1. 建立輸出目錄
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"建立目錄: {output_dir}")

    # 2. 開啟影片檔案
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"錯誤: 無法開啟影片 {video_path}")
        return

    frame_count = 0
    print(f"開始轉換影片: {video_path}")

    while True:
        # 讀取一幀
        ret, frame = cap.read()
        
        # 如果讀取失敗（影片結束），跳出迴圈
        if not ret:
            break
        
        # 3. 儲存圖片（檔名補零以方便排序，例如 frame_0001.jpg）
        frame_count += 1
        file_name = f"frame_{frame_count:04d}.jpg"
        file_path = os.path.join(output_dir, file_name)
        
        cv2.imwrite(file_path, frame)
        
        # 每 100 幀顯示一次進度
        if frame_count % 100 == 0:
            print(f"已處理 {frame_count} 幀...")

    cap.release()
    print(f"轉換完成！總共儲存了 {frame_count} 張圖片至 {output_dir}")

if __name__ == "__main__":
    # --- 設定區 ---
    VIDEO_INPUT = "output_cut/sequential_tracked.mp4"
    IMAGE_OUTPUT_DIR = "/home/jeter/pipeline_release/output_frames"
    # --------------
    
    video_to_images(VIDEO_INPUT, IMAGE_OUTPUT_DIR)
