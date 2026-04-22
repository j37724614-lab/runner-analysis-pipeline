import os
import subprocess

def convert_mov_to_mp4(input_path, output_path=None):
    """
    將 .mov 影片檔案轉換為 .mp4 格式（使用 ffmpeg 直接轉碼，支援 HEVC/H.265 iPhone 影片）。

    :param input_path: 輸入的 .mov 檔案路徑
    :param output_path: 輸出的 .mp4 檔案路徑 (若為 None 則自動生成)
    """
    if not input_path.lower().endswith('.mov'):
        print(f"錯誤: {input_path} 不是 .mov 檔案")
        return

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + '.mp4'

    print(f"正在轉換: {input_path} -> {output_path}")

    # 優先嘗試 libx264；若不支援則 fallback 到 copy（速度更快但保留原始編碼）
    for codec_args in [
        ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"],
        ["-c:v", "copy"],
    ]:
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            *codec_args,
            "-c:a", "aac",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("轉換完成！")
            return
        else:
            label = "libx264" if "-c:v" in codec_args and codec_args[codec_args.index("-c:v") + 1] == "libx264" else "copy"
            print(f"{label} 轉碼失敗，嘗試下一個方法...")

    print(f"轉換失敗: {result.stderr[-500:]}")

if __name__ == "__main__":
    # 批次處理：填入你要轉換的 .MOV 檔案路徑
    video_list = [
        "/home/jeter/pipeline_release/video/IMG_2526_11.mov",
        "/home/jeter/pipeline_release/video/0420_10.MOV",


    ]
    
    for video in video_list:
        if os.path.exists(video):
            convert_mov_to_mp4(video)
        else:
            print(f"找不到檔案: {video}")
