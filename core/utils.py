"""
core/utils.py

提供專案全域路徑管理、資源加載與通用的輔助工具。
此模組為專案中所有檔案（字型、模型權重、相機設定）定位的單一事實來源。
"""

import os
import sys
import subprocess
from pathlib import Path
from PIL import ImageFont
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 定義 Repo 根目錄
REPO_ROOT = Path(__file__).resolve().parent.parent

# 預設資源路徑
DEFAULT_FONT_PATH = str(REPO_ROOT / "assets" / "fonts" / "ChineseFont.ttf")
DEFAULT_MODEL_PATH = str(REPO_ROOT / "models" / "yolo26x.pt")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "output_cut")

def get_font_path():
    """取得中文字型的路徑，若不存在則會嘗試從多個備份路徑尋找。"""
    paths_to_check = [
        DEFAULT_FONT_PATH,
        str(REPO_ROOT / "ChineseFont.ttf"),  # 舊位置相容
        str(REPO_ROOT / "MotionAGFormer" / "ChineseFont.ttf"),  # 舊 scripts 位置相容
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            return p
    return None

def get_model_path(model_name="yolo26x.pt"):
    """取得指定模型權重檔案的路徑。"""
    paths_to_check = [
        str(REPO_ROOT / "models" / model_name),
        str(REPO_ROOT / model_name),  # 舊位置相容
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            return p
    # 如果找不到，預設回傳 Repo models 目錄下的預期路徑
    return str(REPO_ROOT / "models" / model_name)

def get_font(size=28, font_path=None):
    """
    載入 PIL 點陣字型，用於 OpenCV/PIL 文字繪製。
    失敗時回傳 None 以便安全降級為 OpenCV 內建 Hershey 字體。
    """
    if font_path is None:
        font_path = get_font_path()
        
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception as e:
            print(f"PIL 字型 {font_path} 載入失敗: {e}")
            return None
    return None

def configure_matplotlib_font(font_path=None):
    """
    設定 Matplotlib 繪圖引擎使用指定中文字型，防止圖片中的中文文字變成亂碼（豆腐塊）。
    """
    if font_path is None:
        font_path = get_font_path()
        
    if font_path and os.path.exists(font_path):
        try:
            if hasattr(fm.fontManager, 'addfont'):
                fm.fontManager.addfont(font_path)
            font_prop = fm.FontProperties(fname=font_path)
            font_name = font_prop.get_name()
            plt.rcParams['font.family'] = [font_name]
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[Core.Utils] 已成功配置 Matplotlib 中文字型: {font_name} ({font_path})")
            return font_prop
        except Exception as e:
            print(f"[Core.Utils] Matplotlib 中文字型配置失敗，將使用預設字型: {e}")
    else:
        print(f"[Core.Utils] 找不到中文字型檔，Matplotlib 改用預設字型: {font_path}")

    plt.rcParams['axes.unicode_minus'] = False
    return None

def convert_to_web_compatible_mp4(input_path, output_path=None):
    """
    使用 ffmpeg 將影片轉換為網頁相容格式 (H.264 編碼，並加入 faststart metadata 以支援串流播放)。
    如果轉換成功，將替換或產生目標檔案。
    """
    if output_path is None:
        output_path = input_path
        
    temp_output = input_path.replace(".mp4", "_web_temp.mp4")
    if temp_output == input_path:
        temp_output = input_path + ".web_temp.mp4"
        
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-preset", "ultrafast",
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        temp_output
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp_output, output_path)
        print(f"[Core.Utils] 影片已成功轉換為網頁相容格式: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Core.Utils] FFMPEG 網頁轉碼失敗: {e}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False
