"""集中管理角度 CSV 的檔案系統副作用。"""

import os
import shutil

import pandas as pd


def exists(path: str | None) -> bool:
    """回傳角度 CSV 路徑是否存在。"""
    return bool(path and os.path.exists(path))


def read(path: str) -> pd.DataFrame:
    """讀取角度 CSV。"""
    return pd.read_csv(path)


def write(path: str, dataframe: pd.DataFrame) -> None:
    """覆寫角度 CSV，且不輸出 DataFrame 索引。"""
    dataframe.to_csv(path, index=False)


def remove(path: str) -> None:
    """刪除角度 CSV。"""
    os.remove(path)


def publish(source_path: str, output_path: str) -> None:
    """保留檔案中繼資料並發佈角度 CSV。"""
    shutil.copy2(source_path, output_path)
