"""
core package

跑者多相機動作分析管線核心實作模組。
暴露了主要的高階流程控制介面，使根目錄入口程式能保持輕量且整潔。
"""

from core.pipeline import run_pipeline, run_analysis
from core.tracker import run_tracker

__all__ = [
    'run_pipeline',
    'run_analysis',
    'run_tracker'
]
