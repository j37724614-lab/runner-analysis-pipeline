"""
track_runners.py — 運動表現分析追蹤器（極簡 CLI 入口，底層由 core.tracker 驅動）

此檔案已重構為輕量包裝器，其核心追蹤與圖表邏輯已移入 core/ 軟體包中。
"""

from core.tracker import main

if __name__ == "__main__":
    main()
