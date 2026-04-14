"""
add_angle_overlay.py

將 vis.py 輸出的 2D 影片與角度 CSV 合併為單一影片：
  - 左半：原始 2D 骨架影片
  - 右半：4 個角度折線圖（2×2 排列）

圖表特性：
  - X 軸：開始即固定為全部幀數範圍，曲線從左向右生長
  - Y 軸：整部影片固定（開始前計算全域 min/max），刻度不跳動
  - 重用同一 matplotlib Figure，逐幀更新線段資料（效能優化）

用法：
  python add_angle_overlay.py \
      --video  .../tracked_cropped_2D.mp4 \
      --csv    .../tracked_cropped_angles.csv \
      --output .../tracked_cropped_2D_angles.mp4 \
      [--chart_w_ratio 1.5] [--dpi 100]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
import numpy as np
import cv2
import pandas as pd


# ---------------------------------------------------------------------------
# 中文字型
# ---------------------------------------------------------------------------
FONT_PATH = '/home/jeter/MotionAGFormer/MotionAGFormer/ChineseFont.ttf'

# CSV 欄位 → 圖例簡短標籤（顯示在折線圖右上角）
COL_ZH = {
    'left_knee_angle':           '左膝',
    'right_knee_angle':          '右膝',
    'left_hip_angle':            '左髖',
    'right_hip_angle':           '右髖',
    'left_arm_torso_angle':      '左臂軀幹',
    'right_arm_torso_angle':     '右臂軀幹',
    'left_elbow_flexion_angle':  '左手肘',
    'right_elbow_flexion_angle': '右手肘',
    'left_shoulder_flexion':     '左肩',
    'right_shoulder_flexion':    '右肩',
    'pelvis_torso_angle':        '骨盆軀幹',
}

# ---------------------------------------------------------------------------
# 4 個角度 Panel 設定（labels 從 COL_ZH 對照）
# ---------------------------------------------------------------------------
PANELS = [
    {
        'title':  '膝關節角度',
        'cols':   ['left_knee_angle', 'right_knee_angle'],
        'colors': ['blue', 'red'],
    },
    {
        'title':  '手肘屈曲角度',
        'cols':   ['left_elbow_flexion_angle', 'right_elbow_flexion_angle'],
        'colors': ['blue', 'red'],
    },
    {
        'title':  '肩關節屈曲角度',
        'cols':   ['left_shoulder_flexion', 'right_shoulder_flexion'],
        'colors': ['blue', 'red'],
    },
    {
        'title':  '骨盆軀幹角度',
        'cols':   ['pelvis_torso_angle'],
        'colors': ['green'],
    },
]


def add_angle_overlay(video_path, csv_path, output_path,
                      chart_w_ratio=1.5, dpi=100):
    # ------------------------------------------------------------------
    # 中文字型
    # ------------------------------------------------------------------
    zh_font = FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None
    if zh_font is None:
        print("  ⚠  ChineseFont.ttf 不存在，改用預設字型")

    # ------------------------------------------------------------------
    # 讀取資料
    # ------------------------------------------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"角度 CSV 不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    csv_len = len(df)
    print(f"  CSV 已讀取：{csv_len} 幀")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟 2D 影片: {video_path}")

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chart_w = int(video_w * chart_w_ratio)

    print(f"  影片: {video_w}×{video_h}，{total} 幀，{fps:.1f} FPS")
    print(f"  圖表面板: {chart_w}×{video_h}")
    print(f"  輸出尺寸: {video_w + chart_w}×{video_h}")

    # ------------------------------------------------------------------
    # 預計算各 panel 的全域 Y 軸範圍（整部影片固定）
    # ------------------------------------------------------------------
    panel_ylims  = []
    panel_yticks = []
    for panel in PANELS:
        series_list = [df[c].dropna() for c in panel['cols'] if c in df.columns]
        if series_list:
            combined = pd.concat(series_list)
            # 用 5–95 百分位數做顯示範圍，排除離群值讓密集區填滿畫面
            lo = combined.quantile(0.05)
            hi = combined.quantile(0.95)
            margin = max((hi - lo) * 0.15, 3.0)
            y0 = lo - margin
            y1 = hi + margin
        else:
            y0, y1 = 0.0, 180.0
        panel_ylims.append((y0, y1))
        # 在 y0–y1 範圍內產生約 5 個整數刻度
        span = y1 - y0
        # 選步進：讓刻度數落在 4–6 個
        for step in [1, 2, 5, 10, 15, 20, 25, 30]:
            if span / step <= 6:
                break
        first = int(np.ceil(y0 / step)) * step
        ticks = np.arange(first, y1 + step * 0.01, step)
        ticks = ticks[(ticks >= y0) & (ticks <= y1)]
        panel_yticks.append(ticks)

    # ------------------------------------------------------------------
    # 建立 matplotlib Figure（只建一次，之後只更新線段資料）
    # ------------------------------------------------------------------
    x_max = max(df['frame'].iloc[-1], csv_len - 1) if csv_len > 0 else 100

    fig, axes = plt.subplots(
        2, 2,
        figsize=(chart_w / dpi, video_h / dpi),
        dpi=dpi,
    )
    fig.patch.set_facecolor('#ffffff')
    axes_flat = axes.flatten()

    # 每個 panel 的 Line2D 物件列表
    panel_lines = []   # panel_lines[i] = [(col, line_obj), ...]
    panel_dots  = []   # panel_dots[i]  = [(col, dot_obj), ...]

    for i, (ax, panel, ylim, yticks) in enumerate(
            zip(axes_flat, PANELS, panel_ylims, panel_yticks)):

        ax.set_facecolor('#ffffff')
        ax.set_xlim(0, x_max)
        ax.set_ylim(ylim[0], ylim[1])
        # 固定刻度，不讓 matplotlib 自動調整
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        # 標題：中文字型
        title_kw = {'fontsize': 6, 'color': 'black', 'pad': 3, 'fontweight': 'bold'}
        if zh_font:
            title_kw['fontproperties'] = zh_font
        ax.set_title(panel['title'], **title_kw)
        ax.tick_params(colors='black', labelsize=6)
        ax.spines[:].set_color('black')
        ax.grid(True, alpha=0.3)

        col_lines = []
        col_dots  = []
        for col, color in zip(panel['cols'], panel['colors']):
            if col not in df.columns:
                continue
            zh_label = COL_ZH.get(col, col)
            ln,  = ax.plot([], [], color=color, lw=1.0, label=zh_label, alpha=1.0)
            dot, = ax.plot([], [], 'o', color=color, ms=6, zorder=5)
            col_lines.append((col, ln))
            col_dots.append((col, dot))

        if len(col_lines) > 1:
            legend_kw = dict(loc='upper right', facecolor='#ffffff',
                             edgecolor='none', labelcolor='black',
                             handlelength=1, fontsize=5.5)
            if zh_font:
                legend_font = zh_font.copy()
                legend_font.set_size(5)   # 在這裡強制縮小圖例字體為 5
                legend_kw['prop'] = legend_font
                legend_kw.pop('fontsize')   # prop 指定時不能同時給 fontsize
            ax.legend(**legend_kw)

        panel_lines.append(col_lines)
        panel_dots.append(col_dots)

    plt.tight_layout(pad=0.4)
    fig.canvas.draw()   # 初始化 renderer

    # ------------------------------------------------------------------
    # 輸出 VideoWriter
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (video_w + chart_w, video_h))

    # ------------------------------------------------------------------
    # 主迴圈：逐幀更新折線、渲染圖表、合併影片
    # ------------------------------------------------------------------
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # CSV 幀數若少於影片幀數，夾住最後一幀
        csv_idx = min(frame_idx, csv_len - 1)

        # 更新各 panel 的線段資料
        for col_lines, col_dots in zip(panel_lines, panel_dots):
            for (col, ln), (_, dot) in zip(col_lines, col_dots):
                x_data = df['frame'].iloc[:csv_idx + 1].values
                y_data = df[col].iloc[:csv_idx + 1].values
                ln.set_data(x_data, y_data)
                dot.set_data([df['frame'].iloc[csv_idx]],
                             [df[col].iloc[csv_idx]])

        # 渲染 → numpy BGR
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        chart_rgb = buf[:, :, :3]
        chart_bgr = cv2.cvtColor(chart_rgb, cv2.COLOR_RGB2BGR)

        # 縮放至精確目標尺寸
        chart_bgr = cv2.resize(chart_bgr, (chart_w, video_h),
                               interpolation=cv2.INTER_LANCZOS4)

        combined = np.hstack([frame, chart_bgr])
        out.write(combined)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  [{frame_idx}/{total}] {frame_idx/total*100:.0f}%")

    cap.release()
    out.release()
    plt.close(fig)
    print(f"\n  輸出完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='在 2D 影片右側加入 4 個角度折線圖（2×2 排列）'
    )
    parser.add_argument('--video',          required=True, help='2D 影片路徑')
    parser.add_argument('--csv',            required=True, help='角度 CSV 路徑')
    parser.add_argument('--output',         required=True, help='輸出影片路徑')
    parser.add_argument('--chart_w_ratio',  type=float, default=2,
                        help='圖表面板寬度比例（相對影片寬，預設 1.5）')
    parser.add_argument('--dpi',            type=int, default=100,
                        help='matplotlib DPI（預設 100）')
    args = parser.parse_args()

    add_angle_overlay(
        args.video, args.csv, args.output,
        chart_w_ratio=args.chart_w_ratio,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
