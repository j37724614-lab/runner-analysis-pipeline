"""
add_angle_overlay.py

將 vis.py 輸出的 2D 影片與角度 CSV 合併為單一影片：
  - 上半：原始影片；右下角疊 vis.py 偵測用的追焦 2D 骨架影片
  - 下半：4 個角度折線圖（2×2 排列）

圖表特性：
  - X 軸：開始即固定為全部幀數範圍，曲線從左向右生長
  - Y 軸：整部影片固定（開始前計算全域 min/max），刻度不跳動
  - 重用同一 matplotlib Figure，逐幀更新線段資料（效能優化）

用法：
  python add_angle_overlay.py \
      --video  .../tracked_cropped_2D.mp4 \
      --csv    .../tracked_cropped_angles.csv \
      --output .../tracked_cropped_2D_angles.mp4 \
      [--main-video .../original.mp4] \
      [--main-videos .../cam1.mp4 .../cam2.mp4 .../cam3.mp4] \
      [--chart_height 200] [--dpi 100]
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# 此腳本位於 scripts/visualization/，往上三層為 repo 根目錄
_BASE_DIR = Path(__file__).resolve().parent.parent.parent

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
FONT_PATH = str(_BASE_DIR / "ChineseFont.ttf")

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


def _resize_keep_aspect(frame, target_w=None, target_h=None):
    h, w = frame.shape[:2]
    if target_h is not None:
        new_h = int(target_h)
        new_w = max(1, int(round(w * new_h / max(h, 1))))
    elif target_w is not None:
        new_w = int(target_w)
        new_h = max(1, int(round(h * new_w / max(w, 1))))
    else:
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _fit_to_canvas(frame, target_w, target_h):
    resized = _resize_keep_aspect(frame, target_h=target_h)
    h, w = resized.shape[:2]
    if w == target_w:
        return resized
    if w < target_w:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x0 = (target_w - w) // 2
        canvas[:, x0:x0 + w] = resized
        return canvas
    x0 = (w - target_w) // 2
    return resized[:, x0:x0 + target_w]


def _compose_main_with_inset(main_frame, inset_frame, target_w, target_h,
                             inset_height_ratio=0.45, inset_margin=10):
    canvas = _fit_to_canvas(main_frame, target_w, target_h)
    if inset_frame is None or inset_frame.size == 0:
        return canvas

    inset_h = max(1, int(round(target_h * inset_height_ratio)))
    inset = _resize_keep_aspect(inset_frame, target_h=inset_h)
    ih, iw = inset.shape[:2]
    max_w = max(1, target_w - inset_margin * 2)
    if iw > max_w:
        inset = _resize_keep_aspect(inset, target_w=max_w)
        ih, iw = inset.shape[:2]

    x1 = target_w - inset_margin - iw
    y1 = target_h - inset_margin - ih
    x2 = x1 + iw
    y2 = y1 + ih
    cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), -1)
    canvas[y1:y2, x1:x2] = inset
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return canvas


def _smooth_camera_boundary_angles(df, frame_map, cols, blend_frames=30):
    """
    Smooth display-only angle discontinuities at camera boundaries.

    The raw CSV is left untouched on disk. For each camera transition, the first
    frame of the new camera is offset to match the previous frame, then that
    offset linearly decays to zero over blend_frames.
    """
    if frame_map is None or blend_frames <= 0:
        return df, []

    frame_to_cam = {
        int(frame): int(info['cam'])
        for frame, info in frame_map.items()
        if 'cam' in info
    }
    if not frame_to_cam:
        return df, []

    out = df.copy()
    frame_values = [int(v) for v in out['frame'].tolist()]
    frame_set = set(frame_values)
    transitions = []
    prev_cam = None
    prev_frame = None
    for frame in frame_values:
        cam = frame_to_cam.get(frame)
        if cam is None:
            continue
        if prev_cam is not None and cam != prev_cam and prev_frame in frame_set:
            transitions.append((frame, prev_frame, prev_cam, cam))
        prev_cam = cam
        prev_frame = frame

    applied = []
    for start_frame, prev_frame, prev_cam, cam in transitions:
        end_frame = start_frame + blend_frames
        mask = (out['frame'] >= start_frame) & (out['frame'] < end_frame)
        blend_idx = out.loc[mask].index.tolist()
        if not blend_idx:
            continue
        for col in cols:
            if col not in out.columns:
                continue
            prev_vals = out.loc[out['frame'] == prev_frame, col]
            start_vals = out.loc[out['frame'] == start_frame, col]
            if prev_vals.empty or start_vals.empty:
                continue
            offset = float(prev_vals.iloc[0] - start_vals.iloc[0])
            for i, idx in enumerate(blend_idx):
                weight = max(0.0, 1.0 - (i / max(blend_frames, 1)))
                out.at[idx, col] = out.at[idx, col] + offset * weight
            applied.append((start_frame, prev_cam, cam, col, offset))

    return out, applied


def add_angle_overlay(video_path, csv_path, output_path,
                      main_video_path=None,
                      main_video_paths=None,
                      frame_map_path=None,
                      chart_height=200, display_height=340,
                      inset_height_ratio=0.45, inset_margin=10,
                      smooth_camera_boundary=True,
                      boundary_blend_frames=30,
                      dpi=100):
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

    inset_cap = cv2.VideoCapture(video_path)
    if not inset_cap.isOpened():
        raise RuntimeError(f"無法開啟 2D 追焦骨架影片: {video_path}")

    if main_video_paths is None:
        main_video_paths = []
    if main_video_path and not main_video_paths:
        main_video_paths = [main_video_path]

    main_caps = []
    for idx, path in enumerate(main_video_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            for opened in main_caps:
                opened.release()
            raise RuntimeError(f"無法開啟原始影片 CAM{idx + 1}: {path}")
        main_caps.append(cap)

    frame_map = None
    if frame_map_path:
        if not os.path.exists(frame_map_path):
            raise FileNotFoundError(f"frame map 不存在: {frame_map_path}")
        frame_map = {}
        with open(frame_map_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                output_frame = int(row['output_frame'])
                frame_map[output_frame] = {
                    'cam': int(row.get('cam') or 1),
                    'source_frame': int(row['source_frame']),
                }
        if smooth_camera_boundary:
            df, applied_boundary_smoothing = _smooth_camera_boundary_angles(
                df,
                frame_map,
                cols=['pelvis_torso_angle'],
                blend_frames=boundary_blend_frames,
            )
        else:
            applied_boundary_smoothing = []
    else:
        applied_boundary_smoothing = []

    inset_w = int(inset_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    inset_h = int(inset_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = inset_cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(inset_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if main_caps:
        main_w = int(main_caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        main_h = int(main_caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_h = max(1, int(display_height))
        video_w = max(1, int(round(main_w * video_h / max(main_h, 1))))
    else:
        video_w, video_h = inset_w, inset_h
    chart_h = max(0, int(chart_height))

    print(f"  追焦骨架影片: {inset_w}×{inset_h}，{total} 幀，{fps:.1f} FPS")
    if main_caps:
        if len(main_video_paths) == 1:
            print(f"  原始主畫面: {main_video_paths[0]}")
        else:
            print(f"  原始主畫面: {len(main_video_paths)} 支")
            for idx, path in enumerate(main_video_paths, start=1):
                print(f"    CAM{idx}: {path}")
    else:
        print("  原始主畫面: 未指定，使用追焦骨架影片作為主畫面")
    if frame_map_path:
        print(f"  Frame map: {frame_map_path}")
    if applied_boundary_smoothing:
        print(f"  Camera boundary smoothing: pelvis_torso_angle, {boundary_blend_frames} 幀")
        for start_frame, prev_cam, cam, col, offset in applied_boundary_smoothing:
            print(f"    frame {start_frame}: CAM{prev_cam} → CAM{cam}, {col} offset {offset:+.2f}°")
    print(f"  圖表面板: {video_w}×{chart_h}")
    print(f"  輸出尺寸: {video_w}×{video_h + chart_h}")

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
        figsize=(video_w / dpi, max(chart_h, 1) / dpi),
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
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Ensure even dimensions for compatibility
    total_w = video_w // 2 * 2
    total_h = (video_h + chart_h) // 2 * 2
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_w, total_h))

    # ------------------------------------------------------------------
    # 主迴圈：逐幀更新折線、渲染圖表、合併影片
    # ------------------------------------------------------------------
    frame_idx = 0
    last_main_frame = None
    while True:
        ret, inset_frame = inset_cap.read()
        if not ret:
            break
        if main_caps:
            main_cap = main_caps[0]
            source_frame = None
            cam_no = 1
            if frame_map is not None and frame_idx in frame_map:
                mapped = frame_map[frame_idx]
                cam_no = mapped['cam']
                source_frame = mapped['source_frame']
                if 1 <= cam_no <= len(main_caps):
                    main_cap = main_caps[cam_no - 1]
                else:
                    print(f"  ⚠  frame_map CAM{cam_no} 超出 main videos 數量，改用 CAM1")
                    cam_no = 1
                main_cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ret_main, main_frame = main_cap.read()
            if ret_main:
                last_main_frame = main_frame
            elif last_main_frame is not None:
                main_frame = last_main_frame
            else:
                main_frame = inset_frame
            frame = _compose_main_with_inset(
                main_frame, inset_frame, video_w, video_h,
                inset_height_ratio=inset_height_ratio,
                inset_margin=inset_margin,
            )
        else:
            frame = _fit_to_canvas(inset_frame, video_w, video_h)

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

        # 縮放至精確目標尺寸，接到影片底部
        chart_bgr = cv2.resize(chart_bgr, (video_w, chart_h),
                               interpolation=cv2.INTER_LANCZOS4)

        combined = np.vstack([frame, chart_bgr])
        if (combined.shape[1], combined.shape[0]) != (total_w, total_h):
            combined = cv2.resize(combined, (total_w, total_h))
        out.write(combined)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  [{frame_idx}/{total}] {frame_idx/total*100:.0f}%")

    inset_cap.release()
    for cap in main_caps:
        cap.release()
    out.release()
    plt.close(fig)
    print(f"\n  輸出完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='在 2D 影片底部加入 4 個角度折線圖（2×2 排列）'
    )
    parser.add_argument('--video',          required=True, help='2D 影片路徑')
    parser.add_argument('--csv',            required=True, help='角度 CSV 路徑')
    parser.add_argument('--output',         required=True, help='輸出影片路徑')
    parser.add_argument('--main-video',     default=None,
                        help='原始主畫面影片路徑；指定後 2D 追焦骨架影片會疊在右下角')
    parser.add_argument('--main-videos',    nargs='+', default=None,
                        help='多台原始主畫面影片路徑；搭配 frame map 的 cam 欄位切換')
    parser.add_argument('--frame-map',      default=None,
                        help='追焦輸出幀對應原始影片幀的 CSV；欄位需含 output_frame/source_frame')
    parser.add_argument('--chart_w_ratio',  type=float, default=2,
                        help='保留相容舊參數；目前圖表固定接在影片底部，不使用此值')
    parser.add_argument('--chart_height',   type=int, default=200,
                        help='底部圖表高度（預設 200）')
    parser.add_argument('--display_height', type=int, default=340,
                        help='上方主畫面高度（預設 340）')
    parser.add_argument('--inset_height_ratio', type=float, default=0.45,
                        help='右下追焦小窗高度比例（相對上方主畫面，預設 0.45）')
    parser.add_argument('--inset_margin',   type=int, default=10,
                        help='右下追焦小窗邊距（預設 10）')
    parser.add_argument('--no-boundary-smooth', dest='smooth_camera_boundary',
                        action='store_false',
                        help='關閉跨相機接縫角度平滑')
    parser.add_argument('--boundary_blend_frames', type=int, default=30,
                        help='跨相機接縫角度 offset 漸退幀數（預設 30）')
    parser.add_argument('--dpi',            type=int, default=100,
                        help='matplotlib DPI（預設 100）')
    parser.set_defaults(smooth_camera_boundary=True)
    args = parser.parse_args()

    add_angle_overlay(
        args.video, args.csv, args.output,
        main_video_path=args.main_video,
        main_video_paths=args.main_videos,
        frame_map_path=args.frame_map,
        chart_height=args.chart_height,
        display_height=args.display_height,
        inset_height_ratio=args.inset_height_ratio,
        inset_margin=args.inset_margin,
        smooth_camera_boundary=args.smooth_camera_boundary,
        boundary_blend_frames=args.boundary_blend_frames,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
