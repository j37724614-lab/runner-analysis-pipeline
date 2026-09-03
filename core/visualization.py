"""
core/visualization.py

將 vis.py 輸出的 2D 影片與角度 CSV 合併為單一影片：
  - 上半：原始影片；右下角疊 vis.py 偵測用的追焦 2D 骨架影片
  - 下半：4 個角度折線圖（2×2 排列）
此模組封裝了原本在 scripts/visualization/add_angle_overlay.py 的核心運算邏輯。
"""

import argparse
import csv
import os
from dataclasses import dataclass, field

import matplotlib

matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.font_manager import FontProperties

from core.utils import get_font_path

# 中文字型路徑（動態獲取）
FONT_PATH = get_font_path()

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

# 4 個角度 Panel 設定（labels 從 COL_ZH 對照）
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
    平滑跨相機交接處的角度不連續問題。
    此調整僅影響圖表顯示，硬碟上的 CSV 檔案不變動。
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


@dataclass
class AngleOverlayConfig:
    """add_angle_overlay() 的顯示/輸入設定；全部有預設值。"""
    main_videos: list = field(default_factory=list)
    frame_map_path: "str | None" = None
    chart_height: int = 200
    display_height: int = 340
    inset_height_ratio: float = 0.45
    inset_margin: int = 10
    smooth_camera_boundary: bool = True
    boundary_blend_frames: int = 30
    dpi: int = 100


@dataclass
class _Geometry:
    inset_w: int
    inset_h: int
    fps: float
    total: int
    video_w: int
    video_h: int
    chart_h: int
    total_w: int      # 輸出影片尺寸（各維向下取偶）
    total_h: int


def _zh_font():
    if FONT_PATH and os.path.exists(FONT_PATH):
        return FontProperties(fname=FONT_PATH)
    print("  ⚠️  [Core.Vis] ChineseFont.ttf 不存在，將使用系統預設字型")
    return None


def _load_angle_dataframe(csv_path, cfg):
    """讀角度 CSV，若有 frame_map 且啟用則做跨相機接縫平滑。回傳 (df, frame_map|None)。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"角度 CSV 不存在: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  [Core.Vis] CSV 數據讀取成功：{len(df)} 幀")

    frame_map = None
    if cfg.frame_map_path:
        if not os.path.exists(cfg.frame_map_path):
            raise FileNotFoundError(f"frame map 不存在: {cfg.frame_map_path}")
        frame_map = {}
        with open(cfg.frame_map_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                frame_map[int(row['output_frame'])] = {
                    'cam': int(row.get('cam') or 1),
                    'source_frame': int(row['source_frame']),
                }
        if cfg.smooth_camera_boundary:
            df, _ = _smooth_camera_boundary_angles(
                df, frame_map, cols=['pelvis_torso_angle'],
                blend_frames=cfg.boundary_blend_frames)
    return df, frame_map


def _open_overlay_caps(video_path, main_videos):
    """開 inset 骨架影片與 0..N 支原始主畫面影片；任一失敗即釋放已開的並拋出。"""
    inset_cap = cv2.VideoCapture(video_path)
    if not inset_cap.isOpened():
        raise RuntimeError(f"無法開啟 2D 追焦骨架影片: {video_path}")
    main_caps = []
    for idx, path in enumerate(main_videos):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            inset_cap.release()
            for opened in main_caps:
                opened.release()
            raise RuntimeError(f"無法開啟原始影片 CAM{idx + 1}: {path}")
        main_caps.append(cap)
    return inset_cap, main_caps


def _output_geometry(inset_cap, main_caps, cfg):
    inset_w = int(inset_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    inset_h = int(inset_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = inset_cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(inset_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if main_caps:
        main_w = int(main_caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        main_h = int(main_caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_h = max(1, int(cfg.display_height))
        video_w = max(1, int(round(main_w * video_h / max(main_h, 1))))
    else:
        video_w, video_h = inset_w, inset_h
    chart_h = max(0, int(cfg.chart_height))
    return _Geometry(inset_w, inset_h, fps, total, video_w, video_h, chart_h,
                     total_w=video_w // 2 * 2, total_h=(video_h + chart_h) // 2 * 2)


def _precompute_panel_axes(df):
    """4 個 panel 的 (ylim, yticks) 與共用 x_max。"""
    panel_ylims, panel_yticks = [], []
    for panel in PANELS:
        series_list = [df[c].dropna() for c in panel['cols'] if c in df.columns]
        if series_list:
            combined = pd.concat(series_list)
            lo, hi = combined.quantile(0.05), combined.quantile(0.95)
            margin = max((hi - lo) * 0.15, 3.0)
            y0, y1 = lo - margin, hi + margin
        else:
            y0, y1 = 0.0, 180.0
        panel_ylims.append((y0, y1))

        span = y1 - y0
        for step in [1, 2, 5, 10, 15, 20, 25, 30]:
            if span / step <= 6:
                break
        first = int(np.ceil(y0 / step)) * step
        ticks = np.arange(first, y1 + step * 0.01, step)
        panel_yticks.append(ticks[(ticks >= y0) & (ticks <= y1)])

    csv_len = len(df)
    x_max = max(df['frame'].iloc[-1], csv_len - 1) if csv_len > 0 else 100
    return panel_ylims, panel_yticks, x_max


def _build_chart_figure(df, geom, cfg, axes_spec, zh_font):
    """建立 2x2 折線圖畫布，回傳 (fig, panel_lines, panel_dots)。line/dot 之後由
    render loop 逐幀 set_data。``axes_spec`` 為 _precompute_panel_axes 的回傳。"""
    panel_ylims, panel_yticks, x_max = axes_spec
    fig, axes = plt.subplots(
        2, 2, figsize=(geom.video_w / cfg.dpi, max(geom.chart_h, 1) / cfg.dpi), dpi=cfg.dpi)
    fig.patch.set_facecolor('#ffffff')

    panel_lines, panel_dots = [], []
    for ax, panel, ylim, yticks in zip(axes.flatten(), PANELS, panel_ylims, panel_yticks):
        ax.set_facecolor('#ffffff')
        ax.set_xlim(0, x_max)
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))

        title_kw = {'fontsize': 6, 'color': 'black', 'pad': 3, 'fontweight': 'bold'}
        if zh_font:
            title_kw['fontproperties'] = zh_font
        ax.set_title(panel['title'], **title_kw)
        ax.tick_params(colors='black', labelsize=6)
        ax.spines[:].set_color('black')
        ax.grid(True, alpha=0.3)

        col_lines, col_dots = [], []
        for col, color in zip(panel['cols'], panel['colors']):
            if col not in df.columns:
                continue
            zh_label = COL_ZH.get(col, col)
            ln, = ax.plot([], [], color=color, lw=1.0, label=zh_label, alpha=1.0)
            dot, = ax.plot([], [], 'o', color=color, ms=6, zorder=5)
            col_lines.append((col, ln))
            col_dots.append((col, dot))

        if len(col_lines) > 1:
            legend_kw = dict(loc='upper right', facecolor='#ffffff',
                             edgecolor='none', labelcolor='black',
                             handlelength=1, fontsize=5.5)
            if zh_font:
                legend_font = zh_font.copy()
                legend_font.set_size(5)
                legend_kw['prop'] = legend_font
                legend_kw.pop('fontsize')
            ax.legend(**legend_kw)

        panel_lines.append(col_lines)
        panel_dots.append(col_dots)

    plt.tight_layout(pad=0.4)
    fig.canvas.draw()
    return fig, panel_lines, panel_dots


def _read_main_frame(frame_idx, main_caps, frame_map, last_main_frame):
    """依 frame_map 選相機/定位，讀一張主畫面幀。回傳 (main_frame, last_main_frame)。
    讀失敗時沿用上一張；連上一張都沒有時回傳 (None, last_main_frame)。"""
    main_cap = main_caps[0]
    if frame_map is not None and frame_idx in frame_map:
        mapped = frame_map[frame_idx]
        cam_no = mapped['cam']
        if 1 <= cam_no <= len(main_caps):
            main_cap = main_caps[cam_no - 1]
        else:
            print(f"  ⚠️  [Core.Vis] CAM{cam_no} 超出 main videos，改用 CAM1")
        main_cap.set(cv2.CAP_PROP_POS_FRAMES, mapped['source_frame'])
    ret_main, main_frame = main_cap.read()
    if ret_main:
        return main_frame, main_frame
    return last_main_frame, last_main_frame


def _chart_band_bgr(fig, geom):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    chart_bgr = cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)
    return cv2.resize(chart_bgr, (geom.video_w, geom.chart_h),
                      interpolation=cv2.INTER_LANCZOS4)


def _update_panel_data(panel_lines, panel_dots, df, csv_idx):
    for col_lines, col_dots in zip(panel_lines, panel_dots):
        for (col, ln), (_, dot) in zip(col_lines, col_dots):
            ln.set_data(df['frame'].iloc[:csv_idx + 1].values,
                        df[col].iloc[:csv_idx + 1].values)
            dot.set_data([df['frame'].iloc[csv_idx]], [df[col].iloc[csv_idx]])


def _render_overlay_video(out, inset_cap, main_caps, frame_map, df, chart, geom, cfg):
    fig, panel_lines, panel_dots = chart
    csv_len = len(df)
    frame_idx = 0
    last_main_frame = None
    while True:
        ret, inset_frame = inset_cap.read()
        if not ret:
            break

        if main_caps:
            main_frame, last_main_frame = _read_main_frame(
                frame_idx, main_caps, frame_map, last_main_frame)
            if main_frame is None:
                main_frame = inset_frame
            top = _compose_main_with_inset(
                main_frame, inset_frame, geom.video_w, geom.video_h,
                inset_height_ratio=cfg.inset_height_ratio,
                inset_margin=cfg.inset_margin,
            )
        else:
            top = _fit_to_canvas(inset_frame, geom.video_w, geom.video_h)

        _update_panel_data(panel_lines, panel_dots, df, min(frame_idx, csv_len - 1))
        combined = np.vstack([top, _chart_band_bgr(fig, geom)])
        if (combined.shape[1], combined.shape[0]) != (geom.total_w, geom.total_h):
            combined = cv2.resize(combined, (geom.total_w, geom.total_h))
        out.write(combined)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  [Core.Vis] 合併進度: {frame_idx}/{geom.total} 幀 "
                  f"({frame_idx / geom.total * 100:.0f}%)")


def add_angle_overlay(video_path, csv_path, output_path, config=None):
    """將追焦 2D 骨架影片與角度數據合併，並繪製下方的 2x2 角度變動折線圖。"""
    cfg = config or AngleOverlayConfig()
    zh_font = _zh_font()

    df, frame_map = _load_angle_dataframe(csv_path, cfg)
    inset_cap, main_caps = _open_overlay_caps(video_path, cfg.main_videos)
    geom = _output_geometry(inset_cap, main_caps, cfg)
    print(f"  [Core.Vis] 2D追焦尺寸: {geom.inset_w}×{geom.inset_h}，{geom.total} 幀")
    print(f"  [Core.Vis] 輸出總解析度: {geom.video_w}×{geom.video_h + geom.chart_h}")

    chart = _build_chart_figure(df, geom, cfg, _precompute_panel_axes(df), zh_font)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          geom.fps, (geom.total_w, geom.total_h))

    try:
        _render_overlay_video(out, inset_cap, main_caps, frame_map, df, chart, geom, cfg)
    finally:
        inset_cap.release()
        for cap in main_caps:
            cap.release()
        out.release()
        plt.close(chart[0])
    print(f"✅ [Core.Vis] 角度圖表合併影片生成完成！儲存至: {output_path}")


# ---------------------------------------------------------------------------
# CLI 參數解析器與入口（供 CLI Wrapper 直接導入呼叫）
# ---------------------------------------------------------------------------
def parse_args():
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
    return parser.parse_args()


def run_cli(args=None):
    if args is None:
        args = parse_args()

    main_videos = getattr(args, 'main_videos', None) or (
        [args.main_video] if args.main_video else [])
    add_angle_overlay(args.video, args.csv, args.output, AngleOverlayConfig(
        main_videos=main_videos,
        frame_map_path=args.frame_map,
        chart_height=args.chart_height,
        display_height=args.display_height,
        inset_height_ratio=args.inset_height_ratio,
        inset_margin=args.inset_margin,
        smooth_camera_boundary=args.smooth_camera_boundary,
        boundary_blend_frames=args.boundary_blend_frames,
        dpi=args.dpi,
    ))
