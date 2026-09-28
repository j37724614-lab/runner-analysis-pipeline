"""相機設定建立，以及跑道投影／ROI 幾何工具。

對外提供 CameraConfig、camera()、_build_camera_from_entry()；
也收留 Pass 1/Pass 2/prescan 共用的 ROI 判斷與跑道區域鄰近度函式。
"""

from dataclasses import dataclass

import cv2
import numpy as np

from core.tracking_geometry import (
    _compute_homography,
    _project_onto_track,
    _transform_point_homography,
    build_lane_world_points,
)


# -----------------------------------------------------------------------
# camera() — 快速建立相機設定的 helper function
#
# 必填：
#   video_path  str or None   影片路徑；填 None 表示此台不使用（自動跳過）
#
# 選填（有預設值）：
#   crop        (x起,y起,x終,y終)  前處理裁剪範圍，None = 不裁剪
#   switch_x    int or None        主跑者 center_x（原始座標）超過此值時切下一台
#                                  None = 跑完整段再切（最後有效台自動忽略）
#
# ── 斜線起終點模式（選用，取代 switch_x）──
# start_line  [(x1,y1), (x2,y2)]  起跑線兩端點（原始影像座標）
# end_line    [(x3,y3), (x4,y4)]  終點線兩端點（原始影像座標）
#             ← 同時填入時：switch_x 自動忽略，改由越線事件觸發切換
#             ← 候選範圍由向量投影計算，不受相機視角偏斜影響
# pre_roll_px int  起跑線前的緩衝距離（投影像素），預設 200
# -----------------------------------------------------------------------
@dataclass(frozen=True)
class CameraConfig:
    """建立追蹤相機設定所需的輸入，避免位置參數持續增加。"""

    video_path: str | None
    switch_x: int | None = None
    start_line: list | None = None
    end_line: list | None = None
    pre_roll_px: int = 200
    end_roll_px: int = 120
    homography_lane_margin_px: int = 80
    distance_m: float | None = None
    homography_matrix: object | None = None
    homography_src_points: object | None = None
    homography_dst_world: object | None = None


@dataclass(frozen=True)
class _TrackLineGeometry:
    start_mid: tuple | None = None
    end_mid: tuple | None = None
    track_dir: tuple | None = None
    pixel_span: float | None = None
    quad_roi: object | None = None


def _track_line_geometry(config: CameraConfig) -> _TrackLineGeometry:
    if config.start_line is None or config.end_line is None:
        return _TrackLineGeometry()

    start_mid = tuple(np.mean(config.start_line, axis=0))
    end_mid = tuple(np.mean(config.end_line, axis=0))
    dx = end_mid[0] - start_mid[0]
    dy = end_mid[1] - start_mid[1]
    pixel_span = (dx ** 2 + dy ** 2) ** 0.5
    track_dir = (dx / pixel_span, dy / pixel_span) if pixel_span > 0 else None
    quad_roi = np.array(
        [config.start_line[0], config.end_line[0], config.end_line[1], config.start_line[1]],
        dtype=np.float32,
    )
    return _TrackLineGeometry(start_mid, end_mid, track_dir, pixel_span, quad_roi)


def _homography_quad(config: CameraConfig, fallback):
    if config.homography_src_points is None:
        return fallback
    source = np.asarray(config.homography_src_points, dtype=np.float32)
    if len(source) >= 6:
        return np.array([source[0], source[1], source[4], source[5]], dtype=np.float32)
    if len(source) == 5:
        return np.array([source[0], source[1], source[3], source[4]], dtype=np.float32)
    return fallback


def _homography_start_x(config: CameraConfig):
    matrix = config.homography_matrix
    if matrix is None:
        return None
    if config.start_line is not None:
        world_points = [_transform_point_homography(point, matrix) for point in config.start_line]
        if all(point is not None for point in world_points):
            return float(np.mean([point[0] for point in world_points]))
        return None
    return 0.0


def camera(config: CameraConfig):
    """
    start_line / end_line（可選）：各由兩個原始影像座標點組成的斜線，
      例如 start_line=[(150, 420), (150, 780)]。
    同時填入兩者時：
      - 候選範圍由跑道方向向量投影計算
      - switch_x 自動設為 None（切換改由 end_line 越線事件觸發）
    """
    switch_x = config.switch_x
    geometry = _track_line_geometry(config)
    if geometry.start_mid is not None:
        switch_x = None
    quad_roi = _homography_quad(config, geometry.quad_roi)

    return {
        'video_path':  config.video_path,
        'switch_x':    switch_x,
        'distance_m':  config.distance_m,
        # 斜線模式欄位（舊模式均為 None）
        'start_line':  config.start_line,
        'end_line':    config.end_line,
        'start_mid':   geometry.start_mid,
        'end_mid':     geometry.end_mid,
        'track_dir':   geometry.track_dir,
        'pixel_span':  geometry.pixel_span,
        'quad_roi':    quad_roi,
        'homography_lane_margin_px': config.homography_lane_margin_px,
        'track_roi':   {'start_mid': geometry.start_mid,
                        'track_dir': geometry.track_dir,
                        'pixel_span': geometry.pixel_span,
                        'pre_roll_px': config.pre_roll_px,
                        'end_roll_px': config.end_roll_px}
                       if geometry.start_mid is not None and geometry.track_dir is not None else None,
        'H_matrix':    config.homography_matrix,
        'homography_start_x': _homography_start_x(config),
        'homography_src_points': (
            np.asarray(config.homography_src_points, dtype=np.float32)
            if config.homography_src_points is not None else None
        ),
        'homography_dst_world': (
            np.asarray(config.homography_dst_world, dtype=np.float32)
            if config.homography_dst_world is not None else None
        ),
    }
# =======================================================================
# 以下為程式邏輯，一般不需修改
# =======================================================================

def _build_camera_from_entry(entry):
    """將 config dict 的單台相機 entry 轉換成 camera dict。"""
    # 解析 start_line / end_line：YAML 格式為 [[x1,y1],[x2,y2]]
    sl = entry.get('start_line')
    el = entry.get('end_line')
    start_line = [tuple(p) for p in sl] if sl else None
    end_line   = [tuple(p) for p in el] if el else None
    H_matrix = None
    src_pts = entry.get('homography_src_points') or entry.get('src_points')
    dst_pts = (
        entry.get('homography_dst_world') or
        entry.get('homography_dst_points_world')
    )
    if src_pts is not None and dst_pts is None and entry.get('start_meter') is not None:
        dst_pts = build_lane_world_points(
            float(entry['start_meter']),
            num_points=len(src_pts),
        )
    if src_pts is not None and dst_pts is not None:
        H_matrix, _ = _compute_homography(src_pts, dst_pts)
    distance_m = entry.get('distance_m')
    if distance_m is None and src_pts and entry.get('start_meter') is not None:
        distance_m = 20.0

    return camera(CameraConfig(
        video_path=entry.get('video_path'),
        switch_x=entry.get('switch_x'),
        start_line=start_line,
        end_line=end_line,
        pre_roll_px=int(entry.get('pre_roll_px', 200)),
        end_roll_px=int(entry.get('end_roll_px', 120)),
        homography_lane_margin_px=int(entry.get('homography_lane_margin_px', 80)),
        distance_m=distance_m,
        homography_matrix=H_matrix,
        homography_src_points=src_pts,
        homography_dst_world=dst_pts,
    ))


def _project_and_check_track_roi(ground_pt, track_roi):
    """依跑道投影判斷地面點是否落在 track_roi 的有效範圍內。

    track_roi 為 None 時視為無限制。回傳 (passes, proj_px)；
    track_roi 為 None 時 proj_px 為 None。供 Pass 1、Pass 2 與 prescan
    共用同一套「-pre_roll <= proj <= pixel_span+end_roll」判斷式。
    """
    if track_roi is None:
        return True, None
    proj_px = _project_onto_track(
        ground_pt, track_roi['start_mid'], track_roi['track_dir'],
    )
    pre_roll = track_roi.get('pre_roll_px', 0)
    end_roll = track_roi.get('end_roll_px', 0)
    passes = -pre_roll <= proj_px <= track_roi['pixel_span'] + end_roll
    return passes, proj_px


def _point_track_area_proximity(point, quad_roi, margin_px=120):
    """
    評估一個點是否貼近起點線與終點線圍出的跑道四邊形。

    回傳值範圍 0..1：
      - 點在四邊形內：1
      - 點在四邊形外：依離邊界距離遞減
    """
    if point is None or quad_roi is None:
        return 1.0

    margin_px = max(float(margin_px), 1.0)
    signed_dist = cv2.pointPolygonTest(
        np.asarray(quad_roi, dtype=np.float32),
        (float(point[0]), float(point[1])),
        True,
    )
    if signed_dist >= 0:
        return 1.0
    return max(0.0, 1.0 + signed_dist / margin_px)


