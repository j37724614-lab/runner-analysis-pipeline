"""共用的跑道投影 / Homography 幾何函式。

這些函式原本在 core/tracking.py 與 core/tracker_impl.py 各自獨立實作一份
（`_compute_homography` 兩邊逐字元相同，其餘幾個也只是文件字串或呼叫介面
不同，核心邏輯完全一致）。抽成這個模組讓兩邊改用同一份實作，避免日後各自
修改導致行為悄悄分岔。

只依賴 numpy 和 cv2，不匯入 YOLO/torch，供不需要追蹤模型的呼叫端（測試、
工具腳本）以較低成本取用。
"""
import cv2
import numpy as np

LANE_WIDTH_M = 1.22


def _project_onto_track(point, start_mid, track_dir):
    """將 point 投影到 track_dir 方向，回傳從 start_mid 起的有號像素距離。"""
    dx = point[0] - start_mid[0]
    dy = point[1] - start_mid[1]
    return dx * track_dir[0] + dy * track_dir[1]


def _project_point_to_track_line(point, start_mid, track_dir):
    """將 point 投影到 start_mid + track_dir 定義的跑道中心線上。"""
    proj = _project_onto_track(point, start_mid, track_dir)
    return (
        start_mid[0] + track_dir[0] * proj,
        start_mid[1] + track_dir[1] * proj,
    )


def _compute_homography(src_points, dst_points_world):
    """根據影像點與世界座標點計算 Homography。"""
    if src_points is None or dst_points_world is None:
        return None, None

    src = np.asarray(src_points, dtype=np.float32)
    dst = np.asarray(dst_points_world, dtype=np.float32)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(
            "homography_src_points / homography_dst_world 必須是 shape=(N,2) 且點數一致"
        )
    if len(src) < 4:
        raise ValueError("Homography 至少需要 4 組對應點")

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("Homography 計算失敗，請檢查點位是否共線或順序是否對應")
    return H, mask


def build_lane_world_points(start_meter, num_points=5, lane_width=LANE_WIDTH_M):
    """建立單一跑道區段的世界座標；支援 5 點或 6 點 Homography 標定。"""
    if num_points == 5:
        return np.float32([
            [start_meter + 0.0, lane_width],
            [start_meter + 0.0, 0.0],
            [start_meter + 10.0, 0.0],
            [start_meter + 20.0, 0.0],
            [start_meter + 20.0, lane_width],
        ])

    if num_points == 6:
        return np.float32([
            [start_meter + 0.0, lane_width],
            [start_meter + 0.0, 0.0],
            [start_meter + 10.0, 0.0],
            [start_meter + 10.0, lane_width],
            [start_meter + 20.0, 0.0],
            [start_meter + 20.0, lane_width],
        ])

    raise ValueError("src_points 目前只支援 5 點或 6 點 Homography 標定")


def _transform_point_homography(point, H):
    """把單一影像點轉到世界座標，回傳 (xw, yw)。"""
    if H is None:
        return None
    pts = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pts, H)
    return tuple(mapped[0, 0])


def _bbox_bottom_center(bbox):
    """回傳 bbox 的底部中心點，較接近跑者落地點。"""
    bx1, _by1, bx2, by2 = bbox
    return ((bx1 + bx2) / 2.0, float(by2))
