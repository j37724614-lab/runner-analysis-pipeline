"""固定尺寸裁剪與插值補幀所需的畫面處理函式。"""

import cv2
import numpy as np

from core import tracking as _tracking


def _interpolate_bbox(left_bbox, right_bbox, ratio):
    """用左右兩個有效 bbox 線性插值出中間 bbox。"""
    left = np.asarray(left_bbox, dtype=float)
    right = np.asarray(right_bbox, dtype=float)
    return tuple(np.rint(left + ratio * (right - left)).astype(int))


def _fixed_size_crop(img, bx1, by1, bx2, by2):
    """以 bbox 中心為基準，裁出 _tracking.CROP_WIDTH x _tracking.CROP_HEIGHT 固定尺寸畫面。
    回傳 (crop_frame, bbox_in_crop, c1x, c1y)；c1x/c1y 為裁剪視窗左上角在 img 座標系中的位置。
    """
    h_img, w_img = img.shape[:2]
    cx = int((bx1 + bx2) / 2)
    cy = int((by1 + by2) / 2)
    c1x = cx - _tracking.CROP_WIDTH // 2
    c1y = cy - _tracking.CROP_HEIGHT // 2
    c2x = c1x + _tracking.CROP_WIDTH
    c2y = c1y + _tracking.CROP_HEIGHT

    if c1x < 0:
        c1x, c2x = 0, _tracking.CROP_WIDTH
    elif c2x > w_img:
        c2x, c1x = w_img, w_img - _tracking.CROP_WIDTH
    if c1y < 0:
        c1y, c2y = 0, _tracking.CROP_HEIGHT
    elif c2y > h_img:
        c2y, c1y = h_img, h_img - _tracking.CROP_HEIGHT
    c1x = max(0, c1x)
    c1y = max(0, c1y)
    c2x = min(c2x, w_img)
    c2y = min(c2y, h_img)

    crop_frame = img[c1y:c2y, c1x:c2x]
    bbox_in_crop = (
        int(np.clip(bx1 - c1x, 0, _tracking.CROP_WIDTH - 1)),
        int(np.clip(by1 - c1y, 0, _tracking.CROP_HEIGHT - 1)),
        int(np.clip(bx2 - c1x, 0, _tracking.CROP_WIDTH - 1)),
        int(np.clip(by2 - c1y, 0, _tracking.CROP_HEIGHT - 1)),
    )

    if crop_frame.shape[:2] != (_tracking.CROP_HEIGHT, _tracking.CROP_WIDTH):
        if crop_frame.size > 0:
            crop_frame = cv2.resize(crop_frame, (_tracking.CROP_WIDTH, _tracking.CROP_HEIGHT),
                                    interpolation=cv2.INTER_LINEAR)
        else:
            crop_frame = np.zeros((_tracking.CROP_HEIGHT, _tracking.CROP_WIDTH, 3), dtype=np.uint8)

    return crop_frame, bbox_in_crop, c1x, c1y


def _crop_from_bbox(img, bbox, label_interpolated=False, track_id=None):
    """
    使用原始影格座標的 bbox 重新裁切一幀，供漏偵幀的插值補幀使用。

    回傳 (crop_frame, bbox_in_crop, off_x, off_y)。
    """
    bx1, by1, bx2, by2 = map(int, bbox)
    h_img, w_img = img.shape[:2]
    bx1 = int(np.clip(bx1, 0, max(w_img - 1, 0)))
    bx2 = int(np.clip(bx2, 0, max(w_img - 1, 0)))
    by1 = int(np.clip(by1, 0, max(h_img - 1, 0)))
    by2 = int(np.clip(by2, 0, max(h_img - 1, 0)))
    if bx2 <= bx1 or by2 <= by1:
        return None, None, None, None

    if _tracking.SHOW_OVERLAY and _tracking.DRAW_BBOX_OVERLAY:
        color = (255, 0, 255) if label_interpolated else (0, 255, 0)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 2)
        label_parts = []
        if track_id is not None:
            label_parts.append(f"ID {track_id}")
        if label_interpolated:
            label_parts.append("interp")
        if label_parts:
            label = " ".join(label_parts)
            label_y = max(by1 - 8, 20)
            cv2.putText(
                img,
                label,
                (bx1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                label,
                (bx1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

    crop_frame, bbox_in_crop, c1x_out, c1y_out = _fixed_size_crop(img, bx1, by1, bx2, by2)
    return crop_frame, bbox_in_crop, c1x_out, c1y_out


