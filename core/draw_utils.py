"""Shared image-drawing helpers.

``draw_dashed_line`` / ``draw_text_bgr`` / ``get_font`` were each copy-pasted
into several modules (core.tracking, core.tracker_impl, core.overlay,
scripts.analysis.ankle_step_stride, scripts.tracking.track_crop_roi). They now
live here once.
"""

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.utils import get_font_path


def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=12, gap_len=8):
    """Draw a dashed straight line from ``pt1`` to ``pt2`` on ``img`` in place."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    drawing = True
    while pos < length:
        seg = dash_len if drawing else gap_len
        end_pos = min(pos + seg, length)
        if drawing:
            x1 = int(pt1[0] + ux * pos)
            y1 = int(pt1[1] + uy * pos)
            x2 = int(pt1[0] + ux * end_pos)
            y2 = int(pt1[1] + uy * end_pos)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        pos = end_pos
        drawing = not drawing


def get_font(size=28, font_path=None):
    """Load a PIL truetype font; return None (safe degradation) when it can't
    be read. ``font_path`` defaults to the project's bundled Chinese font."""
    font_path = font_path or get_font_path()
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            return None
    return None


def draw_text_bgr(img, text, org, font=None, color=(255, 255, 255), thickness=2,
                  outline_color=(0, 0, 0)):
    """Draw CJK-capable text on a BGR image; ``org`` is the top-left corner.
    Falls back to cv2.putText (no CJK) when no font is available."""
    if not text:
        return img

    font = font or get_font(size=28)
    if font is None:
        cv2.putText(img, str(text), org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)
        return img

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    x, y = int(org[0]), int(org[1])

    if outline_color is not None:
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), str(text), font=font, fill=outline_color[::-1])

    draw.text((x, y), str(text), font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
