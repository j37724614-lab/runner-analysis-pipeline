"""Temporal HRNet crop stabilization shared by video and streaming inference."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass
class _CropState:
    raw: tuple[float, float, float, float]
    stable: tuple[float, float, float, float]
    context: object


class TemporalPoseCropStabilizer:
    """Follow bbox centres quickly while allowing bbox size to shrink slowly."""

    def __init__(self, center_alpha=0.70, shrink_retention=0.90):
        self.center_alpha = float(center_alpha)
        self.shrink_retention = float(shrink_retention)
        self._state = None

    def reset(self):
        self._state = None

    def update(self, bbox, frame_size, context=None):
        current = tuple(float(value) for value in bbox[:4])
        width = current[2] - current[0]
        height = current[3] - current[1]
        if width <= 0 or height <= 0:
            self.reset()
            return list(current)

        previous = self._state
        if (
            previous is None
            or not self._same_sequence(previous.context, context)
            or not self._plausible(previous.raw, current)
        ):
            self._state = _CropState(current, current, context)
            return list(current)

        alpha = min(1.0, max(0.0, self.center_alpha))
        retention = min(1.0, max(0.0, self.shrink_retention))
        current_center = self._center(current)
        stable_center = self._center(previous.stable)
        center_x = alpha * current_center[0] + (1.0 - alpha) * stable_center[0]
        center_y = alpha * current_center[1] + (1.0 - alpha) * stable_center[1]
        stable_width = max(width, self._width(previous.stable) * retention)
        stable_height = max(height, self._height(previous.stable) * retention)
        frame_width, frame_height = map(float, frame_size)
        stable = (
            max(0.0, center_x - stable_width * 0.5),
            max(0.0, center_y - stable_height * 0.5),
            min(frame_width, center_x + stable_width * 0.5),
            min(frame_height, center_y + stable_height * 0.5),
        )
        self._state = _CropState(current, stable, context)
        return list(stable)

    @staticmethod
    def _same_sequence(previous, current):
        if previous is None or current is None:
            return True
        previous_camera, previous_track, previous_frame = previous
        current_camera, current_track, current_frame = current
        return (
            previous_camera == current_camera
            and previous_track == current_track
            and current_frame == previous_frame + 1
        )

    @classmethod
    def _plausible(cls, previous, current):
        previous_center = cls._center(previous)
        current_center = cls._center(current)
        reference = max(
            cls._width(previous), cls._height(previous),
            cls._width(current), cls._height(current),
        )
        return hypot(
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        ) <= reference * 0.75

    @staticmethod
    def _center(bbox):
        return ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)

    @staticmethod
    def _width(bbox):
        return bbox[2] - bbox[0]

    @staticmethod
    def _height(bbox):
        return bbox[3] - bbox[1]
