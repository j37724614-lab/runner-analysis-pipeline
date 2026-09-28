"""Optional visual diagnostics for the tracking pipeline.

The production tracking path depends only on ``TrackingOverviewWriter``.
When disabled, it does not copy frames, create a video writer, or write files.
"""

from __future__ import annotations

import cv2
import numpy as np


class TrackingOverviewWriter:
    """Write full-frame runner/track overlays when explicitly enabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        frame_map_path: str | None,
        camera_index: int,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        self._writer = None
        self.output_path = None
        if not enabled or not frame_map_path:
            return

        self.output_path = frame_map_path.replace(
            "_frame_map.csv",
            f"_cam{camera_index + 1}_overview.mp4",
        )
        codec = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self._writer = cv2.VideoWriter(
            self.output_path,
            codec,
            fps,
            frame_size,
        )

    def write(
        self,
        image,
        *,
        tracks: dict,
        selected_id: int | None,
        crop_offset: tuple[int, int],
        quadrilateral,
    ) -> None:
        """Copy and annotate one source frame; do nothing when disabled."""
        if self._writer is None:
            return

        overview = image.copy()
        if quadrilateral is not None:
            cv2.polylines(
                overview,
                [quadrilateral.reshape(-1, 1, 2).astype(np.int32)],
                True,
                (255, 200, 50),
                2,
            )

        offset_x, offset_y = crop_offset
        for track_id, track in tracks.items():
            if track["frames_since_detected"] != 0:
                continue
            x1, y1, x2, y2 = track["bbox"]
            x1 += offset_x
            y1 += offset_y
            x2 += offset_x
            y2 += offset_y
            color = (0, 255, 0) if track_id == selected_id else (0, 165, 255)
            cv2.rectangle(overview, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id}"
            label_y = max(y1 - 8, 20)
            cv2.putText(
                overview,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                overview,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        self._writer.write(overview)

    def close(self) -> None:
        """Finalize the optional MP4 and report its path."""
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None
        print(f"  Overview:  {self.output_path}")
