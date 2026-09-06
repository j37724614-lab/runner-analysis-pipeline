"""Render an equal-scale schematic runway review without warping camera pixels.

Only ground-plane coordinates are transformed by the homography. The runway is
drawn from scratch, landing contacts are connected in time order, and the
runner is represented by a fixed-size glyph anchored at the projected ground
position. This avoids stretching above-ground subjects and background pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


class UnicodeTextSpec(NamedTuple):
    text: str
    position: tuple[int, int]
    font_size: int
    color_bgr: tuple[int, int, int]
    rotate_degrees: int = 0


class RenderRequest(NamedTuple):
    video_path: Path
    output_path: Path
    positions: dict[int, tuple[float, float]]
    events: list[dict]
    size: tuple[int, int]
    bounds: tuple[float, float, float, float]
    px_per_meter: float
    padding_px: int
    camera_segments: list[tuple[int, float, float]] | None = None
    title: str = "Equal-scale top-down"
    chart_axes: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", required=True, help="Source video used for timing metadata"
    )
    calibration = parser.add_mutually_exclusive_group(required=True)
    calibration.add_argument("--points-json", help="One camera's image/world controls")
    calibration.add_argument(
        "--calibrations-json",
        help="All camera controls for a stitched full-trial review",
    )
    parser.add_argument(
        "--metrics-csv", required=True, help="Per-frame runner ground-point metrics"
    )
    parser.add_argument(
        "--step-events-csv", required=True, help="Detected landing/contact events"
    )
    parser.add_argument("--output", required=True, help="Destination H.264 MP4")
    parser.add_argument(
        "--camera-index", type=int, default=0, help="Zero-based camera index"
    )
    parser.add_argument(
        "--px-per-meter", type=float, default=75.0, help="Equal X/Y output scale"
    )
    parser.add_argument("--padding-px", type=int, default=60)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_homography(points_path: Path) -> tuple[np.ndarray, np.ndarray]:
    points = json.loads(points_path.read_text(encoding="utf-8")).get("points", [])
    if len(points) < 4:
        raise ValueError("points JSON must contain at least four controls")
    image = np.asarray([[point["x"], point["y"]] for point in points], dtype=np.float64)
    world = np.asarray(
        [[point["world_x_m"], point["world_y_m"]] for point in points],
        dtype=np.float64,
    )
    matrix, _ = cv2.findHomography(image, world, method=0)
    if matrix is None:
        raise RuntimeError("cv2.findHomography failed")
    return matrix, world


def load_calibrations(
    calibrations_path: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load per-camera homographies used to build one continuous race path."""
    payload = json.loads(calibrations_path.read_text(encoding="utf-8"))
    calibrations = {}
    for camera in payload.get("cameras", []):
        camera_index = int(camera["camera_index"])
        image = np.asarray(camera.get("image_points", []), dtype=np.float64)
        world = np.asarray(camera.get("world_points", []), dtype=np.float64)
        if image.shape[0] < 4 or image.shape != world.shape or image.shape[1:] != (2,):
            raise ValueError(
                f"camera {camera_index} must contain matching image/world controls"
            )
        matrix, _ = cv2.findHomography(image, world, method=0)
        if matrix is None:
            raise RuntimeError(f"cv2.findHomography failed for camera {camera_index}")
        calibrations[camera_index] = (matrix, world)
    if not calibrations:
        raise ValueError("calibrations JSON must contain at least one camera")
    return calibrations


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(values, matrix).reshape(-1, 2)


def load_runner_positions(
    metrics_path: Path,
    matrix: np.ndarray,
    camera_index: int,
) -> dict[int, tuple[float, float]]:
    selected = []
    expected_cam = camera_index + 1  # metrics.csv uses one-based camera IDs.
    for row in read_csv(metrics_path):
        if row.get("cam") and int(row["cam"]) != expected_cam:
            continue
        if not row.get("image_point_x") or not row.get("image_point_y"):
            continue
        selected.append(row)
    if not selected:
        return {}
    pixels = np.asarray(
        [
            [float(row["image_point_x"]), float(row["image_point_y"])]
            for row in selected
        ],
        dtype=np.float64,
    )
    world = project(matrix, pixels)
    return {
        int(row["source_frame"]): (float(point[0]), float(point[1]))
        for row, point in zip(selected, world)
    }


def load_landing_events(
    events_path: Path,
    matrix: np.ndarray,
    camera_index: int,
) -> list[dict]:
    events = []
    for row in read_csv(events_path):
        if row.get("cam") and int(row["cam"]) != camera_index:
            continue
        if row.get("contact_valid", "").lower() != "true":
            continue
        if not row.get("contact_x") or not row.get("contact_y"):
            continue
        world = project(
            matrix,
            np.asarray(
                [[float(row["contact_x"]), float(row["contact_y"])]], dtype=np.float64
            ),
        )[0]
        events.append(
            {
                "frame": int(row["orig_frame"]),
                "step": int(row["step_index"]),
                "foot": row["foot"],
                "world": (float(world[0]), float(world[1])),
            }
        )
    return sorted(events, key=lambda event: event["frame"])


def _load_global_runner_positions(
    metrics_path: Path,
    calibrations: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[int, tuple[float, float]], dict[int, list[float]]]:
    positions: dict[int, tuple[float, float]] = {}
    offset_samples: dict[int, list[float]] = {index: [] for index in calibrations}
    for row in read_csv(metrics_path):
        if not row.get("cam") or not row.get("absolute_frame") or not row.get("dist_m"):
            continue
        if not row.get("image_point_x") or not row.get("image_point_y"):
            continue
        camera_index = int(row["cam"]) - 1
        calibration = calibrations.get(camera_index)
        if calibration is None:
            continue
        pixel = np.asarray(
            [[float(row["image_point_x"]), float(row["image_point_y"])]],
            dtype=np.float64,
        )
        global_x = float(row["dist_m"])
        if not np.isfinite(pixel).all() or not math.isfinite(global_x):
            continue
        local = project(
            calibration[0],
            pixel,
        )[0]
        if not np.isfinite(local).all():
            continue
        positions[int(row["absolute_frame"])] = (global_x, float(local[1]))
        offset_samples[camera_index].append(global_x - float(local[0]))
    return positions, offset_samples


def _estimate_camera_offsets(
    calibrations: dict[int, tuple[np.ndarray, np.ndarray]],
    offset_samples: dict[int, list[float]],
) -> dict[int, float]:
    camera_offsets: dict[int, float] = {}
    fallback_x = 0.0
    for camera_index in sorted(calibrations):
        world = calibrations[camera_index][1]
        local_min_x = float(world[:, 0].min())
        local_max_x = float(world[:, 0].max())
        samples = offset_samples[camera_index]
        camera_offsets[camera_index] = (
            float(np.median(np.asarray(samples, dtype=np.float64)))
            if samples
            else fallback_x - local_min_x
        )
        fallback_x += local_max_x - local_min_x
    return camera_offsets


def _global_camera_geometry(
    calibrations: dict[int, tuple[np.ndarray, np.ndarray]],
    camera_offsets: dict[int, float],
) -> tuple[np.ndarray, list[tuple[int, float, float]]]:
    global_controls = []
    camera_segments = []
    for camera_index in sorted(calibrations):
        world = calibrations[camera_index][1]
        offset = camera_offsets[camera_index]
        adjusted = world.copy()
        adjusted[:, 0] += offset
        global_controls.extend(adjusted.tolist())
        camera_segments.append(
            (camera_index, float(adjusted[:, 0].min()), float(adjusted[:, 0].max()))
        )
    return np.asarray(global_controls, dtype=np.float64), camera_segments


def _load_global_landing_events(
    events_path: Path,
    calibrations: dict[int, tuple[np.ndarray, np.ndarray]],
    camera_offsets: dict[int, float],
) -> list[dict]:
    events = []
    for row in read_csv(events_path):
        if row.get("contact_valid", "").lower() != "true":
            continue
        if (
            not row.get("contact_x")
            or not row.get("contact_y")
            or not row.get("seq_frame")
        ):
            continue
        camera_index = int(row.get("cam", -1))
        calibration = calibrations.get(camera_index)
        if calibration is None:
            continue
        pixel = np.asarray(
            [[float(row["contact_x"]), float(row["contact_y"])]],
            dtype=np.float64,
        )
        if not np.isfinite(pixel).all():
            continue
        local = project(
            calibration[0],
            pixel,
        )[0]
        if not np.isfinite(local).all():
            continue
        events.append(
            {
                "frame": int(row["seq_frame"]),
                "step": int(row["step_index"]),
                "foot": row["foot"],
                "world": (
                    float(local[0]) + camera_offsets[camera_index],
                    float(local[1]),
                ),
            }
        )
    return sorted(events, key=lambda event: event["frame"])


def load_full_trial_data(
    metrics_path: Path,
    events_path: Path,
    calibrations: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[
    dict[int, tuple[float, float]],
    list[dict],
    np.ndarray,
    list[tuple[int, float, float]],
    dict[int, float],
]:
    """Join camera-local homographies on the global distance/timeline axes."""
    positions, offset_samples = _load_global_runner_positions(
        metrics_path,
        calibrations,
    )
    camera_offsets = _estimate_camera_offsets(calibrations, offset_samples)
    global_controls, camera_segments = _global_camera_geometry(
        calibrations,
        camera_offsets,
    )
    events = _load_global_landing_events(
        events_path,
        calibrations,
        camera_offsets,
    )
    return (
        positions,
        events,
        global_controls,
        camera_segments,
        camera_offsets,
    )


def output_geometry(
    world_controls: np.ndarray,
    px_per_meter: float,
    padding_px: int,
) -> tuple[tuple[int, int], tuple[float, float, float, float]]:
    if px_per_meter <= 0:
        raise ValueError("--px-per-meter must be greater than zero")
    if padding_px < 0:
        raise ValueError("--padding-px cannot be negative")
    min_x, min_y = world_controls.min(axis=0)
    max_x, max_y = world_controls.max(axis=0)
    width = math.ceil((max_x - min_x) * px_per_meter + 2 * padding_px)
    height = math.ceil((max_y - min_y) * px_per_meter + 2 * padding_px)
    width += width % 2
    height += height % 2
    return (width, height), (float(min_x), float(max_x), float(min_y), float(max_y))


def world_to_pixel(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
    px_per_meter: float,
    padding_px: int,
) -> tuple[int, int]:
    min_x, _, min_y, _ = bounds
    return (
        padding_px + round((point[0] - min_x) * px_per_meter),
        padding_px + round((point[1] - min_y) * px_per_meter),
    )


def draw_runway(
    size: tuple[int, int],
    bounds: tuple[float, float, float, float],
    px_per_meter: float,
    padding_px: int,
    camera_segments: list[tuple[int, float, float]] | None = None,
) -> np.ndarray:
    width, height = size
    min_x, max_x, min_y, max_y = bounds
    canvas = np.full((height, width, 3), (28, 36, 31), dtype=np.uint8)
    start = (padding_px, padding_px)
    end = world_to_pixel((max_x, max_y), bounds, px_per_meter, padding_px)
    cv2.rectangle(canvas, start, end, (64, 74, 177), -1)
    cv2.rectangle(canvas, start, end, (245, 245, 245), 3, cv2.LINE_AA)
    center_y = world_to_pixel(
        (min_x, (min_y + max_y) / 2), bounds, px_per_meter, padding_px
    )[1]
    cv2.line(
        canvas,
        (start[0], center_y),
        (end[0], center_y),
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    for metres in range(math.ceil(min_x), math.floor(max_x) + 1):
        x = world_to_pixel((metres, min_y), bounds, px_per_meter, padding_px)[0]
        tick = 12 if metres % 5 == 0 else 6
        cv2.line(
            canvas, (x, end[1]), (x, end[1] - tick), (240, 240, 240), 2, cv2.LINE_AA
        )
        if metres % 5 == 0:
            label_x = min(x + 4, width - 48)
            cv2.putText(
                canvas,
                f"{metres} m",
                (label_x, start[1] + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )
    for segment_index, (camera_index, segment_start, _) in enumerate(
        camera_segments or []
    ):
        x = world_to_pixel((segment_start, min_y), bounds, px_per_meter, padding_px)[0]
        if segment_index:
            for y in range(start[1], end[1], 14):
                cv2.line(canvas, (x, y), (x, min(y + 7, end[1])), (225, 225, 225), 1)
        cv2.putText(
            canvas,
            f"CAM {camera_index + 1}",
            (min(x + 6, width - 74), end[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _draw_unicode_text(
    canvas: np.ndarray,
    spec: UnicodeTextSpec,
) -> None:
    """Draw Traditional Chinese labels with the system Noto CJK font."""
    font = ImageFont.truetype(_CJK_FONT, spec.font_size)
    color_rgb = (
        spec.color_bgr[2],
        spec.color_bgr[1],
        spec.color_bgr[0],
        255,
    )
    if spec.rotate_degrees:
        left, top, right, bottom = font.getbbox(spec.text)
        label = Image.new("RGBA", (right - left + 8, bottom - top + 8), (0, 0, 0, 0))
        ImageDraw.Draw(label).text(
            (4 - left, 4 - top),
            spec.text,
            font=font,
            fill=color_rgb,
        )
        label = label.rotate(
            spec.rotate_degrees,
            expand=True,
            resample=Image.Resampling.BICUBIC,
        )
        base = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).convert("RGBA")
        base.alpha_composite(label, spec.position)
        canvas[:] = cv2.cvtColor(np.asarray(base.convert("RGB")), cv2.COLOR_RGB2BGR)
        return
    base = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(base).text(
        spec.position,
        spec.text,
        font=font,
        fill=color_rgb[:3],
    )
    canvas[:] = cv2.cvtColor(np.asarray(base), cv2.COLOR_RGB2BGR)


def _draw_step_axis(
    canvas: np.ndarray,
    events: list[dict],
    bounds: tuple[float, float, float, float],
    px_per_meter: float,
    padding_px: int,
) -> None:
    height, width = canvas.shape[:2]
    min_x, max_x, min_y, max_y = bounds
    track_start = world_to_pixel((min_x, min_y), bounds, px_per_meter, padding_px)
    track_end = world_to_pixel((max_x, max_y), bounds, px_per_meter, padding_px)
    color = (235, 235, 235)

    x_axis_y = track_end[1] + 18
    cv2.line(canvas, (track_start[0], x_axis_y), (track_end[0], x_axis_y), color, 1)
    previous_x = None
    for index, event in enumerate(sorted(events, key=lambda value: value["frame"])):
        x = world_to_pixel(event["world"], bounds, px_per_meter, padding_px)[0]
        cv2.line(canvas, (x, x_axis_y - 4), (x, x_axis_y + 4), color, 1)
        shift = 0
        if previous_x is not None and abs(x - previous_x) < 13:
            shift = 8 if index % 2 else -8
        previous_x = x
        cv2.putText(
            canvas,
            str(event["step"]),
            (x - 4 + shift, x_axis_y + 17 + (index % 2) * 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    _draw_unicode_text(
        canvas,
        UnicodeTextSpec(
            text="步序",
            position=(max(0, width // 2 - 22), height - 30),
            font_size=19,
            color_bgr=color,
        ),
    )


def _draw_lateral_axis(
    canvas: np.ndarray,
    bounds: tuple[float, float, float, float],
    px_per_meter: float,
    padding_px: int,
) -> None:
    min_x, max_x, min_y, max_y = bounds
    track_start = world_to_pixel((min_x, min_y), bounds, px_per_meter, padding_px)
    track_end = world_to_pixel((max_x, max_y), bounds, px_per_meter, padding_px)
    color = (235, 235, 235)
    y_axis_x = track_start[0] - 12
    cv2.line(canvas, (y_axis_x, track_start[1]), (y_axis_x, track_end[1]), color, 1)
    for metres in range(math.ceil(min_y), math.floor(max_y) + 1):
        y = world_to_pixel((min_x, metres), bounds, px_per_meter, padding_px)[1]
        cv2.line(canvas, (y_axis_x - 4, y), (y_axis_x + 4, y), color, 1)
        cv2.putText(
            canvas,
            f"{metres:g}",
            (2, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    label = "跑道橫向位置 (m)"
    font = ImageFont.truetype(_CJK_FONT, 17)
    left, _, right, _ = font.getbbox(label)
    rotated_height = right - left + 8
    _draw_unicode_text(
        canvas,
        UnicodeTextSpec(
            text=label,
            position=(
                18,
                max(0, (track_start[1] + track_end[1] - rotated_height) // 2),
            ),
            font_size=17,
            color_bgr=color,
            rotate_degrees=90,
        ),
    )


def draw_chart_axes(
    canvas: np.ndarray,
    events: list[dict],
    bounds: tuple[float, float, float, float],
    px_per_meter: float,
    padding_px: int,
) -> None:
    """Add a real step-order axis and Homography lateral-position axis."""
    _draw_step_axis(canvas, events, bounds, px_per_meter, padding_px)
    _draw_lateral_axis(canvas, bounds, px_per_meter, padding_px)


def draw_footprint(
    canvas: np.ndarray,
    point: tuple[int, int],
    foot: str,
    color: tuple[int, int, int],
) -> None:
    x, y = point
    lateral_offset = -4 if foot == "left" else 4
    cv2.ellipse(
        canvas, (x, y + lateral_offset), (10, 5), 0, 0, 360, color, -1, cv2.LINE_AA
    )
    cv2.circle(canvas, (x + 9, y + lateral_offset), 3, color, -1, cv2.LINE_AA)


def draw_runner(canvas: np.ndarray, point: tuple[int, int]) -> None:
    """Draw a fixed-size glyph anchored at the ground point; never warp it."""
    x, ground_y = point
    accent = (85, 230, 255)
    white = (245, 245, 245)
    cv2.circle(canvas, (x, ground_y - 35), 7, white, -1, cv2.LINE_AA)
    cv2.line(canvas, (x, ground_y - 27), (x, ground_y - 10), white, 5, cv2.LINE_AA)
    cv2.line(
        canvas, (x, ground_y - 21), (x - 12, ground_y - 14), accent, 4, cv2.LINE_AA
    )
    cv2.line(
        canvas, (x, ground_y - 21), (x + 13, ground_y - 25), accent, 4, cv2.LINE_AA
    )
    cv2.line(canvas, (x, ground_y - 10), (x - 10, ground_y), accent, 5, cv2.LINE_AA)
    cv2.line(canvas, (x, ground_y - 10), (x + 12, ground_y - 3), accent, 5, cv2.LINE_AA)
    cv2.circle(canvas, (x, ground_y), 4, (255, 255, 255), -1, cv2.LINE_AA)


class _SchematicRenderer:
    def __init__(self, request: RenderRequest) -> None:
        self.request = request
        self.fps, self.frame_count = self._video_metadata()
        self.base_canvas = self._base_canvas()

    def render(self) -> dict:
        self.request.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="schematic-topdown-") as temp_dir:
            temp_video = Path(temp_dir) / "preview.mp4"
            self._write_video(temp_video)
            self._transcode(temp_video)
        return self._summary()

    def _video_metadata(self) -> tuple[float, int]:
        capture = cv2.VideoCapture(str(self.request.video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.request.video_path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 60.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        return fps, frame_count

    def _base_canvas(self) -> np.ndarray:
        canvas = draw_runway(
            self.request.size,
            self.request.bounds,
            self.request.px_per_meter,
            self.request.padding_px,
            camera_segments=self.request.camera_segments,
        )
        if self.request.chart_axes:
            draw_chart_axes(
                canvas,
                self.request.events,
                self.request.bounds,
                self.request.px_per_meter,
                self.request.padding_px,
            )
        return canvas

    def _write_video(self, temp_video: Path) -> None:
        writer = cv2.VideoWriter(
            str(temp_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.request.size,
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot open schematic VideoWriter")
        try:
            for frame_index in range(self.frame_count):
                writer.write(self._render_frame(frame_index))
        finally:
            writer.release()

    def _render_frame(self, frame_index: int) -> np.ndarray:
        canvas = self.base_canvas.copy()
        prior_events = [
            event for event in self.request.events if event["frame"] <= frame_index
        ]
        landing_pixels = [
            self._world_to_pixel(event["world"]) for event in prior_events
        ]
        if len(landing_pixels) > 1:
            cv2.polylines(
                canvas,
                [np.asarray(landing_pixels)],
                False,
                (110, 235, 250),
                2,
                cv2.LINE_AA,
            )
        for event, point in zip(prior_events, landing_pixels):
            color = (255, 170, 45) if event["foot"] == "left" else (60, 220, 255)
            draw_footprint(canvas, point, event["foot"], color)
        if frame_index in self.request.positions:
            draw_runner(
                canvas, self._world_to_pixel(self.request.positions[frame_index])
            )
        self._draw_title(canvas)
        return canvas

    def _world_to_pixel(self, point: tuple[float, float]) -> tuple[int, int]:
        return world_to_pixel(
            point,
            self.request.bounds,
            self.request.px_per_meter,
            self.request.padding_px,
        )

    def _draw_title(self, canvas: np.ndarray) -> None:
        cv2.putText(
            canvas,
            f"{self.request.title}  |  {self.request.px_per_meter:g} px/m",
            (self.request.padding_px, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )

    def _transcode(self, temp_video: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(temp_video),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.request.output_path),
            ],
            check=True,
        )

    def _summary(self) -> dict:
        return {
            "output": str(self.request.output_path),
            "size": {
                "width": self.request.size[0],
                "height": self.request.size[1],
            },
            "fps": self.fps,
            "frames": self.frame_count,
            "runner_position_frames": len(self.request.positions),
            "projected_landings": len(self.request.events),
            "px_per_meter_x": self.request.px_per_meter,
            "px_per_meter_y": self.request.px_per_meter,
        }


def render(request: RenderRequest) -> dict:
    return _SchematicRenderer(request).render()


def main() -> None:
    options = parse_args()
    video_path = Path(options.video).expanduser().resolve()
    metrics_path = Path(options.metrics_csv).expanduser().resolve()
    events_path = Path(options.step_events_csv).expanduser().resolve()
    output_path = Path(options.output).expanduser().resolve()
    camera_segments = None
    camera_offsets = None
    title = "Equal-scale top-down"
    if options.calibrations_json:
        calibrations_path = Path(options.calibrations_json).expanduser().resolve()
        calibrations = load_calibrations(calibrations_path)
        positions, events, world_controls, camera_segments, camera_offsets = (
            load_full_trial_data(metrics_path, events_path, calibrations)
        )
        title = "Full-trial equal-scale top-down"
        chart_axes = True
    else:
        points_path = Path(options.points_json).expanduser().resolve()
        matrix, world_controls = load_homography(points_path)
        positions = load_runner_positions(metrics_path, matrix, options.camera_index)
        events = load_landing_events(events_path, matrix, options.camera_index)
        chart_axes = False
    size, bounds = output_geometry(
        world_controls, options.px_per_meter, options.padding_px
    )
    if chart_axes:
        size = (size[0], size[1] + 62)
    summary = render(
        RenderRequest(
            video_path=video_path,
            output_path=output_path,
            positions=positions,
            events=events,
            size=size,
            bounds=bounds,
            px_per_meter=options.px_per_meter,
            padding_px=options.padding_px,
            camera_segments=camera_segments,
            title=title,
            chart_axes=chart_axes,
        )
    )
    if camera_offsets is not None:
        summary["camera_offsets_m"] = {
            str(camera_index): offset for camera_index, offset in camera_offsets.items()
        }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
