"""
Build a homography from clicked cone ground points and export:
  1. homography_rectified_preview.mp4
  2. cone_distance_check.csv
  3. homography_debug_overlay.png

The input JSON should contain at least four points. Each point needs image
coordinates and real-world ground-plane coordinates in meters:

{
  "points": [
    {"id": 1, "x": 1234, "y": 567, "world_x_m": 0.0, "world_y_m": 0.0},
    {"id": 2, "x": 1240, "y": 620, "world_x_m": 0.0, "world_y_m": 1.22},
    {"id": 3, "x": 2100, "y": 540, "world_x_m": 10.0, "world_y_m": 0.0},
    {"id": 4, "x": 2115, "y": 590, "world_x_m": 10.0, "world_y_m": 1.22}
  ]
}

If the JSON only has x/y for exactly four points, pass:
  --world-points "0,0;0,1.22;10,0;10,1.22"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

_MARKER_COLOURS = {
    "heel": (0, 180, 0),
    "big_toe": (0, 165, 255),
    "estimated": (255, 0, 255),
    "ankle": (0, 0, 255),
}


class DistanceReportRequest(NamedTuple):
    path: Path
    labels: list[str]
    image_points: np.ndarray
    world_points: np.ndarray
    image_to_world: np.ndarray
    control_indices: set[int]


class DebugOverlayRequest(NamedTuple):
    path: Path
    original_frame: np.ndarray
    rectified_frame: np.ndarray
    image_points: np.ndarray
    projected_rect_points: np.ndarray
    expected_rect_points: np.ndarray
    labels: list[str]


class PreviewVideoRequest(NamedTuple):
    video_path: Path
    output_path: Path
    image_to_rect: np.ndarray
    size: tuple[int, int]
    start_frame: int
    max_frames: int
    step_events: list[dict]


class RectificationInputs(NamedTuple):
    video_path: Path
    points_path: Path
    output_dir: Path
    image_points: np.ndarray
    world_points: np.ndarray
    labels: list[str]
    control_indices: list[int]
    frame_index: int
    frame: np.ndarray
    px_per_meter_x: float
    px_per_meter_y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--points-json", required=True, help="Clicked points JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Debug frame index; defaults to JSON meta/frame",
    )
    parser.add_argument(
        "--world-points",
        default=None,
        help='Fallback world coords, e.g. "0,0;0,1.22;10,0;10,1.22"',
    )
    parser.add_argument(
        "--control-count",
        type=int,
        default=4,
        help="First N points are used to solve Homography",
    )
    parser.add_argument(
        "--control-indices",
        default=None,
        help="Comma-separated 1-based point indices used to solve Homography, e.g. 1,2,3,4,9,10",
    )
    parser.add_argument(
        "--px-per-meter", type=float, default=100.0, help="Rectified preview scale"
    )
    parser.add_argument(
        "--px-per-meter-x",
        type=float,
        default=None,
        help="Optional horizontal scale; overrides --px-per-meter for X only",
    )
    parser.add_argument(
        "--px-per-meter-y",
        type=float,
        default=None,
        help="Optional vertical display scale; overrides --px-per-meter for Y only",
    )
    parser.add_argument(
        "--padding-px",
        type=int,
        default=80,
        help="Padding around rectified world bounds",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Max frames in preview; 0 means whole video",
    )
    parser.add_argument(
        "--start-frame", type=int, default=0, help="Preview start frame"
    )
    parser.add_argument(
        "--step-events-csv",
        default=None,
        help="Optional step_events.csv. Valid world-coordinate contacts are drawn on the rectified video.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Optional zero-based camera index; only that camera's step events are drawn.",
    )
    return parser.parse_args()


def parse_world_points(text: str | None) -> list[tuple[float, float]] | None:
    if not text:
        return None
    pts = []
    for item in text.split(";"):
        x_s, y_s = item.split(",", 1)
        pts.append((float(x_s), float(y_s)))
    return pts


def parse_control_indices(
    text: str | None, control_count: int, n_points: int
) -> list[int]:
    if text:
        indices = [int(x.strip()) - 1 for x in text.split(",") if x.strip()]
        if len(indices) < 4:
            raise ValueError("--control-indices must contain at least four points")
        if len(set(indices)) != len(indices):
            raise ValueError("--control-indices contains duplicate points")
        if min(indices) < 0 or max(indices) >= n_points:
            raise ValueError("--control-indices outside point range")
        return indices
    if control_count < 4:
        raise ValueError("--control-count must be at least 4")
    if control_count > n_points:
        raise ValueError("--control-count cannot exceed point count")
    return list(range(control_count))


def load_points(
    path: Path, world_points_arg: str | None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    points = data.get("points")
    if not isinstance(points, list) or len(points) < 4:
        raise ValueError("points-json must contain at least four points")

    fallback_world = parse_world_points(world_points_arg)
    if fallback_world is not None and len(fallback_world) != len(points):
        raise ValueError("--world-points count must match clicked point count")

    img_pts = []
    world_pts = []
    labels = []
    for i, p in enumerate(points):
        if "x" not in p or "y" not in p:
            raise ValueError(f"point {i + 1} missing x/y")
        img_pts.append([float(p["x"]), float(p["y"])])
        label = str(p.get("id", p.get("index", i + 1)))
        labels.append(label)

        wx = p.get("world_x_m")
        wy = p.get("world_y_m")
        if wx is None or wy is None:
            if fallback_world is None:
                raise ValueError(
                    f"point {i + 1} missing world_x_m/world_y_m; "
                    "provide them in JSON or pass --world-points"
                )
            wx, wy = fallback_world[i]
        world_pts.append([float(wx), float(wy)])

    return (
        np.asarray(img_pts, dtype=np.float64),
        np.asarray(world_pts, dtype=np.float64),
        labels,
    )


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index}")
    return frame


def infer_frame_from_json(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "frame_index" in data:
        return int(data["frame_index"])
    meta = data.get("meta") or {}
    if "frame" in meta:
        return int(meta["frame"])
    return 0


def build_homographies(
    img_pts: np.ndarray,
    world_pts: np.ndarray,
    px_per_meter_x: float,
    px_per_meter_y: float,
    padding_px: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray]:
    if px_per_meter_x <= 0 or px_per_meter_y <= 0:
        raise ValueError("Pixel-per-meter scales must be greater than zero")

    h_img_to_world, _ = cv2.findHomography(img_pts, world_pts, method=0)
    if h_img_to_world is None:
        raise RuntimeError("cv2.findHomography failed")

    min_xy = world_pts.min(axis=0)
    max_xy = world_pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    width_px = math.ceil(span[0] * px_per_meter_x + 2 * padding_px)
    height_px = math.ceil(span[1] * px_per_meter_y + 2 * padding_px)
    # VideoWriter/H.264 yuv420p require even dimensions. Round upward so the
    # requested calibrated area is never silently cropped by the encoder.
    width_px += width_px % 2
    height_px += height_px % 2

    world_to_rect = np.array(
        [
            [px_per_meter_x, 0.0, padding_px - min_xy[0] * px_per_meter_x],
            [0.0, px_per_meter_y, padding_px - min_xy[1] * px_per_meter_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    h_img_to_rect = world_to_rect @ h_img_to_world
    return h_img_to_world, h_img_to_rect, (width_px, height_px), world_to_rect


def transform_points(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_in = pts.reshape(-1, 1, 2).astype(np.float64)
    pts_out = cv2.perspectiveTransform(pts_in, h).reshape(-1, 2)
    return pts_out


def draw_points(
    img: np.ndarray, pts: np.ndarray, labels: list[str], title: str
) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        title,
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )
    for p, label in zip(pts, labels):
        x, y = round(p[0]), round(p[1])
        cv2.circle(out, (x, y), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 11, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            str(label),
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def draw_rectified_comparison(
    img: np.ndarray,
    projected_pts: np.ndarray,
    expected_pts: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out,
        "Rectified: red=projected clicked point, green=expected world point",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for p_proj, p_exp, label in zip(projected_pts, expected_pts, labels):
        x1, y1 = round(p_proj[0]), round(p_proj[1])
        x2, y2 = round(p_exp[0]), round(p_exp[1])
        cv2.line(out, (x1, y1), (x2, y2), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(out, (x1, y1), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (x1, y1), 11, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (x2, y2), 8, (0, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(out, (x2, y2), 11, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            str(label),
            (x1 + 12, y1 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def _distance_rows(request: DistanceReportRequest) -> list[dict[str, str]]:
    measured_world = transform_points(request.image_to_world, request.image_points)
    rows = []
    for first_index, second_index in combinations(range(len(request.labels)), 2):
        measured_distance = float(
            np.linalg.norm(measured_world[second_index] - measured_world[first_index])
        )
        true_distance = float(
            np.linalg.norm(
                request.world_points[second_index] - request.world_points[first_index]
            )
        )
        error = measured_distance - true_distance
        error_percent = error / true_distance * 100.0 if true_distance > 1e-9 else 0.0
        rows.append(
            {
                "point_a": request.labels[first_index],
                "point_b": request.labels[second_index],
                "measured_distance_m": f"{measured_distance:.6f}",
                "true_distance_m": f"{true_distance:.6f}",
                "error_m": f"{error:.6f}",
                "error_percent": f"{error_percent:.3f}",
                "measured_a_x_m": f"{measured_world[first_index, 0]:.6f}",
                "measured_a_y_m": f"{measured_world[first_index, 1]:.6f}",
                "measured_b_x_m": f"{measured_world[second_index, 0]:.6f}",
                "measured_b_y_m": f"{measured_world[second_index, 1]:.6f}",
                "pair_type": (
                    "control-control"
                    if first_index in request.control_indices
                    and second_index in request.control_indices
                    else "validation"
                ),
            }
        )
    return rows


def write_distance_csv(request: DistanceReportRequest) -> None:
    with request.path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "point_a",
                "point_b",
                "measured_distance_m",
                "true_distance_m",
                "error_m",
                "error_percent",
                "measured_a_x_m",
                "measured_a_y_m",
                "measured_b_x_m",
                "measured_b_y_m",
                "pair_type",
            ],
        )
        writer.writeheader()
        writer.writerows(_distance_rows(request))


def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    scale = target_height / image.shape[0]
    target_width = round(image.shape[1] * scale)
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def write_debug_overlay(request: DebugOverlayRequest) -> None:
    left = draw_points(
        request.original_frame,
        request.image_points,
        request.labels,
        "Original clicked cone points",
    )
    right = draw_rectified_comparison(
        request.rectified_frame,
        request.projected_rect_points,
        request.expected_rect_points,
        request.labels,
    )
    canvas = np.hstack([_resize_to_height(left, 720), _resize_to_height(right, 720)])
    cv2.imwrite(str(request.path), canvas)


def _contact_label(event: dict, joint: str) -> str:
    if str(event.get("event_type")) == "final_landing":
        return "LANDING"
    joint_tag = next(
        (
            tag
            for name, tag in (
                ("heel", "H"),
                ("big_toe", "T"),
                ("estimated", "E"),
            )
            if name in joint
        ),
        "A",
    )
    label = f"S{event['step_index']} {joint_tag}"
    try:
        return f"{label} L={float(event.get('step_length_m')):.2f}m"
    except (TypeError, ValueError):
        return label


def _draw_contact_history(frame: np.ndarray, history: list[dict]) -> None:
    for event in history:
        joint = str(event.get("contact_joint") or "ankle")
        colour = next(
            (value for name, value in _MARKER_COLOURS.items() if name in joint),
            _MARKER_COLOURS["ankle"],
        )
        point = (round(float(event["rect_x"])), round(float(event["rect_y"])))
        cv2.circle(frame, point, 8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, point, 4, colour, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            _contact_label(event, joint),
            (point[0] + 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )


class _PreviewVideoRenderer:
    def __init__(self, request: PreviewVideoRequest) -> None:
        self.request = request
        self.events = sorted(
            request.step_events,
            key=lambda item: int(item["orig_frame"]),
        )
        self.event_index = 0
        self.history: list[dict] = []

    def render(self) -> int:
        capture = self._open_capture()
        writer = self._open_writer(capture)
        try:
            written = self._write_frames(capture, writer)
        finally:
            writer.release()
            capture.release()
        self._transcode_to_h264()
        return written

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(self.request.video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.request.video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, self.request.start_frame))
        return capture

    def _open_writer(self, capture: cv2.VideoCapture) -> cv2.VideoWriter:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        writer = cv2.VideoWriter(
            str(self.request.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            self.request.size,
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open video writer: {self.request.output_path}")
        return writer

    def _write_frames(
        self,
        capture: cv2.VideoCapture,
        writer: cv2.VideoWriter,
    ) -> int:
        written = 0
        frame_limit = self._frame_limit(capture)
        while written < frame_limit:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            writer.write(self._render_frame(frame, written))
            written += 1
        return written

    def _frame_limit(self, capture: cv2.VideoCapture) -> int:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        available_frames = max(0, total_frames - max(0, self.request.start_frame))
        if self.request.max_frames == 0:
            return available_frames
        return min(self.request.max_frames, available_frames)

    def _render_frame(self, frame: np.ndarray, written: int) -> np.ndarray:
        rectified = cv2.warpPerspective(
            frame,
            self.request.image_to_rect,
            self.request.size,
            flags=cv2.INTER_LINEAR,
        )
        source_frame = max(0, self.request.start_frame) + written
        while (
            self.event_index < len(self.events)
            and int(self.events[self.event_index]["orig_frame"]) <= source_frame
        ):
            self.history.append(self.events[self.event_index])
            self.event_index += 1
        _draw_contact_history(rectified, self.history)
        return rectified

    def _transcode_to_h264(self) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return
        h264_path = self.request.output_path.with_name(
            f"{self.request.output_path.stem}_h264.mp4"
        )
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(self.request.output_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(h264_path),
            ],
            check=False,
        )
        if h264_path.exists() and h264_path.stat().st_size > 0:
            h264_path.replace(self.request.output_path)


def write_preview_video(request: PreviewVideoRequest) -> int:
    return _PreviewVideoRenderer(request).render()


def _resolve_inputs(args: argparse.Namespace) -> RectificationInputs:
    px_per_meter_x = (
        args.px_per_meter_x if args.px_per_meter_x is not None else args.px_per_meter
    )
    px_per_meter_y = (
        args.px_per_meter_y if args.px_per_meter_y is not None else args.px_per_meter
    )
    video_path = Path(args.video).expanduser().resolve()
    points_path = Path(args.points_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    img_pts, world_pts, labels = load_points(points_path, args.world_points)
    control_indices = parse_control_indices(
        args.control_indices, args.control_count, len(labels)
    )
    frame_index = (
        args.frame if args.frame is not None else infer_frame_from_json(points_path)
    )
    return RectificationInputs(
        video_path=video_path,
        points_path=points_path,
        output_dir=output_dir,
        image_points=img_pts,
        world_points=world_pts,
        labels=labels,
        control_indices=control_indices,
        frame_index=frame_index,
        frame=read_frame(video_path, frame_index),
        px_per_meter_x=px_per_meter_x,
        px_per_meter_y=px_per_meter_y,
    )


def _load_step_events(
    csv_path: str | None,
    camera_index: int | None,
    world_to_rect: np.ndarray,
) -> list[dict]:
    if not csv_path:
        return []
    step_events = []
    with (
        Path(csv_path)
        .expanduser()
        .resolve()
        .open(
            newline="",
            encoding="utf-8",
        ) as input_file
    ):
        for event in csv.DictReader(input_file):
            try:
                if (
                    camera_index is not None
                    and int(event.get("cam", -1)) != camera_index
                ):
                    continue
                world = np.array(
                    [[float(event["world_x_m"]), float(event["world_y_m"])]],
                )
                rect_x, rect_y = transform_points(world_to_rect, world)[0]
                event["rect_x"], event["rect_y"] = float(rect_x), float(rect_y)
                step_events.append(event)
            except (KeyError, TypeError, ValueError):
                continue
    return step_events


class _RectificationRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.inputs = _resolve_inputs(args)
        self.image_to_world: np.ndarray
        self.image_to_rect: np.ndarray
        self.rect_size: tuple[int, int]
        self.world_to_rect: np.ndarray
        self.step_events: list[dict] = []

    def run(self) -> dict:
        self._build_homography()
        self.step_events = _load_step_events(
            self.args.step_events_csv,
            self.args.camera_index,
            self.world_to_rect,
        )
        preview_frames = self._write_outputs()
        summary = self._summary(preview_frames)
        self._publish_summary(summary)
        return summary

    def _build_homography(self) -> None:
        selected = self.inputs.control_indices
        (
            self.image_to_world,
            self.image_to_rect,
            self.rect_size,
            self.world_to_rect,
        ) = build_homographies(
            self.inputs.image_points[selected],
            self.inputs.world_points[selected],
            self.inputs.px_per_meter_x,
            self.inputs.px_per_meter_y,
            self.args.padding_px,
        )

    def _write_outputs(self) -> int:
        preview_frames = write_preview_video(
            PreviewVideoRequest(
                video_path=self.inputs.video_path,
                output_path=self.inputs.output_dir / "homography_rectified_preview.mp4",
                image_to_rect=self.image_to_rect,
                size=self.rect_size,
                start_frame=self.args.start_frame,
                max_frames=self.args.max_frames,
                step_events=self.step_events,
            )
        )
        write_distance_csv(self._distance_report_request())
        write_debug_overlay(self._debug_overlay_request())
        return preview_frames

    def _distance_report_request(self) -> DistanceReportRequest:
        return DistanceReportRequest(
            path=self.inputs.output_dir / "cone_distance_check.csv",
            labels=self.inputs.labels,
            image_points=self.inputs.image_points,
            world_points=self.inputs.world_points,
            image_to_world=self.image_to_world,
            control_indices=set(self.inputs.control_indices),
        )

    def _debug_overlay_request(self) -> DebugOverlayRequest:
        rectified_frame = cv2.warpPerspective(
            self.inputs.frame,
            self.image_to_rect,
            self.rect_size,
            flags=cv2.INTER_LINEAR,
        )
        return DebugOverlayRequest(
            path=self.inputs.output_dir / "homography_debug_overlay.png",
            original_frame=self.inputs.frame,
            rectified_frame=rectified_frame,
            image_points=self.inputs.image_points,
            projected_rect_points=transform_points(
                self.image_to_rect,
                self.inputs.image_points,
            ),
            expected_rect_points=transform_points(
                self.world_to_rect,
                self.inputs.world_points,
            ),
            labels=self.inputs.labels,
        )

    def _summary(self, preview_frames: int) -> dict:
        output_dir = self.inputs.output_dir
        return {
            "video": str(self.inputs.video_path),
            "points_json": str(self.inputs.points_path),
            "frame": self.inputs.frame_index,
            "rectified_size": {
                "width": self.rect_size[0],
                "height": self.rect_size[1],
            },
            "px_per_meter": self.args.px_per_meter,
            "px_per_meter_x": self.inputs.px_per_meter_x,
            "px_per_meter_y": self.inputs.px_per_meter_y,
            "is_isotropic_scale": math.isclose(
                self.inputs.px_per_meter_x,
                self.inputs.px_per_meter_y,
            ),
            "control_indices_1based": [
                index + 1 for index in self.inputs.control_indices
            ],
            "control_count": len(self.inputs.control_indices),
            "validation_count": max(
                0,
                len(self.inputs.labels) - len(self.inputs.control_indices),
            ),
            "preview_frames_written": preview_frames,
            "step_events_drawn": len(self.step_events),
            "homography_image_to_world": self.image_to_world.tolist(),
            "outputs": {
                "homography_rectified_preview": str(
                    output_dir / "homography_rectified_preview.mp4"
                ),
                "cone_distance_check": str(output_dir / "cone_distance_check.csv"),
                "homography_debug_overlay": str(
                    output_dir / "homography_debug_overlay.png"
                ),
            },
        }

    def _publish_summary(self, summary: dict) -> None:
        summary_path = self.inputs.output_dir / "homography_rectification_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(summary["outputs"], ensure_ascii=False, indent=2))


def main() -> None:
    _RectificationRunner(parse_args()).run()


if __name__ == "__main__":
    main()
