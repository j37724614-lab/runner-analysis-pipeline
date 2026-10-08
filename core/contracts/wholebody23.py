"""Publish Server HRNet WholeBody23 in the Local pose artifact shape."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np


JOINT_ORDER = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]


def write_wholebody23_artifact(
    *,
    raw_npz: str | os.PathLike[str],
    offsets_npz: str | os.PathLike[str],
    bbox_map_csv: str | os.PathLike[str] | None,
    destination: str | os.PathLike[str],
    frames_per_second: float,
) -> Path:
    """Convert crop-space model output to Local-compatible original pixels.

    The existing H36M17 and foot NPZ files remain the inputs to legacy Server
    post-processing.  This document is an additional immutable comparison
    artifact; it never feeds back into the numerical pipeline.
    """
    if frames_per_second <= 0:
        raise ValueError("frames_per_second must be positive")

    with np.load(raw_npz, allow_pickle=False) as raw:
        keypoints = np.asarray(raw["keypoints"], dtype=np.float64)
        scores = np.asarray(raw["scores"], dtype=np.float64)
    if keypoints.ndim != 4 or keypoints.shape[0] != 1 or keypoints.shape[2:] != (23, 2):
        raise ValueError(f"expected keypoints shape (1, frames, 23, 2), got {keypoints.shape}")
    if scores.shape != keypoints.shape[:3]:
        raise ValueError(f"scores shape {scores.shape} does not match {keypoints.shape[:3]}")

    with np.load(offsets_npz, allow_pickle=False) as mapping:
        offsets = np.asarray(mapping["offsets"], dtype=np.float64)
        source_frames = np.asarray(mapping["orig_frames"], dtype=np.int64).reshape(-1)
        camera_indices = (
            np.asarray(mapping["cam_indices"], dtype=np.int64).reshape(-1)
            if "cam_indices" in mapping
            else np.zeros(len(source_frames), dtype=np.int64)
        )

    frame_count = keypoints.shape[1]
    observed_counts = {frame_count, len(offsets), len(source_frames), len(camera_indices)}
    if len(observed_counts) != 1:
        raise ValueError(
            "WholeBody23/frame-map length mismatch: "
            f"pose={frame_count}, offsets={len(offsets)}, "
            f"source_frames={len(source_frames)}, cameras={len(camera_indices)}"
        )
    if offsets.shape != (frame_count, 2):
        raise ValueError(f"expected offsets shape ({frame_count}, 2), got {offsets.shape}")

    bbox_rows = _bbox_rows_by_output_frame(bbox_map_csv)
    frames = []
    for index in range(frame_count):
        offset_x, offset_y = offsets[index]
        frame_scores = scores[0, index]
        frame_keypoints = keypoints[0, index]
        valid = bool(
            np.isfinite(frame_keypoints).all()
            and np.isfinite(frame_scores).all()
            and np.any(frame_scores > 0)
        )
        joints = []
        if valid:
            joints = [
                {
                    "x": float(point[0] + offset_x),
                    "y": float(point[1] + offset_y),
                    "score": float(score),
                }
                for point, score in zip(frame_keypoints, frame_scores)
            ]

        bbox_row = bbox_rows.get(index)
        bbox = None
        bbox_extrapolated = False
        if bbox_row is not None:
            bbox = {
                "x1": float(bbox_row["x1"]) + float(offset_x),
                "y1": float(bbox_row["y1"]) + float(offset_y),
                "x2": float(bbox_row["x2"]) + float(offset_x),
                "y2": float(bbox_row["y2"]) + float(offset_y),
            }
            bbox_extrapolated = bool(int(bbox_row.get("is_interpolated") or 0))

        source_frame = int(source_frames[index])
        frames.append({
            "camera_index": int(camera_indices[index]),
            "source_frame": source_frame,
            "timestamp_seconds": source_frame / frames_per_second,
            "bbox": bbox,
            "joints": joints,
            "valid": valid,
            "bbox_extrapolated": bbox_extrapolated,
        })

    document = {
        "schema_version": "1.0.0",
        "joint_order": JOINT_ORDER,
        "frames": frames,
    }
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(f".{destination_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, destination_path)
    return destination_path


def _bbox_rows_by_output_frame(
    bbox_map_csv: str | os.PathLike[str] | None,
) -> Mapping[int, dict[str, str]]:
    if bbox_map_csv is None or not Path(bbox_map_csv).is_file():
        return {}
    with Path(bbox_map_csv).open(newline="", encoding="utf-8-sig") as source:
        return {
            int(row["output_frame"]): row
            for row in csv.DictReader(source)
        }
