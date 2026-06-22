"""
Compute per-joint frame-to-frame displacement statistics from keypoints.npz.

The input files are MotionAGFormer/HRNet 2D keypoints files with:
  reconstruction: shape (person, frame, joint, 3)

For each joint, this script measures:
  displacement[t] = distance(keypoint[t], keypoint[t - 1])

Only frame pairs whose two confidence values are both >= --min-confidence are
included by default. The p95 and p99 columns can be used as data-driven
suggestions for joint_max and joint_hard.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


H36M_JOINT_NAMES = [
    "Hip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "Spine",
    "Thorax",
    "Neck_Nose",
    "Head",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RShoulder",
    "RElbow",
    "RWrist",
]


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            paths.extend(Path(match) for match in matches)
            continue
        path = Path(pattern)
        if path.exists():
            paths.append(path)

    unique = sorted({path.resolve() for path in paths})
    return [path for path in unique if path.is_file()]


def _joint_name(joint_idx: int) -> str:
    if joint_idx < len(H36M_JOINT_NAMES):
        return H36M_JOINT_NAMES[joint_idx]
    return f"joint_{joint_idx}"


def _load_displacements(npz_path: Path, min_confidence: float) -> dict[int, list[float]]:
    data = np.load(npz_path, allow_pickle=True)
    keypoints = data["reconstruction"]
    if keypoints.ndim != 4 or keypoints.shape[-1] < 2:
        raise ValueError(f"{npz_path} reconstruction must have shape (M,T,J,2+)") 

    xy = keypoints[..., :2].astype(np.float64)
    if keypoints.shape[-1] >= 3:
        conf = keypoints[..., 2].astype(np.float64)
    else:
        conf = np.ones(keypoints.shape[:-1], dtype=np.float64)

    num_joints = keypoints.shape[2]
    by_joint = {joint_idx: [] for joint_idx in range(num_joints)}

    if keypoints.shape[1] < 2:
        return by_joint

    deltas = xy[:, 1:, :, :] - xy[:, :-1, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    valid_pairs = (conf[:, 1:, :] >= min_confidence) & (conf[:, :-1, :] >= min_confidence)

    for joint_idx in range(num_joints):
        values = distances[:, :, joint_idx][valid_pairs[:, :, joint_idx]]
        if values.size:
            by_joint[joint_idx].extend(float(v) for v in values[np.isfinite(values)])

    return by_joint


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def compute_stats(npz_paths: list[Path], min_confidence: float) -> list[dict]:
    all_values: dict[int, list[float]] = {}

    for npz_path in npz_paths:
        by_joint = _load_displacements(npz_path, min_confidence)
        for joint_idx, values in by_joint.items():
            all_values.setdefault(joint_idx, []).extend(values)

    rows = []
    for joint_idx in sorted(all_values):
        stats = _summary(all_values[joint_idx])
        rows.append({
            "joint_index": joint_idx,
            "joint": _joint_name(joint_idx),
            **stats,
            "suggested_joint_max": stats["p95"],
            "suggested_joint_hard": stats["p99"],
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "joint_index",
        "joint",
        "count",
        "mean",
        "std",
        "p50",
        "p90",
        "p95",
        "p99",
        "max",
        "suggested_joint_max",
        "suggested_joint_hard",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-joint displacement statistics from keypoints.npz files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="keypoints.npz paths or glob patterns, e.g. 'output*/**/keypoints.npz'",
    )
    parser.add_argument(
        "--output",
        default="joint_motion_stats.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.50,
        help="Only count frame pairs where both endpoint confidences are >= this value.",
    )
    args = parser.parse_args()

    npz_paths = _expand_inputs(args.inputs)
    if not npz_paths:
        raise FileNotFoundError("No input keypoints.npz files found.")

    rows = compute_stats(npz_paths, args.min_confidence)
    output_path = Path(args.output).resolve()
    write_csv(output_path, rows)

    print(f"Inputs: {len(npz_paths)}")
    print(f"Output: {output_path}")
    for row in rows:
        print(
            f"{row['joint_index']:>2} {row['joint']:<12} "
            f"count={row['count']:<6} p95={row['p95']} p99={row['p99']}"
        )


if __name__ == "__main__":
    main()
