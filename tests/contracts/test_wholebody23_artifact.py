import csv
import json

import numpy as np
import pytest

from core.contracts.wholebody23 import JOINT_ORDER, write_wholebody23_artifact


def test_writes_local_compatible_original_coordinate_wholebody23(tmp_path):
    keypoints = np.zeros((1, 2, 23, 2), dtype=np.float32)
    scores = np.zeros((1, 2, 23), dtype=np.float32)
    keypoints[0, 0] = [[10 + joint, 20 + joint] for joint in range(23)]
    scores[0, 0] = 0.9
    raw = tmp_path / "wholebody23_raw.npz"
    np.savez_compressed(raw, keypoints=keypoints, scores=scores)

    offsets = tmp_path / "cam1_offsets.npz"
    np.savez_compressed(
        offsets,
        offsets=np.asarray([[100, 200], [300, 400]]),
        orig_frames=np.asarray([7, 11]),
        cam_indices=np.asarray([0, 1]),
    )
    bbox = tmp_path / "tracked_bbox_map.csv"
    with bbox.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["output_frame", "x1", "y1", "x2", "y2", "is_interpolated"],
        )
        writer.writeheader()
        writer.writerow({
            "output_frame": 0,
            "x1": 1,
            "y1": 2,
            "x2": 30,
            "y2": 40,
            "is_interpolated": 1,
        })

    destination = tmp_path / "pose" / "keypoints_2d.json"
    result = write_wholebody23_artifact(
        raw_npz=raw,
        offsets_npz=offsets,
        bbox_map_csv=bbox,
        destination=destination,
        frames_per_second=50,
    )

    assert result == destination
    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["schema_version"] == "1.0.0"
    assert document["joint_order"] == JOINT_ORDER
    assert len(document["frames"]) == 2
    first = document["frames"][0]
    assert first["camera_index"] == 0
    assert first["source_frame"] == 7
    assert first["timestamp_seconds"] == pytest.approx(0.14)
    assert first["joints"][0] == {"x": 110.0, "y": 220.0, "score": pytest.approx(0.9)}
    assert first["joints"][22]["x"] == 132.0
    assert first["bbox"] == {"x1": 101.0, "y1": 202.0, "x2": 130.0, "y2": 240.0}
    assert first["bbox_extrapolated"] is True
    assert first["valid"] is True

    second = document["frames"][1]
    assert second["camera_index"] == 1
    assert second["source_frame"] == 11
    assert second["joints"] == []
    assert second["bbox"] is None
    assert second["valid"] is False


def test_rejects_frame_map_length_mismatch(tmp_path):
    raw = tmp_path / "wholebody23_raw.npz"
    np.savez_compressed(
        raw,
        keypoints=np.zeros((1, 2, 23, 2)),
        scores=np.zeros((1, 2, 23)),
    )
    offsets = tmp_path / "offsets.npz"
    np.savez_compressed(
        offsets,
        offsets=np.zeros((1, 2)),
        orig_frames=np.asarray([0]),
        cam_indices=np.asarray([0]),
    )

    with pytest.raises(ValueError, match="length mismatch"):
        write_wholebody23_artifact(
            raw_npz=raw,
            offsets_npz=offsets,
            bbox_map_csv=None,
            destination=tmp_path / "pose.json",
            frames_per_second=60,
        )
