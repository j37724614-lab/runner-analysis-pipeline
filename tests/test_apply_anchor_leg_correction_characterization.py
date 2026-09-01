"""Characterization tests for apply_anchor_leg_correction().

These lock the current behaviour of the anchor-DP leg-identity solver plus its
persistence step (fill leg gaps -> smooth/limit legs -> overwrite keypoints.npz)
on a deterministic synthetic skeleton, so the planned extraction of the Viterbi
solver into its own unit and the npz writes into one place can be verified to
preserve output exactly.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis.ankle_step_stride import (  # noqa: E402
    apply_anchor_leg_correction,
)

R_HIP, R_KNEE, R_ANKLE = 1, 2, 3
L_HIP, L_KNEE, L_ANKLE = 4, 5, 6

_SWAPPED_MASK = [
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
]
_LEG_CHECKSUM = 36073.7131


def _synthetic_recon(frames=30, label_error_frames=range(10, 20)):
    recon = np.zeros((1, frames, 17, 3), dtype=np.float64)
    for t in range(frames):
        x = 10.0 + 4.0 * t
        phase = 2 * np.pi * t / 10.0
        recon[0, t, 0] = (x, 100.0, 0.9)
        recon[0, t, R_HIP] = (x + 5, 105.0, 0.9)
        recon[0, t, R_KNEE] = (x + 5, 132.0, 0.9)
        recon[0, t, R_ANKLE] = (x + 5, 160.0 + 6.0 * np.cos(phase), 0.9)
        recon[0, t, L_HIP] = (x - 5, 105.0, 0.9)
        recon[0, t, L_KNEE] = (x - 5, 132.0, 0.9)
        recon[0, t, L_ANKLE] = (x - 5, 160.0 + 6.0 * np.cos(phase + np.pi), 0.9)
        for joint in range(7, 17):
            recon[0, t, joint] = (x, 60.0, 0.9)
    for t in label_error_frames:
        for right, left in ((R_HIP, L_HIP), (R_KNEE, L_KNEE), (R_ANKLE, L_ANKLE)):
            swap = recon[0, t, right].copy()
            recon[0, t, right] = recon[0, t, left]
            recon[0, t, left] = swap
    return recon


def _alternating_anchors():
    fps = 60.0
    frames = [4, 12, 20, 28]
    feet = ["right", "left", "right", "left"]
    return [
        {"seq_frame": f, "foot": ft, "seq_time_s": f / fps, "time_s": f / fps}
        for f, ft in zip(frames, feet)
    ]


def _write_keypoints(tmp_path, recon):
    path = tmp_path / "keypoints.npz"
    np.savez_compressed(
        path, reconstruction=recon, valid_frames=np.arange(recon.shape[1])
    )
    return path


def _write_status(tmp_path, frames):
    np.savez_compressed(
        tmp_path / "keypoint_status.npz",
        raw_confidence=np.ones((1, frames, 17), dtype=np.float32),
        low_conf_mask=np.zeros((1, frames, 17), dtype=bool),
        ankle_crossing_mask=np.zeros((1, frames, 17), dtype=bool),
        pre_dp_leg_swap_mask=np.zeros((1, frames), dtype=bool),
        bbox_heights=np.full((1, frames), 100.0, dtype=np.float32),
        bbox_ref_height=np.asarray(100.0, dtype=np.float32),
    )


def test_fewer_than_two_anchors_returns_none(tmp_path):
    path = _write_keypoints(tmp_path, _synthetic_recon())

    assert apply_anchor_leg_correction(str(path), _alternating_anchors()[:1]) is None


def test_solver_swap_mask_and_rewritten_keypoints(tmp_path):
    path = _write_keypoints(tmp_path, _synthetic_recon())

    swapped = apply_anchor_leg_correction(str(path), _alternating_anchors())

    assert swapped.astype(int).tolist() == _SWAPPED_MASK
    rewritten = np.load(path, allow_pickle=True)["reconstruction"]
    assert rewritten.shape == (1, 30, 17, 3)
    leg_checksum = float(np.nansum(np.abs(rewritten[0, :, 1:7, :2])))
    assert round(leg_checksum, 4) == _LEG_CHECKSUM


def test_runs_with_keypoint_status_file(tmp_path):
    recon = _synthetic_recon()
    path = _write_keypoints(tmp_path, recon)
    _write_status(tmp_path, recon.shape[1])

    swapped = apply_anchor_leg_correction(str(path), _alternating_anchors())

    assert swapped.astype(int).tolist() == _SWAPPED_MASK
    assert (tmp_path / "post_dp_ankle_fill_mask.npz").exists()
    rewritten = np.load(path, allow_pickle=True)["reconstruction"]
    leg_checksum = float(np.nansum(np.abs(rewritten[0, :, 1:7, :2])))
    assert round(leg_checksum, 4) == _LEG_CHECKSUM
