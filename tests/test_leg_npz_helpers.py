"""Characterization tests for the leg / foot npz helper functions that were
previously untested (and are adjacent to the clean-code follow-up work):
_post_dp_smooth_and_limit_legs, apply_foot_leg_correction (the merged foot
L/R-swap + re-anchor step), update_leg_swap_metadata.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis.ankle_step_stride import (  # noqa: E402
    KNEE_MIN_ANGLE_DEG,
    apply_foot_leg_correction,
    _post_dp_smooth_and_limit_legs,
    update_leg_swap_metadata,
)

R_ANKLE, L_ANKLE = 3, 6


def _knee_angle_deg(hip, knee, ankle):
    v1, v2 = hip - knee, ankle - knee
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _hyperflexed_leg_recon(frames=40):
    """Straight leg for most frames, with a sustained right-knee hyper-flex over
    frames 16-24 (ankle folded up beside the hip) that survives SG smoothing."""
    recon = np.zeros((1, frames, 17, 3))
    for t in range(frames):
        x = 10.0 + 2.0 * t
        recon[0, t, 0] = (x, 100, 0.9)
        recon[0, t, 1] = (x + 4, 110, 0.9)
        recon[0, t, 2] = (x + 4, 140, 0.9)
        recon[0, t, 3] = (x + 4, 170, 0.9)
        recon[0, t, 4] = (x - 4, 110, 0.9)
        recon[0, t, 5] = (x - 4, 140, 0.9)
        recon[0, t, 6] = (x - 4, 170, 0.9)
        for joint in range(7, 17):
            recon[0, t, joint] = (x, 60, 0.9)
        if 16 <= t <= 24:
            recon[0, t, 3, :2] = recon[0, t, 1, :2] + np.array([2.0, 3.0])
    return recon


def test_post_dp_smooth_and_limit_legs_opens_hyperflexed_knee():
    recon = _hyperflexed_leg_recon()
    before = _knee_angle_deg(recon[0, 20, 1, :2], recon[0, 20, 2, :2], recon[0, 20, 3, :2])

    out = _post_dp_smooth_and_limit_legs(_hyperflexed_leg_recon())

    assert out.shape == (1, 40, 17, 3)
    after = _knee_angle_deg(out[0, 20, 1, :2], out[0, 20, 2, :2], out[0, 20, 3, :2])
    assert before < KNEE_MIN_ANGLE_DEG
    assert after == pytest.approx(KNEE_MIN_ANGLE_DEG)
    # non-leg joints and the confidence channel are untouched
    assert np.array_equal(out[0, :, 7:, :], recon[0, :, 7:, :])
    assert np.array_equal(out[0, :, :, 2], recon[0, :, :, 2])


def _approaching_runner_recon(frames=40):
    """Straight legs whose on-screen length grows 2x as the runner walks toward
    the camera -- perspective, not a keypoint error. All trajectories are linear
    in t so SG smoothing leaves them unchanged."""
    recon = np.zeros((1, frames, 17, 3))
    for t in range(frames):
        scale = 1.0 + t / (frames - 1)
        x = 10.0 + 2.0 * t
        recon[0, t, 0] = (x, 100, 0.9)
        for hip, knee, ankle, dx in ((1, 2, 3, 4.0), (4, 5, 6, -4.0)):
            recon[0, t, hip] = (x + dx, 100 + 10 * scale, 0.9)
            recon[0, t, knee] = (x + dx, 100 + 40 * scale, 0.9)
            recon[0, t, ankle] = (x + dx, 100 + 70 * scale, 0.9)
        for joint in range(7, 17):
            recon[0, t, joint] = (x, 60, 0.9)
    return recon


def test_post_dp_smooth_and_limit_legs_keeps_ankles_when_runner_approaches_camera():
    recon = _approaching_runner_recon()

    out = _post_dp_smooth_and_limit_legs(_approaching_runner_recon())

    for ankle in (R_ANKLE, L_ANKLE):
        assert np.allclose(out[0, :, ankle, :2], recon[0, :, ankle, :2], atol=1e-6)


def test_post_dp_smooth_and_limit_legs_is_deterministic():
    a = _post_dp_smooth_and_limit_legs(_hyperflexed_leg_recon())
    b = _post_dp_smooth_and_limit_legs(_hyperflexed_leg_recon())
    assert np.array_equal(a, b)
    assert round(float(np.nansum(np.abs(a[0, :, 1:7, :2]))), 4) == 44944.3677


def _write_foot_npz(path, frames=6):
    keypoints = np.arange(1 * frames * 6 * 2, dtype=np.float64).reshape(1, frames, 6, 2)
    scores = np.arange(1 * frames * 6, dtype=np.float64).reshape(1, frames, 6)
    np.savez_compressed(
        path,
        keypoints=keypoints,
        scores=scores,
        valid_frames=np.arange(frames),
        keypoint_names=np.array(
            ["Lbt", "Lst", "Lh", "Rbt", "Rst", "Rh"], dtype=object
        ),
    )
    return keypoints, scores


def _write_body_npz(path, ankles_by_frame):
    """ankles_by_frame: list of ((l_x, l_y), (r_x, r_y)) per frame."""
    recon = np.zeros((1, len(ankles_by_frame), 17, 3))
    for t, ((lx, ly), (rx, ry)) in enumerate(ankles_by_frame):
        recon[0, t, L_ANKLE, :2] = (lx, ly)
        recon[0, t, R_ANKLE, :2] = (rx, ry)
    np.savez_compressed(path, reconstruction=recon, valid_frames=np.arange(len(ankles_by_frame)))


def test_apply_foot_leg_correction_swap_only_when_keypoints_missing(tmp_path):
    path = tmp_path / "foot_keypoints.npz"
    keypoints, scores = _write_foot_npz(path)
    mask = np.array([False, True, False, True, False, False])

    # re-anchor stage skipped: raw/final keypoints paths do not exist
    apply_foot_leg_correction(
        str(path), str(tmp_path / "raw.npz"), str(tmp_path / "kp.npz"), mask
    )

    out = np.load(path, allow_pickle=True)
    for t in range(6):
        if mask[t]:
            assert np.array_equal(out["keypoints"][:, t, [0, 1, 2]], keypoints[:, t, [3, 4, 5]])
            assert np.array_equal(out["scores"][:, t, [0, 1, 2]], scores[:, t, [3, 4, 5]])
        else:
            assert np.array_equal(out["keypoints"][:, t], keypoints[:, t])


def test_apply_foot_leg_correction_swap_then_reanchor(tmp_path):
    foot = tmp_path / "foot_keypoints.npz"
    foot_kp = np.arange(1 * 5 * 6 * 2, dtype=np.float64).reshape(1, 5, 6, 2)
    scores = (np.arange(1 * 5 * 6, dtype=np.float64) / 10).reshape(1, 5, 6)
    np.savez_compressed(
        foot, keypoints=foot_kp, scores=scores, valid_frames=np.arange(5),
        keypoint_names=np.array(list("abcdef"), dtype=object),
    )
    raw = tmp_path / "raw.npz"
    fin = tmp_path / "kp.npz"
    _write_body_npz(raw, [((10 + t, 20), (30 + t, 40)) for t in range(5)])
    _write_body_npz(fin, [((10 + t + 2, 21), (30 + t - 1, 43)) for t in range(5)])
    mask = np.array([False, True, False, False, True])

    apply_foot_leg_correction(str(foot), str(raw), str(fin), mask)

    out = np.load(foot, allow_pickle=True)
    assert round(float(np.sum(out["keypoints"])), 4) == 1845.0
    assert round(float(np.sum(out["scores"])), 4) == 43.5
    assert out["keypoints"][0, 0].tolist() == [
        [2.0, 2.0], [4.0, 4.0], [6.0, 6.0], [5.0, 10.0], [7.0, 12.0], [9.0, 14.0],
    ]
    assert out["keypoints"][0, 1].tolist() == [
        [0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [31.0, 36.0], [33.0, 38.0], [35.0, 40.0],
    ]


def test_apply_foot_leg_correction_noops_on_none_or_missing_foot(tmp_path):
    path = tmp_path / "foot_keypoints.npz"
    keypoints, _ = _write_foot_npz(path)
    apply_foot_leg_correction(str(path), "x", "y", None)
    assert np.array_equal(np.load(path, allow_pickle=True)["keypoints"], keypoints)
    apply_foot_leg_correction(str(tmp_path / "nope.npz"), "x", "y", np.array([True]))


def test_update_leg_swap_metadata_returns_xor_and_persists_keys(tmp_path):
    kp_path = tmp_path / "keypoints.npz"
    kp_path.write_bytes(b"")  # only its parent dir matters
    pre = np.array([True, True, False, False])
    anchor = np.array([True, False, True, False])

    final_mask = update_leg_swap_metadata(str(kp_path), pre, anchor)

    assert final_mask.tolist() == [False, True, True, False]
    stored = np.load(tmp_path / "post_dp_ankle_fill_mask.npz", allow_pickle=True)
    assert stored["pre_dp_leg_swapped"].tolist() == pre.tolist()
    assert stored["anchor_dp_leg_swapped"].tolist() == anchor.tolist()
    assert stored["final_leg_swapped"].tolist() == [False, True, True, False]
    assert stored["dp_leg_swapped"].tolist() == [False, True, True, False]


def test_update_leg_swap_metadata_returns_none_when_both_masks_none(tmp_path):
    assert update_leg_swap_metadata(str(tmp_path / "keypoints.npz"), None, None) is None
