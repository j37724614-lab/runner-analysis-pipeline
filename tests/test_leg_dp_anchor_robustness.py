"""Anchor robustness for the leg-identity DP.

Covers the four changes to how anchors reach the Viterbi solver:
  1. alternation parity is decided by all touchdowns, not only the first one;
  2. a ~2x touchdown gap is treated as a missed step (covered in
     test_detect_steps_characterization.py);
  3. long jump: anchors after the first flight-sized gap are dropped;
  4. a run whose DP would swap most anchored frames is left unswapped.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scripts.analysis.ankle_step_stride as ass  # noqa: E402
from scripts.analysis.ankle_step_stride import (  # noqa: E402
    _leg_identity_anchors,
    apply_anchor_leg_correction,
    detect_steps,
)
from test_apply_anchor_leg_correction_characterization import (  # noqa: E402
    _SWAPPED_MASK,
    _alternating_anchors,
    _synthetic_recon,
    _write_keypoints,
)
from test_detect_steps_characterization import PIXEL_CAL, _running_ankle_rows  # noqa: E402


def test_wrong_first_touchdown_vote_is_outvoted_by_later_touchdowns():
    rows = _running_ankle_rows(n=90)
    # Corrupt only the lower_foot votes around the first touchdown (frame 12).
    for row in rows[8:17]:
        row["lower_foot"] = "right"

    events = detect_steps(rows, PIXEL_CAL, fps=60.0)

    assert [(e["seq_frame"], e["foot"]) for e in events] == [
        (12, "left"), (24, "right"), (36, "left"), (48, "right"),
        (60, "left"), (72, "right"), (84, "left"),
    ]


def _events(frames, cams=None):
    cams = cams or [None] * len(frames)
    feet = ["right", "left"] * len(frames)
    return [
        {"seq_frame": f, "foot": feet[i], "cam": c}
        for i, (f, c) in enumerate(zip(frames, cams))
    ]


def test_long_jump_anchors_stop_at_the_flight_gap():
    events = _events([10, 24, 38, 52, 100, 130, 160])  # 52 -> 100 is the flight

    kept = _leg_identity_anchors(events, truncate_after_flight=True)

    assert [frame for frame, _, _ in kept] == [10, 24, 38, 52]
    assert len(_leg_identity_anchors(events)) == 7  # running keeps every anchor


def test_flight_truncation_ignores_camera_handoff_gaps():
    events = _events([10, 24, 38, 90, 104, 118], cams=[1, 1, 1, 2, 2, 2])

    kept = _leg_identity_anchors(events, truncate_after_flight=True)

    assert len(kept) == 6


def _force_solver(monkeypatch, swapped_fraction):
    def solve(self, t_a, foot_a, t_b, foot_b):
        frames = list(range(t_a, t_b + 1))
        cut = round(len(frames) * swapped_fraction)
        return {frame: int(i < cut) for i, frame in enumerate(frames)}
    monkeypatch.setattr(ass._LegIdentitySolver, "solve_segment", solve)


def test_dp_keeps_pose_identity_when_it_would_swap_most_frames(tmp_path, monkeypatch):
    # A DP that swaps most anchored frames points at wrong anchors, not legs.
    _force_solver(monkeypatch, swapped_fraction=0.8)
    path = _write_keypoints(tmp_path, _synthetic_recon(label_error_frames=()))
    before = np.load(path)["reconstruction"].copy()

    swapped = apply_anchor_leg_correction(str(path), _alternating_anchors())

    assert not swapped.any()
    after = np.load(path)["reconstruction"]
    # identity kept: each leg stays on its own side (only smoothing moved it)
    assert np.all(after[0, :, 3, 0] > after[0, :, 6, 0])
    assert np.all(before[0, :, 3, 0] > before[0, :, 6, 0])


def test_dp_swaps_are_applied_below_the_guard(tmp_path, monkeypatch):
    _force_solver(monkeypatch, swapped_fraction=0.3)
    path = _write_keypoints(tmp_path, _synthetic_recon(label_error_frames=()))

    swapped = apply_anchor_leg_correction(str(path), _alternating_anchors())

    assert swapped.any()


def test_guard_does_not_block_a_normal_partial_correction(tmp_path):
    path = _write_keypoints(tmp_path, _synthetic_recon())

    swapped = apply_anchor_leg_correction(
        str(path), _alternating_anchors(), truncate_after_flight=True,
    )

    assert swapped.astype(int).tolist() == _SWAPPED_MASK
