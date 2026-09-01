"""Characterization tests for detect_steps().

These lock the current end-to-end behaviour of the step detector (peak
detection -> gap suppression -> raw-peak rescue -> short-contact dedupe ->
alternating foot labels -> event build) on deterministic synthetic ankle
signals, so the planned decomposition of detect_steps() into smaller
single-purpose helpers can be verified to preserve output exactly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis.ankle_step_stride import (  # noqa: E402
    detect_steps,
    _homography_calibration,
    _track_calibration,
)


def _running_ankle_rows(
    n=180, fps=60.0, cam=0, x0=200.0, vx=6.0, stride_period=24,
    flatten=(), low_conf=(),
):
    """Synthetic running: two ankles oscillate in anti-phase, the lower one
    (larger image y) is the stance foot, x advances at a constant speed."""
    rows = []
    for i in range(n):
        phase = 2 * np.pi * i / stride_period
        right_y = 300.0 + 30.0 * np.cos(phase)
        left_y = 300.0 + 30.0 * np.cos(phase + np.pi)
        if i in flatten:
            right_y = left_y = 285.0
        right_x = x0 + vx * i + 5.0
        left_x = x0 + vx * i - 5.0
        if right_y >= left_y:
            foot, lower_x, lower_y = "right", right_x, right_y
        else:
            foot, lower_x, lower_y = "left", left_x, left_y
        conf = 0.1 if i in low_conf else 0.9
        rows.append({
            "seq_frame": i, "offset_index": i, "orig_frame": i,
            "time_s": i / fps, "seq_time_s": i / fps, "cam": cam,
            "right_ankle_x": right_x, "right_ankle_y": right_y, "right_ankle_conf": 0.9,
            "left_ankle_x": left_x, "left_ankle_y": left_y, "left_ankle_conf": 0.9,
            "lower_foot": foot,
            "lower_ankle_x": lower_x, "lower_ankle_y": lower_y, "lower_ankle_conf": conf,
        })
    return rows


PIXEL_CAL = _track_calibration(None, None, None, None)
SCALE_CAL = _track_calibration(None, None, None, 0.01)
HOMOGRAPHY_CAL = _homography_calibration(
    [[100, 400], [500, 400], [500, 200], [100, 200]],
    [[0.0, 0.0], [10.0, 0.0], [10.0, 2.0], [0.0, 2.0]],
)


def test_pixel_calibration_events():
    events = detect_steps(_running_ankle_rows(), PIXEL_CAL, fps=60.0)

    assert [(e["seq_frame"], e["foot"]) for e in events] == [
        (12, "left"), (24, "right"), (36, "left"), (48, "right"),
        (60, "left"), (72, "right"), (84, "left"), (96, "right"),
        (108, "left"), (120, "right"), (132, "left"), (144, "right"),
        (156, "left"), (168, "right"),
    ]
    assert [e["step_length_px"] for e in events] == [
        None, 82.0, 62.0, 82.0, 62.0, 82.0, 62.0,
        82.0, 62.0, 82.0, 62.0, 82.0, 62.0, 82.0,
    ]
    assert all(e["step_length_m"] is None for e in events)
    assert all(e["world_x_m"] is None for e in events)
    assert [e["step_index"] for e in events] == list(range(1, 15))
    assert events[0]["ankle_y"] == 330.0
    assert events[0]["track_position_px"] == 267.0


def test_scale_calibration_reports_meters():
    events = detect_steps(_running_ankle_rows(), SCALE_CAL, fps=60.0)

    assert len(events) == 14
    assert events[0]["step_length_m"] is None
    assert [e["step_length_m"] for e in events[1:]] == pytest.approx(
        [0.82, 0.62, 0.82, 0.62, 0.82, 0.62,
         0.82, 0.62, 0.82, 0.62, 0.82, 0.62, 0.82]
    )


def test_homography_calibration_world_coords_and_range_filter():
    events = detect_steps(_running_ankle_rows(), HOMOGRAPHY_CAL, fps=60.0)

    # Candidates past world_x_max + 0.5 m are dropped by the calibrated-range filter.
    assert [(e["seq_frame"], e["foot"]) for e in events] == [
        (12, "left"), (24, "right"), (36, "left"), (48, "right"),
    ]
    assert [e["world_x_m"] for e in events] == pytest.approx([4.175, 6.225, 7.775, 9.825], abs=1e-4)
    assert [e["world_y_m"] for e in events] == pytest.approx([0.7, 0.7, 0.7, 0.7], abs=1e-4)
    assert all(e["track_position_px"] is None for e in events)
    assert events[0]["step_length_m"] is None
    assert [e["step_length_m"] for e in events[1:]] == pytest.approx([2.05, 1.55, 2.05], abs=1e-4)
    assert all(e["step_length_px"] is None for e in events)


def test_lookahead_rows_do_not_become_events():
    rows = _running_ankle_rows(n=60)
    lookahead = _running_ankle_rows(n=200)[60:80]

    events = detect_steps(rows, PIXEL_CAL, fps=60.0, lookahead_rows=lookahead)

    assert [(e["seq_frame"], e["foot"]) for e in events] == [
        (12, "left"), (24, "right"), (36, "left"), (48, "right"),
    ]


def test_explicit_min_step_frames_and_prominence():
    events = detect_steps(
        _running_ankle_rows(), PIXEL_CAL, fps=60.0,
        min_step_frames=8, prominence=3.0,
    )

    assert [(e["seq_frame"], e["foot"]) for e in events] == [
        (12, "left"), (24, "right"), (36, "left"), (48, "right"),
        (60, "left"), (72, "right"), (84, "left"), (96, "right"),
        (108, "left"), (120, "right"), (132, "left"), (144, "right"),
        (156, "left"), (168, "right"),
    ]


def test_low_confidence_touchdown_is_neither_kept_nor_rescued():
    rows = _running_ankle_rows(n=90, low_conf=range(30, 43))

    events = detect_steps(rows, PIXEL_CAL, fps=60.0)

    assert [(e["seq_frame"], e["foot"]) for e in events] == [
        (12, "left"), (24, "right"), (48, "left"),
        (60, "right"), (72, "left"), (84, "right"),
    ]


def test_narrow_touchdown_in_flattened_stride_is_still_recovered():
    # The low-pass filter suppresses a one-frame dip; the raw-peak rescue path
    # brings it back so the alternating-foot sequence stays intact.
    rows = _running_ankle_rows(n=120)
    for row in rows:
        if 32 <= row["seq_frame"] <= 40:
            row["lower_ankle_y"] = 288.0
    rows[36]["lower_ankle_y"] = 331.0

    events = detect_steps(rows, PIXEL_CAL, fps=60.0)

    assert [(e["seq_frame"], e["foot"]) for e in events] == [
        (12, "left"), (24, "right"), (36, "left"), (48, "right"),
        (60, "left"), (72, "right"), (84, "left"), (96, "right"), (108, "left"),
    ]


def test_empty_input_returns_empty_list():
    assert detect_steps([], PIXEL_CAL, fps=60.0) == []
