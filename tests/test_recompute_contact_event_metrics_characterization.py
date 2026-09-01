"""Characterization tests for _recompute_contact_event_metrics().

These lock the current end-to-end behaviour of the six sequential passes
(project contacts -> reject lateral temporal jumps -> interpolate rejected
world_y -> reject implausible velocity -> recompute global homography step
lengths -> strip internal keys) so the planned split into one function per
pass can be verified to preserve the mutated event fields exactly.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis.ankle_step_stride import (  # noqa: E402
    _homography_calibration,
    _recompute_contact_event_metrics,
    _track_calibration,
)

_HOM_SRC = [[100, 400], [700, 400], [700, 250], [100, 250]]
_HOM_DST = [[0.0, 0.0], [12.0, 0.0], [12.0, 1.8], [0.0, 1.8]]


def _homography_cal(offset_m=0.0):
    cal = _homography_calibration(_HOM_SRC, _HOM_DST)
    cal["camera_offset_m"] = offset_m
    return cal


def _event(step_index, seq_frame, cam, contact_x, contact_y, time_s, event_type="run_step"):
    return {
        "step_index": step_index, "seq_frame": seq_frame, "cam": cam,
        "contact_x": contact_x, "contact_y": contact_y,
        "time_s": time_s, "seq_time_s": time_s,
        "event_type": event_type, "contact_rejection_reason": "",
    }


def test_line_calibration_step_lengths():
    cal = _track_calibration(
        [(100.0, 300.0), (100.0, 340.0)], [(600.0, 300.0), (600.0, 340.0)], 10.0, None,
    )
    cal["camera_offset_m"] = 0.0
    events = [
        _event(1, 10, 0, 150.0, 320.0, 0.10),
        _event(2, 22, 0, 260.0, 320.0, 0.30),
        _event(3, 34, 0, 370.0, 320.0, 0.50),
    ]

    _recompute_contact_event_metrics(events, {0: cal}, {})

    assert [e["track_position_px"] for e in events] == [50.0, 160.0, 270.0]
    assert [e["step_length_px"] for e in events] == [None, 110.0, 110.0]
    assert [e["step_length_m"] for e in events] == pytest.approx([None, 2.2, 2.2])
    assert all(e["world_x_m"] is None for e in events)
    assert all("_homography_raw_world_x" not in e for e in events)


def test_scale_calibration_step_lengths():
    cal = _track_calibration(None, None, None, 0.02)
    cal["camera_offset_m"] = 0.0
    events = [
        _event(1, 10, 0, 150.0, 320.0, 0.10),
        _event(2, 22, 0, 260.0, 320.0, 0.30),
    ]

    _recompute_contact_event_metrics(events, {0: cal}, {})

    assert [e["track_position_px"] for e in events] == [150.0, 260.0]
    assert [e["step_length_px"] for e in events] == [None, 110.0]
    assert [e["step_length_m"] for e in events] == pytest.approx([None, 2.2])


def test_homography_clean_world_coords():
    events = [
        _event(1, 10, 0, 200.0, 330.0, 0.10),
        _event(2, 22, 0, 320.0, 330.0, 0.30),
        _event(3, 34, 0, 440.0, 330.0, 0.50),
        _event(4, 46, 0, 560.0, 330.0, 0.70),
    ]

    _recompute_contact_event_metrics(events, {0: _homography_cal()}, {})

    assert all(e["track_position_px"] is None for e in events)
    assert [e["world_x_m"] for e in events] == pytest.approx([2.0, 4.4, 6.8, 9.2], abs=1e-4)
    assert [e["world_y_m"] for e in events] == pytest.approx([0.84] * 4, abs=1e-4)
    assert [e["step_length_m"] for e in events] == pytest.approx([None, 2.4, 2.4, 2.4], abs=1e-4)
    assert all(e["homography_lateral_valid"] is True for e in events)
    assert all(e["homography_y_interpolated"] is False for e in events)
    assert all("_homography_raw_world_x" not in e for e in events)


def test_homography_interior_lateral_jump_is_flagged_then_y_interpolated():
    # Event 3's contact sits well off the lane; pass 2 flags the temporal jump,
    # pass 3 restores its world_y from the two neighbours and keeps its raw X.
    events = [
        _event(1, 10, 0, 200.0, 330.0, 0.10),
        _event(2, 22, 0, 320.0, 330.0, 0.30),
        _event(3, 34, 0, 440.0, 306.0, 0.50),
        _event(4, 46, 0, 560.0, 330.0, 0.70),
    ]

    _recompute_contact_event_metrics(events, {0: _homography_cal()}, {})

    assert [e["world_x_m"] for e in events] == pytest.approx([2.0, 4.4, 6.8, 9.2], abs=1e-4)
    assert [e["world_y_m"] for e in events] == pytest.approx([0.84] * 4, abs=1e-4)
    assert [e["step_length_m"] for e in events] == pytest.approx([None, 2.4, 2.4, 2.4], abs=1e-4)
    assert [e["homography_y_interpolated"] for e in events] == [False, False, True, False]
    assert events[2]["contact_rejection_reason"] == (
        "homography_lateral_temporal_jump;homography_y_interpolated"
    )
    assert all(e["contact_rejection_reason"] == "" for e in (events[0], events[1], events[3]))


def test_homography_implausible_velocity_is_rejected():
    events = [
        _event(1, 10, 0, 150.0, 330.0, 0.10),
        _event(2, 22, 0, 250.0, 330.0, 0.30),
        _event(3, 34, 0, 350.0, 330.0, 0.50),
        _event(4, 40, 0, 690.0, 330.0, 0.52),
    ]

    _recompute_contact_event_metrics(events, {0: _homography_cal()}, {})

    assert [e["world_x_m"] for e in events] == pytest.approx([1.0, 3.0, 5.0, None], abs=1e-4)
    assert [e["step_length_m"] for e in events] == pytest.approx([None, 2.0, 2.0, None], abs=1e-4)
    assert [e["homography_lateral_valid"] for e in events] == [True, True, True, False]
    assert events[3]["contact_rejection_reason"] == "homography_velocity_implausible"


def test_two_camera_line_cross_camera_bridge():
    cal0 = _track_calibration(
        [(100.0, 300.0), (100.0, 340.0)], [(600.0, 300.0), (600.0, 340.0)], 10.0, None,
    )
    cal0["camera_offset_m"] = 0.0
    cal1 = _track_calibration(
        [(100.0, 300.0), (100.0, 340.0)], [(600.0, 300.0), (600.0, 340.0)], 10.0, None,
    )
    cal1["camera_offset_m"] = 10.0
    events = [
        _event(1, 10, 0, 300.0, 320.0, 0.10),
        _event(2, 22, 0, 500.0, 320.0, 0.30),
        _event(3, 34, 1, 150.0, 320.0, 0.10),
        _event(4, 46, 1, 350.0, 320.0, 0.30),
    ]

    _recompute_contact_event_metrics(events, {0: cal0, 1: cal1}, {})

    assert [e["track_position_px"] for e in events] == [200.0, 400.0, 50.0, 250.0]
    assert [e["step_length_px"] for e in events] == [None, 200.0, None, 200.0]
    # Step 3 spans the camera cut: measured in the shared global metre coordinate.
    assert [e["step_length_m"] for e in events] == pytest.approx([None, 4.0, 3.0, 4.0])
