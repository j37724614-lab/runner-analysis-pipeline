"""Characterization tests for refresh_step_analysis_after_leg_correction().

These lock the current end-to-end behaviour (build per-camera calibrations ->
re-derive every event from the corrected keypoints -> optional long-jump
landing pass -> recompute metrics -> renumber + summarise -> write CSVs) on a
deterministic synthetic clip, so the planned decomposition into one function
per phase can be verified to preserve the returned step_analysis exactly.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis.ankle_step_stride import (  # noqa: E402
    refresh_step_analysis_after_leg_correction,
)

R_ANKLE, L_ANKLE = 3, 6


def _write_inputs(tmp_path, frames=40):
    recon = np.zeros((1, frames, 17, 3), dtype=np.float64)
    for t in range(frames):
        x = 20.0 + 5.0 * t
        phase = 2 * np.pi * t / 12.0
        for joint in range(17):
            recon[0, t, joint] = (x, 80.0, 0.9)
        recon[0, t, R_ANKLE] = (x + 6, 200.0 + 8.0 * np.cos(phase), 0.9)
        recon[0, t, L_ANKLE] = (x - 6, 200.0 + 8.0 * np.cos(phase + np.pi), 0.9)
    kp_path = tmp_path / "keypoints.npz"
    np.savez_compressed(kp_path, reconstruction=recon, valid_frames=np.arange(frames))
    off_path = tmp_path / "offsets.npz"
    np.savez_compressed(
        off_path,
        offsets=np.zeros((frames, 2)),
        orig_frames=np.arange(frames),
        cam_indices=np.zeros(frames, dtype=int),
    )
    return str(kp_path), str(off_path)


def _step_events():
    fps = 60.0
    return [
        {"seq_frame": f, "foot": ft, "seq_time_s": f / fps, "time_s": f / fps, "step_index": i}
        for i, (f, ft) in enumerate(
            [(6, "right"), (18, "left"), (30, "right")], start=1
        )
    ]


def _step_analysis(tmp_path):
    return {
        "step_events": _step_events(),
        "ankle_csv": str(tmp_path / "ankle.csv"),
        "steps_csv": str(tmp_path / "steps.csv"),
    }


def test_returns_input_unchanged_without_step_analysis_or_cameras(tmp_path):
    kp_path, off_path = _write_inputs(tmp_path)
    assert refresh_step_analysis_after_leg_correction(
        {}, {"cameras": [{}]}, str(tmp_path), kp_path, off_path
    ) == {}
    sentinel = {"step_events": _step_events()}
    assert refresh_step_analysis_after_leg_correction(
        sentinel, {}, str(tmp_path), kp_path, off_path
    ) is sentinel


def test_events_rederived_from_corrected_keypoints_pixel_mode(tmp_path):
    kp_path, off_path = _write_inputs(tmp_path)
    step_analysis = _step_analysis(tmp_path)

    result = refresh_step_analysis_after_leg_correction(
        step_analysis, {"cameras": [{}]}, str(tmp_path), kp_path, off_path,
    )

    assert result is step_analysis  # mutated in place and returned
    assert result["rejected_step_events"] == []
    assert result["avg_step_length_m"] is None
    assert result["avg_cadence_spm"] == pytest.approx(276.923077, abs=1e-5)

    events = result["step_events"]
    assert [(e["step_index"], e["seq_frame"], e["foot"]) for e in events] == [
        (1, 6, "right"), (2, 18, "left"), (3, 30, "right"),
    ]
    assert [e["contact_joint"] for e in events] == [
        "right_ankle", "left_ankle", "right_ankle",
    ]
    assert [e["ankle_x"] for e in events] == [56.0, 104.0, 176.0]
    assert [e["ankle_y"] for e in events] == [192.0, 208.0, 192.0]
    assert [e["track_position_px"] for e in events] == [56.0, 104.0, 176.0]
    assert [e["step_length_px"] for e in events] == [None, 48.0, 72.0]
    assert all(e["step_length_m"] is None for e in events)
    assert [e["cadence_spm"] for e in events] == [None, 300.0, 300.0]
    assert Path(step_analysis["steps_csv"]).exists()
    assert Path(step_analysis["ankle_csv"]).exists()


def test_long_jump_flag_without_a_detected_flight_is_a_noop(tmp_path):
    kp_path, off_path = _write_inputs(tmp_path)
    step_analysis = _step_analysis(tmp_path)

    result = refresh_step_analysis_after_leg_correction(
        step_analysis,
        {"cameras": [{}], "long_jump_final_landing": True},
        str(tmp_path), kp_path, off_path,
    )

    events = result["step_events"]
    assert [(e["seq_frame"], e["foot"], e["step_length_px"]) for e in events] == [
        (6, "right", None), (18, "left", 48.0), (30, "right", 72.0),
    ]
    assert result["avg_cadence_spm"] == pytest.approx(276.923077, abs=1e-5)
