"""Characterization tests for refresh_step_analysis_after_leg_correction().

These lock the current end-to-end behaviour (build per-camera calibrations ->
re-derive every event from the corrected keypoints -> optional long-jump
landing pass -> recompute metrics -> renumber + summarise -> write CSVs) on a
deterministic synthetic clip, so the planned decomposition into one function
per phase can be verified to preserve the returned step_analysis exactly.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis import ankle_step_stride as ankle_module
from scripts.analysis.ankle_step_stride import (
    StepAnalysisRefreshRequest,
    refresh_step_analysis_after_leg_correction,
)


def _refresh(step_analysis, config, output_dir, keypoints, offsets):
    return refresh_step_analysis_after_leg_correction(StepAnalysisRefreshRequest(
        step_analysis=step_analysis,
        config=config,
        output_dir=str(output_dir),
        keypoints_npz=keypoints,
        offsets_npz=offsets,
    ))

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
    assert _refresh({}, {"cameras": [{}]}, tmp_path, kp_path, off_path) == {}
    sentinel = {"step_events": _step_events()}
    assert _refresh(sentinel, {}, tmp_path, kp_path, off_path) is sentinel


def test_events_rederived_from_corrected_keypoints_pixel_mode(tmp_path):
    kp_path, off_path = _write_inputs(tmp_path)
    step_analysis = _step_analysis(tmp_path)

    result = _refresh(
        step_analysis,
        {"cameras": [{}]},
        tmp_path,
        kp_path,
        off_path,
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

    result = _refresh(
        step_analysis,
        {"cameras": [{}], "long_jump_final_landing": True},
        tmp_path,
        kp_path,
        off_path,
    )

    events = result["step_events"]
    assert [(e["seq_frame"], e["foot"], e["step_length_px"]) for e in events] == [
        (6, "right", None), (18, "left", 48.0), (30, "right", 72.0),
    ]
    assert result["avg_cadence_spm"] == pytest.approx(276.923077, abs=1e-5)


def test_opt_in_hybrid_algorithm_writes_result_and_adapts_touchdown(
    tmp_path, monkeypatch,
):
    kp_path, off_path = _write_inputs(tmp_path)
    step_analysis = _step_analysis(tmp_path)

    def fake_detector(request):
        row = request.ankle_rows[30]
        return {
            "detected": True,
            "algorithm": "anchor_step_hybrid_longest_flight",
            "camera_id": 0,
            "takeoff": {"seq_frame": 18},
            "first_airborne": {"seq_frame": 19},
            "last_airborne": {"seq_frame": 29},
            "touchdown": {
                "seq_frame": 30,
                "orig_frame": 30,
                "time_s": 0.5,
                "seq_time_s": 0.5,
                "kind": "observed",
                "foot": "right",
                "contact_joint": "heel",
                "contact_x": float(row["right_ankle_x"]),
                "contact_y": float(row["right_ankle_y"]),
                "contact_conf": 0.9,
            },
            "max_compression": {"seq_frame": 32},
            "flight_duration_frames": 11,
            "flight_duration_seconds": 11 / 60,
            "ground_model": {"source": "anchor_step_hybrid"},
            "candidates": [],
            "debug_trace": [],
            "quality_score": 1.0,
            "needs_review": False,
            "reasons": [],
        }

    monkeypatch.setattr(ankle_module, "detect_running_long_jump", fake_detector)
    monkeypatch.setattr(
        ankle_module,
        "_detect_long_jump_final_landing",
        lambda *_args: pytest.fail("legacy detector must not run in hybrid mode"),
    )
    monkeypatch.setattr(
        ankle_module,
        "_recover_terminal_preflight_contacts",
        lambda *_args: pytest.fail("single-pass hybrid mode must not redetect steps"),
    )

    result = _refresh(
        step_analysis,
        {
            "cameras": [{}],
            "long_jump_final_landing": True,
            "running_long_jump": {
                "enabled": True,
                "algorithm": "anchor_step_hybrid_longest_flight",
            },
        },
        tmp_path,
        kp_path,
        off_path,
    )

    landing = result["step_events"][-1]
    assert landing["event_type"] == "final_landing"
    assert landing["seq_frame"] == 30
    assert landing["contact_selection_reason"] == (
        "long_jump_hybrid_longest_bilateral_flight"
    )
    assert result["running_long_jump_result"]["max_compression"]["seq_frame"] == 32
    result_json = Path(result["running_long_jump_result_json"])
    assert result_json.exists()
    assert "anchor_step_hybrid_longest_flight" in result_json.read_text()
    diagnostic_logs = result["running_long_jump_diagnostic_logs"]
    for path in diagnostic_logs.values():
        assert Path(path).exists()
    frame_log = Path(diagnostic_logs["frame_decisions"]).read_text()
    assert "decision_reason" in frame_log
    assert "left_distance_px" in frame_log
    candidate_log = Path(diagnostic_logs["candidate_decisions"]).read_text()
    assert "selected" in candidate_log


def test_multicamera_boundary_flight_replaces_terminal_false_steps(monkeypatch):
    rows = [
        {
            "seq_frame": frame,
            "orig_frame": frame,
            "time_s": frame / 60,
            "seq_time_s": frame / 60,
            "cam": 1,
            "lower_foot": "left",
            "left_ankle_x": 100.0,
            "left_ankle_y": 200.0,
            "left_ankle_conf": 0.9,
        }
        for frame in (20, 40)
    ]
    prior_contact = {"seq_frame": 19, "cam": 0, "foot": "right"}
    terminal_false_contact = {"seq_frame": 30, "cam": 1, "foot": "left"}
    context = SimpleNamespace(
        ankle_rows=rows,
        foot_contacts_by_seq={},
        fps=60.0,
    )

    def fake_detector(request):
        assert request.config["allow_initial_airborne_at_camera_boundary"] is True
        return {
            "detected": True,
            "algorithm": "anchor_step_hybrid_longest_flight",
            "camera_id": 1,
            "takeoff": None,
            "first_airborne": {"seq_frame": 20},
            "last_airborne": {"seq_frame": 39},
            "touchdown": {
                "seq_frame": 40,
                "orig_frame": 40,
                "time_s": 40 / 60,
                "seq_time_s": 40 / 60,
                "kind": "observed",
                "foot": "left",
                "contact_joint": "heel",
                "contact_x": 100.0,
                "contact_y": 200.0,
                "contact_conf": 0.9,
            },
            "flight_duration_frames": 20,
        }

    monkeypatch.setattr(ankle_module, "detect_running_long_jump", fake_detector)
    events, result = ankle_module._apply_hybrid_running_long_jump(
        [prior_contact, terminal_false_contact],
        context,
        [{}, {}],
        {"enabled": True},
    )

    assert result["takeoff"]["seq_frame"] == 19
    assert result["takeoff"]["kind"] == "previous_camera_contact"
    assert result["event_filter_trace"] == [
        {
            "seq_frame": 19,
            "cam": 0,
            "foot": "right",
            "event_type": "run_step",
            "decision": "retained",
            "reason": "outside_terminal_camera",
        },
        {
            "seq_frame": 30,
            "cam": 1,
            "foot": "left",
            "event_type": "run_step",
            "decision": "dropped",
            "reason": "after_takeoff_reference_frame_19",
        },
        {
            "seq_frame": 40,
            "cam": 1,
            "foot": "left",
            "event_type": "final_landing",
            "decision": "added",
            "reason": "selected_running_long_jump_touchdown",
        },
    ]
    assert [(event["cam"], event["seq_frame"]) for event in events] == [
        (0, 19),
        (1, 40),
    ]


def test_preflight_recovery_replaces_adjacent_duplicate_with_better_contact(
    monkeypatch,
):
    existing = {
        "seq_frame": 3,
        "cam": 0,
        "foot": "left",
        "contact_joint": "left_heel",
        "contact_x": 10.0,
        "contact_y": 20.0,
        "contact_conf": 0.80,
    }
    row = {
        "seq_frame": 4,
        "orig_frame": 53,
        "time_s": 53 / 60,
        "seq_time_s": 4 / 60,
        "cam": 0,
    }
    candidate = {
        "seq_frame": 4,
        "foot": "left",
        "ankle_x": 10.05,
        "ankle_y": 20.0,
        "ankle_conf": 0.95,
    }
    calibration = {
        "mode": "homography",
        "homography": np.eye(3),
    }
    context = SimpleNamespace(
        calibrations={0: calibration},
        ankle_rows=[row],
        rows_by_seq={4: row},
        foot_contacts_by_seq={},
        contact_contexts={0: None},
        fps=60.0,
    )
    recovered_contact = {
        "contact_valid": True,
        "contact_joint": "left_heel",
        "contact_x": 10.05,
        "contact_y": 20.0,
        "contact_conf": 0.95,
        "contact_selection_reason": "heel_confidence_ge_0.50",
    }
    monkeypatch.setattr(
        ankle_module,
        "detect_steps",
        lambda *_args, **_kwargs: [candidate],
    )
    monkeypatch.setattr(
        ankle_module,
        "select_contact_point",
        lambda *_args, **_kwargs: recovered_contact,
    )

    events = [existing]
    ankle_module._recover_terminal_preflight_contacts(
        ankle_module._TerminalPreflightRecoveryRequest(
            refreshed_events=events,
            context=context,
            terminal_camera=0,
            keep_through_sequence=10,
        )
    )

    assert len(events) == 1
    assert events[0]["seq_frame"] == 4
    assert events[0]["contact_conf"] == pytest.approx(0.95)
