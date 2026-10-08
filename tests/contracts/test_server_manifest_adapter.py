import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from core.contracts import ServerManifestRequest, write_server_manifest


def _write(path: Path, content: bytes = b"fixture") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def _request(tmp_path: Path) -> ServerManifestRequest:
    output_dir = tmp_path / "result"
    video_path = tmp_path / "input.mov"
    model_path = tmp_path / "model.pt"
    _write(video_path, b"video bytes")
    _write(model_path, b"model bytes")

    metrics = _write(output_dir / "metrics.csv", b"speed_mps\n8.7\n")
    angles = _write(output_dir / "angles.csv", b"frame,left_knee_angle\n0,90\n")
    steps = _write(output_dir / "step_events.csv", b"step_index\n1\n")
    ankle = _write(output_dir / "ankle.csv", b"seq_frame\n0\n")
    overlay = _write(output_dir / "output_final.mp4", b"mp4")
    _write(output_dir / "sequential_tracked/input_2D/keypoints.npz", b"pose2d")
    _write(output_dir / "sequential_tracked/input_2D/foot_keypoints.npz", b"feet")
    _write(output_dir / "pose/keypoints_2d.json", b"canonical-wholebody23")
    _write(output_dir / "sequential_tracked/pred_3D/3Dkeypoints.npz", b"pose3d")
    timing = _write(
        output_dir / "timing_report.json",
        json.dumps({
            "timings": [
                {"stage": "Step1/total_tracking", "elapsed_sec": 2.0},
                {"stage": "Step2/hrnet_2d_pose", "elapsed_sec": 3.0},
                {"stage": "Analysis/rerun_3d_angles_after_leg_dp", "elapsed_sec": 1.0},
                {"stage": "Analysis/speed_metrics_from_bbox_map", "elapsed_sec": 0.5},
                {"stage": "Analysis/step_stride_analysis", "elapsed_sec": 0.4},
                {"stage": "Analysis/overlay_original_video", "elapsed_sec": 0.3},
            ]
        }).encode(),
    )
    return ServerManifestRequest(
        analysis_config={
            "cameras": [{"video_path": str(video_path)}],
            "long_jump_final_landing": False,
        },
        legacy_result={
            "metrics_csv": metrics,
            "angles_csv": angles,
            "uncropped_video": overlay,
            "timing_report": timing,
            "step_analysis": {
                "steps_csv": steps,
                "ankle_csv": ankle,
                "detected_steps": 14,
                "avg_cadence_spm": 281.5,
            },
            "total_time": 12.13,
            "avg_velocity": 8.7,
            "avg_acceleration": 0.4,
            "avg_step_length": 1.85,
        },
        output_dir=output_dir,
        request_id="11111111-1111-4111-8111-111111111111",
        run_id="66666666-6666-4666-8666-666666666666",
        comparison_group_id="44444444-4444-4444-8444-444444444444",
        engine_version="97e7d6e",
        model_files={"server-model": model_path},
        created_at=datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc),
    )


def test_server_manifest_maps_legacy_result_and_hashes_files(tmp_path):
    result = write_server_manifest(_request(tmp_path))
    manifest = result.manifest

    assert result.manifest_path.exists()
    assert result.digest_path.read_text().strip() == f"{result.sha256}  manifest.json"
    assert hashlib.sha256(result.manifest_path.read_bytes()).hexdigest() == result.sha256
    assert manifest["compute_location"] == "server"
    assert manifest["status"] == "completed"
    assert manifest["summary"]["detected_steps"] == 14
    assert manifest["summary"]["average_speed_mps"] == 8.7
    assert manifest["input_videos"][0]["sha256"] == hashlib.sha256(b"video bytes").hexdigest()
    assert manifest["engine"]["models"][0]["sha256"] == hashlib.sha256(b"model bytes").hexdigest()
    assert {artifact["type"] for artifact in manifest["artifacts"]} >= {
        "metrics", "angles", "steps", "pose2d", "pose3d", "overlay", "timing"
    }
    assert any(
        artifact["relative_path"] == "pose/keypoints_2d.json"
        for artifact in manifest["artifacts"]
    )
    assert [stage["name"] for stage in manifest["stages"]] == [
        "validating", "prescan", "tracking", "pose2d", "pose3d", "speed", "gait", "export"
    ]
    assert not list(Path(result.manifest_path.parent).glob(".*.tmp"))


def test_server_manifest_has_deterministic_structure(tmp_path):
    request = _request(tmp_path)
    first = write_server_manifest(request)
    first_bytes = first.manifest_path.read_bytes()
    second = write_server_manifest(request)

    assert second.manifest_path.read_bytes() == first_bytes
    assert second.sha256 == first.sha256
    assert [item["relative_path"] for item in second.manifest["artifacts"]] == sorted(
        item["relative_path"] for item in second.manifest["artifacts"]
    )


def test_server_manifest_rejects_missing_input_video(tmp_path):
    request = ServerManifestRequest(
        analysis_config={"cameras": [{"video_path": str(tmp_path / "missing.mov")}]},
        legacy_result={},
        output_dir=tmp_path / "result",
    )

    try:
        write_server_manifest(request)
    except FileNotFoundError as error:
        assert error.filename == str(tmp_path / "missing.mov") or str(error) == str(tmp_path / "missing.mov")
    else:
        raise AssertionError("missing input video must be rejected")
