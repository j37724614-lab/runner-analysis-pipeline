"""`run_analysis_with_manifest()` 不改動 `run_analysis()` 行為，只額外發布 manifest。

規劃書（runner-pose-ondevice/report/integration_guide.md）§13 Step 3：
「在 run_analysis() 外加 manifest adapter，不改動既有演算法結果」。這裡用
monkeypatch 取代真正的 `run_analysis`（避免跑真實 GPU/影片），驗證 wrapper：

1. 原封不動把參數轉給 `run_analysis()`，回傳值完全不變。
2. 用跟 `run_analysis()` 內部一致的 `_resolve_analysis_output_directory` 推導
   出同一個輸出目錄，寫出通過 schema 驗證的 `manifest.json`。
"""
from pathlib import Path

import core.pipeline.final_export as final_export
from core.pipeline import AnalysisOptions
from core.pipeline.final_export import run_analysis_with_manifest


def _write(path: Path, content: bytes = b"fixture") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


def test_run_analysis_with_manifest_forwards_args_and_return_value_unchanged(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "result"
    video_path = _write(tmp_path / "input.mov", b"video bytes")
    fake_result = {"total_time": 1.0, "avg_velocity": 0.0}

    calls = []

    def fake_run_analysis(analysis_config, options):
        calls.append((analysis_config, options))
        return fake_result

    monkeypatch.setattr(final_export, "run_analysis", fake_run_analysis)

    analysis_config = {"cameras": [{"video_path": video_path}]}
    options = AnalysisOptions(output_dest=str(output_dir))

    result = run_analysis_with_manifest(analysis_config, options)

    assert result is fake_result
    assert calls == [(analysis_config, options)]


def test_run_analysis_with_manifest_writes_manifest_to_resolved_output_dir(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "result"
    video_path = _write(tmp_path / "input.mov", b"video bytes")
    model_path = _write(tmp_path / "model.pt", b"model bytes")
    metrics = _write(output_dir / "metrics.csv", b"speed_mps\n8.7\n")
    angles = _write(output_dir / "angles.csv", b"frame,left_knee_angle\n0,90\n")

    monkeypatch.setattr(
        final_export,
        "run_analysis",
        lambda analysis_config, options: {
            "metrics_csv": metrics,
            "angles_csv": angles,
            "uncropped_video": None,
            "timing_report": None,
            "step_analysis": {},
            "total_time": 12.0,
            "avg_velocity": 8.7,
            "avg_acceleration": 0.4,
            "avg_step_length": 1.85,
        },
    )

    analysis_config = {"cameras": [{"video_path": video_path}]}
    options = AnalysisOptions(output_dest=str(output_dir))

    run_analysis_with_manifest(
        analysis_config,
        options,
        engine_version="test-engine",
        model_files={"server-model": model_path},
    )

    manifest_path = output_dir / "manifest.json"
    assert manifest_path.exists()
    assert not list(output_dir.glob(".*.tmp"))
