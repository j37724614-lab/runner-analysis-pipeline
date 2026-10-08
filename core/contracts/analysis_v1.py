"""Map the legacy Server analysis result into AnalysisResultManifest v1."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID, uuid4

from jsonschema import Draft202012Validator, FormatChecker

from core.utils import REPO_ROOT


SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = (
    REPO_ROOT
    / "contracts"
    / "analysis"
    / "v1"
    / "analysis-result-manifest.schema.json"
)
JOINT_ORDER = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]
STAGE_ORDER = (
    "validating", "prescan", "tracking", "pose2d", "pose3d",
    "speed", "gait", "export",
)


@dataclass(frozen=True)
class ServerManifestRequest:
    """Everything needed to publish one legacy Server result as contract v1."""

    analysis_config: Mapping[str, Any]
    legacy_result: Mapping[str, Any]
    output_dir: str | os.PathLike[str]
    request_id: str | UUID = field(default_factory=uuid4)
    run_id: str | UUID = field(default_factory=uuid4)
    comparison_group_id: str | UUID | None = None
    engine_version: str = "unknown"
    model_files: Mapping[str, str | os.PathLike[str]] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class ManifestWriteResult:
    manifest: dict[str, Any]
    manifest_path: Path
    digest_path: Path
    sha256: str


def write_server_manifest(request: ServerManifestRequest) -> ManifestWriteResult:
    """Create, validate, and atomically publish a deterministic Server manifest."""
    output_dir = Path(request.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    timing_rows = _read_timing_rows(request.legacy_result.get("timing_report"), warnings)
    artifacts = _collect_artifacts(request.legacy_result, output_dir, warnings)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "request_id": str(request.request_id),
        "run_id": str(request.run_id),
        "comparison_group_id": (
            str(request.comparison_group_id)
            if request.comparison_group_id is not None
            else None
        ),
        "compute_location": "server",
        "status": _manifest_status(artifacts),
        "created_at": _iso8601(request.created_at),
        "config_sha256": _canonical_sha256(request.analysis_config),
        "engine": {
            "name": "runner-analysis-pipeline",
            "version": request.engine_version,
            "models": _model_descriptors(request.model_files, warnings),
        },
        "input_videos": _input_video_descriptors(request.analysis_config),
        "coordinate_system": {
            "pose2d_origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "pose2d_unit": "original_video_pixel",
            "bbox_format": "x1_y1_x2_y2",
            "joint_order": JOINT_ORDER,
        },
        "stages": _stage_descriptors(timing_rows, artifacts),
        "summary": _summary(request.legacy_result),
        "artifacts": artifacts,
        "warnings": sorted(set(warnings)),
    }
    _validator().validate(manifest)

    payload = _canonical_json(manifest) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    manifest_path = output_dir / "manifest.json"
    digest_path = output_dir / "manifest.sha256"
    _atomic_write(manifest_path, payload)
    _atomic_write(digest_path, f"{digest}  manifest.json\n".encode())
    return ManifestWriteResult(manifest, manifest_path, digest_path, digest)


def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"analysis config contains unsupported value: {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_video_descriptors(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    descriptors = []
    for fallback_index, camera in enumerate(config.get("cameras", [])):
        video = camera.get("video", {})
        video_path = video.get("uri") or camera.get("video_path")
        if not video_path:
            raise ValueError(f"camera {fallback_index} has no video path")
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        descriptors.append({
            "camera_index": int(camera.get("camera_index", fallback_index)),
            "sha256": _sha256_file(path),
        })
    if not descriptors:
        raise ValueError("analysis config must contain at least one camera")
    return sorted(descriptors, key=lambda item: item["camera_index"])


def _model_descriptors(
    model_files: Mapping[str, str | os.PathLike[str]],
    warnings: list[str],
) -> list[dict[str, Any]]:
    descriptors = []
    for name, model_file in sorted(model_files.items()):
        path = Path(model_file)
        if not path.is_file():
            warnings.append(f"model file missing: {name}")
            continue
        descriptors.append({
            "name": name,
            "sha256": _sha256_file(path),
            "compute_units": None,
        })
    return descriptors


def _read_timing_rows(timing_path: Any, warnings: list[str]) -> list[dict[str, Any]]:
    if not timing_path:
        warnings.append("timing report missing")
        return []
    path = Path(timing_path)
    if not path.is_file():
        warnings.append("timing report file missing")
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload.get("timings", []))
    except (OSError, json.JSONDecodeError, TypeError) as error:
        warnings.append(f"timing report unreadable: {error}")
        return []


def _artifact_candidates(result: Mapping[str, Any], output_dir: Path):
    direct = (
        ("metrics_csv", "metrics"),
        ("angles_csv", "angles"),
        ("uncropped_video", "overlay"),
        ("timing_report", "timing"),
    )
    for key, artifact_type in direct:
        if result.get(key):
            yield Path(result[key]), artifact_type, None
    step_analysis = result.get("step_analysis") or {}
    for key, artifact_type in (
        ("steps_csv", "steps"),
        ("ankle_csv", "diagnostics"),
        ("overlay_video", "overlay"),
    ):
        if step_analysis.get(key):
            yield Path(step_analysis[key]), artifact_type, None

    patterns = (
        ("pose/keypoints_2d.json", "pose2d"),
        ("**/input_2D/keypoints.npz", "pose2d"),
        ("**/input_2D/foot_keypoints.npz", "pose2d"),
        ("**/pred_3D/3Dkeypoints.npz", "pose3d"),
    )
    for pattern, artifact_type in patterns:
        for path in output_dir.glob(pattern):
            yield path, artifact_type, None


def _collect_artifacts(
    result: Mapping[str, Any],
    output_dir: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    unique: dict[str, tuple[Path, str, int | None]] = {}
    for path, artifact_type, camera_index in _artifact_candidates(result, output_dir):
        resolved = path.resolve()
        if not resolved.is_file():
            warnings.append(f"artifact missing: {path.name}")
            continue
        try:
            relative = resolved.relative_to(output_dir).as_posix()
        except ValueError:
            warnings.append(f"artifact outside result bundle: {resolved.name}")
            continue
        unique[relative] = (resolved, artifact_type, camera_index)

    artifacts = []
    for relative, (path, artifact_type, camera_index) in sorted(unique.items()):
        digest = _sha256_file(path)
        artifact_id = _stable_artifact_uuid(relative, digest)
        artifacts.append({
            "artifact_id": artifact_id,
            "type": artifact_type,
            "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            "relative_path": relative,
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "camera_index": camera_index,
        })
    return artifacts


def _stable_artifact_uuid(relative_path: str, digest: str) -> str:
    seed = hashlib.sha256(f"{relative_path}\0{digest}".encode()).hexdigest()
    return str(UUID(f"{seed[:8]}-{seed[8:12]}-4{seed[13:16]}-8{seed[17:20]}-{seed[20:32]}"))


def _stage_for_timing(label: str) -> str | None:
    lower = label.lower()
    if "prescan" in lower:
        return "prescan"
    if "step1" in lower or "tracking" in lower:
        return "tracking"
    if "3d" in lower or "angle" in lower:
        return "pose3d"
    if "step2" in lower or "pose" in lower or "hrnet" in lower:
        return "pose2d"
    if "speed" in lower:
        return "speed"
    if "step" in lower or "leg" in lower or "gait" in lower:
        return "gait"
    if any(token in lower for token in ("overlay", "transcode", "archive")):
        return "export"
    return None


def _stage_descriptors(
    timing_rows: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    durations = {stage: 0.0 for stage in STAGE_ORDER}
    observed = {"validating"}
    for row in timing_rows:
        stage = _stage_for_timing(str(row.get("stage", "")))
        if stage:
            observed.add(stage)
            durations[stage] += max(0.0, float(row.get("elapsed_sec", 0.0)))

    artifact_types = {artifact["type"] for artifact in artifacts}
    observed.update({"pose2d"} if "pose2d" in artifact_types else set())
    observed.update({"pose3d"} if "pose3d" in artifact_types else set())
    observed.update({"speed"} if "metrics" in artifact_types else set())
    observed.update({"gait"} if "steps" in artifact_types else set())
    observed.update({"export"} if "overlay" in artifact_types else set())

    return [
        {
            "name": stage,
            "status": "completed" if stage in observed else "skipped",
            "duration_seconds": round(durations[stage], 4) if durations[stage] else None,
            "warnings": [],
        }
        for stage in STAGE_ORDER
    ]


def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
    step_analysis = result.get("step_analysis") or {}
    return {
        "total_time_seconds": max(0.0, float(result.get("total_time") or 0.0)),
        "average_speed_mps": _optional_float(result.get("avg_velocity")),
        "average_acceleration_mps2": _optional_float(result.get("avg_acceleration")),
        "detected_steps": _optional_int(step_analysis.get("detected_steps")),
        "average_cadence_spm": _optional_float(step_analysis.get("avg_cadence_spm")),
        "average_step_length_m": _optional_float(
            result.get("avg_step_length", step_analysis.get("avg_step_length_m"))
        ),
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _manifest_status(artifacts: list[dict[str, Any]]) -> str:
    artifact_types = {artifact["type"] for artifact in artifacts}
    return "completed" if {"metrics", "angles", "steps"} <= artifact_types else "degraded"


def _iso8601(value: datetime | None) -> str:
    timestamp = value or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
