import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker


CONTRACT_DIR = (
    Path(__file__).resolve().parents[2] / "contracts" / "analysis" / "v1"
)
VALID_FIXTURE_DIR = CONTRACT_DIR / "fixtures" / "valid"
INVALID_FIXTURE_DIR = CONTRACT_DIR / "fixtures" / "invalid"
SCHEMAS = sorted(CONTRACT_DIR.glob("*.schema.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validator(schema_path: Path) -> Draft202012Validator:
    schema = _load(schema_path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


@pytest.mark.parametrize("schema_path", SCHEMAS, ids=lambda path: path.stem)
def test_schema_is_valid_draft_2020_12(schema_path: Path):
    Draft202012Validator.check_schema(_load(schema_path))


@pytest.mark.parametrize("schema_path", SCHEMAS, ids=lambda path: path.stem)
def test_valid_fixture_matches_schema(schema_path: Path):
    fixture_path = VALID_FIXTURE_DIR / schema_path.name.replace(".schema", "")
    assert fixture_path.exists(), f"missing valid fixture: {fixture_path}"
    errors = sorted(
        _validator(schema_path).iter_errors(_load(fixture_path)),
        key=lambda error: list(error.path),
    )
    assert not errors, "\n".join(error.message for error in errors)


@pytest.mark.parametrize("schema_path", SCHEMAS, ids=lambda path: path.stem)
def test_valid_fixture_round_trip(schema_path: Path):
    fixture_path = VALID_FIXTURE_DIR / schema_path.name.replace(".schema", "")
    original = _load(fixture_path)
    decoded = json.loads(json.dumps(original, ensure_ascii=False, sort_keys=True))
    _validator(schema_path).validate(decoded)
    assert decoded == original


@pytest.mark.parametrize("schema_path", SCHEMAS, ids=lambda path: path.stem)
def test_invalid_fixture_is_rejected(schema_path: Path):
    fixture_path = INVALID_FIXTURE_DIR / schema_path.name.replace(".schema", "")
    assert fixture_path.exists(), f"missing invalid fixture: {fixture_path}"
    errors = list(_validator(schema_path).iter_errors(_load(fixture_path)))
    assert errors, f"invalid fixture unexpectedly passed: {fixture_path}"


def test_request_camera_indices_are_contiguous_and_sorted():
    request = _load(VALID_FIXTURE_DIR / "analysis-request.json")
    indices = [camera["camera_index"] for camera in request["cameras"]]
    assert indices == list(range(len(indices)))


def test_request_homography_point_counts_match():
    request = _load(VALID_FIXTURE_DIR / "analysis-request.json")
    for camera in request["cameras"]:
        calibration = camera.get("calibration", {})
        image_points = calibration.get("homography_image_points")
        world_points = calibration.get("homography_world_points_m")
        if image_points is not None or world_points is not None:
            assert image_points is not None
            assert world_points is not None
            assert len(image_points) == len(world_points)
