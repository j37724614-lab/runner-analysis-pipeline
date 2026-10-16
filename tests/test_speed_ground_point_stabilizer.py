import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.tracker_impl import _stabilize_bbox_speed_points


def test_repairs_single_tight_shifted_bbox_measurement():
    points = []
    for index in range(7):
        points.append({
            "center_x": 140.0 + index * 10.0,
            "center_y": 200.0,
            "bottom_y": 300.0,
            "width": 80.0,
            "height": 200.0,
        })
    points[3].update({
        "center_x": 165.0,
        "center_y": 220.0,
        "bottom_y": 260.0,
        "width": 20.0,
        "height": 80.0,
    })

    result = _stabilize_bbox_speed_points(points)

    assert result[3]["center_x"] == pytest.approx(170.0)
    assert result[3]["center_y"] == pytest.approx(200.0)
    assert result[3]["bottom_y"] == pytest.approx(300.0)
    assert result[3]["repaired"] is True


def test_preserves_monotonic_constant_size_track():
    points = [{
        "center_x": 100.0 + index * 15.0,
        "center_y": 200.0,
        "bottom_y": 300.0,
        "width": 80.0,
        "height": 200.0,
    } for index in range(20)]

    result = _stabilize_bbox_speed_points(points)

    assert [point["center_x"] for point in result] == [
        point["center_x"] for point in points
    ]
    assert not any(point["repaired"] for point in result)
