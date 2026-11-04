"""Tests for sparse fast-person motion detection used by production prescan."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import tracking
from core.tracking.prescan import (
    _FastPersonMotionDetector,
    _PersonObservation,
)


def _person(x, y=100.0, height=100.0):
    return _PersonObservation(float(x), float(y), float(height))


def test_stationary_person_never_marks_prescan_hit(monkeypatch):
    monkeypatch.setattr(tracking, "PRESCAN_MOTION_MIN_SAMPLES", 3)
    detector = _FastPersonMotionDetector(fps=60.0, stride=15)

    updates = [
        detector.update(frame, [_person(x)])
        for frame, x in [(0, 100), (15, 103), (30, 101), (45, 102)]
    ]

    assert all(not update.hit_frames for update in updates)


def test_fast_large_motion_backfills_confirmation_window(monkeypatch):
    monkeypatch.setattr(tracking, "PRESCAN_MOTION_MIN_SAMPLES", 3)
    detector = _FastPersonMotionDetector(fps=60.0, stride=15)

    detector.update(0, [_person(100)])
    detector.update(15, [_person(150)])
    update = detector.update(30, [_person(200)])

    assert update.hit_frames == (0, 15, 30)
    assert update.running_tracks == 1
    assert update.max_displacement_heights == 1.0
    assert update.max_speed_heights_per_sec == 2.0
    assert update.max_straightness == 1.0


def test_out_and_back_box_jitter_is_not_running(monkeypatch):
    monkeypatch.setattr(tracking, "PRESCAN_MOTION_MIN_SAMPLES", 3)
    detector = _FastPersonMotionDetector(fps=60.0, stride=15)

    detector.update(0, [_person(100)])
    detector.update(15, [_person(160)])
    update = detector.update(30, [_person(100)])

    assert update.hit_frames == ()
    assert update.max_straightness == 0.0


def test_moving_person_is_found_next_to_stationary_people(monkeypatch):
    monkeypatch.setattr(tracking, "PRESCAN_MOTION_MIN_SAMPLES", 3)
    detector = _FastPersonMotionDetector(fps=60.0, stride=15)

    detector.update(0, [_person(100), _person(600)])
    detector.update(15, [_person(102), _person(650)])
    update = detector.update(30, [_person(101), _person(700)])

    assert update.hit_frames == (0, 15, 30)
    assert update.running_tracks == 1
