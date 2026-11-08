"""Unit tests for the long-jump analysis prototype, on synthetic signals and images only."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.prototypes.longjump import longjump_analysis as la
from scripts.prototypes.longjump.longjump_analysis import (
    BoardContactRule,
    FlightRule,
    ManualMarkRecord,
    PoseTrack,
    RunSpec,
    TakeoffLine,
    board_side_contact,
    box_iou,
    contact_sheet,
    detect_board,
    detect_jump_events,
    fill_nan,
    find_board_contact,
    find_board_contact_with_board,
    find_flight,
    find_sand_landing,
    find_toe_off,
    find_trim_end,
    first_sustained,
    fit_takeoff_line,
    longest_true_run,
    nearest_scored_joint,
    sand_surface_y,
)

JOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
          "right_knee", "left_ankle", "right_ankle", "left_big_toe", "left_small_toe", "left_heel",
          "right_big_toe", "right_small_toe", "right_heel"]
FPS = 60.0
GROUND_Y = 700.0
BOX_HEIGHT = 200.0
FLIGHT_FIRST_AIRBORNE, FLIGHT_LAST_AIRBORNE = 57, 83
APEX, TOUCHDOWN = 70, 90
STANCE_FIRST, STANCE_LAST = 40, 55


def make_track(bottom_y, heights=None, hip_x=None, left_foot_x=None, right_foot_x=None, first_frame=0):
    """Synthetic PoseTrack: every joint at (hip_x, 600) except the hips and foot joints given here."""
    count = len(bottom_y)
    heights = np.full(count, BOX_HEIGHT) if heights is None else np.asarray(heights, float)
    hip_x = np.full(count, 1000.0) if hip_x is None else np.asarray(hip_x, float)
    joints = np.zeros((count, len(JOINTS), 2))
    joints[:, :, 0] = hip_x[:, None]
    joints[:, :, 1] = 600.0
    for side, foot_x in (("left", left_foot_x), ("right", right_foot_x)):
        if foot_x is not None:
            for joint in ("heel", "big_toe", "ankle", "small_toe"):
                joints[:, JOINTS.index(f"{side}_{joint}"), 0] = foot_x
    bottom_y = np.asarray(bottom_y, float)
    boxes = np.stack([hip_x - 50, bottom_y - heights, hip_x + 50, bottom_y], axis=1)
    return PoseTrack(
        frames=np.arange(first_frame, first_frame + count),
        timestamps=np.arange(count) / FPS,
        joints_xy=joints,
        joint_scores=np.ones((count, len(JOINTS))),
        joint_index={name: index for index, name in enumerate(JOINTS)},
        boxes=boxes,
    )


def jump_bottom_y(count=150):
    """y2 on the ground at 700, linear rise to 600 at APEX, linear fall back to 700 at TOUCHDOWN."""
    frames = np.arange(count, dtype=float)
    bottom_y = np.full(count, GROUND_Y)
    rising = (frames >= 50) & (frames <= APEX)
    falling = (frames > APEX) & (frames <= TOUCHDOWN)
    bottom_y[rising] = GROUND_Y - 5 * (frames[rising] - 50)
    bottom_y[falling] = 600 + 5 * (frames[falling] - APEX)
    return bottom_y


def take_off_foot_x(count=150):
    """Foot moving 10 px/frame toward the pit, standing still on the board over STANCE_FIRST..STANCE_LAST."""
    frames = np.arange(count, dtype=float)
    foot_x = 1500 - 10 * frames
    foot_x[STANCE_FIRST:STANCE_LAST + 1] = 1500 - 10 * STANCE_FIRST
    after = frames > STANCE_LAST
    foot_x[after] = 1500 - 10 * STANCE_FIRST - 10 * (frames[after] - STANCE_LAST)
    return foot_x


# --------------------------------------------------------------------------- small helpers

def test_fill_nan_interpolates_inner_and_holds_edges():
    filled = fill_nan([np.nan, 1.0, np.nan, 3.0, np.nan])

    assert filled.tolist() == [1.0, 1.0, 2.0, 3.0, 3.0]


def test_fill_nan_leaves_values_when_fewer_than_two_are_finite():
    filled = fill_nan([np.nan, 5.0, np.nan])

    assert np.isnan(filled[0]) and filled[1] == 5.0 and np.isnan(filled[2])


def test_longest_true_run_returns_half_open_range_and_first_wins_ties():
    mask = np.array([1, 1, 0, 1, 1, 0, 1, 1, 1, 0], bool)

    assert longest_true_run(mask) == (6, 9)
    assert longest_true_run(np.array([1, 1, 0, 1, 1], bool)) == (0, 2)


def test_longest_true_run_respects_min_length_and_handles_no_run():
    mask = np.array([1, 1, 0, 1, 1, 1], bool)

    assert longest_true_run(mask, min_length=4) is None
    assert longest_true_run(np.zeros(5, bool)) is None


def test_first_sustained_needs_the_full_hold_after_start():
    condition = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0], bool)

    assert first_sustained(condition, start=0, hold=3) == 5
    assert first_sustained(condition, start=0, hold=2) == 2
    assert first_sustained(condition, start=6, hold=3) is None


def test_box_iou():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_runner_segmenter_only_infers_inside_expanded_runner_bbox():
    class ArrayProxy:
        def __init__(self, value):
            self.value = np.asarray(value)

        def cpu(self):
            return self

        def numpy(self):
            return self.value

        def __getitem__(self, index):
            return ArrayProxy(self.value[index])

    class FakeModel:
        def __init__(self):
            self.input_shape = None

        def predict(self, image, **_):
            self.input_shape = image.shape
            height, width = image.shape[:2]
            mask = np.ones((1, height, width), np.uint8)
            result = type("Result", (), {})()
            result.boxes = type("Boxes", (), {"xyxy": ArrayProxy([[0, 0, width, height]])})()
            result.masks = type("Masks", (), {"data": ArrayProxy(mask)})()
            return [result]

    frame = np.zeros((200, 200, 3), np.uint8)
    runner_box = np.array([80, 80, 120, 120], float)
    segmenter = object.__new__(la.RunnerSegmenter)
    segmenter.model = FakeModel()
    segmenter.rule = la.SandMarkRule()

    mask = segmenter.runner_mask(frame, runner_box)

    assert segmenter.model.input_shape[:2] == (120, 120)
    assert mask.shape == frame.shape[:2]
    assert mask[40:160, 40:160].all()
    assert not mask[:40].any()
    assert not mask[:, :40].any()


def test_event_review_segments_clean_frame_before_drawing_runner_box(tmp_path):
    class RecordingSegmenter:
        def __init__(self):
            self.input = None

        def runner_mask(self, image, _box):
            self.input = image.copy()
            return None

    track = make_track(
        np.full(2, 80.0),
        heights=np.full(2, 40.0),
        hip_x=np.full(2, 100.0),
    )
    run = RunSpec("jump", tmp_path, "jump", 0, tmp_path / "jump.mp4", tmp_path)
    events = la.JumpEvents(0, 1, 0, None, None, 1)
    analysis = la.JumpAnalysis(run, track, la.TrimResult(2, None), track, events)
    segmenter = RecordingSegmenter()
    review = la.EventReviewVideo(analysis, tmp_path / "review.mp4", segmenter)
    image = np.zeros((100, 200, 3), np.uint8)

    review._draw(image, frame=0, fps=60.0)

    assert segmenter.input.max() == 0


def test_event_review_render_stops_at_detected_trim_frame(monkeypatch, tmp_path):
    class FakeCapture:
        def get(self, _):
            return 100

        def release(self):
            pass

    class FakeWriter:
        def __init__(self, *_):
            pass

        def write(self, _):
            pass

        def close(self):
            pass

    requested_range = []

    def fake_iterate_frames(_video, first, last):
        requested_range.append((first, last))
        return iter(())

    track = make_track(np.full(2, GROUND_Y))
    run = RunSpec("jump", tmp_path, "jump", 0, tmp_path / "jump.mp4", tmp_path)
    events = la.JumpEvents(0, 1, 0, None, None, 1)
    analysis = la.JumpAnalysis(run, track, la.TrimResult(2, None), track, events)
    monkeypatch.setattr(la, "video_geometry", lambda _: (60.0, 1920, 1080))
    monkeypatch.setattr(la.cv2, "VideoCapture", lambda _: FakeCapture())
    monkeypatch.setattr(la, "Mp4Writer", FakeWriter)
    monkeypatch.setattr(la, "iterate_frames", fake_iterate_frames)

    la.EventReviewVideo(analysis, tmp_path / "review.mp4").render()

    assert requested_range == [(0, analysis.last_frame)]


def test_takeoff_line_x_at_interpolates_along_the_slanted_edge():
    line = TakeoffLine(top=(100.0, 10.0), bottom=(130.0, 20.0))

    assert line.x_at(10.0) == pytest.approx(100.0)
    assert line.x_at(15.0) == pytest.approx(115.0)


def test_contact_sheet_pads_an_odd_tile_count_with_a_blank_tile():
    tiles = [np.full((2, 3, 3), value, np.uint8) for value in (10, 20, 30)]

    sheet = contact_sheet(tiles)

    assert sheet.shape == (4, 6, 3)
    assert sheet[2:, 3:].max() == 0


# --------------------------------------------------------------------------- PoseTrack

def test_pose_track_signals():
    hip_x = 1000 - 12 * np.arange(10.0)
    track = make_track(np.full(10, GROUND_Y), hip_x=hip_x, first_frame=100)

    assert track.fps == pytest.approx(FPS)
    assert np.allclose(track.hip_speed_toward_pit(), 12.0)
    assert track.index_by_frame()[105] == 5
    shortened = track.until(4)
    assert shortened.frames.tolist() == [100, 101, 102, 103]
    assert shortened.boxes.shape == (4, 4)


def test_slowest_foot_speed_is_the_same_when_left_and_right_are_swapped():
    moving, still = 1000 - 10 * np.arange(20.0), np.full(20, 500.0)
    left_still = make_track(np.full(20, GROUND_Y), left_foot_x=still, right_foot_x=moving)
    right_still = make_track(np.full(20, GROUND_Y), left_foot_x=moving, right_foot_x=still)

    assert np.array_equal(left_still.slowest_foot_speed(), right_still.slowest_foot_speed())
    assert np.allclose(left_still.slowest_foot_speed(), 0.0)


# --------------------------------------------------------------------------- events

def test_find_flight_returns_the_airborne_frames():
    track = make_track(jump_bottom_y())

    assert find_flight(track) == (FLIGHT_FIRST_AIRBORNE, FLIGHT_LAST_AIRBORNE + 1)


def test_find_flight_ignores_short_hops_and_walking_toward_the_camera():
    bottom_y = np.full(200, GROUND_Y)
    bottom_y[20:26] = GROUND_Y - 60          # 0.1 s run-up hop, shorter than min_flight_s
    bottom_y[120:] = np.linspace(GROUND_Y, 900, 80)   # y2 keeps increasing: approaching the camera

    assert find_flight(make_track(bottom_y)) is None


def test_find_flight_min_duration_is_in_seconds():
    track = make_track(jump_bottom_y())

    assert find_flight(track, FlightRule(min_flight_s=1.0)) is None


def test_find_sand_landing_is_the_first_frame_where_y2_stops_increasing():
    bottom_y = jump_bottom_y()

    assert find_sand_landing(bottom_y, (FLIGHT_FIRST_AIRBORNE, FLIGHT_LAST_AIRBORNE + 1)) == TOUCHDOWN


def test_sand_landing_ignores_a_mid_air_bbox_plateau_after_an_apex_spike():
    # 16Pro jump 07: one-frame +12 px jitter at the apex, then the box stalls for a few
    # frames while the athlete is still in the air.  Landing cannot precede flight end.
    bottom_y = jump_bottom_y()
    bottom_y[APEX + 1] = 615.0
    bottom_y[APEX + 2:APEX + 5] = 612.0

    assert find_sand_landing(bottom_y, (FLIGHT_FIRST_AIRBORNE, FLIGHT_LAST_AIRBORNE + 1)) == TOUCHDOWN


def test_sand_landing_ignores_a_one_frame_spike_late_in_the_descent():
    # After flight end the order constraint no longer helps; a +14 px one-frame box jump
    # would look like "stopped" over the next 3 frames without smoothing.
    bottom_y = jump_bottom_y()
    bottom_y[TOUCHDOWN - 5] += 14

    assert find_sand_landing(bottom_y, (FLIGHT_FIRST_AIRBORNE, FLIGHT_LAST_AIRBORNE + 1)) == TOUCHDOWN


def test_sand_landing_review_flags_missing_and_late_landings():
    flight = (FLIGHT_FIRST_AIRBORNE, FLIGHT_LAST_AIRBORNE + 1)
    late = FLIGHT_LAST_AIRBORNE + round(0.25 * FPS) + 1

    assert la.review_sand_landing(TOUCHDOWN, flight, FPS) == ()
    assert la.review_sand_landing(None, flight, FPS) == ("sand_landing_not_found",)
    assert la.review_sand_landing(late, flight, FPS) == ("sand_landing_late_after_flight",)


def test_detect_jump_events_carries_the_sand_landing_review():
    events = detect_jump_events(make_track(jump_bottom_y()))

    assert events.sand_landing == TOUCHDOWN
    assert events.sand_landing_review == ()


SAND_LANDING_GT = json.loads(
    (Path(__file__).parent / "fixtures/longjump_sand_landing_gt.json").read_text(encoding="utf-8"))["jumps"]


@pytest.mark.parametrize("jump", SAND_LANDING_GT, ids=[jump["name"] for jump in SAND_LANDING_GT])
def test_sand_landing_regression_against_cvat_ground_truth(jump):
    """Real bbox signals of every jump with CVAT GT; add new GT jumps to the fixture."""
    bottom_y, heights = np.array(jump["bottom_y"]), np.array(jump["box_height"])
    frames = np.array(jump["frames"])
    track = make_track(bottom_y, heights=heights)
    track.frames = frames
    track.timestamps = np.arange(len(frames)) / jump["fps"]
    flight = find_flight(track)

    landing = find_sand_landing(track.bottom_y, flight)

    assert abs(int(frames[landing]) - jump["gt_sand_landing_frame"]) <= 3
    assert la.review_sand_landing(landing, flight, jump["fps"]) == ()


def test_find_board_contact_walks_back_to_the_start_of_the_still_foot_run():
    track = make_track(jump_bottom_y(), left_foot_x=take_off_foot_x(), right_foot_x=1500 - 10 * np.arange(150.0))

    assert find_board_contact(track, FLIGHT_FIRST_AIRBORNE) == STANCE_FIRST


def test_find_board_contact_gives_up_when_the_foot_is_not_still_just_before_flight():
    track = make_track(jump_bottom_y(), left_foot_x=take_off_foot_x(), right_foot_x=1500 - 10 * np.arange(150.0))

    assert find_board_contact(track, FLIGHT_FIRST_AIRBORNE, BoardContactRule(max_gap_before_flight=1)) is None


def test_find_board_contact_stops_after_max_frames_back():
    track = make_track(jump_bottom_y(), left_foot_x=take_off_foot_x(), right_foot_x=1500 - 10 * np.arange(150.0))

    contact = find_board_contact(track, FLIGHT_FIRST_AIRBORNE, BoardContactRule(max_frames_back=5))

    assert contact == FLIGHT_FIRST_AIRBORNE - 5


def test_find_toe_off_is_the_last_frame_of_the_still_foot_run():
    track = make_track(jump_bottom_y(), left_foot_x=take_off_foot_x(), right_foot_x=1500 - 10 * np.arange(150.0))

    assert find_toe_off(track, STANCE_FIRST) == STANCE_LAST


def test_find_toe_off_is_none_without_a_board_contact():
    track = make_track(jump_bottom_y(), left_foot_x=take_off_foot_x(), right_foot_x=1500 - 10 * np.arange(150.0))

    assert find_toe_off(track, None) is None


def test_board_aware_contact_identifies_the_foot_that_stops_on_the_board():
    count = 40
    moving = 900 - 10 * np.arange(count, dtype=float)
    left = moving.copy()
    left[10:16] = 700.0
    left[16:] = 700.0 - 10 * np.arange(1, count - 15)
    track = make_track(
        np.full(count, GROUND_Y),
        left_foot_x=left,
        right_foot_x=moving,
    )
    board = np.zeros((800, 1000), bool)
    board[580:621, 680:721] = True

    contact = find_board_contact_with_board(track, flight_start=20, board_mask=board)

    assert contact.foot_side == "left"
    assert contact.board_contact_index == 11
    assert contact.toe_off_index == 15
    assert contact.takeoff_position == "ON_BOARD"


def test_detect_jump_events_uses_2d_foot_speed_to_select_last_support_foot():
    count = 150
    left_x = take_off_foot_x(count)
    right_x = np.full(count, 705.0)
    track = make_track(
        jump_bottom_y(count),
        left_foot_x=left_x,
        right_foot_x=right_x,
    )
    # The right foot has no horizontal movement but is moving rapidly in y, so
    # it is a swing foot rather than the planted take-off foot.
    right_y = 300.0 + 10.0 * np.arange(count)
    for joint in ("heel", "big_toe", "small_toe", "ankle"):
        track.joints_xy[:, JOINTS.index(f"right_{joint}"), 1] = right_y
    board = np.zeros((1800, 1000), bool)
    board[:, 690:711] = True

    events = detect_jump_events(track, board_mask=board)

    assert events.takeoff_assessment.foot_side == "left"
    assert events.board_contact == 40
    assert events.toe_off == 55


def test_detect_jump_events_prefers_board_touching_foot_over_later_slow_swing_foot():
    count = 150
    track = make_track(
        jump_bottom_y(count),
        left_foot_x=np.full(count, 1200.0),
        right_foot_x=take_off_foot_x(count),
    )
    board = np.zeros((800, 1600), bool)
    board[580:621, 1080:1121] = True

    events = detect_jump_events(track, board_mask=board)

    assert events.takeoff_assessment.foot_side == "right"
    assert events.board_contact == 40
    assert events.toe_off == 55


def test_board_aware_contact_reports_toe_clearance_and_crossing():
    count = 40
    moving = 900 - 10 * np.arange(count, dtype=float)
    left = moving.copy()
    left[10:16] = 700.0
    left[16:] = 700.0 - 10 * np.arange(1, count - 15)
    track = make_track(
        np.full(count, GROUND_Y),
        left_foot_x=left,
        right_foot_x=moving,
    )
    board = np.zeros((800, 1000), bool)
    board[580:621, 680:721] = True
    takeoff_line = TakeoffLine((705.0, 580.0), (705.0, 620.0))

    contact = find_board_contact_with_board(
        track,
        flight_start=20,
        board_mask=board,
        takeoff_line=takeoff_line,
    )

    assert contact.min_toe_clearance_px == pytest.approx(-5.0)
    assert contact.clearance_frame_index == 11
    assert contact.toe_over_line is True
    assert contact.takeoff_position == "OVER_LINE"


def test_takeoff_stance_is_detected_and_classified_before_board_when_foot_is_short():
    count = 40
    moving = 1000 - 10 * np.arange(count, dtype=float)
    left = moving.copy()
    left[10:16] = 800.0
    left[16:] = 800.0 - 10 * np.arange(1, count - 15)
    track = make_track(
        np.full(count, GROUND_Y),
        left_foot_x=left,
        right_foot_x=moving,
    )
    board = np.zeros((800, 1200), bool)
    board[580:621, 680:721] = True
    takeoff_line = TakeoffLine((680.0, 580.0), (680.0, 620.0))

    contact = find_board_contact_with_board(
        track,
        flight_start=20,
        board_mask=board,
        takeoff_line=takeoff_line,
    )

    assert contact.foot_side == "left"
    assert contact.board_contact_index == 11
    assert contact.toe_off_index == 15
    assert contact.takeoff_position == "BEFORE_BOARD"
    assert contact.min_toe_clearance_px == pytest.approx(120.0)
    assert contact.toe_over_line is False


def test_on_board_stance_is_refined_to_the_frames_where_the_sole_remains_near_board():
    count = 40
    moving = 900 - 10 * np.arange(count, dtype=float)
    left = moving.copy()
    left[10:18] = 700.0
    left[18:] = 700.0 - 10 * np.arange(1, count - 17)
    track = make_track(
        np.full(count, GROUND_Y),
        left_foot_x=left,
        right_foot_x=moving,
    )
    # Keep the representative foot x stationary while the sole points rotate
    # away from the board after frame 14.
    for frame in range(15, 18):
        for joint in ("heel", "big_toe", "small_toe"):
            track.joints_xy[frame, JOINTS.index(f"left_{joint}"), 0] = 750.0
        track.joints_xy[frame, JOINTS.index("left_ankle"), 0] = 600.0
    board = np.zeros((800, 1000), bool)
    board[580:621, 680:721] = True

    contact = find_board_contact_with_board(track, flight_start=20, board_mask=board)

    assert contact.takeoff_position == "ON_BOARD"
    assert contact.board_contact_index == 11
    assert contact.toe_off_index == 14


def test_detect_jump_events_reports_video_frame_numbers():
    track = make_track(jump_bottom_y(), left_foot_x=take_off_foot_x(),
                       right_foot_x=1500 - 10 * np.arange(150.0), first_frame=1000)

    events = detect_jump_events(track)

    assert (events.flight_start, events.flight_end) == (1000 + FLIGHT_FIRST_AIRBORNE, 1000 + FLIGHT_LAST_AIRBORNE)
    assert events.apex == 1000 + APEX
    assert events.board_contact == 1000 + STANCE_FIRST
    assert events.toe_off == 1000 + STANCE_LAST
    assert events.sand_landing == 1000 + TOUCHDOWN


def test_detect_jump_events_exposes_board_aware_takeoff_assessment():
    track = make_track(
        jump_bottom_y(),
        left_foot_x=take_off_foot_x(),
        right_foot_x=1500 - 10 * np.arange(150.0),
    )
    board = np.zeros((800, 1600), bool)
    board[580:621, 1080:1121] = True
    line = TakeoffLine((1105.0, 580.0), (1105.0, 620.0))

    events = detect_jump_events(track, board_mask=board, takeoff_line=line)

    assert events.board_contact == STANCE_FIRST
    assert events.toe_off == STANCE_LAST
    assert events.takeoff_assessment.foot_side == "left"
    assert events.takeoff_assessment.toe_over_line is True


def test_detect_jump_events_is_none_without_a_flight():
    assert detect_jump_events(make_track(np.full(100, GROUND_Y))) is None


def test_event_status_follows_the_detected_jump_phases():
    events = la.JumpEvents(
        flight_start=57,
        flight_end=83,
        apex=70,
        board_contact=41,
        toe_off=54,
        sand_landing=90,
    )

    assert la.event_status(events, 40) == "RUN-UP"
    assert la.event_status(events, 41) == "TAKE-OFF SUPPORT"
    assert la.event_status(events, 54) == "FLIGHT"
    assert la.event_status(events, 84) == "DESCENT / CONTACT"
    assert la.event_status(events, 90) == "LANDED"


def test_event_summary_lines_include_frames_and_clip_relative_seconds():
    events = la.JumpEvents(57, 83, 70, 41, 54, 90)

    lines = la.event_summary_lines(events, fps=60.0)

    assert lines == [
        "support:  f41   0.683s",
        "toe-off:  f54   0.900s",
        "apex:  f70   1.167s",
        "flight end:  f83   1.383s",
        "sand landing:  f90   1.500s",
    ]


def test_analyse_jump_uses_the_detected_board_for_takeoff(monkeypatch, tmp_path):
    track = make_track(
        jump_bottom_y(),
        left_foot_x=take_off_foot_x(),
        right_foot_x=1500 - 10 * np.arange(150.0),
    )
    board = np.zeros((800, 1600), bool)
    board[580:621, 1080:1121] = True
    run = RunSpec("jump", tmp_path, "jump", 0, tmp_path / "jump.mp4", tmp_path)
    monkeypatch.setattr(la, "load_pose_track", lambda _: track)
    monkeypatch.setattr(la, "read_first_frame", lambda _: np.zeros((800, 1600, 3), np.uint8))
    monkeypatch.setattr(la, "detect_board", lambda _: la.BoardDetection(board, []))

    analysis = la.analyse_jump(run)

    assert analysis.events.takeoff_assessment is not None
    assert analysis.events.takeoff_assessment.foot_side == "left"


# --------------------------------------------------------------------------- trim

def trim_track(count=200, stop_frame=100, sit_from=105, rise_from=140):
    """Hip runs at 12 px/frame until stop_frame, then stays; bbox height 200 -> 100 (sitting) -> 200 (standing)."""
    frames = np.arange(count, dtype=float)
    hip_x = 1500 - 12 * np.minimum(frames, stop_frame)
    heights = np.full(count, BOX_HEIGHT)
    heights[sit_from:rise_from] = 100.0
    return make_track(np.full(count, GROUND_Y), heights=heights, hip_x=hip_x)


def test_find_trim_end_cuts_when_the_athlete_starts_getting_up():
    trim = find_trim_end(trim_track())

    # np.gradient gives 6 px/frame at the stop frame 100 (= 0.5 x 12, still fast), so the fast
    # segment ends after frame 100 and the athlete is slow from frame 101 on.
    assert trim.fast_segment == (0, 101)
    assert trim.reference_speed == pytest.approx(12.0)
    assert trim.anchor_index == 101
    assert trim.end_index == 141   # height >= 1.2 x 100 from frame 140 on; the cut keeps that frame


def test_find_trim_end_keeps_everything_when_the_athlete_never_stops():
    track = make_track(np.full(120, GROUND_Y), hip_x=1500 - 12 * np.arange(120.0))

    trim = find_trim_end(track)

    assert trim.anchor_index is None
    assert trim.end_index == 120


def test_find_trim_end_keeps_everything_when_the_athlete_never_gets_up():
    trim = find_trim_end(trim_track(rise_from=200))

    assert trim.anchor_index == 101
    assert trim.end_index == 200


# --------------------------------------------------------------------------- board

GREEN_BGR, RED_BGR, WHITE_BGR = (40, 140, 60), (40, 40, 180), (250, 250, 250)


def slanted_board_mask(shape, top_y=100, rows=20, left_x=200, width=12, slope=3):
    mask = np.zeros(shape, bool)
    for row in range(rows):
        x = left_x + slope * row
        mask[top_y + row, x:x + width] = True
    return mask


def runway_scene():
    """Green field, red runway band, a slanted white board in the runway and a white cone on the grass."""
    image = np.zeros((300, 500, 3), np.uint8)
    image[:] = GREEN_BGR
    image[80:140, :] = RED_BGR
    board = slanted_board_mask(image.shape[:2])
    image[board] = WHITE_BGR
    image[200:220, 50:70] = WHITE_BGR
    return image, board


def test_detect_board_picks_the_white_blob_surrounded_by_red_runway():
    image, board = runway_scene()

    detection = detect_board(image)

    # the 3x3 opening in white_mask trims the stair-step corners of the slanted board
    assert not (detection.mask & ~board).any()
    assert detection.mask.sum() >= 0.9 * board.sum()
    assert detection.candidates[0].ring_red > 0.95   # trimmed white corners fall inside the ring
    assert detection.candidates[1].ring_red == pytest.approx(0.0)


def test_detect_board_finds_nothing_without_red_around_white():
    image = np.zeros((300, 500, 3), np.uint8)
    image[:] = GREEN_BGR
    image[200:220, 50:70] = WHITE_BGR

    detection = detect_board(image)

    assert detection.mask is None
    assert len(detection.candidates) == 1


def test_detect_board_ignores_white_areas_larger_than_max_area():
    image, _ = runway_scene()
    image[150:300, 300:500] = WHITE_BGR      # sand pit: one huge white region

    areas = [candidate.area for candidate in detect_board(image).candidates]

    assert max(areas) <= la.BoardRule().max_area


def test_fit_takeoff_line_uses_the_pit_side_edge():
    board = slanted_board_mask((300, 500))

    left = fit_takeoff_line(board, la.PIT_LEFT)
    right = fit_takeoff_line(board, la.PIT_RIGHT)

    assert left.top == pytest.approx((200.0, 100.0), abs=0.01)
    assert left.bottom == pytest.approx((257.0, 119.0), abs=0.01)
    assert right.top == pytest.approx((211.0, 100.0), abs=0.01)


# --------------------------------------------------------------------------- sand mark

def test_board_side_contact_takes_the_column_nearest_the_board_that_touches_the_sand():
    mask = np.zeros((100, 100), bool)
    mask[60:90, 10:30] = True     # foot, lowest y 89
    mask[60:88, 40:60] = True     # buttocks, lowest y 87
    mask[20:50, 70:90] = True     # arm in the air, lowest y 49

    assert board_side_contact(mask, sand_y=90, tolerance=3) == (59, 87)
    assert board_side_contact(mask, sand_y=90, tolerance=1) == (29, 89)
    assert board_side_contact(mask, sand_y=99, tolerance=1) is None
    assert board_side_contact(np.zeros((10, 10), bool), sand_y=5, tolerance=1) is None


def test_sand_surface_y_is_the_largest_foot_y_at_landing():
    track = make_track(np.full(5, GROUND_Y), first_frame=10)
    track.joints_xy[2, JOINTS.index("left_heel"), 1] = 690
    track.joints_xy[2, JOINTS.index("right_big_toe"), 1] = 702
    track.joints_xy[2, JOINTS.index("left_hip"), 1] = 800     # not a foot joint

    assert sand_surface_y(track, 12) == 702


def test_nearest_scored_joint_skips_low_confidence_joints():
    track = make_track(np.full(3, GROUND_Y))
    track.joints_xy[1, JOINTS.index("right_eye")] = (500, 700)
    track.joint_scores[1, JOINTS.index("right_eye")] = 0.01
    track.joints_xy[1, JOINTS.index("left_hip")] = (510, 690)

    assert nearest_scored_joint(track, 1, (500, 700), min_score=0.3) == "left_hip"


def test_sand_mark_row_reports_distances_and_manual_comparison():
    run = RunSpec("IMG_TEST", la.REPO, "stem", 0, la.REPO / "video.mp4", la.GT_DATASET_DIR)
    analysis = la.JumpAnalysis(run, None, None, make_track(np.full(5, GROUND_Y), first_frame=130),
                               la.JumpEvents(100, 120, 110, 90, 97, 131))
    mark = la.SandMark(analysis, 700.0, TakeoffLine((1500.0, 690.0), (1520.0, 710.0)), (900, 700), 133,
                       "left_hip", None, None, (880, 701), ManualMarkRecord(135, 890.0, 699.0, "buttocks"))

    row = la.sand_mark_row(mark)

    assert row["mark_part"] == "buttocks"
    assert row["window_end"] == 134
    assert row["takeoff_line_x_at_mark_y"] == 1510.0
    assert row["takeoff_to_mark_px"] == 610.0
    assert row["foot_x_at_landing"] == 880
    assert (row["err_frame"], row["err_x"]) == (-2, 10.0)


def test_box_size_text_reports_width_height_and_their_ratio():
    assert la.box_size_text(np.array([100.0, 200.0, 138.0, 346.0])) == "W=38 L=146 W/L=0.26"


def test_jump_event_row_reports_gt_errors_and_toe_line_clearance(monkeypatch):
    run = RunSpec("IMG_TEST", la.REPO, "stem", 0, la.REPO / "video.mp4", la.GT_DATASET_DIR)
    track = make_track(np.full(5, GROUND_Y), first_frame=130)
    takeoff = la.BoardContactAssessment("left", 1, 3, -4.5, 2, True, "OVER_LINE", 0.0)
    events = la.JumpEvents(130, 134, 132, 131, 133, 134, takeoff)
    analysis = la.JumpAnalysis(run, track, la.TrimResult(5, (0, 1)), track, events)
    monkeypatch.setattr(
        la,
        "load_event_ground_truth",
        lambda _: {"board_contact": 130, "toe_off": 133, "sand_landing_foot": 135},
    )

    row = la.jump_event_row(analysis)

    assert row["board_contact_error"] == 1
    assert row["toe_off_error"] == 0
    assert row["sand_landing_error"] == -1
    assert row["takeoff_foot"] == "left"
    assert row["min_toe_clearance_px"] == -4.5
    assert row["clearance_frame"] == 132
    assert row["toe_over_line"] is True
    assert row["takeoff_position"] == "OVER_LINE"
    assert row["min_board_distance_px"] == 0.0


# --------------------------------------------------------------------------- loading

def test_load_pose_track_keeps_valid_frames_and_the_selected_runner_box(tmp_path):
    joints = [{"x": float(i), "y": float(i + 1), "score": 0.9} for i in range(3)]
    frames = [{"valid": True, "source_frame": 10, "timestamp_seconds": 0.0, "joints": joints},
              {"valid": False, "source_frame": 11, "timestamp_seconds": 0.1, "joints": joints},
              {"valid": True, "source_frame": 12, "timestamp_seconds": 0.2, "joints": joints}]
    (tmp_path / "pose").mkdir()
    (tmp_path / "pose/keypoints_2d.json").write_text(
        json.dumps({"joint_order": ["a", "b", "c"], "frames": frames}))
    (tmp_path / "clip_selected_runner.json").write_text(json.dumps([{"track_id": 7}]))
    (tmp_path / "clip_all_tracks.csv").write_text(
        "frame_idx,track_id,bx1,by1,bx2,by2\n10,7,1,2,3,4\n10,8,9,9,9,9\n11,7,5,5,5,5\n")
    run = RunSpec("clip", tmp_path, "clip", 0, tmp_path / "clip.mp4", tmp_path)

    track = la.load_pose_track(run)

    assert track.frames.tolist() == [10, 12]
    assert track.joint_index == {"a": 0, "b": 1, "c": 2}
    assert track.boxes[0].tolist() == [1, 2, 3, 4]
    assert np.isnan(track.boxes[1]).all()


def test_load_event_ground_truth_adds_the_clip_offset(tmp_path, monkeypatch):
    gt_csv = tmp_path / "gt.csv"
    gt_csv.write_text("video,event,clip_frame\nIMG_0319.MOV,board_contact,85\nIMG_0320.MOV,board_contact,85\n")
    other_csv = tmp_path / "other.csv"
    other_csv.write_text("video,event,clip_frame\nIMG_0319.MOV,toe_off,92\n")
    monkeypatch.setattr(la, "EVENT_GT_CSVS", (gt_csv, other_csv))
    run = RunSpec("IMG_0319", tmp_path, "IMG_0319", 604, tmp_path / "v.MOV", tmp_path)

    assert la.load_event_ground_truth(run) == {"board_contact": 689, "toe_off": 696}
