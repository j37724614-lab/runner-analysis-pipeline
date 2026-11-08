"""Long-jump event analysis prototype (single fixed side camera, athlete runs right -> left).

Algorithm notes: docs/developer/longjump_board_sand_detection.md

    trim       hip speed profile -> cut the clip when the athlete starts to get up in the sand
    events     bbox bottom y2 + foot speed -> flight, board contact, sand landing
    board      first video frame -> white board set in the red runway -> take-off line
    sand mark  person segmentation -> silhouette contact with the sand nearest the take-off line

Image coordinates: y grows downward, so y2 = bbox bottom is largest when the feet are on the ground.

Usage (from this folder):
    python longjump_analysis.py board <video> <output_prefix> [--pit-side left|right]
    python longjump_analysis.py event-video <video> <pose_dir> <file_stem> <output> [--name NAME]
    python longjump_analysis.py full-video <video> <output_dir> [--name NAME]
    python longjump_analysis.py marks
    python longjump_analysis.py videos
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from scipy.ndimage import median_filter

REPO = Path(__file__).resolve().parents[3]
GT_DATASET_DIR = REPO / "data/videos/2026-10-19跳遠最後一機畫面"
NEW_DATASET_DIR = REPO / "data/videos/longjump_test_0117_0118"
EVENT_GT_CSVS = (GT_DATASET_DIR / "cvat_export/longjump_events_gt_20261026.csv",
                 NEW_DATASET_DIR / "cvat_export/longjump_events_gt_20261028.csv")
MANUAL_MARK_CSV = GT_DATASET_DIR / "prescan/vscode/landing_annotations/landing_events.csv"
SEGMENTATION_WEIGHTS = REPO / "models/yolo26x-seg.pt"

PIT_LEFT, PIT_RIGHT = "left", "right"
FOOT_JOINTS_FOR_SPEED = ("heel", "big_toe", "ankle")
FOOT_JOINTS_ON_SAND = ("left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel")
BODY_PART_OF_JOINT = {
    "left_hip": "buttocks", "right_hip": "buttocks",
    "left_wrist": "hand_left", "right_wrist": "hand_right",
    "left_knee": "knee_left", "right_knee": "knee_right",
    **{f"{side}_{joint}": f"foot_{side}"
       for joint in ("ankle", "big_toe", "small_toe", "heel") for side in ("left", "right")},
}

YELLOW, RED, GREEN, WHITE, BLACK = (0, 255, 255), (0, 0, 255), (0, 255, 0), (255, 255, 255), (0, 0, 0)
MAGENTA, CYAN, ORANGE, GREY = (255, 0, 255), (255, 200, 0), (0, 165, 255), (200, 200, 200)


# --------------------------------------------------------------------------- rules (thresholds)

@dataclass(frozen=True)
class FlightRule:
    """Airborne = local ground y2 - y2 > rise_ratio x bbox height, for at least min_flight_s."""
    min_flight_s: float = 0.3
    rise_ratio: float = 0.15
    ground_reach_s: float = 1.5


@dataclass(frozen=True)
class SandLandingRule:
    """Landing = first frame after the steepest descent where y2 rises < settle_px over the next look frames.

    y2 is median-smoothed first, and the search never starts earlier than flight end - search_before_flight_end:
    a one-frame box jitter or a short mid-air stall must not pass as touchdown.  A landing later than
    max_after_flight_s after the flight is flagged for review instead of being trusted silently.
    """
    settle_px: float = 3.0
    look_frames: int = 3
    descent_search_after_flight: int = 10
    smoothing: int = 5
    search_before_flight_end: int = 2
    max_after_flight_s: float = 0.25


@dataclass(frozen=True)
class BoardContactRule:
    """Take-off stance timing from normalized 2D foot speed, followed by board classification."""
    max_still_speed_per_s: float = 2.7
    speed_smoothing: int = 3
    board_distance_ratio: float = 0.03
    max_gap_before_flight: int = 6
    max_frames_back: int = 25


@dataclass(frozen=True)
class TrimRule:
    """Cut the clip when the athlete starts getting up after sitting in the sand."""
    coarse_percentile: float = 95
    fast_ratio: float = 0.5
    slow_ratio: float = 0.35
    slow_hold: int = 10
    settle_window: int = 60
    rise_ratio: float = 1.2
    rise_hold: int = 5
    speed_smoothing: int = 9
    height_smoothing: int = 5


@dataclass(frozen=True)
class BoardRule:
    """White blob (low S, high V) whose surrounding ring is mostly red runway."""
    ring_px: int = 6
    min_area: int = 150
    max_area: int = 5000
    min_ring_red: float = 0.5


@dataclass(frozen=True)
class SandMarkRule:
    """Silhouette column touches the sand when its lowest pixel y >= sand y - contact_ratio x bbox height."""
    contact_ratio: float = 0.03
    min_joint_score: float = 0.3
    image_size: int = 1280
    person_confidence: float = 0.2
    # One bbox width/height on every side gives YOLO enough body context while
    # still excluding distant bystanders from the segmentation input.
    bbox_margin_ratio: float = 1.0
    min_runner_iou: float = 0.1


# --------------------------------------------------------------------------- data

@dataclass(frozen=True)
class RunSpec:
    """One jump video plus where its pipeline outputs live."""
    name: str
    pose_dir: Path
    file_stem: str
    clip_to_run_offset: int
    video: Path
    dataset_dir: Path


def _gt_run(name: str, pose_subdir: str, file_stem: str, offset: int, video: str) -> RunSpec:
    return RunSpec(name, GT_DATASET_DIR / pose_subdir, file_stem, offset, GT_DATASET_DIR / video, GT_DATASET_DIR)


def _new_run(name: str) -> RunSpec:
    return RunSpec(name, NEW_DATASET_DIR / name, name, 0, REPO / f"{name}.MOV", NEW_DATASET_DIR)


RUNS = (
    _gt_run("IMG_0311", "hrnet_preannotation/IMG_0311", "IMG_0311_valid_ranges", 0,
            "prescan/vscode/IMG_0311_valid_ranges.mp4"),
    _gt_run("IMG_0312", "hrnet_preannotation/IMG_0312", "IMG_0312_valid_ranges", 0,
            "prescan/vscode/IMG_0312_valid_ranges.mp4"),
    _gt_run("IMG_0314", "hrnet_preannotation/IMG_0314", "IMG_0314_valid_ranges", 0,
            "prescan/vscode/IMG_0314_valid_ranges.mp4"),
    _gt_run("IMG_0319", "hrnet_original_mov/IMG_0319", "IMG_0319", 604, "IMG_0319.MOV"),
    _gt_run("IMG_0320", "hrnet_original_mov/IMG_0320", "IMG_0320", 396, "IMG_0320.MOV"),
    _new_run("1_3IMG_0117"),
    _new_run("2_3IMG_0118"),
)


def fill_nan(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs (unchanged when fewer than two finite values)."""
    filled = np.asarray(values, float).copy()
    finite = np.isfinite(filled)
    if finite.sum() >= 2:
        filled[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), filled[finite])
    return filled


@dataclass
class PoseTrack:
    """Per-frame signals of the selected runner; index i <-> video frame frames[i]."""
    frames: np.ndarray
    timestamps: np.ndarray
    joints_xy: np.ndarray
    joint_scores: np.ndarray
    joint_index: dict
    boxes: np.ndarray

    @property
    def fps(self) -> float:
        return (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])

    @property
    def bottom_y(self) -> np.ndarray:
        return fill_nan(self.boxes[:, 3])

    @property
    def box_heights(self) -> np.ndarray:
        return fill_nan(self.boxes[:, 3] - self.boxes[:, 1])

    def joint_xy(self, name: str) -> np.ndarray:
        return self.joints_xy[:, self.joint_index[name]]

    def hip_speed_toward_pit(self) -> np.ndarray:
        hip_x = (self.joint_xy("left_hip")[:, 0] + self.joint_xy("right_hip")[:, 0]) / 2
        return -np.gradient(hip_x)

    def slowest_foot_speed(self) -> np.ndarray:
        """|vx| of the slower foot, so a left/right keypoint swap cannot affect it."""
        speeds = []
        for side in ("left", "right"):
            foot_x = np.mean([self.joint_xy(f"{side}_{joint}")[:, 0] for joint in FOOT_JOINTS_FOR_SPEED], axis=0)
            speeds.append(np.abs(np.gradient(foot_x)))
        return np.minimum(*speeds)

    def foot_speed_2d_per_s(self, side: str, smoothing: int = 3) -> np.ndarray:
        """2D speed of one foot, normalized by bbox height and seconds."""
        points = np.mean(
            [self.joint_xy(f"{side}_{joint}") for joint in FOOT_JOINTS_FOR_SPEED],
            axis=0,
        )
        points = np.column_stack([fill_nan(points[:, axis]) for axis in range(2)])
        speed_px_per_frame = np.linalg.norm(np.gradient(points, axis=0), axis=1)
        normalized = speed_px_per_frame * self.fps / self.box_heights
        return median_filter(normalized, size=smoothing, mode="nearest")

    def index_by_frame(self) -> dict:
        return {int(frame): index for index, frame in enumerate(self.frames)}

    def until(self, end_index: int) -> PoseTrack:
        return PoseTrack(self.frames[:end_index], self.timestamps[:end_index], self.joints_xy[:end_index],
                         self.joint_scores[:end_index], self.joint_index, self.boxes[:end_index])


def load_pose_track(run: RunSpec) -> PoseTrack:
    """Valid frames of pose/keypoints_2d.json + the selected runner's tracked bbox."""
    document = json.loads((run.pose_dir / "pose/keypoints_2d.json").read_text())
    valid = [frame for frame in document["frames"] if frame["valid"]]
    frames = np.array([int(frame["source_frame"]) for frame in valid])
    return PoseTrack(
        frames=frames,
        timestamps=np.array([float(frame["timestamp_seconds"]) for frame in valid]),
        joints_xy=np.array([[[joint["x"], joint["y"]] for joint in frame["joints"]] for frame in valid]),
        joint_scores=np.array([[joint["score"] for joint in frame["joints"]] for frame in valid]),
        joint_index={name: index for index, name in enumerate(document["joint_order"])},
        boxes=_runner_boxes(run, frames),
    )


def _runner_boxes(run: RunSpec, frames: np.ndarray) -> np.ndarray:
    selected = json.loads((run.pose_dir / f"{run.file_stem}_selected_runner.json").read_text())
    track_id = selected[0]["track_id"]
    with open(run.pose_dir / f"{run.file_stem}_all_tracks.csv") as tracks:
        box_by_frame = {int(row["frame_idx"]): [float(row[key]) for key in ("bx1", "by1", "bx2", "by2")]
                        for row in csv.DictReader(tracks) if int(row["track_id"]) == track_id}
    return np.array([box_by_frame.get(int(frame), [np.nan] * 4) for frame in frames])


def load_event_ground_truth(run: RunSpec) -> dict:
    """CVAT GT events {event: run frame}; empty for videos without GT."""
    ground_truth = {}
    for gt_csv in EVENT_GT_CSVS:
        with open(gt_csv) as gt_file:
            ground_truth.update({row["event"]: int(row["clip_frame"]) + run.clip_to_run_offset
                                 for row in csv.DictReader(gt_file) if row["video"].startswith(run.name)})
    return ground_truth


@dataclass(frozen=True)
class ManualMarkRecord:
    frame: int
    x: float
    y: float
    body_part: str


def load_manual_mark(run: RunSpec) -> Optional[ManualMarkRecord]:
    """Provisional manual record of the first non-foot sand contact (landing_events.csv)."""
    with open(MANUAL_MARK_CSV) as records:
        for row in csv.DictReader(records):
            if row["video"].startswith(run.name):
                return ManualMarkRecord(int(row["non_foot_contact_frame_1"]) + run.clip_to_run_offset,
                                        float(row["non_foot_contact_x_px"]), float(row["non_foot_contact_y_px"]),
                                        row["non_foot_body_part"])
    return None


# --------------------------------------------------------------------------- trim

@dataclass(frozen=True)
class TrimResult:
    end_index: int
    fast_segment: Optional[tuple]
    reference_speed: Optional[float] = None
    anchor_index: Optional[int] = None


def longest_true_run(mask: np.ndarray, min_length: int = 1) -> Optional[tuple]:
    """[start, end) of the longest run of True values (first one wins ties), or None."""
    best, start = None, None
    for index, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            length = index - start
            if length >= min_length and (best is None or length > best[1] - best[0]):
                best = (start, index)
            start = None
    return best


def find_trim_end(track: PoseTrack, rule: TrimRule = TrimRule()) -> TrimResult:
    """Exclusive end index of the jump clip; keeps the whole clip when a step fails.

    1. Coarse speed = coarse_percentile of hip speed toward the pit.
    2. Fast segment = longest run with speed >= fast_ratio x coarse (run-up + flight); reference = its median.
    3. Anchor = after the fast segment, first time speed < slow_ratio x reference for slow_hold frames.
    4. End = the athlete starts getting up (see find_getting_up_index).
    """
    speed = median_filter(track.hip_speed_toward_pit(), size=rule.speed_smoothing, mode="nearest")
    coarse_speed = np.percentile(speed, rule.coarse_percentile)
    fast_segment = longest_true_run(speed >= rule.fast_ratio * coarse_speed)
    if fast_segment is None:
        return TrimResult(len(speed), None)
    reference_speed = float(np.median(speed[fast_segment[0]:fast_segment[1]]))
    anchor = first_sustained(speed < rule.slow_ratio * reference_speed, fast_segment[1], rule.slow_hold)
    end_index = len(speed) if anchor is None else find_getting_up_index(track, anchor, rule)
    return TrimResult(end_index, fast_segment, reference_speed, anchor)


def first_sustained(condition: np.ndarray, start: int, hold: int) -> Optional[int]:
    """First index >= start where condition stays True for `hold` frames."""
    return next((index for index in range(start, len(condition) - hold) if condition[index:index + hold].all()), None)


def find_getting_up_index(track: PoseTrack, anchor: int, rule: TrimRule) -> int:
    """Lowest bbox height within settle_window of the anchor (sitting), then the first frame
    where the height stays >= rise_ratio x that minimum for rise_hold frames; returns that frame + 1."""
    heights = median_filter(track.box_heights, rule.height_smoothing, mode="nearest")
    sitting = anchor + int(np.argmin(heights[anchor:anchor + rule.settle_window]))
    risen = heights >= rule.rise_ratio * heights[sitting]
    getting_up = first_sustained(risen, sitting, rule.rise_hold)
    return len(heights) if getting_up is None else getting_up + 1


# --------------------------------------------------------------------------- events

@dataclass(frozen=True)
class JumpEvents:
    """Video frame numbers of the detected events (None when not found)."""
    flight_start: int
    flight_end: int
    apex: int
    board_contact: Optional[int]
    toe_off: Optional[int]
    sand_landing: Optional[int]
    takeoff_assessment: Optional[BoardContactAssessment] = None
    sand_landing_review: tuple = ()


def find_flight(track: PoseTrack, rule: FlightRule = FlightRule()) -> Optional[tuple]:
    """[start, end) of the longest airborne run.

    Ground y2 at frame t = the smaller of max(y2) over the ground_reach_s before t and after t
    (a side without frames is skipped), so walking toward the camera after landing is never airborne.
    """
    bottom_y, heights = track.bottom_y, track.box_heights
    reach = max(1, round(rule.ground_reach_s * track.fps))
    airborne = np.zeros(len(bottom_y), bool)
    for index in range(len(bottom_y)):
        ground_y = _local_ground_y(bottom_y, index, reach)
        if ground_y is not None:
            airborne[index] = bottom_y[index] < ground_y - rule.rise_ratio * heights[index]
    return longest_true_run(airborne, min_length=max(1, round(rule.min_flight_s * track.fps)))


def _local_ground_y(bottom_y: np.ndarray, index: int, reach: int) -> Optional[float]:
    before, after = bottom_y[max(0, index - reach):index], bottom_y[index + 1:index + 1 + reach]
    side_maxima = [np.nanmax(window) for window in (before, after) if len(window) and np.isfinite(window).any()]
    return min(side_maxima) if side_maxima else None


def flight_apex(bottom_y: np.ndarray, flight: tuple) -> int:
    """Index of the highest point of the jump (smallest y2) inside the flight [start, end)."""
    flight_start, flight_end = flight
    return flight_start + int(np.nanargmin(bottom_y[flight_start:flight_end]))


def find_sand_landing(bottom_y: np.ndarray, flight: tuple, rule: SandLandingRule = SandLandingRule()) -> Optional[int]:
    """Index of the first frame after the steepest descent where the smoothed y2 stops increasing.

    1. Median-smooth y2 over rule.smoothing frames (removes one-frame box jitter).
    2. Steepest descent between the apex and flight end + descent_search_after_flight.
    3. Search for the stop from max(steepest, flight end - search_before_flight_end): the feet cannot
       reach the sand while the flight detector still sees the athlete clearly airborne.
    """
    smoothed = median_filter(fill_nan(bottom_y), size=rule.smoothing, mode="nearest")
    flight_end = flight[1]
    apex = flight_apex(smoothed, flight)
    descent = np.diff(smoothed)
    search_end = min(flight_end + rule.descent_search_after_flight, len(descent))
    steepest = apex + int(np.nanargmax(descent[apex:search_end]))
    for index in range(max(steepest, flight_end - rule.search_before_flight_end), len(smoothed) - rule.look_frames):
        if np.nanmax(smoothed[index:index + rule.look_frames + 1]) - smoothed[index] < rule.settle_px:
            return index
    return None


def review_sand_landing(sand_index: Optional[int], flight: tuple, fps: float,
                        rule: SandLandingRule = SandLandingRule()) -> tuple:
    """Reasons to check the landing by hand (empty when it is plausible)."""
    if sand_index is None:
        return ("sand_landing_not_found",)
    if sand_index - (flight[1] - 1) > rule.max_after_flight_s * fps:
        return ("sand_landing_late_after_flight",)
    return ()


def find_board_contact(track: PoseTrack, flight_start: int, rule: BoardContactRule = BoardContactRule()) -> Optional[int]:
    """Index where the take-off foot lands: walk back from flight start into the still-foot run, then to its start."""
    still = _foot_still(track, rule)
    index = flight_start
    while index > 0 and not still[index]:
        index -= 1
        if flight_start - index > rule.max_gap_before_flight:
            return None
    while index > 0 and still[index - 1] and flight_start - index < rule.max_frames_back:
        index -= 1
    return index


def find_toe_off(track: PoseTrack, board_contact: Optional[int],
                 rule: BoardContactRule = BoardContactRule()) -> Optional[int]:
    """Index where the take-off foot leaves the board: the last frame of the still-foot run that starts at board contact."""
    if board_contact is None:
        return None
    still = _foot_still(track, rule)
    index = board_contact
    while index + 1 < len(still) and still[index + 1]:
        index += 1
    return index


@dataclass(frozen=True)
class BoardContactAssessment:
    """Take-off stance timing plus its position relative to the board and take-off line."""
    foot_side: Optional[str]
    board_contact_index: Optional[int]
    toe_off_index: Optional[int]
    min_toe_clearance_px: Optional[float] = None
    clearance_frame_index: Optional[int] = None
    toe_over_line: Optional[bool] = None
    takeoff_position: Optional[str] = None
    min_board_distance_px: Optional[float] = None


def find_board_contact_with_board(
    track: PoseTrack,
    flight_start: int,
    board_mask: np.ndarray,
    takeoff_line: Optional[TakeoffLine] = None,
    pit_side: str = PIT_LEFT,
    rule: BoardContactRule = BoardContactRule(),
) -> BoardContactAssessment:
    """Find the last single-foot stance before flight, then classify where it occurred.

    Timing depends on the foot-still run, not board proximity, so an athlete taking
    off short of the white board is still detected.  Board geometry classifies the
    stance as BEFORE_BOARD, ON_BOARD, or OVER_LINE afterwards.
    """
    distance = cv2.distanceTransform((~board_mask.astype(bool)).astype(np.uint8), cv2.DIST_L2, 5)
    candidates = []
    for side in ("left", "right"):
        still = track.foot_speed_2d_per_s(side, rule.speed_smoothing) < rule.max_still_speed_per_s
        near = np.zeros(len(track.frames), bool)
        for index in range(len(track.frames)):
            distances = []
            for joint in ("heel", "big_toe", "small_toe"):
                x, y = track.joint_xy(f"{side}_{joint}")[index]
                if np.isfinite((x, y)).all():
                    column = int(np.clip(round(x), 0, distance.shape[1] - 1))
                    row = int(np.clip(round(y), 0, distance.shape[0] - 1))
                    distances.append(distance[row, column])
            if distances:
                near[index] = min(distances) <= rule.board_distance_ratio * track.box_heights[index]
        stance = still
        index = flight_start
        while index > 0 and not stance[index]:
            index -= 1
            if flight_start - index > rule.max_gap_before_flight:
                break
        if not stance[index] or flight_start - index > rule.max_gap_before_flight:
            continue
        toe_off = index
        while index > 0 and stance[index - 1] and flight_start - index < rule.max_frames_back:
            index -= 1
        board_distances = []
        for stance_index in range(index, toe_off + 1):
            for joint in ("heel", "big_toe", "small_toe"):
                x, y = track.joint_xy(f"{side}_{joint}")[stance_index]
                if np.isfinite((x, y)).all():
                    column = int(np.clip(round(x), 0, distance.shape[1] - 1))
                    row = int(np.clip(round(y), 0, distance.shape[0] - 1))
                    board_distances.append(float(distance[row, column]))
        minimum_board_distance = min(board_distances) if board_distances else float("inf")
        touches_board = bool(near[index:toe_off + 1].any())
        contact_event, toe_off_event = index, toe_off
        if touches_board:
            near_indices = np.flatnonzero(near[index:toe_off + 1]) + index
            contact_event, toe_off_event = int(near_indices[0]), int(near_indices[-1])
        candidates.append(
            (toe_off, minimum_board_distance, side, index, touches_board, contact_event, toe_off_event)
        )
    if not candidates:
        return BoardContactAssessment(None, None, None)
    _, minimum_board_distance, side, _, touches_board, contact, toe_off = max(
        candidates,
        key=lambda candidate: (candidate[4], candidate[0], -candidate[1]),
    )
    if takeoff_line is None:
        position = "ON_BOARD" if touches_board else "BEFORE_BOARD"
        return BoardContactAssessment(
            side,
            contact,
            toe_off,
            takeoff_position=position,
            min_board_distance_px=minimum_board_distance,
        )
    clearances = []
    for index in range(contact, toe_off + 1):
        for joint in ("big_toe", "small_toe"):
            x, y = track.joint_xy(f"{side}_{joint}")[index]
            line_x = takeoff_line.x_at(y)
            clearance = x - line_x if pit_side == PIT_LEFT else line_x - x
            clearances.append((float(clearance), index))
    min_clearance, clearance_index = min(clearances)
    position = "OVER_LINE" if min_clearance < 0 else ("ON_BOARD" if touches_board else "BEFORE_BOARD")
    return BoardContactAssessment(
        side,
        contact,
        toe_off,
        min_clearance,
        clearance_index,
        min_clearance < 0,
        position,
        minimum_board_distance,
    )


def _foot_still(track: PoseTrack, rule: BoardContactRule) -> np.ndarray:
    speeds = [track.foot_speed_2d_per_s(side, rule.speed_smoothing) for side in ("left", "right")]
    return np.minimum(*speeds) < rule.max_still_speed_per_s


def detect_jump_events(
    track: PoseTrack,
    board_mask: Optional[np.ndarray] = None,
    takeoff_line: Optional[TakeoffLine] = None,
    pit_side: str = PIT_LEFT,
) -> Optional[JumpEvents]:
    flight = find_flight(track)
    if flight is None:
        return None
    assessment = None
    if board_mask is not None:
        assessment = find_board_contact_with_board(
            track,
            flight[0],
            board_mask,
            takeoff_line,
            pit_side,
        )
    board_index = (
        assessment.board_contact_index
        if assessment and assessment.board_contact_index is not None
        else find_board_contact(track, flight[0])
    )
    toe_off_index = (
        assessment.toe_off_index
        if assessment and assessment.toe_off_index is not None
        else find_toe_off(track, board_index)
    )
    sand_index = find_sand_landing(track.bottom_y, flight)
    return JumpEvents(_frame_at(track, flight[0]), _frame_at(track, flight[1] - 1),
                      _frame_at(track, flight_apex(track.bottom_y, flight)),
                      _frame_at(track, board_index), _frame_at(track, toe_off_index),
                      _frame_at(track, sand_index), assessment,
                      review_sand_landing(sand_index, flight, track.fps))


def _frame_at(track: PoseTrack, index: Optional[int]) -> Optional[int]:
    return None if index is None else int(track.frames[index])


@dataclass
class JumpAnalysis:
    run: RunSpec
    full_track: PoseTrack
    trim: TrimResult
    track: PoseTrack
    events: Optional[JumpEvents]

    @property
    def last_frame(self) -> int:
        return int(self.track.frames[-1])


def analyse_jump(run: RunSpec) -> JumpAnalysis:
    """Trim first (speed profile), then detect events on the trimmed track."""
    full_track = load_pose_track(run)
    trim = find_trim_end(full_track)
    track = full_track.until(trim.end_index)
    board = detect_board(read_first_frame(run.video))
    if board.mask is None:
        events = detect_jump_events(track)
    else:
        line = fit_takeoff_line(board.mask, PIT_LEFT)
        events = detect_jump_events(track, board.mask, line, PIT_LEFT)
    return JumpAnalysis(run, full_track, trim, track, events)


# --------------------------------------------------------------------------- board

@dataclass(frozen=True)
class BoardCandidate:
    ring_red: float
    area: int
    label: int
    bbox: tuple


@dataclass
class BoardDetection:
    mask: Optional[np.ndarray]
    candidates: list


@dataclass(frozen=True)
class TakeoffLine:
    """Board edge nearest the pit, as its endpoints at the board's smallest and largest y."""
    top: tuple
    bottom: tuple

    def x_at(self, y: float) -> float:
        (x_top, y_top), (x_bottom, y_bottom) = self.top, self.bottom
        return x_top + (y - y_top) * (x_bottom - x_top) / (y_bottom - y_top)


def read_first_frame(video: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read first frame: {video}")
    return frame


def red_runway_mask(image: np.ndarray) -> np.ndarray:
    hue, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    return (((hue <= 10) | (hue >= 165)) & (saturation > 70) & (value > 50)).astype(np.uint8)


def white_mask(image: np.ndarray) -> np.ndarray:
    _, saturation, value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    white = ((saturation < 60) & (value > 160)).astype(np.uint8)
    return cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def detect_board(image: np.ndarray, rule: BoardRule = BoardRule()) -> BoardDetection:
    """Pick the white blob whose ring is most red; no board when that ratio < min_ring_red."""
    red = red_runway_mask(image)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask(image))
    candidates = [_score_candidate(labels, stats[label], label, red, rule) for label in range(1, count)
                  if rule.min_area <= stats[label][cv2.CC_STAT_AREA] <= rule.max_area]
    candidates.sort(key=lambda c: (c.ring_red, c.area, c.label, c.bbox), reverse=True)
    if not candidates or candidates[0].ring_red < rule.min_ring_red:
        return BoardDetection(None, candidates)
    return BoardDetection(labels == candidates[0].label, candidates)


def _score_candidate(labels: np.ndarray, stat: np.ndarray, label: int, red: np.ndarray, rule: BoardRule) -> BoardCandidate:
    x, y, width, height, area = stat
    ring = rule.ring_px
    x0, y0 = max(x - ring, 0), max(y - ring, 0)
    window = (slice(y0, y + height + ring), slice(x0, x + width + ring))
    blob = (labels[window] == label).astype(np.uint8)
    ring_pixels = cv2.dilate(blob, np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)) - blob
    ring_red = float(red[window][ring_pixels > 0].mean())
    return BoardCandidate(ring_red, int(area), label, (int(x), int(y), int(width), int(height)))


def fit_takeoff_line(board_mask: np.ndarray, pit_side: str) -> TakeoffLine:
    """Huber line fit through the pit-side edge pixel of every board row."""
    ys, xs = np.nonzero(board_mask)
    edge = [((xs[ys == y].min() if pit_side == PIT_LEFT else xs[ys == y].max()), y) for y in np.unique(ys)]
    edge = np.array(edge, dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(edge, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()

    def point_at(y: float) -> tuple:
        return float(x0 + (y - y0) * vx / vy), float(y)

    return TakeoffLine(point_at(edge[:, 1].min()), point_at(edge[:, 1].max()))


def find_takeoff_line(video: Path, pit_side: str = PIT_LEFT) -> Optional[TakeoffLine]:
    board = detect_board(read_first_frame(video))
    return None if board.mask is None else fit_takeoff_line(board.mask, pit_side)


# --------------------------------------------------------------------------- sand mark

def box_iou(box_a, box_b) -> float:
    overlap_x = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    overlap_y = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    overlap = overlap_x * overlap_y
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return overlap / (area_a + area_b - overlap + 1e-9)


class RunnerSegmenter:
    """Person segmentation restricted to an expanded tracked-runner bbox."""

    def __init__(self, rule: SandMarkRule = SandMarkRule()):
        from ultralytics import YOLO  # heavy import, only needed for the sand-mark commands
        self.model = YOLO(str(SEGMENTATION_WEIGHTS))
        self.rule = rule

    def runner_mask(self, frame: np.ndarray, runner_box: np.ndarray) -> Optional[np.ndarray]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = np.asarray(runner_box, dtype=float)
        margin_x = (x2 - x1) * self.rule.bbox_margin_ratio
        margin_y = (y2 - y1) * self.rule.bbox_margin_ratio
        left = max(0, int(np.floor(x1 - margin_x)))
        top = max(0, int(np.floor(y1 - margin_y)))
        right = min(width, int(np.ceil(x2 + margin_x)))
        bottom = min(height, int(np.ceil(y2 + margin_y)))
        if right <= left or bottom <= top:
            return None

        roi = frame[top:bottom, left:right]
        result = self.model.predict(roi, imgsz=self.rule.image_size, conf=self.rule.person_confidence,
                                    classes=[0], retina_masks=True, verbose=False)[0]
        if result.masks is None:
            return None
        local_runner_box = np.array([x1 - left, y1 - top, x2 - left, y2 - top])
        overlaps = np.array([
            box_iou(box, local_runner_box) for box in result.boxes.xyxy.cpu().numpy()
        ])
        if overlaps.size == 0 or float(overlaps.max()) < self.rule.min_runner_iou:
            return None
        best = int(overlaps.argmax())
        roi_mask = result.masks.data[best].cpu().numpy() > 0.5
        mask = np.zeros((height, width), dtype=bool)
        mask[top:bottom, left:right] = roi_mask
        return mask


def board_side_contact(mask: np.ndarray, sand_y: float, tolerance: float) -> Optional[tuple]:
    """(x, y) of the silhouette column nearest the board (largest x) whose lowest pixel touches the sand."""
    columns = np.nonzero(mask.any(axis=0))[0]
    if columns.size == 0:
        return None
    lowest_y = np.array([np.nonzero(mask[:, column])[0].max() for column in columns])
    touching = lowest_y >= sand_y - tolerance
    if not touching.any():
        return None
    column = columns[touching].max()
    return int(column), int(lowest_y[columns == column][0])


def sand_surface_y(track: PoseTrack, sand_landing_frame: int) -> float:
    """Largest heel/toe y at the sand-landing frame: the feet sit on the sand there."""
    index = track.index_by_frame()[sand_landing_frame]
    return float(np.nanmax([track.joint_xy(joint)[index, 1] for joint in FOOT_JOINTS_ON_SAND]))


@dataclass
class ContactFrame:
    frame: int
    image: np.ndarray
    mask: np.ndarray
    contact: Optional[tuple]


@dataclass
class RunnerOutline:
    """Silhouette contours of one frame and its board-side sand contact (None before sand landing)."""
    contours: tuple
    contact: Optional[tuple] = None


@dataclass
class SandMark:
    """Nearest sand mark to the take-off line over sand landing .. trim end."""
    analysis: JumpAnalysis
    sand_y: float
    takeoff_line: TakeoffLine
    point: tuple
    frame: int
    joint: str
    image: np.ndarray
    mask: np.ndarray
    foot_contact_at_landing: Optional[tuple]
    manual: Optional[ManualMarkRecord] = None
    outlines: dict = field(default_factory=dict)  # frame -> RunnerOutline over the search window

    @property
    def body_part(self) -> str:
        return BODY_PART_OF_JOINT.get(self.joint, self.joint)

    @property
    def takeoff_x(self) -> float:
        return self.takeoff_line.x_at(self.point[1])


def iterate_frames(video: Path, first: int, last: int) -> Iterator[tuple]:
    """(frame number, image) for first..last, stopping early at the end of the video.

    Decodes from frame 0 instead of seeking: seeking 1_3IMG_0117 / 2_3IMG_0118 lands up to 3 frames off.
    """
    capture = cv2.VideoCapture(str(video))
    for _ in range(first):
        capture.grab()
    for frame in range(first, last + 1):
        ok, image = capture.read()
        if not ok:
            break
        yield frame, image
    capture.release()


def contact_frames(analysis: JumpAnalysis, sand_y: float, segmenter: RunnerSegmenter) -> Iterator[ContactFrame]:
    """Segmented runner + board-side sand contact for every tracked frame of the search window."""
    track, index_by_frame = analysis.track, analysis.track.index_by_frame()
    for frame, image in iterate_frames(analysis.run.video, analysis.events.sand_landing, analysis.last_frame):
        index = index_by_frame.get(frame)
        if index is None or not np.isfinite(track.boxes[index]).all():
            continue
        box = track.boxes[index]
        mask = segmenter.runner_mask(image, box)
        if mask is None:
            continue
        tolerance = segmenter.rule.contact_ratio * (box[3] - box[1])
        yield ContactFrame(frame, image, mask, board_side_contact(mask, sand_y, tolerance))


def nearest_scored_joint(track: PoseTrack, frame: int, point: tuple, min_score: float) -> str:
    index = track.index_by_frame()[frame]
    names = list(track.joint_index)
    distances = [np.hypot(*(track.joints_xy[index, track.joint_index[name]] - point))
                 if track.joint_scores[index, track.joint_index[name]] >= min_score else np.nan for name in names]
    return names[int(np.nanargmin(distances))]


def find_sand_mark(analysis: JumpAnalysis, segmenter: RunnerSegmenter) -> SandMark:
    sand_landing = analysis.events.sand_landing
    sand_y = sand_surface_y(analysis.track, sand_landing)
    best, foot_contact, outlines = None, None, {}
    for observed in contact_frames(analysis, sand_y, segmenter):
        outlines[observed.frame] = RunnerOutline(silhouette_contours(observed.mask), observed.contact)
        if observed.frame == sand_landing and observed.contact:
            foot_contact = observed.contact
        if observed.contact and (best is None or observed.contact[0] > best.contact[0]):
            best = observed
    joint = nearest_scored_joint(analysis.track, best.frame, best.contact, segmenter.rule.min_joint_score)
    return SandMark(analysis, sand_y, find_takeoff_line(analysis.run.video), best.contact, best.frame, joint,
                    best.image.copy(), best.mask, foot_contact, load_manual_mark(analysis.run), outlines)


def sand_mark_row(mark: SandMark) -> dict:
    row = dict(video=mark.analysis.run.name, sand_frame=mark.analysis.events.sand_landing,
               window_end=mark.analysis.last_frame, mark_frame=mark.frame, mark_x=mark.point[0],
               mark_y=mark.point[1], mark_part=mark.body_part, nearest_joint=mark.joint,
               foot_x_at_landing=mark.foot_contact_at_landing[0] if mark.foot_contact_at_landing else "",
               takeoff_line_x_at_mark_y=round(mark.takeoff_x, 1),
               takeoff_to_mark_px=round(mark.takeoff_x - mark.point[0], 1))
    if mark.manual:
        row.update(ref_frame=mark.manual.frame, ref_x=mark.manual.x, ref_part=mark.manual.body_part,
                   err_frame=mark.frame - mark.manual.frame, err_x=mark.point[0] - mark.manual.x)
    return row


# --------------------------------------------------------------------------- drawing

def put_label(image: np.ndarray, text: str, y: int, color: tuple, x: int = 30) -> None:
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, BLACK, 6, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)


def draw_takeoff_line(image: np.ndarray, line: TakeoffLine) -> None:
    cv2.line(image, tuple(map(int, line.top)), tuple(map(int, line.bottom)), RED, 3)


def silhouette_contours(mask: np.ndarray) -> tuple:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def draw_silhouette(image: np.ndarray, contours: tuple) -> None:
    cv2.drawContours(image, contours, -1, GREEN, 2)


def box_size_text(box: np.ndarray) -> str:
    """W = bbox width, L = bbox height (pixels), as in the prescan aspect-ratio CSV."""
    width, height = box[2] - box[0], box[3] - box[1]
    return f"W={width:.0f} L={height:.0f} W/L={width / height:.2f}"


def draw_runner_box(image: np.ndarray, box: np.ndarray) -> None:
    """Tracked runner bbox with its W, L and W/L above it (below it when there is no room above),
    shifted left when the runner is at the right edge of the frame."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 2)
    text = box_size_text(box)
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0][0]
    put_label(image, text, y1 - 12 if y1 > 60 else y2 + 40, WHITE, x=max(0, min(x1, image.shape[1] - text_width - 10)))


def draw_manual_mark(image: np.ndarray, manual: ManualMarkRecord, size: int) -> None:
    cv2.drawMarker(image, (int(manual.x), int(manual.y)), MAGENTA, cv2.MARKER_CROSS, size, 3)


class Mp4Writer:
    """Pipes BGR frames into ffmpeg (libx264, crf 20)."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
             "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE)

    def write(self, image: np.ndarray) -> None:
        self.process.stdin.write(image.tobytes())

    def close(self) -> None:
        self.process.stdin.close()
        self.process.wait()


def video_geometry(video: Path) -> tuple:
    capture = cv2.VideoCapture(str(video))
    geometry = (capture.get(cv2.CAP_PROP_FPS), int(capture.get(3)), int(capture.get(4)))
    capture.release()
    return geometry


# --------------------------------------------------------------------------- sand mark sheet

SHEET_TILE = (900, 340)


def sand_mark_tile(mark: SandMark) -> np.ndarray:
    image = mark.image.copy()
    draw_silhouette(image, silhouette_contours(mark.mask))
    cv2.line(image, (0, int(mark.sand_y)), (image.shape[1], int(mark.sand_y)), CYAN, 1)
    draw_takeoff_line(image, mark.takeoff_line)
    cv2.circle(image, mark.point, 8, YELLOW, -1)
    if mark.manual:
        draw_manual_mark(image, mark.manual, 22)
    tile_width, tile_height = SHEET_TILE
    x0 = int(np.clip(mark.point[0] - 450, 0, image.shape[1] - tile_width))
    y0 = int(np.clip(mark.point[1] - 260, 0, image.shape[0] - tile_height))
    tile = image[y0:y0 + tile_height, x0:x0 + tile_width].copy()
    cv2.putText(tile, f"{mark.analysis.run.name} f{mark.frame} mark={mark.body_part}", (10, 30), 0, 0.9, WHITE, 2)
    return tile


def contact_sheet(tiles: list, columns: int = 2) -> np.ndarray:
    blank = np.zeros_like(tiles[0])
    padded = tiles + [blank] * (-len(tiles) % columns)
    return np.vstack([np.hstack(padded[start:start + columns]) for start in range(0, len(padded), columns)])


# --------------------------------------------------------------------------- review video

EVENT_LABEL_FRAMES = 12
TRIM_NOTICE_FRAMES = 20
LABEL_LINE_HEIGHT = 45
LEGEND = "green box=bbox  green line=runner  red=take-off  cyan=sand y  white=contact  yellow=nearest mark"


SLOW_MOTION_FACTOR = 4


def event_status(events: JumpEvents, frame: int) -> str:
    """Persistent phase label for a review video frame."""
    if events.board_contact is not None and frame < events.board_contact:
        return "RUN-UP"
    if events.board_contact is not None and (events.toe_off is None or frame < events.toe_off):
        return "TAKE-OFF SUPPORT"
    if frame <= events.flight_end:
        return "FLIGHT"
    if events.sand_landing is not None and frame < events.sand_landing:
        return "DESCENT / CONTACT"
    return "LANDED"


def event_summary_lines(events: JumpEvents, fps: float) -> list[str]:
    """Stable event summary used by the persistent top-left review panel."""
    values = (
        ("support", events.board_contact),
        ("toe-off", events.toe_off),
        ("apex", events.apex),
        ("flight end", events.flight_end),
        ("sand landing", events.sand_landing),
    )
    return [f"{name}:  f{frame}   {frame / fps:.3f}s" for name, frame in values if frame is not None]


def draw_event_panel(image: np.ndarray, analysis: JumpAnalysis, frame: int, fps: float) -> None:
    """Draw the compact persistent event panel used by board-independent reviews."""
    events = analysis.events
    panel_width, panel_height = 620, 235
    overlay = image.copy()
    cv2.rectangle(overlay, (15, 15), (panel_width, panel_height), BLACK, -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)

    def panel_text(text: str, y: int, color: tuple, scale: float = 0.68, thickness: int = 2) -> None:
        cv2.putText(image, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, cv2.LINE_AA)

    panel_text(f"{analysis.run.name}   frame={frame}   t={frame / fps:.3f}s", 48, WHITE, 0.72)
    panel_text(event_status(events, frame), 82, YELLOW, 0.88, 3)
    for row, text in enumerate(event_summary_lines(events, fps)):
        panel_text(text, 112 + row * 24, WHITE, 0.58)


class EventReviewVideo:
    """Board-independent review ending at the detected analysis trim point.

    The tracked bbox is drawn wherever it exists.  When a segmenter is supplied,
    the runner silhouette is drawn from the detected apex through the analysis
    trim point.  A missing board therefore disables only board geometry and sand
    mark measurement; it does not prevent event-video rendering.
    """

    def __init__(self, analysis: JumpAnalysis, output: Path,
                 segmenter: RunnerSegmenter | None = None):
        if analysis.events is None:
            raise RuntimeError(f"{analysis.run.name}: no jump flight detected")
        self.analysis = analysis
        self.output = output
        self.segmenter = segmenter
        self.index_by_frame = analysis.track.index_by_frame()

    def render(self) -> Path:
        fps, width, height = video_geometry(self.analysis.run.video)
        writer = Mp4Writer(self.output, width, height, fps)
        for frame, image in iterate_frames(self.analysis.run.video, 0, self.analysis.last_frame):
            self._draw(image, frame, fps)
            writer.write(image)
        writer.close()
        return self.output

    def _draw(self, image: np.ndarray, frame: int, fps: float) -> None:
        index = self.index_by_frame.get(frame)
        if index is not None and np.isfinite(self.analysis.track.boxes[index]).all():
            box = self.analysis.track.boxes[index]
            mask = self._runner_mask(image, frame, box)
            draw_runner_box(image, box)
            if mask is not None:
                draw_silhouette(image, silhouette_contours(mask))
        draw_event_panel(image, self.analysis, frame, fps)

    def _runner_mask(self, image: np.ndarray, frame: int, box: np.ndarray) -> np.ndarray | None:
        events = self.analysis.events
        if (self.segmenter is None or frame < events.apex
                or frame > self.analysis.last_frame):
            return None
        return self.segmenter.runner_mask(image, box)


def review_video_paths(run: RunSpec) -> tuple:
    """(normal speed, SLOW_MOTION_FACTOR x slow motion) review video paths."""
    folder = run.dataset_dir / "jump_review"
    return folder / f"{run.name}_review.mp4", folder / f"{run.name}_review_slow{SLOW_MOTION_FACTOR}x.mp4"


class JumpReviewVideo:
    """One video per jump over the trimmed clip: take-off line, board contact and sand landing (detected vs GT),
    trim milestones, and from the jump apex on the segmented runner, sand contacts and the nearest sand mark.

    Search-window outlines are reused from find_sand_mark; only apex .. sand landing is segmented here.
    """

    def __init__(self, mark: SandMark, segmenter: RunnerSegmenter):
        self.mark = mark
        self.analysis = mark.analysis
        self.segmenter = segmenter
        self.index_by_frame = self.analysis.track.index_by_frame()
        self.events = _event_labels(self.analysis)
        self.milestones = _trim_milestones(self.analysis)
        self.outlines = dict(mark.outlines)
        self.nearest_contact = None

    def render(self) -> tuple:
        """Writes the same drawn frames at native fps and at fps / SLOW_MOTION_FACTOR; returns both paths."""
        run, track = self.analysis.run, self.analysis.track
        fps, width, height = video_geometry(run.video)
        outputs = review_video_paths(run)
        writers = [Mp4Writer(output, width, height, rate) for output, rate in zip(outputs, (fps, fps / SLOW_MOTION_FACTOR))]
        for frame, image in iterate_frames(run.video, int(track.frames[0]), self.analysis.last_frame):
            self._draw(image, frame)
            for writer in writers:
                writer.write(image)
        for writer in writers:
            writer.close()
        return outputs

    def _in_search_window(self, frame: int) -> bool:
        return self.analysis.events.sand_landing <= frame <= self.analysis.last_frame

    def _draw(self, image: np.ndarray, frame: int) -> None:
        outline = self.outlines.get(frame) or self._segment_descent(image, frame)
        draw_takeoff_line(image, self.mark.takeoff_line)
        if self._in_search_window(frame):
            cv2.line(image, (0, int(self.mark.sand_y)), (image.shape[1], int(self.mark.sand_y)), CYAN, 1)
        self._draw_box(image, frame)
        if outline is not None:
            draw_silhouette(image, outline.contours)
            self._draw_contact(image, outline.contact)
        if self.nearest_contact is not None:
            cv2.circle(image, self.nearest_contact, 9, YELLOW, -1)
        manual = self.mark.manual
        if manual and frame >= manual.frame:
            draw_manual_mark(image, manual, 26)
        self._label(image, frame)

    def _segment_descent(self, image: np.ndarray, frame: int) -> Optional[RunnerOutline]:
        """Segment frames from the jump apex up to sand landing; earlier frames are not segmented."""
        events, index = self.analysis.events, self.index_by_frame.get(frame)
        if not events.apex <= frame < events.sand_landing or index is None:
            return None
        box = self.analysis.track.boxes[index]
        mask = self.segmenter.runner_mask(image, box) if np.isfinite(box).all() else None
        return None if mask is None else RunnerOutline(silhouette_contours(mask))

    def _draw_box(self, image: np.ndarray, frame: int) -> None:
        index = self.index_by_frame.get(frame)
        if index is not None and np.isfinite(self.analysis.track.boxes[index]).all():
            draw_runner_box(image, self.analysis.track.boxes[index])

    def _draw_contact(self, image: np.ndarray, contact: Optional[tuple]) -> None:
        if contact is None:
            return
        cv2.circle(image, contact, 6, WHITE, 2)
        if self.nearest_contact is None or contact[0] > self.nearest_contact[0]:
            self.nearest_contact = contact

    def _label(self, image: np.ndarray, frame: int) -> None:
        draw_event_panel(image, self.analysis, frame, self.analysis.track.fps)
        lines = _recent_event_lines(frame, self.events, self.milestones) + self._mark_lines(frame)
        for row, (text, color) in enumerate(lines):
            put_label(image, text, 290 + row * LABEL_LINE_HEIGHT, color)
        height = image.shape[0]
        if self.analysis.last_frame - frame < TRIM_NOTICE_FRAMES:
            put_label(image, f"TRIM POINT: video cut after frame {self.analysis.last_frame}", height - 80, RED)
        put_label(image, LEGEND, height - 30, WHITE)

    def _mark_lines(self, frame: int) -> list:
        mark, manual, lines = self.mark, self.mark.manual, []
        if frame >= mark.frame:
            lines.append((f"NEAREST MARK = {mark.body_part} @ {mark.frame}  x={mark.point[0]}", YELLOW))
        if manual and frame >= manual.frame:
            lines.append((f"manual record: {manual.body_part} @ {manual.frame}  x={manual.x:.0f}", MAGENTA))
        return lines


def _event_labels(analysis: JumpAnalysis) -> list:
    """(title, detected frame, GT frame) for the events shown in the review video."""
    ground_truth = load_event_ground_truth(analysis.run)
    return [("BOARD CONTACT", analysis.events.board_contact, ground_truth.get("board_contact")),
            ("TOE-OFF", analysis.events.toe_off, ground_truth.get("toe_off")),
            ("SAND LANDING", analysis.events.sand_landing, ground_truth.get("sand_landing_foot"))]


def _trim_milestones(analysis: JumpAnalysis) -> list:
    """(frame, text) for where the high-speed segment ends and where the athlete stopped."""
    trim, frames = analysis.trim, analysis.full_track.frames
    milestones = [(int(frames[trim.fast_segment[1] - 1]), "HIGH-SPEED SEGMENT ENDS")]
    if trim.anchor_index is not None:
        milestones.append((int(frames[trim.anchor_index]), "ANCHOR (athlete stopped)"))
    return milestones


def _recent_event_lines(frame: int, events: list, milestones: list) -> list:
    """(text, color) for events and milestones that happened within the last EVENT_LABEL_FRAMES frames."""
    lines = [(f"{title} ({source} @ {event_frame})", color)
             for title, detected, truth in events
             for source, event_frame, color in (("detected", detected, YELLOW), ("GT", truth, ORANGE))
             if _is_recent(frame, event_frame)]
    lines += [(f"{text} @ {milestone_frame}", CYAN)
              for milestone_frame, text in milestones if _is_recent(frame, milestone_frame)]
    return lines


def _is_recent(frame: int, event_frame: Optional[int]) -> bool:
    return event_frame is not None and 0 <= frame - event_frame < EVENT_LABEL_FRAMES


# --------------------------------------------------------------------------- commands

def jump_event_row(analysis: JumpAnalysis) -> dict:
    """One repeatable evaluation row for events and board-aware take-off geometry."""
    events = analysis.events
    truth = load_event_ground_truth(analysis.run)
    row = {"video": analysis.run.name}
    for name, detected, gt_name in (
        ("board_contact", events.board_contact, "board_contact"),
        ("toe_off", events.toe_off, "toe_off"),
        ("sand_landing", events.sand_landing, "sand_landing_foot"),
    ):
        expected = truth.get(gt_name)
        row[f"{name}_detected"] = detected
        row[f"{name}_gt"] = "" if expected is None else expected
        row[f"{name}_error"] = "" if expected is None or detected is None else detected - expected
    row["sand_landing_review"] = ";".join(events.sand_landing_review)
    takeoff = events.takeoff_assessment
    row.update(
        takeoff_foot="" if takeoff is None or takeoff.foot_side is None else takeoff.foot_side,
        min_toe_clearance_px=("" if takeoff is None or takeoff.min_toe_clearance_px is None
                              else round(takeoff.min_toe_clearance_px, 1)),
        clearance_frame=("" if takeoff is None or takeoff.clearance_frame_index is None
                         else int(analysis.track.frames[takeoff.clearance_frame_index])),
        toe_over_line="" if takeoff is None or takeoff.toe_over_line is None else takeoff.toe_over_line,
        takeoff_position=("" if takeoff is None or takeoff.takeoff_position is None
                          else takeoff.takeoff_position),
        min_board_distance_px=("" if takeoff is None or takeoff.min_board_distance_px is None
                               else round(takeoff.min_board_distance_px, 1)),
    )
    return row


def command_events(_: argparse.Namespace) -> None:
    rows = [jump_event_row(analyse_jump(run)) for run in RUNS]
    output_dir = GT_DATASET_DIR / "jump_events"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "board_aware_results.csv"
    with open(output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(f"output={output}")


def _takeoff_clearance_frame(analysis: JumpAnalysis) -> tuple:
    """Annotated source frame and a board/foot crop at the minimum toe clearance."""
    takeoff = analysis.events.takeoff_assessment
    if takeoff is None or takeoff.clearance_frame_index is None:
        raise RuntimeError(f"{analysis.run.name}: no board-aware toe clearance")
    frame_number = int(analysis.track.frames[takeoff.clearance_frame_index])
    decoded = list(iterate_frames(analysis.run.video, frame_number, frame_number))
    if not decoded:
        raise RuntimeError(f"{analysis.run.name}: cannot decode frame {frame_number}")
    image = decoded[0][1]
    board = detect_board(read_first_frame(analysis.run.video))
    if board.mask is None:
        raise RuntimeError(f"{analysis.run.name}: take-off board not found")
    line = fit_takeoff_line(board.mask, PIT_LEFT)

    overlay = image.copy()
    overlay[board.mask] = YELLOW
    cv2.addWeighted(overlay, 0.22, image, 0.78, 0, image)
    cv2.drawContours(image, silhouette_contours(board.mask), -1, YELLOW, 1)
    cv2.line(image, tuple(map(int, line.top)), tuple(map(int, line.bottom)), RED, 1, cv2.LINE_AA)

    toe_points = []
    for joint in ("big_toe", "small_toe"):
        point = analysis.track.joint_xy(f"{takeoff.foot_side}_{joint}")[takeoff.clearance_frame_index]
        x, y = map(float, point)
        clearance = float(x - line.x_at(y))
        toe_points.append((clearance, joint, tuple(map(int, np.rint(point)))))
    clearance, joint, toe = min(toe_points)
    line_point = (int(round(line.x_at(toe[1]))), toe[1])
    position = takeoff.takeoff_position or ("OVER_LINE" if clearance < 0 else "ON_BOARD")
    status = position.replace("_", " ")
    status_color = RED if position == "OVER_LINE" else (ORANGE if position == "BEFORE_BOARD" else GREEN)
    cv2.line(image, line_point, toe, status_color, 1, cv2.LINE_AA)
    cv2.circle(image, line_point, 2, WHITE, -1)
    for _, toe_joint, point in toe_points:
        cv2.circle(image, point, 3, CYAN if toe_joint != joint else status_color, -1)

    put_label(image, f"{analysis.run.name}  frame {frame_number}", 50, WHITE)
    put_label(image, f"take-off foot={takeoff.foot_side.upper()}  {status}", 95, status_color)
    put_label(image, f"minimum toe clearance={clearance:+.1f}px", 140, status_color)

    ys, xs = np.nonzero(board.mask)
    points_x = [point[0] for _, _, point in toe_points]
    points_y = [point[1] for _, _, point in toe_points]
    margin = 100
    x0 = max(0, min(int(xs.min()), *points_x) - margin)
    x1 = min(image.shape[1], max(int(xs.max()), *points_x) + margin + 1)
    y0 = max(0, min(int(ys.min()), *points_y) - margin)
    y1 = min(image.shape[0], max(int(ys.max()), *points_y) + margin + 1)
    zoom = cv2.resize(image[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    put_label(zoom, f"{analysis.run.name} f{frame_number}  clearance={clearance:+.1f}px  {status}", 45,
              status_color, 20)

    # Presentation sheet: match the user's compact 2-column reference layout
    # with a wide crop centered on the board and the athlete's lower body.
    tile_width, tile_height = 1200, 480
    source_width, source_height = 900, 360
    center_x = int(round((xs.min() + xs.max()) / 2))
    board_center_y = int(round((ys.min() + ys.max()) / 2))
    sheet_x0 = int(np.clip(center_x - source_width // 2, 0, image.shape[1] - source_width))
    sheet_y0 = int(np.clip(board_center_y - 270, 0, image.shape[0] - source_height))
    sheet_crop = image[sheet_y0:sheet_y0 + source_height, sheet_x0:sheet_x0 + source_width]
    sheet_tile = cv2.resize(sheet_crop, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
    put_label(
        sheet_tile,
        f"{analysis.run.name} frame {frame_number}  clearance={clearance:+.1f}px  {status}",
        38,
        status_color,
        10,
    )
    return frame_number, image, zoom, sheet_tile


def command_takeoff_frames(_: argparse.Namespace) -> None:
    output_dir = GT_DATASET_DIR / "jump_events/takeoff_line_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_tiles = []
    for run in RUNS:
        analysis = analyse_jump(run)
        frame, full, zoom, sheet_tile = _takeoff_clearance_frame(analysis)
        full_path = output_dir / f"{run.name}_f{frame}_full.jpg"
        zoom_path = output_dir / f"{run.name}_f{frame}_zoom.jpg"
        cv2.imwrite(str(full_path), full)
        cv2.imwrite(str(zoom_path), zoom)
        sheet_tiles.append(sheet_tile)
        print(f"{run.name}: {full_path}, {zoom_path}")
    sheet_path = output_dir / "takeoff_clearance_sheet.jpg"
    cv2.imwrite(str(sheet_path), contact_sheet(sheet_tiles, columns=2))
    print(f"sheet={sheet_path}")

def command_board(args: argparse.Namespace) -> None:
    image = read_first_frame(args.video)
    board = detect_board(image)
    result = {"video": str(args.video),
              "candidates": [[round(c.ring_red, 2), c.area, list(c.bbox)] for c in board.candidates[:3]]}
    if board.mask is not None:
        line = fit_takeoff_line(board.mask, args.pit_side)
        image[board.mask] = YELLOW
        draw_takeoff_line(image, line)
        result["takeoff_line"] = [line.top, line.bottom]
    cv2.imwrite(f"{args.output_prefix}_det.jpg", image)
    print(json.dumps(result))


def command_event_video(args: argparse.Namespace) -> None:
    """Analyse one prepared video and render the board-independent full-clip review."""
    run = RunSpec(
        name=args.name or args.video.stem,
        pose_dir=args.pose_dir,
        file_stem=args.file_stem,
        clip_to_run_offset=0,
        video=args.video,
        dataset_dir=args.output.parent,
    )
    analysis = analyse_jump(run)
    segmenter = None if args.no_segmentation else RunnerSegmenter()
    output = EventReviewVideo(analysis, args.output, segmenter).render()
    result = jump_event_row(analysis)
    result["output"] = str(output)
    result["segmentation"] = segmenter is not None
    print(json.dumps(result, ensure_ascii=False, default=lambda value: value.item()))


def longjump_tracking_config(video_stem: str) -> dict:
    """Production tracking settings for one-click full long-jump videos.

    ``prescan_enabled`` deliberately routes through ``core.tracking.prescan``;
    Pass 1 and Pass 2 then retain original source-frame numbers through the
    normal frame-map and offsets artifacts.
    """
    return {
        "output_name": f"{video_stem}_longjump_tracked.mp4",
        "tracking_mode": "two_pass",
        "runner_selection_strategy": "max_progress",
        "auto_crop": True,
        "prescan_enabled": True,
        "show_overlay": False,
        "draw_bbox_overlay": False,
        "write_overview_video": True,
    }


def _video_fps(video: Path) -> float:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if fps <= 0:
        raise RuntimeError(f"cannot read FPS: {video}")
    return fps


def _prescan_override_path(output_dir: Path, video_stem: str) -> Path:
    return output_dir / f"{video_stem}_prescan_frame_ranges.json"


def command_prescan_worker(args: argparse.Namespace) -> None:
    """Isolated worker so a native TensorRT failure cannot kill full-video."""
    import sys

    repo_path = str(REPO)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from core import tracking
    from core.tracking_runtime import TrackingRuntimeOptions, temporary_tracking_runtime

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = longjump_tracking_config(args.video.stem)
    config["prescan_engine_path"] = str(args.model.resolve())
    with temporary_tracking_runtime(TrackingRuntimeOptions(
        config=config,
        gpu=args.gpu,
        output_directory=str(output_dir),
    )):
        ranges = tracking.run_temporal_prescan(
            [{"video_path": str(args.video.resolve())}],
            output_dir=str(output_dir),
        )
    serializable = {
        str(camera): [[int(start), int(end)] for start, end in camera_ranges]
        for camera, camera_ranges in (ranges or {}).items()
    }
    destination = _prescan_override_path(output_dir, args.video.stem)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def run_isolated_prescan(video: Path, output_dir: Path, gpu: str) -> dict:
    """Run motion prescan safely, falling back from TensorRT to PyTorch.

    TensorRT engine incompatibility can terminate the interpreter with SIGBUS,
    so it must be tried in a child process.  Both attempts use the same
    ``core.tracking.prescan`` motion detector and thresholds; only the YOLO
    inference backend differs.
    """
    import sys

    destination = _prescan_override_path(output_dir, video.stem)
    candidates = (
        ("TensorRT INT8", REPO / "models/yolo26x_ultralytics_int8.engine"),
        ("PyTorch", REPO / "models/yolo26x.pt"),
    )
    for label, model in candidates:
        if not model.is_file():
            continue
        destination.unlink(missing_ok=True)
        completed = subprocess.run([
            sys.executable,
            str(Path(__file__).resolve()),
            "prescan-worker",
            str(video),
            str(output_dir),
            str(model),
            "--gpu", gpu,
        ], check=False)
        if completed.returncode != 0 or not destination.is_file():
            print(f"  [prescan] {label} worker failed (exit={completed.returncode})")
            continue
        document = json.loads(destination.read_text(encoding="utf-8"))
        ranges = {
            int(camera): [(int(start), int(end)) for start, end in camera_ranges]
            for camera, camera_ranges in document.items()
        }
        print(f"  [prescan] isolated backend={label}, ranges={ranges}")
        return ranges
    print("  [prescan] all isolated backends failed; tracking will scan the full video")
    return {}


def command_full_video(args: argparse.Namespace) -> None:
    """Run prescan, tracking, WholeBody23 pose, events and review in one command."""
    import sys

    repo_path = str(REPO)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from core.contracts.wholebody23 import write_wholebody23_artifact
    from core.pipeline import (
        PoseEstimationRequest,
        PoseScope,
        Step1TrackingRequest,
        VideoOutput,
        step1_track,
        step2_pose,
    )

    video = args.video.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or video.stem
    frame_ranges = run_isolated_prescan(video, output_dir, args.gpu)
    tracked_video = Path(step1_track(Step1TrackingRequest(
        camera_configs=[{"video_path": str(video)}],
        extra_config=longjump_tracking_config(video.stem),
        gpu=args.gpu,
        output_dir=str(output_dir),
        frame_ranges_by_camera=frame_ranges,
    )))
    pose_dir = Path(step2_pose(PoseEstimationRequest(
        tracked_video_path=str(tracked_video),
        output_base_dir=str(output_dir),
        gpu=args.gpu,
        motion_ag_dir=REPO / "MotionAGFormer",
        pose_scope=PoseScope.TWO_D_ONLY,
        video_output=VideoOutput.OMIT,
    )))
    bbox_map = tracked_video.with_name(f"{tracked_video.stem}_bbox_map.csv")
    offsets = output_dir / f"{video.stem}_offsets.npz"
    pose_artifact = write_wholebody23_artifact(
        raw_npz=pose_dir / "input_2D/wholebody23_raw.npz",
        offsets_npz=offsets,
        bbox_map_csv=bbox_map,
        destination=output_dir / "pose/keypoints_2d.json",
        frames_per_second=_video_fps(video),
    )
    run = RunSpec(name, output_dir, video.stem, 0, video, output_dir)
    analysis = analyse_jump(run)
    segmenter = None if args.no_segmentation else RunnerSegmenter()
    review_output = args.review_output or output_dir / f"{name}_event_review.mp4"
    EventReviewVideo(analysis, review_output, segmenter).render()
    result = jump_event_row(analysis)
    events = analysis.events
    fps = _video_fps(video)
    result.update({
        "input": str(video),
        "flight_start_detected": events.flight_start,
        "flight_end_detected": events.flight_end,
        "apex_detected": events.apex,
        "flight_start_sec": round(events.flight_start / fps, 3),
        "flight_end_sec": round(events.flight_end / fps, 3),
        "apex_sec": round(events.apex / fps, 3),
        "sand_landing_sec": (
            None if events.sand_landing is None
            else round(events.sand_landing / fps, 3)
        ),
        "prescan_report": str(output_dir / f"{video.stem}_prescan_ranges.json"),
        "pose": str(pose_artifact),
        "review": str(review_output),
        "segmentation": segmenter is not None,
    })
    result_path = output_dir / f"{name}_event_results.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=lambda value: value.item()) + "\n",
        encoding="utf-8",
    )
    result["results"] = str(result_path)
    print(json.dumps(result, ensure_ascii=False, default=lambda value: value.item()))


def command_marks(_: argparse.Namespace) -> None:
    segmenter = RunnerSegmenter()
    marks = [find_sand_mark(analyse_jump(run), segmenter) for run in RUNS]
    rows = [sand_mark_row(mark) for mark in marks]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False, default=lambda value: value.item()))
    output_dir = GT_DATASET_DIR / "sand_mark"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "mark_results.csv", "w", newline="") as results:
        writer = csv.DictWriter(results, list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)
    cv2.imwrite(str(output_dir / "mark_sheet.jpg"), contact_sheet([sand_mark_tile(mark) for mark in marks]))


def command_videos(_: argparse.Namespace) -> None:
    segmenter = RunnerSegmenter()
    for run in RUNS:
        analysis = analyse_jump(run)
        normal, slow = JumpReviewVideo(find_sand_mark(analysis, segmenter), segmenter).render()
        print(f"{run.name}: kept {int(analysis.track.frames[0])}-{analysis.last_frame} -> {normal}, {slow}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    board = commands.add_parser("board", help="detect the take-off board in the first frame of a video")
    board.add_argument("video", type=Path)
    board.add_argument("output_prefix")
    board.add_argument("--pit-side", choices=(PIT_LEFT, PIT_RIGHT), default=PIT_LEFT)
    board.set_defaults(handler=command_board)
    event_video = commands.add_parser(
        "event-video",
        help="analyse one prepared video and render bbox, segmentation and a persistent event panel",
    )
    event_video.add_argument("video", type=Path)
    event_video.add_argument("pose_dir", type=Path,
                             help="folder containing pose/keypoints_2d.json and tracking sidecars")
    event_video.add_argument("file_stem",
                             help="prefix of <stem>_selected_runner.json and <stem>_all_tracks.csv")
    event_video.add_argument("output", type=Path)
    event_video.add_argument("--name", help="display name; defaults to the video stem")
    event_video.add_argument("--no-segmentation", action="store_true",
                             help="render events and bbox without running person segmentation")
    event_video.set_defaults(handler=command_event_video)
    full_video = commands.add_parser(
        "full-video",
        help="one-click full video: motion prescan, tracking, pose, events and review",
    )
    full_video.add_argument("video", type=Path)
    full_video.add_argument("output_dir", type=Path)
    full_video.add_argument("--name", help="display/output name; defaults to the video stem")
    full_video.add_argument("--gpu", default="0")
    full_video.add_argument("--review-output", type=Path)
    full_video.add_argument("--no-segmentation", action="store_true")
    full_video.set_defaults(handler=command_full_video)
    prescan_worker = commands.add_parser("prescan-worker", help=argparse.SUPPRESS)
    prescan_worker.add_argument("video", type=Path)
    prescan_worker.add_argument("output_dir", type=Path)
    prescan_worker.add_argument("model", type=Path)
    prescan_worker.add_argument("--gpu", default="0")
    prescan_worker.set_defaults(handler=command_prescan_worker)
    for name, handler, text in (("events", command_events, "board-aware event evaluation CSV for all runs"),
                                ("takeoff-frames", command_takeoff_frames,
                                 "annotated full and zoomed frames at minimum toe clearance"),
                                ("marks", command_marks, "nearest sand mark CSV + check sheet for all runs"),
                                ("videos", command_videos, "one review video per jump with every detection")):
        commands.add_parser(name, help=text).set_defaults(handler=handler)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.handler(arguments)
