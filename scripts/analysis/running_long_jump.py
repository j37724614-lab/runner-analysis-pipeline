"""Anchor/step hybrid detection for a running long jump.

This module intentionally does not replace the legacy terminal landing
detector in ``ankle_step_stride.py``.  Its single public interface is
``detect_running_long_jump()``; the caller decides which implementation to
select with configuration.

The detector uses a two-pass offline process:

1. Build a provisional ground model from accepted running contacts.
2. Find every complete bilateral-airborne interval and select the longest.
3. Rebuild/freeze the model from the last reliable contacts before take-off.
4. Re-run the evidence sequence and report take-off, first touchdown and
   maximum compression separately.

Missing pose points are UNKNOWN evidence.  They never become artificial
airborne or contact observations and sand-pit frames never update the frozen
ground model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from math import exp, isfinite

import cv2
import numpy as np

ALGORITHM_NAME = "anchor_step_hybrid_longest_flight"
_SIDES = ("left", "right")


@dataclass(frozen=True)
class RunningLongJumpRequest:
    """Inputs required by :func:`detect_running_long_jump`.

    ``ankle_rows`` and ``foot_contacts_by_seq`` must already contain the
    project's corrected left/right identities and original-image pixels.
    ``accepted_events`` are used only as pre-flight calibration evidence.
    """

    ankle_rows: Sequence[Mapping]
    accepted_events: Sequence[Mapping]
    foot_contacts_by_seq: Mapping[int, Mapping]
    camera: Mapping
    fps: float
    config: Mapping = field(default_factory=dict)


@dataclass(frozen=True)
class _GroundModel:
    source: str
    geometry_mode: str
    camera_id: int
    surface_scope: str
    surface_plane_mode: str
    foot_slope: float
    foot_intercept: float
    ankle_slope: float
    ankle_intercept: float
    x_min: float | None
    x_max: float | None
    runner_lane_ratio: float | None
    runner_lane_world_y: float | None
    foot_offset_px: float | None
    ankle_offset_px: float | None
    foot_noise_px: float
    ankle_noise_px: float
    foot_threshold_px: float
    ankle_threshold_px: float
    foot_scale_px: float
    ankle_scale_px: float
    ankle_to_heel_y_offset_px: Mapping[str, float]
    ankle_to_heel_residual_noise_px: Mapping[str, float]
    ankle_to_heel_sample_count: Mapping[str, int]
    anchor_frames: tuple[int, ...]
    reprojection_error_px: float | None
    quality_score: float
    reasons: tuple[str, ...]

    def ground_y(self, x: float, joint_kind: str) -> float:
        if joint_kind == "foot":
            return self.foot_slope * float(x) + self.foot_intercept
        return self.ankle_slope * float(x) + self.ankle_intercept

    def supports_x(self, x: float, margin_px: float) -> bool:
        if self.x_min is None or self.x_max is None:
            return True
        return self.x_min - margin_px <= float(x) <= self.x_max + margin_px

    def summary(self) -> dict:
        return {
            "source": self.source,
            "geometry_mode": self.geometry_mode,
            "camera_id": self.camera_id,
            "calibrated_surface_scope": self.surface_scope,
            "surface_plane_mode": self.surface_plane_mode,
            "sand_model_source": (
                "shared_anchor_surface"
                if self.surface_scope == "runway_and_sand"
                and self.surface_plane_mode == "shared"
                else "not_shared"
            ),
            "runner_lane_ratio_u": self.runner_lane_ratio,
            "runner_lane_world_y": self.runner_lane_world_y,
            "anchor_frames": list(self.anchor_frames),
            "anchor_count": len(self.anchor_frames),
            "foot_offset_px": self.foot_offset_px,
            "ankle_offset_px": self.ankle_offset_px,
            "foot_residual_mad_px": self.foot_noise_px / 1.4826,
            "ankle_residual_mad_px": self.ankle_noise_px / 1.4826,
            "foot_off_threshold_px": self.foot_threshold_px,
            "ankle_off_threshold_px": self.ankle_threshold_px,
            "ankle_to_heel_y_offset_px": dict(self.ankle_to_heel_y_offset_px),
            "ankle_to_heel_residual_noise_px": dict(
                self.ankle_to_heel_residual_noise_px
            ),
            "ankle_to_heel_sample_count": dict(self.ankle_to_heel_sample_count),
            "reprojection_error_px": self.reprojection_error_px,
            "valid_x_min": self.x_min,
            "valid_x_max": self.x_max,
            "frozen": True,
            "quality_score": self.quality_score,
            "needs_review": bool(self.reasons),
            "reasons": list(self.reasons),
        }


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _setting(config: Mapping, name: str, default):
    value = config.get(name, default)
    return default if value is None else value


def _robust_center(values: Sequence[float], default: float = 0.0) -> float:
    finite = np.asarray([value for value in values if isfinite(value)], dtype=float)
    return float(np.median(finite)) if len(finite) else float(default)


def _robust_noise(residuals: Sequence[float]) -> float:
    if not residuals:
        return 0.0
    values = np.asarray(residuals, dtype=float)
    center = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - center)))


def _robust_line(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Fit one line and remove gross MAD outliers once."""
    if not points:
        raise ValueError("at least one point is required")
    xs = np.asarray([point[0] for point in points], dtype=float)
    ys = np.asarray([point[1] for point in points], dtype=float)
    if len(points) < 2 or float(np.ptp(xs)) < 1.0:
        return 0.0, float(np.median(ys))
    slope, intercept = np.polyfit(xs, ys, 1)
    residuals = ys - (slope * xs + intercept)
    center = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - center)))
    if mad > 0:
        keep = np.abs(residuals - center) <= 3.5 * 1.4826 * mad
        if int(np.count_nonzero(keep)) >= 2 and float(np.ptp(xs[keep])) >= 1.0:
            slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
    return float(slope), float(intercept)


def _foot_joint_point(
    per_frame: Mapping | None, side: str, joint: str, min_conf: float
):
    if not per_frame:
        return None
    values = per_frame.get(side) or {}
    point = values.get(joint) or {}
    x, y, confidence = (
        _finite(point.get("x")),
        _finite(point.get("y")),
        _finite(point.get("conf")),
    )
    if (
        x is None
        or y is None
        or confidence is None
        or confidence < min_conf
    ):
        return None
    return x, y, confidence, joint


def _foot_point(per_frame: Mapping | None, side: str, min_conf: float):
    valid = []
    for joint in ("heel", "big_toe"):
        point = _foot_joint_point(per_frame, side, joint, min_conf)
        if point is not None:
            valid.append(point)
    return max(valid, key=lambda point: point[1], default=None)


def _ankle_point(row: Mapping, side: str, min_conf: float):
    x = _finite(row.get(f"{side}_ankle_x"))
    y = _finite(row.get(f"{side}_ankle_y"))
    confidence = _finite(row.get(f"{side}_ankle_conf"))
    if x is None or y is None or confidence is None or confidence < min_conf:
        return None
    return x, y, confidence, "ankle"


def _event_contact_point(event: Mapping, foot_contacts: Mapping, min_conf: float):
    sequence = int(event["seq_frame"])
    side = str(event.get("foot", "")).lower()
    point = _foot_point(foot_contacts.get(sequence), side, min_conf)
    if point is not None:
        return point
    joint = str(event.get("contact_joint", ""))
    confidence = _finite(event.get("contact_conf"))
    x, y = _finite(event.get("contact_x")), _finite(event.get("contact_y"))
    if (
        event.get("contact_valid", True)
        and ("heel" in joint or "toe" in joint)
        and confidence is not None
        and confidence >= min_conf
        and x is not None
        and y is not None
    ):
        return x, y, confidence, joint
    return None


def _reliable_events(events: Sequence[Mapping], rows_by_seq: Mapping[int, Mapping]):
    result = []
    for event in sorted(events, key=lambda item: int(item["seq_frame"])):
        sequence = int(event["seq_frame"])
        if sequence not in rows_by_seq:
            continue
        if str(event.get("event_type") or "run_step") != "run_step":
            continue
        if event.get("contact_valid") is False:
            continue
        rejection_reason = str(event.get("contact_rejection_reason") or "")
        if "world_x_clamped_to_calibrated_range" in rejection_reason:
            # A contact outside the calibrated surface may still be retained
            # for display after its longitudinal coordinate is clamped.  It is
            # not a measured point on the ground plane and must not teach the
            # frozen ground baseline.
            continue
        result.append(event)
    return result


def _homography_geometry(camera: Mapping):
    src = camera.get("homography_src_points")
    dst = camera.get("homography_dst_world")
    if src is None or dst is None:
        return None
    source = np.asarray(src, dtype=np.float64)
    world = np.asarray(dst, dtype=np.float64)
    if source.ndim != 2 or world.ndim != 2 or source.shape != world.shape:
        return None
    if source.shape[0] < 4 or source.shape[1] != 2:
        return None
    matrix, _ = cv2.findHomography(source, world, method=0)
    if matrix is None:
        return None
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    reconstructed = cv2.perspectiveTransform(
        world.astype(np.float32).reshape(-1, 1, 2), inverse
    ).reshape(-1, 2)
    error = float(np.mean(np.linalg.norm(reconstructed - source, axis=1)))
    return source, world, matrix, inverse, error


def _transform(point: tuple[float, float], matrix: np.ndarray):
    values = np.asarray([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(values, matrix)[0, 0]
    return float(mapped[0]), float(mapped[1])


def _samples_within_world_x(samples, matrix, world_x_min, world_x_max):
    """Discard contacts that project outside the calibrated runway length."""
    return [
        sample
        for sample in samples
        if world_x_min <= _transform((sample[0], sample[1]), matrix)[0] <= world_x_max
    ]


def _model_threshold(noise: float, config: Mapping, joint: str):
    # Floors default to the paper's fixed T_off=8px / s=2.5px. Real calibration
    # residuals still win when they are larger (more preflight contacts, more
    # scatter), but a handful of near-identical anchor samples must not be
    # allowed to collapse the curve tighter than the paper's own baseline --
    # that is what let a few pixels of normal push-off noise register as
    # 96%+ confident "airborne" when only 2 anchors were available.
    explicit = _finite(config.get(f"{joint}_off_distance_threshold_px"))
    minimum = float(_setting(config, "minimum_off_distance_px", 8.0))
    multiplier = float(_setting(config, "noise_threshold_multiplier", 2.5))
    threshold = explicit if explicit is not None else max(minimum, multiplier * noise)
    scale = max(
        float(_setting(config, "minimum_logistic_scale_px", 2.5)),
        noise,
        threshold / 3.0,
    )
    return float(threshold), float(scale)


def _build_ground_model(request: RunningLongJumpRequest, events: Sequence[Mapping]):
    rows_by_seq = {int(row["seq_frame"]): row for row in request.ankle_rows}
    min_foot_conf = float(_setting(request.config, "foot_confidence_min", 0.50))
    min_ankle_conf = float(_setting(request.config, "ankle_confidence_min", 0.50))
    events = _reliable_events(events, rows_by_seq)
    foot_samples = []
    ankle_samples = []
    ankle_to_heel_residuals = {side: [] for side in _SIDES}
    for event in events:
        sequence = int(event["seq_frame"])
        side = str(
            event.get("foot", rows_by_seq[sequence].get("lower_foot", ""))
        ).lower()
        foot = _event_contact_point(event, request.foot_contacts_by_seq, min_foot_conf)
        if foot is not None:
            foot_samples.append((foot[0], foot[1], sequence))
        ankle = _ankle_point(rows_by_seq[sequence], side, min_ankle_conf)
        if ankle is not None:
            ankle_samples.append((ankle[0], ankle[1], sequence))
        heel = _foot_joint_point(
            request.foot_contacts_by_seq.get(sequence),
            side,
            "heel",
            min_foot_conf,
        )
        if heel is not None and ankle is not None:
            ankle_to_heel_residuals[side].append(heel[1] - ankle[1])

    minimum_residual_samples = max(
        1, int(_setting(request.config, "ankle_heel_residual_min_samples", 2))
    )
    maximum_residual_noise = float(
        _setting(request.config, "max_ankle_heel_residual_noise_px", 6.0)
    )
    ankle_to_heel_offsets = {}
    ankle_to_heel_noise = {}
    ankle_to_heel_counts = {}
    for side, residuals in ankle_to_heel_residuals.items():
        count = len(residuals)
        noise = _robust_noise(residuals)
        ankle_to_heel_counts[side] = count
        ankle_to_heel_noise[side] = noise
        if count >= minimum_residual_samples and noise <= maximum_residual_noise:
            ankle_to_heel_offsets[side] = _robust_center(residuals)

    geometry = _homography_geometry(request.camera)
    reasons = []
    lane_ratio = lane_world_y = reprojection_error = None
    x_min = x_max = None
    foot_offset = ankle_offset = None
    surface_scope = str(
        request.camera.get(
            "calibrated_surface_scope",
            _setting(request.config, "calibrated_surface_scope", "runway_only"),
        )
    )
    plane_mode = str(
        request.camera.get(
            "surface_plane_mode",
            _setting(request.config, "surface_plane_mode", "shared"),
        )
    )

    if geometry is not None:
        _, world, matrix, inverse, reprojection_error = geometry
        world_y_min, world_y_max = (
            float(np.min(world[:, 1])),
            float(np.max(world[:, 1])),
        )
        world_x_min, world_x_max = (
            float(np.min(world[:, 0])),
            float(np.max(world[:, 0])),
        )
        foot_samples = _samples_within_world_x(
            foot_samples, matrix, world_x_min, world_x_max,
        )
        ankle_samples = _samples_within_world_x(
            ankle_samples, matrix, world_x_min, world_x_max,
        )
        lane_values = [_transform((x, y), matrix)[1] for x, y, _ in foot_samples]
        # A median of 1 sample is just that one sample -- with no protection,
        # a single contact sitting right at the edge of the calibrated range
        # (e.g. the first frame recovered inside a widened margin) becomes
        # the runner's entire lateral path with no outlier resistance at all.
        minimum_lane_samples = max(
            1, int(_setting(request.config, "minimum_lane_samples", 2))
        )
        if len(lane_values) >= minimum_lane_samples:
            lane_world_y = float(
                np.clip(_robust_center(lane_values), world_y_min, world_y_max)
            )
        else:
            ratio = float(_setting(request.config, "anchor_only_lane_ratio", 0.50))
            lane_world_y = world_y_min + np.clip(ratio, 0.0, 1.0) * (
                world_y_max - world_y_min
            )
            reasons.append("reliable_foot_contacts_insufficient_for_lane")
        lane_ratio = (
            0.5
            if world_y_max == world_y_min
            else (lane_world_y - world_y_min) / (world_y_max - world_y_min)
        )
        image_a = _transform((world_x_min, lane_world_y), inverse)
        image_b = _transform((world_x_max, lane_world_y), inverse)
        if abs(image_b[0] - image_a[0]) < 1e-6:
            return None
        anchor_slope = (image_b[1] - image_a[1]) / (image_b[0] - image_a[0])
        anchor_intercept = image_a[1] - anchor_slope * image_a[0]
        x_min, x_max = sorted((image_a[0], image_b[0]))
        foot_residuals = [
            y - (anchor_slope * x + anchor_intercept) for x, y, _ in foot_samples
        ]
        ankle_residuals = [
            y - (anchor_slope * x + anchor_intercept) for x, y, _ in ankle_samples
        ]
        foot_offset = _robust_center(foot_residuals)
        ankle_offset = _robust_center(ankle_residuals)
        if not foot_residuals:
            reasons.append("foot_offset_uncalibrated")
        if not ankle_residuals:
            reasons.append("ankle_offset_uncalibrated")
        foot_slope = ankle_slope = anchor_slope
        foot_intercept = anchor_intercept + foot_offset
        ankle_intercept = anchor_intercept + ankle_offset
        source = (
            "anchor_step_hybrid"
            if foot_residuals and ankle_residuals
            else "anchor_only"
        )
        geometry_mode = "homography"
    else:
        if not foot_samples and not ankle_samples:
            return None
        if foot_samples:
            foot_slope, foot_intercept = _robust_line(
                [(x, y) for x, y, _ in foot_samples]
            )
        else:
            foot_slope, foot_intercept = _robust_line(
                [(x, y) for x, y, _ in ankle_samples]
            )
            reasons.append("foot_model_uses_ankle_fallback")
        if ankle_samples:
            ankle_slope, ankle_intercept = _robust_line(
                [(x, y) for x, y, _ in ankle_samples]
            )
        else:
            ankle_slope, ankle_intercept = foot_slope, foot_intercept
            reasons.append("ankle_model_uses_foot_fallback")
        foot_residuals = [
            y - (foot_slope * x + foot_intercept) for x, y, _ in foot_samples
        ]
        ankle_residuals = [
            y - (ankle_slope * x + ankle_intercept) for x, y, _ in ankle_samples
        ]
        all_x = [x for x, _, _ in foot_samples + ankle_samples]
        x_min, x_max = (min(all_x), max(all_x)) if all_x else (None, None)
        has_linear_span = (
            x_min is not None and x_max is not None and x_max - x_min >= 1.0
        )
        if x_min is not None and x_max is not None:
            contact_span = max(1.0, x_max - x_min)
            extrapolation = float(
                _setting(
                    request.config,
                    "step_only_extrapolation_px",
                    max(30.0, contact_span),
                )
            )
            x_min -= extrapolation
            x_max += extrapolation
        source = "step_only_fallback"
        geometry_mode = "contact_linear" if has_linear_span else "contact_constant"
        reasons.append("anchor_geometry_unavailable")

    foot_noise = _robust_noise(foot_residuals)
    ankle_noise = _robust_noise(ankle_residuals)
    foot_threshold, foot_scale = _model_threshold(foot_noise, request.config, "foot")
    ankle_threshold, ankle_scale = _model_threshold(
        ankle_noise, request.config, "ankle"
    )
    frame_ids = sorted({sequence for _, _, sequence in foot_samples + ankle_samples})
    sample_target = max(1, int(_setting(request.config, "preflight_contact_count", 5)))
    quality = min(1.0, len(frame_ids) / sample_target)
    if geometry is None:
        quality *= 0.65
        if bool(_setting(request.config, "anchor_geometry_required", False)):
            reasons.append("anchor_geometry_required_but_unavailable")
    maximum_reprojection_error = float(
        _setting(request.config, "max_homography_reprojection_error_px", 5.0)
    )
    if (
        reprojection_error is not None
        and reprojection_error > maximum_reprojection_error
    ):
        reasons.append("homography_reprojection_error_too_large")
        quality *= 0.5
    if surface_scope == "runway_and_sand" and plane_mode != "shared":
        reasons.append("sand_surface_requires_piecewise_model")

    return _GroundModel(
        source=source,
        geometry_mode=geometry_mode,
        camera_id=int(request.camera.get("camera_id", 0)),
        surface_scope=surface_scope,
        surface_plane_mode=plane_mode,
        foot_slope=foot_slope,
        foot_intercept=foot_intercept,
        ankle_slope=ankle_slope,
        ankle_intercept=ankle_intercept,
        x_min=x_min,
        x_max=x_max,
        runner_lane_ratio=None if lane_ratio is None else float(lane_ratio),
        runner_lane_world_y=None if lane_world_y is None else float(lane_world_y),
        foot_offset_px=foot_offset,
        ankle_offset_px=ankle_offset,
        foot_noise_px=foot_noise,
        ankle_noise_px=ankle_noise,
        foot_threshold_px=foot_threshold,
        ankle_threshold_px=ankle_threshold,
        foot_scale_px=foot_scale,
        ankle_scale_px=ankle_scale,
        ankle_to_heel_y_offset_px=ankle_to_heel_offsets,
        ankle_to_heel_residual_noise_px=ankle_to_heel_noise,
        ankle_to_heel_sample_count=ankle_to_heel_counts,
        anchor_frames=tuple(frame_ids),
        reprojection_error_px=reprojection_error,
        quality_score=float(quality),
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _off_probability(distance: float, threshold: float, scale: float) -> float:
    z = np.clip((float(distance) - threshold) / scale, -60.0, 60.0)
    return float(1.0 / (1.0 + exp(-float(z))))


def _side_evidence(request, row, side, model):
    sequence = int(row["seq_frame"])
    foot_conf = float(_setting(request.config, "foot_confidence_min", 0.50))
    ankle_conf = float(_setting(request.config, "ankle_confidence_min", 0.50))
    point = _foot_point(request.foot_contacts_by_seq.get(sequence), side, foot_conf)
    joint_kind = "foot"
    if point is None:
        ankle = _ankle_point(row, side, ankle_conf)
        heel_offset = model.ankle_to_heel_y_offset_px.get(side)
        if ankle is not None and heel_offset is not None:
            point = (
                ankle[0],
                ankle[1] + heel_offset,
                ankle[2],
                "heel_estimated_from_ankle",
            )
        else:
            point = ankle
            joint_kind = "ankle"
    if point is None:
        return {
            "valid": False,
            "state": "UNKNOWN",
            "source": None,
            "reason": "foot_and_ankle_unavailable_or_low_confidence",
        }
    x, y, confidence, joint = point
    margin = float(_setting(request.config, "anchor_polygon_margin_px", 3.0))
    edge_excess_px = max(
        0.0,
        0.0 if model.x_min is None else model.x_min - x,
        0.0 if model.x_max is None else x - model.x_max,
    )
    if not model.supports_x(x, margin):
        return {
            "valid": False,
            "state": "UNKNOWN",
            "source": joint,
            "reason": "outside_calibrated_surface",
            "x": x,
            "y": y,
            "confidence": confidence,
            "edge_excess_px": edge_excess_px,
            "edge_tolerance_px": margin,
        }
    ground_y = model.ground_y(x, joint_kind)
    distance = ground_y - y
    if joint_kind == "foot":
        threshold, scale = model.foot_threshold_px, model.foot_scale_px
    else:
        threshold, scale = model.ankle_threshold_px, model.ankle_scale_px
    probability = _off_probability(distance, threshold, scale)
    contact_limit = float(
        _setting(request.config, "per_foot_contact_probability_max", 0.50)
    )
    state = "CONTACT" if probability <= contact_limit else "AIRBORNE"
    return {
        "valid": True,
        "state": state,
        "source": joint,
        "x": x,
        "y": y,
        "confidence": confidence,
        "edge_excess_px": edge_excess_px,
        "edge_tolerance_px": margin,
        "ground_y": ground_y,
        "distance_px": distance,
        "contact_threshold_px": threshold,
        "logistic_scale_px": scale,
        "p_off": probability,
    }


def _evidence_sequence(request, model):
    result = []
    joint_boundary = float(_setting(request.config, "joint_air_boundary", 0.50))
    for index, row in enumerate(request.ankle_rows):
        left = _side_evidence(request, row, "left", model)
        right = _side_evidence(request, row, "right", model)
        p_air = contact_score = None
        if left["valid"] and right["valid"]:
            p_air = float(left["p_off"] * right["p_off"])
            contact_score = 1.0 - p_air
            state = "AIRBORNE" if p_air > joint_boundary else "CONTACT"
        elif left.get("state") == "CONTACT" or right.get("state") == "CONTACT":
            state = "CONTACT"
        else:
            state = "UNKNOWN"
        result.append(
            {
                "index": index,
                "seq_frame": int(row["seq_frame"]),
                "state": state,
                "left": left,
                "right": right,
                "p_air": p_air,
                "contact_score": contact_score,
            }
        )
    return result


def _airborne_runs(evidence, config):
    minimum = max(1, int(_setting(config, "debounce_frames", 2)))
    max_unknown = max(0, int(_setting(config, "max_unknown_gap_frames", 2)))
    air_indices = [
        index for index, item in enumerate(evidence) if item["state"] == "AIRBORNE"
    ]
    if not air_indices:
        return []
    groups = [[air_indices[0]]]
    for index in air_indices[1:]:
        previous = groups[-1][-1]
        gap = evidence[previous + 1 : index]
        if len(gap) <= max_unknown and all(item["state"] == "UNKNOWN" for item in gap):
            groups[-1].append(index)
        else:
            groups.append([index])
    return [
        (group[0], group[-1])
        for group in groups
        if sum(
            evidence[i]["state"] == "AIRBORNE" for i in range(group[0], group[-1] + 1)
        )
        >= minimum
    ]


def _side_has_sustained_contact(evidence, start, side, debounce, stop):
    limit = min(len(evidence), stop)
    for index in range(max(0, start), limit):
        end = min(limit, index + debounce)
        if end - index < debounce:
            break
        if all(
            evidence[position][side].get("state") == "CONTACT"
            for position in range(index, end)
        ):
            return index
    return None


def _last_contact(evidence, start, lookback):
    for index in range(start - 1, max(-1, start - lookback - 1), -1):
        if any(evidence[index][side].get("state") == "CONTACT" for side in _SIDES):
            return index
    return None


def _estimated_touchdown(evidence, start, stop, config):
    """Estimate a ground crossing from reliable descending foot evidence.

    A confidence collapse by itself is deliberately insufficient.  At least
    two reliable samples from the same foot must be approaching the frozen
    ground model, and their short linear extrapolation must cross the contact
    threshold inside an UNKNOWN run.
    """
    history = max(2, int(_setting(config, "landing_trajectory_history_frames", 5)))
    minimum_descent = float(
        _setting(config, "landing_min_ground_approach_px_per_frame", 0.75)
    )
    limit = min(stop, len(evidence))
    estimates = []
    for side in _SIDES:
        samples = []
        for index in range(max(0, start - history), start):
            side_evidence = evidence[index][side]
            distance = _finite(side_evidence.get("distance_px"))
            if side_evidence.get("valid") and distance is not None:
                samples.append((index, distance))
        if len(samples) < 2:
            continue
        recent = samples[-min(3, len(samples)) :]
        slopes = [
            (current_distance - previous_distance) / (current_index - previous_index)
            for (previous_index, previous_distance), (
                current_index,
                current_distance,
            ) in pairwise(recent)
            if current_index > previous_index
        ]
        approach = _robust_center(slopes)
        if approach > -minimum_descent:
            continue
        last_index, last_distance = recent[-1]
        threshold = float(evidence[last_index][side].get("contact_threshold_px", 0.0))
        frames_to_ground = max(
            1,
            int(np.ceil(max(0.0, last_distance - threshold) / -approach)),
        )
        predicted = last_index + frames_to_ground
        if start <= predicted < limit and evidence[predicted]["state"] == "UNKNOWN":
            estimates.append(predicted)
    return min(estimates, default=None)


def _contact_detail(evidence_item, preferred_side=None):
    choices = []
    for side in _SIDES:
        item = evidence_item[side]
        if item.get("state") == "CONTACT":
            choices.append((float(item.get("p_off", 1.0)), side, item))
    if preferred_side is not None:
        choices.sort(key=lambda choice: (choice[1] != preferred_side, choice[0]))
    else:
        choices.sort(key=lambda choice: choice[0])
    return choices[0][1:] if choices else (None, None)


def _candidate_records(request, evidence):
    fps = max(float(request.fps), 1.0)
    debounce = max(1, int(_setting(request.config, "debounce_frames", 2)))
    pre_window = int(
        _setting(request.config, "pre_contact_search_frames", round(0.35 * fps))
    )
    post_window = int(
        _setting(request.config, "post_contact_search_frames", round(0.75 * fps))
    )
    max_unknown_ratio = float(
        _setting(request.config, "max_candidate_unknown_ratio", 0.40)
    )
    allow_initial_airborne = bool(
        _setting(
            request.config,
            "allow_initial_airborne_at_camera_boundary",
            False,
        )
    )
    max_unknown_gap = max(
        0,
        int(_setting(request.config, "max_unknown_gap_frames", 2)),
    )
    candidates = []
    for start, end in _airborne_runs(evidence, request.config):
        before = _last_contact(evidence, start, pre_window)
        search_stop = min(len(evidence), end + post_window + 1)
        contacts = []
        for side in _SIDES:
            index = _side_has_sustained_contact(
                evidence, end + 1, side, debounce, search_stop
            )
            if index is not None:
                contacts.append((index, side))
        estimated_index = _estimated_touchdown(
            evidence, end + 1, search_stop, request.config
        )
        observed = min(contacts) if contacts else None
        if observed is not None and (
            estimated_index is None or observed[0] <= estimated_index
        ):
            touchdown_index, landing_foot = observed
            touchdown_kind = "observed"
        else:
            touchdown_index = estimated_index
            landing_foot = None
            touchdown_kind = "estimated" if touchdown_index is not None else None
        span_stop = touchdown_index if touchdown_index is not None else end
        span = evidence[start : span_stop + 1]
        unknown_ratio = (
            sum(item["state"] == "UNKNOWN" for item in span) / len(span)
            if span
            else 1.0
        )
        reasons = []
        warnings = []
        leading_gap = evidence[:start]
        entered_airborne_at_boundary = (
            before is None
            and start <= max_unknown_gap
            and all(item["state"] == "UNKNOWN" for item in leading_gap)
        )
        if before is None and not (
            allow_initial_airborne and entered_airborne_at_boundary
        ):
            reasons.append("missing_pre_contact")
        if touchdown_index is None:
            reasons.append("missing_post_contact_or_landing_evidence")
        if unknown_ratio > max_unknown_ratio:
            reasons.append("too_many_unknown_frames")
        if (
            touchdown_kind == "observed"
            and touchdown_index is not None
            and any(
                item["state"] == "UNKNOWN"
                for item in evidence[end + 1 : touchdown_index]
            )
        ):
            warnings.append("unknown_gap_before_observed_touchdown")
        candidates.append(
            {
                "start_index": start,
                "end_index": end,
                "takeoff_index": before,
                "touchdown_index": touchdown_index,
                "touchdown_kind": touchdown_kind,
                "landing_foot": landing_foot,
                "duration_frames": end - start + 1,
                "unknown_ratio": float(unknown_ratio),
                "valid": not reasons,
                "reasons": reasons,
                "warnings": warnings,
                "entered_airborne_at_camera_boundary": bool(
                    allow_initial_airborne and entered_airborne_at_boundary
                ),
            }
        )
    return candidates


def _choose_longest(candidates):
    valid = [candidate for candidate in candidates if candidate["valid"]]
    return max(
        valid,
        key=lambda candidate: (
            candidate.get("entered_airborne_at_camera_boundary", False),
            candidate["duration_frames"],
            -candidate["unknown_ratio"],
        ),
        default=None,
    )


def _event_subset_before(events, sequence_frame, count, rows_by_seq):
    eligible = [
        event
        for event in _reliable_events(events, rows_by_seq)
        if int(event["seq_frame"]) <= int(sequence_frame)
    ]
    return eligible[-max(1, int(count)) :]


def _maximum_compression_index(request, touchdown_index):
    stop = min(
        len(request.ankle_rows),
        touchdown_index + max(2, round(float(request.fps) * 0.40)),
    )

    def vertical_position(index):
        value = _finite(request.ankle_rows[index].get("lower_ankle_y"))
        return -np.inf if value is None else value

    return max(range(touchdown_index, stop), key=vertical_position)


def _event_frame(row, index, extra=None):
    result = {
        "index": int(index),
        "seq_frame": int(row["seq_frame"]),
        "orig_frame": int(row.get("orig_frame", row["seq_frame"])),
        "time_s": float(row.get("time_s", 0.0)),
        "seq_time_s": float(row.get("seq_time_s", 0.0)),
    }
    if extra:
        result.update(extra)
    return result


def _public_candidates(candidates, evidence, selected=None):
    return [
        {
            "candidate_id": candidate_id,
            "first_airborne_frame": evidence[item["start_index"]]["seq_frame"],
            "last_airborne_frame": evidence[item["end_index"]]["seq_frame"],
            "takeoff_frame": (
                None
                if item["takeoff_index"] is None
                else evidence[item["takeoff_index"]]["seq_frame"]
            ),
            "touchdown_frame": (
                None
                if item["touchdown_index"] is None
                else evidence[item["touchdown_index"]]["seq_frame"]
            ),
            "touchdown_kind": item["touchdown_kind"],
            "duration_frames": item["duration_frames"],
            "unknown_ratio": item["unknown_ratio"],
            "valid": item["valid"],
            "selected": item is selected,
            "entered_airborne_at_camera_boundary": item.get(
                "entered_airborne_at_camera_boundary", False
            ),
            "reasons": list(item["reasons"]),
            "warnings": list(item.get("warnings", [])),
        }
        for candidate_id, item in enumerate(candidates, start=1)
    ]


def _frame_decision_reason(item):
    if item["state"] == "AIRBORNE":
        return "bilateral_p_air_above_boundary"
    if item["state"] == "CONTACT":
        if item["p_air"] is not None:
            return "bilateral_p_air_at_or_below_boundary"
        contacts = [
            side for side in _SIDES if item[side].get("state") == "CONTACT"
        ]
        return "contact_from_" + "_and_".join(contacts)
    reasons = []
    for side in _SIDES:
        side_item = item[side]
        if not side_item.get("valid"):
            reasons.append(f"{side}:{side_item.get('reason', 'invalid_evidence')}")
    return "|".join(reasons) or "insufficient_bilateral_evidence"


def _debug_trace(evidence, candidates=None, selected=None):
    candidates = candidates or []
    trace = []
    for index, item in enumerate(evidence):
        row = {
            "index": index,
            "seq_frame": item["seq_frame"],
            "state": item["state"],
            "decision_reason": _frame_decision_reason(item),
            "p_off_left": item["left"].get("p_off"),
            "p_off_right": item["right"].get("p_off"),
            "p_air": item["p_air"],
            "contact_score": item["contact_score"],
        }
        for side in _SIDES:
            side_item = item[side]
            for diagnostic_field in (
                "valid",
                "state",
                "source",
                "reason",
                "x",
                "y",
                "confidence",
                "edge_excess_px",
                "edge_tolerance_px",
                "ground_y",
                "distance_px",
                "contact_threshold_px",
                "logistic_scale_px",
                "p_off",
            ):
                row[f"{side}_{diagnostic_field}"] = side_item.get(
                    diagnostic_field
                )
        memberships = [
            str(candidate_id)
            for candidate_id, candidate in enumerate(candidates, start=1)
            if candidate["start_index"] <= index <= (
                candidate["touchdown_index"]
                if candidate["touchdown_index"] is not None
                else candidate["end_index"]
            )
        ]
        roles = []
        if selected is not None:
            for role, field in (
                ("takeoff_contact", "takeoff_index"),
                ("first_airborne", "start_index"),
                ("last_airborne", "end_index"),
                ("touchdown", "touchdown_index"),
            ):
                if selected.get(field) == index:
                    roles.append(role)
        row["candidate_ids"] = "|".join(memberships)
        row["selected_candidate"] = bool(
            selected is not None
            and selected["start_index"] <= index <= (
                selected["touchdown_index"]
                if selected["touchdown_index"] is not None
                else selected["end_index"]
            )
        )
        row["decision_roles"] = "|".join(roles)
        trace.append(row)
    return trace


def _empty_result(reason: str, model=None, candidates=None, evidence=None):
    return {
        "detected": False,
        "algorithm": ALGORITHM_NAME,
        "takeoff": None,
        "first_airborne": None,
        "last_airborne": None,
        "touchdown": None,
        "max_compression": None,
        "flight_duration_frames": None,
        "flight_duration_seconds": None,
        "ground_model": None if model is None else model.summary(),
        "candidates": _public_candidates(candidates or [], evidence or []),
        "debug_trace": _debug_trace(evidence or [], candidates or []),
        "quality_score": 0.0,
        "needs_review": True,
        "reasons": [reason],
    }


def _model_blocker(model):
    if "anchor_geometry_required_but_unavailable" in model.reasons:
        return "anchor_geometry_required_but_unavailable"
    if "homography_reprojection_error_too_large" in model.reasons:
        return "homography_reprojection_error_too_large"
    if (
        model.surface_scope == "runway_and_sand"
        and model.surface_plane_mode != "shared"
    ):
        return "piecewise_sand_surface_model_not_configured"
    return None


def detect_running_long_jump(request: RunningLongJumpRequest) -> dict:
    """Detect the longest valid bilateral flight behind one stable interface."""
    rows = list(request.ankle_rows)
    if len(rows) < 3:
        return _empty_result("insufficient_pose_rows")
    rows.sort(key=lambda row: int(row["seq_frame"]))
    if rows != list(request.ankle_rows):
        request = RunningLongJumpRequest(
            ankle_rows=rows,
            accepted_events=request.accepted_events,
            foot_contacts_by_seq=request.foot_contacts_by_seq,
            camera=request.camera,
            fps=request.fps,
            config=request.config,
        )
    rows_by_seq = {int(row["seq_frame"]): row for row in rows}
    events = _reliable_events(request.accepted_events, rows_by_seq)
    if not events:
        return _empty_result("reliable_preflight_contacts_unavailable")

    provisional_model = _build_ground_model(request, events)
    if provisional_model is None:
        return _empty_result("ground_model_unavailable")
    blocker = _model_blocker(provisional_model)
    if blocker is not None:
        return _empty_result(blocker, provisional_model)
    provisional_evidence = _evidence_sequence(request, provisional_model)
    provisional_candidates = _candidate_records(request, provisional_evidence)
    provisional = _choose_longest(provisional_candidates)
    if provisional is None:
        return _empty_result(
            "no_complete_bilateral_flight",
            provisional_model,
            provisional_candidates,
            provisional_evidence,
        )

    boundary_entry = provisional.get("entered_airborne_at_camera_boundary", False)
    contact_count = int(_setting(request.config, "preflight_contact_count", 5))
    if boundary_entry:
        # The provisional model judged this candidate's flight already under
        # way at frame 0 of this camera, so there is no takeoff_index to
        # anchor "preflight" on. That must not mean giving up on refinement
        # entirely -- the events already accepted for this camera (whatever
        # led to entered_airborne_at_camera_boundary in the first place) are
        # still the best available preflight evidence. Try refining against
        # them, anchored on the airborne run's own start frame, and only
        # fall back to the sparse provisional model if that refinement
        # itself is unusable.
        boundary_sequence = provisional_evidence[provisional["start_index"]][
            "seq_frame"
        ]
        frozen_events = _event_subset_before(
            events, boundary_sequence, contact_count, rows_by_seq
        )
        model = provisional_model
        if frozen_events:
            refined_model = _build_ground_model(request, frozen_events)
            if refined_model is not None and _model_blocker(refined_model) is None:
                model = refined_model
    else:
        takeoff_sequence = provisional_evidence[provisional["takeoff_index"]][
            "seq_frame"
        ]
        frozen_events = _event_subset_before(
            events, takeoff_sequence, contact_count, rows_by_seq
        )
        model = _build_ground_model(request, frozen_events)
    if model is None:
        return _empty_result("frozen_ground_model_unavailable")
    blocker = _model_blocker(model)
    if blocker is not None:
        return _empty_result(blocker, model)
    evidence = _evidence_sequence(request, model)
    candidates = _candidate_records(request, evidence)
    selected = _choose_longest(candidates)
    if selected is None:
        return _empty_result(
            "no_complete_bilateral_flight_after_freeze", model, candidates, evidence
        )

    takeoff_index = selected["takeoff_index"]
    touchdown_index = selected["touchdown_index"]
    max_compression_index = _maximum_compression_index(request, touchdown_index)
    if takeoff_index is None:
        takeoff_foot = takeoff_contact = None
    else:
        takeoff_foot, takeoff_contact = _contact_detail(evidence[takeoff_index])
    touchdown_evidence = evidence[touchdown_index]
    landing_foot, contact = _contact_detail(
        touchdown_evidence, selected.get("landing_foot")
    )
    touchdown_extra = {
        "kind": selected["touchdown_kind"],
        "foot": landing_foot,
        "contact_joint": None if contact is None else contact.get("source"),
        "contact_x": None if contact is None else contact.get("x"),
        "contact_y": None if contact is None else contact.get("y"),
        "contact_conf": None if contact is None else contact.get("confidence"),
    }
    duration = int(selected["duration_frames"])
    reasons = list(model.reasons)
    reasons.extend(selected.get("warnings", []))
    if selected["touchdown_kind"] == "estimated":
        reasons.append("touchdown_estimated_from_ground_approach_trajectory")
    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    if len(valid_candidates) > 1:
        ordered = sorted(
            (candidate["duration_frames"] for candidate in valid_candidates),
            reverse=True,
        )
        ambiguity = int(_setting(request.config, "ambiguous_duration_gap_frames", 2))
        if ordered[0] - ordered[1] <= ambiguity:
            reasons.append("longest_candidate_is_ambiguous")

    return {
        "detected": True,
        "algorithm": ALGORITHM_NAME,
        "camera_id": int(request.camera.get("camera_id", rows[0].get("cam", 0))),
        "takeoff": (
            None
            if takeoff_index is None
            else _event_frame(
                rows[takeoff_index],
                takeoff_index,
                {
                    "foot": takeoff_foot,
                    "contact_joint": (
                        None
                        if takeoff_contact is None
                        else takeoff_contact.get("source")
                    ),
                },
            )
        ),
        "first_airborne": _event_frame(
            rows[selected["start_index"]], selected["start_index"]
        ),
        "last_airborne": _event_frame(
            rows[selected["end_index"]], selected["end_index"]
        ),
        "touchdown": _event_frame(
            rows[touchdown_index], touchdown_index, touchdown_extra
        ),
        "max_compression": _event_frame(
            rows[max_compression_index], max_compression_index
        ),
        "flight_duration_frames": duration,
        "flight_duration_seconds": duration / max(float(request.fps), 1.0),
        "ground_model": model.summary(),
        "candidates": _public_candidates(candidates, evidence, selected),
        "debug_trace": _debug_trace(evidence, candidates, selected),
        "quality_score": float(model.quality_score * (1.0 - selected["unknown_ratio"])),
        "needs_review": bool(reasons),
        "reasons": list(dict.fromkeys(reasons)),
    }


__all__ = ["ALGORITHM_NAME", "RunningLongJumpRequest", "detect_running_long_jump"]
