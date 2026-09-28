"""Choose a running long jump from one completed set of touchdown events.

This detector deliberately does not estimate a ground Y line.  Camera
homography remains responsible for measuring positions; temporal jump
selection uses the corrected ankle trajectory *between* accepted contacts.
Neither a sand-pit pose nor a later step may redefine the takeoff baseline.
The older probabilistic ground-model detector remains separately selectable.
"""

from __future__ import annotations

from math import isfinite

try:
    from scripts.analysis.running_long_jump import RunningLongJumpRequest
except ModuleNotFoundError:  # Support direct ``python ankle_step_stride.py`` use.
    from running_long_jump import RunningLongJumpRequest  # type: ignore[no-redef]

ALGORITHM_NAME = "event_pair_bilateral_valley"


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _frame(row, kind=None):
    result = {key: row[key] for key in (
        "seq_frame", "orig_frame", "time_s", "seq_time_s",
    ) if key in row}
    if kind is not None:
        result["kind"] = kind
    return result


def _valid_event(event, rows_by_frame):
    frame = int(event["seq_frame"])
    return (
        frame in rows_by_frame
        and event.get("event_type", "run_step") == "run_step"
        and event.get("contact_valid", True) is not False
    )


def _bilateral_elevation(row, start, end, min_confidence):
    elevations = []
    for side in ("left", "right"):
        key = f"{side}_ankle_y"
        confidence = _number(row.get(f"{side}_ankle_conf"))
        values = [_number(source.get(key)) for source in (row, start, end)]
        if confidence is None or confidence < min_confidence or None in values:
            return None
        elevations.append(min(values[1:]) - values[0])
    return min(elevations)


def _candidate(start_event, end_event, rows_by_frame, min_confidence, min_rise):
    start_frame = int(start_event["seq_frame"])
    end_frame = int(end_event["seq_frame"])
    start = rows_by_frame[start_frame]
    end = rows_by_frame[end_frame]
    middle = [rows_by_frame[frame] for frame in range(start_frame + 1, end_frame)
              if frame in rows_by_frame]
    start_y = _number(start.get("lower_ankle_y"))
    end_y = _number(end.get("lower_ankle_y"))
    valid_y = [_number(row.get("lower_ankle_y")) for row in middle]
    valid_y = [value for value in valid_y if value is not None]
    valley_depth = (
        min(start_y, end_y) - min(valid_y)
        if start_y is not None and end_y is not None and valid_y else None
    )
    elevated_frames = []
    for row in middle:
        elevation = _bilateral_elevation(row, start, end, min_confidence)
        if elevation is not None and elevation >= min_rise:
            elevated_frames.append(int(row["seq_frame"]))
    longest_run = 0
    run = 0
    previous = None
    for frame in elevated_frames:
        run = run + 1 if previous is not None and frame == previous + 1 else 1
        longest_run = max(longest_run, run)
        previous = frame
    valid = longest_run >= 3 and valley_depth is not None and valley_depth >= min_rise
    return {
        "candidate_id": f"contact_{start_frame}_{end_frame}",
        "takeoff_frame": start_frame,
        "first_airborne_frame": start_frame + 1,
        "last_airborne_frame": end_frame - 1,
        "touchdown_frame": end_frame,
        "touchdown_kind": "accepted_step_candidate",
        "duration_frames": end_frame - start_frame - 1,
        "valley_depth_px": valley_depth,
        "bilateral_elevated_run_frames": longest_run,
        "valid": valid,
        "selected": False,
        "reasons": [] if valid else ["no_sustained_bilateral_ankle_valley"],
        "warnings": [],
    }


def detect_event_pair_long_jump(request: RunningLongJumpRequest) -> dict:
    """Find the longest sustained two-ankle valley between accepted contacts.

    The contact detector has already run exactly once.  A candidate outside
    the calibrated X range can still define the *time* of takeoff; its
    position quality is a separate concern for distance measurement.
    """
    rows = sorted(request.ankle_rows, key=lambda row: int(row["seq_frame"]))
    rows_by_frame = {int(row["seq_frame"]): row for row in rows}
    events = sorted(
        (event for event in request.accepted_events
         if _valid_event(event, rows_by_frame)),
        key=lambda event: int(event["seq_frame"]),
    )
    camera_id = int(request.camera.get("camera_id", rows[0]["cam"] if rows else 0))
    min_confidence = float(request.config.get("ankle_confidence_min", 0.5))
    min_rise = float(request.config.get("bilateral_valley_min_rise_px", 8.0))
    candidates = [_candidate(a, b, rows_by_frame, min_confidence, min_rise)
                  for a, b in zip(events, events[1:])]
    valid = [candidate for candidate in candidates if candidate["valid"]]
    selected = max(valid, key=lambda candidate: (
        candidate["duration_frames"], candidate["valley_depth_px"],
    )) if valid else None
    if selected is not None:
        selected["selected"] = True

    # This is provenance, not a Y-ground model.  In particular, 518 and 657
    # must never become preflight calibration samples for a 476 takeoff.
    preflight = [int(event["seq_frame"]) for event in events
                 if selected is not None
                 and int(event["seq_frame"]) <= selected["takeoff_frame"]]
    model = {
        "source": "not_used_for_event_pair",
        "anchor_frames": [],
        "preflight_contact_frames": preflight,
        "frozen": None,
        "used_for_temporal_decision": False,
        "needs_review": not bool(preflight),
        "reasons": [] if preflight else ["preflight_contact_unavailable"],
    }
    trace = []
    selected_start = selected["takeoff_frame"] if selected else None
    selected_end = selected["touchdown_frame"] if selected else None
    for row in rows:
        frame = int(row["seq_frame"])
        roles = []
        if frame == selected_start:
            roles.append("takeoff")
        if frame == selected_end:
            roles.append("touchdown")
        elevation = None
        if selected_start is not None and selected_start < frame < selected_end:
            elevation = _bilateral_elevation(
                row, rows_by_frame[selected_start], rows_by_frame[selected_end],
                min_confidence,
            )
        trace.append({
            "seq_frame": frame, "camera_id": camera_id,
            "state": "BILATERAL_ELEVATED" if elevation is not None
            and elevation >= min_rise else "NOT_CONFIRMED_AIRBORNE",
            "decision_reason": "accepted_contact_pair_and_ankle_valley",
            "bilateral_elevation_px": elevation,
            "candidate_ids": [candidate["candidate_id"] for candidate in candidates
                              if candidate["takeoff_frame"] <= frame <= candidate["touchdown_frame"]],
            "selected_candidate": selected["candidate_id"] if selected is not None
            and selected_start <= frame <= selected_end else None,
            "decision_roles": "|".join(roles),
        })

    base = {
        "detected": selected is not None, "algorithm": ALGORITHM_NAME,
        "camera_id": camera_id, "ground_model": model,
        "candidates": candidates, "debug_trace": trace,
        "needs_review": True, "quality_score": 0.0,
    }
    if selected is None:
        return {**base, "takeoff": None, "first_airborne": None,
                "last_airborne": None, "touchdown": None,
                "max_compression": None, "flight_duration_frames": None,
                "flight_duration_seconds": None,
                "reasons": ["no_complete_bilateral_valley_between_contacts"]}

    start = selected["takeoff_frame"]
    end = selected["touchdown_frame"]
    takeoff = next(event for event in events if int(event["seq_frame"]) == start)
    landing = next(event for event in events if int(event["seq_frame"]) == end)
    touchdown = _frame(rows_by_frame[end], "accepted_step_candidate")
    touchdown.update({key: landing.get(key) for key in (
        "foot", "contact_joint", "contact_x", "contact_y", "contact_conf",
    )})
    reasons = ["airborne_bounds_inferred_between_contact_candidates"]
    if _number(landing.get("contact_conf")) is None or (
        _number(landing.get("contact_conf")) or 0.0
    ) < float(request.config.get("foot_confidence_min", 0.5)):
        reasons.append("landing_contact_low_or_missing_confidence")
    if takeoff.get("contact_rejection_reason"):
        reasons.append("takeoff_position_has_spatial_warning")
    return {**base,
        "takeoff": {**_frame(rows_by_frame[start], "accepted_step_candidate"),
                    "foot": takeoff.get("foot"),
                    "contact_joint": takeoff.get("contact_joint")},
        "first_airborne": _frame(rows_by_frame[start + 1], "inferred_between_contacts"),
        "last_airborne": _frame(rows_by_frame[end - 1], "inferred_between_contacts"),
        "touchdown": touchdown,
        "max_compression": _frame(rows_by_frame[end], "not_separately_estimated"),
        "flight_duration_frames": selected["duration_frames"],
        "flight_duration_seconds": selected["duration_frames"] / max(float(request.fps), 1.0),
        "quality_score": min(1.0, selected["bilateral_elevated_run_frames"] / 10.0),
        "reasons": reasons,
    }


__all__ = ["ALGORITHM_NAME", "detect_event_pair_long_jump"]
