"""修補已選主跑者短暫消失時的 track_id 斷點。"""

from dataclasses import dataclass
from itertools import pairwise


def _target_frame_indices(frame_cache, camera_index, target_id):
    return sorted(
        frame_index
        for (cached_camera, frame_index), detections in frame_cache.items()
        if cached_camera == camera_index
        and any(detection['track_id'] == target_id for detection in detections)
    )


def _target_detection(frame_cache, camera_index, frame_index, target_id):
    return next(
        detection for detection in frame_cache[(camera_index, frame_index)]
        if detection['track_id'] == target_id
    )


def _interpolated_track_profile(start_detection, end_detection, ratio):
    def profile(detection):
        return (
            (detection['bx1'] + detection['bx2']) / 2,
            (detection['by1'] + detection['by2']) / 2,
            detection['by2'] - detection['by1'],
        )

    start = profile(start_detection)
    end = profile(end_detection)
    return tuple(left + ratio * (right - left) for left, right in zip(start, end))


def _closest_stitch_candidate(detections, expected_profile, max_distance):
    expected_x, expected_y, expected_height = expected_profile
    best_detection, best_score = None, float('inf')
    for detection in detections:
        center_x = (detection['bx1'] + detection['bx2']) / 2
        center_y = (detection['by1'] + detection['by2']) / 2
        height = detection['by2'] - detection['by1']
        distance = ((center_x - expected_x) ** 2 + (center_y - expected_y) ** 2) ** 0.5
        size_ratio = max(height, expected_height) / max(min(height, expected_height), 1.0)
        if distance > max_distance or size_ratio > 1.5:
            continue
        score = distance + 50.0 * (size_ratio - 1.0)
        if score < best_score:
            best_detection, best_score = detection, score
    return best_detection


@dataclass(frozen=True)
class _TrackGap:
    frame_cache: dict
    camera_index: int
    target_id: int
    start_frame: int
    end_frame: int
    max_distance: float


def _stitch_track_gap(gap):
    start = _target_detection(
        gap.frame_cache, gap.camera_index, gap.start_frame, gap.target_id,
    )
    end = _target_detection(
        gap.frame_cache, gap.camera_index, gap.end_frame, gap.target_id,
    )
    stitched = 0
    for frame_index in range(gap.start_frame + 1, gap.end_frame):
        detections = gap.frame_cache.get((gap.camera_index, frame_index))
        if not detections or any(d['track_id'] == gap.target_id for d in detections):
            continue
        ratio = (frame_index - gap.start_frame) / (gap.end_frame - gap.start_frame)
        candidate = _closest_stitch_candidate(
            detections,
            _interpolated_track_profile(start, end, ratio),
            gap.max_distance,
        )
        if candidate is not None:
            candidate['track_id'] = gap.target_id
            stitched += 1
    return stitched


def _stitch_target_id(frame_cache, preset_ids, fps=30.0, max_dist_px=100):
    """
    In-place 修補 frame_cache：當 target_id 短暫消失，
    若空缺幀有其他 ID 的 bbox 位置與大小接近預期（線性插值），就把 track_id 改成 target_id。
    max_gap 由 fps 自動推算（約 0.2 秒），上限 15 幀。
    """
    max_gap = max(5, min(15, round(fps * 0.2)))
    for cam_idx, target_id in preset_ids.items():
        target_frames = _target_frame_indices(frame_cache, cam_idx, target_id)
        if len(target_frames) < 2:
            continue

        stitched = 0
        for start_f, end_f in pairwise(target_frames):
            gap = end_f - start_f - 1
            if gap < 1 or gap > max_gap:
                continue
            stitched += _stitch_track_gap(_TrackGap(
                frame_cache, cam_idx, target_id, start_f, end_f, max_dist_px,
            ))

        if stitched:
            print(f"  [stitch] 相機 {cam_idx + 1}: 修補 {stitched} 幀 target_id={target_id}")


