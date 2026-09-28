"""依軌跡品質指標為每台相機評分並選出主跑者。"""

from itertools import pairwise

from .camera_setup import _point_track_area_proximity


def _compute_camera_total_frames(all_detections, frame_ranges_by_cam):
    """算出各相機的分母幀數，供 coverage 計算使用。

    frame_ranges_by_cam 有給時優先採用（prescan 有效區間總長）；否則用該相機所有偵測中
    最大的 frame_idx+1 推估。回傳 defaultdict(int)，缺項時安全回傳 0。
    """
    from collections import defaultdict

    cam_total_frames = defaultdict(int)
    if frame_ranges_by_cam:
        for cam_idx, ranges in frame_ranges_by_cam.items():
            cam_total_frames[cam_idx] = sum(int(end) - int(start) + 1 for start, end in ranges)
    else:
        for row in all_detections:
            ci = row['cam_idx']
            cam_total_frames[ci] = max(cam_total_frames[ci], row['frame_idx'] + 1)
    return cam_total_frames


def _candidate_progress(rows, projections, pixel_span):
    if projections and pixel_span:
        return min(1.0, max(0.0, (max(projections) - min(projections)) / pixel_span))
    centers = [row['center_x'] for row in rows]
    return min(1.0, (max(centers) - min(centers)) / 1920.0) if len(centers) > 1 else 0.0


def _candidate_monotonicity(rows, projections):
    values = projections or [row['center_x'] for row in rows]
    if len(values) <= 1:
        return 0.0
    return sum(1 for previous, current in pairwise(values) if current > previous) / (
        len(values) - 1
    )


def _candidate_track_area_proximity(rows, camera):
    quadrilateral = camera.get('quad_roi')
    points = [
        (row['ground_x'], row['ground_y'])
        for row in rows
        if row.get('ground_x') is not None and row.get('ground_y') is not None
    ]
    if quadrilateral is None or not points:
        return 1.0
    margin = camera.get('homography_lane_margin_px', 120)
    return sum(
        _point_track_area_proximity(point, quadrilateral, margin)
        for point in points
    ) / len(points)


def _score_track_candidate(rows, cam, total_frames):
    """依單一 (cam_idx, track_id) 的偵測序列，計算軌跡品質指標與綜合分數。

    total_frames 為該相機的分母幀數（用於 coverage）。回傳 dict：
    n_frames/coverage/progress/monotonic/start_proximity/roi_ratio/
    track_area_proximity/score（不含 cam_idx/track_id，由呼叫端補上）。
    """
    rows_s = sorted(rows, key=lambda r: r['frame_idx'])
    n = len(rows_s)
    coverage = n / (total_frames or 1)

    pixel_span = cam.get('pixel_span')
    projections = [r['proj_px'] for r in rows_s if r['proj_px'] is not None]
    progress = _candidate_progress(rows_s, projections, pixel_span)
    monotonic = _candidate_monotonicity(rows_s, projections)
    first_proj = projections[0] if projections else 0.0
    start_proximity = 1.0 / (1.0 + max(0.0, float(first_proj)))

    if pixel_span and projections:
        roi_ratio = sum(1 for p in projections if 0 <= p <= pixel_span) / len(projections)
    else:
        roi_ratio = 1.0

    track_area_proximity = _candidate_track_area_proximity(rows_s, cam)

    score = (0.10 * coverage + 0.10 * progress + 0.15 * monotonic +
             0.10 * start_proximity + 0.10 * roi_ratio +
             0.45 * track_area_proximity)

    return {
        'n_frames':        n,
        'coverage':        round(coverage, 4),
        'progress':        round(progress, 4),
        'monotonic':       round(monotonic, 4),
        'start_proximity': round(start_proximity, 4),
        'roi_ratio':       round(roi_ratio, 4),
        'track_area_proximity': round(track_area_proximity, 4),
        'score':           round(score, 4),
    }


def _select_best_candidate(summaries, cam_idx):
    """從 summaries 中選出指定相機分數最高的候選 track_id，並印出診斷訊息。

    回傳 track_id；找不到候選人時回傳 None。
    """
    candidates = [s for s in summaries if s['cam_idx'] == cam_idx
                  and s['n_frames'] >= 10
                  and s['progress'] >= 0.3
                  and s['monotonic'] >= 0.55]
    if not candidates:
        candidates = [s for s in summaries if s['cam_idx'] == cam_idx]
    if not candidates:
        print(f"  ⚠️  相機 {cam_idx+1}: 無候選主跑者")
        return None

    candidates.sort(key=lambda s: s['score'], reverse=True)
    best = candidates[0]

    if len(candidates) > 1:
        second = candidates[1]
        if best['score'] > 0 and (best['score'] - second['score']) / best['score'] < 0.10:
            print(f"  ⚠️  相機 {cam_idx+1}: 主跑者選擇不穩定"
                  f"（第1={best['track_id']} {best['score']:.3f}，"
                  f"第2={second['track_id']} {second['score']:.3f}，差距<10%）")

    print(f"  [two_pass] 相機 {cam_idx+1}: 選中 ID={best['track_id']} "
          f"score={best['score']:.3f} "
          f"coverage={best['coverage']:.2f} progress={best['progress']:.2f} "
          f"monotonic={best['monotonic']:.2f} "
          f"track_area={best['track_area_proximity']:.2f}")

    return best['track_id']


def _score_and_select_runners(all_detections, cameras, frame_ranges_by_cam=None):
    """
    依軌跡品質評分，選出每台相機的主跑者。
    回傳 (preset_ids, summaries)：
      preset_ids  dict {cam_idx: track_id}
      summaries   list[dict] 所有 ID 的指標與分數
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for row in all_detections:
        groups[(row['cam_idx'], row['track_id'])].append(row)

    cam_total_frames = _compute_camera_total_frames(all_detections, frame_ranges_by_cam)

    summaries = []
    for (cam_idx, track_id), rows in groups.items():
        metrics = _score_track_candidate(rows, cameras[cam_idx], cam_total_frames[cam_idx])
        summaries.append({'cam_idx': cam_idx, 'track_id': track_id, **metrics})

    preset_ids = {}
    for cam_idx in range(len(cameras)):
        track_id = _select_best_candidate(summaries, cam_idx)
        if track_id is None:
            raise ValueError(
                f"相機 {cam_idx + 1} 無法選出主跑者；已停止分析，不會退回逐幀選人"
            )
        preset_ids[cam_idx] = track_id

    return preset_ids, summaries


