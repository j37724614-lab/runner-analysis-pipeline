"""相機切換（或最後一台退出 ROI）判定：Homography 距離 → 斜線投影 → switch_x。"""

from dataclasses import dataclass

from core.tracking_geometry import (
    _bbox_bottom_center,
    _project_onto_track,
    _project_point_to_track_line,
    _transform_point_homography,
)


@dataclass(frozen=True)
class _CameraSwitchContext:
    camera: dict
    runner_states: dict
    runner_id: int | None
    runner_center_x: float | None
    runner_right_x: float | None
    is_last_camera: bool

    @property
    def action(self):
        return '退出ROI' if self.is_last_camera else '切換'

    @property
    def runner(self):
        return self.runner_states.get(self.runner_id)


def _homography_switch(context):
    camera = context.camera
    if camera.get('H_matrix') is None or camera.get('distance_m') is None:
        return None
    if context.runner_id is None or context.runner is None:
        return False, None
    runner = context.runner
    bx1, _by1, bx2, by2 = runner['bbox']
    ground_point = runner.get('smoothed_ground_point') or runner.get('ground_point')
    if ground_point is None:
        image_point = ((bx1 + bx2) / 2, by2)
    else:
        image_point = (float(ground_point[0]), float(ground_point[1]))
        if camera.get('start_mid') is not None and camera.get('track_dir') is not None:
            image_point = _project_point_to_track_line(
                image_point, camera['start_mid'], camera['track_dir'],
            )
    world_point = _transform_point_homography(image_point, camera['H_matrix'])
    if world_point is None:
        return False, None
    distance = max(0.0, float(world_point[0]) - (camera.get('homography_start_x') or 0.0))
    threshold = float(camera['distance_m'])
    if distance < threshold:
        return False, None
    return True, (
        f"  → 觸發{context.action}：Homography距離={distance:.2f}m >= {threshold:.2f}m"
    )


def _track_projection_switch(context):
    camera = context.camera
    if camera.get('track_roi') is None or not camera.get('pixel_span'):
        return None
    if context.runner is None:
        return False, None
    _bx1, _by1, bx2, _by2 = context.runner['bbox']
    ground_x, ground_y = context.runner.get(
        'ground_point', _bbox_bottom_center(context.runner['bbox']),
    )
    reference_x = bx2 if context.is_last_camera else ground_x
    projected = _project_onto_track(
        (reference_x, ground_y),
        camera['start_mid'], camera['track_dir'],
    )
    if projected < camera['pixel_span']:
        return False, None
    return True, (
        f"  → 觸發{context.action}：投影={projected:.0f}px "
        f">= {camera['pixel_span']:.0f}px"
    )


def _legacy_x_switch(context):
    threshold = context.camera.get('switch_x')
    if threshold is None or context.runner_id is None:
        return False, None
    trigger = context.runner_right_x if context.is_last_camera else context.runner_center_x
    if trigger is None or trigger <= threshold:
        return False, None
    reference = 'bx2' if context.is_last_camera else 'center_x'
    return True, f"  → 觸發{context.action}：{reference}={trigger:.0f} > {threshold}"


def _should_switch_camera(context):
    """判斷是否觸發相機切換（或最後一台相機退出 ROI）。

    優先序：Homography 距離 → 斜線投影 pixel_span → switch_x，與 tracker_impl.py 對齊。
    純判斷、不印 log、不修改任何狀態。回傳 (should_switch, log_message)；
    log_message 在不切換時為 None。
    """
    homography_result = _homography_switch(context)
    if homography_result is not None:
        return homography_result
    projection_result = _track_projection_switch(context)
    if projection_result is not None:
        return projection_result
    return _legacy_x_switch(context)


