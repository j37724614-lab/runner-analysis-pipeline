import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "MotionAGFormer/demo/lib/hrnet/lib/utils/pose_crop.py"
)
SPEC = importlib.util.spec_from_file_location("pose_crop", MODULE_PATH)
pose_crop = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pose_crop
SPEC.loader.exec_module(pose_crop)
TemporalPoseCropStabilizer = pose_crop.TemporalPoseCropStabilizer


def test_center_follows_current_and_size_shrinks_gradually():
    stabilizer = TemporalPoseCropStabilizer()
    previous = [1034.25, 546, 1171.5, 664.5]
    current = [1025.25, 543.75, 1116, 660.75]

    assert stabilizer.update(previous, (1920, 1080)) == previous
    stable = stabilizer.update(current, (1920, 1080))

    assert stable == pytest.approx([1018.5375, 544.65, 1142.0625, 661.65])


def test_context_change_resets_stabilization():
    stabilizer = TemporalPoseCropStabilizer()
    stabilizer.update([100, 100, 200, 300], (1920, 1080), (1, "7", 10))
    current = [120, 100, 180, 300]

    assert stabilizer.update(current, (1920, 1080), (2, "7", 11)) == current


def test_implausible_center_jump_resets_stabilization():
    stabilizer = TemporalPoseCropStabilizer()
    stabilizer.update([100, 100, 200, 300], (1920, 1080))
    current = [900, 100, 1000, 300]

    assert stabilizer.update(current, (1920, 1080)) == current


def test_server_auto_crop_contains_padded_portrait_hrnet_view():
    from core.tracking.debug_output import _auto_crop_side_from_bbox_sizes

    assert _auto_crop_side_from_bbox_sizes(
        widths=[137.25, 90.75],
        heights=[118.5, 117],
    ) == 229
