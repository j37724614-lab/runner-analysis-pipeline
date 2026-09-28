"""暫時切換行程狀態以載入並執行 MotionAGFormer 的工具。

被 core/pipeline 內的 angle_post.py（DP 修正後重算 3D 角度）與 pose_step.py
（Step 2 姿態估計）共用，core/pose_stream.py 也會在執行期動態 import 這兩個函式。
"""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from core.process_runtime import (
    PROCESS_STATE_LOCK as _PROCESS_STATE_LOCK,
)
from core.process_runtime import (
    temporary_environment_variable as _temporary_environment_variable,
)


@contextmanager
def _temporary_motion_agformer_runtime(motion_ag_dir: Path, gpu: str):
    """暫時切換 MotionAGFormer 所需的程序狀態，完成後完整還原。"""
    with _PROCESS_STATE_LOCK:
        original_argv = sys.argv[:]
        original_cwd = os.getcwd()
        original_sys_path = sys.path[:]
        demo_dir = str(motion_ag_dir / "demo")
        motion_ag_path = str(motion_ag_dir)

        try:
            sys.argv = [sys.argv[0]]
            os.chdir(motion_ag_path)
            sys.path[:] = [
                demo_dir,
                motion_ag_path,
                *(
                    path
                    for path in original_sys_path
                    if path not in {demo_dir, motion_ag_path}
                ),
            ]
            with _temporary_environment_variable("CUDA_VISIBLE_DEVICES", gpu):
                yield
        finally:
            os.chdir(original_cwd)
            sys.argv = original_argv
            sys.path[:] = original_sys_path




def _import_vis_module(motion_ag_dir: Path):
    """動態載入 MotionAGFormer 的 vis.py；呼叫端負責暫時程序環境。"""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vis", str(motion_ag_dir / "demo" / "vis.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"無法載入 MotionAGFormer vis.py: {motion_ag_dir}")
    vis_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vis_module)
    return vis_module

