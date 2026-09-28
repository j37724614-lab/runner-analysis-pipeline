"""Step 2/Step 3：HRNet 2D 姿態估計，以及 2D 骨架與角度折線圖合併。"""
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from core.pipeline import (
    OUTPUT_SEPARATOR,
    STALE_POSE_OUTPUT_DIRECTORIES,
    PoseScope,
    VideoOutput,
    _record_timing,
)
from core.visualization import AngleOverlayConfig, add_angle_overlay

from .motionag_runtime import _import_vis_module, _temporary_motion_agformer_runtime


@dataclass(frozen=True)
class PoseEstimationRequest:
    """描述一次姿態估計工作及其輸出內容。"""

    tracked_video_path: str
    output_base_dir: str
    gpu: str
    motion_ag_dir: Path
    pose_scope: PoseScope = PoseScope.TWO_D_AND_3D
    video_output: VideoOutput = VideoOutput.GENERATE
    timings: list | None = None
    pose_model_path: str | None = None


@dataclass(frozen=True)
class PoseWorkspace:
    """保存姿態估計執行時使用的已解析路徑。"""

    video_path: str
    output_dir: str
    result_dir: str
    bbox_csv_path: str | None


def _prepare_pose_workspace(request: PoseEstimationRequest) -> PoseWorkspace:
    """建立輸出目錄，並解析姿態估計需要的輸入路徑。"""
    video_stem = Path(request.tracked_video_path).stem
    result_dir = os.path.join(request.output_base_dir, video_stem) + "/"
    os.makedirs(result_dir, exist_ok=True)

    video_path = os.path.abspath(request.tracked_video_path)
    bbox_csv_candidate = video_path.replace(
        ".mp4",
        "_bbox_map.csv",
    )
    return PoseWorkspace(
        video_path=video_path,
        output_dir=os.path.abspath(result_dir),
        result_dir=result_dir,
        bbox_csv_path=(
            bbox_csv_candidate if os.path.exists(bbox_csv_candidate) else None
        ),
    )


def _clear_stale_pose_outputs(workspace: PoseWorkspace) -> None:
    """清除舊影格，避免新舊輸出混合而污染生成影片。"""
    for folder_name in STALE_POSE_OUTPUT_DIRECTORIES:
        folder_path = os.path.join(workspace.output_dir, folder_name)
        if os.path.exists(folder_path):
            print(
                f"  [Step 2] 偵測到舊的 {folder_name} 資料夾，進行清理以避免新舊影格污染..."
            )
            try:
                shutil.rmtree(folder_path)
            except OSError as error:
                print(f"  ⚠️  [Step 2] 清理 {folder_name} 失敗: {error}")


def _print_pose_estimation_plan(
    request: PoseEstimationRequest,
    workspace: PoseWorkspace,
) -> None:
    """顯示本次姿態估計的範圍與輸出位置。"""
    includes_3d = request.pose_scope is PoseScope.TWO_D_AND_3D
    analysis_scope = "2D + 3D + 角度" if includes_3d else "2D only"
    print(OUTPUT_SEPARATOR)
    print(f"Step 2 — 姿態估計（{analysis_scope}）")
    print(f"  影片: {request.tracked_video_path}")
    print(f"  輸出: {workspace.result_dir}")
    print(OUTPUT_SEPARATOR)


def _execute_pose_estimation(
    request: PoseEstimationRequest,
    workspace: PoseWorkspace,
) -> None:
    """在隔離的 MotionAGFormer 執行環境中完成姿態估計。"""
    includes_3d = request.pose_scope is PoseScope.TWO_D_AND_3D
    generates_video = request.video_output is VideoOutput.GENERATE

    # 暫時改變程序環境，離開區塊後會完整還原。
    with _temporary_motion_agformer_runtime(request.motion_ag_dir, request.gpu):
        run_pose_estimation = _import_vis_module(
            request.motion_ag_dir
        ).run_pose_estimation
        pose_started_at = time.perf_counter()
        run_pose_estimation(
            video_path=workspace.video_path,
            output_dir=workspace.output_dir,
            only_2d=not includes_3d,
            gpu=request.gpu,
            bbox_csv=workspace.bbox_csv_path,
            skip_video=not generates_video,
            model_path=request.pose_model_path,
        )
        _record_timing(
            request.timings,
            "Step2/pose_estimation_total",
            pose_started_at,
            video_path=workspace.video_path,
            output_dir=workspace.output_dir,
            only_2d=not includes_3d,
            skip_video=not generates_video,
        )


def step2_pose(request: PoseEstimationRequest) -> str:
    """協調工作區準備、舊輸出清理與姿態估計。"""
    workspace = _prepare_pose_workspace(request)
    _clear_stale_pose_outputs(workspace)
    _print_pose_estimation_plan(request, workspace)
    _execute_pose_estimation(request, workspace)

    print(f"\nStep 2 完成，骨架與角度數據輸出至：{workspace.result_dir}\n")
    return workspace.result_dir


@dataclass(frozen=True)
class OverlayWorkspace:
    """保存 Step 3 角度疊圖使用的輸入與輸出路徑。"""

    pose_video: str
    angle_csv: str
    output_video: str
    frame_map: str | None
    config_marker: str


def _prepare_overlay_workspace(
    pose_output_dir: str,
    video_stem: str,
) -> OverlayWorkspace:
    """解析角度疊圖階段使用的所有路徑。"""
    pose_video = os.path.join(pose_output_dir, f"{video_stem}_2D.mp4")
    angle_csv = os.path.join(
        pose_output_dir,
        "pred_3D",
        "angles",
        f"{video_stem}_angles.csv",
    )
    pipeline_output_dir = os.path.dirname(
        os.path.normpath(pose_output_dir)
    )
    frame_map_candidate = os.path.join(
        pipeline_output_dir,
        f"{video_stem}_frame_map.csv",
    )
    return OverlayWorkspace(
        pose_video=pose_video,
        angle_csv=angle_csv,
        output_video=os.path.join(
            pose_output_dir,
            f"{video_stem}_2D_angles.mp4",
        ),
        frame_map=(
            frame_map_candidate if os.path.exists(frame_map_candidate) else None
        ),
        config_marker=os.path.join(pipeline_output_dir, ".config.json"),
    )


def _overlay_inputs_exist(workspace: OverlayWorkspace) -> bool:
    """確認角度疊圖必要輸入存在，並顯示略過原因。"""
    if not os.path.exists(workspace.angle_csv):
        print("  ⚠️  [Step 3] 角度 CSV 不存在，略過 Step 3 合併 (可能是 only_2d=True)")
        return False
    if not os.path.exists(workspace.pose_video):
        print(f"  ⚠️  [Step 3] 2D 骨架影片不存在: {workspace.pose_video}，略過")
        return False
    return True


def _load_overlay_main_video_paths(config_marker: str) -> list[str]:
    """從 Pipeline 暫存設定讀取原始相機影片路徑。"""
    if not os.path.exists(config_marker):
        return []
    try:
        with open(config_marker, encoding="utf-8") as config_file:
            saved_config = json.load(config_file)
        return [
            camera["video_path"]
            for camera in saved_config.get("cameras", [])
            if camera.get("video_path")
        ]
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        AttributeError,
        TypeError,
    ) as error:
        print(f"  [Step 3] 無法讀取 config 暫存: {error}")
        return []


def _render_angle_overlay(
    workspace: OverlayWorkspace,
    main_video_paths: list[str],
) -> None:
    """以固定顯示設定產生 2D 骨架與角度折線圖合併影片。"""
    add_angle_overlay(
        workspace.pose_video,
        workspace.angle_csv,
        workspace.output_video,
        AngleOverlayConfig(
            main_videos=main_video_paths,
            frame_map_path=workspace.frame_map,
        ),
    )


def step3_overlay(pose_output_dir: str, video_stem: str) -> str | None:
    """協調 2D 骨架影片與角度折線圖合併。"""
    workspace = _prepare_overlay_workspace(pose_output_dir, video_stem)

    print(OUTPUT_SEPARATOR)
    print("Step 3 — 2D 影片 + 角度折線圖合併")
    print(OUTPUT_SEPARATOR)

    if not _overlay_inputs_exist(workspace):
        return None
    _render_angle_overlay(
        workspace,
        _load_overlay_main_video_paths(workspace.config_marker),
    )
    return workspace.output_video


# -----------------------------------------------------------------------
# CLI/Python Orchestration API
# -----------------------------------------------------------------------

