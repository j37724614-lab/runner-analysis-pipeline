"""底層追蹤+姿態 Pipeline 排程（run_pipeline）與高階分析內容準備（prepare_analysis_context）。"""
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from core.pipeline import PoseScope, TrackedVideoSource, VideoOutput
from core.utils import REPO_ROOT

from .pose_step import PoseEstimationRequest, step2_pose, step3_overlay
from .tracking_step import Step1TrackingRequest, step1_track


@dataclass(frozen=True)
class PipelineOptions:
    """保存底層追蹤與姿態 Pipeline 的執行選項。"""

    output_dir: str | None = None
    gpu: str = "0"
    pose_scope: PoseScope = PoseScope.TWO_D_AND_3D
    tracked_video_source: TrackedVideoSource = TrackedVideoSource.GENERATE
    video_output: VideoOutput = VideoOutput.GENERATE
    timings: list | None = None


@dataclass(frozen=True)
class PipelineRequest:
    """描述一次底層追蹤與姿態 Pipeline 工作。"""

    cameras: list
    extra_config: dict = field(default_factory=dict)
    options: PipelineOptions = field(default_factory=PipelineOptions)


@dataclass(frozen=True)
class PipelineWorkspace:
    """保存 Pipeline 各階段依序產生的主要路徑。"""

    output_dir: str
    tracked_video: str
    pose_directory: str | None = None


def _pipeline_output_directory(request: PipelineRequest) -> str:
    """解析 Pipeline 所有階段共用的輸出目錄。"""
    return request.options.output_dir or request.extra_config.get(
        "output_dir",
        str(REPO_ROOT / "output_cut"),
    )


def _find_existing_tracked_video(output_dir: str) -> str:
    """依輸出標記或目錄內容尋找既有追蹤影片。"""
    marker_path = os.path.join(output_dir, ".last_output_name")
    if os.path.exists(marker_path):
        with open(marker_path, encoding="utf-8") as marker_file:
            output_name = marker_file.read().strip()
        selected_path = os.path.join(output_dir, output_name)
        if os.path.exists(selected_path):
            return selected_path
        raise FileNotFoundError(
            f"上一輪使用骨架串流，未保留可重用的追焦影片：{selected_path}"
        )

    videos = [
        filename
        for filename in (
            os.listdir(output_dir) if os.path.isdir(output_dir) else []
        )
        if filename.endswith(".mp4") and not filename.endswith("_2D.mp4")
    ]
    if not videos:
        raise FileNotFoundError("找不到已追蹤的影片，無法略過 Step 1")
    return os.path.join(output_dir, min(videos))


def _obtain_tracked_video(
    request: PipelineRequest,
    output_dir: str,
    stream_pose: bool = False,
) -> str:
    """依設定產生追蹤影片，或沿用既有輸出。"""
    if request.options.tracked_video_source is TrackedVideoSource.GENERATE:
        return step1_track(
            Step1TrackingRequest(
                camera_configs=request.cameras,
                extra_config=request.extra_config,
                gpu=request.options.gpu,
                output_dir=output_dir,
                timings=request.options.timings,
                stream_pose=stream_pose,
                pose_model_path=request.extra_config.get("pose_model_path"),
                write_tracked_video=not stream_pose,
            )
        )
    print("略過 Step 1，讀取上一次的輸出結果...")
    return _find_existing_tracked_video(output_dir)


def _write_pipeline_config(
    request: PipelineRequest,
    output_dir: str,
) -> None:
    """保存 Step 3 建立圖表底圖需要的相機設定。"""
    pipeline_config = {"cameras": request.cameras}
    pipeline_config.update(request.extra_config)
    config_marker = os.path.join(output_dir, ".config.json")
    with open(config_marker, "w", encoding="utf-8") as config_file:
        json.dump(pipeline_config, config_file, ensure_ascii=False, indent=2)


def _run_pose_pipeline_step(
    request: PipelineRequest,
    workspace: PipelineWorkspace,
) -> str:
    """執行 Pipeline 的姿態估計階段。"""
    return step2_pose(
        PoseEstimationRequest(
            tracked_video_path=workspace.tracked_video,
            output_base_dir=workspace.output_dir,
            gpu=request.options.gpu,
            motion_ag_dir=REPO_ROOT / "MotionAGFormer",
            pose_scope=request.options.pose_scope,
            video_output=request.options.video_output,
            timings=request.options.timings,
            pose_model_path=request.extra_config.get("pose_model_path"),
        )
    )


def _run_overlay_pipeline_step(
    request: PipelineRequest,
    workspace: PipelineWorkspace,
) -> str | None:
    """需要影片輸出時執行角度折線圖合併。"""
    if request.options.video_output is VideoOutput.OMIT:
        return None
    assert workspace.pose_directory is not None
    return step3_overlay(
        workspace.pose_directory,
        Path(workspace.tracked_video).stem,
    )


def run_pipeline(request: PipelineRequest) -> dict:
    """協調追蹤、姿態估計與角度影片輸出。"""
    stream_pose = request.extra_config.get("pose_handoff", "video") == "bounded_stream"
    if stream_pose and (
        request.options.tracked_video_source is not TrackedVideoSource.GENERATE
        or request.options.pose_scope is not PoseScope.TWO_D_ONLY
        or request.options.video_output is not VideoOutput.OMIT
        or request.extra_config.get("tracking_mode", "two_pass") != "two_pass"
    ):
        raise ValueError("bounded_stream requires generated two_pass tracking and 2D-only pose without pose video")
    output_dir = _pipeline_output_directory(request)
    tracked_video = (
        _obtain_tracked_video(request, output_dir, stream_pose=True)
        if stream_pose else _obtain_tracked_video(request, output_dir)
    )
    _write_pipeline_config(request, output_dir)
    workspace = PipelineWorkspace(
        output_dir=output_dir,
        tracked_video=tracked_video,
    )
    workspace = replace(
        workspace,
        pose_directory=(
            os.path.join(output_dir, Path(tracked_video).stem)
            if stream_pose else _run_pose_pipeline_step(request, workspace)
        ),
    )
    overlay_video = _run_overlay_pipeline_step(request, workspace)

    stream_metadata = None
    if stream_pose:
        metadata_path = tracked_video.replace(".mp4", "_stream_metadata.json")
        with open(metadata_path, encoding="utf-8") as metadata_file:
            stream_metadata = json.load(metadata_file)

    result = {
        "output_dir": workspace.pose_directory,
        "tracked_video": workspace.tracked_video,
        "overlay_video": overlay_video,
    }
    if stream_pose:
        result["tracked_video_metadata"] = stream_metadata
    return result


@dataclass(frozen=True)
class AnalysisOptions:
    """保存單次高階分析流程的執行選項。"""

    gpu: str = "0"
    pose_scope: PoseScope = PoseScope.TWO_D_AND_3D
    tracked_video_source: TrackedVideoSource = TrackedVideoSource.GENERATE
    output_dest: str | None = None
    progress_callback: Callable[[int], None] | None = None
    started_at: float = field(default_factory=time.perf_counter, repr=False)


@dataclass
class AnalysisContext:
    """保存所有內部分析階段共用且穩定的設定。"""

    config: dict
    cameras: list
    tracking_cameras: list
    output_dest: str
    options: AnalysisOptions
    started_at: float
    timings: list = field(default_factory=list)
    motion_ag_dir: Path = field(default_factory=lambda: REPO_ROOT / "MotionAGFormer")

    def report_progress(self, percentage: int) -> None:
        """呼叫使用者提供的 callback 回報目前進度。"""
        if self.options.progress_callback:
            self.options.progress_callback(percentage)


@dataclass
class AnalysisState:
    """保存各分析階段依序產生的路徑、資料與統計結果。"""

    tracked_video: str | None
    track_output_dir: str
    track_output_name: str
    metrics_csv: str
    final_pose_dir: str
    angles_csv: str | None = None
    output_video: str | None = None
    step_analysis: dict | None = None
    avg_step_length: float | None = None
    offsets_npz: str | None = None
    keypoints_npz: str | None = None
    foot_npz: str | None = None
    step_overlay_started_at: float | None = None
    tracked_video_metadata: dict | None = None


def prepare_analysis_context(
    analysis_config: dict,
    options: AnalysisOptions,
) -> AnalysisContext:
    """協調分析設定正規化、追蹤設定轉換與輸出目錄準備。"""
    normalized_config = _normalize_analysis_config(analysis_config)
    cameras = normalized_config.get("cameras", [])
    output_dir = _resolve_analysis_output_directory(
        normalized_config,
        options,
    )
    os.makedirs(output_dir, exist_ok=True)
    normalized_config["output_dir"] = output_dir

    return AnalysisContext(
        config=normalized_config,
        cameras=cameras,
        tracking_cameras=_tracking_camera_configs(cameras),
        output_dest=output_dir,
        options=options,
        started_at=options.started_at,
    )


def _normalize_analysis_config(analysis_config: dict) -> dict:
    """複製分析設定，並在未指定裁切尺寸時啟用自動裁切。"""
    normalized_config = dict(analysis_config)
    if (
        "auto_crop" not in normalized_config
        and "crop_width" not in normalized_config
        and "crop_height" not in normalized_config
    ):
        normalized_config["auto_crop"] = True
    return normalized_config


def _tracking_camera_configs(cameras: list) -> list:
    """建立不含 Homography 控制點的追蹤相機設定。"""
    return [_tracking_camera_config(camera) for camera in cameras]


def _tracking_camera_config(camera: dict) -> dict:
    """將完整分析相機設定轉成 tracking 所需設定。"""
    tracking_camera = dict(camera)
    destination_world = tracking_camera.pop("homography_dst_world", None)
    tracking_camera.pop("homography_src_points", None)
    if (
        tracking_camera.get("distance_m") is None
        and destination_world
        and tracking_camera.get("start_line")
        and tracking_camera.get("end_line")
    ):
        world_x_coordinates = [point[0] for point in destination_world]
        distance_span = max(world_x_coordinates) - min(world_x_coordinates)
        if distance_span > 0:
            tracking_camera["distance_m"] = distance_span
    return tracking_camera


def _resolve_analysis_output_directory(
    normalized_config: dict,
    options: AnalysisOptions,
) -> str:
    """依明確選項、相機位置或預設值解析分析輸出目錄。"""
    configured_output = options.output_dest or normalized_config.get("output_dest")
    if configured_output:
        return configured_output

    cameras = normalized_config.get("cameras", [])
    camera_output = next(
        (
            os.path.dirname(os.path.abspath(camera["video_path"]))
            for camera in cameras
            if camera.get("video_path")
        ),
        None,
    )
    if camera_output:
        return camera_output
    return normalized_config.get(
        "output_dir",
        str(REPO_ROOT / "output_cut"),
    )

