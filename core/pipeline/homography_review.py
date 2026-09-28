"""Homography 俯視回顧影片的輸出協調：單鏡頭校正回顧、示意影片、完整賽事路徑影片。"""
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from core.pipeline import HOMOGRAPHY_CONTROL_POINT_COUNT, VideoOutput
from core.utils import REPO_ROOT


@dataclass(frozen=True)
class HomographyReviewOptions:
    """保存俯視回顧影片的輸出路徑與呈現設定。"""

    output_dest: str
    timeline_video: str | None = None
    camera_schematic_output: VideoOutput = VideoOutput.OMIT
    camera_pixels_per_meter: float = 75.0
    padding_pixels: int = 60
    trial_pixels_per_meter: float = 30.0


@dataclass(frozen=True)
class HomographyReviewRequest:
    """描述一組相機與完整賽事的俯視回顧輸出。"""

    cameras: list
    steps_csv: str | None
    options: HomographyReviewOptions


@dataclass(frozen=True)
class CameraHomographyInputs:
    """保存單鏡頭 Homography 回顧共用的輸入。"""

    video_path: str
    camera_index: int
    points_path: Path
    control_count: int
    steps_csv: str


@dataclass(frozen=True)
class TrialHomographyInputs:
    """保存完整賽事 Homography 回顧需要的輸入。"""

    calibrations: list[dict[str, object]]
    steps_csv: str


@dataclass(frozen=True)
class HomographyRenderJob:
    """描述一項外部 Homography 影片渲染工作。"""

    tool_path: Path
    arguments: tuple[str, ...]
    output_path: Path
    failure_message: str
    generated_path: Path | None = None




def _complete_homography_calibration(camera: dict, camera_index: int):
    """將一台相機的完整六點校正轉成可序列化資料。"""
    image_points = camera.get("homography_src_points")
    world_points = camera.get("homography_dst_world")
    if not isinstance(image_points, list) or not isinstance(world_points, list):
        return None
    if (
        len(image_points) != HOMOGRAPHY_CONTROL_POINT_COUNT
        or len(world_points) != HOMOGRAPHY_CONTROL_POINT_COUNT
    ):
        return None
    return {
        "camera_index": camera_index,
        "image_points": [[float(point[0]), float(point[1])] for point in image_points],
        "world_points": [[float(point[0]), float(point[1])] for point in world_points],
    }


def _write_homography_control_points(
    calibration: dict,
    output_dest: str,
) -> tuple[Path, list[dict[str, float | int]]]:
    """寫出單鏡頭校正工具需要的控制點 JSON。"""
    camera_index = int(calibration["camera_index"])
    points = [
        {
            "id": index + 1,
            "x": image_point[0],
            "y": image_point[1],
            "world_x_m": world_point[0],
            "world_y_m": world_point[1],
        }
        for index, (image_point, world_point) in enumerate(
            zip(calibration["image_points"], calibration["world_points"])
        )
    ]

    points_path = (
        Path(output_dest) / f"topdown_review_cam{camera_index + 1}_controls.json"
    )
    points_path.write_text(
        json.dumps({"points": points}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return points_path, points


def _run_homography_render(job: HomographyRenderJob) -> str | None:
    """執行外部渲染工具，驗證並發佈產生的影片。"""
    try:
        subprocess.run(
            (
                sys.executable,
                str(job.tool_path),
                *job.arguments,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        generated_path = job.generated_path or job.output_path
        if not generated_path.exists():
            return None
        if generated_path != job.output_path:
            os.replace(generated_path, job.output_path)
        return str(job.output_path)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"  ▶ {job.failure_message}: {error}")
        return None


def _export_rectified_camera_review(
    inputs: CameraHomographyInputs,
    options: HomographyReviewOptions,
) -> str | None:
    """輸出最後一台相機的透視校正回顧影片。"""
    output_dir = Path(options.output_dest)
    return _run_homography_render(
        HomographyRenderJob(
            tool_path=(
                REPO_ROOT
                / "scripts"
                / "tools"
                / "rectify_video_from_cone_points.py"
            ),
            arguments=(
                "--video",
                inputs.video_path,
                "--points-json",
                str(inputs.points_path),
                "--output-dir",
                options.output_dest,
                "--control-count",
                str(inputs.control_count),
                "--px-per-meter",
                "100",
                "--padding-px",
                "40",
                "--max-frames",
                "0",
                "--step-events-csv",
                inputs.steps_csv,
                "--camera-index",
                str(inputs.camera_index),
            ),
            generated_path=(
                output_dir / "homography_rectified_preview.mp4"
            ),
            output_path=(
                output_dir
                / f"cam{inputs.camera_index + 1}_topdown_review.mp4"
            ),
            failure_message=(
                f"Cam {inputs.camera_index + 1} 俯視回顧影片輸出失敗"
            ),
        )
    )


def _export_schematic_camera_review(
    inputs: CameraHomographyInputs,
    options: HomographyReviewOptions,
) -> str | None:
    """輸出單鏡頭等比例跑道示意影片。"""
    metrics_path = Path(options.output_dest) / "metrics.csv"
    schematic_output = (
        Path(options.output_dest)
        / f"cam{inputs.camera_index + 1}_topdown_schematic_review.mp4"
    )
    if not metrics_path.exists():
        print(
            f"  ▶ 略過 Cam {inputs.camera_index + 1} "
            f"俯視示意影片：找不到 {metrics_path}"
        )
        return None
    return _run_homography_render(
        HomographyRenderJob(
            tool_path=(
                REPO_ROOT
                / "scripts"
                / "tools"
                / "render_schematic_topdown_review.py"
            ),
            arguments=(
                "--video",
                inputs.video_path,
                "--points-json",
                str(inputs.points_path),
                "--metrics-csv",
                str(metrics_path),
                "--step-events-csv",
                inputs.steps_csv,
                "--output",
                str(schematic_output),
                "--camera-index",
                str(inputs.camera_index),
                "--px-per-meter",
                str(options.camera_pixels_per_meter),
                "--padding-px",
                str(options.padding_pixels),
            ),
            output_path=schematic_output,
            failure_message=(
                f"Cam {inputs.camera_index + 1} 俯視示意影片輸出失敗"
            ),
        )
    )


def _collect_trial_calibrations(cameras: list) -> list[dict[str, object]]:
    """只有全部相機皆有完整六點校正時才回傳校正集合。"""
    complete_calibrations: list[dict[str, object]] = []
    for index, current_camera in enumerate(cameras):
        calibration = _complete_homography_calibration(current_camera, index)
        if calibration is None:
            return []
        complete_calibrations.append(calibration)
    return complete_calibrations


def _prepare_full_trial_render_job(
    inputs: TrialHomographyInputs,
    options: HomographyReviewOptions,
) -> HomographyRenderJob | None:
    """驗證完整賽事輸入、寫出校正設定並建立渲染工作。"""
    metrics_path = Path(options.output_dest) / "metrics.csv"
    timeline_path = Path(options.timeline_video) if options.timeline_video else None
    if (
        not inputs.calibrations
        or timeline_path is None
        or not timeline_path.exists()
        or not metrics_path.exists()
    ):
        return None

    calibrations_path = Path(options.output_dest) / "trial_topdown_calibrations.json"
    calibrations_path.write_text(
        json.dumps(
            {"cameras": inputs.calibrations},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output_path = Path(options.output_dest) / "trial_topdown_review.mp4"
    return HomographyRenderJob(
        tool_path=(
            REPO_ROOT
            / "scripts"
            / "tools"
            / "render_schematic_topdown_review.py"
        ),
        arguments=(
            "--video",
            str(timeline_path),
            "--calibrations-json",
            str(calibrations_path),
            "--metrics-csv",
            str(metrics_path),
            "--step-events-csv",
            inputs.steps_csv,
            "--output",
            str(output_path),
            "--px-per-meter",
            str(options.trial_pixels_per_meter),
            "--padding-px",
            str(options.padding_pixels),
        ),
        output_path=output_path,
        failure_message="完整賽事俯視路徑影片輸出失敗",
    )


def _export_full_trial_topdown_review(
    inputs: TrialHomographyInputs,
    options: HomographyReviewOptions,
) -> str | None:
    """將所有鏡頭校正投影到同一時間與距離軸的示意影片。"""
    render_job = _prepare_full_trial_render_job(inputs, options)
    if render_job is None:
        print("  ▶ 略過完整賽事俯視路徑：所有鏡頭皆需 6 點校正、metrics 與完整影片")
        return None
    return _run_homography_render(render_job)


def _export_homography_review_videos(
    request: HomographyReviewRequest,
) -> list[str]:
    """協調單鏡頭與完整賽事的 Homography 回顧影片輸出。"""
    if (
        not request.steps_csv
        or not os.path.exists(request.steps_csv)
        or not request.cameras
    ):
        return []

    camera_inputs = _last_camera_homography_inputs(request)
    if camera_inputs is None:
        return []
    outputs = _export_camera_homography_reviews(camera_inputs, request.options)
    trial_output = _export_full_trial_topdown_review(
        TrialHomographyInputs(
            calibrations=_collect_trial_calibrations(request.cameras),
            steps_csv=request.steps_csv,
        ),
        request.options,
    )
    if trial_output:
        outputs.append(trial_output)
    return outputs


def _last_camera_homography_inputs(request):
    camera_index = len(request.cameras) - 1
    camera = request.cameras[camera_index]
    video_path = camera.get("video_path")
    calibration = _complete_homography_calibration(camera, camera_index)
    if not video_path or not os.path.exists(video_path) or calibration is None:
        return None
    points_path, points = _write_homography_control_points(
        calibration, request.options.output_dest,
    )
    return CameraHomographyInputs(
        video_path=video_path, camera_index=camera_index,
        points_path=points_path, control_count=len(points),
        steps_csv=request.steps_csv,
    )


def _export_camera_homography_reviews(camera_inputs, options):
    outputs = []
    rectified_output = _export_rectified_camera_review(
        camera_inputs, options,
    )
    if rectified_output:
        outputs.append(rectified_output)

    if options.camera_schematic_output is VideoOutput.GENERATE:
        schematic_output = _export_schematic_camera_review(
            camera_inputs, options,
        )
        if schematic_output:
            outputs.append(schematic_output)

    return outputs

