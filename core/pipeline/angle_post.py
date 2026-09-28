"""角度 CSV 後處理：時間欄位補齊、DP 左右腿身份同步、修正後 2D 關鍵點重算 3D 角度。

``_read_video_frames_per_second`` 本來跟速度分析放在一起，但因為這裡的
``_resolve_angle_frames_per_second`` 也需要它、而 speed_and_legs.py 又需要
本檔案的 DP 左右腿/角度重算函式，兩邊互相依賴會形成循環 import，所以把這個
純函式移來這裡，讓 speed_and_legs.py 只單向依賴本檔案。
"""
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from core import angle_csv_store
from core.pipeline import (
    CSV_IO_ERRORS,
    DEFAULT_VIDEO_FPS,
    KEYPOINT_VIDEO_SIZE_RATIO_THRESHOLD,
    LEG_ANGLE_COLUMN_PAIRS,
    _record_timing,
)

from .motionag_runtime import _import_vis_module, _temporary_motion_agformer_runtime


def _read_video_frames_per_second(
    video_path: str | None,
    fallback: float | None,
) -> float | None:
    """讀取影片 FPS；路徑或影片無效時回傳指定 fallback。"""
    if not video_path:
        return fallback
    video_capture = cv2.VideoCapture(video_path)
    try:
        if not video_capture.isOpened():
            return fallback
        return video_capture.get(cv2.CAP_PROP_FPS) or fallback
    finally:
        video_capture.release()


@dataclass(frozen=True)
class AngleTimeRequest:
    """描述角度 CSV 時間欄位補齊工作。"""

    angle_csv_path: str | None
    video_path: str | None = None
    frames_per_second: float | None = None


def _resolve_angle_frames_per_second(request: AngleTimeRequest) -> float:
    """依明確設定、影片資訊與預設值解析角度資料 FPS。"""
    if request.video_path and os.path.exists(request.video_path):
        video_fps = _read_video_frames_per_second(request.video_path, None)
        if video_fps:
            return video_fps
    if request.frames_per_second and request.frames_per_second > 0:
        return request.frames_per_second
    return DEFAULT_VIDEO_FPS


def _add_angle_time_columns(
    angle_dataframe: pd.DataFrame,
    frames_per_second: float,
) -> pd.DataFrame:
    """回傳補齊時間欄位的新角度資料，不修改輸入。"""
    timed_dataframe = angle_dataframe.copy(deep=True)
    time_seconds = timed_dataframe["frame"].astype(float) / frames_per_second
    for time_column in ("time_s", "time_sec"):
        if time_column in timed_dataframe.columns:
            timed_dataframe[time_column] = time_seconds
        else:
            timed_dataframe.insert(1, time_column, time_seconds)
    leading_columns = [
        column
        for column in ("frame", "time_sec", "time_s")
        if column in timed_dataframe.columns
    ]
    remaining_columns = [
        column
        for column in timed_dataframe.columns
        if column not in leading_columns
    ]
    return timed_dataframe[leading_columns + remaining_columns]


def _add_time_to_angles_csv(request: AngleTimeRequest) -> str | None:
    """依影片 FPS 為角度 CSV 補上相對時間欄位。"""
    if not angle_csv_store.exists(request.angle_csv_path):
        return None
    assert request.angle_csv_path is not None
    resolved_fps = _resolve_angle_frames_per_second(request)

    try:
        angle_dataframe = angle_csv_store.read(request.angle_csv_path)
        if "frame" not in angle_dataframe.columns:
            print(
                "  ▶ 角度 CSV 缺少 frame 欄位，略過時間欄位補齊: "
                f"{request.angle_csv_path}"
            )
            return None
        timed_dataframe = _add_angle_time_columns(angle_dataframe, resolved_fps)
        angle_csv_store.write(request.angle_csv_path, timed_dataframe)
        print(
            f"  ▶ 已補齊角度時間欄位: {request.angle_csv_path} "
            f"(fps={resolved_fps:.3f})"
        )
        return request.angle_csv_path
    except CSV_IO_ERRORS as error:
        print(f"  ▶ 補齊角度時間欄位失敗: {error}")
        return None


@dataclass(frozen=True)
class LegSwapMaskRequest:
    """描述 DP 左右腿交換紀錄的輸出內容。"""

    swapped_mask: np.ndarray | None
    output_dir: str | None = None
    pre_dp_swapped_mask: np.ndarray | None = None
    anchor_dp_swapped_mask: np.ndarray | None = None


def _padded_boolean_mask(mask, target_length: int) -> np.ndarray:
    """將布林遮罩裁切或補齊至指定長度。"""
    values = np.asarray(mask, dtype=bool).reshape(-1)
    return np.pad(
        values[:target_length],
        (0, max(0, target_length - len(values))),
    )


def _leg_swap_mask_dataframe(request: LegSwapMaskRequest) -> pd.DataFrame:
    """建立 DP 左右腿交換紀錄，不進行檔案寫入。"""
    assert request.swapped_mask is not None
    swapped = np.asarray(request.swapped_mask, dtype=bool).reshape(-1)
    columns = {
        "frame": np.arange(len(swapped), dtype=int),
        "dp_leg_swapped": swapped,
    }
    if request.pre_dp_swapped_mask is not None:
        columns["pre_dp_leg_swapped"] = _padded_boolean_mask(
            request.pre_dp_swapped_mask,
            len(swapped),
        )
    if request.anchor_dp_swapped_mask is not None:
        columns["anchor_dp_leg_swapped"] = _padded_boolean_mask(
            request.anchor_dp_swapped_mask,
            len(swapped),
        )
    return pd.DataFrame(columns)


def _write_dp_leg_swap_mask(request: LegSwapMaskRequest):
    """保存 DP 左右腿交換結果及可選的分階段遮罩。"""
    if request.swapped_mask is None or not request.output_dir:
        return None

    try:
        swapped = np.asarray(request.swapped_mask, dtype=bool).reshape(-1)
        swap_csv = os.path.join(
            request.output_dir,
            "dp_leg_identity_swaps.csv",
        )
        _leg_swap_mask_dataframe(request).to_csv(swap_csv, index=False)
        print(f"  ▶ DP 左右腿交換紀錄: {swap_csv}")
        return {
            "swap_csv": swap_csv,
            "swapped_frames": int(swapped.sum()),
        }
    except (OSError, ValueError, TypeError) as error:
        print(f"  ▶ 寫出 DP 左右腿交換紀錄失敗: {error}")
        return None


@dataclass(frozen=True)
class AngleAlignmentResult:
    """保存角度欄位交換後的資料與套用摘要。"""

    dataframe: pd.DataFrame
    swapped_frames: int
    applied_pairs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class AngleCsvAlignmentRequest:
    """描述既有角度 CSV 與腿部身份遮罩的同步工作。"""

    angle_csv_path: str | None
    swapped_mask: np.ndarray | None
    output_dir: str | None = None


def _align_angle_dataframe_to_leg_identity(
    angle_dataframe: pd.DataFrame,
    swapped_mask,
) -> AngleAlignmentResult:
    """依左右腿交換遮罩轉換角度資料，不讀寫檔案或修改輸入。"""
    aligned_dataframe = angle_dataframe.copy(deep=True)
    swapped = np.asarray(swapped_mask, dtype=bool).reshape(-1)
    aligned_frame_count = min(len(aligned_dataframe), len(swapped))
    active_swap_mask = swapped[:aligned_frame_count]
    applied_pairs: list[tuple[str, str]] = []

    if aligned_frame_count and np.any(active_swap_mask):
        for left_column, right_column in LEG_ANGLE_COLUMN_PAIRS:
            if (
                left_column not in aligned_dataframe.columns
                or right_column not in aligned_dataframe.columns
            ):
                continue
            left_index = aligned_dataframe.columns.get_loc(left_column)
            right_index = aligned_dataframe.columns.get_loc(right_column)
            left_values = aligned_dataframe.iloc[
                :aligned_frame_count,
                left_index,
            ].copy()
            right_values = aligned_dataframe.iloc[
                :aligned_frame_count,
                right_index,
            ].copy()
            aligned_dataframe.iloc[:aligned_frame_count, left_index] = np.where(
                active_swap_mask,
                right_values,
                left_values,
            )
            aligned_dataframe.iloc[:aligned_frame_count, right_index] = np.where(
                active_swap_mask,
                left_values,
                right_values,
            )
            applied_pairs.append((left_column, right_column))

    return AngleAlignmentResult(
        dataframe=aligned_dataframe,
        swapped_frames=int(active_swap_mask.sum()),
        applied_pairs=tuple(applied_pairs),
    )


def _align_angle_csv_to_leg_identity(
    request: AngleCsvAlignmentRequest,
) -> dict[str, object] | None:
    """在 DP 交換 2D 腿部身份的影格同步交換左右腿角度欄位。

    Preferred flow is to run MotionAGFormer 3D + angle computation only after
    apply_anchor_leg_correction() has already rewritten input_2D/keypoints.npz.
    In that flow this fallback is not needed because angles are computed from
    the corrected 2D identities. Keep it available for older outputs where only
    a pre-DP angles.csv exists.
    """
    if (
        not request.angle_csv_path
        or request.swapped_mask is None
        or not angle_csv_store.exists(request.angle_csv_path)
    ):
        _write_dp_leg_swap_mask(
            LegSwapMaskRequest(request.swapped_mask, request.output_dir)
        )
        return None

    try:
        angle_dataframe = angle_csv_store.read(request.angle_csv_path)
        alignment = _align_angle_dataframe_to_leg_identity(
            angle_dataframe,
            request.swapped_mask,
        )
        if alignment.swapped_frames == 0:
            _write_dp_leg_swap_mask(
                LegSwapMaskRequest(request.swapped_mask, request.output_dir)
            )
            return None
        if not alignment.applied_pairs:
            return None

        return _persist_angle_alignment(request, alignment)
    except CSV_IO_ERRORS as error:
        print(f"  ▶ 同步 DP 左右腿身份到角度 CSV 失敗: {error}")
        return None


def _persist_angle_alignment(request, alignment):
    angle_csv_store.write(request.angle_csv_path, alignment.dataframe)
    swap_info = _write_dp_leg_swap_mask(
        LegSwapMaskRequest(request.swapped_mask, request.output_dir)
    )
    swap_csv = swap_info.get("swap_csv") if swap_info else None
    print(
        "  ▶ 已依 DP 左右腿身份修正同步角度 CSV: "
        f"{request.angle_csv_path} "
        f"(swapped_frames={alignment.swapped_frames}, "
        f"pairs={list(alignment.applied_pairs)})"
    )
    return {
        "angle_csv": request.angle_csv_path,
        "swap_csv": swap_csv,
        "swapped_frames": alignment.swapped_frames,
        "applied_pairs": list(alignment.applied_pairs),
    }


# -----------------------------------------------------------------------
# 延遲 import：只在真正需要時才載入 GPU-heavy 的 3D 重建函式庫
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectedPose3DRequest:
    """描述使用修正後 2D 關鍵點重新產生 3D 角度的工作。"""

    tracked_video_path: str | None
    pose_output_dir: str
    analysis_output_dir: str
    gpu: str
    motion_ag_dir: Path
    timings: list | None = None
    video_metadata: dict | None = None


def _video_dimensions(video_path: str) -> tuple[float, float]:
    """讀取影片寬高；影片無法開啟時回傳零值。"""
    video_capture = cv2.VideoCapture(video_path)
    try:
        if not video_capture.isOpened():
            return 0.0, 0.0
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0
        return float(width), float(height)
    finally:
        video_capture.release()


def _archived_tracked_video_path(pose_output_dir: str) -> str | None:
    """尋找關鍵點封存目錄中的第一相機追蹤影片。"""
    pointer_path = (
        Path(pose_output_dir) / "input_2D" / "keypoints_raw_archive_dir.txt"
    )
    if not pointer_path.exists():
        return None
    try:
        archive_dir = pointer_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    archived_video = Path(archive_dir) / "cam1_tracked.mp4"
    return str(archived_video) if archived_video.exists() else None


def _keypoint_coordinate_extent(keypoints_npz: str) -> tuple[float, float]:
    """讀取關鍵點資料並回傳 X、Y 座標最大值。"""
    reconstruction = np.load(keypoints_npz, allow_pickle=True)["reconstruction"]
    keypoint_coordinates = (
        reconstruction[0, :, :, :2]
        if reconstruction.ndim == 4
        else reconstruction[:, :, :2]
    )
    return (
        float(np.nanmax(keypoint_coordinates[..., 0])),
        float(np.nanmax(keypoint_coordinates[..., 1])),
    )


def _coordinate_system_may_differ(
    keypoint_extent: tuple[float, float],
    video_dimensions: tuple[float, float],
) -> bool:
    """判斷關鍵點與影片是否可能使用不同尺寸的座標系。"""
    max_keypoint_x, max_keypoint_y = keypoint_extent
    video_width, video_height = video_dimensions
    return (
        video_width > 0
        and video_height > 0
        and (
            max_keypoint_x < video_width * KEYPOINT_VIDEO_SIZE_RATIO_THRESHOLD
            or max_keypoint_y < video_height * KEYPOINT_VIDEO_SIZE_RATIO_THRESHOLD
        )
    )


def _select_compatible_3d_video(
    request: CorrectedPose3DRequest,
    keypoints_npz: str,
) -> str:
    """選擇與修正後關鍵點座標系相容的 3D 重算影片。"""
    assert request.tracked_video_path is not None
    if request.video_metadata is not None:
        return request.tracked_video_path
    selected_video = request.tracked_video_path
    try:
        keypoint_extent = _keypoint_coordinate_extent(keypoints_npz)
        selected_dimensions = _video_dimensions(selected_video)
        if not _coordinate_system_may_differ(keypoint_extent, selected_dimensions):
            return selected_video

        archived_video = _archived_tracked_video_path(request.pose_output_dir)
        if archived_video:
            archived_dimensions = _video_dimensions(archived_video)
            if all(dimension > 0 for dimension in archived_dimensions):
                print(
                    "  ▶ 偵測到 3D 重算影片尺寸與 keypoints 座標系不一致，"
                    f"改用 archive tracked video: {archived_video} "
                    f"({selected_dimensions[0]:.0f}x{selected_dimensions[1]:.0f} -> "
                    f"{archived_dimensions[0]:.0f}x{archived_dimensions[1]:.0f})"
                )
                return archived_video

        print(
            "  ▶ 警告：3D 重算影片尺寸可能與 keypoints 座標系不一致，"
            f"video={selected_dimensions[0]:.0f}x{selected_dimensions[1]:.0f}, "
            f"keypoints max=({keypoint_extent[0]:.1f},{keypoint_extent[1]:.1f})"
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        cv2.error,
    ) as error:
        print(f"  ▶ 檢查 3D 重算影片尺寸失敗，繼續使用原影片: {error}")
    return selected_video


def _generate_corrected_3d_angles(
    request: CorrectedPose3DRequest,
    video_path: str,
) -> None:
    """在隔離環境中執行 MotionAGFormer 3D 姿態與角度重算。"""
    with _temporary_motion_agformer_runtime(request.motion_ag_dir, request.gpu):
        started_at = time.perf_counter()
        visualization_module = _import_vis_module(request.motion_ag_dir)
        visualization_module.get_pose3D(
            video_path,
            request.pose_output_dir,
            skip_video=True,
            video_metadata=request.video_metadata,
        )
        _record_timing(
            request.timings,
            "Analysis/rerun_3d_angles_after_leg_dp",
            started_at,
        )


def _publish_corrected_angle_csv(request: CorrectedPose3DRequest) -> str | None:
    """將重算產生的角度 CSV 發佈到分析輸出目錄。"""
    source_angle_csv = os.path.join(
        request.pose_output_dir,
        "pred_3D",
        "angles",
        f"{Path(request.pose_output_dir).name}_angles.csv",
    )
    if not angle_csv_store.exists(source_angle_csv):
        print(f"  ▶ 重算後角度 CSV 不存在: {source_angle_csv}")
        return None

    output_angle_csv = os.path.join(request.analysis_output_dir, "angles.csv")
    try:
        angle_csv_store.publish(source_angle_csv, output_angle_csv)
        print(f"  ▶ 已用 DP 修正後 2D keypoints 重算 3D 角度: {output_angle_csv}")
        return output_angle_csv
    except OSError as error:
        print(f"  ▶ 複製重算後角度 CSV 失敗: {error}")
        return source_angle_csv


def _rerun_3d_angles_from_corrected_2d(
    request: CorrectedPose3DRequest,
) -> str | None:
    """協調修正後 2D 關鍵點的 3D 角度重算與結果發佈。"""
    if (
        not request.tracked_video_path
        or not request.pose_output_dir
        or not request.analysis_output_dir
    ):
        return None

    keypoints_npz = os.path.join(
        request.pose_output_dir,
        "input_2D",
        "keypoints.npz",
    )
    if not os.path.exists(keypoints_npz):
        print(f"  ▶ 無法重算 3D 角度，找不到修正後 keypoints: {keypoints_npz}")
        return None

    selected_video = _select_compatible_3d_video(request, keypoints_npz)
    _generate_corrected_3d_angles(request, selected_video)
    return _publish_corrected_angle_csv(request)


# -----------------------------------------------------------------------
# 各步驟實作函式
# -----------------------------------------------------------------------

