"""隔離單次 tracking 執行需要的全域設定。"""

from contextlib import contextmanager
from dataclasses import dataclass

from core import tracking
from core.process_runtime import (
    PROCESS_STATE_LOCK,
    temporary_environment_variable,
)

TRACKING_CONFIG_FIELDS = {
    "output_dir": ("OUTPUT_DIR", str),
    "crop_width": ("CROP_WIDTH", int),
    "crop_height": ("CROP_HEIGHT", int),
    "auto_crop": ("AUTO_CROP", bool),
    "show_overlay": ("SHOW_OVERLAY", bool),
    "draw_bbox_overlay": ("DRAW_BBOX_OVERLAY", bool),
    "movement_threshold": ("MOVEMENT_THRESHOLD", int),
    "min_movement_frames": ("MIN_MOVEMENT_FRAMES", int),
    "stationary_decay": ("STATIONARY_DECAY", int),
    "max_person_memory": ("MAX_PERSON_MEMORY", int),
    "tracking_mode": ("TRACKING_MODE", str),
    "prescan_enabled": ("PRESCAN_ENABLED", bool),
    "prescan_engine_path": ("PRESCAN_ENGINE_PATH", str),
    "prescan_stride": ("PRESCAN_STRIDE", int),
    "prescan_imgsz": ("PRESCAN_IMGSZ", int),
    "prescan_conf": ("PRESCAN_CONF", float),
    "prescan_iou": ("PRESCAN_IOU", float),
    "prescan_buffer_sec": ("PRESCAN_BUFFER_SEC", float),
    "prescan_max_gap_sec": ("PRESCAN_MAX_GAP_SEC", float),
    "prescan_use_grab": ("PRESCAN_USE_GRAB", bool),
}


@dataclass(frozen=True)
class TrackingRuntimeOptions:
    """保存單次 tracking 執行時暫時套用的程序設定。"""

    config: dict
    gpu: str
    output_directory: str


@contextmanager
def temporary_tracking_runtime(options: TrackingRuntimeOptions):
    """只在單次追蹤期間套用 tracking 模組設定與 GPU 選擇。"""
    with PROCESS_STATE_LOCK:
        tracked_attributes = {
            attribute_name
            for attribute_name, _converter in TRACKING_CONFIG_FIELDS.values()
        }
        original_values = {
            attribute_name: getattr(tracking, attribute_name)
            for attribute_name in tracked_attributes
        }

        try:
            for config_name, (
                attribute_name,
                converter,
            ) in TRACKING_CONFIG_FIELDS.items():
                if config_name in options.config:
                    setattr(
                        tracking,
                        attribute_name,
                        converter(options.config[config_name]),
                    )
            tracking.OUTPUT_DIR = options.output_directory
            with temporary_environment_variable(
                "CUDA_VISIBLE_DEVICES",
                options.gpu,
            ):
                yield
        finally:
            for attribute_name, original_value in original_values.items():
                setattr(tracking, attribute_name, original_value)
