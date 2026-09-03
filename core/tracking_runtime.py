"""隔離單次 tracking 執行需要的全域設定。"""

from contextlib import contextmanager
from dataclasses import dataclass

from core import tracking
from core.process_runtime import (
    PROCESS_STATE_LOCK,
    temporary_environment_variable,
)

# Single source of truth lives in core.tracking (so the CLI path
# _apply_config_overrides and this pipeline path never drift apart).
from core.tracking import _TRACKING_CONFIG_FIELDS as TRACKING_CONFIG_FIELDS


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
