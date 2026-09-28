"""Two-pass 多鏡位主跑者追蹤與追焦輸出。

第一遍在原始影格上使用 YOLO/ByteTrack 選定每台相機的主跑者；
第二遍讀取已選 bbox 快取，輸出固定尺寸追焦影格、映射資料，
並可同步送入 HRNet。跑道投影或 Homography 負責範圍驗證與鏡位切換。
"""
# =======================================================================
# 負責系統資源控制與底層函式庫的初始化設定。在使用 Python 處理多鏡頭影片（OpenCV / FFMPEG）加上深度學習模型（PyTorch / YOLO）時
# 很容易因為「狂開執行緒（Threads）」而導致系統崩潰（報錯 Resource temporarily unavailable 或是直接死當）。
# 動作：透過環境變數，強制將這些底層運算的執行緒降到 1。FFMPEG 解碼影片也強制只用 1 個 thread。
# =======================================================================
import logging
import resource as _res

logger = logging.getLogger(__name__)

try:
    _soft, _hard = _res.getrlimit(_res.RLIMIT_NPROC)
    if _soft < _hard:
        _res.setrlimit(_res.RLIMIT_NPROC, (_hard, _hard))
except (OSError, ValueError):
    logger.warning(
        "無法調整 RLIMIT_NPROC，將使用原本的系統限制",
        exc_info=True,
    )

import os

# 限制執行緒數，避免 RLIMIT_NPROC 超限（在 import cv2 之前設定）
for _k, _v in {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "GOMP_SPINCOUNT": "0",
    "OPENCV_FFMPEG_CAPTURE_OPTIONS": "threads;1",  # FFMPEG option name is "threads"
}.items():
    os.environ[_k] = _v  # 強制覆蓋，不用 setdefault

import cv2

try:
    cv2.setLogLevel(3)  # type: ignore[attr-defined]  # 抑制 swscaler 色彩轉換警告
except AttributeError:
    pass
from core.utils import DEFAULT_OUTPUT_DIR, get_model_path

# =======================================================================
# 設定區（每次修改只需改這裡）
# =======================================================================

# 使用哪張實體 GPU（'0' = 第 0 張，'1' = 第 1 張）
CUDA_VISIBLE_DEVICES = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
DEVICE = 0

# 模型權重路徑（下載方式見 README）
MODEL_PATH = get_model_path("yolo26x.pt")
TWO_PASS_TRACKER_CONFIG = "bytetrack.yaml"

# 輸出目錄（檔名依第一台有效相機自動命名：{輸入檔名}_tracked.mp4）
OUTPUT_DIR = DEFAULT_OUTPUT_DIR

# 未啟用 auto_crop 時的固定裁剪尺寸
# 一般分析流程應使用 auto_crop，避免不同解析度或人物大小都被限制在固定尺寸。
# 建議值：先執行一次看底部「建議 CROP_WIDTH/CROP_HEIGHT」的統計輸出再調整
# 或在 config 加入 auto_crop: true，讓程式依 bbox 統計自動決定正方形尺寸
CROP_WIDTH  = 200
CROP_HEIGHT = 260
AUTO_CROP   = False  # True → 依第一遍選定主跑者的 bbox 自動設定裁剪尺寸
TRACKING_MODE = 'two_pass'

# Optional temporal pre-scan for the two-pass tracking flow.
# It scans sampled frames with a TensorRT INT8 YOLO engine, finds ranges where
# a valid person appears, expands the ranges with buffer, and limits both
# two-pass Pass 1 and Pass 2 to those original-frame ranges.
PRESCAN_ENABLED = False
PRESCAN_ENGINE_PATH = get_model_path("yolo26x_ultralytics_int8.engine")
PRESCAN_STRIDE = 15
PRESCAN_IMGSZ = 640
PRESCAN_CONF = 0.25
PRESCAN_IOU = 0.7
PRESCAN_BUFFER_SEC = 0.5
PRESCAN_MAX_GAP_SEC = 1.0
PRESCAN_USE_GRAB = True

# 是否在裁剪畫面上疊加框
#   True  = 綠色 bbox（已選主跑者）+ 藍色 ROI 框
#   False = 輸出乾淨畫面（無任何疊加）
SHOW_OVERLAY = True
DRAW_BBOX_OVERLAY = True  # True → 輸出追焦影片畫 bbox 與 track ID；bbox_map.csv 仍會照常輸出給 HRNet
WRITE_OVERVIEW_VIDEO = False  # True → 額外輸出原始全畫面的追蹤除錯影片

# -----------------------------------------------------------------------
# 第二遍主跑者狀態
# -----------------------------------------------------------------------
SELECTED_RUNNER_MEMORY_FRAMES = 30  # 連續漏偵超過此幀數後，清除舊的 bbox/平滑地面點
MIN_PERSON_HEIGHT   = 40  # bbox 高度小於此值（前處理裁剪後像素）視為背景遠景人物，略過
GROUND_POINT_EMA_ALPHA = 0.35  # 與 tracker_impl.py 對齊：bbox 底部中心點平滑係數


# =======================================================================
# 對外公開介面：從各子模組匯入，讓 `from core import tracking` 的既有呼叫方
# （core/pipeline.py、core/tracking_runtime.py、測試）完全不用修改。
# =======================================================================
from core.tracking.camera_setup import (
    CameraConfig,
    _build_camera_from_entry,
    camera,
)
from core.tracking.camera_switch import _should_switch_camera
from core.tracking.crop import _crop_from_bbox
from core.tracking.debug_output import (
    _auto_crop_from_selected_cache,
    _write_two_pass_debug,
)
from core.tracking.pass1 import (
    _collect_all_detections,
    _PassOneDetectionCollector,
    _PassOneRoi,
)
from core.tracking.pass2_frame import (
    FrameProcessingConfig,
    _cached_detections_to_arrays,
    process_frame,
)
from core.tracking.pass2_render import (
    CameraBatchRequest,
    TrackedFramePacket,
    _process_cameras,
    _process_single_camera,
    _resolve_cached_detections,
    _SingleCameraProcessor,
)
from core.tracking.prescan import run_temporal_prescan
from core.tracking.scoring import _score_and_select_runners
from core.tracking.stitch import _stitch_target_id

# 上面這些名稱只透過 `tracking.xxx` 被外部呼叫，這裡本身不會用到；
# 明確宣告 __all__ 讓這是有意的重新匯出，而不是遺留的死 import。
__all__ = [
    "CameraConfig",
    "_build_camera_from_entry",
    "camera",
    "_should_switch_camera",
    "_crop_from_bbox",
    "_auto_crop_from_selected_cache",
    "_write_two_pass_debug",
    "_collect_all_detections",
    "_PassOneDetectionCollector",
    "_PassOneRoi",
    "FrameProcessingConfig",
    "_cached_detections_to_arrays",
    "process_frame",
    "CameraBatchRequest",
    "TrackedFramePacket",
    "_process_cameras",
    "_process_single_camera",
    "_resolve_cached_detections",
    "_SingleCameraProcessor",
    "run_temporal_prescan",
    "_score_and_select_runners",
    "_stitch_target_id",
]


# config-key → (module-constant name, value converter). Single source of truth,
# also consumed by core.tracking_runtime.temporary_tracking_runtime.
_TRACKING_CONFIG_FIELDS = {
    'output_dir':          ('OUTPUT_DIR',          str),
    'crop_width':          ('CROP_WIDTH',          int),
    'crop_height':         ('CROP_HEIGHT',         int),
    'auto_crop':           ('AUTO_CROP',           bool),
    'show_overlay':        ('SHOW_OVERLAY',        bool),
    'draw_bbox_overlay':   ('DRAW_BBOX_OVERLAY',   bool),
    'write_overview_video': ('WRITE_OVERVIEW_VIDEO', bool),
    # 保留舊 config key，避免現有設定檔失效；它現在只控制主跑者狀態的漏偵保留幀數。
    'max_person_memory':   ('SELECTED_RUNNER_MEMORY_FRAMES', int),
    'min_person_height':   ('MIN_PERSON_HEIGHT',   int),
    'tracking_mode':       ('TRACKING_MODE',       str),
    'prescan_enabled':     ('PRESCAN_ENABLED',     bool),
    'prescan_engine_path': ('PRESCAN_ENGINE_PATH', str),
    'prescan_stride':      ('PRESCAN_STRIDE',      int),
    'prescan_imgsz':       ('PRESCAN_IMGSZ',       int),
    'prescan_conf':        ('PRESCAN_CONF',        float),
    'prescan_iou':         ('PRESCAN_IOU',         float),
    'prescan_buffer_sec':  ('PRESCAN_BUFFER_SEC',  float),
    'prescan_max_gap_sec': ('PRESCAN_MAX_GAP_SEC', float),
    'prescan_use_grab':    ('PRESCAN_USE_GRAB',    bool),
}
