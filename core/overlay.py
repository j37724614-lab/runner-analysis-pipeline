"""
core/overlay.py

在原始（未裁切）影片上疊加 2D 骨架與起終點標線，支援多相機串接。
此模組封裝了原本在根目錄下 overlay_original.py 的核心運算邏輯。
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm

from core.draw_utils import draw_dashed_line as _draw_dashed_line


# ---------------------------------------------------------------------------
# 骨架繪製（H36M 17 關節格式）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PoseOverlayRequest:
    """在原始影格繪製一幀 2D 姿態所需的資料。"""

    keypoints: "np.ndarray"
    image: "np.ndarray"
    offset: tuple
    foot_keypoints: "np.ndarray | None" = None
    foot_scores: "np.ndarray | None" = None


class _OriginalPoseRenderer:
    """將裁切空間的 H36M 與腳部關節還原並繪製到原始影格。"""

    # Neck/Nose -> Head 與 Thorax -> Neck/Nose 刻意省略；目前 checkpoint
    # 對這兩組關節不穩定，繪製後會產生明顯跳動。
    BODY_CONNECTIONS = (
        (0, 1, False),
        (1, 2, False),
        (2, 3, False),
        (0, 4, True),
        (4, 5, True),
        (5, 6, True),
        (0, 7, True),
        (7, 8, True),
        (8, 11, True),
        (11, 12, True),
        (12, 13, True),
        (8, 14, False),
        (14, 15, False),
        (15, 16, False),
    )
    FOOT_CONNECTIONS = (
        (0, 6),
        (1, 6),
        (2, 6),
        (3, 3),
        (4, 3),
        (5, 3),
    )
    LEFT_BODY_COLOR = (255, 0, 0)
    RIGHT_BODY_COLOR = (0, 0, 255)
    LEFT_FOOT_COLOR = (0, 255, 255)
    RIGHT_FOOT_COLOR = (255, 0, 255)
    JOINT_COLOR = (0, 255, 0)
    FOOT_SCORE_THRESHOLD = 0.3
    BODY_THICKNESS = 2

    def __init__(self, request):
        self.request = request

    def render(self):
        self._draw_body()
        if self.request.foot_keypoints is not None:
            self._draw_feet()
        return self.request.image

    def _draw_body(self):
        for start_index, end_index, is_left in self.BODY_CONNECTIONS:
            start = self._original_point(self.request.keypoints[start_index])
            end = self._original_point(self.request.keypoints[end_index])
            color = self.LEFT_BODY_COLOR if is_left else self.RIGHT_BODY_COLOR
            cv2.line(
                self.request.image,
                start,
                end,
                color,
                self.BODY_THICKNESS,
            )
            self._draw_joint(start)
            self._draw_joint(end)

    def _draw_feet(self):
        for foot_index, ankle_index in self.FOOT_CONNECTIONS:
            if not self._foot_point_is_visible(foot_index):
                continue
            foot = self._original_point(
                self.request.foot_keypoints[foot_index]
            )
            ankle = self._original_point(self.request.keypoints[ankle_index])
            color = (
                self.LEFT_FOOT_COLOR
                if foot_index < 3
                else self.RIGHT_FOOT_COLOR
            )
            cv2.line(self.request.image, ankle, foot, color, 2)
            cv2.circle(self.request.image, foot, 4, color, -1)

    def _original_point(self, point):
        offset_x, offset_y = self.request.offset
        return int(point[0] + offset_x), int(point[1] + offset_y)

    def _foot_point_is_visible(self, foot_index):
        return (
            self.request.foot_scores is None
            or self.request.foot_scores[foot_index] >= self.FOOT_SCORE_THRESHOLD
        )

    def _draw_joint(self, point):
        cv2.circle(
            self.request.image,
            point,
            thickness=-1,
            color=self.JOINT_COLOR,
            radius=2,
        )


def draw_pose_on_original_frame(request):
    """在原始影格繪製 H36M 17 關節骨架與可選的腳部關節。"""
    return _OriginalPoseRenderer(request).render()


# ---------------------------------------------------------------------------
# 在影格上畫起終點線與跑道範圍
# ---------------------------------------------------------------------------
def _draw_lines(frame, start_line, end_line, homography_points=None):
    """在原始影格上繪製起跑線、終點線以及中間包夾的虛線跑道區間。

    4 點線性投影校正的相機有 start_line/end_line，用兩條實線 + 四邊虛線框住
    中間的跑道範圍。6 點 homography 校正的相機沒有這兩條線（沒有對應的兩線
    語意），改成把 homography_points（該相機的 6 個校正點，pixel 座標）依序
    連成一圈虛線多邊形，畫出同樣的「框住跑道範圍」效果，讓兩種校正模式在疊圖
    影片上的視覺呈現一致。
    """
    if start_line and end_line:
        p0 = (int(start_line[0][0]), int(start_line[0][1]))
        p3 = (int(start_line[1][0]), int(start_line[1][1]))
        p1 = (int(end_line[0][0]), int(end_line[0][1]))
        p2 = (int(end_line[1][0]), int(end_line[1][1]))
        for a, b in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
            _draw_dashed_line(frame, a, b, (255, 255, 255), thickness=2)
        cv2.line(frame, p0, p3, (0, 0, 0), 5)
        cv2.line(frame, p0, p3, (180, 255, 255), 3)  # 黃色起跑線
        cv2.line(frame, p1, p2, (0, 0, 0), 5)
        cv2.line(frame, p1, p2, (255, 200, 100), 3)  # 天藍色終點線
    elif start_line:
        pt1 = (int(start_line[0][0]), int(start_line[0][1]))
        pt2 = (int(start_line[1][0]), int(start_line[1][1]))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 5)
        cv2.line(frame, pt1, pt2, (180, 255, 255), 3)
    elif end_line:
        pt1 = (int(end_line[0][0]), int(end_line[0][1]))
        pt2 = (int(end_line[1][0]), int(end_line[1][1]))
        cv2.line(frame, pt1, pt2, (0, 0, 0), 5)
        cv2.line(frame, pt1, pt2, (255, 200, 100), 3)
    elif homography_points and len(homography_points) >= 2:
        pts = [(int(p[0]), int(p[1])) for p in homography_points]
        for a, b in zip(pts, pts[1:] + [pts[0]]):
            _draw_dashed_line(frame, a, b, (255, 255, 255), thickness=2)


# ---------------------------------------------------------------------------
# 共用：載入 offsets / keypoints / foot npz 與 config 合併
# ---------------------------------------------------------------------------
@dataclass
class _OverlaySources:
    """overlay_videos() 與 overlay_videos_per_camera() 共用的輸入資料。"""
    cameras: list
    offsets: "np.ndarray"
    orig_frames: "np.ndarray"
    cam_indices: "np.ndarray"
    kps_map: dict
    foot_kps_map: dict
    foot_scores_map: dict
    num_cams: int


def _merge_line_config(cameras, config):
    """以 config['cameras'] 的 start_line/end_line 補上各相機（不覆蓋已有值）。"""
    if not (config and 'cameras' in config):
        return cameras
    cfg_cams = config['cameras']
    merged = []
    for i, cam in enumerate(cameras):
        c = dict(cam)
        if i < len(cfg_cams):
            c.setdefault('start_line', cfg_cams[i].get('start_line'))
            c.setdefault('end_line', cfg_cams[i].get('end_line'))
        merged.append(c)
    return merged


def _load_overlay_sources(cameras, offsets_npz, kps_npz, config):
    """讀 offsets / keypoints /（可選）foot npz，建立 v_idx → keypoints 對應表。"""
    cameras = _merge_line_config(cameras, config)

    offsets_data = np.load(offsets_npz)
    offsets = offsets_data['offsets']
    orig_frames = offsets_data['orig_frames']
    if 'cam_indices' in offsets_data:
        cam_indices = offsets_data['cam_indices'].astype(int)
    else:
        print("  ⚠️  offsets.npz 未含 cam_indices，假設所有幀皆來自相機 0")
        cam_indices = np.zeros(len(orig_frames), dtype=int)

    kps_data = np.load(kps_npz, allow_pickle=True)
    keypoints = kps_data['reconstruction'][0]
    valid_frames = np.asarray(kps_data['valid_frames']).flatten().astype(int)
    kps_map = {v: keypoints[i] for i, v in enumerate(valid_frames)
               if v < len(orig_frames)}

    foot_kps_map, foot_scores_map = {}, {}
    foot_npz = os.path.join(os.path.dirname(kps_npz), 'foot_keypoints.npz')
    if os.path.exists(foot_npz):
        foot_data = np.load(foot_npz, allow_pickle=True)
        foot_keypoints = foot_data['keypoints'][0]
        foot_scores = foot_data['scores'][0]
        for i, v in enumerate(valid_frames):
            if v < len(orig_frames) and i < len(foot_keypoints):
                foot_kps_map[v] = foot_keypoints[i]
                foot_scores_map[v] = foot_scores[i]

    num_cams = int(max(cam_indices)) + 1 if len(cam_indices) > 0 else len(cameras)
    return _OverlaySources(cameras, offsets, orig_frames, cam_indices,
                           kps_map, foot_kps_map, foot_scores_map, num_cams)


def _iter_original_frames(sources, wanted_cams=None):
    """依 orig_frames 順序產出 (v_idx, c_idx, frame)。每台相機只開一次
    VideoCapture 並循序讀取（cap.set() 在壓縮影片上定位不準）。
    不在 wanted_cams、或無法開啟/讀取的幀 → frame 為 None。"""
    current_cam_idx = -1
    cap = None
    current_frame_pos = 0
    try:
        for v_idx in range(len(sources.orig_frames)):
            c_idx = int(sources.cam_indices[v_idx])
            orig_idx = int(sources.orig_frames[v_idx])

            if wanted_cams is not None and c_idx not in wanted_cams:
                yield v_idx, c_idx, None
                continue

            if c_idx != current_cam_idx:
                if cap is not None:
                    cap.release()
                    cap = None
                video_path = (sources.cameras[c_idx].get('video_path')
                              if c_idx < len(sources.cameras) else None)
                if not video_path or not os.path.exists(video_path):
                    print(f"  ⚠️  找不到相機 {c_idx} 的影片: {video_path}，跳過")
                    yield v_idx, c_idx, None
                    continue
                cap = cv2.VideoCapture(video_path)
                current_cam_idx = c_idx
                current_frame_pos = 0

            ret = False
            frame = None
            while current_frame_pos <= orig_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame_pos += 1
            yield v_idx, c_idx, (frame if ret else None)
    finally:
        if cap is not None:
            cap.release()


def _draw_skeleton_on_frame(frame, sources, v_idx):
    """在原始幀上畫起終點線與（若有）骨架，回傳更新後的 frame。"""
    if v_idx not in sources.kps_map:
        return frame
    off_x, off_y = sources.offsets[v_idx]
    return draw_pose_on_original_frame(PoseOverlayRequest(
        keypoints=sources.kps_map[v_idx],
        image=frame,
        offset=(off_x, off_y),
        foot_keypoints=sources.foot_kps_map.get(v_idx),
        foot_scores=sources.foot_scores_map.get(v_idx),
    ))


# ---------------------------------------------------------------------------
# 主要公開 API
# ---------------------------------------------------------------------------
def overlay_videos(cameras, offsets_npz, kps_npz, output_video, config=None):
    """
    在各台相機的原始影片上疊加 2D 骨架與起終點標線，輸出為單支合成影片。

    cameras / offsets_npz / kps_npz / output_video 同舊版；config 提供時以其
    start/end_line 覆寫。
    """
    sources = _load_overlay_sources(cameras, offsets_npz, kps_npz, config)

    first_video = sources.cameras[0].get('video_path')
    if not first_video or not os.path.exists(first_video):
        raise FileNotFoundError(f"找不到影片: {first_video}")
    cap0 = cv2.VideoCapture(first_video)
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap0.get(cv2.CAP_PROP_FPS)
    cap0.release()
    if fps <= 0:
        fps = 30.0

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    print(f"[Core.Overlay] 正在輸出原解析度骨架影片: {output_video} "
          f"(幀數: {len(sources.orig_frames)}, 相機數: {sources.num_cams})")

    with tqdm(total=len(sources.orig_frames), desc="Overlaying") as pbar:
        for v_idx, c_idx, frame in _iter_original_frames(sources):
            if frame is not None:
                cam_cfg = sources.cameras[c_idx] if c_idx < len(sources.cameras) else {}
                _draw_lines(frame, cam_cfg.get('start_line'), cam_cfg.get('end_line'),
                            cam_cfg.get('homography_src_points'))
                frame = _draw_skeleton_on_frame(frame, sources, v_idx)
                out.write(frame)
            pbar.update(1)

    out.release()
    print(f"✅ [Core.Overlay] 原影片骨架疊加完成！儲存至: {output_video}")


def _open_per_camera_writers(sources, output_paths):
    """依 output_paths 為每台有效相機建立 VideoWriter（以該相機影片尺寸）。"""
    writers = {}
    for c_idx in range(sources.num_cams):
        out_path = output_paths[c_idx] if c_idx < len(output_paths) else None
        if not out_path:
            continue
        video_path = (sources.cameras[c_idx].get('video_path')
                      if c_idx < len(sources.cameras) else None)
        if not video_path or not os.path.exists(video_path):
            print(f"  ⚠️  找不到相機 {c_idx} 的影片，略過該相機的疊圖輸出: {video_path}")
            continue
        probe = cv2.VideoCapture(video_path)
        w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
        probe.release()
        writers[c_idx] = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    return writers


@dataclass(frozen=True)
class _LandingAnnotation:
    """單幀落地資訊疊圖所需的資料。"""

    frame: "np.ndarray"
    ankle_row: object
    camera_history: list
    total_steps: int
    contact_display: object


def _annotate_landing(request):
    """在一幀上畫腳踝點、過去 20 個落地事件標記、時間/步數文字。"""
    from scripts.analysis.ankle_step_stride import TEXT_COLOR

    frame = request.frame
    row = request.ankle_row
    if row:
        cv2.circle(frame, (int(row["right_ankle_x"]), int(row["right_ankle_y"])), 3, (0, 0, 255), -1)
        cv2.circle(frame, (int(row["left_ankle_x"]), int(row["left_ankle_y"])), 3, (255, 0, 0), -1)

    for past in request.camera_history[-20:]:
        # homography_lateral_valid 只在 homography 相機上設定；False = 該點世界座標
        # 被標為離群值 → 不畫。
        if past.get("homography_lateral_valid") is False:
            continue
        px, py, colour, joint_tag = request.contact_display(past)
        cv2.circle(frame, (px, py), 6, TEXT_COLOR, 2)
        cv2.circle(frame, (px, py), 3, colour, -1)
        label = f"S{past['step_index']} {joint_tag}"
        if past["step_length_m"] is not None:
            label += f" L={past['step_length_m']:.2f}m"
        elif past["step_length_px"] is not None:
            label += f" L={past['step_length_px']:.0f}px"
        label_y = py + 43 if (past['step_index'] % 2 == 1) else py + 70
        cv2.putText(frame, label, (px + 8, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 2, cv2.LINE_AA)

    if row:
        cv2.putText(frame, f"Time: {row['seq_time_s']:.2f}s", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Steps: {request.total_steps}", (30, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)


@dataclass(frozen=True)
class PerCameraOverlayRequest:
    """產生各相機骨架與落地點回顧影片所需的完整輸入。"""

    cameras: list
    offsets_npz: str
    keypoints_npz: str
    ankle_rows: list
    step_events: list
    output_paths: list
    config: object = None


class _PerCameraOverlayRenderer:
    """管理各相機 overlay 的資料索引、逐幀繪製與輸出資源。"""

    def __init__(self, request):
        from scripts.analysis.ankle_step_stride import _event_contact_display

        self.sources = _load_overlay_sources(
            request.cameras,
            request.offsets_npz,
            request.keypoints_npz,
            request.config,
        )
        self.ankle_rows = {
            int(row["seq_frame"]): row for row in request.ankle_rows
        }
        self.events = {
            int(event["seq_frame"]): event for event in request.step_events
        }
        self.event_history = {}
        self.contact_display = _event_contact_display
        self.writers = _open_per_camera_writers(
            self.sources,
            request.output_paths,
        )

    def render(self):
        """逐幀寫入各相機回顧影片，並確保所有 writer 都會釋放。"""
        print(
            "[Core.Overlay] 正在輸出各相機獨立骨架+落地點疊圖影片"
            f"（相機數: {len(self.writers)}）"
        )
        try:
            self._render_frames()
        finally:
            self._release_writers()
        print(
            "✅ [Core.Overlay] 各相機獨立疊圖完成，"
            f"共 {len(self.writers)} 支影片"
        )

    def _render_frames(self):
        wanted_cameras = set(self.writers)
        with tqdm(
            total=len(self.sources.orig_frames),
            desc="Per-camera overlaying",
        ) as progress:
            for sequence_frame, camera_index, frame in _iter_original_frames(
                self.sources,
                wanted_cams=wanted_cameras,
            ):
                if frame is not None:
                    self._render_frame(sequence_frame, camera_index, frame)
                progress.update(1)

    def _render_frame(self, sequence_frame, camera_index, frame):
        camera = self._camera_config(camera_index)
        _draw_lines(
            frame,
            camera.get('start_line'),
            camera.get('end_line'),
            camera.get('homography_src_points'),
        )
        frame = _draw_skeleton_on_frame(frame, self.sources, sequence_frame)
        camera_history = self._record_event(sequence_frame, camera_index)
        _annotate_landing(_LandingAnnotation(
            frame=frame,
            ankle_row=self.ankle_rows.get(sequence_frame),
            camera_history=camera_history,
            total_steps=self._total_steps(),
            contact_display=self.contact_display,
        ))
        self.writers[camera_index].write(frame)

    def _camera_config(self, camera_index):
        if camera_index < len(self.sources.cameras):
            return self.sources.cameras[camera_index]
        return {}

    def _record_event(self, sequence_frame, camera_index):
        camera_history = self.event_history.setdefault(camera_index, [])
        event = self.events.get(sequence_frame)
        if event:
            camera_history.append(event)
        return camera_history

    def _total_steps(self):
        return sum(len(history) for history in self.event_history.values())

    def _release_writers(self):
        for writer in self.writers.values():
            writer.release()


def overlay_videos_per_camera(request):
    """分別輸出各相機的骨架與落地點回顧影片。"""
    _PerCameraOverlayRenderer(request).render()
