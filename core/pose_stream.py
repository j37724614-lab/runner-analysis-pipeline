"""Bounded tracked-frame handoff to an isolated 2D HRNet worker process."""

from __future__ import annotations

from dataclasses import replace
from collections import defaultdict
import multiprocessing
from pathlib import Path
import queue
import time
import traceback

import numpy as np

from core.utils import REPO_ROOT


def summarize_queue_depth(enqueued_ns, dequeued_ns, capacity):
    """Time-weighted FIFO depth from successful put/get events on one host."""
    if len(enqueued_ns) != len(dequeued_ns):
        raise ValueError("Queue put/get traces have different counts")
    if not enqueued_ns:
        return {
            "queue_depth_avg": 0.0,
            "queue_depth_p95": 0,
            "queue_depth_max": 0,
            "queue_trace_clock_adjustments": 0,
        }

    events = []
    adjustments = 0
    for index, (put_at, get_at) in enumerate(zip(enqueued_ns, dequeued_ns)):
        # The consumer may run between Queue.put() returning and the producer
        # recording its timestamp. This race is at most one packet's trace,
        # not a queue ordering violation; give that packet zero residence time.
        if put_at > get_at:
            put_at = get_at
            adjustments += 1
        # Conversely, Queue.get() frees a slot before the consumer timestamps
        # it. A later put can be recorded before that get; preserve the real
        # bounded-queue happens-before relation when reconstructing depth.
        if index >= capacity and put_at < dequeued_ns[index - capacity]:
            put_at = dequeued_ns[index - capacity]
            adjustments += 1
        events.extend(((put_at, 1), (get_at, -1)))

    events.sort()
    depth = 0
    maximum = 0
    duration_by_depth = defaultdict(int)
    previous_at = events[0][0]
    index = 0
    while index < len(events):
        at = events[index][0]
        duration_by_depth[depth] += at - previous_at
        while index < len(events) and events[index][0] == at:
            depth += events[index][1]
            index += 1
        if depth < 0 or depth > capacity:
            raise ValueError(f"Queue trace depth {depth} outside 0..{capacity}")
        maximum = max(maximum, depth)
        previous_at = at
    if depth != 0:
        raise ValueError("Queue trace did not drain")

    total_ns = sum(duration_by_depth.values())
    if not total_ns:
        average = 0.0
        p95 = maximum
    else:
        average = sum(level * duration for level, duration in duration_by_depth.items()) / total_ns
        threshold = total_ns * 0.95
        cumulative = 0
        p95 = maximum
        for level, duration in sorted(duration_by_depth.items()):
            cumulative += duration
            if cumulative >= threshold:
                p95 = level
                break
    return {
        "queue_depth_avg": round(average, 4),
        "queue_depth_p95": p95,
        "queue_depth_max": maximum,
        "queue_trace_clock_adjustments": adjustments,
    }


def _pose_worker(frames, result, video_path, output_dir, bbox_csv,
                 gpu, model_path):
    """Run the existing global 2D postprocessing after consuming all frames."""
    try:
        pose_started_at = time.perf_counter()
        from core.pipeline import _import_vis_module, _temporary_motion_agformer_runtime

        motion_dir = REPO_ROOT / "MotionAGFormer"
        with _temporary_motion_agformer_runtime(motion_dir, gpu):
            vis = _import_vis_module(motion_dir)
            count = 0
            queue_wait_sec = 0.0
            idle_sec = 0.0
            dequeued_ns = []
            pose_metrics = {}
            first_frame_at = None
            last_frame_at = None

            def frame_source():
                nonlocal count, queue_wait_sec, idle_sec, first_frame_at, last_frame_at
                while True:
                    waiting_since = time.perf_counter()
                    try:
                        packet = frames.get_nowait()
                        empty_wait = 0.0
                    except queue.Empty:
                        empty_started_at = time.perf_counter()
                        packet = frames.get()
                        empty_wait = time.perf_counter() - empty_started_at
                    received_at = time.perf_counter()
                    queue_wait_sec += received_at - waiting_since
                    if packet is None:
                        return
                    if packet.output_frame != count:
                        raise ValueError(
                            f"Non-contiguous pose frame {packet.output_frame}; expected {count}"
                        )
                    if count:
                        idle_sec += empty_wait
                    dequeued_ns.append(time.perf_counter_ns())
                    count += 1
                    if first_frame_at is None:
                        first_frame_at = received_at
                    last_frame_at = received_at
                    yield packet.image, packet.bbox

            vis.get_pose2D(
                video_path, output_dir, bbox_csv=bbox_csv,
                model_path=model_path, frame_source=frame_source(),
                profiling=pose_metrics,
            )
        result.put({
            "ok": True,
            "frames": count,
            "elapsed_sec": round(time.perf_counter() - pose_started_at, 4),
            "queue_wait_sec": round(queue_wait_sec, 4),
            "consumer_idle_sec": round(idle_sec, 4),
            "dequeued_ns": dequeued_ns,
            "pose_metrics": pose_metrics,
            "frame_span_sec": round(last_frame_at - first_frame_at, 4)
            if first_frame_at is not None and last_frame_at is not None else 0.0,
        })
    except BaseException:
        result.put({"ok": False, "error": traceback.format_exc()})


class PoseStreamSession:
    """Small producer interface; queue, worker lifecycle and errors stay inside."""

    def __init__(self, video_path: str, output_dir: str, gpu: str,
                 model_path: str | None = None, capacity: int = 16):
        context = multiprocessing.get_context("spawn")
        self.frames = context.Queue(maxsize=capacity)
        self.result = context.Queue(maxsize=1)
        self.capacity = capacity
        self.video_path = video_path
        self.output_dir = output_dir
        self.bbox_csv = video_path.replace(".mp4", "_bbox_map.csv")
        self.worker = context.Process(
            target=_pose_worker,
            args=(self.frames, self.result, video_path, output_dir,
                  self.bbox_csv, gpu, model_path),
            name="hrnet-pose-stream",
        )
        self.produced = 0
        self.pose_elapsed_sec = None
        self.worker_queue_wait_sec = None
        self.worker_frame_span_sec = None
        self.enqueue_sec = 0.0
        self.blocked_sec = 0.0
        self.enqueue_events_ns = []
        self.queue_depth_metrics = {}
        self.queue_full_count = 0
        self.queue_full_frames = 0
        self.queue_full_retries = 0
        self.consumer_idle_sec = None
        self.pose_metrics = {}
        self.worker.start()

    def _put(self, item):
        started_at = time.perf_counter()
        if not self.worker.is_alive():
            raise RuntimeError(
                f"HRNet stream worker exited early (exit={self.worker.exitcode})"
            )
        try:
            self.frames.put_nowait(item)
        except queue.Full:
            self.queue_full_count += 1
            if item is not None:
                self.queue_full_frames += 1
            blocked_at = time.perf_counter()
            while True:
                if not self.worker.is_alive():
                    raise RuntimeError(
                        f"HRNet stream worker exited early (exit={self.worker.exitcode})"
                    )
                try:
                    self.frames.put(item, timeout=0.25)
                    break
                except queue.Full:
                    self.queue_full_count += 1
                    self.queue_full_retries += 1
            if item is not None:
                self.blocked_sec += time.perf_counter() - blocked_at
        self.enqueue_sec += time.perf_counter() - started_at
        if item is not None:
            self.enqueue_events_ns.append(time.perf_counter_ns())

    def emit(self, packet):
        if packet.output_frame != self.produced:
            raise ValueError(
                f"Tracking emitted frame {packet.output_frame}, expected {self.produced}"
            )
        # multiprocessing.Queue serializes on a feeder thread after put().
        # Own the image bytes before the tracker advances to another frame.
        self._put(replace(packet, image=packet.image.copy()))
        self.produced += 1

    def finish(self):
        try:
            self._put(None)
            self.worker.join(timeout=300)
            if self.worker.is_alive():
                raise TimeoutError("HRNet stream worker did not finish within 300 seconds")
            try:
                outcome = self.result.get(timeout=5)
            except queue.Empty as error:
                raise RuntimeError(
                    f"HRNet stream worker returned no result (exit={self.worker.exitcode})"
                ) from error
            if not outcome["ok"]:
                raise RuntimeError(f"HRNet stream failed:\n{outcome['error']}")
            self.pose_elapsed_sec = outcome["elapsed_sec"]
            self.worker_queue_wait_sec = outcome["queue_wait_sec"]
            self.worker_frame_span_sec = outcome["frame_span_sec"]
            self.consumer_idle_sec = outcome["consumer_idle_sec"]
            self.pose_metrics = outcome["pose_metrics"]
            self.queue_depth_metrics = summarize_queue_depth(
                self.enqueue_events_ns, outcome["dequeued_ns"], self.capacity,
            )
            pose_path = Path(self.output_dir) / "input_2D" / "keypoints.npz"
            with np.load(pose_path) as keypoints:
                pose_frames = int(keypoints["reconstruction"].shape[1])
            if self.produced != outcome["frames"] or self.produced != pose_frames:
                raise RuntimeError(
                    "HRNet frame count differs from tracking: "
                    f"tracked={self.produced}, consumed={outcome['frames']}, pose={pose_frames}"
                )
            return self.produced
        finally:
            self.abort()

    def abort(self):
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=5)
        self.frames.close()
        self.result.close()
