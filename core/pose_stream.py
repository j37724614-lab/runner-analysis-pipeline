"""Bounded tracked-frame handoff to an isolated 2D HRNet worker process."""

from __future__ import annotations

from dataclasses import replace
from collections import defaultdict
import atexit
import multiprocessing
import os
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
                 gpu, model_path, pose_backend):
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
                profiling=pose_metrics, backend=pose_backend,
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


def _persistent_pose_worker_loop(control, frames, result, gpu):
    """Long-lived MogaNet worker: runs one job after another with the same
    process, instead of a fresh spawn per job. The only reason this exists is
    to amortize torch.compile's one-off warmup (model_load_sec jumps from
    ~0.5s to ~6s the first time a job compiles) across many analysis runs
    instead of re-paying it every single time, since _model_cache otherwise
    starts empty on every freshly spawned process.

    Each job's parameters arrive on `control`; a None job is the shutdown
    signal. `frames`/`result` are reused across jobs exactly like _pose_worker
    already reuses them within one job (they are plain Queue objects, not
    tied to a single call).
    """
    while True:
        job = control.get()
        if job is None:
            return
        video_path, output_dir, bbox_csv, model_path, pose_backend = job
        _pose_worker(frames, result, video_path, output_dir, bbox_csv, gpu, model_path, pose_backend)


class _PersistentMogaNetWorker:
    """Module-level singleton owning the long-lived MogaNet worker process.

    Scoped to the MogaNet backend only (opt-in via MOGANET_PERSISTENT_WORKER=1)
    -- HRNet keeps the existing spawn-per-job PoseStreamSession path untouched.
    A crashed or never-started worker is (re)started lazily on next use; any
    job that ends in error tears the whole thing down so the next job starts
    from a clean process, same as the existing per-job isolation guarantee.
    """

    _instance = None

    def __init__(self, gpu, capacity):
        context = multiprocessing.get_context("spawn")
        self.gpu = gpu
        self.control = context.Queue()
        self.frames = context.Queue(maxsize=capacity)
        self.result = context.Queue(maxsize=1)
        self.process = context.Process(
            target=_persistent_pose_worker_loop,
            args=(self.control, self.frames, self.result, gpu),
            name="moganet-persistent-pose-worker",
            daemon=True,
        )
        self.process.start()
        # Without this, a normal interpreter exit (e.g. the FastAPI server
        # restarting) leaves this GPU-resident worker running as an orphan --
        # confirmed by hand: it held ~23GB of GPU memory indefinitely after
        # the owning process exited. daemon=True covers most shutdown paths;
        # this is a second, more graceful attempt (lets the worker's current
        # job finish before it exits instead of killing it mid-job).
        atexit.register(self.shutdown)

    @classmethod
    def get_or_start(cls, gpu, capacity):
        existing = cls._instance
        if existing is not None and existing.gpu == gpu and existing.process.is_alive():
            return existing
        if existing is not None:
            existing.shutdown()
        cls._instance = cls(gpu, capacity)
        return cls._instance

    @classmethod
    def discard(cls):
        """Tear down the current singleton; the next get_or_start() starts fresh."""
        if cls._instance is not None:
            cls._instance.shutdown()
            cls._instance = None

    def shutdown(self):
        if getattr(self, "_shut_down", False):
            return  # idempotent: may run once via discard() and again via atexit
        self._shut_down = True
        if self.process.is_alive():
            try:
                self.control.put(None)
                self.process.join(timeout=5)
            except Exception:
                pass
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.control.close()
        self.frames.close()
        self.result.close()


class PoseStreamSession:
    """Small producer interface; queue, worker lifecycle and errors stay inside."""

    def __init__(self, video_path: str, output_dir: str, gpu: str,
                 model_path: str | None = None, capacity: int = 16,
                 pose_backend: str = "hrnet"):
        self.capacity = capacity
        self.video_path = video_path
        self.output_dir = output_dir
        self.bbox_csv = video_path.replace(".mp4", "_bbox_map.csv")
        self.persistent = (
            pose_backend == "moganet"
            and os.environ.get("MOGANET_PERSISTENT_WORKER") == "1"
        )
        self._job_ok = False
        if self.persistent:
            shared = _PersistentMogaNetWorker.get_or_start(gpu, capacity)
            self.frames = shared.frames
            self.result = shared.result
            self.worker = shared.process
            shared.control.put(
                (video_path, output_dir, self.bbox_csv, model_path, pose_backend)
            )
        else:
            context = multiprocessing.get_context("spawn")
            self.frames = context.Queue(maxsize=capacity)
            self.result = context.Queue(maxsize=1)
            self.worker = context.Process(
                target=_pose_worker,
                args=(self.frames, self.result, video_path, output_dir,
                      self.bbox_csv, gpu, model_path, pose_backend),
                name="hrnet-pose-stream",
            )
            self.worker.start()
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
            if self.persistent:
                # The shared worker loops forever across jobs; it never exits,
                # so joining it would just hang until the timeout below fires.
                # Wait for this job's result directly instead.
                try:
                    outcome = self.result.get(timeout=300)
                except queue.Empty as error:
                    raise RuntimeError(
                        "MogaNet persistent worker returned no result "
                        f"(exit={self.worker.exitcode})"
                    ) from error
            else:
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
            self._job_ok = True
            return self.produced
        finally:
            self.abort()

    def abort(self):
        if self.persistent:
            # Happy path: leave the shared worker running so the next session
            # reuses its already-compiled model instead of paying the warmup
            # cost again. Only recycle it when this job ended in error, same
            # isolation guarantee the non-persistent path gets for free by
            # always using a brand-new process.
            if not self._job_ok:
                _PersistentMogaNetWorker.discard()
            return
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=5)
        self.frames.close()
        self.result.close()
