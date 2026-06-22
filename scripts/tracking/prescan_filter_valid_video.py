"""
Use a TensorRT YOLO engine to find valid runner frame ranges and keep only
those ranges in a new video.

This is a standalone temporal pre-scan tool. It does not run the full two-pass
tracker and does not choose the final runner ID. It only:
  1. samples frames with a low-cost detector,
  2. marks sampled frames that contain a valid person bbox,
  3. merges nearby hits into frame ranges,
  4. expands ranges with buffer,
  5. writes a clipped/concatenated video containing only those ranges.

Example:
  TRT_HOME=/home/jeter/pipeline_release/tools/tensorrt_11_cuda12_9
  PYTRT=/home/jeter/pipeline_release/tools/tensorrt_python_11_cuda12_9/usr/lib/python3.10/dist-packages
  PYTHONPATH="$PYTRT" LD_LIBRARY_PATH="$TRT_HOME/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" \
    /home/jeter/.conda/envs/trt310/bin/python scripts/tracking/prescan_filter_valid_video.py \
      --video MotionAGFormer/demo/video/IMG_0033.MOV
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE = REPO_ROOT / "models" / "yolo26x_ultralytics_int8.engine"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output_cut" / "prescan_valid_segments"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _seconds_to_frames(seconds: float | None, fps: float) -> int | None:
    if seconds is None:
        return None
    return max(0, int(round(float(seconds) * fps)))


def _merge_hit_frames(
    hit_frames: list[int],
    total_frames: int,
    stride: int,
    buffer_frames: int,
    max_gap_frames: int,
) -> list[dict]:
    if not hit_frames:
        return []

    ranges: list[tuple[int, int]] = []
    start = end = int(hit_frames[0])
    for frame_idx in hit_frames[1:]:
        frame_idx = int(frame_idx)
        if frame_idx - end <= max_gap_frames:
            end = frame_idx
        else:
            ranges.append((start, end))
            start = end = frame_idx
    ranges.append((start, end))

    expanded: list[tuple[int, int]] = []
    last_start = last_end = None
    for start, end in ranges:
        start = max(0, start - buffer_frames)
        end = min(max(total_frames - 1, 0), end + buffer_frames + stride - 1)
        if last_start is None:
            last_start, last_end = start, end
        elif start <= last_end + 1:
            last_end = max(last_end, end)
        else:
            expanded.append((last_start, last_end))
            last_start, last_end = start, end
    expanded.append((last_start, last_end))

    return [
        {
            "start_frame": int(start),
            "end_frame": int(end),
            "num_frames": int(end - start + 1),
        }
        for start, end in expanded
    ]


def _scan_valid_ranges(args, fps: float, total_frames: int) -> tuple[list[dict], list[dict], float]:
    model = YOLO(str(args.engine), task="detect")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    sampled_rows: list[dict] = []
    hit_frames: list[int] = []
    sampled_count = 0
    start_time = time.perf_counter()
    frame_idx = 0

    while frame_idx < total_frames:
        if frame_idx % args.stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            sampled_count += 1

            result = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]

            valid_boxes = 0
            max_conf = 0.0
            max_height = 0.0
            if result.boxes is not None and len(result.boxes):
                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()
                for box, conf in zip(xyxy, confs):
                    x1, y1, x2, y2 = map(float, box[:4])
                    height = y2 - y1
                    max_height = max(max_height, height)
                    max_conf = max(max_conf, float(conf))
                    if height >= args.min_height:
                        valid_boxes += 1

            is_hit = valid_boxes > 0
            if is_hit:
                hit_frames.append(frame_idx)

            sampled_rows.append(
                {
                    "frame": frame_idx,
                    "time_s": frame_idx / fps if fps > 0 else None,
                    "hit": int(is_hit),
                    "valid_boxes": valid_boxes,
                    "max_conf": round(max_conf, 6),
                    "max_height": round(max_height, 2),
                }
            )
        else:
            ok = cap.grab() if args.use_grab else cap.read()[0]
            if not ok:
                break
        frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - start_time

    ranges = _merge_hit_frames(
        hit_frames=hit_frames,
        total_frames=total_frames,
        stride=args.stride,
        buffer_frames=args.buffer_frames,
        max_gap_frames=args.max_gap_frames,
    )
    for item in ranges:
        item["start_time_s"] = item["start_frame"] / fps if fps > 0 else None
        item["end_time_s"] = item["end_frame"] / fps if fps > 0 else None

    print(
        "[prescan] "
        f"sampled={sampled_count}, hits={len(hit_frames)}, "
        f"ranges={len(ranges)}, elapsed={elapsed:.3f}s, "
        f"source_fps={total_frames / elapsed if elapsed else 0:.2f}"
    )
    return ranges, sampled_rows, elapsed


def _write_kept_video(args, ranges: list[dict], fps: float, width: int, height: int) -> tuple[Path | None, float]:
    if not ranges:
        return None, 0.0

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    out_path = args.output_dir / f"{args.video.stem}_valid_ranges.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open writer: {out_path}")

    start_time = time.perf_counter()
    written = 0
    for item in ranges:
        start = int(item["start_frame"])
        end = int(item["end_frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frame_idx = start
        while frame_idx <= end:
            ok, frame = cap.read()
            if not ok:
                break
            if args.draw_range_label:
                text = f"src frame {frame_idx}  kept range {start}-{end}"
                cv2.putText(frame, text, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(frame, text, (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            written += 1
            frame_idx += 1

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - start_time
    print(f"[write] {out_path} frames={written}, elapsed={elapsed:.3f}s")
    return out_path, elapsed


def _write_reports(
    args,
    ranges: list[dict],
    sampled_rows: list[dict],
    prescan_elapsed: float,
    write_elapsed: float,
    output_video: Path | None,
    fps: float,
    total_frames: int,
    width: int,
    height: int,
) -> Path:
    csv_path = args.output_dir / f"{args.video.stem}_prescan_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "time_s", "hit", "valid_boxes", "max_conf", "max_height"],
        )
        writer.writeheader()
        writer.writerows(sampled_rows)

    kept_frames = sum(item["num_frames"] for item in ranges)
    report = {
        "video": str(args.video),
        "engine": str(args.engine),
        "output_video": str(output_video) if output_video else None,
        "sample_csv": str(csv_path),
        "video_info": {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
        },
        "params": {
            "stride": args.stride,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "min_height": args.min_height,
            "buffer_frames": args.buffer_frames,
            "max_gap_frames": args.max_gap_frames,
            "use_grab": args.use_grab,
        },
        "ranges": ranges,
        "summary": {
            "num_ranges": len(ranges),
            "kept_frames": kept_frames,
            "kept_ratio": kept_frames / total_frames if total_frames else 0.0,
            "prescan_elapsed_sec": prescan_elapsed,
            "write_elapsed_sec": write_elapsed,
            "total_elapsed_sec": prescan_elapsed + write_elapsed,
        },
    }

    json_path = args.output_dir / f"{args.video.stem}_valid_ranges.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[report] {json_path}")
    print(f"[report] {csv_path}")
    return json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--engine", default=DEFAULT_ENGINE, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--stride", default=8, type=_positive_int)
    parser.add_argument("--imgsz", default=640, type=_positive_int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument("--min-height", default=40, type=float)
    parser.add_argument("--device", default=0)
    parser.add_argument("--buffer-frames", default=None, type=int)
    parser.add_argument("--buffer-sec", default=1.0, type=float)
    parser.add_argument("--max-gap-frames", default=None, type=int)
    parser.add_argument("--max-gap-sec", default=1.0, type=float)
    parser.add_argument("--no-grab", action="store_true", help="Decode skipped frames instead of using cap.grab().")
    parser.add_argument("--draw-range-label", action="store_true")
    args = parser.parse_args()

    args.video = args.video.resolve()
    args.engine = args.engine.resolve()
    args.output_dir = args.output_dir.resolve()
    args.use_grab = not args.no_grab

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.engine.exists():
        raise FileNotFoundError(args.engine)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    if args.buffer_frames is None:
        args.buffer_frames = _seconds_to_frames(args.buffer_sec, fps)
    if args.max_gap_frames is None:
        args.max_gap_frames = _seconds_to_frames(args.max_gap_sec, fps)

    print(f"[video] {args.video}")
    print(f"[video] frames={total_frames}, fps={fps:.3f}, size={width}x{height}")
    print(f"[engine] {args.engine}")
    print(
        "[params] "
        f"stride={args.stride}, buffer_frames={args.buffer_frames}, "
        f"max_gap_frames={args.max_gap_frames}, imgsz={args.imgsz}"
    )

    ranges, sampled_rows, prescan_elapsed = _scan_valid_ranges(args, fps, total_frames)
    output_video, write_elapsed = _write_kept_video(args, ranges, fps, width, height)
    _write_reports(
        args=args,
        ranges=ranges,
        sampled_rows=sampled_rows,
        prescan_elapsed=prescan_elapsed,
        write_elapsed=write_elapsed,
        output_video=output_video,
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
    )

    if not ranges:
        print("[done] no valid ranges found")
    else:
        print("[done] valid ranges:")
        for item in ranges:
            print(
                f"  {item['start_frame']} -> {item['end_frame']} "
                f"({item['num_frames']} frames)"
            )


if __name__ == "__main__":
    main()
