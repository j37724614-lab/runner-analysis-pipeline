"""
core.tracker

Thin public entry point for the runner speed tracker.

The full implementation lives in ``core.tracker_impl`` so this module stays
easy to scan while preserving the old import path:

    from core.tracker import main, run_tracker

Responsibilities of the tracker:
  - detect and track the main runner with YOLO/ByteTrack
  - estimate distance from bbox ground points, start/end lines, or homography
  - smooth distance/speed/acceleration with interpolation and Kalman filtering
  - render the tracking video and write the metrics CSV
"""

from core.tracker_impl import *  # noqa: F401,F403 - keep backward-compatible API
from core.tracker_impl import main, run_tracker


if __name__ == "__main__":
    main()
