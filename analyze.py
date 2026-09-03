"""
analyze.py — 綜合分析入口

整合了速度與加速度追蹤 (track_runners.py) 以及姿態與關節角度分析 (run_pipeline.py)。
透過此腳本，可以一鍵完成所有分析流程。
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

from core.pipeline import AnalysisOptions, run_analysis


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True


_POINT_PICKER_PAGE = """\
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pick Line Points</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #111; color: #eee; }}
    header {{ padding: 12px 16px; background: #222; position: sticky; top: 0; z-index: 2; }}
    #wrap {{ position: relative; display: inline-block; margin: 16px; }}
    #frame {{ max-width: calc(100vw - 32px); height: auto; display: block; }}
    #overlay {{ position: absolute; left: 0; top: 0; pointer-events: none; }}
    button {{ margin-right: 8px; padding: 8px 12px; }}
    code {{ color: #9ee; }}
  </style>
</head>
<body>
  <header>
    <div>Video: <code>{video_path}</code></div>
    <div>Click order: 1 start top, 2 end top, 3 end bottom, 4 start bottom.</div>
    <div>
      <button onclick="undoPoint()">Undo</button>
      <button onclick="resetPoints()">Reset</button>
      <button onclick="submitPoints()">Submit 4 points</button>
      <span id="status">0 / 4 points</span>
    </div>
  </header>
  <div id="wrap">
    <img id="frame" src="/frame.jpg" alt="first frame">
    <canvas id="overlay"></canvas>
  </div>
  <script>
    const naturalWidth = {image_width};
    const naturalHeight = {image_height};
    const img = document.getElementById("frame");
    const canvas = document.getElementById("overlay");
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("status");
    const points = [];
    const labels = ["1 start top", "2 end top", "3 end bottom", "4 start bottom"];

    function syncCanvas() {{
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
      redraw();
    }}

    function toDisplay(point) {{
      return [
        point[0] * canvas.width / naturalWidth,
        point[1] * canvas.height / naturalHeight,
      ];
    }}

    function redraw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.font = "16px Arial";
      points.forEach((point, idx) => {{
        const [x, y] = toDisplay(point);
        ctx.fillStyle = "#ff3b30";
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(labels[idx], x + 8, y - 8);
      }});
      if (points.length >= 2) drawLine(points[0], points[1], "#4da3ff");
      if (points.length >= 4) {{
        drawLine(points[3], points[2], "#4da3ff");
        drawLine(points[0], points[3], "#34c759");
        drawLine(points[1], points[2], "#34c759");
      }}
      statusEl.textContent = `${{points.length}} / 4 points`;
    }}

    function drawLine(a, b, color) {{
      const [x1, y1] = toDisplay(a);
      const [x2, y2] = toDisplay(b);
      ctx.strokeStyle = color;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }}

    img.addEventListener("load", syncCanvas);
    window.addEventListener("resize", syncCanvas);
    img.addEventListener("click", (event) => {{
      if (points.length >= 4) return;
      const rect = img.getBoundingClientRect();
      const x = Math.round((event.clientX - rect.left) * naturalWidth / rect.width);
      const y = Math.round((event.clientY - rect.top) * naturalHeight / rect.height);
      points.push([x, y]);
      redraw();
    }});

    function undoPoint() {{
      points.pop();
      redraw();
    }}

    function resetPoints() {{
      points.length = 0;
      redraw();
    }}

    async function submitPoints() {{
      if (points.length !== 4) {{
        alert("Please select exactly 4 points.");
        return;
      }}
      const response = await fetch("/submit", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ points }}),
      }});
      if (!response.ok) {{
        alert("Submit failed.");
        return;
      }}
      document.body.innerHTML = "<h2 style='font-family: Arial; padding: 24px;'>Submitted. Return to terminal.</h2>";
    }}
  </script>
</body>
</html>"""


def _first_frame_jpeg(video_path: str) -> tuple[bytes, int, int]:
    """Read the first video frame; return (jpeg_bytes, native_width, native_height).
    The JPEG is downscaled to <=1280px wide for the browser, but the reported
    size is the native resolution so picked points map back correctly."""
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError(f"cannot read first frame: {video_path}")

    max_width = 1280
    scale = min(1.0, max_width / frame.shape[1])
    display = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
    ok, encoded = cv2.imencode(".jpg", display)
    if not ok:
        raise RuntimeError(f"cannot encode first frame: {video_path}")
    return encoded.tobytes(), frame.shape[1], frame.shape[0]


def _serve_until(handler_class, preferred_port: int, is_done) -> None:
    """Bind the first free port at/above preferred_port and handle requests until
    is_done() is true."""
    server = None
    for port in range(preferred_port, preferred_port + 20):
        try:
            server = ReusableHTTPServer(("0.0.0.0", port), handler_class)
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError(
            f"cannot bind point picker port from {preferred_port} to {preferred_port + 19}"
        )

    public_host = os.getenv("POINT_PICKER_HOST", "catslab.ee.ncku.edu.tw")
    print("\n" + "=" * 60)
    print("Open this URL in your browser and pick 4 points:")
    print(f"http://{public_host}:{server.server_port}")
    print("Waiting for browser submission...")
    print("=" * 60)

    try:
        while not is_done():
            server.handle_request()
    finally:
        server.server_close()


def pick_line_points(video_path: str) -> tuple[list[list[int]], list[list[int]]]:
    image_bytes, image_width, image_height = _first_frame_jpeg(video_path)
    page = _POINT_PICKER_PAGE.format(
        video_path=video_path, image_width=image_width, image_height=image_height
    ).encode("utf-8")
    selected: dict[str, list[list[int]] | None] = {"points": None}

    class PointPickerHandler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args) -> None:
            return

        def _respond(self, content_type: str, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/frame.jpg":
                self._respond("image/jpeg", image_bytes)
            else:
                self._respond("text/html; charset=utf-8", page)

        def do_POST(self) -> None:
            if self.path != "/submit":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            points = payload.get("points")
            if not isinstance(points, list) or len(points) != 4:
                self.send_error(400, "expected 4 points")
                return
            selected["points"] = [[int(p[0]), int(p[1])] for p in points]
            self.send_response(200)
            self.end_headers()

    _serve_until(
        PointPickerHandler,
        int(os.getenv("POINT_PICKER_PORT", "18002")),
        lambda: selected["points"] is not None,
    )

    points = selected["points"]
    start_line = [points[0], points[3]]
    end_line = [points[1], points[2]]
    return start_line, end_line


def build_camera_config(
    video_path: str, distance_m: float, lane_margin_px: int = 10, pre_roll_px: int = 100
) -> dict:
    start_line, end_line = pick_line_points(video_path)
    return {
        "video_path": video_path,
        "start_line": start_line,
        "end_line": end_line,
        "distance_m": distance_m,
        "homography_lane_margin_px": lane_margin_px,
        "pre_roll_px": pre_roll_px,
    }


if __name__ == "__main__":
    start_time = time.perf_counter()

    test_cameras = [
        {
            "video_path": "/home/jeter/runner-analysis-pipeline/MotionAGFormer/demo/video/IMG_5707 (1).mp4",
            "distance_m": 20.0,
        },
        {
            "video_path": "/home/jeter/runner-analysis-pipeline/video/0506_4.mp4",
            "distance_m": 20.0,
        },
    ]

    # cameras 裡每一筆代表一支攝影機影片；start/end line 由視窗點選產生，不再手寫 crop。
    config_dict = {
        "cameras": [
            build_camera_config(cam["video_path"], cam["distance_m"])
            for cam in test_cameras
        ],
        "tracking_mode": "two_pass",
    }

    run_analysis(
        config_dict=config_dict,
        options=AnalysisOptions(gpu="0"),
    )

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 60)
    print(f"總執行時間: {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分鐘)")
    print("=" * 60)
