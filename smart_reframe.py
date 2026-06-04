"""
Smart Reframe API — production-grade 9:16 clip extraction from landscape video.

Pipeline (per clip segment):
  1. FFmpeg lossless slice at source interval
  2. Frame-by-frame face detection (OpenCV DNN) with temporal smoothing
  3. Dynamic 9:16 crop window centered on tracked speaker
  4. FFmpeg re-encode with libx264 high profile + AAC audio at 1080p

Quick start:
  pip install "fastapi[standard]" opencv-python numpy
  uvicorn smart_reframe:app --host 0.0.0.0 --port 8000

  → Open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import urllib.request
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

SMOOTHING_ALPHA = 0.35
DEADBAND_PX = 3.0
FACE_VERTICAL_TARGET = 0.33
LOST_FACE_TIMEOUT = 60
ADAPTIVE_TRACKING = True   # larger movements → faster response

FFMPEG_PRESET = "medium"
FFMPEG_CRF = 18
FFMPEG_DENOISE = "hqdn3d=1.0:1.0:0.5:0.5"
FFMPEG_VT_AVAIL = False
AUDIO_BITRATE = "192k"
MAX_UPLOAD_SIZE = 10 * 1024**3  # 10 GB

EMOJI_FONT = "/System/Library/Fonts/Apple Color Emoji.ttc"
CAPTION_FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
CAPTION_STYLE = (
    "fontfile={font}:fontsize={size}:fontcolor=white:"
    "box=1:boxcolor=black@0.5:boxborderw=12:"
    "x=(w-text_w)/2:y=h-th-{bottom}"
)

ENGAGEMENT_WINDOW = 5   # seconds per scoring block
ENGAGEMENT_TOP_N = 3    # max clips for smart mode
ENGAGEMENT_MIN_CLIP = 20  # minimum clip duration in seconds
ENGAGEMENT_MAX_CLIP = 45  # maximum clip duration in seconds
MOTION_WEIGHT = 0.4
FACE_WEIGHT = 0.6

MODEL_DIR = Path(tempfile.gettempdir()) / "smart_reframe_models"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "smart_reframe_output"
BIN_DIR = Path(__file__).parent / "bin"

logger = logging.getLogger("smart_reframe")


# ---------------------------------------------------------------------------
# FFmpeg discovery
# ---------------------------------------------------------------------------


def _ensure_ffmpeg() -> None:
    """If ffmpeg isn't on PATH, try to use imageio_ffmpeg's bundled
    binary or download a static build to ``BIN_DIR``."""
    if shutil.which("ffmpeg"):
        return

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = BIN_DIR / "ffmpeg"

    if ffmpeg_path.is_file():
        os.environ["PATH"] = f"{BIN_DIR}:{os.environ.get('PATH', '')}"
        return

    # Try bundled imageio_ffmpeg.
    try:
        import imageio_ffmpeg
        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        shutil.copy2(str(src), str(ffmpeg_path))
        ffmpeg_path.chmod(0o755)
        os.environ["PATH"] = f"{BIN_DIR}:{os.environ.get('PATH', '')}"
        logger.info("FFmpeg installed from imageio_ffmpeg: %s", ffmpeg_path)
        return
    except (ImportError, Exception):
        pass

    logger.warning("FFmpeg not found. Run: pip install imageio-ffmpeg")


# ---------------------------------------------------------------------------
# YouTube downloader
# ---------------------------------------------------------------------------

YT_DLP_CMD: list[str] = []


def _ensure_yt_dlp() -> None:
    """Locate yt-dlp (prefer standalone binary, fallback pip)."""
    global YT_DLP_CMD
    if YT_DLP_CMD:
        return
    # 1. Standalone binary (bundles Python 3.12 + OpenSSL, avoids SSL EOF bugs)
    binary_path = BIN_DIR / "yt-dlp"
    if binary_path.is_file():
        YT_DLP_CMD = [str(binary_path)]
        logger.info("yt-dlp binary at: %s", binary_path)
        return
    # 2. python -m yt_dlp (pip install)
    try:
        import yt_dlp  # noqa: F401
        YT_DLP_CMD = [sys.executable, "-m", "yt_dlp"]
        logger.info("yt-dlp resolved via python -m yt_dlp")
        return
    except ImportError:
        pass
    # 3. PATH
    found = shutil.which("yt-dlp")
    if found:
        YT_DLP_CMD = [found]
        logger.info("yt-dlp resolved at: %s", found)
        return
    candidates = [
        str(Path.home() / "Library/Python/3.9/bin/yt-dlp"),
        str(Path.home() / ".local/bin/yt-dlp"),
        "/usr/local/bin/yt-dlp",
    ]
    for c in candidates:
        if Path(c).is_file():
            YT_DLP_CMD = [c]
            logger.info("yt-dlp resolved at: %s", c)
            return
    logger.warning("yt-dlp not found — YouTube downloads disabled")


_YT_URL_RE = re.compile(
    r"^(https?://)?"
    r"(www\.|m\.)?"
    r"(youtube\.com|youtu\.be)"
    r"(/(watch\?v=|embed/|shorts/|live/|v/)[\w-]{11}"
    r"|/watch\?.+v=[\w-]{11})"
    r"([&#]\S*)?$"
)

def _validate_video_url(url: str) -> str:
    """Check *url* points to a single video (not a channel/playlist).
    Returns the cleaned URL or raises ValueError."""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    # Block known non-video patterns.
    blocked = re.search(
        r"(youtube\.com/(c/|@|channel/|playlist\?|user/|feed/|gaming))",
        url, re.I,
    )
    if blocked:
        raise ValueError(
            "That looks like a channel or playlist URL, not a video. "
            "Paste a specific video link (e.g. youtube.com/watch?v=...)"
        )
    if not _YT_URL_RE.match(url):
        raise ValueError(
            "Could not parse YouTube video URL. "
            "Expected format: https://youtube.com/watch?v=..."
        )
    return url


def _run_ytdlp(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        YT_DLP_CMD + list(args), capture_output=True, text=True, timeout=600,
    )


def _probe_youtube_duration(url: str) -> float:
    """Get video duration (seconds) via yt-dlp --print, no download."""
    if not YT_DLP_CMD:
        _ensure_yt_dlp()
    url = _validate_video_url(url)
    result = _run_ytdlp(
        "--print", "duration",
        "--no-playlist",
        "--flat-playlist",
        "--extractor-args", "youtube:player_client=web,mweb,android",
        url,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp probe failed (exit {result.returncode}):\n{result.stderr}"
        )
    return float(result.stdout.strip())


def _download_youtube(url: str, output_dir: Path) -> Path:
    """Download a YouTube video to *output_dir* using yt-dlp.

    Returns the path to the downloaded file.
    """
    if not YT_DLP_CMD:
        _ensure_yt_dlp()
    url = _validate_video_url(url)
    logger.info("Downloading YouTube video: %s", url)
    t0 = time.perf_counter()

    # Cookies: HF secret YT_COOKIES > cookies.txt > Safari on macOS
    cookie_args: list[str] = []
    cookies_text = os.environ.get("YT_COOKIES")
    if cookies_text:
        cp = Path(f"/tmp/yt-cookies-{uuid.uuid4().hex}.txt")
        cp.write_text(cookies_text)
        cookie_args = ["--cookies", str(cp)]
    elif Path("cookies.txt").is_file():
        cookie_args = ["--cookies", "cookies.txt"]
    elif sys.platform == "darwin":
        cookie_args = ["--cookies-from-browser", "safari"]

    result = _run_ytdlp(
        "-f", "18/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--force-ipv4",
        "--throttled-rate", "100K",
        "--extractor-args", "youtube:player_client=web,mweb,android",
        *cookie_args,
        "-o", str(output_dir / "%(id)s.%(ext)s"),
        url,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {result.returncode}):\n{result.stderr}"
        )

    # Find the downloaded file.
    files = sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    mp4_files = [f for f in files if f.suffix == ".mp4"]
    if not mp4_files:
        all_video = [f for f in files if f.suffix in (".mp4", ".webm", ".mkv")]
        if not all_video:
            raise RuntimeError("yt-dlp completed but no video file found")
        out_path = max(all_video, key=lambda p: p.stat().st_size)
    else:
        out_path = mp4_files[0]

    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    logger.info(
        "YouTube download complete: %.1f MB in %.2fs → %s",
        size_mb, elapsed, out_path,
    )
    return out_path

# ---------------------------------------------------------------------------
# Subtitle download & parsing (auto captions)
# ---------------------------------------------------------------------------


def _get_video_title(url: str) -> str:
    """Fetch the YouTube video title via yt-dlp ``--print title``."""
    try:
        result = _run_ytdlp(
            "--no-playlist", "--force-ipv4",
            "--print", "title",
            url,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as exc:
        logger.warning("Title fetch failed: %s", exc)
    return ""


# ---------------------------------------------------------------------------
# Face Detector — OpenCV DNN (Caffe SSD)
# Auto-downloads model files on first use; no extra dependencies.
# ---------------------------------------------------------------------------


def _download_model(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading face detection model: %s", url.split("/")[-1])
    urllib.request.urlretrieve(url, str(dest))
    logger.info("Model downloaded: %s", dest)


def _init_face_detector() -> cv2.dnn.Net:
    proto_url = (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    )
    model_url = (
        "https://github.com/opencv/opencv_3rdparty/raw/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    proto_path = MODEL_DIR / "deploy.prototxt"
    model_path = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

    _download_model(proto_url, proto_path)
    _download_model(model_url, model_path)

    net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


_face_net: Optional[cv2.dnn.Net] = None

# ---------------------------------------------------------------------------
# Upload store  (upload_id → local path)
# ---------------------------------------------------------------------------
UPLOAD_STORE: dict[str, Path] = {}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class ClipInterval(BaseModel):
    start: float = Field(..., ge=0, description="Start time in seconds.")
    end: float = Field(..., ge=0, description="End time in seconds.")

    @field_validator("end")
    @classmethod
    def end_must_exceed_start(cls, v, info):
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError("end must be greater than start")
        return v


class ReframeRequest(BaseModel):
    video_path: str = Field(..., min_length=1, description="Path to the source MP4.")
    clips: list[ClipInterval] = Field(..., min_length=1)


class ClipResult(BaseModel):
    index: int
    interval: dict
    output_path: str
    download_url: str = ""
    duration_sec: float
    face_detection_rate: float
    pipeline_time_sec: float
    engagement_score: float = 0.0
    caption_text: str = ""


class CaptionsConfig(BaseModel):
    text: str = ""
    emoji: str = "🔥"
    style: str = "modern"
    font_size: int = 28


class ReframeResponse(BaseModel):
    status: str
    job_id: str = ""
    clips: list[ClipResult]
    total_time_sec: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEMP_DIR: Optional[Path] = None

_SMART_REFRAME_TEMP = os.environ.get("SMART_REFRAME_TEMP", "")


def _get_temp_dir() -> Path:
    global TEMP_DIR
    if TEMP_DIR is None:
        if _SMART_REFRAME_TEMP:
            base = Path(_SMART_REFRAME_TEMP)
            base.mkdir(parents=True, exist_ok=True)
            TEMP_DIR = base
        else:
            TEMP_DIR = Path(tempfile.mkdtemp(prefix="smart_reframe_"))
    return TEMP_DIR


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float
    has_audio: bool
    codec: str


def probe_video(path: str) -> VideoMeta:
    """Read stream metadata using OpenCV (avoids ffprobe dependency)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(400, f"Cannot open video: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Check for audio by attempting to read the audio stream via ffmpeg
    has_audio = False
    codec = "unknown"
    cap.release()

    # Use ffmpeg -i to check for audio stream and codec
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    try:
        result = subprocess.run(
            [ffmpeg, "-i", path],
            capture_output=True, text=True, timeout=15,
        )
        stderr = result.stderr
        has_audio = "Audio:" in stderr
        for line in stderr.split("\n"):
            if "Stream" in line and "Video:" in line:
                # e.g. "Stream #0:0(und): Video: h264 (avc1 / 0x31637661)..."
                parts = line.split("Video: ")
                if len(parts) > 1:
                    codec = parts[1].split()[0].strip(",")
                    break
    except Exception:
        pass

    return VideoMeta(
        path=path,
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration=duration,
        has_audio=has_audio,
        codec=codec,
    )

    video_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video_stream is None:
        raise HTTPException(400, f"No video stream found in {path}")

    audio_stream = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)

    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    duration = float(video_stream.get("duration", 0) or 0)
    total_frames = int(video_stream.get("nb_frames", 0))
    if total_frames == 0 and duration > 0:
        total_frames = int(duration * fps)

    return VideoMeta(
        path=path,
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=fps,
        total_frames=total_frames,
        duration=duration,
        has_audio=audio_stream is not None,
        codec=video_stream.get("codec_name", "unknown"),
    )


# ---------------------------------------------------------------------------
# Face Tracking
# ---------------------------------------------------------------------------


class FaceTracker:
    """Adaptive exponential moving average for smooth crop centering.

    Larger face movements get a higher alpha (faster response) to reduce
    perceived lag, while small movements use the base alpha for smoothness.
    """

    def __init__(self, alpha: float = SMOOTHING_ALPHA, deadband: float = DEADBAND_PX):
        self.alpha = alpha
        self.deadband = deadband
        self._cx: Optional[float] = None
        self._cy: Optional[float] = None

    def reset(self) -> None:
        self._cx = self._cy = None

    def update(self, cx: float, cy: float) -> Tuple[float, float]:
        if self._cx is None:
            self._cx, self._cy = float(cx), float(cy)
            return self._cx, self._cy

        dx, dy = cx - self._cx, cy - self._cy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < self.deadband:
            return self._cx, self._cy

        if ADAPTIVE_TRACKING:
            # Scale alpha: small moves → base alpha, large moves → up to 0.8
            a = min(self.alpha + (dist / 300.0) * (0.8 - self.alpha), 0.8)
        else:
            a = self.alpha

        self._cx += a * dx
        self._cy += a * dy
        return self._cx, self._cy


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def _detect_primary_face(
    rgb_frame: np.ndarray,
    prev_bbox: Optional[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float, float, float]]:
    """
    Detect the most relevant face using OpenCV DNN SSD.

    Selection heuristic (in order):
      1. Confidence ≥ 0.7
      2. Prefer face closest to previous bounding box (temporal coherence)
      3. Tie-break: largest area
    """
    global _face_net
    if _face_net is None:
        _face_net = _init_face_detector()

    h, w = rgb_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(rgb_frame, 1.0, (300, 300), (104, 177, 123))
    _face_net.setInput(blob)
    detections = _face_net.forward()

    candidates = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.7:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        candidates.append((cx, cy, bw, bh, area))

    if not candidates:
        return None

    if prev_bbox is not None:
        px, py, _pbw, _pbh = prev_bbox
        candidates.sort(key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
    else:
        candidates.sort(key=lambda c: c[4], reverse=True)

    cx, cy, bw, bh, _area = candidates[0]
    return (cx, cy, bw, bh)


def _compute_crop(
    face_bbox: Optional[Tuple[float, float, float, float]],
    prev_bbox: Optional[Tuple[float, float, float, float]],
    lost_counter: int,
    frame_w: int,
    frame_h: int,
    target_w: int,
    target_h: int,
) -> Tuple[Tuple[int, int, int, int], int, Optional[Tuple[float, float, float, float]]]:
    crop_w = int(frame_h * target_w / target_h)
    crop_h = frame_h
    crop_w = min(crop_w, frame_w)

    if face_bbox is not None:
        lost_counter = 0
        cx, cy, bw, bh = face_bbox
        prev_bbox = (cx, cy, bw, bh)
    else:
        lost_counter += 1
        if prev_bbox is not None:
            cx, cy, _, _ = prev_bbox
        else:
            cx, cy = frame_w / 2, frame_h / 3

    if lost_counter >= LOST_FACE_TIMEOUT:
        cx, cy = frame_w / 2, frame_h / 2
        prev_bbox = None

    crop_x = int(cx - crop_w // 2)
    crop_x = max(0, min(frame_w - crop_w, crop_x))

    return (crop_x, 0, crop_w, crop_h), lost_counter, prev_bbox


def _slice_segment(input_path: str, start: float, end: float, output_path: str) -> dict:
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid interval: {start} -> {end}")

    logger.info("Slicing segment %.2f-%.2f from %s", start, end, input_path)
    t0 = time.perf_counter()

    slice_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-to", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "1",
        "-fflags", "+genpts",
        output_path,
    ]
    subprocess.run(slice_cmd, check=True, capture_output=True, text=True)

    meta = probe_video(output_path)
    logger.info(
        "Slice complete: %.1fs (%.1f MB) in %.2fs",
        meta.duration, Path(output_path).stat().st_size / 1e6, time.perf_counter() - t0,
    )
    return {"meta": meta, "slice_time": time.perf_counter() - t0}


def _compute_frame_crops(
    video_path: str,
    tracker: FaceTracker,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
) -> Tuple[list[Tuple[int, int, int, int]], float, float]:
    meta = probe_video(video_path)
    cap = cv2.VideoCapture(video_path)
    n_frames = 0
    n_detected = 0
    crops: list[Tuple[int, int, int, int]] = []
    prev_bbox: Optional[Tuple[float, float, float, float]] = None
    lost_counter = 0
    tracker.reset()
    t0 = time.perf_counter()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        n_frames += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        face_bbox = _detect_primary_face(frame_rgb, prev_bbox)
        if face_bbox is not None:
            n_detected += 1

        if face_bbox is not None:
            smoothed_cx, smoothed_cy = tracker.update(face_bbox[0], face_bbox[1])
            smoothed_bbox = (smoothed_cx, smoothed_cy, face_bbox[2], face_bbox[3])
        else:
            smoothed_cx, smoothed_cy = tracker.update(
                prev_bbox[0] if prev_bbox else meta.width / 2,
                prev_bbox[1] if prev_bbox else meta.height / 3,
            )
            smoothed_bbox = None

        crop_rect, lost_counter, prev_bbox = _compute_crop(
            smoothed_bbox, prev_bbox, lost_counter,
            meta.width, meta.height, target_w, target_h,
        )
        crops.append(crop_rect)

        if n_frames % 300 == 0:
            logger.info("  Face detection: %d / %d frames", n_frames, meta.total_frames)

    cap.release()
    elapsed = time.perf_counter() - t0
    detect_rate = n_detected / n_frames if n_frames > 0 else 0.0
    logger.info(
        "Face detection done: %d/%d frames (%.1f%%) in %.2fs",
        n_detected, n_frames, detect_rate * 100, elapsed,
    )
    return crops, detect_rate, elapsed


def _score_segments(
    video_path: str,
    window_sec: float = ENGAGEMENT_WINDOW,
) -> list[float]:
    """Score each *window_sec* block of the video by motion only (fast).

    Uses motion (frame diff) as a proxy for engagement — high motion
    correlates with interesting content. Downscales frames heavily for
    speed. Skips face detection (too slow on full-length video); faces
    are analyzed on the final clips.

    Returns a list of scores, one per *window_sec* block.
    """
    meta = probe_video(video_path)
    cap = cv2.VideoCapture(video_path)
    window_frames = int(meta.fps * window_sec)
    n_windows = max(1, math.ceil(meta.total_frames / window_frames))

    scores = [0.0] * n_windows
    prev_small: Optional[np.ndarray] = None
    frame_idx = 0
    win_idx = 0
    motion_acc = 0.0

    # Sample at ~3 fps, further downscale to 160px wide for speed
    step = max(1, int(meta.fps / 3))
    small_w = 160

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            small = cv2.resize(frame_bgr, (small_w, int(frame_bgr.shape[0] * small_w / frame_bgr.shape[1])))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_small is not None:
                diff = cv2.absdiff(gray, prev_small).mean()
                motion_acc += min(diff / 15.0, 1.0)
            prev_small = gray

        frame_idx += 1

        if frame_idx % window_frames == 0 or not ret:
            m = motion_acc / max(1, (frame_idx - (win_idx * window_frames)) // step)
            scores[win_idx] = m
            win_idx += 1
            motion_acc = 0.0

    cap.release()
    return scores


def _select_engaging_clips(
    scores: list[float],
    window_sec: float,
    fps: float,
    max_clips: int = ENGAGEMENT_TOP_N,
    min_clip: float = ENGAGEMENT_MIN_CLIP,
    max_clip: float = ENGAGEMENT_MAX_CLIP,
    gap: float = 5.0,
) -> list[dict]:
    """Pick up to *max_clips* non-overlapping segments with highest
    engagement scores.

    Each returned dict: ``{start, end, score}``.
    """
    if not scores:
        return []

    clip_windows = max(1, int(min_clip / window_sec))
    gap_windows = max(1, int(gap / window_sec))
    n = len(scores)

    # Sliding window average for smoother selection
    win_avgs = []
    for i in range(n - clip_windows + 1):
        s = sum(scores[i: i + clip_windows]) / clip_windows
        win_avgs.append((s, i))

    win_avgs.sort(key=lambda x: -x[0])
    used: list[range] = []
    selected: list[dict] = []

    for _score, start_win in win_avgs:
        start_sec = start_win * window_sec
        end_sec = (start_win + clip_windows) * window_sec
        end_sec = min(end_sec, start_sec + max_clip)

        # Check overlap
        sr = range(start_win - gap_windows, start_win + clip_windows + gap_windows)
        if any(sr.start < u.stop and u.start < sr.stop for u in used):
            continue

        used.append(range(start_win, start_win + clip_windows))
        selected.append({
            "start": round(start_sec, 1),
            "end": round(end_sec, 1),
            "score": round(_score / max(1, clip_windows), 3),
        })
        if len(selected) >= max_clips:
            break

    return sorted(selected, key=lambda x: x["start"])


def _build_caption_filter(
    caption_text: str,
    video_duration: float,
    fps: float,
    out_w: int,
    out_h: int,
) -> str:
    """Build an FFmpeg ``drawtext`` filter string for emoji caption overlay.

    The caption slides in, holds, and fades out — styled with an emoji prefix
    and a dark semi-transparent bar at the bottom of the frame.
    """
    if not caption_text:
        return ""

    font_size = max(22, min(34, out_h // 52))
    bar_h = font_size + 36

    # Check which fonts exist, fallback gracefully
    fonts_to_try = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    font_path = next((f for f in fonts_to_try if Path(f).is_file()), "")
    if not font_path:
        return ""

    # Bottom bar
    bar = (
        f"drawbox=x=0:y={out_h - bar_h}:w={out_w}:h={bar_h}:"
        f"color=black@0.45:t=fill,"
    )

    # Caption text centered in bar — strip emoji that Arial Bold can't render
    display_text = caption_text.replace("'", "’").replace(":", "\\:")
    # Remove common emoji and symbol ranges that aren't in Arial Bold
    display_text = re.sub(r'[\U0001F000-\U0001FFFF\u2600-\u27BF\u2B50\u2728\u2702-\u27B0\uFE00-\uFE0F]', '', display_text)
    if not display_text.strip():
        display_text = "Smart Reframe Clip"

    label = (
        f"drawtext=text='{display_text}':"
        f"fontfile={font_path}:fontsize={font_size}:"
        f"fontcolor=white:x=(w-text_w)/2:y={out_h - bar_h + (bar_h - font_size) // 2}:"
        f"shadowcolor=black@0.6:shadowx=2:shadowy=2"
    )

    return f"{bar}{label}"


def _apply_crops_render(
    input_path: str,
    output_path: str,
    crops: list[Tuple[int, int, int, int]],
    meta: VideoMeta,
    caption_text: str = "",
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
    denoise: bool = True,
) -> float:
    _cx, _cy, crop_w, crop_h = crops[0]
    out_w = target_w
    out_h = target_h

    logger.info(
        "Rendering %d frames (%dx%d → %dx%d) to %s%s",
        len(crops), crop_w, crop_h, out_w, out_h, output_path,
        "  [captions]" if caption_text else "",
    )
    t0 = time.perf_counter()

    cap = cv2.VideoCapture(input_path)

    # Build filter chain
    filters = f"scale={out_w}:{out_h}:flags=lanczos"
    if denoise:
        filters += f",{FFMPEG_DENOISE}"
    caption_filter = _build_caption_filter(
        caption_text, meta.duration, meta.fps, out_w, out_h,
    )
    if caption_filter:
        filters = filters + "," + caption_filter

    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{crop_w}x{crop_h}",
        "-r", str(meta.fps),
        "-i", "-",
        "-i", input_path,
        "-map", "0:v",
        "-pix_fmt", "yuv420p",
        "-vf", filters,
        "-shortest",
        output_path,
    ]

    if FFMPEG_VT_AVAIL:
        encode_cmd[-1:-1] = [
            "-c:v", "h264_videotoolbox",
            "-b:v", "5000k",
            "-movflags", "+faststart",
        ]
    else:
        encode_cmd[-1:-1] = [
            "-c:v", "libx264",
            "-profile:v", "high",
            "-preset", FFMPEG_PRESET,
            "-crf", str(FFMPEG_CRF),
            "-movflags", "+faststart",
        ]

    if meta.has_audio:
        encode_cmd[-1:-1] = [
            "-map", "1:a?",
            "-c:a", "aac",
            "-b:a", AUDIO_BITRATE,
        ]

    proc = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(crops):
                break
            x, y, w, h = crops[frame_idx]
            cropped = np.ascontiguousarray(frame[y: y + h, x: x + w])
            proc.stdin.write(cropped.tobytes())  # type: ignore[union-attr]
            frame_idx += 1
            if frame_idx % 300 == 0:
                logger.info("  Rendered %d / %d frames", frame_idx, len(crops))
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg encode failed (exit code {proc.returncode})")

    elapsed = time.perf_counter() - t0
    output_size = Path(output_path).stat().st_size
    logger.info(
        "Render complete: %.1f MB in %.2fs → %s",
        output_size / 1e6, elapsed, output_path,
    )
    return elapsed


def _process_single_clip(
    video_path: str,
    start: float,
    end: float,
    clip_index: int,
    output_dir: Path,
    caption_text: str = "",
    engagement_score: float = 0.0,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
    denoise: bool = True,
) -> ClipResult:
    clip_id = uuid.uuid4().hex[:12]
    sliced_path = str(output_dir / f"slice_{clip_id}.mp4")
    output_path = str(output_dir / f"clip_{clip_index:02d}_{clip_id}.mp4")

    pipe_start = time.perf_counter()

    try:
        slice_result = _slice_segment(video_path, start, end, sliced_path)
        meta = slice_result["meta"]

        crops, detect_rate, _detect_time = _compute_frame_crops(
            sliced_path, FaceTracker(), target_w, target_h,
        )

        if not crops:
            raise RuntimeError("No frames produced from sliced segment")

        _render_time = _apply_crops_render(
            sliced_path, output_path, crops, meta, caption_text,
            target_w, target_h, denoise,
        )

        total_time = time.perf_counter() - pipe_start
        logger.info(
            "Clip %d done: %.1fs pipeline, face=%.0f%% → %s",
            clip_index, total_time, detect_rate * 100, output_path,
        )

        return ClipResult(
            index=clip_index,
            interval={"start": start, "end": end},
            output_path=output_path,
            duration_sec=end - start,
            face_detection_rate=round(detect_rate, 3),
            pipeline_time_sec=round(total_time, 2),
            engagement_score=round(engagement_score, 3),
            caption_text=caption_text,
        )
    finally:
        try:
            Path(sliced_path).unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# HTML frontend (single-page app embedded)
# ---------------------------------------------------------------------------

INDEX_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Smart Reframe — Clip Extractor</title>
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {
  theme: {
    extend: {
      colors: {
        glass: { border: 'rgba(255,255,255,0.08)', light: 'rgba(255,255,255,0.06)', mid: 'rgba(255,255,255,0.10)' },
      }
    }
  }
}
</script>
<style>
  :root {
    --bg: radial-gradient(ellipse at 20% 50%, #0a1628 0%, #030712 100%);
    --card: rgba(255,255,255,0.04);
    --card-border: rgba(255,255,255,0.08);
    --card-hover: rgba(255,255,255,0.08);
    --text: #fff;
    --text-secondary: rgba(255,255,255,0.55);
    --text-tertiary: rgba(255,255,255,0.3);
    --input-bg: rgba(255,255,255,0.05);
    --input-border: rgba(255,255,255,0.10);
    --input-text: #e2e8f0;
    --divider: rgba(255,255,255,0.06);
    --tab-bg: rgba(255,255,255,0.05);
    --tab-active: rgba(255,255,255,0.10);
    --tab-text: rgba(255,255,255,0.40);
    --tab-text-active: #fff;
    --toggle-bg: rgba(255,255,255,0.05);
    --toggle-border: rgba(255,255,255,0.15);
    --scrollbar: rgba(255,255,255,0.1);
    --shimmer: linear-gradient(90deg, rgba(255,255,255,0.02) 25%, rgba(255,255,255,0.06) 50%, rgba(255,255,255,0.02) 75%);
    --dropzone-bg: rgba(255,255,255,0.04);
    --error-bg: rgba(239,68,68,0.10);
    --error-border: rgba(239,68,68,0.20);
    --error-text: #fca5a5;
    --success-text: #34d399;
    --info-bg: rgba(6,182,212,0.10);
    --info-border: rgba(6,182,212,0.20);
    --info-text: #67e8f9;
    --result-bg: rgba(255,255,255,0.04);
  }
  .light {
    --bg: radial-gradient(ellipse at 20% 50%, #e2e8f0 0%, #f1f5f9 100%);
    --card: rgba(255,255,255,0.70);
    --card-border: rgba(0,0,0,0.06);
    --card-hover: rgba(255,255,255,0.90);
    --text: #0f172a;
    --text-secondary: rgba(15,23,42,0.55);
    --text-tertiary: rgba(15,23,42,0.30);
    --input-bg: rgba(255,255,255,0.80);
    --input-border: rgba(0,0,0,0.12);
    --input-text: #1e293b;
    --divider: rgba(0,0,0,0.06);
    --tab-bg: rgba(0,0,0,0.04);
    --tab-active: #fff;
    --tab-text: rgba(0,0,0,0.35);
    --tab-text-active: #0f172a;
    --toggle-bg: #fff;
    --toggle-border: rgba(0,0,0,0.15);
    --scrollbar: rgba(0,0,0,0.10);
    --shimmer: linear-gradient(90deg, rgba(255,255,255,0.3) 25%, rgba(255,255,255,0.6) 50%, rgba(255,255,255,0.3) 75%);
    --dropzone-bg: rgba(255,255,255,0.50);
    --error-bg: rgba(239,68,68,0.06);
    --error-border: rgba(239,68,68,0.15);
    --error-text: #dc2626;
    --success-text: #059669;
    --info-bg: rgba(6,182,212,0.06);
    --info-border: rgba(6,182,212,0.15);
    --info-text: #0891b2;
    --result-bg: rgba(0,0,0,0.03);
  }
  * { scrollbar-width: thin; scrollbar-color: var(--scrollbar) transparent; }
  body {
    background: var(--bg);
    color: var(--text);
    transition: background .35s ease, color .35s ease;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
  }
  .glass {
    background: var(--card);
    backdrop-filter: blur(24px) saturate(1.4);
    -webkit-backdrop-filter: blur(24px) saturate(1.4);
    border: 1px solid var(--card-border);
    transition: background .35s ease, border-color .35s ease, box-shadow .2s ease;
  }
  .glass-hover:hover {
    background: var(--card-hover);
    border-color: var(--card-border);
  }
  .glass-input {
    background: var(--input-bg);
    border: 1px solid var(--input-border);
    color: var(--input-text);
    outline: none;
    transition: all .2s ease;
  }
  .glass-input:focus {
    border-color: rgba(6,182,212,0.5);
    box-shadow: 0 0 0 3px rgba(6,182,212,0.15);
    background: var(--card-hover);
  }
  .glass-input::placeholder { color: var(--text-tertiary); }
  .drop-zone { transition: all .15s ease; background: var(--dropzone-bg); }
  .drop-zone.dragover { border-color: rgba(6,182,212,0.6) !important; background: rgba(6,182,212,0.08) !important; }
  #progress-bar { transition: width .3s ease; }
  .clip-row:not(:last-child) { border-bottom: 1px solid var(--divider); }
  .glow-btn {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    transition: all .2s ease;
  }
  .glow-btn:hover:not(:disabled) {
    box-shadow: 0 0 24px rgba(6,182,212,0.35), 0 0 48px rgba(37,99,235,0.15);
    transform: translateY(-1px);
  }
  .glow-btn:disabled { opacity: 0.3; cursor: not-allowed; }
  .tab-btn { transition: all .2s ease; }
  .tab-btn.active { background: var(--tab-active); color: var(--tab-text-active); box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .tab-btn:not(.active) { color: var(--tab-text); }
  .tab-btn:not(.active):hover { color: var(--text); background: var(--tab-bg); }
  .format-btn { transition: all .2s ease; }
  .format-btn.active { background: rgba(6,182,212,0.15); color: #0891b2; border-color: rgba(6,182,212,0.25); }
  .format-btn:not(.active) { color: var(--text-secondary); }
  .format-btn:not(.active):hover { color: var(--text); background: var(--tab-bg); }
  .slim-scroll::-webkit-scrollbar { width: 4px; }
  .slim-scroll::-webkit-scrollbar-thumb { background: var(--scrollbar); border-radius: 2px; }
  @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
  .shimmer {
    background: var(--shimmer);
    background-size: 200% 100%;
    animation: shimmer 4s ease-in-out infinite;
  }
  @keyframes theme-spin {
    0% { transform: rotate(0deg) scale(1); }
    50% { transform: rotate(180deg) scale(0.8); }
    100% { transform: rotate(360deg) scale(1); }
  }
  .theme-spin { animation: theme-spin .5s ease; }
  @keyframes fade-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  .fade-in { animation: fade-in .3s ease; }
</style>
</head>
<body>
<div class="max-w-3xl mx-auto px-4 py-6 sm:py-10">
  <!-- Header with theme toggle -->
  <div class="flex items-center justify-between mb-8">
    <div class="text-center flex-1">
      <h1 class="text-3xl sm:text-4xl font-extrabold tracking-tight bg-gradient-to-r from-blue-500 via-cyan-400 to-teal-400 bg-clip-text text-transparent">Smart Reframe</h1>
      <p class="mt-1.5 text-sm" style="color:var(--text-secondary)">Upload a landscape video, pick timestamps, get vertical clips</p>
    </div>
    <button id="theme-toggle" type="button"
            class="shrink-0 p-2.5 rounded-xl glass hover:scale-105 active:scale-95 transition-all duration-200"
            style="margin-left:12px" title="Toggle theme">
      <svg id="theme-icon-sun" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="color:#f59e0b">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/>
      </svg>
      <svg id="theme-icon-moon" class="w-5 h-5 hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="color:#818cf8">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>
      </svg>
    </button>
  </div>

  <!-- Source card -->
  <div class="glass rounded-2xl p-5 sm:p-6 mb-5 fade-in">
    <div class="flex gap-1 mb-4" style="background:var(--tab-bg);border-radius:0.5rem;padding:3px;width:fit-content">
      <button id="tab-upload" class="tab-btn px-4 py-2 text-sm font-medium rounded-md active">Upload Video</button>
      <button id="tab-youtube" class="tab-btn px-4 py-2 text-sm font-medium rounded-md">YouTube URL</button>
    </div>

    <!-- Upload mode -->
    <div id="source-upload">
      <div id="drop-zone"
           class="drop-zone rounded-xl p-8 sm:p-10 text-center cursor-pointer"
           style="border:2px dashed var(--input-border)">
        <div id="upload-prompt">
          <svg class="mx-auto h-12 w-12" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="color:var(--text-tertiary)">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
          </svg>
          <p class="mt-3 text-sm" style="color:var(--text-secondary)">Drop an MP4 here or click to browse</p>
          <p class="text-xs mt-1" style="color:var(--text-tertiary)">Max 10 GB · 16:9 landscape recommended</p>
        </div>
        <div id="upload-status" class="hidden">
          <div class="flex items-center gap-3 justify-center">
            <svg class="animate-spin h-5 w-5" viewBox="0 0 24 24" style="color:#06b6d4">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            <span class="text-sm" style="color:var(--text-secondary)">Uploading…</span>
          </div>
        </div>
        <div id="upload-done" class="hidden">
          <p style="color:var(--success-text);font-weight:500">✓ Video uploaded</p>
          <p id="upload-filename" class="text-sm mt-1" style="color:var(--text-secondary)"></p>
        </div>
      </div>
      <input type="file" id="file-input" accept="video/mp4,video/quicktime,video/x-msvideo" class="hidden">
      <div id="upload-error" class="hidden mt-3 p-3 rounded-lg text-sm" style="background:var(--error-bg);border:1px solid var(--error-border);color:var(--error-text)"></div>
    </div>

    <!-- YouTube mode -->
    <div id="source-youtube" class="hidden">
      <label class="block text-sm font-medium mb-1.5" style="color:var(--text-secondary)">Paste a video URL</label>
      <input type="url" id="youtube-input" placeholder="https://youtube.com/watch?v=..."
             class="glass-input w-full rounded-xl px-4 py-3 text-sm">
      <p class="text-xs mt-2" style="color:var(--text-tertiary)">Supports YouTube, Twitch, Vimeo · Downloads best quality ≤1080p</p>
      <div id="youtube-status" class="hidden mt-3 p-3 rounded-lg text-sm" style="background:var(--info-bg);border:1px solid var(--info-border);color:var(--info-text)"></div>
    </div>
  </div>

  <!-- Clip intervals card -->
  <div class="glass rounded-2xl p-5 sm:p-6 mb-5 fade-in" style="animation-delay:0.05s">
    <div class="flex flex-wrap items-center justify-between gap-3 mb-3">
      <div class="flex items-center gap-3 sm:gap-4 flex-wrap">
        <h2 class="text-lg font-semibold" style="color:var(--text)">2. Clip Intervals</h2>
        <div class="flex rounded-lg overflow-hidden text-sm" style="border:1px solid var(--input-border)">
          <button type="button" id="format-portrait"
                  class="format-btn active px-3 py-1.5 font-medium">9:16</button>
          <button type="button" id="format-square"
                  class="format-btn px-3 py-1.5 font-medium">1:1</button>
        </div>
        <label class="flex items-center gap-1.5 text-sm cursor-pointer select-none" style="color:var(--text-secondary)">
          <input type="checkbox" id="denoise-toggle" checked
                 class="w-4 h-4 rounded" style="border-color:var(--toggle-border);background:var(--toggle-bg);color:#06b6d4">
          <span>Denoise</span>
        </label>
      </div>
      <label class="flex items-center gap-2 text-sm cursor-pointer select-none shrink-0" style="color:var(--text-secondary)">
        <input type="checkbox" id="smart-toggle"
               class="w-4 h-4 rounded" style="border-color:var(--toggle-border);background:var(--toggle-bg);color:#06b6d4">
        <span>🤖 Smart Mode</span>
      </label>
    </div>
    <p id="smart-info" class="hidden text-xs mb-3 p-2.5 rounded-lg" style="background:var(--info-bg);border:1px solid var(--info-border);color:var(--info-text)">
      🧠 Auto-finds the 3 most engaging parts using motion analysis + face detection
    </p>
    <div id="clips-container">
      <div class="clip-row flex gap-3 items-end pb-3 mb-3">
        <div class="flex-1">
          <label class="block text-xs font-medium mb-1" style="color:var(--text-secondary)">Start</label>
          <input type="number" step="0.1" min="0" value="0"
                 class="glass-input start w-full rounded-lg px-3 py-2 text-sm">
        </div>
        <div class="flex-1">
          <label class="block text-xs font-medium mb-1" style="color:var(--text-secondary)">End</label>
          <input type="number" step="0.1" min="0" value="30"
                 class="glass-input end w-full rounded-lg px-3 py-2 text-sm">
        </div>
        <button type="button" class="remove-clip p-2 transition-colors rounded-lg hover:bg-red-500/10" disabled style="color:rgba(239,68,68,0.5)">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
          </svg>
        </button>
      </div>
    </div>
    <div class="mt-3 flex flex-wrap items-center gap-2">
      <button id="add-clip" type="button"
              class="text-sm font-medium transition-colors" style="color:#06b6d4">+ Add clip</button>
      <span class="text-sm" style="color:var(--divider)">|</span>
      <span class="text-xs" style="color:var(--text-tertiary)">Batch:</span>
      <input type="number" id="batch-count" min="1" max="999" value="10"
             class="glass-input w-16 rounded px-2 py-1 text-sm text-center">
      <span class="text-xs" style="color:var(--text-tertiary)">clips ×</span>
      <input type="number" id="batch-dur" min="1" max="600" value="30"
             class="glass-input w-16 rounded px-2 py-1 text-sm text-center">
      <span class="text-xs" style="color:var(--text-tertiary)">s</span>
      <button id="fill-clips" type="button"
              class="text-sm glass glass-hover px-3 py-1 rounded-lg font-medium" style="color:var(--text-secondary)">Fill</button>
      <button id="add-n-clips" type="button"
              class="text-sm font-medium transition-colors" style="color:#06b6d4">+ Add N</button>
    </div>
  </div>

  <!-- Process button -->
  <button id="process-btn" type="button" disabled
          class="glow-btn w-full py-3.5 px-6 rounded-xl text-white font-semibold text-lg transition-all">
    Process Clips
  </button>

  <!-- Progress area -->
  <div id="progress-area" class="hidden mt-5">
    <div class="glass rounded-2xl p-5 sm:p-6 fade-in">
      <div class="flex justify-between text-sm mb-2">
        <span id="progress-label" class="font-medium" style="color:var(--text-secondary)">Processing…</span>
        <span id="progress-pct" style="color:var(--text-tertiary)">0%</span>
      </div>
      <div class="w-full rounded-full h-2.5 overflow-hidden" style="background:var(--input-bg)">
        <div id="progress-bar" class="h-2.5 rounded-full bg-gradient-to-r from-blue-500 to-cyan-400" style="width:0%"></div>
      </div>
      <pre id="progress-log" class="mt-4 text-xs rounded-lg p-3 max-h-40 overflow-y-auto font-mono slim-scroll" style="color:var(--text-tertiary);background:var(--input-bg)"></pre>
    </div>
  </div>

  <!-- Results area -->
  <div id="results-area" class="hidden mt-5">
    <div class="glass rounded-2xl p-5 sm:p-6 fade-in">
      <h2 class="text-lg font-semibold mb-4" style="color:var(--text)">Results</h2>
      <div id="results-list" class="space-y-3"></div>
    </div>
  </div>

  <!-- Error area -->
  <div id="error-area" class="hidden mt-5">
    <div class="rounded-xl p-4 fade-in" style="background:var(--error-bg);border:1px solid var(--error-border);color:var(--error-text)">
      <p class="font-semibold">Error</p>
      <p id="error-message" class="text-sm mt-1"></p>
    </div>
  </div>

  <!-- History -->
  <div id="history-area" class="mt-8">
    <div class="glass rounded-2xl p-5 sm:p-6">
      <div class="flex items-center justify-between mb-4">
        <h2 class="text-lg font-semibold" style="color:var(--text)">📋 History</h2>
        <button id="clear-history-btn" type="button"
                class="text-xs px-3 py-1.5 rounded-lg transition-all hover:scale-105"
                style="background:var(--error-bg);color:var(--error-text);border:1px solid var(--error-border)">Clear All</button>
      </div>
      <div id="history-list" class="space-y-2">
        <p class="text-sm text-center py-8" style="color:var(--text-tertiary)">No history yet — process some clips!</p>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <p class="text-center text-xs mt-8" style="color:var(--text-tertiary)">Smart Reframe · AI-powered vertical clip extraction</p>
</div>

<script>
// --- Theme toggle ---
(function() {
  const toggle = document.getElementById('theme-toggle');
  const sunIcon = document.getElementById('theme-icon-sun');
  const moonIcon = document.getElementById('theme-icon-moon');
  const stored = localStorage.getItem('theme');
  function setTheme(dark) {
    document.documentElement.classList.toggle('light', !dark);
    sunIcon.classList.toggle('hidden', !dark);
    moonIcon.classList.toggle('hidden', dark);
    localStorage.setItem('theme', dark ? 'dark' : 'light');
  }
  // Default: respect system preference
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (stored) setTheme(stored !== 'light');
  else setTheme(prefersDark);
  toggle.addEventListener('click', () => {
    const isDark = !sunIcon.classList.contains('hidden');
    setTheme(!isDark);
    toggle.classList.add('theme-spin');
    setTimeout(() => toggle.classList.remove('theme-spin'), 500);
  });
})();

const tabUpload = document.getElementById('tab-upload');
const tabYoutube = document.getElementById('tab-youtube');
const sourceUpload = document.getElementById('source-upload');
const sourceYoutube = document.getElementById('source-youtube');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const youtubeInput = document.getElementById('youtube-input');
const uploadPrompt = document.getElementById('upload-prompt');
const uploadStatus = document.getElementById('upload-status');
const uploadDone = document.getElementById('upload-done');
const uploadFilename = document.getElementById('upload-filename');
const uploadProgressText = document.getElementById('upload-progress-text');
const uploadError = document.getElementById('upload-error');
const youtubeStatus = document.getElementById('youtube-status');
const clipsContainer = document.getElementById('clips-container');
const addClipBtn = document.getElementById('add-clip');
const processBtn = document.getElementById('process-btn');
const smartToggle = document.getElementById('smart-toggle');
const smartInfo = document.getElementById('smart-info');
const formatBtns = document.querySelectorAll('.format-btn');
const formatPortrait = document.getElementById('format-portrait');
const formatSquare = document.getElementById('format-square');
const denoiseToggle = document.getElementById('denoise-toggle');
let selectedFormat = 'portrait';
const progressArea = document.getElementById('progress-area');
const progressBar = document.getElementById('progress-bar');
const progressPct = document.getElementById('progress-pct');
const progressLabel = document.getElementById('progress-label');
const progressLog = document.getElementById('progress-log');
const resultsArea = document.getElementById('results-area');
const resultsList = document.getElementById('results-list');
const errorArea = document.getElementById('error-area');
const errorMessage = document.getElementById('error-message');

let uploadId = null;
let youtubeUrl = null;
let sourceMode = 'upload';
let videoDuration = 0;

// --- Tab switching ---
function setMode(mode) {
  sourceMode = mode;
  document.querySelectorAll('.source-tab').forEach(t => {
    t.classList.remove('active');
  });
  if (mode === 'upload') {
    tabUpload.classList.add('active');
    sourceUpload.classList.remove('hidden');
    sourceYoutube.classList.add('hidden');
    processBtn.disabled = !uploadId;
  } else {
    tabYoutube.classList.add('active');
    sourceUpload.classList.add('hidden');
    sourceYoutube.classList.remove('hidden');
    processBtn.disabled = !youtubeUrl;
  }
}
tabUpload.addEventListener('click', () => setMode('upload'));
tabYoutube.addEventListener('click', () => setMode('youtube'));

// --- Drag / Drop ---
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

async function handleFile(file) {
  uploadError.classList.add('hidden');
  uploadPrompt.classList.add('hidden');
  uploadStatus.classList.remove('hidden');
  uploadDone.classList.add('hidden');
  processBtn.disabled = true;

  const form = new FormData();
  form.append('file', file);

  try {
    const resp = await fetch('/api/v1/upload', { method: 'POST', body: form });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Upload failed');

    uploadId = data.upload_id;
    youtubeUrl = null;
    uploadStatus.classList.add('hidden');
    uploadDone.classList.remove('hidden');
    uploadFilename.textContent = file.name + ' (' + (file.size / 1e6).toFixed(1) + ' MB)';
    autoSetClips('/api/v1/probe?upload_id=' + uploadId);
  } catch (err) {
    uploadStatus.classList.add('hidden');
    uploadPrompt.classList.remove('hidden');
    uploadError.classList.remove('hidden');
    uploadError.textContent = err.message;
  }
}

// --- YouTube URL input ---
youtubeInput.addEventListener('input', () => {
  const val = youtubeInput.value.trim();
  uploadError.classList.add('hidden');
  if (!val) {
    youtubeUrl = null;
    youtubeStatus.classList.add('hidden');
    processBtn.disabled = !uploadId;
    return;
  }
  if (!val.startsWith('http://') && !val.startsWith('https://')) {
    youtubeUrl = null;
    youtubeStatus.classList.add('hidden');
    processBtn.disabled = true;
    return;
  }
  // Quick client-side check for channel/playlist URLs
  if (/(youtube\.com\/(c\/|@|channel\/|playlist\?|user\/|feed\/|gaming))/.test(val)) {
    youtubeUrl = null;
    youtubeStatus.classList.remove('hidden');
    youtubeStatus.style.cssText = 'margin-top:12px;padding:12px;border-radius:8px;font-size:14px';
    youtubeStatus.style.background = 'rgba(239,68,68,0.08)';
    youtubeStatus.style.border = '1px solid rgba(239,68,68,0.15)';
    youtubeStatus.style.color = '#ef4444';
    youtubeStatus.textContent = 'That looks like a channel or playlist URL. Paste a specific video link.';
    processBtn.disabled = true;
    return;
  }
  youtubeUrl = val;
  uploadId = null;
  youtubeStatus.style.cssText = 'margin-top:12px;padding:12px;border-radius:8px;font-size:14px';
  youtubeStatus.style.background = 'var(--info-bg)';
  youtubeStatus.style.border = '1px solid var(--info-border)';
  youtubeStatus.style.color = 'var(--info-text)';
  youtubeStatus.classList.remove('hidden');
  youtubeStatus.textContent = '✓ URL set — probing...';
  processBtn.disabled = true;
  autoSetClips('/api/v1/probe?youtube_url=' + encodeURIComponent(val));
});

// --- Auto-set clip intervals for long videos ---
async function autoSetClips(probeUrl) {
  try {
    const resp = await fetch(probeUrl);
    if (!resp.ok) return;
    const info = await resp.json();
    videoDuration = info.duration;
    if (info.duration > 600) {
      const clipLen = 30;
      const p1 = Math.round(info.duration * 0.20 * 10) / 10;
      const p2 = Math.round(info.duration * 0.60 * 10) / 10;
      clipsContainer.innerHTML = '';
      addClipRow(p1, Math.round((p1 + clipLen) * 10) / 10);
      addClipRow(p2, Math.round((p2 + clipLen) * 10) / 10);
      youtubeStatus.textContent = `✓ ${(info.duration/60).toFixed(0)} min — 2 auto-clips added`;
    } else {
      youtubeStatus.textContent = '✓ URL set';
    }
  } catch (_) {
    youtubeStatus.textContent = '✓ URL set';
  } finally {
    processBtn.disabled = false;
  }
}

// --- Format toggle ---
function setFormat(fmt) {
  selectedFormat = fmt;
  formatBtns.forEach(b => b.classList.remove('active'));
  const btn = fmt === 'portrait' ? formatPortrait : formatSquare;
  btn.classList.add('active');
}
formatPortrait.addEventListener('click', () => setFormat('portrait'));
formatSquare.addEventListener('click', () => setFormat('square'));

// --- Smart mode toggle ---
let smartMode = false;
smartToggle.addEventListener('change', () => {
  smartMode = smartToggle.checked;
  smartInfo.classList.toggle('hidden', !smartMode);
  document.getElementById('clips-container').style.opacity = smartMode ? '0.4' : '1';
  document.getElementById('add-clip').style.display = smartMode ? 'none' : '';
  if (smartMode) {
    clipsContainer.querySelectorAll('.clip-row').forEach((r, i) => {
      if (i > 0) r.remove();
    });
  }
});

// --- Clip rows ---
function addClipRow(startVal, endVal) {
  const row = document.createElement('div');
  row.className = 'clip-row flex gap-3 items-end pb-3 mb-3';
  row.innerHTML = `
    <div class="flex-1">
      <label class="block text-xs font-medium mb-1" style="color:var(--text-secondary)">Start (s)</label>
      <input type="number" step="0.1" min="0" value="${startVal}"
             class="start glass-input w-full rounded-lg px-3 py-2 text-sm">
    </div>
    <div class="flex-1">
      <label class="block text-xs font-medium mb-1" style="color:var(--text-secondary)">End (s)</label>
      <input type="number" step="0.1" min="0" value="${endVal}"
             class="end glass-input w-full rounded-lg px-3 py-2 text-sm">
    </div>
    <button type="button" class="remove-clip p-2 transition-colors rounded-lg hover:bg-red-500/10" style="color:rgba(239,68,68,0.6)">
      <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/>
      </svg>
    </button>`;
  clipsContainer.appendChild(row);
  row.querySelector('.remove-clip').addEventListener('click', () => { row.remove(); });
  updateRemoveButtons();
}

addClipBtn.addEventListener('click', () => addClipRow(0, 30));

function updateRemoveButtons() {
  const btns = clipsContainer.querySelectorAll('.remove-clip');
  btns.forEach((btn, i) => btn.disabled = i === 0);
}

// --- Batch clip controls ---
const batchCount = document.getElementById('batch-count');
const batchDur = document.getElementById('batch-dur');
const fillClipsBtn = document.getElementById('fill-clips');
const addNClipsBtn = document.getElementById('add-n-clips');

fillClipsBtn.addEventListener('click', () => {
  const n = parseInt(batchCount.value) || 10;
  const dur = parseInt(batchDur.value) || 30;
  const total = n * dur;
  if (!videoDuration) {
    alert('Set a video source (upload or YouTube URL) first.');
    return;
  }
  if (total > videoDuration) {
    alert(`Need ${total}s but video is only ${Math.round(videoDuration)}s. Reduce clips or duration.`);
    return;
  }
  // Clear all existing clips
  clipsContainer.innerHTML = '';
  const gap = (videoDuration - total) / Math.max(1, n - 1);
  for (let i = 0; i < n; i++) {
    const start = Math.round(i * (dur + gap) * 10) / 10;
    const end = Math.round((start + dur) * 10) / 10;
    addClipRow(start, end);
  }
});

addNClipsBtn.addEventListener('click', () => {
  const n = parseInt(batchCount.value) || 10;
  const dur = parseInt(batchDur.value) || 30;
  // Find the last clip's end to append after it
  const rows = clipsContainer.querySelectorAll('.clip-row');
  let lastEnd = 0;
  rows.forEach(r => {
    const e = parseFloat(r.querySelector('.end').value);
    if (!isNaN(e) && e > lastEnd) lastEnd = e;
  });
  for (let i = 0; i < n; i++) {
    const start = Math.round((lastEnd + i * dur) * 10) / 10;
    const end = Math.round((start + dur) * 10) / 10;
    if (videoDuration && end > videoDuration) break;
    addClipRow(start, end);
  }
});

// --- Simulated progress animation ---
let _progressTimer = null;
function startProgressSim() {
  let pct = 0;
  const start = Date.now();
  // Fast initial ramp, then slow asymptotic approach to ~90%
  function tick() {
    const elapsed = (Date.now() - start) / 1000;
    // Sigmoid-like: quick to 30%, then slow approach to 90%
    const target = Math.min(90, 30 * (1 - Math.exp(-elapsed * 0.15)) + 60 * (1 - Math.exp(-elapsed * 0.025)));
    pct = Math.max(pct, Math.round(target));
    progressBar.style.width = pct + '%';
    progressPct.textContent = pct + '%';
    _progressTimer = setTimeout(tick, 400);
  }
  tick();
}
function stopProgressSim(full) {
  if (_progressTimer) { clearTimeout(_progressTimer); _progressTimer = null; }
  if (full) {
    progressBar.style.width = '100%';
    progressPct.textContent = '100%';
  }
}

// --- Process ---
processBtn.addEventListener('click', async () => {
  if (!uploadId && !youtubeUrl) return;

  errorArea.classList.add('hidden');
  resultsArea.classList.add('hidden');
  progressArea.classList.remove('hidden');
  progressBar.style.width = '0%';
  progressPct.textContent = '0%';
  progressLog.textContent = '';
  processBtn.disabled = true;

  const clips = [];
  document.querySelectorAll('.clip-row').forEach(row => {
    const start = parseFloat(row.querySelector('.start').value);
    const end = parseFloat(row.querySelector('.end').value);
    if (!isNaN(start) && !isNaN(end) && end > start) clips.push({ start, end });
  });

  if (!clips.length) {
    showError('No valid clip intervals.');
    processBtn.disabled = false;
    return;
  }

  const logEvent = (msg) => {
    progressLog.textContent += msg + '\n';
    progressLog.scrollTop = progressLog.scrollHeight;
  };
  logEvent('Queuing job...');

  const form = new FormData();
  if (uploadId) {
    form.append('upload_id', uploadId);
    logEvent('Source: uploaded video');
  } else {
    form.append('youtube_url', youtubeUrl);
    logEvent('Downloading: ' + youtubeUrl);
  }
  form.append('clips', JSON.stringify(clips));
  if (smartMode) form.append('smart', '1');
  form.append('format', selectedFormat);
  form.append('denoise', denoiseToggle.checked ? '1' : '0');

  startProgressSim();

  try {
    const resp = await fetch('/api/v1/process', { method: 'POST', body: form });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Processing failed');

    stopProgressSim(true);
    progressLabel.textContent = 'Complete';
    logEvent('Done in ' + data.total_time_sec + 's');

    // Save to history
    data.job_id = data.job_id || '';
    const history = JSON.parse(localStorage.getItem('clip_history') || '[]');
    history.unshift({
      job_id: data.job_id,
      time: Date.now(),
      total: data.total_time_sec,
      source: youtubeUrl || uploadId || '',
      caption: ((data.clips[0] || {}).caption_text || '').trim().slice(0, 60) || 'Clips',
      clips: data.clips,
    });
    localStorage.setItem('clip_history', JSON.stringify(history.slice(0, 50)));
    renderHistory();

    setTimeout(() => {
      progressArea.classList.add('hidden');
      showResults(data.clips, data.total_time_sec);
      processBtn.disabled = false;
    }, 600);
  } catch (err) {
    stopProgressSim(false);
    progressArea.classList.add('hidden');
    showError(err.message);
    processBtn.disabled = false;
  }
});

function showResults(clips, totalTime) {
  resultsArea.classList.remove('hidden');
  resultsList.innerHTML = '';
  clips.forEach((clip, i) => {
    const div = document.createElement('div');
    div.className = 'flex items-center justify-between p-4 rounded-xl glass-hover fade-in';
    div.style.background = 'var(--result-bg)';
    div.innerHTML = `
      <div class="min-w-0 mr-3">
        <p class="font-medium" style="color:var(--text)">${clip.caption_text ? '✨ ' : ''}Clip ${i + 1}</p>
        <p class="text-sm" style="color:var(--text-secondary)">${clip.interval.start}s → ${clip.interval.end}s · ${clip.duration_sec.toFixed(1)}s · face ${(clip.face_detection_rate * 100).toFixed(0)}%
          ${clip.engagement_score ? ' · 🎯 engagement ' + (clip.engagement_score * 100).toFixed(0) + '%' : ''}
          · ${clip.pipeline_time_sec.toFixed(1)}s</p>
        ${clip.caption_text ? '<p class="text-xs mt-0.5" style="color:var(--text-tertiary)">' + clip.caption_text + '</p>' : ''}
      </div>
      <a href="${clip.download_url}" download
         class="shrink-0 px-4 py-2 rounded-lg text-sm font-medium transition-all hover:scale-105 active:scale-95"
         style="background:linear-gradient(135deg,#2563eb,#06b6d4);color:white">
        Download
      </a>`;
    resultsList.appendChild(div);
  });
  resultsList.innerHTML += `<p class="text-sm text-center pt-2" style="color:var(--text-tertiary)">Total: ${totalTime.toFixed(1)}s</p>`;
}

function showError(msg) {
  errorArea.classList.remove('hidden');
  errorMessage.textContent = msg;
}

// --- History ---
function renderHistory() {
  const list = document.getElementById('history-list');
  const history = JSON.parse(localStorage.getItem('clip_history') || '[]');
  if (!history.length) {
    list.innerHTML = '<p class="text-sm text-center py-8" style="color:var(--text-tertiary)">No history yet — process some clips!</p>';
    return;
  }
  list.innerHTML = history.map((h, idx) => {
    const date = new Date(h.time).toLocaleString();
    const clipCount = h.clips ? h.clips.length : 0;
    return `
      <div class="glass-hover rounded-xl p-4 fade-in" style="background:var(--result-bg)">
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0 flex-1">
            <p class="text-sm font-medium truncate" style="color:var(--text)">${h.caption}</p>
            <p class="text-xs mt-0.5" style="color:var(--text-tertiary)">${date} · ${clipCount} clips · ${h.total.toFixed(1)}s</p>
            ${h.source ? `<p class="text-xs truncate" style="color:var(--text-tertiary)">${h.source}</p>` : ''}
          </div>
          <div class="flex gap-2 shrink-0">
            <button onclick="deleteHistoryItem(${idx})"
                    class="text-xs px-3 py-1.5 rounded-lg transition-all hover:scale-105"
                    style="background:var(--error-bg);color:var(--error-text);border:1px solid var(--error-border)">✕</button>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function deleteHistoryItem(idx) {
  const history = JSON.parse(localStorage.getItem('clip_history') || '[]');
  const entry = history[idx];
  if (entry && entry.job_id) {
    fetch('/api/v1/history/' + entry.job_id, { method: 'DELETE' }).catch(() => {});
  }
  history.splice(idx, 1);
  localStorage.setItem('clip_history', JSON.stringify(history));
  renderHistory();
}

document.getElementById('clear-history-btn').addEventListener('click', () => {
  if (!confirm('Clear all history?')) return;
  const history = JSON.parse(localStorage.getItem('clip_history') || '[]');
  history.forEach(h => {
    if (h.job_id) fetch('/api/v1/history/' + h.job_id, { method: 'DELETE' }).catch(() => {});
  });
  localStorage.removeItem('clip_history');
  renderHistory();
});

// Load history on page load
renderHistory();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smart Reframe API",
    version="1.1.0",
    description="AI-powered 9:16 vertical clip extraction from horizontal video.",
)
application = app


@app.on_event("startup")
def _startup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _ensure_ffmpeg()
    _ensure_yt_dlp()
    os.makedirs(_get_temp_dir(), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _init_face_detector()
    logger.info("OpenCV DNN face detector loaded")
    logger.info("Temp dir: %s", _get_temp_dir())
    logger.info("Model dir: %s", MODEL_DIR)

    # Detect hardware encoder availability
    global FFMPEG_VT_AVAIL
    try:
        r = subprocess.run(
            [str(BIN_DIR / "ffmpeg"), "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        FFMPEG_VT_AVAIL = "h264_videotoolbox" in r.stdout
        logger.info(
            "Hardware encoder (h264_videotoolbox): %s",
            "available" if FFMPEG_VT_AVAIL else "not available",
        )
    except Exception:
        FFMPEG_VT_AVAIL = False


@app.on_event("shutdown")
def _shutdown():
    if TEMP_DIR is not None:
        try:
            shutil.rmtree(str(TEMP_DIR), ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------


@app.post("/api/v1/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file. Returns an upload_id for later processing."""
    if not file.filename or not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(400, "Unsupported file type. Expected .mp4, .mov, .avi, .mkv, or .webm")

    upload_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix or ".mp4"
    dest = _get_temp_dir() / f"upload_{upload_id}{ext}"

    logger.info("Receiving upload %s (%s)", upload_id, file.filename)
    t0 = time.perf_counter()

    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, "File too large (max 10 GB)")
            f.write(chunk)

    elapsed = time.perf_counter() - t0
    logger.info("Upload %s complete: %.1f MB in %.2fs", upload_id, size / 1e6, elapsed)

    UPLOAD_STORE[upload_id] = dest
    return {"upload_id": upload_id, "filename": file.filename, "size_bytes": size}


# ---------------------------------------------------------------------------
# Process endpoint (multipart: upload_id + clips JSON)
# ---------------------------------------------------------------------------


@app.post("/api/v1/process", response_model=ReframeResponse)
async def process_clips(
    background_tasks: BackgroundTasks,
    upload_id: str = Form(""),
    youtube_url: str = Form(""),
    clips: str = Form(...),
    smart: str = Form(""),
    format: str = Form("portrait"),
    denoise: str = Form("1"),
):
    """
    Process video with given clip intervals.

    Provide one of:
    - upload_id: from the /upload endpoint (file upload)
    - youtube_url: a YouTube / Twitch / Vimeo URL to download first

    clips: JSON array of {start, end}
    format: "portrait" (9:16, 1080x1920) or "square" (1:1, 1080x1080)
    denoise: "1" to apply light denoising, "0" to skip
    """
    # Resolve video path — either upload or YouTube download.
    job_id = uuid.uuid4().hex[:8]
    work_dir = _get_temp_dir() / f"job_{job_id}"         # temp working dir (cleaned up)
    output_dir = OUTPUT_DIR / f"job_{job_id}"             # persistent output dir
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    video_path: Optional[Path] = None

    if youtube_url:
        try:
            video_path = _download_youtube(youtube_url, work_dir)
        except Exception as exc:
            raise HTTPException(400, f"YouTube download failed: {exc}")
    elif upload_id:
        video_path = UPLOAD_STORE.get(upload_id)
        if video_path is None:
            raise HTTPException(404, f"Upload {upload_id} not found (expired or invalid)")
    else:
        raise HTTPException(400, "Provide either upload_id or youtube_url")

    # Validate video
    try:
        meta = probe_video(str(video_path))
    except Exception as exc:
        raise HTTPException(400, f"Cannot probe video: {exc}")

    # Smart mode: auto-select exactly 3 engaging clips
    is_smart = smart.lower() in ("1", "true", "yes")
    if is_smart:
        logger.info(
            "Smart mode: analyzing %s for engagement (%.1fs)",
            youtube_url or upload_id, meta.duration,
        )
        scores = _score_segments(str(video_path))
        smart_clips = _select_engaging_clips(
            scores, ENGAGEMENT_WINDOW, meta.fps, max_clips=ENGAGEMENT_TOP_N,
        )
        if not smart_clips:
            raise HTTPException(400, "Could not find engaging segments in video")
        clip_intervals = [ClipInterval(**c) for c in smart_clips]
        logger.info(
            "Smart mode: selected %d engaging clips", len(clip_intervals),
        )
    else:
        # Parse intervals from user input
        try:
            intervals = json.loads(clips)
            clip_intervals = [ClipInterval(**c) for c in intervals]
        except Exception as exc:
            raise HTTPException(400, f"Invalid clips JSON: {exc}")

        if not clip_intervals:
            raise HTTPException(400, "At least one clip interval required")

    for ci in clip_intervals:
        if ci.end > meta.duration:
            raise HTTPException(
                400,
                f"Interval [{ci.start}, {ci.end}] exceeds video duration ({meta.duration:.1f}s)",
            )

    logger.info(
        "Processing %d clips from %s (%dx%d @ %.2f fps, %.1fs)",
        len(clip_intervals), youtube_url or upload_id, meta.width, meta.height,
        meta.fps, meta.duration,
    )

    # Get video title for auto-captions
    video_title = _get_video_title(youtube_url) if youtube_url else ""

    # Determine target dimensions from format
    do_denoise = denoise.lower() in ("1", "true", "yes")
    if format == "square":
        target_w, target_h = 1080, 1080
    else:
        target_w, target_h = TARGET_WIDTH, TARGET_HEIGHT

    # Process each clip
    t_start = time.perf_counter()
    results: list[ClipResult] = []

    for i, interval in enumerate(clip_intervals):
        try:
            # Build caption from video title
            if video_title:
                caption_text = video_title
            else:
                caption_text = "Smart Reframe Clip"
            eng_score = getattr(interval, "score", 0.0) if is_smart else 0.0
            result = _process_single_clip(
                str(video_path), interval.start, interval.end, i, work_dir,
                caption_text=caption_text, engagement_score=eng_score,
                target_w=target_w, target_h=target_h, denoise=do_denoise,
            )
            # Move output to persistent dir
            src = Path(result.output_path)
            dst = output_dir / src.name
            shutil.move(str(src), str(dst))
            result.output_path = str(dst)
            # Build download URL
            filename = dst.name
            result.download_url = f"/api/v1/output/{job_id}/{filename}"
            results.append(result)
            logger.info(
                "Clip %d done in %.2fs (detection rate: %.0f%%)",
                i, result.pipeline_time_sec, result.face_detection_rate * 100,
            )
        except Exception as exc:
            logger.exception("Clip %d failed: %s", i, exc)
            raise HTTPException(500, f"Clip {i} processing failed: {exc}")

    total = time.perf_counter() - t_start

    # Cleanup working dir (deletes downloaded video, slices, etc.)
    background_tasks.add_task(_cleanup_later, work_dir)
    background_tasks.add_task(_cleanup_upload, upload_id)

    return ReframeResponse(
        status="success", job_id=job_id, clips=results, total_time_sec=round(total, 2),
    )


# ---------------------------------------------------------------------------
# Download endpoint
# ---------------------------------------------------------------------------


@app.get("/api/v1/output/{job_id}/{filename}")
async def download_clip(job_id: str, filename: str):
    """Serve a rendered clip file."""
    if not re.match(r"^[\w\-]+\.mp4$", filename):
        raise HTTPException(400, "Invalid filename")
    filepath = OUTPUT_DIR / f"job_{job_id}" / filename
    if not filepath.is_file():
        raise HTTPException(404, "File not found or expired")
    return FileResponse(
        str(filepath),
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/v1/history")
async def list_history():
    """List all job output directories with clip info."""
    if not OUTPUT_DIR.is_dir():
        return {"jobs": []}
    jobs: list[dict] = []
    for job_dir in sorted(OUTPUT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not job_dir.is_dir() or not job_dir.name.startswith("job_"):
            continue
        clips = sorted(job_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        jobs.append({
            "job_id": job_dir.name.replace("job_", "", 1),
            "created": job_dir.stat().st_mtime,
            "clips": [{
                "filename": c.name,
                "size_mb": round(c.stat().st_size / 1e6, 1),
                "url": f"/api/v1/output/{job_dir.name.replace('job_', '', 1)}/{c.name}",
            } for c in clips],
        })
    return {"jobs": jobs}


@app.delete("/api/v1/history/{job_id}")
async def delete_history(job_id: str):
    """Delete a job's output directory."""
    if not re.match(r"^[\w-]+$", job_id):
        raise HTTPException(400, "Invalid job_id")
    job_dir = OUTPUT_DIR / f"job_{job_id}"
    if job_dir.is_dir():
        shutil.rmtree(str(job_dir), ignore_errors=True)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Reframe endpoint (JSON body, local file path — original)
# ---------------------------------------------------------------------------


@app.post("/api/v1/reframe", response_model=ReframeResponse)
async def reframe_clips(
    body: ReframeRequest,
    background_tasks: BackgroundTasks,
):
    t_start = time.perf_counter()
    output_dir = _get_temp_dir() / f"job_{uuid.uuid4().hex[:8]}"
    os.makedirs(output_dir, exist_ok=True)

    if not Path(body.video_path).is_file():
        raise HTTPException(400, f"Video file not found: {body.video_path}")

    try:
        meta = probe_video(body.video_path)
    except Exception as exc:
        raise HTTPException(400, f"Cannot probe video: {exc}")

    for ci in body.clips:
        if ci.end > meta.duration:
            raise HTTPException(
                400,
                f"Interval [{ci.start}, {ci.end}] exceeds video duration ({meta.duration:.1f}s)",
            )

    results: list[ClipResult] = []
    for i, interval in enumerate(body.clips):
        try:
            t_clip = time.perf_counter()
            result = _process_single_clip(
                body.video_path, interval.start, interval.end, i, output_dir,
            )
            filename = Path(result.output_path).name
            result.download_url = f"/api/v1/output/{output_dir.name}/{filename}"
            result.pipeline_time_sec = round(time.perf_counter() - t_clip, 2)
            results.append(result)
        except Exception as exc:
            logger.exception("Clip %d failed: %s", i, exc)
            raise HTTPException(500, f"Clip {i} processing failed: {exc}")

    total = time.perf_counter() - t_start
    background_tasks.add_task(_cleanup_later, output_dir)

    return ReframeResponse(status="success", clips=results, total_time_sec=round(total, 2))


# ---------------------------------------------------------------------------
# Health + Frontend
# ---------------------------------------------------------------------------


@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}


@app.get("/api/v1/probe")
async def probe_source(upload_id: str = "", youtube_url: str = ""):
    """Return metadata (duration, width, height, fps) for an uploaded file
    or a YouTube URL (probed via yt-dlp, no download)."""
    if upload_id:
        path = UPLOAD_STORE.get(upload_id)
        if path is None:
            raise HTTPException(404, "Upload not found")
        meta = probe_video(str(path))
        return {
            "duration": meta.duration,
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "filename": path.name,
        }
    elif youtube_url:
        if not YT_DLP_CMD:
            _ensure_yt_dlp()
        try:
            duration = _probe_youtube_duration(youtube_url)
        except Exception as exc:
            raise HTTPException(400, f"Cannot probe YouTube URL: {exc}")
        return {
            "duration": duration,
            "width": 0,
            "height": 0,
            "fps": 0,
            "filename": youtube_url,
        }
    else:
        raise HTTPException(400, "Provide upload_id or youtube_url")


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------


def _cleanup_later(path: Path, delay: float = 300.0):
    time.sleep(delay)
    try:
        shutil.rmtree(str(path), ignore_errors=True)
    except Exception:
        pass


def _cleanup_upload(upload_id: str):
    path = UPLOAD_STORE.pop(upload_id, None)
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 4:
        print("Usage: python smart_reframe.py <video.mp4> <start> <end> [<start2> <end2> ...]")
        sys.exit(1)

    video_path = sys.argv[1]
    args = sys.argv[2:]
    clips = []
    for i in range(0, len(args), 2):
        if i + 1 >= len(args):
            break
        clips.append(ClipInterval(start=float(args[i]), end=float(args[i + 1])))

    if not clips:
        print("No valid intervals provided.")
        sys.exit(1)

    output_dir = _get_temp_dir() / "cli"
    os.makedirs(output_dir, exist_ok=True)

    for idx, clip in enumerate(clips):
        print(f"\n=== Processing clip {idx}: [{clip.start} → {clip.end}] ===")
        result = _process_single_clip(video_path, clip.start, clip.end, idx, output_dir)
        print(f"  Output: {result.output_path}")
        print(f"  Face detection rate: {result.face_detection_rate*100:.0f}%")
        print(f"  Time: {result.pipeline_time_sec:.2f}s")


if __name__ == "__main__":
    main()
