"""
GPU detection, memory management, and video metadata reading.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import cv2

from vidtrace.models.events import VideoInfo

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────
# GPU memory management
# ───────────────────────────────────────────────────────────────

def release_gpu_memory() -> None:
    """
    Release GPU memory after a stage completes.

    Tries both PyTorch (for Whisper/CTranslate2) and PaddlePaddle.
    All exceptions are silently caught — this is best-effort cleanup.
    """
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    try:
        import paddle
        if paddle.is_compiled_with_cuda():
            with contextlib.suppress(Exception):
                paddle.device.cuda.empty_cache()
    except Exception:
        pass


# ───────────────────────────────────────────────────────────────
# GPU detection
# ───────────────────────────────────────────────────────────────

def detect_gpu() -> dict[str, Any]:
    """
    Detect available GPU capabilities.

    Checks nvidia-smi, CTranslate2 CUDA, and PaddlePaddle CUDA.
    """
    result: dict[str, Any] = {
        "nvidia_smi_available": False,
        "gpu_name": None,
        "faster_whisper_cuda_available": False,
        "paddle_cuda_available": False,
    }

    # ── NVIDIA-SMI ────────────────────────────────────────────
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if process.returncode == 0:
            result["nvidia_smi_available"] = True
            output = process.stdout.strip()
            if output:
                result["gpu_name"] = output.splitlines()[0]

    except Exception:
        pass

    # ── CTranslate2 ───────────────────────────────────────────
    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")
        if supported:
            result["faster_whisper_cuda_available"] = True
    except Exception:
        pass

    # ── PaddlePaddle ──────────────────────────────────────────
    try:
        import paddle
        result["paddle_cuda_available"] = bool(
            paddle.is_compiled_with_cuda()
        )
    except Exception:
        pass

    return result


# ───────────────────────────────────────────────────────────────
# Video metadata
# ───────────────────────────────────────────────────────────────

def read_video_info(video_path: Path) -> VideoInfo:
    """
    Read metadata from a video file using OpenCV and optionally ffprobe.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    duration = (
        frame_count / fps
        if fps > 0 and frame_count > 0
        else 0.0
    )

    cap.release()

    codec = "unknown"

    # Try ffprobe if available.
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,duration",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])

            if streams:
                stream = streams[0]
                codec = str(stream.get("codec_name") or "unknown")

                if duration <= 0:
                    duration = float(stream.get("duration") or 0)

    except Exception:
        pass

    return VideoInfo(
        path=str(video_path.resolve()),
        name=video_path.name,
        duration_seconds=duration,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        codec=codec,
    )
