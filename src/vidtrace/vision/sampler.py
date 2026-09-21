"""
Adaptive frame sampling for video analysis.

Reads frames sequentially (avoiding expensive random seeks on compressed
recordings) and dynamically switches between normal and dense sampling
based on detected visual changes.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import cv2
import numpy as np


def frame_signature(frame: np.ndarray) -> np.ndarray:
    """
    Compute a compact visual signature of a frame.

    Downsizes to 96×54, converts to grayscale, applies slight blur,
    and normalizes to [0, 1] float. Used for fast frame comparison.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
    resized = cv2.GaussianBlur(resized, (3, 3), 0)
    return resized.astype(np.float32) / 255.0


def frame_difference(
    a: np.ndarray | None,
    b: np.ndarray | None,
) -> float:
    """
    Compute mean absolute difference between two frame signatures.

    Returns 1.0 if either signature is None (e.g., first frame).
    """
    if a is None or b is None:
        return 1.0
    return float(np.mean(np.abs(a - b)))


def adaptive_frame_sampler(
    cap: Any,
    fps: float,
    duration: float,
    normal_interval: float,
    active_interval: float,
    active_window: float,
    scene_threshold: float,
) -> Generator[tuple[int, float, np.ndarray, float], None, None]:
    """
    Sequential frame decoder with adaptive sampling.

    This reads frames continuously but only yields frames at the current
    sampling target. When visual changes are detected above the threshold,
    it switches to dense sampling for `active_window` seconds.

    Yields:
        (frame_index, timestamp, frame, visual_difference)
    """
    if fps <= 0:
        fps = 30.0

    next_sample = 0.0
    active_until = -1.0
    last_signature = None
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_index / fps
        frame_index += 1

        if timestamp + 0.01 < next_sample:
            continue

        signature = frame_signature(frame)
        difference = frame_difference(last_signature, signature)
        last_signature = signature

        # Visual change detected — switch to dense sampling.
        if difference >= scene_threshold:
            active_until = max(active_until, timestamp + active_window)

        yield (frame_index - 1, timestamp, frame, difference)

        # Choose next interval based on activity state.
        interval = active_interval if timestamp < active_until else normal_interval

        next_sample = timestamp + interval

        if timestamp >= duration:
            break
