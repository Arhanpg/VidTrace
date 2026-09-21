"""
Image preprocessing for OCR enhancement.

When the initial OCR pass returns low-confidence results, these
variants are tried to improve text detection on difficult frames.
"""

from __future__ import annotations

import cv2
import numpy as np


def prepare_image(image: np.ndarray, variant: str) -> np.ndarray:
    """
    Apply an image preprocessing variant for OCR retry.

    Args:
        image: BGR image from OpenCV.
        variant: One of "original", "clahe", "sharpen", "threshold".

    Returns:
        Preprocessed BGR image.
    """
    if variant == "original":
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if variant == "clahe":
        return _clahe(gray)
    elif variant == "sharpen":
        return _sharpen(gray)
    elif variant == "threshold":
        return _adaptive_threshold(gray)
    else:
        raise ValueError(f"Unknown OCR variant: {variant}")


def _clahe(gray: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement with slight upscale."""
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.resize(
        enhanced, None,
        fx=1.35, fy=1.35,
        interpolation=cv2.INTER_CUBIC,
    )
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp mask sharpening with slight upscale."""
    resized = cv2.resize(
        gray, None,
        fx=1.25, fy=1.25,
        interpolation=cv2.INTER_CUBIC,
    )
    blur = cv2.GaussianBlur(resized, (0, 0), 1.2)
    sharpened = cv2.addWeighted(resized, 1.7, blur, -0.7, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """Adaptive Gaussian thresholding with slight upscale."""
    resized = cv2.resize(
        gray, None,
        fx=1.25, fy=1.25,
        interpolation=cv2.INTER_CUBIC,
    )
    thresholded = cv2.adaptiveThreshold(
        resized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11,
    )
    return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)


# Enhancement variants tried when confidence is low.
ENHANCEMENT_VARIANTS = ["clahe", "sharpen", "threshold"]
