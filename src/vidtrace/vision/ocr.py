"""
PaddleOCR engine with automatic enhancement retries.

Handles OCR result parsing across PaddleOCR versions, bounding box
conversion (polygons → xyxy), reading-order sorting, and multi-variant
retry when initial OCR confidence is low.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np

from vidtrace.models.events import OCRBox
from vidtrace.pipeline.gpu import release_gpu_memory
from vidtrace.vision.preprocessing import ENHANCEMENT_VARIANTS, prepare_image

logger = logging.getLogger(__name__)

# Don't waste OCR on essentially empty frames.
DEFAULT_MIN_OCR_CHARS = 4

# Weak OCR triggers enhanced retry.
DEFAULT_RETRY_CONFIDENCE = 0.62


class PaddleOCREngine:
    """
    PaddleOCR wrapper with automatic enhancement retries.

    When the initial OCR pass returns low-confidence or too few characters,
    the engine retries with CLAHE, sharpening, and adaptive threshold
    variants, selecting the best result.
    """

    def __init__(
        self,
        prefer_gpu: bool = True,
        min_ocr_chars: int = DEFAULT_MIN_OCR_CHARS,
        retry_confidence: float = DEFAULT_RETRY_CONFIDENCE,
    ):
        from paddleocr import PaddleOCR

        self.min_ocr_chars = min_ocr_chars
        self.retry_confidence = retry_confidence
        self.device = "cpu"

        try:
            import paddle
            if prefer_gpu and paddle.is_compiled_with_cuda():
                self.device = "gpu:0"
        except Exception:
            pass

        logger.info("PaddleOCR | device=%s", self.device)

        # PaddleOCR 3.x configuration.
        # Orientation/unwarping disabled: lecture recordings contain flat
        # desktop screenshots, not rotated documents.
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="paddle",
            device=self.device,
            # Higher resolution helps with source-code text.
            text_det_limit_side_len=2400,
            # Lower thresholds increase recall.
            text_det_thresh=0.20,
            text_det_box_thresh=0.45,
            text_rec_score_thresh=0.20,
        )

    # ───────────────────────────────────────────────────────────
    # Result parsing
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def get_result_dict(result_obj: Any) -> dict:
        """Parse PaddleOCR result into a stable dictionary format."""
        # PaddleOCR 3.x commonly exposes .json
        try:
            value = getattr(result_obj, "json", None)
            if callable(value):
                value = value()
            if isinstance(value, str):
                value = json.loads(value)
            if isinstance(value, dict):
                if "res" in value and isinstance(value["res"], dict):
                    return value["res"]
                return value
        except Exception:
            pass

        # Some environments return dictionaries directly.
        if isinstance(result_obj, dict):
            value = result_obj.get("res", result_obj)
            if isinstance(value, dict):
                return value

        return {}

    # ───────────────────────────────────────────────────────────
    # Box conversion
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def convert_boxes(boxes: Any) -> list[tuple[int, int, int, int]]:
        """Convert polygon or xyxy boxes to (x1, y1, x2, y2) tuples."""
        if boxes is None:
            return []

        arr = np.asarray(boxes)
        output = []

        # xyxy format
        if arr.ndim == 2 and arr.shape[-1] == 4:
            for box in arr:
                x1, y1, x2, y2 = box
                output.append((int(x1), int(y1), int(x2), int(y2)))

        # polygon format (4 points × 2 coords)
        elif arr.ndim == 3 and arr.shape[1:] == (4, 2):
            for polygon in arr:
                xs = polygon[:, 0]
                ys = polygon[:, 1]
                output.append((
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ))

        return output

    # ───────────────────────────────────────────────────────────
    # Reading-order sort
    # ───────────────────────────────────────────────────────────

    @staticmethod
    def sort_boxes(boxes: list[OCRBox]) -> list[OCRBox]:
        """Sort OCR boxes in reading order (top-to-bottom, left-to-right)."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: ((b.y1 + b.y2) / 2, b.x1))

        lines: list[list[OCRBox]] = []

        for box in boxes:
            center_y = (box.y1 + box.y2) / 2
            box_height = max(1, box.y2 - box.y1)

            placed = False

            for group in reversed(lines[-5:]):
                group_center = np.mean([(x.y1 + x.y2) / 2 for x in group])
                group_height = np.mean([max(1, x.y2 - x.y1) for x in group])

                if abs(center_y - group_center) <= max(box_height, group_height) * 0.60:
                    group.append(box)
                    placed = True
                    break

            if not placed:
                lines.append([box])

        output = []
        for group in lines:
            group.sort(key=lambda b: b.x1)
            output.extend(group)

        return output

    # ───────────────────────────────────────────────────────────
    # Single OCR pass
    # ───────────────────────────────────────────────────────────

    def predict_once(self, image: Any) -> list[OCRBox]:
        """Run a single OCR prediction pass on an image."""
        results = self.ocr.predict(image)
        output = []

        for result in results:
            data = self.get_result_dict(result)

            texts = data.get("rec_texts") or data.get("texts") or []
            scores = data.get("rec_scores") or data.get("scores") or []

            # IMPORTANT: Never use "a or b" with numpy arrays.
            boxes = data.get("rec_boxes")
            if boxes is None:
                boxes = data.get("rec_polys")
            if boxes is None:
                boxes = data.get("dt_polys")

            xyxy = self.convert_boxes(boxes)

            for i, text in enumerate(texts):
                text = str(text).strip()
                if not text:
                    continue

                confidence = float(scores[i]) if i < len(scores) else 1.0

                if i < len(xyxy):
                    x1, y1, x2, y2 = xyxy[i]
                else:
                    x1 = y1 = x2 = y2 = 0

                output.append(OCRBox(
                    text=text,
                    confidence=confidence,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))

        return self.sort_boxes(output)

    # ───────────────────────────────────────────────────────────
    # Main OCR with enhancement retries
    # ───────────────────────────────────────────────────────────

    def run(self, image: Any) -> tuple[list[OCRBox], str]:
        """
        Run OCR with automatic enhancement retries.

        Returns:
            (boxes, preprocessing_variant_used)
        """
        # First pass on original image.
        original = self.predict_once(image)

        original_chars = sum(len(x.text) for x in original)
        original_confidence = (
            float(np.mean([x.confidence for x in original]))
            if original
            else 0.0
        )

        # If OCR looks good, stop here.
        if (
            original
            and original_chars >= self.min_ocr_chars
            and original_confidence >= self.retry_confidence
        ):
            return original, "original"

        # Weak result → enhanced retries.
        best_boxes = original
        best_variant = "original"
        best_quality = (
            original_confidence
            * math.log1p(max(1, original_chars))
        )

        for variant in ENHANCEMENT_VARIANTS:
            try:
                processed = prepare_image(image, variant)
                candidate = self.predict_once(processed)

                chars = sum(len(x.text) for x in candidate)
                confidence = (
                    float(np.mean([x.confidence for x in candidate]))
                    if candidate
                    else 0.0
                )

                quality = confidence * math.log1p(max(1, chars))

                if quality > best_quality + 0.02:
                    best_boxes = candidate
                    best_variant = variant
                    best_quality = quality

            except Exception as exc:
                logger.warning("OCR %s retry failed: %s", variant, exc)

        return best_boxes, best_variant

    def close(self) -> None:
        """Release OCR engine and GPU memory."""
        self.ocr = None
        release_gpu_memory()
