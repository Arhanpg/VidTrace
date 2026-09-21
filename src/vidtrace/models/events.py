"""
Core data models for VidTrace.

All pipeline stages produce and consume these dataclasses.
The unified TimelineEvent schema merges speech, OCR, and visual events
into a single temporal stream.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ───────────────────────────────────────────────────────────────
# Video metadata
# ───────────────────────────────────────────────────────────────

@dataclass
class VideoInfo:
    """Metadata extracted from a video file."""

    path: str
    name: str
    duration_seconds: float
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ───────────────────────────────────────────────────────────────
# Transcript segment (from Whisper)
# ───────────────────────────────────────────────────────────────

@dataclass
class TranscriptSegment:
    """A single speech segment with word-level timestamps."""

    index: int
    start: float
    end: float
    text: str
    words: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ───────────────────────────────────────────────────────────────
# OCR bounding box
# ───────────────────────────────────────────────────────────────

@dataclass
class OCRBox:
    """A single text region detected by OCR."""

    text: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ───────────────────────────────────────────────────────────────
# OCR event (deduplicated, timestamped screen state)
# ───────────────────────────────────────────────────────────────

@dataclass
class OCREvent:
    """
    A semantically distinct OCR observation.

    OCR candidates are deduplicated: only screen states that differ
    meaningfully from the previous observation become events.
    """

    event_id: int
    timestamp: float
    frame_index: int
    text: str
    lines: list[str]
    boxes: list[dict[str, Any]]
    mean_confidence: float
    code_likelihood: bool
    preprocessing: str
    evidence_frame: str | None
    speech_context: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ───────────────────────────────────────────────────────────────
# Unified timeline event
# ───────────────────────────────────────────────────────────────

@dataclass
class TimelineEvent:
    """
    A single event in the multimodal timeline.

    This is the unified schema: every speech segment, OCR observation,
    and visual change is converted into a TimelineEvent for downstream
    consumption (search, RAG, export).

    Types:
        "speech"  — transcribed speech segment
        "ocr"     — screen text detected by OCR
        "visual"  — visual scene change detected
        "code"    — code-heavy screen event (subset of OCR)

    Sources:
        "whisper"    — faster-whisper transcription
        "paddleocr"  — PaddleOCR engine
        "opencv"     — OpenCV frame analysis
    """

    timestamp: float
    type: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_transcript(cls, segment: TranscriptSegment) -> TimelineEvent:
        """Convert a transcript segment into a timeline event."""
        return cls(
            timestamp=segment.start,
            type="speech",
            source="whisper",
            text=segment.text,
            metadata={
                "start": segment.start,
                "end": segment.end,
                "index": segment.index,
                "word_count": len(segment.words),
            },
        )

    @classmethod
    def from_ocr_event(cls, event: OCREvent) -> TimelineEvent:
        """Convert an OCR event into a timeline event."""
        event_type = "code" if event.code_likelihood else "ocr"
        return cls(
            timestamp=event.timestamp,
            type=event_type,
            source="paddleocr",
            text=event.text,
            metadata={
                "event_id": event.event_id,
                "frame_index": event.frame_index,
                "mean_confidence": event.mean_confidence,
                "preprocessing": event.preprocessing,
                "evidence_frame": event.evidence_frame,
                "line_count": len(event.lines),
            },
        )
