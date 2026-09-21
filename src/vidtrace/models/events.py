"""
Core data models for VidTrace.

All pipeline stages produce and consume these dataclasses.
The unified TimelineEvent schema merges speech, OCR, and visual events
into a single temporal stream.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


# ───────────────────────────────────────────────────────────────
# Event type taxonomy
# ───────────────────────────────────────────────────────────────

class EventType(str, Enum):
    """
    Taxonomy of timeline event types.

    Core types are produced by built-in extractors.
    Extension types are reserved for plugin extractors.

    Using str, Enum so values serialize directly to JSON strings.
    """

    # ── Core types ────────────────────────────────────────────
    SPEECH = "speech"
    OCR = "ocr"
    CODE = "code"
    VISUAL = "visual"
    SCENE = "scene"

    # ── Extension types (future extractors) ───────────────────
    TERMINAL = "terminal"
    SLIDE = "slide"
    UI = "ui"
    URL = "url"
    COMMAND = "command"
    EQUATION = "equation"
    ARTIFACT = "artifact"


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

def _generate_event_id() -> str:
    """Generate a short unique event ID."""
    return uuid.uuid4().hex[:12]


@dataclass
class TimelineEvent:
    """
    A single event in the multimodal timeline.

    This is the unified schema: every speech segment, OCR observation,
    visual change, code block, and scene change is converted into a
    TimelineEvent for downstream consumption (search, RAG, export).

    Fields:
        id:          Unique event identifier (12-char hex).
        timestamp:   Seconds from video start.
        duration:    Duration in seconds (None for point events).
        type:        Event type from EventType enum.
        source:      Extractor that produced this event.
        text:        Extracted text content (may be None for visual events).
        confidence:  Extraction confidence [0.0, 1.0] (None if N/A).
        frame_path:  Relative path to the evidence frame (None if N/A).
        metadata:    Type-specific extra data.
    """

    id: str
    timestamp: float
    duration: float | None
    type: str  # EventType value — kept as str for JSON compat
    source: str
    text: str | None
    confidence: float | None = None
    frame_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_transcript(cls, segment: TranscriptSegment) -> TimelineEvent:
        """Convert a transcript segment into a timeline event."""
        return cls(
            id=_generate_event_id(),
            timestamp=segment.start,
            duration=segment.end - segment.start,
            type=EventType.SPEECH.value,
            source="whisper",
            text=segment.text,
            confidence=None,
            frame_path=None,
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
        event_type = (
            EventType.CODE.value
            if event.code_likelihood
            else EventType.OCR.value
        )
        return cls(
            id=_generate_event_id(),
            timestamp=event.timestamp,
            duration=None,
            type=event_type,
            source="paddleocr",
            text=event.text,
            confidence=event.mean_confidence,
            frame_path=event.evidence_frame,
            metadata={
                "event_id": event.event_id,
                "frame_index": event.frame_index,
                "preprocessing": event.preprocessing,
                "line_count": len(event.lines),
            },
        )

    @classmethod
    def from_scene_change(
        cls,
        timestamp: float,
        visual_difference: float,
        frame_path: str | None = None,
    ) -> TimelineEvent:
        """Create a scene-change event from visual analysis."""
        return cls(
            id=_generate_event_id(),
            timestamp=timestamp,
            duration=None,
            type=EventType.SCENE.value,
            source="opencv",
            text=None,
            confidence=visual_difference,
            frame_path=frame_path,
            metadata={
                "visual_difference": visual_difference,
            },
        )
