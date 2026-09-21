"""
Temporal fusion and multimodal alignment.

Merges speech transcription and OCR events into a unified
chronological timeline, and provides speech-context lookup
for aligning OCR events with nearby spoken words.
"""

from __future__ import annotations

from vidtrace.models.events import (
    OCREvent,
    TimelineEvent,
    TranscriptSegment,
)
from vidtrace.utils import format_time


def build_timeline(
    transcript: list[TranscriptSegment],
    ocr_events: list[OCREvent],
) -> list[TimelineEvent]:
    """
    Merge transcript segments and OCR events into a unified timeline.

    All events are converted to TimelineEvent and sorted chronologically.
    """
    events: list[TimelineEvent] = []

    for segment in transcript:
        events.append(TimelineEvent.from_transcript(segment))

    for ocr_event in ocr_events:
        events.append(TimelineEvent.from_ocr_event(ocr_event))

    events.sort(key=lambda e: e.timestamp)
    return events


def nearby_speech(
    transcript: list[TranscriptSegment],
    timestamp: float,
    radius: float = 10.0,
) -> str:
    """
    Find speech segments within a time radius of a given timestamp.

    Used to provide spoken context for OCR events, so the explanation
    can be connected to the code currently visible on screen.

    Args:
        transcript: List of transcript segments.
        timestamp: The target timestamp in seconds.
        radius: Time window in seconds (before and after).

    Returns:
        Concatenated speech context string.
    """
    output = []

    for segment in transcript:
        if (
            segment.end >= timestamp - radius
            and segment.start <= timestamp + radius
        ):
            output.append(
                f"[{format_time(segment.start)}"
                f" → "
                f"{format_time(segment.end)}] "
                f"{segment.text}"
            )

    return " ".join(output)
