"""
Tests for timeline fusion and speech context.
"""

from __future__ import annotations

import unittest

from vidtrace.fusion.timeline import build_timeline, nearby_speech
from vidtrace.models.events import OCREvent, TranscriptSegment


def _make_segment(index: int, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        index=index, start=start, end=end, text=text,
    )


def _make_ocr(event_id: int, timestamp: float, text: str, code: bool = False) -> OCREvent:
    return OCREvent(
        event_id=event_id,
        timestamp=timestamp,
        frame_index=int(timestamp * 30),
        text=text,
        lines=text.split("\n"),
        boxes=[],
        mean_confidence=0.90,
        code_likelihood=code,
        preprocessing="raw",
        evidence_frame=None,
        speech_context="",
    )


class TestBuildTimeline(unittest.TestCase):
    """Tests for build_timeline()."""

    def test_empty(self) -> None:
        result = build_timeline([], [])
        assert result == []

    def test_speech_only(self) -> None:
        segments = [
            _make_segment(0, 0.0, 2.0, "Hello"),
            _make_segment(1, 5.0, 7.0, "World"),
        ]
        timeline = build_timeline(segments, [])
        assert len(timeline) == 2
        assert timeline[0].type == "speech"
        assert timeline[0].text == "Hello"
        assert timeline[1].text == "World"

    def test_ocr_only(self) -> None:
        events = [
            _make_ocr(0, 3.0, "Screen text"),
            _make_ocr(1, 8.0, "More text"),
        ]
        timeline = build_timeline([], events)
        assert len(timeline) == 2
        assert timeline[0].type == "ocr"

    def test_merged_chronological(self) -> None:
        segments = [
            _make_segment(0, 1.0, 3.0, "Speaking"),
        ]
        events = [
            _make_ocr(0, 0.5, "Before speech"),
            _make_ocr(1, 2.0, "During speech"),
            _make_ocr(2, 4.0, "After speech"),
        ]
        timeline = build_timeline(segments, events)
        assert len(timeline) == 4

        timestamps = [e.timestamp for e in timeline]
        assert timestamps == sorted(timestamps)

    def test_code_events_typed_correctly(self) -> None:
        events = [_make_ocr(0, 1.0, "def main():", code=True)]
        timeline = build_timeline([], events)
        assert timeline[0].type == "code"

    def test_event_ids_unique(self) -> None:
        segments = [_make_segment(0, 0.0, 1.0, "A")]
        events = [_make_ocr(0, 0.5, "B")]
        timeline = build_timeline(segments, events)
        ids = [e.id for e in timeline]
        assert len(ids) == len(set(ids))


class TestNearbySpeech(unittest.TestCase):
    """Tests for nearby_speech()."""

    def test_empty_transcript(self) -> None:
        result = nearby_speech([], 10.0)
        assert result == ""

    def test_no_match(self) -> None:
        segments = [_make_segment(0, 100.0, 105.0, "Far away")]
        result = nearby_speech(segments, 10.0, radius=5.0)
        assert result == ""

    def test_within_radius(self) -> None:
        segments = [
            _make_segment(0, 8.0, 12.0, "Context speech"),
        ]
        result = nearby_speech(segments, 10.0, radius=5.0)
        assert "Context speech" in result

    def test_multiple_matches(self) -> None:
        segments = [
            _make_segment(0, 5.0, 8.0, "First"),
            _make_segment(1, 9.0, 12.0, "Second"),
            _make_segment(2, 100.0, 105.0, "Too far"),
        ]
        result = nearby_speech(segments, 10.0, radius=5.0)
        assert "First" in result
        assert "Second" in result
        assert "Too far" not in result

    def test_radius_zero(self) -> None:
        segments = [_make_segment(0, 10.0, 12.0, "Exact")]
        result = nearby_speech(segments, 10.0, radius=0.0)
        assert "Exact" in result


if __name__ == "__main__":
    unittest.main()
