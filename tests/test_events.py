"""
Tests for the event model and TimelineEvent.
"""

from __future__ import annotations

import json
import unittest

from vidtrace.models.events import (
    EventType,
    OCREvent,
    TimelineEvent,
    TranscriptSegment,
    VideoInfo,
)


class TestEventType(unittest.TestCase):
    """Tests for EventType enum."""

    def test_core_types_exist(self) -> None:
        assert EventType.SPEECH.value == "speech"
        assert EventType.OCR.value == "ocr"
        assert EventType.CODE.value == "code"
        assert EventType.VISUAL.value == "visual"
        assert EventType.SCENE.value == "scene"

    def test_extension_types_exist(self) -> None:
        assert EventType.TERMINAL.value == "terminal"
        assert EventType.SLIDE.value == "slide"
        assert EventType.UI.value == "ui"
        assert EventType.URL.value == "url"
        assert EventType.COMMAND.value == "command"
        assert EventType.EQUATION.value == "equation"
        assert EventType.ARTIFACT.value == "artifact"

    def test_str_enum_serialization(self) -> None:
        """EventType should serialize directly as a string in JSON."""
        data = {"type": EventType.SPEECH}
        result = json.dumps(data)
        assert '"speech"' in result

    def test_all_types_count(self) -> None:
        """Ensure we have 12 event types."""
        assert len(EventType) == 12


class TestTimelineEvent(unittest.TestCase):
    """Tests for TimelineEvent dataclass."""

    def test_construction(self) -> None:
        event = TimelineEvent(
            id="abc123",
            timestamp=42.5,
            duration=3.0,
            type="speech",
            source="whisper",
            text="Hello world",
            confidence=0.95,
            frame_path=None,
        )
        assert event.id == "abc123"
        assert event.timestamp == 42.5
        assert event.duration == 3.0
        assert event.type == "speech"
        assert event.source == "whisper"
        assert event.text == "Hello world"
        assert event.confidence == 0.95
        assert event.frame_path is None
        assert event.metadata == {}

    def test_to_dict(self) -> None:
        event = TimelineEvent(
            id="x",
            timestamp=1.0,
            duration=None,
            type="ocr",
            source="paddleocr",
            text="code",
        )
        d = event.to_dict()
        assert d["id"] == "x"
        assert d["timestamp"] == 1.0
        assert d["duration"] is None
        assert d["type"] == "ocr"

    def test_from_transcript(self) -> None:
        segment = TranscriptSegment(
            index=0,
            start=10.0,
            end=15.0,
            text="Testing speech",
            words=[{"word": "Testing", "start": 10.0}],
        )
        event = TimelineEvent.from_transcript(segment)
        assert event.type == "speech"
        assert event.source == "whisper"
        assert event.text == "Testing speech"
        assert event.timestamp == 10.0
        assert event.duration == 5.0
        assert event.confidence is None
        assert len(event.id) == 12

    def test_from_ocr_event(self) -> None:
        ocr = OCREvent(
            event_id=0,
            timestamp=25.0,
            frame_index=750,
            text="def main():",
            lines=["def main():"],
            boxes=[],
            mean_confidence=0.92,
            code_likelihood=True,
            preprocessing="clahe",
            evidence_frame="evidence_frames/000000_025.000s.jpg",
            speech_context="",
        )
        event = TimelineEvent.from_ocr_event(ocr)
        assert event.type == "code"  # code_likelihood=True
        assert event.source == "paddleocr"
        assert event.confidence == 0.92
        assert event.frame_path == "evidence_frames/000000_025.000s.jpg"
        assert event.duration is None

    def test_from_ocr_event_non_code(self) -> None:
        ocr = OCREvent(
            event_id=1,
            timestamp=30.0,
            frame_index=900,
            text="Regular text on screen",
            lines=["Regular text on screen"],
            boxes=[],
            mean_confidence=0.85,
            code_likelihood=False,
            preprocessing="raw",
            evidence_frame=None,
            speech_context="",
        )
        event = TimelineEvent.from_ocr_event(ocr)
        assert event.type == "ocr"

    def test_from_scene_change(self) -> None:
        event = TimelineEvent.from_scene_change(
            timestamp=60.0,
            visual_difference=0.25,
            frame_path="frames/scene_001.jpg",
        )
        assert event.type == "scene"
        assert event.source == "opencv"
        assert event.text is None
        assert event.confidence == 0.25
        assert event.frame_path == "frames/scene_001.jpg"
        assert event.metadata["visual_difference"] == 0.25

    def test_unique_ids(self) -> None:
        """Factory methods should generate unique IDs."""
        segment = TranscriptSegment(index=0, start=0, end=1, text="a")
        event1 = TimelineEvent.from_transcript(segment)
        event2 = TimelineEvent.from_transcript(segment)
        assert event1.id != event2.id


class TestVideoInfo(unittest.TestCase):
    """Tests for VideoInfo dataclass."""

    def test_to_dict(self) -> None:
        info = VideoInfo(
            path="/video.mp4",
            name="video.mp4",
            duration_seconds=120.0,
            fps=30.0,
            frame_count=3600,
            width=1920,
            height=1080,
            codec="h264",
        )
        d = info.to_dict()
        assert d["path"] == "/video.mp4"
        assert d["fps"] == 30.0
        assert d["width"] == 1920


class TestTranscriptSegment(unittest.TestCase):
    """Tests for TranscriptSegment."""

    def test_to_dict(self) -> None:
        seg = TranscriptSegment(
            index=5,
            start=100.0,
            end=105.0,
            text="Hello",
        )
        d = seg.to_dict()
        assert d["index"] == 5
        assert d["words"] == []


if __name__ == "__main__":
    unittest.main()
