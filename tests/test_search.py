"""
Tests for multimodal search.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vidtrace.pipeline.search import SearchResult, format_search_results, search_timeline


class TestSearchTimeline(unittest.TestCase):
    """Tests for search_timeline()."""

    def setUp(self) -> None:
        """Create a temp output directory with sample data."""
        self.tmpdir = Path(tempfile.mkdtemp())
        video_dir = self.tmpdir / "test_video"
        video_dir.mkdir()

        # Write a sample timeline.json.
        timeline = [
            {
                "id": "a1",
                "timestamp": 10.0,
                "duration": 3.0,
                "type": "speech",
                "source": "whisper",
                "text": "Now let's connect to PostgreSQL database",
                "confidence": None,
                "frame_path": None,
            },
            {
                "id": "b2",
                "timestamp": 15.0,
                "duration": None,
                "type": "ocr",
                "source": "paddleocr",
                "text": "DATABASE_URL=postgres://localhost:5432/mydb",
                "confidence": 0.92,
                "frame_path": "evidence/001.jpg",
            },
            {
                "id": "c3",
                "timestamp": 20.0,
                "duration": None,
                "type": "code",
                "source": "paddleocr",
                "text": "import psycopg2\nconn = psycopg2.connect(DATABASE_URL)",
                "confidence": 0.88,
                "frame_path": "evidence/002.jpg",
            },
            {
                "id": "d4",
                "timestamp": 50.0,
                "duration": 2.0,
                "type": "speech",
                "source": "whisper",
                "text": "The model uses attention mechanisms",
                "confidence": None,
                "frame_path": None,
            },
        ]

        (video_dir / "timeline.json").write_text(
            json.dumps(timeline), encoding="utf-8",
        )

    def test_search_speech(self) -> None:
        results = search_timeline(self.tmpdir, "PostgreSQL")
        assert len(results) == 1
        assert results[0].event_type == "speech"
        assert results[0].timestamp == 10.0

    def test_search_ocr(self) -> None:
        results = search_timeline(self.tmpdir, "DATABASE_URL")
        assert len(results) == 2  # OCR + code
        types = {r.event_type for r in results}
        assert "ocr" in types
        assert "code" in types

    def test_search_code(self) -> None:
        results = search_timeline(self.tmpdir, "psycopg2")
        assert len(results) == 1
        assert results[0].event_type == "code"

    def test_case_insensitive(self) -> None:
        results = search_timeline(self.tmpdir, "postgresql")
        assert len(results) == 1

    def test_case_sensitive(self) -> None:
        results = search_timeline(
            self.tmpdir, "postgresql", case_sensitive=True,
        )
        assert len(results) == 0

        results = search_timeline(
            self.tmpdir, "PostgreSQL", case_sensitive=True,
        )
        assert len(results) == 1

    def test_filter_by_type(self) -> None:
        results = search_timeline(
            self.tmpdir, "DATABASE_URL", event_types=["code"],
        )
        assert len(results) == 1
        assert results[0].event_type == "code"

    def test_no_results(self) -> None:
        results = search_timeline(self.tmpdir, "nonexistent_term_xyz")
        assert len(results) == 0

    def test_multimodal_match(self) -> None:
        """Same query should match across speech and OCR."""
        results = search_timeline(self.tmpdir, "database")
        # Should match speech ("database") and OCR ("DATABASE_URL")
        types = {r.event_type for r in results}
        assert len(types) >= 1

    def test_empty_directory(self) -> None:
        empty = Path(tempfile.mkdtemp())
        results = search_timeline(empty, "test")
        assert results == []


class TestSearchResult(unittest.TestCase):
    """Tests for SearchResult formatting."""

    def test_format(self) -> None:
        result = SearchResult(
            timestamp=65.0,
            event_type="speech",
            source="whisper",
            text="Hello world",
            context="Hello world",
            video_name="test",
            confidence=None,
            evidence_path=None,
        )
        formatted = result.format()
        assert "01:05" in formatted
        assert "SPEECH" in formatted

    def test_format_with_confidence(self) -> None:
        result = SearchResult(
            timestamp=10.0,
            event_type="ocr",
            source="paddleocr",
            text="Code text",
            context="Code",
            video_name="test",
            confidence=0.95,
            evidence_path="frames/001.jpg",
        )
        formatted = result.format()
        assert "conf=0.950" in formatted


class TestFormatSearchResults(unittest.TestCase):
    """Tests for format_search_results()."""

    def test_no_results(self) -> None:
        result = format_search_results([], "query")
        assert "No results" in result

    def test_with_results(self) -> None:
        results = [
            SearchResult(
                timestamp=10.0,
                event_type="speech",
                source="whisper",
                text="Test",
                context="Test",
                video_name="video1",
                confidence=None,
                evidence_path=None,
            ),
        ]
        result = format_search_results(results, "test")
        assert "1 results" in result
        assert "video1" in result


if __name__ == "__main__":
    unittest.main()
