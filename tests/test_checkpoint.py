"""
Tests for the checkpoint system.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from vidtrace.pipeline.checkpoint import (
    StageStatus,
    VideoCheckpoint,
    VideoIdentity,
    format_checkpoint_status,
)


class TestVideoIdentity(unittest.TestCase):
    """Tests for VideoIdentity."""

    def test_from_file(self) -> None:
        """Should create a valid identity from a real file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content " * 500)
            f.flush()
            path = Path(f.name)

        identity = VideoIdentity.from_file(path)
        assert identity.size_bytes > 0
        assert len(identity.head_hash) == 16
        assert identity.modified_at != ""

        path.unlink()

    def test_matches_same_file(self) -> None:
        """Same file should match itself."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"test content " * 100)
            f.flush()
            path = Path(f.name)

        id1 = VideoIdentity.from_file(path)
        id2 = VideoIdentity.from_file(path)
        assert id1.matches(id2)

        path.unlink()

    def test_different_content_no_match(self) -> None:
        """Different content should not match."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"content A " * 100)
            f.flush()
            path1 = Path(f.name)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"content B " * 100)
            f.flush()
            path2 = Path(f.name)

        id1 = VideoIdentity.from_file(path1)
        id2 = VideoIdentity.from_file(path2)
        assert not id1.matches(id2)

        path1.unlink()
        path2.unlink()


class TestVideoCheckpoint(unittest.TestCase):
    """Tests for VideoCheckpoint."""

    def test_new_checkpoint(self) -> None:
        cp = VideoCheckpoint()
        assert cp.started_at == ""
        assert cp.stages == {}
        assert cp.ocr_last_timestamp == -1.0

    def test_stage_default_pending(self) -> None:
        cp = VideoCheckpoint()
        assert cp.get_stage_status("transcription") == "pending"

    def test_mark_stage(self) -> None:
        cp = VideoCheckpoint()
        cp.mark_stage("transcription", StageStatus.RUNNING)
        assert cp.get_stage_status("transcription") == "running"
        assert cp.stages["transcription"]["started_at"] is not None

    def test_mark_complete(self) -> None:
        cp = VideoCheckpoint()
        cp.mark_stage("transcription", StageStatus.COMPLETE, segments=42)
        assert cp.get_stage_status("transcription") == "complete"
        assert cp.stages["transcription"]["segments"] == 42

    def test_should_run_pending(self) -> None:
        cp = VideoCheckpoint()
        assert cp.should_run_stage("ocr") is True

    def test_should_run_complete(self) -> None:
        cp = VideoCheckpoint()
        cp.mark_stage("ocr", StageStatus.COMPLETE)
        assert cp.should_run_stage("ocr") is False

    def test_should_run_force(self) -> None:
        cp = VideoCheckpoint()
        cp.mark_stage("ocr", StageStatus.COMPLETE)
        assert cp.should_run_stage("ocr", force=True) is True

    def test_save_and_load(self) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = Path(f.name)

        cp = VideoCheckpoint()
        cp.started_at = "2024-01-01T00:00:00"
        cp.mark_stage("transcription", StageStatus.COMPLETE, segments=10)
        cp.update_ocr_progress(42.5, 1275, 5)
        cp.save(path)

        loaded = VideoCheckpoint.load(path)
        assert loaded is not None
        assert loaded.started_at == "2024-01-01T00:00:00"
        assert loaded.get_stage_status("transcription") == "complete"
        assert loaded.ocr_last_timestamp == 42.5
        assert loaded.ocr_last_frame == 1275
        assert loaded.ocr_event_count == 5

        path.unlink()

    def test_load_nonexistent(self) -> None:
        result = VideoCheckpoint.load(Path("/nonexistent/state.json"))
        assert result is None

    def test_update_ocr_progress(self) -> None:
        cp = VideoCheckpoint()
        cp.update_ocr_progress(100.0, 3000, 15)
        assert cp.ocr_last_timestamp == 100.0
        assert cp.ocr_last_frame == 3000
        assert cp.ocr_event_count == 15


class TestFormatCheckpointStatus(unittest.TestCase):
    """Tests for format_checkpoint_status()."""

    def test_empty_checkpoint(self) -> None:
        cp = VideoCheckpoint()
        result = format_checkpoint_status(cp)
        assert "pending" in result

    def test_partial_checkpoint(self) -> None:
        cp = VideoCheckpoint()
        cp.started_at = "2024-01-01"
        cp.mark_stage("transcription", StageStatus.COMPLETE)
        cp.mark_stage("ocr", StageStatus.PARTIAL)
        cp.ocr_last_timestamp = 42.5
        cp.ocr_last_frame = 1275
        cp.ocr_event_count = 5

        result = format_checkpoint_status(cp)
        assert "complete" in result
        assert "partial" in result
        assert "42.5" in result


if __name__ == "__main__":
    unittest.main()
