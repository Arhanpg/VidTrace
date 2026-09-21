"""
Checkpoint and resume system for VidTrace.

Tracks per-stage pipeline progress with video identity verification.
Supports partial OCR resume and automatic cache invalidation when
the source video changes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from vidtrace.utils import now_iso

logger = logging.getLogger(__name__)


class StageStatus(str, Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    PARTIAL = "partial"
    COMPLETE = "complete"


@dataclass
class StageState:
    """State of a single pipeline stage."""

    status: str = StageStatus.PENDING.value
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoIdentity:
    """
    Fingerprint of the source video for cache invalidation.

    Uses file size + modification time + hash of first 8KB.
    If any of these change, all cached results are invalidated.
    """

    path: str
    size_bytes: int
    modified_at: str
    head_hash: str

    @classmethod
    def from_file(cls, video_path: Path) -> VideoIdentity:
        """Create a video identity from a file on disk."""
        stat = video_path.stat()

        # Hash first 8KB for fast identity check.
        hasher = hashlib.sha256()
        with open(video_path, "rb") as f:
            hasher.update(f.read(8192))

        return cls(
            path=str(video_path.resolve()),
            size_bytes=stat.st_size,
            modified_at=str(stat.st_mtime),
            head_hash=hasher.hexdigest()[:16],
        )

    def matches(self, other: VideoIdentity) -> bool:
        """Check if two video identities refer to the same file content."""
        return (
            self.size_bytes == other.size_bytes
            and self.modified_at == other.modified_at
            and self.head_hash == other.head_hash
        )


@dataclass
class VideoCheckpoint:
    """
    Full checkpoint state for a single video processing run.

    Tracks video identity, per-stage progress, and OCR resume
    position for partial recovery.
    """

    video: dict[str, Any] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    # OCR partial resume state.
    ocr_last_timestamp: float = -1.0
    ocr_last_frame: int = -1
    ocr_event_count: int = 0

    @classmethod
    def load(cls, path: Path) -> VideoCheckpoint | None:
        """Load a checkpoint from disk. Returns None if not found."""
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            checkpoint = cls()
            checkpoint.video = data.get("video", {})
            checkpoint.gpu = data.get("gpu", {})
            checkpoint.started_at = data.get("started_at", "")
            checkpoint.stages = data.get("stages", {})
            checkpoint.ocr_last_timestamp = data.get(
                "ocr_last_timestamp", -1.0
            )
            checkpoint.ocr_last_frame = data.get("ocr_last_frame", -1)
            checkpoint.ocr_event_count = data.get("ocr_event_count", 0)
            return checkpoint
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s", exc)
            return None

    def save(self, path: Path) -> None:
        """Save checkpoint to disk."""
        data = {
            "video": self.video,
            "gpu": self.gpu,
            "started_at": self.started_at,
            "stages": self.stages,
            "ocr_last_timestamp": self.ocr_last_timestamp,
            "ocr_last_frame": self.ocr_last_frame,
            "ocr_event_count": self.ocr_event_count,
        }
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_stage_status(self, stage: str) -> str:
        """Get the status of a pipeline stage."""
        stage_data = self.stages.get(stage, {})
        return stage_data.get("status", StageStatus.PENDING.value)

    def should_run_stage(self, stage: str, force: bool = False) -> bool:
        """Determine if a stage should run based on checkpoint state."""
        if force:
            return True
        status = self.get_stage_status(stage)
        return status != StageStatus.COMPLETE.value

    def mark_stage(
        self,
        stage: str,
        status: StageStatus,
        path: Path | None = None,
        **extra: Any,
    ) -> None:
        """Update the status of a pipeline stage."""
        if stage not in self.stages:
            self.stages[stage] = {}

        self.stages[stage]["status"] = status.value

        if status == StageStatus.RUNNING:
            self.stages[stage]["started_at"] = now_iso()
        elif status in (StageStatus.COMPLETE, StageStatus.PARTIAL):
            self.stages[stage]["finished_at"] = now_iso()

        self.stages[stage].update(extra)

        if path is not None:
            self.save(path)

    def is_video_changed(self, video_path: Path) -> bool:
        """
        Check if the source video has changed since last run.

        Returns True if the video identity doesn't match, meaning
        all cached results should be invalidated.
        """
        if not self.video:
            return True

        try:
            current = VideoIdentity.from_file(video_path)
            saved = VideoIdentity(
                path=self.video.get("path", ""),
                size_bytes=self.video.get("size_bytes", 0),
                modified_at=self.video.get("modified_at", ""),
                head_hash=self.video.get("head_hash", ""),
            )
            return not current.matches(saved)
        except Exception:
            return True

    def update_ocr_progress(
        self,
        timestamp: float,
        frame_index: int,
        event_count: int,
        path: Path | None = None,
    ) -> None:
        """Update OCR partial resume position."""
        self.ocr_last_timestamp = timestamp
        self.ocr_last_frame = frame_index
        self.ocr_event_count = event_count

        if path is not None:
            self.save(path)


def format_checkpoint_status(checkpoint: VideoCheckpoint) -> str:
    """Format checkpoint state as a human-readable status string."""
    lines = []
    lines.append(f"Started: {checkpoint.started_at or 'N/A'}")

    for stage_name in ["metadata", "transcription", "ocr", "fusion", "output"]:
        status = checkpoint.get_stage_status(stage_name)
        icon = {
            StageStatus.PENDING.value: "⏳",
            StageStatus.RUNNING.value: "🔄",
            StageStatus.PARTIAL.value: "⚠️",
            StageStatus.COMPLETE.value: "✅",
        }.get(status, "❓")

        lines.append(f"  {icon} {stage_name}: {status}")

        if stage_name == "ocr" and status == StageStatus.PARTIAL.value:
            lines.append(
                f"     Resume from: t={checkpoint.ocr_last_timestamp:.1f}s "
                f"frame={checkpoint.ocr_last_frame} "
                f"events={checkpoint.ocr_event_count}"
            )

    return "\n".join(lines)
