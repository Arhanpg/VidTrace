"""
Plugin and extractor architecture for VidTrace.

Defines the Extractor protocol that all extractors (built-in and
third-party) must implement, plus a registry for managing them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from vidtrace.models.events import TimelineEvent, VideoInfo

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────
# Video context (passed to extractors)
# ───────────────────────────────────────────────────────────────

@dataclass
class VideoContext:
    """
    Context object passed to every extractor.

    Contains everything an extractor needs to process a video
    without coupling to the pipeline orchestrator.
    """

    video_path: Path
    video_info: VideoInfo
    output_dir: Path
    evidence_dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    existing_events: list[TimelineEvent] = field(default_factory=list)


# ───────────────────────────────────────────────────────────────
# Extractor protocol
# ───────────────────────────────────────────────────────────────

@runtime_checkable
class Extractor(Protocol):
    """
    Protocol for VidTrace extractors.

    All extractors — built-in and third-party — should implement
    this protocol. Each extractor receives a VideoContext and
    returns a list of TimelineEvents.

    Built-in extractors:
        - WhisperExtractor (audio → speech)
        - OCRExtractor (vision → ocr/code)
        - SceneExtractor (vision → scene)
        - CodeExtractor (vision → code blocks)

    Plugin extractors (future):
        - SpeakerDiarization
        - QwenVL
        - YOLO
        - EquationOCR
        - URLDetector
    """

    @property
    def name(self) -> str:
        """Unique name of this extractor."""
        ...

    @property
    def version(self) -> str:
        """Version string."""
        ...

    @property
    def event_types(self) -> list[str]:
        """Event types this extractor produces."""
        ...

    def extract(self, context: VideoContext) -> list[TimelineEvent]:
        """
        Run extraction on the video.

        Args:
            context: Video context with paths, config, and prior events.

        Returns:
            List of TimelineEvents produced by this extractor.
        """
        ...

    def is_available(self) -> bool:
        """
        Check if this extractor's dependencies are installed.

        Returns False if required packages are missing.
        """
        ...


# ───────────────────────────────────────────────────────────────
# Extractor registry
# ───────────────────────────────────────────────────────────────

class ExtractorRegistry:
    """
    Manages available extractors.

    Extractors can be registered by name. The registry supports
    listing, enabling/disabling, and running extractors in order.
    """

    def __init__(self) -> None:
        self._extractors: dict[str, Extractor] = {}
        self._enabled: set[str] = set()

    def register(self, extractor: Extractor) -> None:
        """Register an extractor."""
        name = extractor.name
        self._extractors[name] = extractor
        self._enabled.add(name)
        logger.debug("Registered extractor: %s (v%s)", name, extractor.version)

    def unregister(self, name: str) -> None:
        """Remove an extractor from the registry."""
        self._extractors.pop(name, None)
        self._enabled.discard(name)

    def enable(self, name: str) -> None:
        """Enable a registered extractor."""
        if name in self._extractors:
            self._enabled.add(name)

    def disable(self, name: str) -> None:
        """Disable a registered extractor without removing it."""
        self._enabled.discard(name)

    def list_extractors(self) -> list[dict[str, Any]]:
        """List all registered extractors with their status."""
        result = []
        for name, ext in self._extractors.items():
            result.append({
                "name": name,
                "version": ext.version,
                "enabled": name in self._enabled,
                "available": ext.is_available(),
                "event_types": ext.event_types,
            })
        return result

    def get_enabled(self) -> list[Extractor]:
        """Get all enabled and available extractors, in registration order."""
        return [
            ext for name, ext in self._extractors.items()
            if name in self._enabled and ext.is_available()
        ]

    def run_all(self, context: VideoContext) -> list[TimelineEvent]:
        """
        Run all enabled extractors and merge their output.

        Returns a chronologically sorted list of TimelineEvents.
        """
        all_events: list[TimelineEvent] = []

        for extractor in self.get_enabled():
            logger.info("Running extractor: %s", extractor.name)
            try:
                events = extractor.extract(context)
                all_events.extend(events)
                logger.info(
                    "Extractor %s produced %d events",
                    extractor.name, len(events),
                )
            except Exception as exc:
                logger.error(
                    "Extractor %s failed: %s", extractor.name, repr(exc),
                )

        all_events.sort(key=lambda e: e.timestamp)
        return all_events


# ───────────────────────────────────────────────────────────────
# Built-in extractor base (for reference implementations)
# ───────────────────────────────────────────────────────────────

class BaseExtractor:
    """
    Convenience base class for built-in extractors.

    Not required — any class implementing the Extractor protocol works.
    This just reduces boilerplate for VidTrace's own extractors.
    """

    _name: str = "base"
    _version: str = "0.1.0"
    _event_types: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @property
    def event_types(self) -> list[str]:
        return self._event_types

    def extract(self, context: VideoContext) -> list[TimelineEvent]:
        raise NotImplementedError

    def is_available(self) -> bool:
        return True


class SceneExtractor(BaseExtractor):
    """
    Detects visual scene changes using OpenCV frame differencing.

    Produces EventType.SCENE events at points where the visual
    content changes significantly.
    """

    _name = "scene"
    _version = "0.1.0"
    _event_types = ["scene"]

    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold

    def extract(self, context: VideoContext) -> list[TimelineEvent]:
        import cv2

        from vidtrace.vision.sampler import frame_difference, frame_signature

        events: list[TimelineEvent] = []
        cap = cv2.VideoCapture(str(context.video_path))

        if not cap.isOpened():
            return events

        fps = context.video_info.fps or 30.0
        last_sig = None
        frame_idx = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = frame_idx / fps
                frame_idx += 1

                # Sample every 0.5 seconds for scene detection.
                if frame_idx % max(1, int(fps * 0.5)) != 0:
                    continue

                sig = frame_signature(frame)
                diff = frame_difference(last_sig, sig)
                last_sig = sig

                if diff >= self.threshold:
                    events.append(
                        TimelineEvent.from_scene_change(
                            timestamp=timestamp,
                            visual_difference=diff,
                        )
                    )
        finally:
            cap.release()

        return events

    def is_available(self) -> bool:
        try:
            import cv2  # noqa: F401
            return True
        except ImportError:
            return False
