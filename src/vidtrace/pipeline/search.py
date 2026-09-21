"""
Multimodal search across VidTrace output.

Searches speech, OCR, code, and scene events across one or more
video output directories.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from vidtrace.utils import format_time

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search match from the timeline."""

    timestamp: float
    event_type: str
    source: str
    text: str
    context: str
    video_name: str
    confidence: float | None
    evidence_path: str | None

    def format(self) -> str:
        """Format as a human-readable line."""
        badge = self.event_type.upper()
        ts = format_time(self.timestamp)
        conf = f" conf={self.confidence:.3f}" if self.confidence else ""
        return (
            f"  [{ts}] {badge:8s} | {self.text[:120]}{conf}"
        )


def search_timeline(
    output_dir: Path,
    query: str,
    event_types: list[str] | None = None,
    case_sensitive: bool = False,
) -> list[SearchResult]:
    """
    Search across all timeline events in an output directory.

    Searches speech text, OCR text, and code events. The query is
    matched as a substring — no regex needed for simple searches.

    Args:
        output_dir: Path to the vidtrace output directory.
        query: Search string.
        event_types: Filter to specific event types (e.g., ["speech", "code"]).
        case_sensitive: Whether to match case.

    Returns:
        List of matching SearchResult objects, sorted by timestamp.
    """
    results: list[SearchResult] = []

    if not case_sensitive:
        query_lower = query.lower()

    # Search all subdirectories (each is a video output).
    for video_dir in sorted(output_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        video_name = video_dir.name

        # Search timeline.json if it exists.
        timeline_path = video_dir / "timeline.json"
        if timeline_path.exists():
            results.extend(
                _search_timeline_file(
                    timeline_path, query, query_lower if not case_sensitive else query,
                    event_types, case_sensitive, video_name,
                )
            )
            continue

        # Fallback: search transcript.json + ocr_events.json separately.
        transcript_path = video_dir / "transcript.json"
        if transcript_path.exists():
            results.extend(
                _search_transcript(
                    transcript_path, query,
                    query_lower if not case_sensitive else query,
                    case_sensitive, video_name,
                )
            )

        ocr_events_path = video_dir / "ocr_events.json"
        if ocr_events_path.exists():
            results.extend(
                _search_ocr_events(
                    ocr_events_path, query,
                    query_lower if not case_sensitive else query,
                    event_types, case_sensitive, video_name,
                )
            )

    results.sort(key=lambda r: (r.video_name, r.timestamp))
    return results


def _search_timeline_file(
    path: Path,
    query: str,
    query_normalized: str,
    event_types: list[str] | None,
    case_sensitive: bool,
    video_name: str,
) -> list[SearchResult]:
    """Search a timeline.json file."""
    results = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    for event in data:
        etype = event.get("type", "")

        if event_types and etype not in event_types:
            continue

        text = event.get("text") or ""
        match_text = text if case_sensitive else text.lower()

        if query_normalized not in match_text:
            continue

        # Extract context: surrounding text snippet.
        idx = match_text.find(query_normalized)
        start = max(0, idx - 40)
        end = min(len(text), idx + len(query) + 40)
        context = text[start:end]

        results.append(SearchResult(
            timestamp=event.get("timestamp", 0.0),
            event_type=etype,
            source=event.get("source", ""),
            text=text[:200],
            context=context,
            video_name=video_name,
            confidence=event.get("confidence"),
            evidence_path=event.get("frame_path"),
        ))

    return results


def _search_transcript(
    path: Path,
    query: str,
    query_normalized: str,
    case_sensitive: bool,
    video_name: str,
) -> list[SearchResult]:
    """Search a transcript.json file."""
    results = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        segments = data.get("segments", [])
    except Exception:
        return []

    for segment in segments:
        text = segment.get("text", "")
        match_text = text if case_sensitive else text.lower()

        if query_normalized not in match_text:
            continue

        results.append(SearchResult(
            timestamp=segment.get("start", 0.0),
            event_type="speech",
            source="whisper",
            text=text,
            context=text[:120],
            video_name=video_name,
            confidence=None,
            evidence_path=None,
        ))

    return results


def _search_ocr_events(
    path: Path,
    query: str,
    query_normalized: str,
    event_types: list[str] | None,
    case_sensitive: bool,
    video_name: str,
) -> list[SearchResult]:
    """Search an ocr_events.json file."""
    results = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    for event in data:
        text = event.get("text", "")
        match_text = text if case_sensitive else text.lower()

        if query_normalized not in match_text:
            continue

        is_code = event.get("code_likelihood", False)
        etype = "code" if is_code else "ocr"

        if event_types and etype not in event_types:
            continue

        results.append(SearchResult(
            timestamp=event.get("timestamp", 0.0),
            event_type=etype,
            source="paddleocr",
            text=text[:200],
            context=text[:120],
            video_name=video_name,
            confidence=event.get("mean_confidence"),
            evidence_path=event.get("evidence_frame"),
        ))

    return results


def format_search_results(
    results: list[SearchResult],
    query: str,
) -> str:
    """Format search results as a readable report."""
    if not results:
        return f'No results found for: "{query}"'

    lines = [
        f'Found {len(results)} results for: "{query}"',
        "",
    ]

    current_video = ""
    for result in results:
        if result.video_name != current_video:
            current_video = result.video_name
            lines.append(f"── {current_video} ──")

        lines.append(result.format())

    return "\n".join(lines)
