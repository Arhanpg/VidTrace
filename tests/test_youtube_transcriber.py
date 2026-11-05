"""Unit tests for the YouTube transcription helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vidtrace.models.events import TranscriptSegment
from vidtrace.youtube.transcriber import (
    PRAKRIT_WARNING,
    collect_video_entries,
    language_name,
    render_markdown,
    render_srt,
    render_vtt,
    resolve_language,
    slugify,
    validate_youtube_url,
)


class TestLanguageResolution:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Hindi", "hi"),
            ("hi", "hi"),
            ("Gujarati", "gu"),
            ("gujrati", "gu"),
            ("Marathi", "mr"),
            ("Kannada", "kn"),
            ("Sanskrit", "sa"),
            ("English", "en"),
            ("auto", None),
        ],
    )
    def test_supported_aliases(self, value: str, expected: str | None) -> None:
        language, warning = resolve_language(value)
        assert language == expected
        assert warning is None

    def test_prakrit_is_explicitly_experimental(self) -> None:
        language, warning = resolve_language("Prakrit")
        assert language is None
        assert warning == PRAKRIT_WARNING

    def test_unknown_language_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            resolve_language("klingon")

    def test_language_name(self) -> None:
        assert language_name("hi") == "Hindi"
        assert language_name("gu") == "Gujarati"
        assert language_name(None) == "Undetermined"


class TestYouTubeInput:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=abc123",
            "https://youtu.be/abc123",
            "https://www.youtube.com/playlist?list=PL123",
            "https://music.youtube.com/watch?v=abc123",
        ],
    )
    def test_valid_youtube_urls(self, url: str) -> None:
        validate_youtube_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/watch?v=abc",
            "file:///C:/video.mp4",
            "not-a-url",
        ],
    )
    def test_non_youtube_urls_rejected(self, url: str) -> None:
        with pytest.raises(ValueError, match="YouTube"):
            validate_youtube_url(url)

    def test_slugify_is_windows_safe(self) -> None:
        result = slugify('A title: Part 1 / Jain "lecture"')
        assert result == "a-title-part-1-jain-lecture"
        assert "/" not in result
        assert ":" not in result


class TestTranscriptRenderers:
    @pytest.fixture
    def segments(self) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(
                index=0,
                start=1.25,
                end=3.50,
                text="नमो अरिहंताणं।",
                words=[],
            ),
            TranscriptSegment(
                index=1,
                start=4.0,
                end=5.25,
                text="આ સમવસરણ છે.",
                words=[],
            ),
        ]

    def test_markdown_contains_source_and_times(self, segments) -> None:
        result = render_markdown(
            title="Jain Lecture",
            video_url="https://youtu.be/abc123",
            video_id="abc123",
            playlist_title="Jain Research",
            playlist_index=2,
            requested_language="auto",
            detected_language="hi",
            language_probability=0.93,
            model_name="small",
            audio_duration=12.5,
            segments=segments,
        )
        assert "# Jain Lecture" in result
        assert "https://youtu.be/abc123" in result
        assert "Detected language:** Hindi" in result
        assert "[00:01.250 – 00:03.500]" in result
        assert "नमो अरिहंताणं।" in result
        assert "આ સમવસરણ છે." in result
        assert "not a verified scriptural transcription" in result

    def test_markdown_contains_prakrit_warning(self, segments) -> None:
        result = render_markdown(
            title="Prakrit",
            video_url="https://youtu.be/abc123",
            video_id="abc123",
            playlist_title="Research",
            playlist_index=1,
            requested_language="prakrit",
            detected_language="sa",
            language_probability=0.4,
            model_name="small",
            audio_duration=12,
            segments=segments,
            warning=PRAKRIT_WARNING,
        )
        assert "Language warning" in result
        assert "no dedicated language token" in result

    def test_srt_and_vtt_timestamps(self, segments) -> None:
        srt = render_srt(segments)
        vtt = render_vtt(segments)
        assert "00:00:01,250 --> 00:00:03,500" in srt
        assert "00:00:04.000 --> 00:00:05.250" in vtt
        assert vtt.startswith("WEBVTT")


def test_collect_video_entries_supports_playlist_and_range(monkeypatch) -> None:
    playlist = {
        "title": "Jain playlist",
        "id": "PL123",
        "entries": [
            {"id": "v1", "title": "One", "playlist_index": 1},
            {"id": "v2", "title": "Two", "playlist_index": 2},
            {"id": "v3", "title": "Three", "playlist_index": 3},
        ],
    }

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            return playlist

    fake_module = ModuleType("yt_dlp")
    fake_module.YoutubeDL = FakeYoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)

    title, playlist_id, entries = collect_video_entries(
        "https://www.youtube.com/playlist?list=PL123",
        playlist_start=2,
        playlist_end=3,
    )
    assert title == "Jain playlist"
    assert playlist_id == "PL123"
    assert [e["playlist_index"] for e in entries] == [2, 3]
    assert entries[0]["webpage_url"] == "https://www.youtube.com/watch?v=v2"


def test_collect_video_entries_limits_videos(monkeypatch) -> None:
    playlist = {
        "title": "Jain playlist",
        "id": "PL123",
        "entries": [
            {"id": f"v{i}", "title": f"Video {i}", "playlist_index": i}
            for i in range(1, 6)
        ],
    }

    class FakeYoutubeDL:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            return playlist

    fake_module = ModuleType("yt_dlp")
    fake_module.YoutubeDL = FakeYoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)

    _, _, entries = collect_video_entries(
        "https://www.youtube.com/playlist?list=PL123",
        max_videos=2,
    )
    assert len(entries) == 2
