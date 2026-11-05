"""YouTube video/playlist downloader and multilingual transcript exporter."""
from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from vidtrace.audio.transcription import WhisperTranscriber
from vidtrace.models.config import VidTraceConfig
from vidtrace.models.events import TranscriptSegment
from vidtrace.utils import format_srt_time, format_time, format_vtt_time

logger = logging.getLogger(__name__)

LANGUAGE_ALIASES: dict[str, str | None] = {
    "auto": None,
    "hi": "hi", "hindi": "hi",
    "gu": "gu", "gujarati": "gu", "gujrati": "gu",
    "mr": "mr", "marathi": "mr",
    "kn": "kn", "kannada": "kn",
    "sa": "sa", "sanskrit": "sa",
    "en": "en", "english": "en",
    "bn": "bn", "bengali": "bn",
    "as": "as", "assamese": "as",
    "pa": "pa", "punjabi": "pa",
    "ta": "ta", "tamil": "ta",
    "te": "te", "telugu": "te",
    "ml": "ml", "malayalam": "ml",
    "ur": "ur", "urdu": "ur",
    "ne": "ne", "nepali": "ne",
    "prakrit": None, "prākrit": None, "pali-prakrit": None,
}
LANGUAGE_NAMES = {
    "hi": "Hindi", "gu": "Gujarati", "mr": "Marathi",
    "kn": "Kannada", "sa": "Sanskrit", "en": "English",
    "bn": "Bengali", "as": "Assamese", "pa": "Punjabi",
    "ta": "Tamil", "te": "Telugu", "ml": "Malayalam",
    "ur": "Urdu", "ne": "Nepali",
}
PRAKRIT_WARNING = (
    "Prakrit has no dedicated language token in the standard Whisper language "
    "set. This mode uses automatic language detection; the output may be "
    "classified as Sanskrit, Hindi, or another language and must be manually "
    "checked by someone familiar with the Prakrit variety."
)
JAIN_INITIAL_PROMPT = (
    "Jain Dharma, Digambara Jainism, Samavasaran, Samavasarana, "
    "Adinath, Rishabhdev, Tirthankara, Kevaljnana, Divyadhvani, "
    "Gandhakuti, Manastambha, pitha, vedi, bhumi, kot, "
    "Tiloyapannatti, Adipurana, Jinasena, Yativrishabha."
)


def resolve_language(value: str) -> tuple[str | None, str | None]:
    """Return Whisper language code and optional warning."""
    key = value.strip().lower()
    if key not in LANGUAGE_ALIASES:
        available = ", ".join(sorted(LANGUAGE_ALIASES))
        raise ValueError(f"Unsupported language '{value}'. Use one of: {available}")
    warning = PRAKRIT_WARNING if key in {"prakrit", "prākrit", "pali-prakrit"} else None
    return LANGUAGE_ALIASES[key], warning


def language_name(code: str | None) -> str:
    if not code:
        return "Undetermined"
    return LANGUAGE_NAMES.get(code, code)


def slugify(value: str, max_length: int = 70) -> str:
    """Create a readable, Windows-safe ASCII folder name."""
    value = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return value[:max_length].strip("-") or "video"


def validate_youtube_url(url: str) -> None:
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    allowed = {
        "youtube.com", "www.youtube.com", "m.youtube.com",
        "music.youtube.com", "youtu.be", "www.youtu.be",
        "youtube-nocookie.com", "www.youtube-nocookie.com",
    }
    if parsed.scheme not in {"http", "https"} or host not in allowed:
        raise ValueError("Please provide a YouTube video or playlist URL.")


def _youtube_watch_url(entry: dict[str, Any]) -> str:
    candidate = entry.get("webpage_url") or entry.get("url")
    if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
        return candidate
    video_id = entry.get("id") or candidate
    if not video_id:
        raise ValueError("A playlist entry has no video ID or URL.")
    return f"https://www.youtube.com/watch?v={video_id}"


def collect_video_entries(
    url: str,
    playlist_start: int = 1,
    playlist_end: int | None = None,
    max_videos: int | None = None,
) -> tuple[str, str | None, list[dict[str, Any]]]:
    """Read video/playlist metadata without downloading media."""
    validate_youtube_url(url)
    if playlist_start < 1:
        raise ValueError("--playlist-start must be at least 1.")
    if playlist_end is not None and playlist_end < playlist_start:
        raise ValueError("--playlist-end must be >= --playlist-start.")
    if max_videos is not None and max_videos < 1:
        raise ValueError("--max-videos must be at least 1.")

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            'yt-dlp is missing. Install with: pip install -e ".[youtube]"'
        ) from exc

    options = {
        "quiet": True,
        "no_warnings": False,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    if not info:
        raise RuntimeError("yt-dlp could not read this YouTube URL.")

    collection_title = str(info.get("title") or "YouTube Transcripts")
    collection_id = info.get("id")
    raw_entries = info.get("entries")

    if raw_entries is None:
        entries = [dict(info)]
        entries[0]["playlist_index"] = 1
        entries[0]["webpage_url"] = _youtube_watch_url(entries[0])
        if playlist_start > 1:
            entries = []
    else:
        entries = []
        for fallback_index, entry in enumerate(raw_entries, start=1):
            if not entry:
                continue
            original_index = int(entry.get("playlist_index") or fallback_index)
            if original_index < playlist_start:
                continue
            if playlist_end is not None and original_index > playlist_end:
                continue
            normalized = dict(entry)
            normalized["playlist_index"] = original_index
            normalized["webpage_url"] = _youtube_watch_url(normalized)
            entries.append(normalized)
            if max_videos is not None and len(entries) >= max_videos:
                break

    if max_videos is not None:
        entries = entries[:max_videos]
    if not entries:
        raise RuntimeError("No available videos were found in the selected range.")
    return collection_title, str(collection_id) if collection_id else None, entries


def _find_cached_audio(work_dir: Path) -> Path | None:
    if not work_dir.exists():
        return None
    candidates = [
        p for p in work_dir.glob("source.*")
        if p.is_file() and not p.name.endswith((".part", ".ytdl"))
    ]
    return candidates[0] if candidates else None


def download_audio(video_url: str, work_dir: Path) -> tuple[Path, dict[str, Any] | None]:
    """Download only the best available audio stream for one video."""
    cached = _find_cached_audio(work_dir)
    if cached:
        logger.info("Using cached audio: %s", cached.name)
        return cached, None

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            'yt-dlp is missing. Install with: pip install -e ".[youtube]"'
        ) from exc

    work_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(work_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "retries": 5,
        "fragment_retries": 5,
        "continuedl": True,
        "overwrites": False,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(video_url, download=True)

    audio_path = _find_cached_audio(work_dir)
    if audio_path is None:
        raise RuntimeError("yt-dlp finished without an audio file. Update yt-dlp and retry.")
    return audio_path, info if isinstance(info, dict) else None


def _segments_to_json(segments: list[TranscriptSegment]) -> list[dict[str, Any]]:
    return [
        {
            "index": seg.index,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text,
            "words": seg.words,
        }
        for seg in segments
    ]


def render_markdown(
    title: str,
    video_url: str,
    video_id: str,
    playlist_title: str,
    playlist_index: int,
    requested_language: str,
    detected_language: str | None,
    language_probability: float | None,
    model_name: str,
    audio_duration: float | None,
    segments: list[TranscriptSegment],
    warning: str | None = None,
) -> str:
    """Render a timestamped Markdown transcript."""
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"# {title}", "",
        "## Source", "",
        f"- **YouTube URL:** {video_url}",
        f"- **Video ID:** {video_id}",
        f"- **Playlist / collection:** {playlist_title}",
        f"- **Playlist position:** {playlist_index}",
        f"- **Generated (UTC):** {generated}", "",
        "## Transcription metadata", "",
        f"- **Detected language:** {language_name(detected_language)} "
        f"({detected_language or 'unknown'})",
        f"- **Requested language:** {requested_language}",
        "- **Language confidence:** " + (
            f"{language_probability:.3f}"
            if language_probability is not None else "not reported"
        ),
        f"- **Whisper model:** {model_name}",
        "- **Audio duration:** " + (
            format_time(audio_duration)
            if audio_duration is not None else "not reported"
        ),
        "- **Task:** Speech transcription in the original spoken language; no translation.",
        "",
        "> **Accuracy note:** This is an automatic speech-recognition draft, not a verified "
        "scriptural transcription. Review names, gāthā numbers, Sanskrit/Prakrit terms, "
        "proper nouns and technical religious vocabulary against the source book/audio.",
        "",
    ]
    if warning:
        lines.extend(["## Language warning", "", f"> {warning}", ""])
    lines.extend(["## Timestamped transcript", ""])
    if not segments:
        lines.extend(["_No speech segments were returned by the model._", ""])
    else:
        for segment in segments:
            lines.extend([
                f"**[{format_time(segment.start)} – {format_time(segment.end)}]**",
                "",
                segment.text.strip(),
                "",
            ])
    return "\n".join(lines).rstrip() + "\n"


def render_srt(segments: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for number, segment in enumerate(segments, start=1):
        lines.extend([
            str(number),
            f"{format_srt_time(segment.start)} --> {format_srt_time(segment.end)}",
            segment.text.strip(),
            "",
        ])
    return "\n".join(lines)


def render_vtt(segments: list[TranscriptSegment]) -> str:
    lines = ["WEBVTT", ""]
    for segment in segments:
        lines.extend([
            f"{format_vtt_time(segment.start)} --> {format_vtt_time(segment.end)}",
            segment.text.strip(),
            "",
        ])
    return "\n".join(lines)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _write_playlist_index(
    path: Path,
    playlist_title: str,
    records: list[dict[str, Any]],
) -> None:
    lines = [
        f"# Transcript index — {playlist_title}", "",
        f"- **Videos listed:** {len(records)}", "",
        "| # | Video | Status | Language | Transcript |",
        "|---:|---|---|---|---|",
    ]
    for record in records:
        transcript_link = record.get("transcript_link")
        link = f"[Open]({transcript_link})" if transcript_link else "—"
        title = str(record.get("title", "Untitled")).replace("|", "\\|")
        language = str(record.get("language_name", "Unknown")).replace("|", "\\|")
        lines.append(
            f"| {record.get('playlist_index', '')} | {title} | "
            f"{record.get('status', 'unknown')} | {language} | {link} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_youtube_transcription(
    url: str,
    output_dir: Path,
    model_name: str = "small",
    language: str = "auto",
    keep_audio: bool = False,
    force: bool = False,
    playlist_start: int = 1,
    playlist_end: int | None = None,
    max_videos: int | None = None,
    initial_prompt: str | None = None,
) -> int:
    """
    Download and transcribe a YouTube video or playlist.

    One faster-whisper instance is reused for the whole playlist to reduce
    repeated model loading and GPU memory churn.
    """
    language_code, language_warning = resolve_language(language)
    collection_title, collection_id, entries = collect_video_entries(
        url=url,
        playlist_start=playlist_start,
        playlist_end=playlist_end,
        max_videos=max_videos,
    )

    root = output_dir.expanduser().resolve() / slugify(collection_title)
    videos_root = root / "videos"
    cache_root = root / ".audio_cache"
    videos_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Found %d video(s) in '%s'. Output: %s",
        len(entries), collection_title, root,
    )
    if language_warning:
        logger.warning(language_warning)

    config = VidTraceConfig(
        whisper_model=model_name,
        language=language_code,
        output_folder=str(root),
    )
    prompt = initial_prompt if initial_prompt is not None else JAIN_INITIAL_PROMPT
    transcriber: WhisperTranscriber | None = None
    records: list[dict[str, Any]] = []
    succeeded = 0
    failed = 0
    skipped = 0

    try:
        for position, entry in enumerate(entries, start=1):
            playlist_index = int(entry.get("playlist_index") or position)
            video_url = str(entry.get("webpage_url") or _youtube_watch_url(entry))
            video_id = str(entry.get("id") or f"entry-{playlist_index}")
            title = str(entry.get("title") or f"YouTube video {video_id}")
            video_dir_name = (
                f"{playlist_index:03d}_{slugify(title, 55)}_{slugify(video_id, 20)}"
            )
            video_dir = videos_root / video_dir_name
            video_dir.mkdir(parents=True, exist_ok=True)

            existing_metadata = video_dir / "metadata.json"
            existing_transcripts = list(video_dir.glob("transcript.*.md"))
            if not force and existing_metadata.exists() and existing_transcripts:
                try:
                    old_meta = json.loads(existing_metadata.read_text(encoding="utf-8"))
                    transcript_filename = old_meta.get("transcript_file", "")
                    transcript_path = video_dir / transcript_filename
                    record = {
                        "playlist_index": playlist_index,
                        "title": title,
                        "status": "skipped (already complete)",
                        "language_name": old_meta.get("detected_language_name", "Unknown"),
                        "transcript_link": (
                            str(transcript_path.relative_to(root)).replace("\\", "/")
                            if transcript_path.exists() else None
                        ),
                    }
                    records.append(record)
                    skipped += 1
                    logger.info("[%d/%d] Already complete: %s", position, len(entries), title)
                    _write_playlist_index(root / "playlist_index.md", collection_title, records)
                    continue
                except (OSError, ValueError, KeyError):
                    logger.warning("Existing transcript metadata is unreadable; reprocessing.")

            work_dir = cache_root / slugify(video_id, 40)
            try:
                logger.info("[%d/%d] Downloading audio: %s", position, len(entries), title)
                audio_path, downloaded_info = download_audio(video_url, work_dir)
                if downloaded_info:
                    title = str(downloaded_info.get("title") or title)
                    video_id = str(downloaded_info.get("id") or video_id)

                if transcriber is None:
                    logger.info(
                        "Loading faster-whisper model '%s'. The first run may download "
                        "the model weights.",
                        model_name,
                    )
                    transcriber = WhisperTranscriber(config, initial_prompt=prompt)

                logger.info("[%d/%d] Transcribing: %s", position, len(entries), title)
                segments, whisper_meta = transcriber.transcribe(audio_path)

                detected_code = (
                    str(whisper_meta.get("language"))
                    if whisper_meta.get("language") else language_code
                )
                if language.lower() in {"prakrit", "prākrit", "pali-prakrit"}:
                    filename_language = "prakrit_experimental"
                else:
                    filename_language = detected_code or "und"

                transcript_filename = f"transcript.{filename_language}.md"
                transcript_path = video_dir / transcript_filename
                markdown = render_markdown(
                    title=title,
                    video_url=video_url,
                    video_id=video_id,
                    playlist_title=collection_title,
                    playlist_index=playlist_index,
                    requested_language=language,
                    detected_language=detected_code,
                    language_probability=whisper_meta.get("language_probability"),
                    model_name=model_name,
                    audio_duration=whisper_meta.get("audio_duration"),
                    segments=segments,
                    warning=language_warning,
                )
                transcript_path.write_text(markdown, encoding="utf-8")

                (video_dir / f"transcript.{filename_language}.srt").write_text(
                    render_srt(segments), encoding="utf-8",
                )
                (video_dir / f"transcript.{filename_language}.vtt").write_text(
                    render_vtt(segments), encoding="utf-8",
                )

                transcript_json = {
                    "source": {
                        "url": video_url,
                        "video_id": video_id,
                        "title": title,
                        "playlist_title": collection_title,
                        "playlist_id": collection_id,
                        "playlist_index": playlist_index,
                    },
                    "requested_language": language,
                    "detected_language": detected_code,
                    "language_probability": whisper_meta.get("language_probability"),
                    "model": model_name,
                    "whisper_metadata": whisper_meta,
                    "segments": _segments_to_json(segments),
                }
                _write_json(video_dir / "transcript.json", transcript_json)

                metadata = {
                    **transcript_json["source"],
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "requested_language": language,
                    "detected_language": detected_code,
                    "detected_language_name": language_name(detected_code),
                    "language_probability": whisper_meta.get("language_probability"),
                    "model": model_name,
                    "transcript_file": transcript_filename,
                    "audio_duration": whisper_meta.get("audio_duration"),
                    "elapsed_seconds": whisper_meta.get("elapsed_seconds"),
                    "device": whisper_meta.get("device"),
                    "compute_type": whisper_meta.get("compute_type"),
                    "language_warning": language_warning,
                }
                _write_json(existing_metadata, metadata)

                if keep_audio:
                    shutil.copy2(audio_path, video_dir / f"source_audio{audio_path.suffix}")
                shutil.rmtree(work_dir, ignore_errors=True)

                records.append({
                    "playlist_index": playlist_index,
                    "title": title,
                    "status": "completed",
                    "language_name": language_name(detected_code),
                    "transcript_link": str(transcript_path.relative_to(root)).replace("\\", "/"),
                })
                succeeded += 1
                logger.info(
                    "[%d/%d] Complete: %s (%s, %d segments)",
                    position, len(entries), title, language_name(detected_code), len(segments),
                )

            except KeyboardInterrupt:
                logger.warning("Interrupted. Successful transcripts are preserved.")
                raise
            except Exception as exc:
                logger.exception("Failed to process '%s': %s", title, exc)
                records.append({
                    "playlist_index": playlist_index,
                    "title": title,
                    "status": f"failed: {exc}",
                    "language_name": "Unknown",
                    "transcript_link": None,
                })
                failed += 1

            _write_playlist_index(root / "playlist_index.md", collection_title, records)
            _write_json(root / "run_summary.json", {
                "playlist_title": collection_title,
                "playlist_id": collection_id,
                "source_url": url,
                "requested_language": language,
                "whisper_model": model_name,
                "completed": succeeded,
                "skipped": skipped,
                "failed": failed,
                "records": records,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            })

    finally:
        if transcriber is not None:
            transcriber.close()

    logger.info(
        "Finished. Completed=%d | Skipped=%d | Failed=%d | Output=%s",
        succeeded, skipped, failed, root,
    )
    return 1 if failed else 0
