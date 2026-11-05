"""YouTube video/playlist downloader and multilingual transcript exporter."""
from __future__ import annotations

import html
import json
import logging
import os
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
from vidtrace.youtube.quality import assess_transcript

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
JAIN_LANGUAGE_PROMPTS: dict[str, str] = {
    "hi": (
        "जैन धर्म, दिगम्बर जैन, समवसरण, आदिनाथ, ऋषभदेव, तीर्थंकर, "
        "केवलज्ञान, दिव्यध्वनि, गंधकुटी, मानस्तंभ, पीठ, वेदी, भूमि, "
        "कोट, तिलोयपण्णत्ति, आदिपुराण, जिनसेन, यतिवृषभ।"
    ),
    "mr": (
        "जैन धर्म, दिगंबर जैन, समवसरण, आदिनाथ, ऋषभदेव, तीर्थंकर, "
        "केवलज्ञान, दिव्यध्वनी, गंधकुटी, मानस्तंभ, पीठ, वेदी, भूमी, "
        "कोट, तिलोयपण्णत्ति, आदिपुराण, जिनसेन, यतिवृषभ।"
    ),
    "gu": (
        "જૈન ધર્મ, દિગંબર જૈન, સમવસરણ, આદિનાથ, ઋષભદેવ, તીર્થંકર, "
        "કેવળજ્ઞાન, દિવ્યધ્વનિ, ગંધકુટી, માનસ્તંભ, પીઠ, વેદી, ભૂમિ, "
        "કોટ, તિલોયપન્નત્તિ, આદિપુરાણ, જિનસેન, યતિવૃષભ."
    ),
    "kn": (
        "ಜೈನ ಧರ್ಮ, ದಿಗಂಬರ ಜೈನ, ಸಮವಸರಣ, ಆದಿನಾಥ, ಋಷಭದೇವ, ತೀರ್ಥಂಕರ, "
        "ಕೇವಲಜ್ಞಾನ, ದಿವ್ಯಧ್ವನಿ, ಗಂಧಕುಟಿ, ಮಾನಸ್ತಂಭ, ಪೀಠ, ವೇದಿ, ಭೂಮಿ, "
        "ಕೋಟ, ತಿಲೋಯಪಣ್ಣತ್ತಿ, ಆದಿಪುರಾಣ, ಜಿನಸೇನ, ಯತಿವೃಷಭ."
    ),
    "sa": (
        "जैनधर्मः, दिगम्बरजैनपरम्परा, समवसरणम्, आदिनाथः, ऋषभदेवः, "
        "तीर्थङ्करः, केवलज्ञानम्, दिव्यध्वनिः, गन्धकुटी, मानस्तम्भः, "
        "पीठम्, वेदी, भूमिः, कोटः, तिलोयपण्णत्ति, आदिपुराणम्, "
        "जिनसेनः, यतिवृषभः।"
    ),
}


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


def _ensure_js_runtime_in_path() -> None:
    """Ensure a JavaScript runtime (like Deno) is accessible to yt-dlp."""
    if shutil.which("deno"):
        return
    candidates: list[Path] = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    user_profile = os.environ.get("USERPROFILE")
    if local_app_data:
        winget_pkgs = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkgs.is_dir():
            candidates.extend(winget_pkgs.glob("DenoLand.Deno*"))
    if user_profile:
        candidates.append(Path(user_profile) / ".deno" / "bin")

    for candidate in candidates:
        deno_exe = candidate / "deno.exe" if os.name == "nt" else candidate / "deno"
        if deno_exe.is_file():
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            logger.debug("Added Deno to PATH from %s", candidate)
            break


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

    _ensure_js_runtime_in_path()
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            'yt-dlp is missing. Install with: pip install -e ".[youtube-transcribe]"'
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

    _ensure_js_runtime_in_path()
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            'yt-dlp is missing. Install with: pip install -e ".[youtube-transcribe]"'
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



def _extract_video_info(video_url: str) -> dict[str, Any]:
    """Read full video metadata, including manual/automatic subtitle tracks."""
    _ensure_js_runtime_in_path()
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            'yt-dlp is missing. Install with: pip install -e ".[youtube-transcribe]"'
        ) from exc
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "ignoreerrors": False,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(video_url, download=False)
    if not isinstance(info, dict):
        raise RuntimeError("yt-dlp did not return video metadata.")
    return info


def _caption_language_key(track_key: str) -> str:
    """Normalize YouTube caption identifiers such as hi-orig and hi-IN."""
    return track_key.lower().replace("_", "-").split("-")[0]


def choose_caption_track(
    video_info: dict[str, Any],
    requested_language: str | None,
) -> tuple[str, str] | None:
    """
    Choose an exact-language YouTube caption track.

    Manual captions take precedence over automatic captions. In auto mode,
    only select when YouTube's declared language or a unique supported
    Indian-language caption family resolves the language unambiguously.
    """
    manual = video_info.get("subtitles") or {}
    automatic = video_info.get("automatic_captions") or {}
    if not isinstance(manual, dict):
        manual = {}
    if not isinstance(automatic, dict):
        automatic = {}

    declared = (
        video_info.get("language")
        or video_info.get("original_language")
        or video_info.get("audio_language")
    )
    declared_code = _caption_language_key(str(declared)) if declared else None
    target = requested_language or declared_code

    if target is None:
        families = {
            _caption_language_key(str(key))
            for key in list(manual) + list(automatic)
            if _caption_language_key(str(key)) in LANGUAGE_NAMES
        }
        original_keys = [
            str(key)
            for key in list(manual) + list(automatic)
            if str(key).lower().endswith("-orig")
            and _caption_language_key(str(key)) in LANGUAGE_NAMES
        ]
        original_families = {_caption_language_key(key) for key in original_keys}
        if len(original_families) == 1:
            target = next(iter(original_families))
        elif len(families) == 1:
            target = next(iter(families))
        else:
            return None

    if target not in LANGUAGE_NAMES:
        return None

    manual_matches = [
        str(key) for key in manual
        if _caption_language_key(str(key)) == target
    ]
    auto_matches = [
        str(key) for key in automatic
        if _caption_language_key(str(key)) == target
    ]

    # YouTube's "-orig" track is usually the original track variant.
    manual_matches.sort(key=lambda key: (not key.lower().endswith("-orig"), key))
    auto_matches.sort(key=lambda key: (not key.lower().endswith("-orig"), key))
    if manual_matches:
        return "manual", manual_matches[0]
    if auto_matches:
        return "automatic", auto_matches[0]
    return None


def _parse_caption_time(value: str) -> float | None:
    value = value.strip().split()[0]
    parts = value.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
    except (ValueError, IndexError):
        return None
    return None


def parse_vtt(path: Path) -> list[TranscriptSegment]:
    """Parse WebVTT cues into timestamped segments, stripping cue markup."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    output: list[TranscriptSegment] = []
    start: float | None = None
    end: float | None = None
    cue_lines: list[str] = []

    def flush() -> None:
        nonlocal cue_lines, start, end
        if start is None or end is None or end <= start:
            cue_lines = []
            return
        cleaned_lines: list[str] = []
        for line in cue_lines:
            line = re.sub(r"<\d{1,2}:\d{2}(?::\d{2})?\.\d+>", " ", line)
            line = re.sub(r"<[^>]*>", " ", line)
            line = html.unescape(line).replace("\u200b", "").replace("\ufeff", "")
            line = re.sub(r"\s+", " ", line).strip()
            if line and (not cleaned_lines or line != cleaned_lines[-1]):
                cleaned_lines.append(line)
        text = " ".join(cleaned_lines).strip()
        if text:
            if output and output[-1].text.casefold() == text.casefold():
                # YouTube sometimes repeats an identical rolling caption cue.
                cue_lines = []
                return
            output.append(
                TranscriptSegment(
                    index=len(output),
                    start=start,
                    end=end,
                    text=text,
                    words=[],
                    quality={},
                )
            )
        cue_lines = []

    for line in raw.splitlines():
        line = line.strip()
        if "-->" in line:
            flush()
            left, right = line.split("-->", 1)
            start = _parse_caption_time(left)
            end_match = re.match(r"\s*([0-9:.]+)", right)
            end = _parse_caption_time(end_match.group(1)) if end_match else None
            continue
        if not line or line == "WEBVTT" or line.startswith(
            ("NOTE", "STYLE", "REGION", "Kind:", "Language:")
        ):
            continue
        # Cue identifiers and settings are not spoken words.
        if start is None and "-->" not in line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        cue_lines.append(line)

    flush()
    return output


def download_caption_track(
    video_url: str,
    work_dir: Path,
    track_kind: str,
    track_language: str,
) -> Path | None:
    """Download one selected caption track only, without downloading video."""
    _ensure_js_runtime_in_path()
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            'yt-dlp is missing. Install with: pip install -e ".[youtube-transcribe]"'
        ) from exc

    caption_dir = work_dir / "captions"
    caption_dir.mkdir(parents=True, exist_ok=True)
    for old_file in caption_dir.glob("source.*"):
        old_file.unlink(missing_ok=True)

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "writesubtitles": track_kind == "manual",
        "writeautomaticsub": track_kind == "automatic",
        "subtitleslangs": [track_language],
        "subtitlesformat": "vtt/best",
        "outtmpl": str(caption_dir / "source.%(ext)s"),
        "overwrites": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        ydl.download([video_url])

    candidates = [
        file for file in caption_dir.glob("source.*")
        if file.is_file() and file.suffix.lower() in {".vtt", ".srt"}
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda file: (file.suffix.lower() != ".vtt", file.name))
    return candidates[0]


def _segments_to_json(segments: list[TranscriptSegment]) -> list[dict[str, Any]]:
    return [
        {
            "index": seg.index,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text,
            "words": seg.words,
            "quality": seg.quality,
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
    quality: dict[str, Any] | None = None,
    transcription_source: str = "faster_whisper",
) -> str:
    """Render a timestamped Markdown transcript with quality diagnostics."""
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
        f"- **Transcription source:** {transcription_source}",
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
    if quality is not None:
        flags = quality.get("flags") or []
        lines.extend([
            "## Quality assessment",
            "",
            f"- **Status:** {quality.get('status', 'unknown')}",
            f"- **Quality score:** {quality.get('score', 'not available')}/100",
            f"- **Flags:** {', '.join(flags) if flags else 'None'}",
            "- **Important:** A passed automated check is not a scholarly verification.",
            "",
        ])
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


PIPELINE_VERSION = "2.0.0"


def run_youtube_transcription(
    url: str,
    output_dir: Path,
    model_name: str = "medium",
    language: str = "auto",
    keep_audio: bool = False,
    force: bool = False,
    playlist_start: int = 1,
    playlist_end: int | None = None,
    max_videos: int | None = None,
    initial_prompt: str | None = None,
    captions: str = "prefer",
) -> int:
    """
    Download and transcribe a YouTube video or playlist.

    Quality strategy:
      1. Prefer a matching manual YouTube caption track.
      2. Use matching automatic captions when they pass quality checks.
      3. Otherwise transcribe the audio using conservative faster-whisper decoding.
      4. Retry with medium if a user-selected tiny/base/small model fails quality.
      5. Store suspicious output as review_required instead of calling it complete.

    A quality score is a warning system, not proof that a transcript is correct.
    """
    if captions not in {"prefer", "off"}:
        raise ValueError("captions must be 'prefer' or 'off'.")

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

    effective_model = model_name
    config = VidTraceConfig(
        whisper_model=effective_model,
        language=language_code,
        output_folder=str(root),
    )
    prompt = (
        initial_prompt
        if initial_prompt is not None
        else JAIN_LANGUAGE_PROMPTS.get(language_code or "", "")
    )

    transcriber: WhisperTranscriber | None = None
    records: list[dict[str, Any]] = []
    succeeded = 0
    passed = 0
    review_required = 0
    failed = 0
    skipped = 0
    upgraded_to_medium = False

    def save_summary() -> None:
        _write_playlist_index(root / "playlist_index.md", collection_title, records)
        _write_json(root / "run_summary.json", {
            "pipeline_version": PIPELINE_VERSION,
            "playlist_title": collection_title,
            "playlist_id": collection_id,
            "source_url": url,
            "requested_language": language,
            "requested_whisper_model": model_name,
            "effective_whisper_model": effective_model,
            "processed": succeeded,
            "completed": passed,
            "review_required": review_required,
            "skipped": skipped,
            "failed": failed,
            "records": records,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

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
            if not force and existing_metadata.exists():
                try:
                    old_meta = json.loads(existing_metadata.read_text(encoding="utf-8"))
                    transcript_path = video_dir / str(old_meta.get("transcript_file", ""))
                    same_request = (
                        old_meta.get("pipeline_version") == PIPELINE_VERSION
                        and old_meta.get("requested_language") == language
                        and old_meta.get("requested_whisper_model") == model_name
                        and old_meta.get("captions") == captions
                    )
                    old_quality = old_meta.get("quality") or {}
                    if (
                        same_request
                        and transcript_path.is_file()
                        and old_quality.get("status") == "passed"
                    ):
                        records.append({
                            "playlist_index": playlist_index,
                            "title": title,
                            "status": "skipped (already passed quality checks)",
                            "quality_score": old_quality.get("score"),
                            "language_name": old_meta.get("detected_language_name", "Unknown"),
                            "transcript_link": transcript_path.relative_to(root).as_posix(),
                        })
                        skipped += 1
                        logger.info("[%d/%d] Skipping previously passed transcript: %s",
                                    position, len(entries), title)
                        save_summary()
                        continue
                except (OSError, ValueError, TypeError):
                    logger.warning("Existing metadata is unreadable; reprocessing %s.", title)

            work_dir = cache_root / slugify(video_id, 40)
            asr_meta: dict[str, Any] = {}
            caption_meta: dict[str, Any] = {}
            selected_segments: list[TranscriptSegment] = []
            selected_quality: dict[str, Any] = {}
            transcription_source = "faster_whisper"
            detected_code: str | None = None
            model_used: str | None = None
            audio_path: Path | None = None

            try:
                logger.info("[%d/%d] Reading metadata: %s", position, len(entries), title)
                video_info = _extract_video_info(video_url)
                title = str(video_info.get("title") or title)
                video_id = str(video_info.get("id") or video_id)
                caption_candidate: tuple[
                    list[TranscriptSegment], dict[str, Any], str
                ] | None = None

                # First preference: YouTube manual captions, then automatic
                # captions for a language we can resolve unambiguously.
                if (
                    captions == "prefer"
                    and language.lower() not in {"prakrit", "prākrit", "pali-prakrit"}
                ):
                    track = choose_caption_track(video_info, language_code)
                    if track is not None:
                        track_kind, track_language = track
                        try:
                            caption_path = download_caption_track(
                                video_url, work_dir, track_kind, track_language
                            )
                            if caption_path is not None:
                                caption_segments = parse_vtt(caption_path)
                                caption_code = _caption_language_key(track_language)
                                caption_source = (
                                    "youtube_manual_caption"
                                    if track_kind == "manual"
                                    else "youtube_auto_caption"
                                )
                                caption_quality = assess_transcript(
                                    caption_segments,
                                    language_probability=None,
                                    duration_seconds=video_info.get("duration"),
                                    source=caption_source,
                                )
                                caption_meta = {
                                    "track_language": track_language,
                                    "track_kind": track_kind,
                                    "caption_file": caption_path.name,
                                }
                                caption_candidate = (
                                    caption_segments, caption_quality, caption_code
                                )
                                logger.info(
                                    "Caption track found | kind=%s | language=%s | "
                                    "segments=%d | quality=%s",
                                    track_kind, track_language, len(caption_segments),
                                    caption_quality["status"],
                                )
                        except Exception as exc:
                            logger.warning("Could not use selected captions: %s", exc)

                # Manual captions are the source's own reviewed subtitles in
                # many cases; keep them, but still flag repetitive/empty text.
                use_caption = False
                if caption_candidate is not None:
                    cap_segments, cap_quality, cap_code = caption_candidate
                    caption_kind = caption_meta.get("track_kind")
                    if cap_quality["status"] == "passed" or (
                        caption_kind == "manual" and cap_quality["status"] == "review_required"
                    ):
                        selected_segments = cap_segments
                        selected_quality = cap_quality
                        detected_code = cap_code
                        transcription_source = (
                            "youtube_manual_caption"
                            if caption_kind == "manual" else "youtube_auto_caption"
                        )
                        use_caption = True

                if not use_caption:
                    logger.info("[%d/%d] Downloading audio: %s", position, len(entries), title)
                    audio_path, downloaded_info = download_audio(video_url, work_dir)
                    if downloaded_info:
                        title = str(downloaded_info.get("title") or title)
                        video_id = str(downloaded_info.get("id") or video_id)

                    if transcriber is None:
                        logger.info(
                            "Loading faster-whisper '%s'; first use downloads its model weights.",
                            effective_model,
                        )
                        transcriber = WhisperTranscriber(
                            config,
                            initial_prompt=prompt or None,
                        )

                    logger.info("[%d/%d] Transcribing audio: %s", position, len(entries), title)
                    asr_segments, asr_meta = transcriber.transcribe(audio_path)
                    detected_code = (
                        str(asr_meta.get("language"))
                        if asr_meta.get("language") else language_code
                    )
                    asr_quality = assess_transcript(
                        asr_segments,
                        language_probability=asr_meta.get("language_probability"),
                        duration_seconds=asr_meta.get("audio_duration"),
                        source="faster_whisper",
                    )

                    # When the user chose a smaller model, make one quality-
                    # triggered medium retry. Reuse it for the rest of the
                    # playlist to avoid repeatedly loading model weights.
                    if (
                        asr_quality["status"] != "passed"
                        and effective_model in {"tiny", "base", "small"}
                        and not upgraded_to_medium
                    ):
                        logger.warning(
                            "Transcript quality requires review; retrying once with medium."
                        )
                        transcriber.close()
                        config.whisper_model = "medium"
                        effective_model = "medium"
                        transcriber = WhisperTranscriber(
                            config,
                            initial_prompt=prompt or None,
                        )
                        upgraded_to_medium = True
                        retry_segments, retry_meta = transcriber.transcribe(audio_path)
                        retry_quality = assess_transcript(
                            retry_segments,
                            language_probability=retry_meta.get("language_probability"),
                            duration_seconds=retry_meta.get("audio_duration"),
                            source="faster_whisper",
                        )
                        if retry_quality["score"] >= asr_quality["score"]:
                            asr_segments = retry_segments
                            asr_meta = retry_meta
                            asr_quality = retry_quality
                            detected_code = (
                                str(retry_meta.get("language"))
                                if retry_meta.get("language") else detected_code
                            )

                    selected_segments = asr_segments
                    selected_quality = asr_quality
                    transcription_source = "faster_whisper"
                    model_used = effective_model

                    # A suspicious automatic caption is retained as an
                    # alternative only when its objective quality score beats
                    # the ASR result; a manual caption was already preferred.
                    if caption_candidate is not None:
                        cap_segments, cap_quality, cap_code = caption_candidate
                        if cap_quality["score"] > selected_quality["score"]:
                            selected_segments = cap_segments
                            selected_quality = cap_quality
                            detected_code = cap_code
                            transcription_source = "youtube_auto_caption"
                            model_used = None
                            logger.info(
                                "Selected automatic captions over ASR by quality score (%s > %s).",
                                cap_quality["score"], asr_quality["score"],
                            )

                if (
                    transcription_source == "youtube_auto_caption"
                    and selected_quality.get("status") == "passed"
                ):
                    # Automatically generated captions are still ASR output.
                    # They may be useful, but should never be presented as
                    # human-verified research transcription.
                    selected_quality = {
                        **selected_quality,
                        "status": "review_required",
                        "flags": [
                            *selected_quality.get("flags", []),
                            "automatic_captions_not_human_verified",
                        ],
                    }

                if language.lower() in {"prakrit", "prākrit", "pali-prakrit"}:
                    filename_language = "prakrit_experimental"
                else:
                    filename_language = detected_code or "und"

                transcript_filename = f"transcript.{filename_language}.md"
                transcript_path = video_dir / transcript_filename
                language_probability = asr_meta.get("language_probability")
                if transcription_source.startswith("youtube_"):
                    language_probability = None

                markdown = render_markdown(
                    title=title,
                    video_url=video_url,
                    video_id=video_id,
                    playlist_title=collection_title,
                    playlist_index=playlist_index,
                    requested_language=language,
                    detected_language=detected_code,
                    language_probability=language_probability,
                    model_name=model_used or "YouTube captions",
                    audio_duration=(
                        asr_meta.get("audio_duration") or video_info.get("duration")
                    ),
                    segments=selected_segments,
                    warning=language_warning,
                    quality=selected_quality,
                    transcription_source=transcription_source,
                )
                transcript_path.write_text(markdown, encoding="utf-8")

                (video_dir / f"transcript.{filename_language}.srt").write_text(
                    render_srt(selected_segments), encoding="utf-8",
                )
                (video_dir / f"transcript.{filename_language}.vtt").write_text(
                    render_vtt(selected_segments), encoding="utf-8",
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
                    "pipeline_version": PIPELINE_VERSION,
                    "requested_language": language,
                    "detected_language": detected_code,
                    "language_probability": language_probability,
                    "model": model_used,
                    "transcription_source": transcription_source,
                    "quality": selected_quality,
                    "caption_metadata": caption_meta,
                    "whisper_metadata": asr_meta,
                    "segments": _segments_to_json(selected_segments),
                }
                _write_json(video_dir / "transcript.json", transcript_json)

                metadata = {
                    "url": video_url,
                    "video_id": video_id,
                    "title": title,
                    "playlist_title": collection_title,
                    "playlist_id": collection_id,
                    "playlist_index": playlist_index,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "pipeline_version": PIPELINE_VERSION,
                    "requested_language": language,
                    "detected_language": detected_code,
                    "detected_language_name": language_name(detected_code),
                    "language_probability": language_probability,
                    "requested_whisper_model": model_name,
                    "model": model_used,
                    "transcription_source": transcription_source,
                    "quality": selected_quality,
                    "caption_metadata": caption_meta,
                    "transcript_file": transcript_filename,
                    "audio_duration": (
                        asr_meta.get("audio_duration") or video_info.get("duration")
                    ),
                    "elapsed_seconds": asr_meta.get("elapsed_seconds"),
                    "device": asr_meta.get("device"),
                    "compute_type": asr_meta.get("compute_type"),
                    "captions": captions,
                    "language_warning": language_warning,
                }
                _write_json(existing_metadata, metadata)

                if keep_audio and audio_path is not None:
                    shutil.copy2(audio_path, video_dir / f"source_audio{audio_path.suffix}")
                shutil.rmtree(work_dir, ignore_errors=True)

                quality_status = str(selected_quality.get("status", "review_required"))
                record_status = "completed" if quality_status == "passed" else quality_status
                records.append({
                    "playlist_index": playlist_index,
                    "title": title,
                    "status": record_status,
                    "quality_score": selected_quality.get("score"),
                    "quality_flags": selected_quality.get("flags", []),
                    "language_name": language_name(detected_code),
                    "transcription_source": transcription_source,
                    "transcript_link": transcript_path.relative_to(root).as_posix(),
                })
                if quality_status == "passed":
                    succeeded += 1
                    passed += 1
                elif quality_status == "review_required":
                    succeeded += 1
                    review_required += 1
                else:
                    failed += 1

                logger.info(
                    "[%d/%d] %s | source=%s | quality=%s/100 | %s",
                    position, len(entries), record_status, transcription_source,
                    selected_quality.get("score", 0), title,
                )

            except KeyboardInterrupt:
                logger.warning("Interrupted. Previously saved transcripts are preserved.")
                raise
            except Exception as exc:
                logger.exception("Failed to process '%s': %s", title, exc)
                records.append({
                    "playlist_index": playlist_index,
                    "title": title,
                    "status": f"failed: {exc}",
                    "language_name": "Unknown",
                    "transcription_source": None,
                    "transcript_link": None,
                })
                failed += 1

            save_summary()

    finally:
        if transcriber is not None:
            transcriber.close()

    logger.info(
        "Finished | passed=%d | review_required=%d | skipped=%d | failed=%d | output=%s",
        passed, review_required, skipped, failed, root,
    )
    if failed:
        return 1
    if review_required:
        return 2
    return 0
