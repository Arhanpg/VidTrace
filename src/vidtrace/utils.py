"""
Shared utility functions for VidTrace.

Text normalization, time formatting, heuristics, and filesystem helpers.
"""

from __future__ import annotations

import re
import time

# ───────────────────────────────────────────────────────────────
# Time helpers
# ───────────────────────────────────────────────────────────────

def now_iso() -> str:
    """Current time in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable timestamp.

    Returns HH:MM:SS.mmm or MM:SS.mmm depending on duration.
    """
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    return f"{minutes:02d}:{secs:06.3f}"


def format_srt_time(seconds: float) -> str:
    """
    Format seconds into SRT timestamp format: HH:MM:SS,mmm

    SRT uses comma as the millisecond separator.
    """
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_time(seconds: float) -> str:
    """
    Format seconds into WebVTT timestamp format: HH:MM:SS.mmm

    VTT uses dot as the millisecond separator.
    """
    seconds = max(0.0, float(seconds))

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# ───────────────────────────────────────────────────────────────
# Text cleaning
# ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean transcript text: normalize whitespace and non-breaking spaces."""
    text = text.replace("\u00a0", " ")
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def safe_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename.

    Replaces special characters and limits length.
    """
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180] or "video"


# ───────────────────────────────────────────────────────────────
# OCR text normalization (for comparison only)
# ───────────────────────────────────────────────────────────────

def normalize_ocr_text(text: str) -> str:
    """
    Normalize OCR text for comparison and deduplication.

    The original OCR text is never replaced by this — this is used
    exclusively for similarity scoring.
    """
    text = text.lower()

    # Normalize smart quotes.
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")

    text = re.sub(r"\s+", " ", text)

    # Reduce whitespace around programming punctuation.
    text = re.sub(
        r"[ \t]*([{}()\[\];,:.=+\-*/<>])[ \t]*",
        r"\1",
        text,
    )

    return text.strip()


def character_bigram_similarity(a: str, b: str) -> float:
    """
    Compute character bigram similarity between two OCR texts.

    Returns a value in [0.0, 1.0] where 1.0 means identical text.
    Used for semantic deduplication of consecutive OCR frames.
    """
    a = normalize_ocr_text(a)
    b = normalize_ocr_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    def grams(s: str) -> set[str]:
        if len(s) <= 1:
            return {s}
        return {s[i:i + 2] for i in range(len(s) - 1)}

    ga = grams(a)
    gb = grams(b)

    return len(ga & gb) / max(1, len(ga | gb))


# ───────────────────────────────────────────────────────────────
# Code detection heuristic
# ───────────────────────────────────────────────────────────────

def looks_like_code(text: str) -> bool:
    """
    Heuristic check for whether OCR text looks like source code.

    This is a label only — OCR itself is NOT restricted to code.
    Requires at least 2 matching patterns to return True.
    """
    if not text:
        return False

    patterns = [
        # Python
        r"\b(def|class|import|from|return|for|while|if|else|elif)\b",
        # Java / C#
        r"\b(public|private|protected|static|void|int|String|boolean|class)\b",
        # JavaScript / TypeScript
        r"\b(const|let|var|function|async|await|console\.log)\b",
        # C/C++
        r'#include\s*[<"]',
        # Arrow functions
        r"=>",
        # Braces / semicolons
        r"[{};]",
        # Function calls
        r"\b[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)",
        # Assignments
        r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*",
    ]

    score = sum(1 for p in patterns if re.search(p, text))
    return score >= 2
