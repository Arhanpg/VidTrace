"""Transcript quality metrics and conservative review flags."""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from typing import Any

from vidtrace.models.events import TranscriptSegment


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def assess_transcript(
    segments: list[TranscriptSegment],
    *,
    language_probability: float | None = None,
    source: str = "faster_whisper",
) -> dict[str, Any]:
    """
    Score a transcript and flag obvious ASR failures.

    The checks intentionally do not rewrite or "correct" recognized text.
    A repeated chant can be legitimate, so repetition is flagged for review
    rather than silently deleted.
    """
    texts = [segment.text.strip() for segment in segments if segment.text.strip()]
    joined = " ".join(texts)
    tokens = re.findall(r"[^\W_]+", joined, flags=re.UNICODE)
    token_count = len(tokens)
    token_counts = Counter(token.lower() for token in tokens)
    unique_token_ratio = (
        len(token_counts) / token_count if token_count else 0.0
    )
    max_token_share = (
        max(token_counts.values()) / token_count if token_count else 0.0
    )

    letter_number_chars = 0
    mark_chars = 0
    punctuation_symbol_chars = 0
    for char in joined:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category.startswith(("L", "N")):
            letter_number_chars += 1
        elif category.startswith("M"):
            mark_chars += 1
        else:
            punctuation_symbol_chars += 1

    total_nonspace = letter_number_chars + mark_chars + punctuation_symbol_chars
    punctuation_symbol_ratio = (
        punctuation_symbol_chars / total_nonspace if total_nonspace else 1.0
    )

    log_probs = [
        value
        for segment in segments
        if (value := _number(segment.quality.get("avg_logprob"))) is not None
    ]
    no_speech_probs = [
        value
        for segment in segments
        if (value := _number(segment.quality.get("no_speech_prob"))) is not None
    ]
    compression_ratios = [
        value
        for segment in segments
        if (value := _number(segment.quality.get("compression_ratio"))) is not None
    ]

    mean_log_prob = statistics.fmean(log_probs) if log_probs else None
    mean_no_speech = statistics.fmean(no_speech_probs) if no_speech_probs else None
    mean_compression = (
        statistics.fmean(compression_ratios) if compression_ratios else None
    )
    lang_prob = _number(language_probability)

    flags: list[str] = []
    score = 100

    if not texts or token_count == 0:
        flags.append("no_recognizable_text")
        score -= 100

    # Repeated-token outputs such as "ौ ौ ौ..." or "ॐ ॐ ॐ..." are
    # often hallucinations. They remain in the file but cannot pass review.
    if token_count >= 12 and max_token_share >= 0.70:
        flags.append("extreme_token_repetition")
        score -= 45
    elif token_count >= 30 and unique_token_ratio <= 0.12:
        flags.append("low_vocabulary_diversity")
        score -= 30

    if len(joined) >= 30 and punctuation_symbol_ratio >= 0.55:
        flags.append("symbol_or_punctuation_heavy")
        score -= 35

    if mean_log_prob is not None and mean_log_prob < -1.0:
        flags.append("low_average_log_probability")
        score -= 20

    if mean_compression is not None and mean_compression > 2.4:
        flags.append("high_compression_ratio")
        score -= 20

    if mean_no_speech is not None and mean_no_speech > 0.70:
        flags.append("high_no_speech_probability")
        score -= 15

    if lang_prob is not None and lang_prob < 0.50:
        flags.append("uncertain_language_detection")
        score -= 15

    score = max(0, min(100, score))
    status = "passed" if not flags and score >= 70 else "review_required"
    if not texts or token_count == 0:
        status = "failed"

    return {
        "status": status,
        "score": score,
        "flags": flags,
        "source": source,
        "segment_count": len(segments),
        "text_character_count": len(joined),
        "token_count": token_count,
        "unique_token_ratio": round(unique_token_ratio, 4),
        "max_token_share": round(max_token_share, 4),
        "punctuation_symbol_ratio": round(punctuation_symbol_ratio, 4),
        "mean_avg_logprob": round(mean_log_prob, 4) if mean_log_prob is not None else None,
        "mean_no_speech_prob": round(mean_no_speech, 4) if mean_no_speech is not None else None,
        "mean_compression_ratio": round(mean_compression, 4) if mean_compression is not None else None,
        "language_probability": round(lang_prob, 4) if lang_prob is not None else None,
    }
