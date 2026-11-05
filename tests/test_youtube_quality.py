"""Tests for conservative transcript quality assessment."""

from vidtrace.models.events import TranscriptSegment
from vidtrace.youtube.quality import assess_transcript


def _segments(*texts: str) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(index=i, start=float(i), end=float(i + 1), text=text)
        for i, text in enumerate(texts)
    ]


def test_repeated_single_token_is_flagged_for_review() -> None:
    segments = _segments(" ".join(["ॐ"] * 30))
    result = assess_transcript(segments, language_probability=0.99)
    assert result["status"] == "review_required"
    assert "extreme_token_repetition" in result["flags"]


def test_symbol_only_transcript_is_not_accepted() -> None:
    segments = _segments("। " * 80)
    result = assess_transcript(segments, language_probability=0.99)
    assert result["status"] in {"failed", "review_required"}
    assert result["score"] < 100


def test_normal_multisentence_hindi_is_not_flagged_as_repetitive() -> None:
    segments = _segments(
        "समवसरण में तीर्थंकर भगवान विराजमान होते हैं।",
        "चारों दिशाओं से जीव धर्म का उपदेश सुनते हैं।",
        "इसका वर्णन जैन आगम और ग्रंथों में मिलता है।",
    )
    result = assess_transcript(segments, language_probability=0.96)
    assert result["status"] == "passed"
    assert result["flags"] == []


def test_low_log_probability_is_flagged() -> None:
    segment = TranscriptSegment(
        index=0,
        start=0,
        end=2,
        text="समवसरण का वर्णन",
        quality={"avg_logprob": -1.7, "no_speech_prob": 0.1, "compression_ratio": 1.2},
    )
    result = assess_transcript([segment], language_probability=0.99)
    assert "low_average_log_probability" in result["flags"]
    assert result["status"] == "review_required"


def test_empty_transcript_is_failed() -> None:
    result = assess_transcript([], language_probability=None)
    assert result["status"] == "failed"
    assert "no_recognizable_text" in result["flags"]
