"""
Unit tests for vidtrace.utils.

Tests text normalization, time formatting, similarity scoring,
and code detection heuristics.
"""

import pytest

from vidtrace.utils import (
    character_bigram_similarity,
    clean_text,
    format_srt_time,
    format_time,
    format_vtt_time,
    looks_like_code,
    normalize_ocr_text,
    safe_filename,
)


# ───────────────────────────────────────────────────────────────
# format_time
# ───────────────────────────────────────────────────────────────

class TestFormatTime:
    def test_zero(self):
        assert format_time(0) == "00:00.000"

    def test_seconds(self):
        assert format_time(5.5) == "00:05.500"

    def test_minutes(self):
        assert format_time(65.123) == "01:05.123"

    def test_hours(self):
        assert format_time(3661.5) == "01:01:01.500"

    def test_negative_clamped(self):
        assert format_time(-5.0) == "00:00.000"

    def test_large_value(self):
        result = format_time(7200.0)
        assert result.startswith("02:00:")


# ───────────────────────────────────────────────────────────────
# format_srt_time / format_vtt_time
# ───────────────────────────────────────────────────────────────

class TestSubtitleTime:
    def test_srt_format(self):
        assert format_srt_time(3661.5) == "01:01:01,500"

    def test_srt_zero(self):
        assert format_srt_time(0) == "00:00:00,000"

    def test_vtt_format(self):
        assert format_vtt_time(3661.5) == "01:01:01.500"

    def test_vtt_zero(self):
        assert format_vtt_time(0) == "00:00:00.000"

    def test_srt_vs_vtt_separator(self):
        srt = format_srt_time(1.234)
        vtt = format_vtt_time(1.234)
        assert "," in srt and "," not in vtt
        assert "." in vtt


# ───────────────────────────────────────────────────────────────
# clean_text
# ───────────────────────────────────────────────────────────────

class TestCleanText:
    def test_whitespace(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_nbsp(self):
        assert clean_text("hello\u00a0world") == "hello world"

    def test_carriage_return(self):
        assert clean_text("hello\r\nworld") == "hello\nworld"

    def test_tabs(self):
        assert clean_text("hello\t\tworld") == "hello world"

    def test_empty(self):
        assert clean_text("") == ""


# ───────────────────────────────────────────────────────────────
# safe_filename
# ───────────────────────────────────────────────────────────────

class TestSafeFilename:
    def test_normal(self):
        assert safe_filename("lecture.mp4") == "lecture.mp4"

    def test_special_chars(self):
        result = safe_filename("file<>with:bad/chars")
        assert "<" not in result
        assert ">" not in result

    def test_long_name(self):
        result = safe_filename("a" * 300)
        assert len(result) <= 180

    def test_empty(self):
        assert safe_filename("") == "video"

    def test_spaces(self):
        result = safe_filename("  hello   world  ")
        assert result == "hello world"


# ───────────────────────────────────────────────────────────────
# normalize_ocr_text
# ───────────────────────────────────────────────────────────────

class TestNormalizeOCRText:
    def test_lowercase(self):
        assert normalize_ocr_text("Hello World") == "hello world"

    def test_smart_quotes(self):
        result = normalize_ocr_text("\u201chello\u201d")
        assert '"' in result
        assert "\u201c" not in result

    def test_whitespace_collapse(self):
        result = normalize_ocr_text("hello    world")
        assert result == "hello world"

    def test_punctuation_spacing(self):
        result = normalize_ocr_text("def foo ( x ) :")
        assert result == "def foo(x):"


# ───────────────────────────────────────────────────────────────
# character_bigram_similarity
# ───────────────────────────────────────────────────────────────

class TestBigramSimilarity:
    def test_identical(self):
        assert character_bigram_similarity("hello", "hello") == 1.0

    def test_empty_both(self):
        assert character_bigram_similarity("", "") == 1.0

    def test_empty_one(self):
        assert character_bigram_similarity("hello", "") == 0.0
        assert character_bigram_similarity("", "hello") == 0.0

    def test_similar(self):
        sim = character_bigram_similarity(
            "def foo(x):\n  return x",
            "def foo(x):\n  return x + 1",
        )
        assert 0.7 < sim < 1.0

    def test_different(self):
        sim = character_bigram_similarity(
            "completely different text",
            "nothing in common here xyz",
        )
        assert sim < 0.5

    def test_case_insensitive(self):
        assert character_bigram_similarity("Hello", "hello") == 1.0

    def test_symmetry(self):
        a = "some text here"
        b = "other text here"
        assert (
            character_bigram_similarity(a, b)
            == character_bigram_similarity(b, a)
        )


# ───────────────────────────────────────────────────────────────
# looks_like_code
# ───────────────────────────────────────────────────────────────

class TestLooksLikeCode:
    def test_python(self):
        assert looks_like_code("def foo(x):\n  return x + 1") is True

    def test_java(self):
        assert looks_like_code(
            "public static void main(String[] args) {"
        ) is True

    def test_javascript(self):
        assert looks_like_code(
            "const result = await fetch(url);"
        ) is True

    def test_plain_text(self):
        assert looks_like_code(
            "The quick brown fox jumps over the lazy dog."
        ) is False

    def test_empty(self):
        assert looks_like_code("") is False

    def test_single_keyword(self):
        # Only one pattern match — should be False (needs >= 2).
        assert looks_like_code("import") is False

    def test_assignment_and_function(self):
        assert looks_like_code("x = foo(bar)") is True
