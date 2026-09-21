"""
Code detection and extraction from OCR results.

Identifies code regions in OCR output using layout analysis,
language detection, and indentation-aware grouping. Goes beyond
the simple looks_like_code() heuristic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProgrammingLanguage(str, Enum):
    """Detected programming language."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    SHELL = "shell"
    HTML = "html"
    CSS = "css"
    UNKNOWN = "unknown"


@dataclass
class CodeRegion:
    """
    A detected code region from OCR output.

    Represents a contiguous block of code identified by layout
    analysis and language detection.
    """

    text: str
    language: str
    confidence: float
    start_line: int
    end_line: int
    indentation_preserved: bool
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ───────────────────────────────────────────────────────────────
# Language detection patterns
# ───────────────────────────────────────────────────────────────

_LANGUAGE_PATTERNS: dict[str, list[tuple[str, float]]] = {
    ProgrammingLanguage.PYTHON.value: [
        (r"\bdef\s+\w+\s*\(", 3.0),
        (r"\bclass\s+\w+.*:", 3.0),
        (r"\bimport\s+\w+", 2.0),
        (r"\bfrom\s+\w+\s+import\b", 3.0),
        (r"\bself\.\w+", 2.5),
        (r"\bif\s+__name__\s*==", 4.0),
        (r"\belif\b", 2.0),
        (r"\bprint\s*\(", 1.5),
        (r"\basync\s+def\b", 3.0),
        (r"\bawait\s+\w+", 2.0),
        (r"^\s*@\w+", 1.5),
        (r"\blambda\s+", 2.0),
        (r":\s*$", 0.5),
    ],
    ProgrammingLanguage.JAVASCRIPT.value: [
        (r"\bconst\s+\w+\s*=", 2.5),
        (r"\blet\s+\w+\s*=", 2.0),
        (r"\bvar\s+\w+\s*=", 1.5),
        (r"\bfunction\s+\w+\s*\(", 2.5),
        (r"=>\s*[{(]?", 2.0),
        (r"\bconsole\.\w+\(", 3.0),
        (r"\bdocument\.\w+", 2.5),
        (r"\brequire\s*\(", 2.0),
        (r"\bmodule\.exports\b", 3.0),
        (r"\basync\s+function\b", 2.5),
    ],
    ProgrammingLanguage.TYPESCRIPT.value: [
        (r":\s*(string|number|boolean|void|any)\b", 3.0),
        (r"\binterface\s+\w+", 3.5),
        (r"\btype\s+\w+\s*=", 3.0),
        (r"<\w+>", 1.0),
        (r"\benum\s+\w+", 2.5),
    ],
    ProgrammingLanguage.JAVA.value: [
        (r"\bpublic\s+(static\s+)?(?:void|int|String|boolean|class)\b", 3.5),
        (r"\bprivate\s+\w+", 2.0),
        (r"\bprotected\s+\w+", 2.0),
        (r"\bSystem\.out\.print", 3.5),
        (r"\bnew\s+\w+\(", 1.5),
        (r"\b(extends|implements)\s+\w+", 2.5),
        (r"@Override", 3.0),
        (r"@Autowired", 3.0),
    ],
    ProgrammingLanguage.CPP.value: [
        (r"#include\s*[<\"]", 4.0),
        (r"\bstd::\w+", 3.5),
        (r"\bcout\s*<<", 3.0),
        (r"\bcin\s*>>", 3.0),
        (r"\bnamespace\s+\w+", 2.5),
        (r"\btemplate\s*<", 3.0),
        (r"\bvirtual\s+\w+", 2.0),
        (r"::\w+\(", 1.5),
    ],
    ProgrammingLanguage.GO.value: [
        (r"\bfunc\s+\w+\(", 3.0),
        (r"\bpackage\s+\w+", 3.5),
        (r"\bfmt\.\w+", 3.0),
        (r":=", 2.0),
        (r"\bgoroutine\b", 3.0),
        (r"\bchan\s+\w+", 2.5),
        (r"\bdefer\s+", 2.0),
    ],
    ProgrammingLanguage.RUST.value: [
        (r"\bfn\s+\w+\(", 3.0),
        (r"\blet\s+mut\s+", 3.5),
        (r"\bimpl\s+\w+", 3.0),
        (r"\bpub\s+fn\b", 3.0),
        (r"\bmatch\s+\w+\s*{", 2.5),
        (r"->", 1.0),
        (r"\bprintln!\(", 4.0),
    ],
    ProgrammingLanguage.SQL.value: [
        (r"\bSELECT\b.*\bFROM\b", 4.0),
        (r"\bCREATE\s+TABLE\b", 4.0),
        (r"\bINSERT\s+INTO\b", 3.5),
        (r"\bALTER\s+TABLE\b", 3.5),
        (r"\bWHERE\b", 1.5),
        (r"\bJOIN\b", 1.5),
        (r"\bGROUP\s+BY\b", 2.0),
    ],
    ProgrammingLanguage.SHELL.value: [
        (r"^\s*#!/bin/(ba)?sh", 5.0),
        (r"\becho\s+", 1.5),
        (r"\bsudo\s+", 2.0),
        (r"\bpip\s+install\b", 2.5),
        (r"\bnpm\s+(install|run)\b", 2.5),
        (r"\bgit\s+(clone|commit|push)\b", 2.5),
        (r"\$\{?\w+\}?", 1.5),
    ],
    ProgrammingLanguage.HTML.value: [
        (r"<\w+[^>]*>", 1.5),
        (r"</\w+>", 1.5),
        (r'class="[^"]*"', 2.0),
        (r"<!DOCTYPE", 4.0),
    ],
    ProgrammingLanguage.CSS.value: [
        (r"\w+\s*:\s*\w+.*;", 1.5),
        (r"\.\w+\s*{", 2.0),
        (r"#\w+\s*{", 2.0),
        (r"@media\b", 3.0),
    ],
}

# Minimum score threshold for language detection.
_LANGUAGE_THRESHOLD = 4.0


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect the programming language of a text block.

    Returns:
        Tuple of (language, confidence_score).
    """
    if not text.strip():
        return ProgrammingLanguage.UNKNOWN.value, 0.0

    scores: dict[str, float] = {}

    for language, patterns in _LANGUAGE_PATTERNS.items():
        score = 0.0
        for pattern, weight in patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            score += len(matches) * weight
        scores[language] = score

    if not scores:
        return ProgrammingLanguage.UNKNOWN.value, 0.0

    best_language = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_language]

    if best_score < _LANGUAGE_THRESHOLD:
        return ProgrammingLanguage.UNKNOWN.value, best_score

    return best_language, best_score


# ───────────────────────────────────────────────────────────────
# Code detection
# ───────────────────────────────────────────────────────────────

# General code indicators (language-agnostic).
_CODE_INDICATORS = [
    (r"[{};]", 1.0),
    (r"\b\w+\([^)]*\)", 1.0),
    (r"\b\w+\s*=\s*", 0.8),
    (r"^\s*(const|let|var|def|class|function|import|from)\b", 1.0), # Keywords
    (r"^\s{2,}\S", 0.5),        # Indentation
    (r"//.*$", 1.5),             # Line comments
    (r"/\*.*\*/", 1.5),          # Block comments
    (r"#\s*\w+", 0.5),           # Directives / comments
    (r"\b(true|false|null|None|undefined)\b", 1.0),
    (r"\b\d+\.\d+\b", 0.3),     # Float literals
    (r'["\'][\w./]+["\']', 0.5),  # String paths
]

_CODE_THRESHOLD = 2.0


def is_code_block(text: str) -> bool:
    """
    Determine if a text block is source code.

    More robust than the simple looks_like_code() — uses weighted
    scoring across language-agnostic indicators.
    """
    if not text.strip():
        return False

    score = 0.0
    for pattern, weight in _CODE_INDICATORS:
        matches = re.findall(pattern, text, re.MULTILINE)
        score += min(len(matches), 5) * weight

    # Bonus for consistent indentation.
    lines = text.strip().split("\n")
    if len(lines) >= 3:
        indented = sum(1 for line in lines if line.startswith(("  ", "\t")))
        if indented / len(lines) > 0.4:
            score += 2.0

    return score >= _CODE_THRESHOLD


class CodeDetector:
    """
    Detects and extracts code regions from OCR output.

    Groups OCR boxes by spatial proximity, identifies code regions
    using language detection, and preserves indentation structure.
    """

    def __init__(self) -> None:
        self.min_lines: int = 2

    def detect_regions(
        self,
        ocr_lines: list[str],
        boxes: list[dict[str, Any]] | None = None,
    ) -> list[CodeRegion]:
        """
        Detect code regions from OCR text lines.

        Args:
            ocr_lines: Lines of OCR text.
            boxes: Optional bounding box data per line.

        Returns:
            List of detected CodeRegion objects.
        """
        if not ocr_lines:
            return []

        # Group consecutive code-like lines.
        regions: list[CodeRegion] = []
        current_block: list[str] = []
        start_line = 0

        for i, line in enumerate(ocr_lines):
            stripped = line.strip()

            if self._is_code_line(stripped):
                if not current_block:
                    start_line = i
                current_block.append(line)
            else:
                if len(current_block) >= self.min_lines:
                    regions.append(
                        self._build_region(
                            current_block, start_line, i - 1, boxes,
                        )
                    )
                current_block = []

        # Handle trailing block.
        if len(current_block) >= self.min_lines:
            regions.append(
                self._build_region(
                    current_block, start_line,
                    len(ocr_lines) - 1, boxes,
                )
            )

        return regions

    def _is_code_line(self, line: str) -> bool:
        """Check if a single line looks like code."""
        if not line:
            return False

        # Strong code indicators for a single line.
        indicators = [
            r"[{};]$",
            r"^\s*(def|class|import|from|return|if|else|for|while)\b",
            r"^\s*(const|let|var|function|public|private)\b",
            r"^\s*#include\b",
            r"=>",
            r"\w+\([^)]*\)\s*[{;:]?\s*$",
            r"^\s*\w+\s*=\s*",
            r"^\s*//",
            r"^\s*#\s",
        ]

        score = sum(
            1 for p in indicators
            if re.search(p, line, re.IGNORECASE)
        )

        return score >= 1

    def _build_region(
        self,
        lines: list[str],
        start_line: int,
        end_line: int,
        boxes: list[dict[str, Any]] | None,
    ) -> CodeRegion:
        """Build a CodeRegion from grouped lines."""
        text = "\n".join(lines)
        language, lang_score = detect_language(text)

        # Check if indentation is preserved.
        has_indentation = any(
            line.startswith(("  ", "\t"))
            for line in lines
            if line.strip()
        )

        # Bounding box (union of relevant boxes).
        x1 = y1 = 0
        x2 = y2 = 0

        if boxes and start_line < len(boxes):
            relevant = boxes[start_line:end_line + 1]
            if relevant:
                x1 = min(b.get("x1", 0) for b in relevant)
                y1 = min(b.get("y1", 0) for b in relevant)
                x2 = max(b.get("x2", 0) for b in relevant)
                y2 = max(b.get("y2", 0) for b in relevant)

        return CodeRegion(
            text=text,
            language=language,
            confidence=min(1.0, lang_score / 10.0),
            start_line=start_line,
            end_line=end_line,
            indentation_preserved=has_indentation,
            x1=x1, y1=y1, x2=x2, y2=y2,
            metadata={
                "line_count": len(lines),
                "language_score": lang_score,
            },
        )
