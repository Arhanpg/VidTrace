"""
Temporal code reconstruction.

Tracks code evolution across consecutive OCR frames,
performs cross-frame syntax correction, and produces
timestamped code evolution artifacts.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from vidtrace.utils import format_time


@dataclass
class CodeSnapshot:
    """A single code state at a point in time."""

    timestamp: float
    text: str
    language: str
    confidence: float
    frame_path: str | None = None


@dataclass
class CodeDiff:
    """A detected change between two code snapshots."""

    timestamp: float
    previous_timestamp: float
    added_lines: list[str]
    removed_lines: list[str]
    change_type: str  # "addition", "deletion", "modification", "refactor"


@dataclass
class CodeEvolution:
    """
    Tracks the evolution of a code block over time.

    Reconstructs the editing timeline by comparing consecutive
    OCR snapshots and detecting insertions, deletions, and
    modifications.
    """

    snapshots: list[CodeSnapshot] = field(default_factory=list)
    diffs: list[CodeDiff] = field(default_factory=list)
    language: str = "unknown"

    def add_snapshot(self, snapshot: CodeSnapshot) -> CodeDiff | None:
        """
        Add a new code snapshot and compute the diff.

        Returns the CodeDiff if there was a meaningful change,
        None if the code is essentially the same.
        """
        if not self.snapshots:
            self.snapshots.append(snapshot)
            self.language = snapshot.language
            return None

        previous = self.snapshots[-1]

        # Compute line-level diff.
        prev_lines = previous.text.splitlines()
        curr_lines = snapshot.text.splitlines()

        differ = difflib.unified_diff(
            prev_lines, curr_lines,
            lineterm="",
        )
        diff_lines = list(differ)

        if not diff_lines:
            return None

        # Classify the change.
        added = [
            line[1:] for line in diff_lines
            if line.startswith("+") and not line.startswith("+++")
        ]
        removed = [
            line[1:] for line in diff_lines
            if line.startswith("-") and not line.startswith("---")
        ]

        # Skip insignificant changes (OCR noise).
        if not added and not removed:
            return None

        change_type = self._classify_change(added, removed)

        diff = CodeDiff(
            timestamp=snapshot.timestamp,
            previous_timestamp=previous.timestamp,
            added_lines=added,
            removed_lines=removed,
            change_type=change_type,
        )

        self.snapshots.append(snapshot)
        self.diffs.append(diff)

        return diff

    def _classify_change(
        self,
        added: list[str],
        removed: list[str],
    ) -> str:
        """Classify the type of code change."""
        if added and not removed:
            return "addition"
        elif removed and not added:
            return "deletion"
        elif len(added) > len(removed) * 2:
            return "addition"
        elif len(removed) > len(added) * 2:
            return "deletion"
        else:
            return "modification"

    def format_evolution(self) -> str:
        """Format the code evolution as a readable timeline."""
        if not self.snapshots:
            return "No code snapshots captured."

        lines = [
            f"CODE EVOLUTION ({self.language})",
            f"Snapshots: {len(self.snapshots)}",
            f"Changes: {len(self.diffs)}",
            "",
        ]

        for i, snapshot in enumerate(self.snapshots):
            ts = format_time(snapshot.timestamp)
            lines.append(f"── t={ts} ──")
            lines.append(snapshot.text)
            lines.append("")

            if i < len(self.diffs):
                diff = self.diffs[i]
                lines.append(
                    f"  ↓ {diff.change_type} "
                    f"(+{len(diff.added_lines)} -{len(diff.removed_lines)})"
                )
                lines.append("")

        return "\n".join(lines)


class CodeReconstructor:
    """
    Reconstructs code from multiple OCR frames using temporal voting.

    When OCR gives ambiguous characters, the reconstructor checks
    neighboring frames to choose the most consistent version.
    """

    def __init__(self) -> None:
        self.evolutions: dict[str, CodeEvolution] = {}

    def process_frame(
        self,
        timestamp: float,
        code_text: str,
        language: str,
        confidence: float,
        region_key: str = "main",
        frame_path: str | None = None,
    ) -> CodeDiff | None:
        """
        Process a code frame and track its evolution.

        Args:
            timestamp: Frame timestamp.
            code_text: Extracted code text.
            language: Detected language.
            confidence: OCR confidence.
            region_key: Key to group related code regions.
            frame_path: Path to evidence frame.

        Returns:
            CodeDiff if a meaningful change was detected.
        """
        if region_key not in self.evolutions:
            self.evolutions[region_key] = CodeEvolution()

        snapshot = CodeSnapshot(
            timestamp=timestamp,
            text=code_text,
            language=language,
            confidence=confidence,
            frame_path=frame_path,
        )

        return self.evolutions[region_key].add_snapshot(snapshot)

    def correct_ocr_errors(
        self,
        current_text: str,
        previous_texts: list[str],
    ) -> str:
        """
        Cross-frame OCR correction via temporal voting.

        When OCR gives ambiguous characters, check if neighboring
        frames consistently show a different character. Common OCR
        confusions: 0/O, 1/l/I, =/-, _/-, (/[, )/{.
        """
        if not previous_texts:
            return current_text

        # Common OCR confusion pairs.
        confusions = [
            (r"for\s+(\w+)\s+in\s+range\((\w+)\s*$",
             r"for \1 in range(\2):"),
            (r"\bif\s+(\w+)\s*$",
             r"if \1:"),
            (r"\belse\s*$",
             r"else:"),
            (r"\bdef\s+(\w+)\s*\(([^)]*)\)\s*$",
             r"def \1(\2):"),
        ]

        result = current_text
        for pattern, replacement in confusions:
            # Only apply if majority of previous frames have the colon.
            if re.search(pattern, result, re.MULTILINE):
                votes = sum(
                    1 for prev in previous_texts
                    if re.search(
                        replacement.replace("\\1", "\\w+").replace("\\2", "[^)]*"),
                        prev,
                        re.MULTILINE,
                    )
                )

                if votes > len(previous_texts) / 2:
                    result = re.sub(
                        pattern, replacement, result,
                        flags=re.MULTILINE,
                    )

        return result

    def get_all_evolutions(self) -> dict[str, CodeEvolution]:
        """Return all tracked code evolutions."""
        return self.evolutions
