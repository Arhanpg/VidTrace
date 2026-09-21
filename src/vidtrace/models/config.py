"""
Configuration system for VidTrace.

Centralizes all pipeline parameters with named presets for common
video types (lectures, coding tutorials, meetings).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ───────────────────────────────────────────────────────────────
# Main configuration
# ───────────────────────────────────────────────────────────────

@dataclass
class VidTraceConfig:
    """All pipeline configuration in a single object."""

    # ── Whisper ───────────────────────────────────────────────
    whisper_model: str = "small"
    language: str | None = None

    # ── Visual sampling ───────────────────────────────────────
    sample_interval: float = 2.0
    active_interval: float = 0.50
    active_window: float = 8.0
    scene_threshold: float = 0.065

    # ── OCR ───────────────────────────────────────────────────
    ocr_change_similarity: float = 0.965
    ocr_heartbeat: float = 3.0
    ocr_retry_confidence: float = 0.62
    min_ocr_chars: int = 4

    # ── Frame filtering ───────────────────────────────────────
    min_luma: float = 8.0

    # ── Output ────────────────────────────────────────────────
    jpeg_quality: int = 92
    output_folder: str = "vidtrace_output"
    output_formats: list[str] = field(
        default_factory=lambda: ["md", "json"]
    )

    # ── Execution ─────────────────────────────────────────────
    cpu_ocr: bool = False
    force: bool = False

    def to_dict(self) -> dict:
        """Serialize config for JSON/YAML output."""
        return {
            "whisper_model": self.whisper_model,
            "language": self.language,
            "sample_interval": self.sample_interval,
            "active_interval": self.active_interval,
            "active_window": self.active_window,
            "scene_threshold": self.scene_threshold,
            "ocr_change_similarity": self.ocr_change_similarity,
            "ocr_heartbeat": self.ocr_heartbeat,
            "ocr_retry_confidence": self.ocr_retry_confidence,
            "min_ocr_chars": self.min_ocr_chars,
            "min_luma": self.min_luma,
            "jpeg_quality": self.jpeg_quality,
            "output_folder": self.output_folder,
            "output_formats": self.output_formats,
            "cpu_ocr": self.cpu_ocr,
            "force": self.force,
        }


# ───────────────────────────────────────────────────────────────
# Presets
# ───────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "default": {},

    "lecture": {
        "sample_interval": 2.0,
        "active_interval": 0.50,
        "active_window": 8.0,
        "ocr_change_similarity": 0.965,
        "output_formats": ["md", "json"],
    },

    "coding": {
        "sample_interval": 1.5,
        "active_interval": 0.40,
        "active_window": 10.0,
        "ocr_change_similarity": 0.950,
        "ocr_heartbeat": 2.0,
        "output_formats": ["md", "json", "html"],
    },

    "meeting": {
        "sample_interval": 5.0,
        "active_interval": 2.0,
        "active_window": 5.0,
        "ocr_change_similarity": 0.980,
        "ocr_heartbeat": 10.0,
        "output_formats": ["md", "json", "srt"],
    },

    "tutorial": {
        "sample_interval": 2.0,
        "active_interval": 0.50,
        "active_window": 8.0,
        "ocr_change_similarity": 0.960,
        "output_formats": ["md", "json", "html", "srt"],
    },
}


def load_preset(name: str) -> VidTraceConfig:
    """Create a config from a named preset."""
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(
            f"Unknown preset '{name}'. Available: {available}"
        )

    overrides = PRESETS[name]
    return VidTraceConfig(**overrides)


def load_config_file(path: Path) -> VidTraceConfig:
    """Load configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return VidTraceConfig(**{
        k: v for k, v in data.items()
        if hasattr(VidTraceConfig, k)
    })


def get_preset_names() -> list[str]:
    """Return all available preset names."""
    return sorted(PRESETS.keys())
