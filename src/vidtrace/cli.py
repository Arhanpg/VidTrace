"""
VidTrace command-line interface.

Usage:
    vidtrace <video_or_folder>
    vidtrace lecture.mp4 --preset coding
    vidtrace ./recordings/ --preset lecture --output-format md,json,html
    vidtrace video.mp4 --transcript-only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vidtrace import __version__
from vidtrace.models.config import (
    VidTraceConfig,
    get_preset_names,
    load_config_file,
    load_preset,
)
from vidtrace.pipeline.orchestrator import run_pipeline


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging with optional rich handler."""
    level = logging.DEBUG if verbose else logging.INFO

    try:
        from rich.logging import RichHandler
        handler = RichHandler(
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler()
        fmt = "[%(levelname)s] %(message)s"

    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[handler],
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="vidtrace",
        description=(
            "VidTrace — Extract the timeline of what was said "
            "and what appeared on screen."
        ),
        epilog=(
            "Examples:\n"
            "  vidtrace lecture.mp4\n"
            "  vidtrace ./recordings/ --preset coding\n"
            "  vidtrace video.mp4 --transcript-only\n"
            "  vidtrace video.mp4 --output-format md,json,html,srt\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Video file or folder containing videos.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"vidtrace {__version__}",
    )

    # ── Presets ────────────────────────────────────────────────
    parser.add_argument(
        "--preset",
        choices=get_preset_names(),
        default=None,
        help="Use a named configuration preset.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML configuration file.",
    )

    # ── Mode flags ────────────────────────────────────────────
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help="Only run speech transcription (skip OCR).",
    )

    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Only run OCR (requires existing transcript).",
    )

    # ── Whisper ────────────────────────────────────────────────
    parser.add_argument(
        "--whisper-model",
        default=None,
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: small).",
    )

    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g., en). Auto-detect if omitted.",
    )

    # ── Sampling ──────────────────────────────────────────────
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=None,
        help="Normal screen OCR interval in seconds.",
    )

    parser.add_argument(
        "--active-interval",
        type=float,
        default=None,
        help="OCR interval while screen is changing.",
    )

    parser.add_argument(
        "--active-window",
        type=float,
        default=None,
        help="Seconds of dense OCR after a visual change.",
    )

    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=None,
        help="Visual difference threshold [0..1].",
    )

    # ── OCR ────────────────────────────────────────────────────
    parser.add_argument(
        "--ocr-change-similarity",
        type=float,
        default=None,
        help="Higher = more OCR snapshots kept.",
    )

    parser.add_argument(
        "--ocr-heartbeat",
        type=float,
        default=None,
        help="Save unchanged screen at this interval.",
    )

    parser.add_argument(
        "--min-ocr-chars",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--min-luma",
        type=float,
        default=None,
        help="Reject almost black frames.",
    )

    # ── Output ────────────────────────────────────────────────
    parser.add_argument(
        "--output-folder",
        default=None,
        help="Output directory name (default: vidtrace_output).",
    )

    parser.add_argument(
        "--output-format",
        default=None,
        help="Comma-separated output formats: md,json,html,srt",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
    )

    # ── Execution ─────────────────────────────────────────────
    parser.add_argument(
        "--cpu-ocr",
        action="store_true",
        help="Force OCR to CPU.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all stages, ignoring existing output.",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser


def _build_config(args: argparse.Namespace) -> VidTraceConfig:
    """Build a VidTraceConfig from CLI arguments."""
    # Start from preset, config file, or defaults.
    if args.config:
        config = load_config_file(args.config)
    elif args.preset:
        config = load_preset(args.preset)
    else:
        config = VidTraceConfig()

    # Apply CLI overrides.
    overrides = {
        "whisper_model": args.whisper_model,
        "language": args.language,
        "sample_interval": args.sample_interval,
        "active_interval": args.active_interval,
        "active_window": args.active_window,
        "scene_threshold": args.scene_threshold,
        "ocr_change_similarity": args.ocr_change_similarity,
        "ocr_heartbeat": args.ocr_heartbeat,
        "min_ocr_chars": args.min_ocr_chars,
        "min_luma": args.min_luma,
        "output_folder": args.output_folder,
        "jpeg_quality": args.jpeg_quality,
    }

    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    if args.output_format:
        config.output_formats = [
            f.strip() for f in args.output_format.split(",")
        ]

    if args.cpu_ocr:
        config.cpu_ocr = True

    if args.force:
        config.force = True

    return config


def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(verbose=args.verbose)

    # Interactive input path.
    if not args.input:
        try:
            args.input = input(
                "\nEnter a video file or folder path:\n> "
            ).strip().strip('"')
        except (EOFError, KeyboardInterrupt):
            print()
            return 1

    input_path = Path(args.input).expanduser().resolve()

    if not input_path.exists():
        logging.error("Path does not exist: %s", input_path)
        return 2

    config = _build_config(args)

    return run_pipeline(input_path, config)


if __name__ == "__main__":
    raise SystemExit(main())
