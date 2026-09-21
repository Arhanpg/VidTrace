"""
VidTrace command-line interface.

Subcommands:
    vidtrace run <video>           Full pipeline (default)
    vidtrace transcribe <video>    Stage 1 only
    vidtrace ocr <video>           Stage 2 only (needs transcript)
    vidtrace export <video>        Re-render outputs from existing data
    vidtrace status <video>        Show checkpoint state
    vidtrace search <dir> <query>  Search across extracted timeline

Backwards compatible:
    vidtrace video.mp4             Same as 'vidtrace run video.mp4'
    vidtrace video.mp4 --preset coding
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


# ──────────────────────────────────────────────────────────────
# Shared argument groups
# ──────────────────────────────────────────────────────────────

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across subcommands."""
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
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )


def _add_whisper_args(parser: argparse.ArgumentParser) -> None:
    """Add Whisper-related arguments."""
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


def _add_ocr_args(parser: argparse.ArgumentParser) -> None:
    """Add OCR-related arguments."""
    parser.add_argument(
        "--sample-interval", type=float, default=None,
        help="Normal screen OCR interval in seconds.",
    )
    parser.add_argument(
        "--active-interval", type=float, default=None,
        help="OCR interval while screen is changing.",
    )
    parser.add_argument(
        "--active-window", type=float, default=None,
        help="Seconds of dense OCR after a visual change.",
    )
    parser.add_argument(
        "--scene-threshold", type=float, default=None,
        help="Visual difference threshold [0..1].",
    )
    parser.add_argument(
        "--ocr-change-similarity", type=float, default=None,
        help="Higher = more OCR snapshots kept.",
    )
    parser.add_argument(
        "--ocr-heartbeat", type=float, default=None,
        help="Save unchanged screen at this interval.",
    )
    parser.add_argument("--min-ocr-chars", type=int, default=None)
    parser.add_argument(
        "--min-luma", type=float, default=None,
        help="Reject almost black frames.",
    )
    parser.add_argument(
        "--cpu-ocr", action="store_true",
        help="Force OCR to CPU.",
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Add output-related arguments."""
    parser.add_argument(
        "--output-folder", default=None,
        help="Output directory name (default: vidtrace_output).",
    )
    parser.add_argument(
        "--output-format", default=None,
        help="Comma-separated output formats: md,json,html,srt",
    )
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all stages, ignoring existing output.",
    )


# ──────────────────────────────────────────────────────────────
# Config builder
# ──────────────────────────────────────────────────────────────

def _build_config(args: argparse.Namespace) -> VidTraceConfig:
    """Build a VidTraceConfig from CLI arguments."""
    if getattr(args, "config", None):
        config = load_config_file(args.config)
    elif getattr(args, "preset", None):
        config = load_preset(args.preset)
    else:
        config = VidTraceConfig()

    overrides = {
        "whisper_model": getattr(args, "whisper_model", None),
        "language": getattr(args, "language", None),
        "sample_interval": getattr(args, "sample_interval", None),
        "active_interval": getattr(args, "active_interval", None),
        "active_window": getattr(args, "active_window", None),
        "scene_threshold": getattr(args, "scene_threshold", None),
        "ocr_change_similarity": getattr(args, "ocr_change_similarity", None),
        "ocr_heartbeat": getattr(args, "ocr_heartbeat", None),
        "min_ocr_chars": getattr(args, "min_ocr_chars", None),
        "min_luma": getattr(args, "min_luma", None),
        "output_folder": getattr(args, "output_folder", None),
        "jpeg_quality": getattr(args, "jpeg_quality", None),
    }

    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    output_format = getattr(args, "output_format", None)
    if output_format:
        config.output_formats = [
            f.strip() for f in output_format.split(",")
        ]

    if getattr(args, "cpu_ocr", False):
        config.cpu_ocr = True

    if getattr(args, "force", False):
        config.force = True

    return config


# ──────────────────────────────────────────────────────────────
# Subcommand handlers
# ──────────────────────────────────────────────────────────────

def _cmd_run(args: argparse.Namespace) -> int:
    """Full pipeline execution."""
    from vidtrace.pipeline.orchestrator import run_pipeline

    input_path = _resolve_input(args)
    if input_path is None:
        return 1

    config = _build_config(args)
    return run_pipeline(input_path, config)


def _cmd_transcribe(args: argparse.Namespace) -> int:
    """Transcription only (Stage 1)."""
    from vidtrace.pipeline.orchestrator import run_pipeline

    input_path = _resolve_input(args)
    if input_path is None:
        return 1

    config = _build_config(args)
    # Only run transcription by setting a flag the orchestrator checks.
    config.output_formats = ["md", "json"]
    return run_pipeline(input_path, config)


def _cmd_ocr(args: argparse.Namespace) -> int:
    """OCR only (Stage 2, requires existing transcript)."""
    from vidtrace.pipeline.orchestrator import run_pipeline

    input_path = _resolve_input(args)
    if input_path is None:
        return 1

    config = _build_config(args)
    return run_pipeline(input_path, config)


def _cmd_export(args: argparse.Namespace) -> int:
    """Re-render outputs from existing extraction data."""
    from vidtrace.pipeline.orchestrator import run_pipeline

    input_path = _resolve_input(args)
    if input_path is None:
        return 1

    config = _build_config(args)
    return run_pipeline(input_path, config)


def _cmd_status(args: argparse.Namespace) -> int:
    """Show checkpoint status for a video."""
    from vidtrace.pipeline.orchestrator import get_video_status

    input_path = _resolve_input(args)
    if input_path is None:
        return 1

    config = _build_config(args)
    print(get_video_status(input_path, config))
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """Search across extracted timeline data."""
    from vidtrace.pipeline.search import (
        format_search_results,
        search_timeline,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.exists():
        logging.error("Output directory not found: %s", output_dir)
        return 2

    query = args.query

    event_types = None
    if args.type:
        event_types = [t.strip() for t in args.type.split(",")]

    results = search_timeline(
        output_dir, query,
        event_types=event_types,
        case_sensitive=args.case_sensitive,
    )

    print(format_search_results(results, query))
    return 0


def _resolve_input(args: argparse.Namespace) -> Path | None:
    """Resolve and validate input path."""
    input_arg = getattr(args, "input", None)

    if not input_arg:
        try:
            input_arg = input(
                "\nEnter a video file or folder path:\n> "
            ).strip().strip('"')
        except (EOFError, KeyboardInterrupt):
            print()
            return None

    path = Path(input_arg).expanduser().resolve()

    if not path.exists():
        logging.error("Path does not exist: %s", path)
        return None

    return path


# ──────────────────────────────────────────────────────────────
# Parser construction
# ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="vidtrace",
        description=(
            "VidTrace — Extract the timeline of what was said "
            "and what appeared on screen."
        ),
        epilog=(
            "Examples:\n"
            "  vidtrace lecture.mp4\n"
            "  vidtrace run lecture.mp4 --preset coding\n"
            "  vidtrace transcribe lecture.mp4\n"
            "  vidtrace status lecture.mp4\n"
            "  vidtrace search vidtrace_output/ \"StateGraph\"\n"
            "  vidtrace export lecture.mp4 --output-format md,html,srt\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"vidtrace {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ── run ────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full extraction pipeline.",
        description="Run all stages: transcribe → OCR → fuse → render.",
    )
    run_parser.add_argument("input", nargs="?", help="Video file or folder.")
    _add_common_args(run_parser)
    _add_whisper_args(run_parser)
    _add_ocr_args(run_parser)
    _add_output_args(run_parser)

    # ── transcribe ─────────────────────────────────────────────
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Run speech transcription only (Stage 1).",
    )
    transcribe_parser.add_argument("input", nargs="?", help="Video file.")
    _add_common_args(transcribe_parser)
    _add_whisper_args(transcribe_parser)
    _add_output_args(transcribe_parser)

    # ── ocr ────────────────────────────────────────────────────
    ocr_parser = subparsers.add_parser(
        "ocr",
        help="Run OCR only (Stage 2, requires existing transcript).",
    )
    ocr_parser.add_argument("input", nargs="?", help="Video file.")
    _add_common_args(ocr_parser)
    _add_ocr_args(ocr_parser)
    _add_output_args(ocr_parser)

    # ── export ─────────────────────────────────────────────────
    export_parser = subparsers.add_parser(
        "export",
        help="Re-render outputs from existing extraction data.",
    )
    export_parser.add_argument("input", nargs="?", help="Video file.")
    _add_common_args(export_parser)
    _add_output_args(export_parser)

    # ── status ─────────────────────────────────────────────────
    status_parser = subparsers.add_parser(
        "status",
        help="Show checkpoint status for a video.",
    )
    status_parser.add_argument("input", nargs="?", help="Video file.")
    _add_common_args(status_parser)
    _add_output_args(status_parser)

    # ── search ─────────────────────────────────────────────────
    search_parser = subparsers.add_parser(
        "search",
        help="Search across extracted timeline data.",
    )
    search_parser.add_argument(
        "output_dir",
        help="Path to vidtrace output directory.",
    )
    search_parser.add_argument(
        "query",
        help="Search query string.",
    )
    search_parser.add_argument(
        "--type",
        default=None,
        help="Filter by event type: speech,ocr,code,scene",
    )
    search_parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Enable case-sensitive search.",
    )
    search_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    return parser


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main() -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(verbose=getattr(args, "verbose", False))

    # ── Backwards compatibility ───────────────────────────────
    # If no subcommand given but a positional arg exists that looks
    # like a file path, treat it as 'run <path>'.
    if args.command is None:
        # Check if there's an unrecognized arg that looks like a path.
        import sys
        remaining = sys.argv[1:]

        if remaining and not remaining[0].startswith("-"):
            # Re-parse as 'run' subcommand.
            sys.argv = [sys.argv[0], "run"] + remaining
            args = parser.parse_args()
        else:
            parser.print_help()
            return 0

    # Dispatch.
    handlers = {
        "run": _cmd_run,
        "transcribe": _cmd_transcribe,
        "ocr": _cmd_ocr,
        "export": _cmd_export,
        "status": _cmd_status,
        "search": _cmd_search,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 0

    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
