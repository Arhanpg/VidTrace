"""Convenience wrapper for VidTrace's YouTube transcription subcommand.

Usage:
    python scripts/transcribe_youtube.py "https://www.youtube.com/playlist?list=..."
    python scripts/transcribe_youtube.py "https://youtu.be/..." --language hi
"""
from __future__ import annotations

import sys

from vidtrace.cli import main


if __name__ == "__main__":
    # Allow the short command to omit the "youtube-transcribe" subcommand.
    sys.argv.insert(1, "youtube-transcribe")
    raise SystemExit(main())
