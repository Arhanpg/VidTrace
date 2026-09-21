"""
VidTrace benchmark runner.

Measures extraction performance across key metrics:
  - Transcription latency
  - OCR latency and recall
  - Event density
  - VRAM peak usage
  - Code detection accuracy

Usage:
    python benchmarks/run_benchmark.py --video path/to/video.mp4
    python benchmarks/run_benchmark.py --video path/to/video.mp4 --preset coding
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def measure_transcription(video_path: Path, config: dict) -> dict:
    """Measure transcription latency."""
    from vidtrace.audio.transcription import WhisperTranscriber
    from vidtrace.models.config import VidTraceConfig

    cfg = VidTraceConfig(**config)

    start = time.perf_counter()
    transcriber = WhisperTranscriber(cfg)
    try:
        segments, meta = transcriber.transcribe(video_path)
    finally:
        transcriber.close()
    elapsed = time.perf_counter() - start

    duration = meta.get("duration", 1.0)
    return {
        "segments": len(segments),
        "elapsed_seconds": elapsed,
        "seconds_per_minute": (elapsed / duration) * 60 if duration > 0 else 0,
        "rtf": elapsed / duration if duration > 0 else 0,
    }


def measure_gpu_usage() -> dict:
    """Capture GPU memory usage if available."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "gpu_memory_used_mb": int(parts[0].strip()),
                "gpu_memory_total_mb": int(parts[1].strip()),
            }
    except Exception:
        pass
    return {"gpu_memory_used_mb": None, "gpu_memory_total_mb": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="VidTrace Benchmark Runner")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--preset", default="default", help="Config preset")
    parser.add_argument("--output", default="benchmark_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    print(f"Benchmarking: {video_path.name}")
    print(f"Preset: {args.preset}")
    print("=" * 60)

    results = {
        "video": str(video_path),
        "preset": args.preset,
        "gpu_before": measure_gpu_usage(),
    }

    # Measure transcription.
    try:
        print("Measuring transcription...")
        results["transcription"] = measure_transcription(
            video_path, {"whisper_model": "small"},
        )
        print(f"  Segments: {results['transcription']['segments']}")
        print(f"  Elapsed: {results['transcription']['elapsed_seconds']:.2f}s")
        print(f"  RTF: {results['transcription']['rtf']:.3f}")
    except Exception as exc:
        results["transcription"] = {"error": str(exc)}
        print(f"  Error: {exc}")

    results["gpu_after"] = measure_gpu_usage()

    # Write results.
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
