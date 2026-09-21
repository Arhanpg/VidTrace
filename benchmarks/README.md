# VidTrace Benchmarks

Measuring extraction quality and performance across video types.

## Metrics

| Metric | Description |
|--------|-------------|
| Transcription latency | Seconds per minute of video (Whisper) |
| OCR latency | Seconds per minute of video (PaddleOCR) |
| OCR recall | % of visible text captured |
| Event count | Timeline events per minute |
| VRAM peak | Maximum GPU memory usage |
| CPU usage | Average CPU utilization |

## Code-Specific Metrics

| Metric | Description |
|--------|-------------|
| Code detection precision | Correctly identified code blocks / total detected |
| Code reconstruction accuracy | Character-level accuracy vs ground truth |
| Language detection accuracy | Correctly identified language / total code blocks |

## Running Benchmarks

```bash
python benchmarks/run_benchmark.py --video path/to/video.mp4 --preset coding
```

## Video Categories

| Category | Characteristics |
|----------|----------------|
| `coding/` | IDE/editor with code typing, terminal output |
| `lecture/` | Slides with text, formulas, diagrams |
| `meeting/` | Shared screens, multiple speakers |
| `tutorial/` | Mixed code + slides + browser |

## Adding Benchmark Videos

1. Use short clips (30-120 seconds) for reproducibility
2. Ensure proper licensing (CC-BY or public domain)
3. Create a ground-truth annotation file (see `golden/README.md`)
4. Add to the appropriate category directory
