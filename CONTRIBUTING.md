# Contributing to VidTrace

Thank you for your interest in contributing to VidTrace! This guide covers everything you need to get started.

## Development Setup

### Prerequisites

- Python 3.10+
- Git

### Clone and Install

```bash
git clone https://github.com/Arhanpg/VidTrace.git
cd VidTrace

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
source .venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/ -v
```

### Lint and Type Check

```bash
ruff check src/
mypy src/vidtrace/
```

### Auto-fix Lint Issues

```bash
ruff check --fix src/
```

## Architecture Overview

VidTrace follows a modular pipeline architecture:

```
CLI (cli.py)
  → Config (models/config.py)
  → Pipeline Orchestrator (pipeline/orchestrator.py)
      → Stage 1: Audio Transcription (audio/transcription.py)
      → Stage 2: Visual OCR (vision/ocr.py, vision/sampler.py)
      → Stage 3: Temporal Fusion (fusion/timeline.py)
      → Stage 4: Output Rendering (output/*.py)
```

### Key Design Decisions

1. **Unified Event Schema**: All data flows through `TimelineEvent` objects — speech, OCR, and visual events share a common structure.

2. **Sequential GPU Access**: On low-VRAM GPUs, Whisper runs first and releases memory before PaddleOCR starts. This is intentional.

3. **Adaptive Sampling**: Instead of OCR-ing every frame, the sampler watches for visual changes and increases density only when the screen is changing.

4. **OCR Deduplication**: Character bigram similarity prevents storing near-identical consecutive OCR results.

## How to Add a New Extractor

The pipeline is designed around extractors. To add a new one:

1. Create a module in the appropriate package (e.g., `vision/slide_detector.py`)
2. Define a class that processes frames or text and produces `TimelineEvent` objects
3. Integrate it into `pipeline/orchestrator.py`
4. Add tests in `tests/`

### Example: Slide Detector

```python
# src/vidtrace/vision/slide_detector.py

from vidtrace.models.events import TimelineEvent

class SlideDetector:
    def detect(self, frame, timestamp):
        # Your logic here
        if is_slide_change:
            return TimelineEvent(
                timestamp=timestamp,
                type="slide",
                source="slide_detector",
                text=f"Slide {slide_number}",
                metadata={"slide_number": slide_number},
            )
```

## How to Add a New Output Format

1. Create a module in `src/vidtrace/output/` (e.g., `csv.py`)
2. Implement an export function that takes timeline data
3. Register the format in `pipeline/orchestrator.py`'s `_render_outputs()` method
4. Add the format name to the config's `output_formats` list

## PR Guidelines

1. **One feature per PR** — keep changes focused
2. **Include tests** for new functionality
3. **Run lint** before submitting: `ruff check src/`
4. **Update docs** if adding new CLI flags or features
5. **Use type hints** for function signatures
6. **Follow existing patterns** — look at how similar code is structured

## Reporting Issues

When reporting bugs, please include:

- Python version (`python --version`)
- GPU model (if applicable)
- Full error traceback
- Video format and approximate size
- The command you ran

## Code Style

- **Formatter**: ruff
- **Type hints**: Yes, for all public function signatures
- **Logging**: Use `logging.getLogger(__name__)` instead of `print()`
- **Docstrings**: Google-style for public classes and functions

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
