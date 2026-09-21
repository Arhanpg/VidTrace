# VidTrace

**Extract what was said and what appeared on screen.**

A local-first, GPU-accelerated pipeline that converts any recorded video into a searchable, timestamped, multimodal knowledge artifact — by understanding both **what was spoken** and **what was visible**.

```
Speech ───────┐
OCR ──────────┤
Scenes ───────┤
Code ─────────┼──→ Unified Timeline ──→ Search / Export / RAG
Screenshots ──┘
```

---

## Quick Start

```bash
pip install vidtrace

# Full extraction
vidtrace lecture.mp4

# With a preset
vidtrace run coding_tutorial.mp4 --preset coding

# Transcribe only
vidtrace transcribe meeting.mp4

# Search across extracted data
vidtrace search vidtrace_output/ "StateGraph"

# Check pipeline status
vidtrace status lecture.mp4
```

### Output

```
vidtrace_output/
├── lecture/
│   ├── transcript.json          # Timestamped speech segments
│   ├── transcript.md            # Readable transcript
│   ├── transcript.srt           # SRT subtitles
│   ├── transcript.vtt           # WebVTT subtitles
│   ├── ocr_events.json          # Deduplicated screen text
│   ├── timeline.json            # Unified multimodal timeline
│   ├── events.jsonl             # Streaming event format
│   ├── timeline.html            # Interactive viewer
│   ├── lecture__analysis.md     # Full analysis report
│   ├── state.json               # Pipeline checkpoint
│   └── evidence_frames/         # Timestamped screenshots
│       ├── 000000_014.220s.jpg
│       ├── 000001_018.410s.jpg
│       └── ...
└── run_summary.json
```

---

## Why VidTrace?

Most video tools treat video as **audio + frames**. VidTrace treats it as a **temporal multimodal stream**.

| Approach | What you get |
|----------|-------------|
| YouTube transcript | Just speech, no screen content |
| Screenshot OCR | Just one frame, no temporal context |
| Video summarizer | Abstract summary, no evidence |
| **VidTrace** | **Timestamped speech + screen text + code + visual changes, cross-referenced** |

**Use cases:**
- 🎓 **Lectures** — Extract slides, formulas, and spoken explanations
- 💻 **Coding tutorials** — Capture code as it's typed, with temporal evolution
- 🏢 **Meetings** — Searchable transcript + shared screen content
- 📹 **Product demos** — Index UI states and narration
- 🎙️ **Conference talks** — Slides + speaker notes + Q&A

---

## Installation

### Core (no ML dependencies)
```bash
pip install vidtrace
```

### With speech transcription
```bash
pip install "vidtrace[whisper]"
```

### With screen OCR
```bash
pip install "vidtrace[ocr]"
```

### Everything
```bash
pip install "vidtrace[all]"
```

### GPU Setup

VidTrace uses a **sequential GPU strategy** optimized for low-VRAM setups (tested on RTX 3050 4GB):

1. Whisper gets the GPU → transcribes → releases VRAM
2. PaddleOCR gets the GPU → extracts screen text → releases VRAM

For GPU acceleration:
```bash
# CUDA 11.8
pip install paddlepaddle-gpu

# CUDA 12.x
pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

> **Note:** PaddlePaddle GPU installation is platform-specific. See the [PaddlePaddle install guide](https://www.paddlepaddle.org.cn/install/quick) for your CUDA version.

---

## CLI Reference

### Commands

| Command | Description |
|---------|-------------|
| `vidtrace run <video>` | Full extraction pipeline (default) |
| `vidtrace transcribe <video>` | Speech transcription only |
| `vidtrace ocr <video>` | Screen OCR only (needs transcript) |
| `vidtrace export <video>` | Re-render outputs from existing data |
| `vidtrace status <video>` | Show checkpoint / pipeline progress |
| `vidtrace search <dir> <query>` | Search across extracted timeline |

### Presets

| Preset | Optimized for |
|--------|-------------|
| `lecture` | Slide-heavy presentations (2s sampling, high dedup) |
| `coding` | Code editors (1.5s sampling, dense on changes) |
| `meeting` | Shared screens (5s sampling, SRT output) |
| `tutorial` | Mixed content (2s sampling, all output formats) |

```bash
vidtrace run video.mp4 --preset coding --output-format md,json,html,srt
```

### Checkpoint & Resume

VidTrace automatically checkpoints after each stage. If interrupted:

```bash
# Check what's done
vidtrace status video.mp4

# Resume — completed stages are skipped
vidtrace run video.mp4

# Force re-run everything
vidtrace run video.mp4 --force
```

If the source video file changes, cached results are automatically invalidated.

---

## Architecture

```
CLI
 ↓
Config / Presets
 ↓
Pipeline Orchestrator ──→ Checkpoint System
 ├── Stage 1: Whisper transcription     (GPU → release)
 ├── Stage 2: PaddleOCR + Adaptive Sampling  (GPU → release)
 ├── Stage 3: Temporal Fusion
 └── Stage 4: Output Rendering
      ↓
 Markdown / JSON / HTML / SRT / VTT
```

### Key Design Decisions

- **Unified `TimelineEvent` schema** — All extractors produce the same event type, enabling consistent search and export
- **Adaptive sampling** — Dense OCR when the screen changes, sparse when it's static
- **Sequential GPU** — Whisper first, then PaddleOCR, to fit in 4GB VRAM
- **Checkpoint/resume** — Partial OCR progress is saved every 20 events
- **Plugin-ready** — `Extractor` protocol allows third-party integrations

### Event Types

| Type | Source | Description |
|------|--------|-------------|
| `speech` | Whisper | Transcribed spoken words |
| `ocr` | PaddleOCR | Screen text (non-code) |
| `code` | PaddleOCR | Code-like screen text |
| `scene` | OpenCV | Visual scene change |
| `terminal` | Plugin | Terminal/CLI output |
| `slide` | Plugin | Presentation slide |
| `url` | Plugin | Detected URL |
| `equation` | Plugin | Mathematical formula |

---

## Configuration

### YAML Config File

```yaml
# vidtrace.yaml
whisper_model: small
language: en
sample_interval: 1.5
active_interval: 0.4
active_window: 10.0
scene_threshold: 0.065
ocr_change_similarity: 0.95
output_formats:
  - md
  - json
  - html
  - srt
```

```bash
vidtrace run video.mp4 --config vidtrace.yaml
```

---

## Project Structure

```
src/vidtrace/
├── audio/
│   └── transcription.py       # Whisper engine
├── vision/
│   ├── sampler.py              # Adaptive frame sampling
│   ├── preprocessing.py        # Image enhancement
│   ├── ocr.py                  # PaddleOCR engine
│   ├── code_detector.py        # Code region detection
│   └── code_reconstructor.py   # Temporal code evolution
├── fusion/
│   └── timeline.py             # Multimodal temporal fusion
├── pipeline/
│   ├── orchestrator.py         # Pipeline coordinator
│   ├── checkpoint.py           # Checkpoint & resume
│   ├── gpu.py                  # GPU memory management
│   ├── search.py               # Multimodal search
│   └── extractors.py           # Plugin architecture
├── output/
│   ├── html.py                 # Interactive timeline viewer
│   ├── json_output.py          # JSON/JSONL export
│   ├── markdown.py             # Markdown reports
│   └── srt.py                  # SRT/VTT subtitles
├── models/
│   ├── events.py               # EventType + TimelineEvent
│   └── config.py               # Config + presets
├── utils.py                    # Shared utilities
└── cli.py                      # CLI with subcommands
```

---

## Roadmap

```
v0.1 — Extraction Core ✅
  ✓ Whisper transcription
  ✓ PaddleOCR with adaptive sampling
  ✓ Unified event schema
  ✓ Markdown / JSON / JSONL output
  ✓ GPU memory management

v0.2 — Evidence Layer ✅
  ✓ Checkpoint / resume system
  ✓ Code detection + language identification
  ✓ Temporal code reconstruction
  ✓ Interactive HTML timeline viewer
  ✓ Multimodal search
  ✓ CLI subcommands
  ✓ Plugin architecture

v0.3 — Intelligence
  □ Semantic search with embeddings
  □ Multimodal RAG layer
  □ Speaker diarization
  □ Slide/scene segmentation

v0.4 — Extensibility
  □ Plugin marketplace
  □ Qwen-VL / local VLM adapters
  □ MCP server
  □ Web UI

v1.0 — Production
  □ Stable API
  □ Benchmarks + golden dataset
  □ PyPI release
  □ Documentation site
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, architecture guide, and contribution guidelines.

### Good First Issues

Look for issues labeled `good first issue`:
- Add new output format
- Add URL detector extractor
- Improve language detection
- Add progress bar to CLI

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.
