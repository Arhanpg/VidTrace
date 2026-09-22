# VidTrace

**Extract what was said and what appeared on screen.**

A local-first, GPU-accelerated pipeline that converts recorded video into a searchable, timestamped, multimodal knowledge artifact by understanding both **what was spoken** and **what was visible**.

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

# Coding preset
vidtrace run coding_tutorial.mp4 --preset coding

# Transcribe only
vidtrace transcribe meeting.mp4

# OCR / vision stage
vidtrace ocr lecture.mp4

# Search extracted timeline
vidtrace search vidtrace_output/ "StateGraph"

# Inspect checkpoint state
vidtrace status lecture.mp4
```

## Output

```
vidtrace_output/
├── lecture/
│   ├── transcript.json
│   ├── transcript.md
│   ├── transcript.srt
│   ├── transcript.vtt
│   ├── ocr_events.json
│   ├── timeline.json
│   ├── events.jsonl
│   ├── timeline.html
│   ├── lecture__analysis.md
│   ├── state.json
│   └── evidence_frames/
└── run_summary.json
```

---

## Why VidTrace?

Most video tools treat video as **audio + frames**. VidTrace treats it as a **temporal multimodal stream**.

| Approach | What you get |
|----------|-------------|
| Transcript only | Speech, without screen context |
| Screenshot OCR | Text from isolated frames |
| Video summarizer | Abstract summary, limited evidence |
| **VidTrace** | **Timestamped speech + screen text + code/visual events + evidence frames** |

### Use cases

- 🎓 Lectures and classroom recordings
- 💻 Coding tutorials and live coding
- 🏢 Meetings and shared-screen recordings
- 📹 Software/product demonstrations
- 🎙️ Conference talks and presentations
- 🧑‍🏫 Technical training and workshops

---

## What VidTrace Extracts

### Speech

faster-whisper produces timestamped transcript segments and word-level timestamps.

### Screen text

PaddleOCR analyzes sampled frames, preserves OCR confidence and bounding boxes, and stores evidence screenshots.

### Temporal visual changes

Adaptive sampling increases visual analysis density when the screen changes and returns to sparse sampling when the screen is stable.

### Code-oriented evidence

Code-like OCR events can be identified and correlated with nearby speech and screenshots. Temporal code reconstruction components can track code changes across observations.

### Unified timeline

Speech and OCR observations are converted to a shared `TimelineEvent` representation so downstream tools can search and consume the extracted timeline consistently.

---

## Real-World Test Case

VidTrace was run locally on **three real Agentic AI lecture recordings** with H.264 video at **1920×1080 / 24 FPS**, using an **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)**. Both faster-whisper CUDA and PaddleOCR CUDA were enabled.

| Recording | Duration | Resolution | FPS | Whisper |
|---|---:|---:|---:|---|
| 2026-08-17 | 01:02:35 | 1920×1080 | 24 | small / CUDA / int8_float16 |
| 2026-08-24 | 00:58:51 | 1920×1080 | 24 | small / CUDA / int8_float16 |
| 2026-09-04 | 01:05:39 | 1920×1080 | 24 | small / CUDA / int8_float16 |

### Run result

```
Videos found:  3
Successful:    3
Failed:        0

GPU:           NVIDIA GeForce RTX 3050 Laptop GPU
Whisper CUDA:  enabled
PaddleOCR CUDA: enabled
```

The recorded end-to-end run time was **36,480.95 seconds (~10.13 hours)**. This is a single local run, not a universal performance claim.

### Per-video OCR result

| Recording | OCR candidate frames | Semantic OCR events | OCR elapsed |
|---|---:|---:|---:|
| 2026-08-17 | 2,362 | 1,918 | 7,074.41 s |
| 2026-08-24 | 2,605 | 1,734 | 23,130.55 s |
| 2026-09-04 | 2,730 | 2,204 | 6,238.26 s |

The generated analysis artifacts include timestamped speech, OCR text, confidence scores, bounding boxes, nearby speech context, evidence-frame screenshots, and code-likelihood annotations.

> **Reproducibility note:** OCR runtime varies with screen content, sampling settings, enhancement retries, model versions, and hardware utilization. These measurements should be treated as one real-world test case rather than a benchmark across machines.

---

## GPU Strategy

VidTrace is designed for low-VRAM local GPUs.

```
Whisper
  ↓
transcription
  ↓
release GPU memory
  ↓
PaddleOCR
  ↓
release GPU memory
  ↓
fusion + output
```

This sequential strategy reduces GPU memory contention on cards such as the RTX 3050 4 GB.

---

## Installation

### Core

```bash
pip install vidtrace
```

### Speech transcription

```bash
pip install "vidtrace[whisper]"
```

### OCR

```bash
pip install "vidtrace[ocr]"
```

### Development

```bash
git clone https://github.com/Arhanpg/VidTrace.git
cd VidTrace
python -m venv .venv
pip install -e ".[dev]"
```

### GPU setup

PaddlePaddle GPU installation is platform- and CUDA-specific. Follow the official PaddlePaddle installation instructions for your platform and supported CUDA runtime.

---

## CLI Reference

| Command | Description |
|---|---|
| `vidtrace run <video>` | Full extraction pipeline |
| `vidtrace transcribe <video>` | Speech transcription |
| `vidtrace ocr <video>` | Visual/OCR stage |
| `vidtrace export <video>` | Re-render outputs |
| `vidtrace status <video>` | Show pipeline state |
| `vidtrace search <dir> <query>` | Search extracted timeline |

### Presets

| Preset | Optimized for |
|---|---|
| `lecture` | Slide-heavy recordings |
| `coding` | IDEs and live coding |
| `meeting` | Shared-screen meetings |
| `tutorial` | Mixed browser/code walkthroughs |

---

## Checkpoint & Resume

VidTrace keeps intermediate artifacts so expensive stages do not need to be repeated unnecessarily.

```
video
  ↓
transcript.json exists?
  ├── yes → reuse transcript
  └── no  → transcribe
  ↓
OCR / visual processing
  ↓
timeline fusion
  ↓
render outputs
```

This is especially important for long recordings where transcription and OCR can dominate runtime.

---

## Architecture

```
CLI
 ↓
Config / Presets
 ↓
Pipeline Orchestrator
 ├── Stage 1: Whisper transcription        (GPU → release)
 ├── Stage 2: PaddleOCR + adaptive vision  (GPU → release)
 ├── Stage 3: Temporal fusion
 └── Stage 4: Output rendering
      ↓
 Markdown / JSON / JSONL / HTML / SRT / VTT
```

### Key design decisions

- **Unified event schema** for downstream consumers
- **Adaptive sampling** for efficient visual analysis
- **Sequential GPU access** for low-VRAM devices
- **Checkpoint/resume** for long-running jobs
- **Temporal code reconstruction** for code-heavy recordings
- **Plugin-ready extractor architecture**

---

## Project Structure

```
src/vidtrace/
├── audio/
│   └── transcription.py
├── vision/
│   ├── sampler.py
│   ├── preprocessing.py
│   ├── ocr.py
│   ├── code_detector.py
│   └── code_reconstructor.py
├── fusion/
│   └── timeline.py
├── pipeline/
│   ├── orchestrator.py
│   ├── checkpoint.py
│   ├── gpu.py
│   ├── search.py
│   └── extractors.py
├── models/
│   ├── events.py
│   └── config.py
├── utils.py
└── cli.py
```

---

## Development & Testing

The repository includes unit tests, benchmark scaffolding, issue templates, contributing guidance, and GitHub Actions CI.

```bash
pytest tests/ -v
ruff check src/
mypy src/vidtrace/
```

Benchmark harness:

```bash
python benchmarks/run_benchmark.py --video path/to/video.mp4 --preset coding
```

---

## Roadmap

### v0.1 — Extraction Core
- [x] Whisper transcription
- [x] PaddleOCR
- [x] Adaptive sampling
- [x] Unified event schema
- [x] Markdown / JSON / JSONL output
- [x] GPU management

### v0.2 — Evidence Layer
- [x] Checkpoint / resume
- [x] Code detection
- [x] Temporal code reconstruction
- [x] CLI stages
- [x] Multimodal search
- [x] Extractor abstraction

### v0.3 — Intelligence
- [ ] Semantic embeddings
- [ ] Multimodal RAG
- [ ] Speaker diarization
- [ ] Slide / scene segmentation
- [ ] Better OCR-to-code correction

### v0.4 — Extensibility
- [ ] Local VLM adapters
- [ ] MCP server
- [ ] Web UI
- [ ] External plugin ecosystem

### v1.0 — Production
- [ ] Stable Python API
- [ ] Public benchmark dataset
- [ ] PyPI release
- [ ] Documentation site

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

Good contribution areas include new extractors, OCR/code improvements, output formats, benchmark datasets, search backends, GPU optimizations, tests, and documentation.

---

## License

Apache 2.0 — See [LICENSE](LICENSE).
