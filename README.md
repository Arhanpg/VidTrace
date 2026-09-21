<div align="center">

# VidTrace

**Extract the timeline of what was said and what appeared on screen.**

Local-first, GPU-accelerated multimodal video extraction for speech, OCR, screen events, code, and timestamped knowledge.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

</div>

---

## What is VidTrace?

VidTrace turns any recorded video into a **searchable, timestamped, multimodal knowledge artifact** by understanding both what was said and what appeared on screen.

```
Video
  ├── Audio Track ──→ Speech Transcription (faster-whisper)
  │                        │
  └── Video Frames ──→ Adaptive Sampling ──→ OCR (PaddleOCR)
                           │                      │
                           └── Scene Detection     ├── Code Detection
                                                   └── Screen Events
                                    │
                                    ▼
                          Temporal Fusion Engine
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
               Markdown          JSON/JSONL     HTML Timeline
               Report            Events         Viewer
                                    │
                                    ▼
                               SRT / VTT
                               Subtitles
```

**Output for each video:**
```
vidtrace_output/
  └── My_Lecture/
      ├── My_Lecture__analysis.md      # Full analysis report
      ├── transcript.json              # Timestamped transcript
      ├── transcript.md                # Human-readable transcript
      ├── transcript.srt               # Subtitles (SRT)
      ├── transcript.vtt               # Subtitles (WebVTT)
      ├── ocr_events.json              # Deduplicated OCR events
      ├── ocr_all_candidates.jsonl     # Raw OCR stream
      ├── timeline.json                # Unified multimodal timeline
      ├── events.jsonl                 # Timeline as JSONL
      ├── timeline.html                # Interactive HTML viewer
      ├── state.json                   # Pipeline state/progress
      └── evidence_frames/             # Timestamped screenshots
          ├── 000001_000120.500s.jpg
          ├── 000002_000125.000s.jpg
          └── ...
```

## Quick Start

### Install

```bash
pip install vidtrace
```

### Run

```bash
# Single video
vidtrace lecture.mp4

# Folder of videos
vidtrace ./recordings/

# With a preset
vidtrace lecture.mp4 --preset coding

# With specific output formats
vidtrace video.mp4 --output-format md,json,html,srt
```

### Example Output

**Unified Timeline Event:**
```json
{
  "timestamp": 82.43,
  "type": "speech",
  "source": "whisper",
  "text": "Now we create the state graph",
  "metadata": {
    "start": 82.43,
    "end": 84.10,
    "index": 42,
    "word_count": 6
  }
}
```

```json
{
  "timestamp": 84.12,
  "type": "code",
  "source": "paddleocr",
  "text": "graph = StateGraph(AgentState)\ngraph.add_node(...)",
  "metadata": {
    "event_id": 15,
    "mean_confidence": 0.934,
    "evidence_frame": "evidence_frames/000015_000084.120s.jpg"
  }
}
```

## Presets

VidTrace includes presets tuned for common video types:

| Preset | OCR Density | Best For |
|--------|------------|----------|
| `lecture` | Normal (2.0s) | University lectures, slide-heavy |
| `coding` | Dense (1.5s) | Programming tutorials, live coding |
| `meeting` | Light (5.0s) | Recorded meetings, mostly speech |
| `tutorial` | Normal (2.0s) | Software demos, walkthroughs |

```bash
vidtrace video.mp4 --preset coding
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    CLI / Config                       │
├──────────────────────────────────────────────────────┤
│                Pipeline Orchestrator                  │
│                                                      │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │   Audio     │  │   Vision    │  │   Fusion     │  │
│  │            │  │             │  │              │  │
│  │ Whisper    │  │ Sampler     │  │ Timeline     │  │
│  │ Transcribe │  │ OCR Engine  │  │ Alignment    │  │
│  │            │  │ Preprocess  │  │              │  │
│  └────────────┘  └─────────────┘  └──────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────────┐│
│  │              Output Renderers                    ││
│  │  Markdown │ JSON/JSONL │ HTML │ SRT/VTT         ││
│  └──────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────┤
│          Models (Events, Config) + Utils             │
└──────────────────────────────────────────────────────┘
```

## GPU Strategy

VidTrace is designed for **low-VRAM GPUs** (tested on RTX 3050 4 GB):

1. **Whisper** gets the GPU first → transcription completes
2. GPU memory is fully released
3. **PaddleOCR** then gets the GPU → OCR processing
4. CPU fallback available for every stage

```bash
# Force CPU-only OCR
vidtrace video.mp4 --cpu-ocr

# Use a smaller Whisper model
vidtrace video.mp4 --whisper-model tiny
```

## Configuration

### CLI Options

```bash
vidtrace video.mp4 \
  --whisper-model small \
  --language en \
  --sample-interval 2.0 \
  --active-interval 0.5 \
  --scene-threshold 0.065 \
  --output-format md,json,html,srt \
  --output-folder my_output \
  --force
```

### YAML Config File

```yaml
# config.yaml
whisper_model: small
language: en
sample_interval: 2.0
active_interval: 0.50
output_formats:
  - md
  - json
  - html
```

```bash
vidtrace video.mp4 --config config.yaml
```

## Supported Video Types

| Use Case | What VidTrace Extracts |
|----------|----------------------|
| **Programming lectures** | Transcript + source code OCR + terminal output + timestamps |
| **YouTube tutorials** | Speech + screen text + chapters + Markdown |
| **Software demos** | UI text + click sequences + timestamps |
| **University lectures** | Speech + slides + equations + diagrams |
| **Conference talks** | Speaker transcript + presentation slides + key moments |
| **Corporate training** | Transcript + screen text + key moments + notes |

## Project Structure

```
VidTrace/
├── src/vidtrace/
│   ├── audio/           # Speech transcription (Whisper)
│   ├── vision/          # Frame sampling, OCR, preprocessing
│   ├── fusion/          # Timeline fusion and alignment
│   ├── pipeline/        # Orchestration, GPU management
│   ├── output/          # Markdown, JSON, HTML, SRT renderers
│   ├── models/          # Data models, config, presets
│   ├── utils.py         # Shared utilities
│   └── cli.py           # Command-line interface
├── tests/               # Unit tests
├── configs/presets/     # YAML preset configurations
├── docs/                # Documentation
├── pyproject.toml       # Package configuration
├── README.md
├── CONTRIBUTING.md
└── LICENSE              # Apache-2.0
```

## Roadmap

### v0.1 ✅ (Current)
- Video ingestion (mp4, mkv, mov, avi, webm, m4v)
- Speech transcription (faster-whisper, GPU + CPU)
- Screen OCR (PaddleOCR with enhancement retries)
- Adaptive visual sampling with scene detection
- Temporal multimodal alignment
- Unified event schema
- Output: Markdown, JSON, JSONL, HTML viewer, SRT/VTT
- Presets (lecture, coding, meeting, tutorial)
- Resume/checkpoint support

### v0.2
- [ ] Plugin architecture for custom extractors
- [ ] Speaker diarization
- [ ] Slide detection
- [ ] URL/command/equation extraction
- [ ] Better code block detection

### v0.3
- [ ] Local multimodal models (Qwen-VL, etc.)
- [ ] Semantic search over timeline
- [ ] Embeddings and RAG interface

### v0.4
- [ ] Web UI with interactive timeline
- [ ] MCP server for agent access
- [ ] Model registry

### v1.0
- [ ] Stable Python API
- [ ] Comprehensive documentation
- [ ] Benchmarks
- [ ] Contributor ecosystem

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[Apache License 2.0](LICENSE)

---

<div align="center">
<sub>Built with 🎬 by <a href="https://github.com/Arhanpg">Arhan</a></sub>
</div>
