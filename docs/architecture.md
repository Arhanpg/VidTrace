# VidTrace Architecture

## System Overview

VidTrace is a **temporal multimodal index for video**. It processes recorded video through a staged extraction pipeline, producing a unified timeline of speech, screen text, code, and visual events.

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLI / API                              │
├─────────────────────────────────────────────────────────────────┤
│                     Pipeline Orchestrator                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Stage 1  │→ │ Stage 2  │→ │ Stage 3  │→ │   Stage 4     │  │
│  │ Whisper  │  │ OCR +    │  │ Temporal │  │ Output        │  │
│  │          │  │ Code Det │  │ Fusion   │  │ Rendering     │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│       ↕              ↕             ↕              ↕            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Checkpoint System (state.json)              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
   transcript.json      timeline.json         timeline.html
   transcript.md        events.jsonl          transcript.srt
   transcript.srt       ocr_events.json       analysis.md
```

## Core Abstractions

### EventType Enum

All events in VidTrace are categorized by type:

| Type | Producer | Description |
|------|----------|-------------|
| `speech` | Whisper | Transcribed speech segment |
| `ocr` | PaddleOCR | Non-code screen text |
| `code` | PaddleOCR + CodeDetector | Code-like screen text |
| `scene` | OpenCV | Visual scene change |
| `visual` | OpenCV | Generic visual event |
| `terminal` | (plugin) | Terminal/CLI output |
| `slide` | (plugin) | Presentation slide |
| `url` | (plugin) | Detected URL |
| `equation` | (plugin) | Mathematical formula |

### TimelineEvent

The unified event schema all extractors produce:

```python
@dataclass
class TimelineEvent:
    id: str                    # Unique 12-char hex ID
    timestamp: float           # Seconds from video start
    duration: float | None     # Duration (None for point events)
    type: str                  # EventType value
    source: str                # Extractor name
    text: str | None           # Extracted content
    confidence: float | None   # [0.0, 1.0]
    frame_path: str | None     # Evidence screenshot path
    metadata: dict[str, Any]   # Type-specific extra data
```

### VidTraceConfig

Centralized configuration with named presets:

```python
config = load_preset("coding")
# Or from YAML:
config = load_config_file("vidtrace.yaml")
```

## Pipeline Stages

### Stage 1: Transcription

- Engine: faster-whisper (CTranslate2)
- GPU: Allocated first, released completely before Stage 2
- Output: `transcript.json`, `transcript.md`
- Resume: Skipped if `transcript.json` exists

### Stage 2: OCR with Adaptive Sampling

- Engine: PaddleOCR
- Sampling: Adaptive — dense when screen changes, sparse when static
- Deduplication: Bigram similarity prevents duplicate snapshots
- Code detection: `CodeDetector` identifies code regions and programming language
- Output: `ocr_events.json`, evidence frames
- Resume: Partial — saves progress every 20 events

### Stage 3: Temporal Fusion

- Merges speech and OCR events chronologically
- Aligns OCR events with nearby speech context
- Output: `timeline.json`, `events.jsonl`

### Stage 4: Output Rendering

- Markdown analysis report
- Interactive HTML timeline viewer
- SRT/VTT subtitles
- JSON/JSONL for programmatic access

## Checkpoint System

```json
{
  "video": { "size_bytes": 123456, "head_hash": "a1b2c3d4..." },
  "stages": {
    "transcription": { "status": "complete" },
    "ocr": { "status": "partial" }
  },
  "ocr_last_timestamp": 42.5,
  "ocr_last_frame": 1275,
  "ocr_event_count": 5
}
```

Key behaviors:
- **Video identity**: Hash of file size + mtime + first 8KB
- **Cache invalidation**: All stages re-run if video changes
- **Partial OCR resume**: Seek to last checkpoint on restart
- **Interrupt safety**: Ctrl+C saves partial OCR progress

## GPU Memory Strategy

Optimized for low-VRAM (4GB) setups:

```
Whisper loads → GPU allocated
Whisper runs  → transcription
Whisper done  → GPU released (torch.cuda.empty_cache)
OCR loads     → GPU allocated
OCR runs      → frame-by-frame extraction
OCR done      → GPU released (paddle.device.cuda.empty_cache)
```

## Plugin Architecture

Any class implementing the `Extractor` protocol can produce events:

```python
class Extractor(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def event_types(self) -> list[str]: ...
    def extract(self, context: VideoContext) -> list[TimelineEvent]: ...
    def is_available(self) -> bool: ...
```

Built-in: `WhisperExtractor`, `OCRExtractor`, `SceneExtractor`, `CodeExtractor`

Future plugins: `SpeakerDiarization`, `QwenVL`, `YOLO`, `EquationOCR`, `URLDetector`

## Code Detection

Beyond the basic `looks_like_code()` heuristic:

1. **Region grouping**: OCR boxes → spatially contiguous code blocks
2. **Language detection**: Weighted pattern matching across 12 languages
3. **Indentation preservation**: Maintains code structure from OCR
4. **Temporal reconstruction**: Cross-frame diffing tracks code evolution

## Search

Multimodal search across the unified timeline:

```bash
vidtrace search output/ "StateGraph"
```

Matches across speech, OCR, and code events simultaneously.
