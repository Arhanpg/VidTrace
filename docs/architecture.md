# Architecture

## Pipeline Stages

VidTrace processes video in four sequential stages:

### Stage 1: Audio Transcription

- Extract audio track from video
- Run faster-whisper with word-level timestamps
- Voice Activity Detection (VAD) filters silence
- Output: `transcript.json`, `transcript.md`

### Stage 2: Visual OCR

- Adaptive frame sampling (normal + dense modes)
- Scene change detection via frame signatures
- PaddleOCR with automatic enhancement retries
- Semantic deduplication (character bigram similarity)
- Evidence frame screenshots
- Output: `ocr_events.json`, `ocr_all_candidates.jsonl`, `evidence_frames/`

### Stage 3: Temporal Fusion

- Convert all observations to unified `TimelineEvent` schema
- Merge speech + OCR chronologically
- Align speech context with OCR events (±10s window)
- Output: `timeline.json`, `events.jsonl`

### Stage 4: Output Rendering

- Markdown analysis report with full transcript, OCR timeline, bounding boxes
- Interactive HTML timeline viewer with search and filtering
- SRT/VTT subtitle files
- Master report linking all processed videos

## Unified Event Schema

Every observation becomes a `TimelineEvent`:

```python
@dataclass
class TimelineEvent:
    timestamp: float       # Seconds from video start
    type: str              # "speech" | "ocr" | "code" | "visual"
    source: str            # "whisper" | "paddleocr" | "opencv"
    text: str              # Extracted text content
    metadata: dict         # Type-specific extra data
```

This is the core abstraction. All downstream consumers (search, RAG, agents, HTML viewer) consume timeline events.

## GPU Memory Strategy

```
┌───────────────┐
│  Stage 1      │  Whisper on GPU (CUDA)
│  Transcribe   │  compute_type: int8_float16
└───────┬───────┘
        │
        ▼ release_gpu_memory()
        │
┌───────┴───────┐
│  Stage 2      │  PaddleOCR on GPU
│  OCR          │
└───────┬───────┘
        │
        ▼ release_gpu_memory()
        │
┌───────┴───────┐
│  Stages 3-4   │  CPU only (fusion + output)
│  Fusion/Out   │
└───────────────┘
```

Sequential GPU access prevents OOM on low-VRAM devices.

## Adaptive Sampling

```
Frame difference > threshold?
  YES → Switch to active_interval (0.5s) for active_window (8s)
  NO  → Stay at normal_interval (2.0s)
```

This means:
- Static screens → OCR every 2 seconds (efficient)
- Typing/scrolling → OCR every 0.5 seconds (captures changes)
- After activity stops → 8-second cooldown before returning to normal

## OCR Deduplication

```
New OCR text
    │
    ▼
character_bigram_similarity(previous, current)
    │
    ├── similarity >= 0.965 → Skip (same screen)
    │                         UNLESS heartbeat interval exceeded
    │
    └── similarity < 0.965 → Store as new OCR event
```
