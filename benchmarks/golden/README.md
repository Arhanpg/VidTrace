# Golden Dataset

Ground-truth annotations for measuring VidTrace accuracy.

## Annotation Format

Each video has a corresponding `.json` annotation file:

```json
{
  "video": "coding_sample_30s.mp4",
  "duration": 30.0,
  "annotations": [
    {
      "timestamp": 5.0,
      "type": "speech",
      "text": "Now we'll define the state class"
    },
    {
      "timestamp": 8.5,
      "type": "code",
      "text": "class AgentState(TypedDict):\n    messages: list[str]",
      "language": "python"
    },
    {
      "timestamp": 15.0,
      "type": "scene",
      "description": "Switch from editor to terminal"
    }
  ]
}
```

## Evaluation Metrics

### Speech
- Word Error Rate (WER) against annotated text

### OCR
- Character Error Rate (CER) against annotated screen text
- Detection recall: % of annotated text regions captured

### Code
- Code detection precision: correctly identified / total detected
- Language detection accuracy

### Temporal
- Event timestamp error: |predicted - annotated| in seconds

## Contributing Annotations

1. Choose a short (30-120s), properly licensed video
2. Watch the video and annotate every event
3. Save as `<video_name>.annotations.json`
4. Submit a PR to `benchmarks/golden/`
