"""Tests for Whisper inference fallback and lazy generator handling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from vidtrace.audio.transcription import WhisperTranscriber


class FakeWord:
    start = 0.0
    end = 0.4
    word = " नमो"
    probability = 0.95


class FakeSegment:
    start = 0.0
    end = 1.0
    text = "नमो अरिहंताणं।"
    words = [FakeWord()]
    avg_logprob = -0.2
    no_speech_prob = 0.01
    compression_ratio = 1.1
    temperature = 0.0
    seek = 0


class FakeInfo:
    language = "hi"
    language_probability = 0.98
    duration = 1.0
    duration_after_vad = 1.0


def test_lazy_cuda_generator_failure_retries_on_cpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    devices: list[str] = []

    class FakeWhisperModel:
        def __init__(self, model_name, device, compute_type, **kwargs):
            self.device = device
            devices.append(device)

        def transcribe(self, *args, **kwargs):
            if self.device == "cuda":
                def broken_generator():
                    yield FakeSegment()
                    raise RuntimeError("simulated CUDA OOM during iteration")
                return broken_generator(), FakeInfo()
            return iter([FakeSegment()]), FakeInfo()

    fake_module = ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    monkeypatch.setattr(
        "vidtrace.audio.transcription.release_gpu_memory",
        lambda: None,
    )

    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")

    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.model_name = "small"
    transcriber.language = "hi"
    transcriber.initial_prompt = None
    transcriber.prefer_gpu = True
    transcriber.requested_compute_type = None
    transcriber.model = FakeWhisperModel("small", "cuda", "int8_float16")
    transcriber.device = "cuda"
    transcriber.compute_type = "int8_float16"

    segments, metadata = transcriber.transcribe(audio)

    assert devices == ["cuda", "cpu"]
    assert transcriber.device == "cpu"
    assert len(segments) == 1
    assert segments[0].text == "नमो अरिहंताणं।"
    assert segments[0].quality["avg_logprob"] == -0.2
    assert "simulated CUDA OOM" in metadata["cuda_retry_error"]
    assert metadata["language"] == "hi"
