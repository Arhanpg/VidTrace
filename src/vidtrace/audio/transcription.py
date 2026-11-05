"""
Robust faster-whisper transcription for VidTrace.

Important: faster-whisper returns a lazy segment generator. The generator must
be fully consumed inside the guarded inference call so CUDA failures are caught
and can be retried on CPU.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from vidtrace.models.config import VidTraceConfig
from vidtrace.models.events import TranscriptSegment
from vidtrace.pipeline.gpu import release_gpu_memory
from vidtrace.utils import clean_text

logger = logging.getLogger(__name__)


def _setup_cuda_dll_paths() -> None:
    """Add CUDA DLL directories installed by NVIDIA pip packages on Windows."""
    if os.name != "nt":
        return

    import sys

    candidate_bases = [
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",
        Path(sys.prefix) / "nvidia",
    ]
    for base in candidate_bases:
        if not base.is_dir():
            continue
        for bin_dir in base.glob("*/bin"):
            if not bin_dir.is_dir():
                continue
            try:
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(bin_dir))
            except (OSError, ValueError):
                logger.debug("Could not register CUDA DLL directory: %s", bin_dir)
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if str(bin_dir) not in path_entries:
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


class WhisperTranscriber:
    """
    faster-whisper engine with conservative decoding and CUDA-to-CPU fallback.

    The caller can pass an initial prompt, but the engine never silently inserts
    an English prompt into a Hindi, Gujarati, Kannada, Marathi or Sanskrit task.
    """

    def __init__(
        self,
        config: VidTraceConfig,
        initial_prompt: str | None = None,
        *,
        prefer_gpu: bool = True,
        compute_type: str | None = None,
    ) -> None:
        self.model_name = config.whisper_model
        self.language = config.language
        self.initial_prompt = initial_prompt
        self.prefer_gpu = prefer_gpu
        self.requested_compute_type = compute_type
        self.model: Any = None
        self.device = "cpu"
        self.compute_type = "int8"
        self._load()

    def _load(self) -> None:
        """Load the model, verifying CUDA availability before allocating it."""
        from faster_whisper import WhisperModel

        _setup_cuda_dll_paths()
        if self.prefer_gpu:
            try:
                import ctranslate2

                supported = ctranslate2.get_supported_compute_types("cuda")
                device_count = ctranslate2.get_cuda_device_count()
                if supported and device_count > 0:
                    requested = self.requested_compute_type or "int8_float16"
                    if requested not in supported:
                        requested = (
                            "int8_float16" if "int8_float16" in supported
                            else "int8" if "int8" in supported
                            else sorted(supported)[0]
                        )
                    self.model = WhisperModel(
                        self.model_name,
                        device="cuda",
                        compute_type=requested,
                        cpu_threads=max(4, os.cpu_count() or 8),
                        num_workers=1,
                    )
                    self.device = "cuda"
                    self.compute_type = requested
                    logger.info(
                        "Whisper loaded | model=%s | device=cuda | compute=%s",
                        self.model_name,
                        self.compute_type,
                    )
                    return
            except Exception as exc:
                logger.warning("CUDA model initialization failed; using CPU: %s", exc)

        self._load_cpu()

    def _load_cpu(self) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=max(4, os.cpu_count() or 8),
            num_workers=1,
        )
        self.device = "cpu"
        self.compute_type = "int8"
        logger.info("Whisper loaded | model=%s | device=cpu | compute=int8", self.model_name)

    def _decode(self, audio_path: Path) -> tuple[list[Any], Any]:
        """
        Run and fully consume the lazy segment generator before returning.

        A CUDA failure anywhere during iteration is therefore caught by the
        outer retry handler, not after it has already returned to the caller.
        """
        if self.model is None:
            raise RuntimeError("Whisper model has not been initialized.")

        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 250,
            },
            word_timestamps=True,
            initial_prompt=self.initial_prompt,
        )
        return list(segments_iter), info

    @staticmethod
    def _serialize_word(word: Any) -> dict[str, Any]:
        return {
            "start": float(word.start) if word.start is not None else None,
            "end": float(word.end) if word.end is not None else None,
            "word": str(word.word),
            "probability": (
                float(word.probability)
                if word.probability is not None else None
            ),
        }

    def transcribe(
        self,
        audio_path: Path,
    ) -> tuple[list[TranscriptSegment], dict[str, Any]]:
        """Transcribe an audio/video path and return timestamped segments + metadata."""
        audio_path = Path(audio_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio input does not exist: {audio_path}")
        if self.model is None:
            raise RuntimeError("Whisper model has not been initialized.")

        started = time.perf_counter()
        cuda_error: str | None = None
        try:
            raw_segments, info = self._decode(audio_path)
        except Exception as exc:
            if self.device != "cuda":
                raise
            cuda_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "CUDA inference/iteration failed; rebuilding on CPU and retrying: %s",
                cuda_error,
            )
            self.model = None
            release_gpu_memory()
            self._load_cpu()
            raw_segments, info = self._decode(audio_path)

        output: list[TranscriptSegment] = []
        for index, segment in enumerate(raw_segments):
            text = clean_text(str(segment.text))
            if not text:
                continue
            words = [
                self._serialize_word(word)
                for word in (getattr(segment, "words", None) or [])
            ]
            output.append(
                TranscriptSegment(
                    index=len(output),
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                    words=words,
                    quality={
                        "avg_logprob": _optional_float(getattr(segment, "avg_logprob", None)),
                        "no_speech_prob": _optional_float(getattr(segment, "no_speech_prob", None)),
                        "compression_ratio": _optional_float(getattr(segment, "compression_ratio", None)),
                        "temperature": _optional_float(getattr(segment, "temperature", None)),
                        "seek": getattr(segment, "seek", None),
                    },
                )
            )

        elapsed = time.perf_counter() - started
        meta = {
            "language": getattr(info, "language", None),
            "language_probability": _optional_float(
                getattr(info, "language_probability", None)
            ),
            "audio_duration": _optional_float(getattr(info, "duration", None)),
            "duration_after_vad": _optional_float(getattr(info, "duration_after_vad", None)),
            "elapsed_seconds": elapsed,
            "device": self.device,
            "compute_type": self.compute_type,
            "model": self.model_name,
            "decoder": {
                "beam_size": 5,
                "temperature_fallback": [0.0, 0.2, 0.4, 0.6],
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": False,
                "vad_filter": True,
            },
            "cuda_retry_error": cuda_error,
            "segment_count": len(output),
        }
        logger.info(
            "Whisper completed | segments=%d | detected=%s (%.3f) | device=%s | %.1fs",
            len(output),
            meta["language"] or "unknown",
            meta["language_probability"] or 0.0,
            self.device,
            elapsed,
        )
        return output, meta

    def close(self) -> None:
        """Release the model and associated GPU memory."""
        self.model = None
        release_gpu_memory()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
