"""
Speech transcription via faster-whisper.

GPU strategy: Whisper gets the NVIDIA GPU first, then releases it
completely before PaddleOCR starts. This is intentional for
low-VRAM setups like the RTX 3050 4 GB.
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
    """Add nvidia pip package bin directories to Windows DLL search path and PATH."""
    if os.name != "nt":
        return
    import sys
    candidate_bases = [
        Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",
        Path(sys.prefix) / "nvidia",
    ]
    for base in candidate_bases:
        if base.is_dir():
            for bin_dir in base.glob("*/bin"):
                if bin_dir.is_dir():
                    try:
                        if hasattr(os, "add_dll_directory"):
                            os.add_dll_directory(str(bin_dir))
                    except (OSError, ValueError):
                        pass
                    if str(bin_dir) not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


class WhisperTranscriber:
    """
    faster-whisper transcription engine.

    Handles GPU/CPU detection, model loading, transcription with
    word-level timestamps, and proper GPU memory release.
    """

    def __init__(
        self,
        config: VidTraceConfig,
        initial_prompt: str | None = None,
    ):
        self.model_name = config.whisper_model
        self.language = config.language
        self.initial_prompt = initial_prompt
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"

        self._load()

    def _load(self) -> None:
        """Load the Whisper model with appropriate device settings."""
        from faster_whisper import WhisperModel

        _setup_cuda_dll_paths()
        gpu_available = False

        try:
            import ctranslate2
            supported = ctranslate2.get_supported_compute_types("cuda")
            if supported and ctranslate2.get_cuda_device_count() > 0:
                gpu_available = True
        except Exception:
            pass

        if gpu_available:
            try:
                self.device = "cuda"
                # Good compromise for a 4 GB RTX GPU.
                self.compute_type = "int8_float16"
                logger.info(
                    "Whisper | model=%s | device=%s | compute=%s",
                    self.model_name,
                    self.device,
                    self.compute_type,
                )
                self.model = WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=max(4, os.cpu_count() or 8),
                    num_workers=1,
                )
                return
            except Exception as exc:
                logger.warning(
                    "Failed to initialize Whisper on CUDA (%s). Falling back to CPU.",
                    exc,
                )

        self.device = "cpu"
        self.compute_type = "int8"
        logger.info(
            "Whisper | model=%s | device=%s | compute=%s",
            self.model_name,
            self.device,
            self.compute_type,
        )
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=max(4, os.cpu_count() or 8),
            num_workers=1,
        )

    def transcribe(
        self,
        video_path: Path,
    ) -> tuple[list[TranscriptSegment], dict[str, Any]]:
        """
        Transcribe a video file.

        Returns:
            Tuple of (segments, metadata_dict).
        """
        assert self.model is not None

        logger.info("Transcribing: %s", video_path.name)

        started = time.perf_counter()

        def _run_transcribe():
            assert self.model is not None
            return self.model.transcribe(
                str(video_path),
                language=self.language,
                task="transcribe",
                # Better recognition of technical speech.
                beam_size=5,
                best_of=5,
                temperature=0.0,
                # Keep context across segments.
                condition_on_previous_text=True,
                # Remove long silent sections.
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 350},
                # Extremely useful for aligning speech with OCR.
                word_timestamps=True,
                initial_prompt=(
                    self.initial_prompt
                    if self.initial_prompt is not None
                    else (
                        "This is a technical programming lecture. "
                        "The lecturer discusses Python, Java, "
                        "Kotlin, JavaScript, APIs, databases, "
                        "AI, machine learning, LLMs, agents, "
                        "LangChain, LangGraph and software engineering."
                    )
                ),
            )

        try:
            segments, info = _run_transcribe()
            segments_iter = iter(segments)
            first_seg = next(segments_iter, None)
        except Exception as exc:
            if self.device == "cuda":
                logger.warning("CUDA transcription failed (%s). Retrying on CPU.", exc)
                from faster_whisper import WhisperModel
                self.device = "cpu"
                self.compute_type = "int8"
                self.model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=max(4, os.cpu_count() or 8),
                    num_workers=1,
                )
                segments, info = _run_transcribe()
                segments_iter = iter(segments)
                first_seg = next(segments_iter, None)
            else:
                raise

        output: list[TranscriptSegment] = []

        all_segments = [first_seg] if first_seg is not None else []
        for seg in segments_iter:
            all_segments.append(seg)

        for index, segment in enumerate(all_segments):
            words = []
            raw_words = getattr(segment, "words", None)

            for word in raw_words or []:
                words.append({
                    "start": (
                        float(word.start)
                        if word.start is not None
                        else None
                    ),
                    "end": (
                        float(word.end)
                        if word.end is not None
                        else None
                    ),
                    "word": str(word.word),
                    "probability": (
                        float(word.probability)
                        if word.probability is not None
                        else None
                    ),
                })

            text = clean_text(str(segment.text))
            if not text:
                continue

            output.append(TranscriptSegment(
                index=index,
                start=float(segment.start),
                end=float(segment.end),
                text=text,
                words=words,
            ))

        elapsed = time.perf_counter() - started

        meta = {
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "audio_duration": getattr(info, "duration", None),
            "duration_after_vad": getattr(info, "duration_after_vad", None),
            "elapsed_seconds": elapsed,
            "device": self.device,
            "compute_type": self.compute_type,
            "model": self.model_name,
        }

        logger.info(
            "Whisper | %d segments | %.2f min",
            len(output),
            elapsed / 60,
        )

        return output, meta

    def close(self) -> None:
        """Release the model and GPU memory."""
        self.model = None
        release_gpu_memory()
