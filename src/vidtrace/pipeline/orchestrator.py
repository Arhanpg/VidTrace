"""
Pipeline orchestrator.

Coordinates the full extraction pipeline:
  Stage 1: Transcription (Whisper)
  Stage 2: OCR (PaddleOCR with adaptive sampling)
  Stage 3: Timeline fusion
  Stage 4: Output rendering (Markdown, JSON, HTML, SRT)

Supports resume via checkpoint: completed stages are skipped,
partial OCR is resumed from the last processed frame.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vidtrace.audio.transcription import WhisperTranscriber
from vidtrace.fusion.timeline import build_timeline, nearby_speech
from vidtrace.models.config import VidTraceConfig
from vidtrace.models.events import OCREvent, TranscriptSegment, VideoInfo
from vidtrace.output.html import export_html_timeline
from vidtrace.output.json_output import (
    append_jsonl,
    export_ocr_events_json,
    export_timeline_json,
    export_timeline_jsonl,
    export_transcript_json,
    write_json,
)
from vidtrace.output.markdown import (
    transcript_markdown,
    write_master_report,
    write_video_report,
)
from vidtrace.output.srt import export_srt, export_vtt
from vidtrace.pipeline.checkpoint import (
    StageStatus,
    VideoCheckpoint,
    VideoIdentity,
    format_checkpoint_status,
)
from vidtrace.pipeline.gpu import (
    detect_gpu,
    read_video_info,
    release_gpu_memory,
)
from vidtrace.utils import (
    character_bigram_similarity,
    format_time,
    looks_like_code,
    now_iso,
    safe_filename,
)
from vidtrace.vision.ocr import PaddleOCREngine
from vidtrace.vision.sampler import adaptive_frame_sampler

logger = logging.getLogger(__name__)

# Supported video extensions.
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
    ".m4v", ".ts", ".mts", ".m2ts",
}


class VideoProcessor:
    """
    Processes a single video through the full extraction pipeline.
    """

    def __init__(self, config: VidTraceConfig, gpu_info: dict[str, Any]):
        self.config = config
        self.gpu_info = gpu_info

    def process(self, video_path: Path) -> Path:
        """
        Run the full pipeline on a video file.

        Returns:
            Path to the generated analysis report.
        """
        logger.info("=" * 70)
        logger.info("VIDEO: %s", video_path.name)
        logger.info("=" * 70)

        info = read_video_info(video_path)

        # ── Output directories ────────────────────────────────
        output_root = video_path.parent / self.config.output_folder
        video_output = output_root / safe_filename(video_path.stem)
        evidence_dir = video_output / "evidence_frames"

        video_output.mkdir(parents=True, exist_ok=True)
        evidence_dir.mkdir(parents=True, exist_ok=True)

        # ── Checkpoint ────────────────────────────────────────
        state_path = video_output / "state.json"
        checkpoint = VideoCheckpoint.load(state_path) or VideoCheckpoint()

        # Check if video changed since last run.
        video_identity = VideoIdentity.from_file(video_path)

        if checkpoint.is_video_changed(video_path):
            if checkpoint.started_at:
                logger.info(
                    "[INVALIDATE] Video file changed. Re-running all stages."
                )
            checkpoint = VideoCheckpoint()

        checkpoint.video = asdict(video_identity)
        checkpoint.gpu = self.gpu_info
        checkpoint.started_at = checkpoint.started_at or now_iso()

        checkpoint.mark_stage(
            "metadata", StageStatus.COMPLETE, state_path,
            video_info=asdict(info),
        )

        # ══════════════════════════════════════════════════════
        # STAGE 1: TRANSCRIPTION
        # ══════════════════════════════════════════════════════
        transcript, whisper_meta = self._run_transcription(
            video_path, video_output, checkpoint, state_path,
        )

        # ══════════════════════════════════════════════════════
        # STAGE 2: OCR
        # ══════════════════════════════════════════════════════
        events = self._run_ocr(
            video_path, info, transcript,
            video_output, evidence_dir, checkpoint, state_path,
        )

        # ══════════════════════════════════════════════════════
        # STAGE 3: TIMELINE FUSION
        # ══════════════════════════════════════════════════════
        if checkpoint.should_run_stage("fusion", self.config.force):
            checkpoint.mark_stage("fusion", StageStatus.RUNNING, state_path)

            timeline = build_timeline(transcript, events)

            export_timeline_json(
                video_output / "timeline.json", timeline,
            )
            export_timeline_jsonl(
                video_output / "events.jsonl", timeline,
            )

            checkpoint.mark_stage(
                "fusion", StageStatus.COMPLETE, state_path,
                timeline_events=len(timeline),
            )
        else:
            logger.info("[SKIP] Fusion already complete.")
            timeline = build_timeline(transcript, events)

        # ══════════════════════════════════════════════════════
        # STAGE 4: OUTPUT RENDERING
        # ══════════════════════════════════════════════════════
        report_path = self._render_outputs(
            video_path, info, whisper_meta, transcript,
            events, video_output,
        )

        checkpoint.mark_stage(
            "output", StageStatus.COMPLETE, state_path,
            formats=self.config.output_formats,
        )

        return report_path

    # ──────────────────────────────────────────────────────────
    # Stage 1: Transcription
    # ──────────────────────────────────────────────────────────

    def _run_transcription(
        self,
        video_path: Path,
        video_output: Path,
        checkpoint: VideoCheckpoint,
        state_path: Path,
    ) -> tuple[list[TranscriptSegment], dict[str, Any]]:
        """Run Whisper transcription with resume support."""
        transcript_json_path = video_output / "transcript.json"
        transcript_md_path = video_output / "transcript.md"

        transcript: list[TranscriptSegment] = []
        whisper_meta: dict[str, Any] = {}

        if not checkpoint.should_run_stage(
            "transcription", self.config.force
        ):
            logger.info("[SKIP] Transcription already complete.")

            saved = json.loads(
                transcript_json_path.read_text(encoding="utf-8")
            )
            transcript = [
                TranscriptSegment(**item)
                for item in saved["segments"]
            ]
            whisper_meta = saved.get("meta", {})

        elif transcript_json_path.exists() and not self.config.force:
            logger.info("[RESUME] Existing transcript found.")

            saved = json.loads(
                transcript_json_path.read_text(encoding="utf-8")
            )
            transcript = [
                TranscriptSegment(**item)
                for item in saved["segments"]
            ]
            whisper_meta = saved.get("meta", {})

            checkpoint.mark_stage(
                "transcription", StageStatus.COMPLETE, state_path,
                segments=len(transcript),
            )

        else:
            checkpoint.mark_stage(
                "transcription", StageStatus.RUNNING, state_path,
            )

            transcriber = WhisperTranscriber(self.config)
            try:
                transcript, whisper_meta = transcriber.transcribe(video_path)
            finally:
                transcriber.close()

            export_transcript_json(
                transcript_json_path,
                str(video_path.resolve()),
                whisper_meta,
                transcript,
            )

            transcript_md_path.write_text(
                transcript_markdown(transcript),
                encoding="utf-8",
            )

            checkpoint.mark_stage(
                "transcription", StageStatus.COMPLETE, state_path,
                segments=len(transcript),
            )

        # Ensure transcript.md exists even on resume.
        if not transcript_md_path.exists():
            transcript_md_path.write_text(
                transcript_markdown(transcript),
                encoding="utf-8",
            )

        return transcript, whisper_meta

    # ──────────────────────────────────────────────────────────
    # Stage 2: OCR
    # ──────────────────────────────────────────────────────────

    def _run_ocr(
        self,
        video_path: Path,
        info: VideoInfo,
        transcript: list[TranscriptSegment],
        video_output: Path,
        evidence_dir: Path,
        checkpoint: VideoCheckpoint,
        state_path: Path,
    ) -> list[OCREvent]:
        """Run OCR with adaptive sampling, deduplication, and resume."""
        ocr_events_path = video_output / "ocr_events.json"
        ocr_candidates_path = video_output / "ocr_all_candidates.jsonl"

        # Skip if already complete.
        if not checkpoint.should_run_stage("ocr", self.config.force):
            logger.info("[SKIP] OCR already complete.")
            if ocr_events_path.exists():
                data = json.loads(
                    ocr_events_path.read_text(encoding="utf-8")
                )
                return [OCREvent(**item) for item in data]
            return []

        logger.info("Starting visual analysis.")

        # If forcing, delete old OCR output.
        if self.config.force:
            ocr_events_path.unlink(missing_ok=True)
            ocr_candidates_path.unlink(missing_ok=True)
            checkpoint.ocr_last_timestamp = -1.0
            checkpoint.ocr_last_frame = -1
            checkpoint.ocr_event_count = 0

        # Partial resume: load existing events.
        events: list[OCREvent] = []
        resume_from_timestamp = -1.0

        if (
            checkpoint.get_stage_status("ocr") == StageStatus.PARTIAL.value
            and checkpoint.ocr_last_timestamp > 0
            and ocr_events_path.exists()
        ):
            logger.info(
                "[RESUME] OCR partial — resuming from t=%.1fs",
                checkpoint.ocr_last_timestamp,
            )
            data = json.loads(
                ocr_events_path.read_text(encoding="utf-8")
            )
            events = [OCREvent(**item) for item in data]
            resume_from_timestamp = checkpoint.ocr_last_timestamp

        checkpoint.mark_stage("ocr", StageStatus.RUNNING, state_path)

        ocr = PaddleOCREngine(
            prefer_gpu=not self.config.cpu_ocr,
            min_ocr_chars=self.config.min_ocr_chars,
            retry_confidence=self.config.ocr_retry_confidence,
        )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            ocr.close()
            raise RuntimeError("Could not reopen video for OCR.")

        previous_text = events[-1].text if events else ""
        previous_event_time = events[-1].timestamp if events else -999.0
        event_id = len(events)
        started = time.perf_counter()
        candidate_count = 0

        try:
            for (
                frame_index, timestamp, frame, visual_difference,
            ) in adaptive_frame_sampler(
                cap,
                fps=info.fps,
                duration=info.duration_seconds,
                normal_interval=self.config.sample_interval,
                active_interval=self.config.active_interval,
                active_window=self.config.active_window,
                scene_threshold=self.config.scene_threshold,
            ):
                # Skip frames already processed (partial resume).
                if timestamp <= resume_from_timestamp:
                    continue

                candidate_count += 1

                # Skip nearly black / empty frames.
                grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(np.mean(grayscale))

                if (
                    brightness < self.config.min_luma
                    and visual_difference < self.config.scene_threshold * 1.5
                ):
                    continue

                # OCR.
                boxes, preprocessing = ocr.run(frame)

                lines = [
                    box.text for box in boxes
                    if box.text.strip()
                ]
                full_text = "\n".join(lines).strip()

                mean_confidence = (
                    float(np.mean([box.confidence for box in boxes]))
                    if boxes else 0.0
                )

                # Log every candidate.
                candidate = {
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "visual_difference": visual_difference,
                    "brightness": brightness,
                    "preprocessing": preprocessing,
                    "mean_confidence": mean_confidence,
                    "text": full_text,
                }
                append_jsonl(ocr_candidates_path, candidate)

                if len(full_text.strip()) < self.config.min_ocr_chars:
                    continue

                # Semantic change detection.
                similarity = character_bigram_similarity(
                    previous_text, full_text,
                )
                should_store = (
                    not previous_text
                    or similarity < self.config.ocr_change_similarity
                    or (timestamp - previous_event_time >= self.config.ocr_heartbeat)
                )

                if not should_store:
                    continue

                # Save evidence screenshot.
                evidence_name = f"{event_id:06d}_{timestamp:012.3f}s.jpg"
                evidence_path = evidence_dir / evidence_name

                cv2.imwrite(
                    str(evidence_path),
                    frame,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        int(self.config.jpeg_quality),
                    ],
                )

                event = OCREvent(
                    event_id=event_id,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    text=full_text,
                    lines=lines,
                    boxes=[asdict(box) for box in boxes],
                    mean_confidence=mean_confidence,
                    code_likelihood=looks_like_code(full_text),
                    preprocessing=preprocessing,
                    evidence_frame=str(
                        evidence_path.relative_to(video_output)
                    ).replace("\\", "/"),
                    speech_context=nearby_speech(
                        transcript, timestamp, radius=10.0,
                    ),
                )

                events.append(event)
                event_id += 1
                previous_text = full_text
                previous_event_time = timestamp

                # Periodic checkpoint save for partial resume.
                if event_id % 20 == 0:
                    elapsed = time.perf_counter() - started
                    logger.info(
                        "OCR | events=%d candidates=%d time=%s elapsed=%.2fm",
                        event_id, candidate_count,
                        format_time(timestamp),
                        elapsed / 60,
                    )
                    checkpoint.update_ocr_progress(
                        timestamp, frame_index, event_id,
                    )
                    checkpoint.mark_stage(
                        "ocr", StageStatus.PARTIAL, state_path,
                    )
                    export_ocr_events_json(ocr_events_path, events)

        except KeyboardInterrupt:
            logger.info("OCR interrupted — saving partial progress.")
            checkpoint.update_ocr_progress(
                previous_event_time,
                events[-1].frame_index if events else -1,
                len(events),
            )
            checkpoint.mark_stage("ocr", StageStatus.PARTIAL, state_path)
            export_ocr_events_json(ocr_events_path, events)
            raise

        finally:
            cap.release()
            export_ocr_events_json(ocr_events_path, events)
            ocr.close()

        checkpoint.mark_stage(
            "ocr", StageStatus.COMPLETE, state_path,
            candidate_frames=candidate_count,
            semantic_events=len(events),
            elapsed_seconds=time.perf_counter() - started,
        )

        return events

    # ──────────────────────────────────────────────────────────
    # Stage 4: Output rendering
    # ──────────────────────────────────────────────────────────

    def _render_outputs(
        self,
        video_path: Path,
        info: VideoInfo,
        whisper_meta: dict[str, Any],
        transcript: list[TranscriptSegment],
        events: list[OCREvent],
        video_output: Path,
    ) -> Path:
        """Render all configured output formats."""
        formats = self.config.output_formats
        ocr_candidates_path = video_output / "ocr_all_candidates.jsonl"

        # Markdown report (always generated).
        report_path = (
            video_output
            / f"{safe_filename(video_path.stem)}__analysis.md"
        )
        write_video_report(
            report_path, info, whisper_meta,
            transcript, events, video_output,
            ocr_candidates_path,
        )

        # HTML timeline viewer.
        if "html" in formats:
            html_path = video_output / "timeline.html"
            export_html_timeline(html_path, info, transcript, events)
            logger.info("Generated: %s", html_path.name)

        # SRT subtitles.
        if "srt" in formats:
            srt_path = video_output / "transcript.srt"
            export_srt(srt_path, transcript)
            vtt_path = video_output / "transcript.vtt"
            export_vtt(vtt_path, transcript)
            logger.info("Generated: %s, %s", srt_path.name, vtt_path.name)

        return report_path


# ──────────────────────────────────────────────────────────────
# Batch processing
# ──────────────────────────────────────────────────────────────

def find_videos(folder: Path, output_folder: str) -> list[Path]:
    """Find all supported video files in a folder recursively."""
    videos = []
    for file in folder.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        # Don't process our own output directory.
        if output_folder in file.parts:
            continue
        videos.append(file)

    videos.sort()
    return videos


def run_pipeline(
    input_path: Path,
    config: VidTraceConfig,
) -> int:
    """
    Run the full VidTrace pipeline.

    Args:
        input_path: Path to a video file or folder of videos.
        config: Pipeline configuration.

    Returns:
        Exit code (0 = success, 1 = no videos, 3 = partial failure).
    """
    gpu_info = detect_gpu()

    # Determine if input is a single file or a folder.
    if input_path.is_file():
        videos = [input_path]
        input_folder = input_path.parent
    else:
        input_folder = input_path
        videos = find_videos(input_folder, config.output_folder)

    if not videos:
        logger.error("No supported videos found.")
        logger.info("Supported: %s", ", ".join(sorted(VIDEO_EXTENSIONS)))
        return 1

    logger.info("=" * 70)
    logger.info("VIDTRACE EXTRACTION PIPELINE")
    logger.info("=" * 70)
    logger.info("Input        : %s", input_path)
    logger.info("Videos       : %d", len(videos))
    logger.info("GPU          : %s", gpu_info.get("gpu_name", "CPU"))
    logger.info("Whisper      : %s", config.whisper_model)
    logger.info("OCR interval : %.1fs normal / %.1fs active",
                config.sample_interval, config.active_interval)
    logger.info("Output       : %s", ", ".join(config.output_formats))
    logger.info("=" * 70)

    processor = VideoProcessor(config, gpu_info)
    reports: list[Path] = []
    failures: list[dict] = []
    total_start = time.perf_counter()

    for number, video in enumerate(videos, start=1):
        logger.info("[%d/%d] %s", number, len(videos), video.name)

        try:
            report = processor.process(video)
            reports.append(report)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as exc:
            logger.error("FAILED: %s — %s", video.name, repr(exc))
            failures.append({
                "video": str(video),
                "error": repr(exc),
            })
            release_gpu_memory()

    # Master report.
    output_root = input_folder / config.output_folder
    output_root.mkdir(parents=True, exist_ok=True)

    master = write_master_report(
        output_root, reports, gpu_info, input_folder,
    )

    elapsed = time.perf_counter() - total_start

    summary = {
        "generated_at": now_iso(),
        "input_folder": str(input_folder),
        "videos_found": len(videos),
        "successful": len(reports),
        "failed": len(failures),
        "failures": failures,
        "elapsed_seconds": elapsed,
        "gpu": gpu_info,
        "master_report": str(master),
    }
    write_json(output_root / "run_summary.json", summary)

    logger.info("=" * 70)
    logger.info("EXTRACTION COMPLETE")
    logger.info("Master report : %s", master)
    logger.info("Successful    : %d/%d", len(reports), len(videos))
    logger.info("Failed        : %d", len(failures))
    logger.info("Total time    : %.2f minutes", elapsed / 60)
    logger.info("=" * 70)

    return 0 if not failures else 3


def get_video_status(input_path: Path, config: VidTraceConfig) -> str:
    """Get checkpoint status for a video file."""
    output_root = input_path.parent / config.output_folder
    video_output = output_root / safe_filename(input_path.stem)
    state_path = video_output / "state.json"

    checkpoint = VideoCheckpoint.load(state_path)
    if checkpoint is None:
        return f"No checkpoint found for: {input_path.name}"

    return (
        f"Video: {input_path.name}\n"
        f"Output: {video_output}\n"
        f"{format_checkpoint_status(checkpoint)}"
    )
