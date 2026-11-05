# YouTube transcript extraction

This guide describes the YouTube video/playlist transcription command added to VidTrace.

## What it does

1. Accepts a public YouTube video URL or playlist URL.
2. Reads video/playlist metadata with yt-dlp.
3. Checks for a matching YouTube manual-caption track and then a matching automatic-caption track.
4. Prefers usable captions; if captions are unavailable or clearly poor, downloads audio only (not the full video) and uses faster-whisper.
5. Uses faster-whisper medium by default, conservative decoding thresholds, temperature fallback, and CUDA-to-CPU retry.
6. Detects the language independently for each video when --language auto is selected.
7. Calculates transcript-quality diagnostics and marks suspicious output review_required instead of silently calling it complete.
8. Writes Markdown, SRT, VTT and JSON for each video, plus a playlist index.
9. Deletes temporary audio after successful processing unless --keep-audio is selected.
10. Reuses only transcripts that passed the current pipeline's quality checks; use --force to regenerate.

The tool does not translate speech. It transcribes the spoken language. A Hindi talk produces a Hindi transcript; a Gujarati talk produces a Gujarati transcript, etc.

## Language choices

| Input | Whisper language |
|---|---|
| auto | Detect each video's dominant language |
| hi / hindi | Hindi |
| gu / gujarati | Gujarati |
| mr / marathi | Marathi |
| kn / kannada | Kannada |
| sa / sanskrit | Sanskrit |
| en / english | English |
| prakrit | Experimental: auto-detection only; no dedicated Whisper Prakrit language token |

Whisper supports Hindi, Gujarati, Marathi, Kannada and Sanskrit language tokens. Prakrit has no dedicated token in the standard Whisper language list. The experimental Prakrit option therefore uses automatic detection and adds a warning to the transcript. A transcript of Prakrit speech must be manually reviewed by someone who knows the relevant Prakrit variety; the model may label it as Sanskrit, Hindi or another related language.

Automatic detection chooses a dominant language for a video. It does not reliably split code-switched speech into separate per-language transcript files. If the speaker switches languages frequently, review the transcript or run the video again with a forced language.

## Installation — Windows PowerShell

Use Python 3.11 or newer for the current YouTube extractor dependency set.

Clone or update VidTrace:

~~~powershell
git clone https://github.com/Arhanpg/VidTrace.git
cd VidTrace
~~~

Create and activate a virtual environment:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

Upgrade pip and install the YouTube transcription feature:

~~~powershell
python -m pip install --upgrade pip
pip install -e ".[youtube-transcribe]"
~~~

For tests and development tools:

~~~powershell
pip install -e ".[youtube-transcribe,dev]"
~~~

### YouTube extractor runtime

For current YouTube extraction support, install a JavaScript runtime such as Deno, then open a new PowerShell window:

~~~powershell
winget install -e --id DenoLand.Deno
deno --version
~~~

Current yt-dlp documentation recommends yt-dlp-ejs plus a supported JavaScript runtime (Deno is recommended) for full YouTube support. The youtube-transcribe extra installs the yt-dlp[default] dependency set; Deno is installed separately.

Official Deno Windows instructions: https://docs.deno.com/runtime/getting_started/installation/

### GPU notes for the RTX 3050 4 GB

The YouTube command now defaults to the medium Whisper model and requests int8_float16 when CUDA supports it. It verifies that a CUDA device exists, then falls back to CPU if CUDA initialization or inference fails. A CPU retry can be significantly slower. Check the log for the actual device and compute type rather than assuming the GPU is active.

Current CTranslate2 GPU inference requires compatible CUDA 12 cuBLAS and cuDNN 9 libraries. See the official faster-whisper GPU requirements: https://github.com/SYSTRAN/faster-whisper#gpu

For a faster, lighter run, select small. If a user-selected tiny/base/small model returns review_required, VidTrace performs one medium-model retry during that playlist run. The medium model may still fall back to CPU when VRAM is insufficient.

## Basic usage

### 1. Transcribe a single video (auto-detect)

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/watch?v=VIDEO_ID"
~~~

### 2. Transcribe a playlist (auto-detect each video)

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/playlist?list=PLAYLIST_ID"
~~~

### 3. Force Hindi

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/playlist?list=PLAYLIST_ID" --language hi
~~~

### 4. Force Gujarati

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/playlist?list=PLAYLIST_ID" --language gu
~~~

### 5. Sanskrit

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --language sa
~~~

### 6. Experimental Prakrit mode

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --language prakrit
~~~

This mode is not a dedicated Prakrit ASR model. Expect manual correction.

### 7. Process only playlist items 10–20

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/playlist?list=PLAYLIST_ID" --playlist-start 10 --playlist-end 20
~~~

### 8. Limit to the first five videos

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/playlist?list=PLAYLIST_ID" --max-videos 5
~~~

### 9. Use a larger model

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --whisper-model medium
~~~

The YouTube command defaults to medium. Medium may improve Indic-language recognition over small but consumes more memory and may use CPU fallback on a 4 GB GPU. Use small when speed or VRAM is the priority.

### 10. Disable YouTube captions and force Whisper

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --captions off
~~~

By default VidTrace prefers a matching manual YouTube caption track, then an automatic caption track. Auto mode uses caption tracks only when the video language or a unique supported caption language can be resolved; it does not guess between several unrelated languages. Use --captions off to force faster-whisper.

### 11. Keep the downloaded audio

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --keep-audio
~~~

Without this flag, downloaded audio is removed after a successful transcript is written. Failed downloads are retained in the temporary cache to help resume/retry.

### 12. Re-run completed videos

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/playlist?list=PLAYLIST_ID" --force
~~~

### Regenerate the existing Jain research outputs

The transcripts committed under `YT text extraction/` were produced by the older pipeline and do not contain quality metadata. Do not use them as verified research sources. Regenerate them with the new pipeline, writing back to the same collection folder:

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/playlist?list=PLAYLIST_ID" --output-dir "YT text extraction" --whisper-model medium --force
~~~

For a video known to be Hindi, specify the language so Whisper can use the Hindi Jain terminology prompt and matching Hindi caption tracks:

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --language hi --output-dir "YT text extraction" --whisper-model medium --force
~~~

Use `--captions off` only when you specifically want to bypass YouTube caption tracks and run speech recognition on the downloaded audio. A suspicious result will be marked `review_required`; it is not automatically corrected or certified.

## Output structure

Default output directory: ./youtube_transcripts/

~~~text
youtube_transcripts/
└── samavasaran-lecture-series/
    ├── playlist_index.md
    ├── run_summary.json
    └── videos/
        ├── 001-introduction-VIDEO_ID/
        │   ├── metadata.json
        │   ├── transcript.json
        │   ├── transcript.hi.md
        │   ├── transcript.hi.srt
        │   └── transcript.hi.vtt
        └── 002-tiloyapannatti-VIDEO_ID/
            ├── metadata.json
            ├── transcript.json
            ├── transcript.sa.md
            ├── transcript.sa.srt
            └── transcript.sa.vtt
~~~

The exact extension depends on the detected language or the caption language. With experimental Prakrit mode, the Markdown filename is transcript.prakrit_experimental.md and includes a language warning.

Each transcript now includes a Quality assessment section. Status values are:
- passed: no obvious automatic quality warning was detected.
- review_required: repetition, symbol-heavy output, low log probability, or another warning was detected. Review the transcript before using it as research evidence.
- failed: no usable text was produced.

These are conservative automated checks. A passed status does not establish that Jain terminology, scripture quotations, or gāthā numbers are correct. Automatically generated YouTube captions are always marked review_required because they are still machine-generated.

### Markdown transcript contents

Each transcript.<language>.md contains:
- Original YouTube video URL and ID.
- Playlist name and position.
- Requested and detected language.
- Language-detection confidence (when reported).
- Model and transcription source (YouTube manual captions, automatic captions, or faster-whisper).
- Language confidence when speech recognition supplied it.
- Quality score, review status and quality flags.
- Audio duration when speech recognition was used.
- Timestamped speech blocks.
- An accuracy warning for scriptural names, gāthās and Jain terminology.

The Markdown transcript is the recommended file to upload for research review. Keep the JSON file as a machine-readable backup and SRT/VTT for subtitles.

### Exit codes

- 0: all processed transcripts passed automated checks (or were skipped because they had previously passed).
- 1: one or more videos failed processing.
- 2: processing finished, but one or more transcripts require human review.

A review_required result is not a claim that the transcript is wrong; it means the program will not silently certify it as clean.

## Resume behaviour

- Only transcripts that passed the current pipeline quality checks are skipped on later runs.
- Existing transcripts from the older pipeline version are regenerated once because they have no quality assessment.
- review_required transcripts are processed again on a later run; use --force to regenerate any transcript.
- A failed video does not stop the rest of the playlist.
- Successful transcript files are preserved if you interrupt the process.
- Failed video audio may remain in .audio_cache/ for retry. Remove this folder manually if you no longer need the cached audio.

## Accuracy workflow for Jain research

Automatic transcription is only a first draft. After transcription:

1. Keep the timestamp in every transcript segment.
2. Correct proper nouns such as Ādinātha, Ṛṣabhadeva, Jinasena and Yativṛṣabha.
3. Check technical terms such as Samavasaraṇa, Gandhakuṭī, Mānastambha, pīṭha, vedī, bhūmi and koṭa.
4. Check every gāthā number against the book edition being discussed.
5. Distinguish what the speaker explains from a direct quotation read from a scripture.
6. Mark unintelligible audio rather than filling in a likely phrase.
7. Retain the original transcript alongside your corrected version.

Do not upload a machine transcript to the project as a verified quotation. Mark it as machine-generated and reviewed/unreviewed.

## Troubleshooting

### yt-dlp cannot extract a YouTube video
- Update the project dependencies: pip install --upgrade "yt-dlp[default]".
- Confirm deno --version works in a new terminal.
- Public videos only are supported by this initial command. Private, members-only, age-restricted or region-restricted videos may require cookies/access configuration that is not included in this first implementation.

### No module named faster_whisper
Run:

~~~powershell
pip install -e ".[youtube-transcribe]"
~~~

### Whisper uses CPU rather than CUDA
Check the startup log's device and compute_type. The current faster-whisper/CTranslate2 GPU setup needs compatible CUDA 12 cuBLAS and cuDNN 9 libraries. See the official faster-whisper README.

### A Sanskrit or Prakrit transcript looks like Hindi noise
- Try --language sa for Sanskrit.
- For Prakrit, the tool can only use auto-detection; review against the recording and a knowledgeable reader.
- A small Whisper model may struggle with rare terms, recitation, old recordings, music or multiple speakers. Test medium if your machine can run it.

### A playlist run is interrupted
Run the same command again. Completed outputs are skipped, and temporary audio for failed/uncompleted videos may be reused.
