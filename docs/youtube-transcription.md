# YouTube transcript extraction

This guide describes the YouTube video/playlist transcription command added to VidTrace.

## What it does

1. Accepts a public YouTube video URL or playlist URL.
2. Reads video/playlist metadata with yt-dlp.
3. Downloads audio only (not the full video) to a temporary local cache.
4. Uses the existing faster-whisper engine to produce timestamped speech segments.
5. Detects the language independently for each video when --language auto is selected.
6. Writes a Markdown transcript for each video, plus SRT, VTT, JSON and a playlist index.
7. Deletes temporary audio after a successful transcript unless --keep-audio is selected.
8. Reuses completed outputs on later runs; use --force to regenerate.

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

### GPU notes for the RTX 3050

VidTrace reuses the existing faster-whisper CUDA detection and low-VRAM compute settings. Current CTranslate2 releases require CUDA 12 cuBLAS and cuDNN 9 for NVIDIA GPU inference. If these libraries are not available, VidTrace may use CPU; check the startup log for the reported device and compute type.

Official faster-whisper GPU requirements: https://github.com/SYSTRAN/faster-whisper#gpu

If you want to test CPU first, use the small or base model and check the run log. Do not assume CUDA is active just because an NVIDIA GPU is installed.

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

A larger model may improve recognition but consumes more VRAM/RAM and takes longer. For a 4 GB laptop GPU, start with small.

### 10. Keep the downloaded audio

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/watch?v=VIDEO_ID" --keep-audio
~~~

Without this flag, downloaded audio is removed after a successful transcript is written. Failed downloads are retained in the temporary cache to help resume/retry.

### 11. Re-run completed videos

~~~powershell
vidtrace youtube-transcribe "https://www.youtube.com/playlist?list=PLAYLIST_ID" --force
~~~

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

The exact extension depends on Whisper's detected language. With experimental Prakrit mode, the Markdown filename is transcript.prakrit_experimental.md and includes a language warning.

### Markdown transcript contents

Each transcript.<language>.md contains:
- Original YouTube video URL and ID.
- Playlist name and position.
- Requested and detected language.
- Language-detection confidence (when reported).
- Whisper model and audio duration.
- Timestamped speech blocks.
- An accuracy warning for scriptural names, gāthās and Jain terminology.

The Markdown transcript is the recommended file to upload for research review. Keep the JSON file as a machine-readable backup and SRT/VTT for subtitles.

## Resume behaviour

- Completed videos with matching metadata are skipped on subsequent runs.
- Use --force to reprocess.
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
