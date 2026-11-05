"""CLI tests for the YouTube transcription command."""

from vidtrace.cli import build_parser


def test_youtube_transcribe_parser_accepts_playlist_options() -> None:
    args = build_parser().parse_args([
        "youtube-transcribe",
        "https://www.youtube.com/playlist?list=PL123",
        "--language", "hi",
        "--playlist-start", "3",
        "--playlist-end", "8",
        "--max-videos", "4",
        "--whisper-model", "small",
    ])
    assert args.command == "youtube-transcribe"
    assert args.language == "hi"
    assert args.playlist_start == 3
    assert args.playlist_end == 8
    assert args.max_videos == 4
    assert args.whisper_model == "small"


def test_youtube_transcribe_parser_defaults_to_auto_detection() -> None:
    args = build_parser().parse_args([
        "youtube-transcribe",
        "https://youtu.be/VIDEO_ID",
    ])
    assert args.language == "auto"
    assert args.whisper_model == "medium"
    assert args.captions == "prefer"
    assert args.output_dir.name == "youtube_transcripts"
    assert args.keep_audio is False
    assert args.force is False



def test_youtube_transcribe_parser_can_disable_captions() -> None:
    args = build_parser().parse_args([
        "youtube-transcribe",
        "https://youtu.be/VIDEO_ID",
        "--captions", "off",
        "--whisper-model", "large-v3",
    ])
    assert args.captions == "off"
    assert args.whisper_model == "large-v3"
