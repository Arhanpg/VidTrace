"""
Unit tests for vidtrace.models.config.

Tests preset loading, config serialization, and validation.
"""

import pytest
import tempfile
from pathlib import Path

from vidtrace.models.config import (
    PRESETS,
    VidTraceConfig,
    get_preset_names,
    load_config_file,
    load_preset,
)


class TestVidTraceConfig:
    def test_defaults(self):
        config = VidTraceConfig()
        assert config.whisper_model == "small"
        assert config.sample_interval == 2.0
        assert config.cpu_ocr is False
        assert config.force is False

    def test_to_dict(self):
        config = VidTraceConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["whisper_model"] == "small"
        assert "sample_interval" in d

    def test_custom_values(self):
        config = VidTraceConfig(
            whisper_model="medium",
            sample_interval=1.0,
            cpu_ocr=True,
        )
        assert config.whisper_model == "medium"
        assert config.sample_interval == 1.0
        assert config.cpu_ocr is True


class TestPresets:
    def test_all_presets_exist(self):
        names = get_preset_names()
        assert "default" in names
        assert "lecture" in names
        assert "coding" in names
        assert "meeting" in names
        assert "tutorial" in names

    def test_load_default(self):
        config = load_preset("default")
        assert isinstance(config, VidTraceConfig)

    def test_load_lecture(self):
        config = load_preset("lecture")
        assert config.sample_interval == 2.0

    def test_load_coding(self):
        config = load_preset("coding")
        assert config.sample_interval == 1.5
        assert config.active_interval == 0.40
        assert "html" in config.output_formats

    def test_load_meeting(self):
        config = load_preset("meeting")
        assert config.sample_interval == 5.0
        assert "srt" in config.output_formats

    def test_load_tutorial(self):
        config = load_preset("tutorial")
        assert "html" in config.output_formats
        assert "srt" in config.output_formats

    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent")


class TestConfigFile:
    def test_load_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            f.write("whisper_model: medium\nsample_interval: 3.0\n")
            f.flush()

            config = load_config_file(Path(f.name))
            assert config.whisper_model == "medium"
            assert config.sample_interval == 3.0

    def test_load_empty_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            f.write("")
            f.flush()

            config = load_config_file(Path(f.name))
            assert config.whisper_model == "small"  # default

    def test_unknown_keys_ignored(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            f.write("whisper_model: small\nunknown_key: value\n")
            f.flush()

            # Should not raise.
            config = load_config_file(Path(f.name))
            assert config.whisper_model == "small"
