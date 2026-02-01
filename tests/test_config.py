"""Config loading, defaults and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from asr.config import (
    DEFAULT_CONFIG,
    AudioConfig,
    ConfigError,
    DecodeConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    SynthConfig,
    TrainConfig,
    VadConfig,
    deep_merge,
    load_config,
    ms_to_frames,
    ms_to_samples,
)


def test_defaults_are_complete():
    config = PipelineConfig.from_mapping({})
    assert config.sample_rate == 8000
    assert config.features.output_dim == 120  # 40 mels x (static, delta, delta-delta)
    assert config.vad.frame_length(8000) == 200
    assert config.vad.hop_length(8000) == 80


def test_every_default_section_round_trips():
    config = PipelineConfig.from_mapping({})
    dumped = config.to_dict()
    for section, values in DEFAULT_CONFIG.items():
        assert section in dumped
        for key in values:
            assert key in dumped[section]


def test_ms_conversions():
    assert ms_to_samples(25.0, 8000) == 200
    assert ms_to_samples(10.0, 16000) == 160
    assert ms_to_frames(60.0, 80, 8000) == 6
    assert ms_to_frames(1.0, 80, 8000) == 1  # never zero frames


def test_unknown_section_rejected():
    with pytest.raises(ConfigError, match="unknown config section"):
        PipelineConfig.from_mapping({"vadd": {}})


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="unknown config key vad.max_zc"):
        PipelineConfig.from_mapping({"vad": {"max_zc": 0.3}})


def test_wrong_type_rejected():
    with pytest.raises(ConfigError, match="must be an int"):
        PipelineConfig.from_mapping({"features": {"n_mels": "forty"}})
