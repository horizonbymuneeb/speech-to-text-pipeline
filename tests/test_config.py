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


def test_bool_is_not_an_int():
    with pytest.raises(ConfigError, match="must be an int"):
        PipelineConfig.from_mapping({"model": {"channels": True}})


def test_deep_merge_is_recursive_and_pure():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"c": 9}})
    assert merged == {"a": {"b": 1, "c": 9}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_load_config_from_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"train": {"epochs": 3}}), encoding="utf-8")
    config = load_config(path)
    assert config.train.epochs == 3
    assert config.checkpoint_path() == (tmp_path / "checkpoints" / "ctc.pt")


def test_load_config_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_rejects_non_mapping(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_load_config_overrides(tmp_path):
    config = load_config(None, overrides={"decode": {"beam_width": 4}})
    assert config.decode.beam_width == 4
    assert config.features.n_mels == 40


def test_audio_config_validation():
    with pytest.raises(ConfigError, match="sample_rate"):
        AudioConfig(sample_rate=1000)
    with pytest.raises(ConfigError, match="peak_amplitude"):
        AudioConfig(peak_amplitude=1.5)


def test_vad_config_validation():
    with pytest.raises(ConfigError, match="hop_ms"):
        VadConfig(frame_ms=10.0, hop_ms=25.0)
    with pytest.raises(ConfigError, match="max_zcr"):
        VadConfig(max_zcr=2.0)
    with pytest.raises(ConfigError, match="non-negative"):
        VadConfig(hangover_ms=-1.0)


def test_feature_config_validation():
    with pytest.raises(ConfigError, match="power of two"):
        FeatureConfig(n_fft=250)
    with pytest.raises(ConfigError, match="n_mels"):
        FeatureConfig(n_mels=200)
    with pytest.raises(ConfigError, match="fmin_hz"):
        FeatureConfig(fmin_hz=4000.0, fmax_hz=3000.0)
