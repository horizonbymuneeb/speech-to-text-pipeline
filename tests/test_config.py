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
