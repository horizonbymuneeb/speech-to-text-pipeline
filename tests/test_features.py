"""Log-mel front end: shapes, band placement and CMVN consistency."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from asr.config import FeatureConfig
from asr.features import (
    FeatureError,
    FeatureExtractor,
    crop_features,
    expected_frames,
    extract_features,
    frame_energy_profile,
    frames_to_samples,
    log_mel_spectrogram,
    normalise_features,
    with_deltas,
)
from asr.synth import AudioConfig, SynthConfig, make_corpus

SR = 8000
CONFIG = FeatureConfig(
    frame_ms=25.0,
    hop_ms=10.0,
    n_fft=256,
    n_mels=32,
    fmin_hz=60.0,
    fmax_hz=3800.0,
    pre_emphasis=0.97,
    use_deltas=True,
    cmvn=True,
)
