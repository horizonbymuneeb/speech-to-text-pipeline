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
