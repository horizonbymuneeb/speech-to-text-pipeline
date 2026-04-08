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


def test_log_mel_shape():
    signal = np.random.default_rng(0).normal(0, 0.1, 4000).astype(np.float32)
    features = log_mel_spectrogram(signal, CONFIG, SR)
    assert features.shape == (expected_frames(signal.size, CONFIG, SR), CONFIG.n_mels)
    assert np.all(np.isfinite(features))


def test_full_features_have_the_configured_dimension(tone):
    extractor = FeatureExtractor(CONFIG, SR)
    features = extractor.transform(tone)
    assert features.shape == (expected_frames(tone.size, CONFIG, SR), 96)
    assert features.dtype == np.float32


def test_features_without_deltas_or_cmvn(tone):
    plain = replace(CONFIG, use_deltas=False, cmvn=False)
    features = extract_features(tone, plain, SR)
    assert features.shape[1] == plain.n_mels
