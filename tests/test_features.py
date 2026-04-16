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


def test_pure_tone_lands_in_the_expected_mel_band():
    """A 440 Hz tone must excite the band that covers 440 Hz, not its neighbours."""
    from asr.dsp import mel_filterbank, mel_to_hz

    sample_rate = 8000
    time = np.arange(4000) / sample_rate
    tone = (0.8 * np.sin(2 * np.pi * 440.0 * time)).astype(np.float32)
    log_mel = log_mel_spectrogram(tone, replace(CONFIG, cmvn=False), sample_rate)
    band = int(np.argmax(log_mel.mean(axis=0)))
    filters = mel_filterbank(sample_rate, CONFIG.n_fft, CONFIG.n_mels, 60.0, 3800.0)
    bins = np.fft.rfftfreq(CONFIG.n_fft, d=1.0 / sample_rate)
    band_centre_hz = mel_to_hz(
        np.linspace(
            __import__("asr.dsp", fromlist=["hz_to_mel"]).hz_to_mel(60.0),
            __import__("asr.dsp", fromlist=["hz_to_mel"]).hz_to_mel(3800.0),
            CONFIG.n_mels + 2,
        )[band + 1]
    )
    assert abs(band_centre_hz - 440.0) < 400.0
    assert filters[band][np.argmin(np.abs(bins - 440.0))] > 0


def test_cmvn_makes_each_dimension_zero_mean_unit_variance(tone):
    features = log_mel_spectrogram(tone, CONFIG, SR)
    normalised = normalise_features(features, CONFIG)
    assert np.allclose(normalised.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(normalised.std(axis=0), 1.0, atol=1e-5)


def test_global_stats_keep_chunks_consistent():
    """Chunked features must match whole-utterance features (streaming sanity)."""
    audio = AudioConfig(sample_rate=SR, peak_amplitude=0.9, noise_snr_db=40.0)
    synth = SynthConfig(
        seed=2, vocabulary=("kasa", "timo"), words_per_utterance=(2, 2), f0_hz=120.0, gap_ms=(40.0, 60.0)
    )
    corpus = make_corpus(audio, synth, 3, seed=5)
    plain = FeatureExtractor(CONFIG, SR)
    stats = plain.corpus_stats([signal.samples for signal in corpus])
    extractor = FeatureExtractor(CONFIG, SR, stats=stats)

    signal = corpus[0].samples
    whole = extractor.transform(signal)
    chunks = [
        extractor.transform(signal[start : start + 1600])
        for start in range(0, signal.size, 1600)
    ]
    stitched = np.concatenate(chunks, axis=0)
    assert stitched.shape[1] == whole.shape[1]
    assert abs(float(stitched.mean()) - float(whole.mean())) < 0.35


def test_corpus_stats_shape_and_validation():
    extractor = FeatureExtractor(CONFIG, SR)
    with pytest.raises(FeatureError, match="at least one signal"):
        extractor.corpus_stats([])


def test_empty_signal_rejected():
    extractor = FeatureExtractor(CONFIG, SR)
    with pytest.raises(FeatureError, match="empty"):
        extractor.transform(np.zeros(0, dtype=np.float32))
    with pytest.raises(FeatureError, match="mono"):
        extractor.transform(np.zeros((2, 100), dtype=np.float32))


def test_extractor_geometry_and_cache():
    extractor = FeatureExtractor(CONFIG, SR)
    assert extractor.frame_length == 200
    assert extractor.hop_length == 80
    assert extractor.output_dim == 96
    assert extractor.filterbank.shape == (CONFIG.n_mels, CONFIG.n_fft // 2 + 1)
    assert extractor.filterbank is extractor.filterbank


def test_stats_report():
    extractor = FeatureExtractor(CONFIG, SR)
    stats = extractor.stats(np.zeros(4000, dtype=np.float32))
    # frame_count(4000, 200, 80) = 1 + ceil((4000 - 200) / 80) = 49
    assert stats.num_frames == 49
    assert stats.feature_dim == 96
    assert stats.duration_s == pytest.approx(0.5)
    assert stats.frame_rate_hz == pytest.approx(98.0)


def test_crop_and_frame_helpers():
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    assert crop_features(features, 3, 2).shape == (2, 2)
    assert crop_features(features, 9, 100).shape == (1, 2)
    assert crop_features(features, 50).shape == (0, 2)
    assert frames_to_samples(0, CONFIG, SR) == 0
    assert frames_to_samples(2, CONFIG, SR) == 280


def test_frame_energy_profile():
    features = np.zeros((4, 3))
    features[:, :] = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [5.0, 5.0, 5.0], [7.0, 7.0, 7.0]])
    assert np.allclose(frame_energy_profile(features), [1.0, 3.0, 5.0, 7.0])
    assert frame_energy_profile(np.zeros((0, 3))).size == 0
