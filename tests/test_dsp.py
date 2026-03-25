"""Framing, spectra and mel-filterbank arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from asr.dsp import (
    DspError,
    apply_cmvn,
    apply_filterbank,
    cmvn,
    cmvn_stats,
    delta,
    frame_count,
    frame_signal,
    frame_times,
    hann_window,
    hz_to_mel,
    log_compress,
    mel_filterbank,
    mel_to_hz,
    power_spectrum,
    pre_emphasis,
    signal_duration,
    stack_deltas,
)


def test_pre_emphasis_formula():
    samples = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    emphasised = pre_emphasis(samples, 0.5)
    assert emphasised[0] == 1.0
    assert emphasised[1] == pytest.approx(1.5)
    assert emphasised[2] == pytest.approx(2.0)


def test_pre_emphasis_rejects_bad_coefficient():
    with pytest.raises(DspError, match="coefficient"):
        pre_emphasis(np.zeros(4), 1.0)


def test_frame_count_and_coverage():
    assert frame_count(1000, 200, 80) == 11
    assert frame_count(0, 200, 80) == 0
    assert frame_count(150, 200, 80) == 1
    frames = frame_signal(np.arange(1000, dtype=np.float64), 200, 80)
    assert frames.shape == (11, 200)
    assert frames[0][0] == 0.0
    assert frames[10][-1] == 999.0  # tail is padded, last sample still covered


def test_frame_signal_without_padding():
    frames = frame_signal(np.arange(1000, dtype=np.float64), 200, 80, pad=False)
    assert frames.shape == (11, 200)


def test_frame_signal_validates_input():
    with pytest.raises(DspError, match="mono"):
        frame_signal(np.zeros((2, 10)), 4, 2)
    with pytest.raises(DspError, match="positive"):
        frame_signal(np.zeros(10), 0, 2)


def test_hann_window_properties():
    window = hann_window(8)
    assert window.shape == (8,)
    assert window[0] == pytest.approx(0.0)
    assert window.max() == pytest.approx(1.0)
    assert np.all(window >= 0.0)
    with pytest.raises(DspError):
        hann_window(1)


def test_power_spectrum_of_a_pure_tone(tone):
    sample_rate = 8000
    frames = frame_signal(tone, 256, 128)
    power = power_spectrum(frames, 256)
    assert power.shape == (frames.shape[0], 129)
    bins = np.fft.rfftfreq(256, d=1.0 / sample_rate)
    peak = bins[int(np.argmax(power.mean(axis=0)))]
    assert peak == pytest.approx(440.0, abs=bins[1] - bins[0])


def test_power_spectrum_requires_big_enough_fft():
    with pytest.raises(DspError, match="n_fft"):
        power_spectrum(np.zeros((2, 300)), 256)


def test_mel_scale_roundtrip():
    frequencies = np.array([0.0, 100.0, 1000.0, 4000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(frequencies)), frequencies)
    assert hz_to_mel(0.0) == pytest.approx(0.0)
    assert hz_to_mel(700.0) == pytest.approx(2595.0 * np.log10(2.0))


def test_mel_filterbank_shape_and_area():
    filters = mel_filterbank(8000, 256, 20, 60.0, 3800.0)
    assert filters.shape == (20, 129)
    assert np.all(filters >= 0.0)
    # every filter is area-normalised over the discrete bins
    assert np.allclose(filters.sum(axis=1), 1.0, atol=1e-9)


def test_mel_filterbank_centres_ascend():
    filters = mel_filterbank(8000, 256, 10, 60.0, 3800.0)
    centres = np.argmax(filters, axis=1)
    assert np.all(np.diff(centres) >= 0)
    assert len(set(centres.tolist())) > 5


def test_mel_filterbank_validation():
    with pytest.raises(DspError, match="fmin_hz"):
        mel_filterbank(8000, 256, 10, 100.0, 50.0)
    with pytest.raises(DspError, match="n_mels"):
        mel_filterbank(8000, 256, 0)


def test_apply_filterbank_checks_bins():
    filters = mel_filterbank(8000, 256, 8, 60.0, 3800.0)
    with pytest.raises(DspError, match="bins"):
        apply_filterbank(np.zeros((3, 64)), filters)


def test_log_compress_floors_at_zero_energy():
    values = np.array([[0.0, np.e, 1.0]])
    compressed = log_compress(values)
    assert compressed[0][0] == pytest.approx(np.log(1e-10))
    assert compressed[0][1] == pytest.approx(1.0)


def test_cmvn_normalises_each_dimension():
    features = np.array([[1.0, 10.0], [3.0, 30.0]])
    normalised = cmvn(features)
    assert np.allclose(normalised.mean(axis=0), 0.0)
    assert np.allclose(normalised.std(axis=0), 1.0)
