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
