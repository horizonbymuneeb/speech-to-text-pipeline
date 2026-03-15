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
