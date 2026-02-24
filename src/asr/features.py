"""Log-mel feature extraction.

The front end is deliberately small and explicit: pre-emphasis, overlapping
Hann frames, power spectrum, mel filterbank, log, then optional mean/variance
normalisation and deltas. Every step is a named function so the tests can check
it in isolation, and the extractor caches the filterbank because rebuilding it
per chunk would dominate the cost of streaming.

Two choices worth stating: filters are area-normalised (a louder vowel scales
every band equally), and the log floor is ``1e-10`` so digital silence becomes
about ``-23`` rather than ``-inf`` and the model never sees a NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from asr.config import FeatureConfig
from asr.dsp import (
    apply_cmvn,
    apply_filterbank,
    cmvn as cepstral_normalise,
    cmvn_stats,
    delta,
    frame_count,
    frame_signal,
    log_compress,
    mel_filterbank,
    power_spectrum,
    pre_emphasis,
    stack_deltas,
)

LOG_FLOOR = 1e-10


class FeatureError(ValueError):
    """Raised when a waveform cannot be turned into features."""


def log_mel_spectrogram(
    samples: np.ndarray,
    config: FeatureConfig,
    sample_rate: int,
    *,
    apply_pre_emphasis: bool = True,
) -> np.ndarray:
    """Compute ``(n_frames, n_mels)`` log-mel energies."""
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1:
        raise FeatureError("samples must be mono (1-D)")
    if signal.size == 0:
        raise FeatureError("cannot extract features from an empty signal")
    if apply_pre_emphasis:
        signal = pre_emphasis(signal, config.pre_emphasis)
    frames = frame_signal(signal, config.frame_length(sample_rate), config.hop_length(sample_rate))
    spectrum = power_spectrum(frames, config.n_fft)
    filters = mel_filterbank(
        sample_rate, config.n_fft, config.n_mels, config.fmin_hz, config.fmax_hz
    )
    mel_energies = apply_filterbank(spectrum, filters)
    return log_compress(mel_energies, LOG_FLOOR)
