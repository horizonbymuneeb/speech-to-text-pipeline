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


def normalise_features(
    features: np.ndarray,
    config: FeatureConfig,
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Apply CMVN when configured.

    ``stats`` holds global mean/std computed once over a training corpus. When it
    is given the normalisation is identical for every chunk, which is what makes
    streaming features match offline features; without it each call normalises
    against its own frames (fine for whole utterances, wrong across chunks).
    """
    if not config.cmvn:
        return features
    if stats is not None:
        mean, std = stats
        return apply_cmvn(features, mean, std)
    return cepstral_normalise(features)


def with_deltas(features: np.ndarray, config: FeatureConfig) -> np.ndarray:
    """Append delta and delta-delta features when configured."""
    return stack_deltas(features) if config.use_deltas else features


def extract_features(
    samples: np.ndarray, config: FeatureConfig, sample_rate: int
) -> np.ndarray:
    """Full front end: log-mel → CMVN → deltas. Shape ``(n_frames, output_dim)``."""
    log_mel = log_mel_spectrogram(samples, config, sample_rate)
    normalised = normalise_features(log_mel, config)
    return with_deltas(normalised, config).astype(np.float32)


def expected_frames(num_samples: int, config: FeatureConfig, sample_rate: int) -> int:
    """Number of frames :func:`extract_features` will produce for a signal."""
    return frame_count(num_samples, config.frame_length(sample_rate), config.hop_length(sample_rate))


def frames_to_samples(num_frames: int, config: FeatureConfig, sample_rate: int) -> int:
    """Approximate sample count covered by ``num_frames`` frames."""
    if num_frames <= 0:
        return 0
    hop = config.hop_length(sample_rate)
    return (num_frames - 1) * hop + config.frame_length(sample_rate)


def crop_features(
    features: np.ndarray, start_frame: int, num_frames: Optional[int] = None
) -> np.ndarray:
    """Slice a feature matrix by frame index, clamped to its bounds."""
    if features.ndim != 2:
        raise FeatureError("features must be 2-D")
    total = features.shape[0]
    start = max(0, min(int(start_frame), total))
    stop = total if num_frames is None else min(total, start + max(0, int(num_frames)))
    return features[start:stop]


def frame_energy_profile(features: np.ndarray) -> np.ndarray:
    """Per-frame loudness proxy from a log-mel matrix (used by the decoder)."""
    if features.ndim != 2:
        raise FeatureError("features must be 2-D")
    if features.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return features.mean(axis=1)


@dataclass(frozen=True)
class FeatureStats:
    """Summary of one extraction, useful in logs and in tests."""

    num_frames: int
    feature_dim: int
    num_samples: int
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return self.num_samples / float(self.sample_rate)

    @property
    def frame_rate_hz(self) -> float:
        return self.num_frames / self.duration_s if self.duration_s else 0.0


class FeatureExtractor:
    """Reusable extractor: caches the mel filterbank between calls.

    The filterbank depends only on ``(sample_rate, n_fft, n_mels, fmin, fmax)``,
    so building it once per extractor (not once per chunk) keeps streaming cheap.
    """

    def __init__(
        self,
        config: FeatureConfig,
        sample_rate: int,
        stats: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise FeatureError("sample_rate must be positive")
        self.config = config
        self.sample_rate = sample_rate
        self.cmvn_stats = stats
        self._filterbank = mel_filterbank(
            sample_rate, config.n_fft, config.n_mels, config.fmin_hz, config.fmax_hz
        )

    @property
    def frame_length(self) -> int:
        return self.config.frame_length(self.sample_rate)

    @property
    def hop_length(self) -> int:
        return self.config.hop_length(self.sample_rate)

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    @property
    def filterbank(self) -> np.ndarray:
        """The cached ``(n_mels, n_fft//2 + 1)`` filter matrix."""
        return self._filterbank

    def log_mel(self, samples: np.ndarray) -> np.ndarray:
        """Log-mel energies without CMVN or deltas."""
        signal = np.asarray(samples, dtype=np.float64)
        if signal.ndim != 1:
            raise FeatureError("samples must be mono (1-D)")
        if signal.size == 0:
            raise FeatureError("cannot extract features from an empty signal")
        emphasised = pre_emphasis(signal, self.config.pre_emphasis)
        frames = frame_signal(emphasised, self.frame_length, self.hop_length)
        spectrum = power_spectrum(frames, self.config.n_fft)
        return log_compress(apply_filterbank(spectrum, self._filterbank), LOG_FLOOR)

    def transform(self, samples: np.ndarray) -> np.ndarray:
        """Full feature matrix for a waveform."""
        features = self.log_mel(samples)
        features = normalise_features(features, self.config, self.cmvn_stats)
        return with_deltas(features, self.config).astype(np.float32)

    def stats(self, samples: np.ndarray) -> FeatureStats:
        """Shape information about what :meth:`transform` would return."""
        count = expected_frames(int(np.asarray(samples).size), self.config, self.sample_rate)
        return FeatureStats(
            num_frames=count,
            feature_dim=self.output_dim,
            num_samples=int(np.asarray(samples).size),
            sample_rate=self.sample_rate,
        )

    def corpus_stats(self, signals) -> tuple[np.ndarray, np.ndarray]:
        """Log-mel statistics over several waveforms (for global CMVN)."""
        frames = [self.log_mel(signal) for signal in signals]
        if not frames:
            raise FeatureError("need at least one signal to compute statistics")
        return cmvn_stats(np.concatenate(frames, axis=0))

    def delta(self, features: np.ndarray, width: int = 2) -> np.ndarray:
        """Expose the delta operator for callers that manage their own stacking."""
        return delta(features, width)
