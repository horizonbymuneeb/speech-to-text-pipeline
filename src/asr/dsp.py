"""Signal-processing primitives: framing, windows, spectra and mel math.

These are the pieces both the detector and the feature extractor need, kept in
one module so that the VAD and the log-mel front end analyse *identical* windows
— a mismatch there is a classic source of "the detector says speech but the
features disagree" bugs.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-10
MEL_CONSTANT = 2595.0
MEL_LOG_BASE = 10.0
MEL_BREAK_HZ = 700.0


class DspError(ValueError):
    """Raised for shapes and parameters that cannot be processed."""


def frame_count(num_samples: int, frame_length: int, hop_length: int) -> int:
    """Number of frames needed to cover ``num_samples`` (last frame padded)."""
    if frame_length <= 0 or hop_length <= 0:
        raise DspError("frame_length and hop_length must be positive")
    if num_samples <= 0:
        return 0
    if num_samples < frame_length:
        return 1
    return 1 + int(np.ceil((num_samples - frame_length) / hop_length))


def frame_signal(
    samples: np.ndarray, frame_length: int, hop_length: int, *, pad: bool = True
) -> np.ndarray:
    """Split a 1-D signal into overlapping frames, shape ``(n_frames, frame_length)``.

    ``pad=True`` zero-pads the tail so every input sample lands in a frame.
    """
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1:
        raise DspError("samples must be mono (1-D)")
    if frame_length <= 0 or hop_length <= 0:
        raise DspError("frame_length and hop_length must be positive")
    if signal.size == 0:
        return np.zeros((0, frame_length), dtype=np.float64)
    if not pad:
        count = 0 if signal.size < frame_length else 1 + (signal.size - frame_length) // hop_length
        if count == 0:
            return np.zeros((0, frame_length), dtype=np.float64)
        return np.lib.stride_tricks.sliding_window_view(signal, frame_length)[::hop_length][:count]
    count = frame_count(signal.size, frame_length, hop_length)
    needed = (count - 1) * hop_length + frame_length
    if needed > signal.size:
        signal = np.pad(signal, (0, needed - signal.size))
    return np.lib.stride_tricks.sliding_window_view(signal, frame_length)[::hop_length][:count]


def pre_emphasis(samples: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    """First-order high-pass ``y[n] = x[n] - a * x[n-1]``.

    Boosts the high frequencies that the glottal source rolls off and that the
    mel filterbank would otherwise see as nearly silent.
    """
    if not 0.0 <= coefficient < 1.0:
        raise DspError("pre-emphasis coefficient must be in [0, 1)")
    signal = np.asarray(samples, dtype=np.float64)
    if signal.ndim != 1:
        raise DspError("samples must be mono (1-D)")
    if signal.size == 0:
        return signal.copy()
    emphasised = np.empty_like(signal)
    emphasised[0] = signal[0]
    emphasised[1:] = signal[1:] - coefficient * signal[:-1]
    return emphasised


def hann_window(length: int) -> np.ndarray:
    """Periodic Hann window (the form that keeps overlapped frames summing flat)."""
    if length < 2:
        raise DspError("window length must be >= 2")
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(length) / length)


def power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Windowed magnitude-squared spectrum, shape ``(n_frames, n_fft//2 + 1)``."""
    if frames.ndim != 2:
        raise DspError("frames must be 2-D")
    if n_fft < frames.shape[1]:
        raise DspError("n_fft must be at least the frame length")
    window = hann_window(frames.shape[1])
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=1)
    return np.square(np.abs(spectrum))


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray:
    """HTK mel scale: ``2595 * log10(1 + f / 700)``."""
    return MEL_CONSTANT * np.log10(1.0 + np.asarray(frequency_hz, dtype=np.float64) / MEL_BREAK_HZ)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`hz_to_mel`."""
    return MEL_BREAK_HZ * (10.0 ** (np.asarray(mel, dtype=np.float64) / MEL_CONSTANT) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin_hz: float = 0.0,
    fmax_hz: float | None = None,
) -> np.ndarray:
    """Triangular mel filters, shape ``(n_mels, n_fft//2 + 1)``.

    Filters are normalised to unit area so that a louder vowel raises every band
    by the same factor; without that, wide high-frequency filters would dominate
    the feature vector.
    """
    if sample_rate <= 0:
        raise DspError("sample_rate must be positive")
    if n_mels < 1:
        raise DspError("n_mels must be >= 1")
    nyquist = sample_rate / 2.0
    top = nyquist if fmax_hz is None else float(fmax_hz)
    if not 0.0 <= fmin_hz < top <= nyquist:
        raise DspError("need 0 <= fmin_hz < fmax_hz <= sample_rate/2")
    mel_points = np.linspace(hz_to_mel(fmin_hz), hz_to_mel(top), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    filters = np.zeros((n_mels, bins.size), dtype=np.float64)
    for index in range(n_mels):
        left, centre, right = hz_points[index], hz_points[index + 1], hz_points[index + 2]
        rising = (bins - left) / max(centre - left, EPSILON)
        falling = (right - bins) / max(right - centre, EPSILON)
        triangle = np.maximum(0.0, np.minimum(rising, falling))
        area = np.sum(triangle)
        if area > 0.0:
            triangle = triangle / area
        filters[index] = triangle
    return filters


def apply_filterbank(power: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Project power spectra onto mel filters."""
    if power.ndim != 2:
        raise DspError("power must be 2-D")
    if filters.shape[1] != power.shape[1]:
        raise DspError(
            f"filterbank has {filters.shape[1]} bins but the spectrum has {power.shape[1]}"
        )
    return power @ filters.T


def log_compress(values: np.ndarray, floor: float = EPSILON) -> np.ndarray:
    """Natural log with a floor, so silent bands become a large negative number."""
    if floor <= 0:
        raise DspError("floor must be positive")
    return np.log(np.maximum(values, floor))


def cmvn(features: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Cepstral mean and variance normalisation, per feature dimension."""
    if features.ndim != 2:
        raise DspError("features must be 2-D")
    if features.shape[0] == 0:
        return features.copy()
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, epsilon)


def cmvn_stats(features: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation per feature dimension, with a std floor."""
    if features.ndim != 2:
        raise DspError("features must be 2-D")
    if features.shape[0] == 0:
        raise DspError("cannot compute statistics of zero frames")
    mean = features.mean(axis=0)
    std = np.maximum(features.std(axis=0), epsilon)
    return mean.astype(np.float64), std.astype(np.float64)


def apply_cmvn(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalise features with fixed statistics (the streaming-safe form)."""
    if features.ndim != 2:
        raise DspError("features must be 2-D")
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != (features.shape[1],) or std.shape != (features.shape[1],):
        raise DspError("CMVN statistics do not match the feature dimension")
    return (features - mean) / np.maximum(std, EPSILON)


def delta(features: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression deltas with edge replication.

    ``width`` is the number of frames on each side; the denominator is
    ``2 * sum(i^2)`` so the result is invariant to the frame rate.
    """
    if features.ndim != 2:
        raise DspError("features must be 2-D")
    if width < 1:
        raise DspError("delta width must be >= 1")
    count = features.shape[0]
    if count == 0:
        return features.copy()
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    denominator = 2.0 * sum(index * index for index in range(1, width + 1))
    result = np.zeros_like(features, dtype=np.float64)
    for index in range(1, width + 1):
        result += index * (padded[width + index : width + index + count] - padded[width - index : width - index + count])
    return result / denominator


def stack_deltas(features: np.ndarray, width: int = 2) -> np.ndarray:
    """Concatenate ``[features, delta, delta-delta]`` along the feature axis."""
    first = delta(features, width)
    second = delta(first, width)
    return np.concatenate([features, first, second], axis=1)
