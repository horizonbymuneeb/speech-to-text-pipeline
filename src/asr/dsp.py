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
