"""Voice activity detection from short-time energy and zero-crossing rate.

The detector works on frames and makes two decisions per frame:

* **energy** — frame power in dB, compared against a noise floor estimated from
  the quietest 15% of the utterance. Comparing against a *relative* floor is what
  makes the same settings work for a quiet recording and a loud one.
* **zero-crossing rate** — a cheap voicing cue. A high ZCR at marginal energy is
  a fricative or background hiss, not speech, so it vetoes the frame. A frame
  that is clearly loud is accepted regardless of ZCR, because rejecting a loud
  voiced frame is worse than admitting a loud noise burst.

A frame decision then survives ``hangover_ms`` after the energy drops, which
keeps word tails that decay below the threshold, and segments shorter than
``min_speech_ms`` are dropped as clicks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from asr.config import VadConfig
from asr.dsp import frame_count, frame_signal

EPSILON = 1e-12


class VadError(ValueError):
    """Raised for invalid detector input."""


@dataclass(frozen=True)
class FrameStats:
    """Per-frame measurements used by the detector (and by the tests)."""

    energy_db: np.ndarray
    zcr: np.ndarray
    frame_length: int
    hop_length: int
    num_samples: int

    def __post_init__(self) -> None:
        if self.energy_db.shape != self.zcr.shape:
            raise VadError("energy_db and zcr must have the same shape")
        if self.energy_db.ndim != 1:
            raise VadError("frame statistics must be one-dimensional")

    @property
    def num_frames(self) -> int:
        return int(self.energy_db.shape[0])


@dataclass(frozen=True)
class SpeechSegment:
    """A detected speech region in sample coordinates."""

    start_sample: int
    end_sample: int
    start_frame: int
    end_frame: int

    def __post_init__(self) -> None:
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise VadError(f"invalid segment [{self.start_sample}, {self.end_sample})")

    @property
    def num_samples(self) -> int:
        return self.end_sample - self.start_sample

    def duration_s(self, sample_rate: int) -> float:
        """Segment duration in seconds."""
        return self.num_samples / float(sample_rate)

    def to_dict(self, sample_rate: int) -> dict[str, float]:
        """Serialisable form with both sample and time coordinates."""
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "start_s": self.start_sample / float(sample_rate),
            "end_s": self.end_sample / float(sample_rate),
            "duration_s": self.duration_s(sample_rate),
        }


def frame_energy(frames: np.ndarray) -> np.ndarray:
    """Mean-square energy of every frame."""
    if frames.ndim != 2:
        raise VadError("frames must be 2-D")
    return np.mean(np.square(frames), axis=1)


def to_db(power: np.ndarray) -> np.ndarray:
    """Convert power to dB, flooring at ``-120 dB`` instead of ``-inf``."""
    return np.maximum(10.0 * np.log10(np.maximum(power, EPSILON)), -120.0)


def energy_db(frames: np.ndarray) -> np.ndarray:
    """Frame energy in dB."""
    return to_db(frame_energy(frames))


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Fraction of adjacent sample pairs that change sign in each frame."""
    if frames.ndim != 2:
        raise VadError("frames must be 2-D")
    if frames.shape[1] < 2:
        return np.zeros(frames.shape[0], dtype=np.float64)
    signs = np.signbit(frames)
    crossings = np.not_equal(signs[:, 1:], signs[:, :-1])
    return np.mean(crossings, axis=1)


def noise_floor_db(energies_db: np.ndarray, percentile: float = 15.0) -> float:
    """Noise floor estimate: a low percentile of the frame energies."""
    if energies_db.size == 0:
        raise VadError("cannot estimate a noise floor from zero frames")
    if not 0.0 <= percentile <= 100.0:
        raise VadError("percentile must be in [0, 100]")
    return float(np.percentile(energies_db, percentile))


def frame_decisions(
    energies_db: np.ndarray, zcr: np.ndarray, config: VadConfig, floor_db: float | None = None
) -> np.ndarray:
    """Per-frame speech decisions (see the module docstring for the rule)."""
    if energies_db.shape != zcr.shape:
        raise VadError("energies_db and zcr must have the same shape")
    if energies_db.size == 0:
        return np.zeros(0, dtype=bool)
    floor = noise_floor_db(energies_db) if floor_db is None else float(floor_db)
    loud = energies_db >= floor + config.energy_threshold_db
    voiced = zcr <= config.max_zcr
    clearly_loud = energies_db >= floor + 2.0 * config.energy_threshold_db
    return np.asarray(loud & (voiced | clearly_loud), dtype=bool)


def apply_hangover(decisions: np.ndarray, hangover_frames: int) -> np.ndarray:
    """Extend every speech run by ``hangover_frames`` frames."""
    if hangover_frames < 0:
        raise VadError("hangover_frames must be non-negative")
    if decisions.size == 0 or hangover_frames == 0:
        return decisions.astype(bool).copy()
    result = decisions.astype(bool).copy()
    indices = np.flatnonzero(decisions)
    for index in indices:
        end = min(result.size, index + hangover_frames + 1)
        result[index:end] = True
    return result


def frames_to_segments(
    decisions: np.ndarray,
    *,
    hop_length: int,
    frame_length: int,
    num_samples: int,
    min_speech_frames: int = 1,
    pre_roll_frames: int = 0,
) -> list[SpeechSegment]:
    """Convert frame decisions into sample-accurate speech segments."""
    if decisions.size == 0:
        return []
    segments: list[SpeechSegment] = []
    index = 0
    while index < decisions.size:
        if not decisions[index]:
            index += 1
            continue
        start = index
        while index + 1 < decisions.size and decisions[index + 1]:
            index += 1
        end = index
        index += 1
        if end - start + 1 < min_speech_frames:
            continue
        start_frame = max(0, start - pre_roll_frames)
        start_sample = start_frame * hop_length
        end_sample = min(num_samples, end * hop_length + frame_length)
        if end_sample <= start_sample:
            continue
        segments.append(
            SpeechSegment(
                start_sample=start_sample,
                end_sample=end_sample,
                start_frame=start_frame,
                end_frame=end,
            )
        )
    return merge_segments(segments)
