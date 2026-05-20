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
