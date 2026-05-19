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
