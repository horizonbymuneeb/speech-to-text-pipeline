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
