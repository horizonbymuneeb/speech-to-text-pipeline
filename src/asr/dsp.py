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
