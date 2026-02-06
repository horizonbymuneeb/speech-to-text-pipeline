"""Signal-processing primitives: framing, windows, spectra and mel math.

These are the pieces both the detector and the feature extractor need, kept in
one module so that the VAD and the log-mel front end analyse *identical* windows
— a mismatch there is a classic source of "the detector says speech but the
features disagree" bugs.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-10
