"""Synthetic speech generation with known ground truth.

A recogniser needs labelled audio. Downloading a corpus is not an option here
(no network at test time, and a real corpus is hundreds of megabytes), so the
pipeline generates its own: each word is built from a consonant burst and a
vowel, and each vowel is additive-synthesised from a harmonic series whose
amplitude envelope is shaped by the first three formants. That produces audio
with the spectral structure a log-mel front end and a CTC model can actually
learn — a pure tone would let the model cheat on pitch alone.

Because the generator knows the word list and the exact sample span of every
word, every downstream metric has ground truth without any alignment step.
Everything is seeded: same seed, same waveform, bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from asr.config import AudioConfig, SynthConfig
from asr.vocab import Vocabulary

# First three formants (Hz) of the five vowels used by the corpus.
VOWEL_FORMANTS: dict[str, tuple[float, float, float]] = {
    "a": (700.0, 1220.0, 2600.0),
    "e": (500.0, 1800.0, 2500.0),
    "i": (300.0, 2200.0, 3000.0),
    "o": (500.0, 900.0, 2400.0),
    "u": (320.0, 800.0, 2300.0),
}

# consonant -> (noise centre Hz, bandwidth Hz, duration seconds)
CONSONANTS: dict[str, tuple[float, float, float]] = {
    "k": (1700.0, 800.0, 0.030),
    "t": (3200.0, 900.0, 0.028),
    "p": (900.0, 500.0, 0.026),
    "b": (750.0, 450.0, 0.026),
    "d": (2600.0, 800.0, 0.028),
    "g": (1900.0, 700.0, 0.030),
    "s": (5200.0, 1200.0, 0.060),
    "v": (1300.0, 600.0, 0.045),
    "m": (280.0, 200.0, 0.050),
    "n": (380.0, 250.0, 0.050),
    "l": (620.0, 350.0, 0.045),
    "r": (950.0, 450.0, 0.045),
}
