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

FORMANT_GAINS = (1.0, 0.55, 0.28)
FORMANT_BANDWIDTHS = (90.0, 120.0, 180.0)
VOWEL_SECONDS = 0.150
FADE_SECONDS = 0.005
LEAD_IN_SECONDS = 0.100
TAIL_SECONDS = 0.120


class SynthesisError(ValueError):
    """Raised when a word cannot be syllabified or a phoneme is unknown."""


@dataclass(frozen=True)
class Syllable:
    """A consonant onset (possibly empty) plus a vowel nucleus."""

    onset: str
    vowel: str

    def __post_init__(self) -> None:
        if self.vowel not in VOWEL_FORMANTS:
            raise SynthesisError(f"unknown vowel {self.vowel!r}")
        if self.onset and self.onset not in CONSONANTS:
            raise SynthesisError(f"unknown consonant {self.onset!r}")


@dataclass(frozen=True)
class Signal:
    """A waveform with its ground-truth words and sample spans."""

    samples: np.ndarray
    sample_rate: int
    words: tuple[str, ...]
    word_spans: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.samples.ndim != 1:
            raise SynthesisError("samples must be mono (1-D)")
        if len(self.word_spans) != len(self.words):
            raise SynthesisError("word_spans must have one entry per word")
        if self.sample_rate <= 0:
            raise SynthesisError("sample_rate must be positive")

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_samples / float(self.sample_rate)

    def tokens(self, vocabulary: Vocabulary) -> list[int]:
        """Ground-truth token ids for this utterance."""
        return vocabulary.encode(self.words)

    def word_at_sample(self, sample: int) -> str | None:
        """Word whose span contains ``sample`` (``None`` in a gap)."""
        for word, (start, end) in zip(self.words, self.word_spans):
            if start <= sample < end:
                return word
        return None

    def transcript(self) -> str:
        """Space-joined ground truth."""
        return " ".join(self.words)


def syllabify(word: str) -> list[Syllable]:
    """Split a lowercase word into (onset, vowel) syllables.

    Every vowel ends a syllable; the consonants collected since the previous
    vowel become the next syllable's onset.
    """
    if not word or not word.isalpha() or word != word.lower():
        raise SynthesisError(f"word {word!r} must be lowercase letters")
    syllables: list[Syllable] = []
    pending = ""
    for char in word:
        if char in VOWEL_FORMANTS:
            syllables.append(Syllable(onset=pending, vowel=char))
            pending = ""
        elif char in CONSONANTS:
            if len(pending) > 1:
                raise SynthesisError(
                    f"word {word!r} has a consonant cluster {pending!r} that this "
                    "synthesiser does not model"
                )
            pending += char
        else:
            raise SynthesisError(f"word {word!r} contains unsupported character {char!r}")
    if not syllables:
        raise SynthesisError(f"word {word!r} has no vowel")
    if pending:
        raise SynthesisError(f"word {word!r} ends with a consonant cluster {pending!r}")
    return syllables


def formant_envelope(
    frequencies: np.ndarray, formants: Sequence[float]
) -> np.ndarray:
    """Amplitude envelope from Gaussian resonances at the formant frequencies."""
    envelope = np.full_like(frequencies, 0.02, dtype=np.float64)
    for gain, centre, bandwidth in zip(FORMANT_GAINS, formants, FORMANT_BANDWIDTHS):
        envelope += gain * np.exp(-0.5 * ((frequencies - centre) / bandwidth) ** 2)
    return envelope


def raised_cosine_envelope(
    length: int, attack: int, release: int
) -> np.ndarray:
    """Attack/release envelope; ``attack + release`` must fit inside ``length``."""
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    attack = max(0, min(attack, length))
    release = max(0, min(release, length - attack))
    envelope = np.ones(length, dtype=np.float64)
    if attack:
        envelope[:attack] = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, attack))
    if release:
        envelope[length - release :] = 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, release))
    return envelope
