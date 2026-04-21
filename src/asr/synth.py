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


def synthesize_vowel(
    vowel: str,
    sample_rate: int,
    f0_hz: float,
    duration_s: float = VOWEL_SECONDS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Additive-synthesis a vowel: harmonic series shaped by formant resonances."""
    if vowel not in VOWEL_FORMANTS:
        raise SynthesisError(f"unknown vowel {vowel!r}")
    generator = rng or np.random.default_rng(0)
    length = int(round(duration_s * sample_rate))
    if length < 16:
        raise SynthesisError("vowel duration is too short")
    time = np.arange(length, dtype=np.float64) / sample_rate
    nyquist = sample_rate / 2.0
    formants = VOWEL_FORMANTS[vowel]
    signal = np.zeros(length, dtype=np.float64)
    harmonic = 1
    while harmonic * f0_hz < nyquist:
        frequency = harmonic * f0_hz
        amplitude = float(formant_envelope(np.array([frequency]), formants)[0]) / harmonic**0.5
        phase = float(generator.uniform(0.0, 2.0 * np.pi))
        signal += amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
        harmonic += 1
    attack = int(0.015 * sample_rate)
    release = int(0.040 * sample_rate)
    signal *= raised_cosine_envelope(length, attack, release)
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak).astype(np.float32)


def synthesize_consonant(
    consonant: str,
    sample_rate: int,
    rng: np.random.Generator,
    duration_s: float | None = None,
) -> np.ndarray:
    """Band-limited noise burst for a consonant onset."""
    if consonant not in CONSONANTS:
        raise SynthesisError(f"unknown consonant {consonant!r}")
    centre, bandwidth, default_duration = CONSONANTS[consonant]
    seconds = default_duration if duration_s is None else duration_s
    length = max(8, int(round(seconds * sample_rate)))
    noise = rng.normal(0.0, 1.0, length)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(length, d=1.0 / sample_rate)
    spectrum *= np.exp(-0.5 * ((freqs - centre) / bandwidth) ** 2)
    shaped = np.fft.irfft(spectrum, n=length)
    shaped *= raised_cosine_envelope(length, int(0.004 * sample_rate), length // 2)
    peak = float(np.max(np.abs(shaped))) or 1.0
    return (0.35 * shaped / peak).astype(np.float32)


def crossfade_concat(parts: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
    """Concatenate parts with a short equal-power crossfade at each join."""
    if not parts:
        return np.zeros(0, dtype=np.float32)
    fade = max(1, int(FADE_SECONDS * sample_rate))
    result = parts[0].astype(np.float64)
    for part in parts[1:]:
        chunk = part.astype(np.float64)
        overlap = min(fade, len(result), len(chunk))
        if overlap <= 0:
            result = np.concatenate([result, chunk])
            continue
        ramp = np.linspace(0.0, 1.0, overlap)
        blended = result[-overlap:] * (1.0 - ramp) + chunk[:overlap] * ramp
        result = np.concatenate([result[:-overlap], blended, chunk[overlap:]])
    return result.astype(np.float32)


def synthesize_word(
    word: str,
    sample_rate: int,
    f0_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesise one word: consonant burst + vowel, per syllable."""
    parts: list[np.ndarray] = []
    for syllable in syllabify(word):
        if syllable.onset:
            parts.append(synthesize_consonant(syllable.onset, sample_rate, rng))
        parts.append(synthesize_vowel(syllable.vowel, sample_rate, f0_hz, rng=rng))
    return crossfade_concat(parts, sample_rate)


def silence(duration_s: float, sample_rate: int) -> np.ndarray:
    """A zero-valued signal of the requested duration."""
    return np.zeros(max(0, int(round(duration_s * sample_rate))), dtype=np.float32)


def rms(samples: np.ndarray) -> float:
    """Root-mean-square level of a signal."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def normalize_peak(samples: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Scale a signal so its loudest sample is ``peak`` (no-op for silence)."""
    if not 0.0 < peak <= 1.0:
        raise SynthesisError("peak must be in (0, 1]")
    current = float(np.max(np.abs(samples))) if samples.size else 0.0
    if current == 0.0:
        return samples.astype(np.float32)
    return (samples.astype(np.float64) * (peak / current)).astype(np.float32)


def add_noise(
    samples: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add white Gaussian noise at the requested SNR."""
    if samples.size == 0:
        return samples.astype(np.float32)
    signal_power = float(np.mean(np.square(samples, dtype=np.float64)))
    if signal_power <= 0.0:
        return samples.astype(np.float32)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), samples.shape[0])
    return (samples.astype(np.float64) + noise).astype(np.float32)


def synthesize_utterance(
    words: Sequence[str],
    audio: AudioConfig,
    synth: SynthConfig,
    rng: np.random.Generator,
) -> Signal:
    """Build one utterance from ``words`` with gaps, lead-in and noise."""
    if not words:
        raise SynthesisError("an utterance needs at least one word")
    sample_rate = audio.sample_rate
    gap_low, gap_high = synth.gap_ms
    pieces: list[np.ndarray] = [silence(LEAD_IN_SECONDS, sample_rate)]
    spans: list[tuple[int, int]] = []
    cursor = len(pieces[0])
    for index, word in enumerate(words):
        if index:
            gap = silence(float(rng.uniform(gap_low, gap_high)) / 1000.0, sample_rate)
            pieces.append(gap)
            cursor += len(gap)
        word_samples = synthesize_word(word, sample_rate, synth.f0_hz, rng)
        pieces.append(word_samples)
        spans.append((cursor, cursor + len(word_samples)))
        cursor += len(word_samples)
    tail = silence(TAIL_SECONDS, sample_rate)
    pieces.append(tail)
    samples = normalize_peak(np.concatenate(pieces), audio.peak_amplitude)
    samples = add_noise(samples, audio.noise_snr_db, rng)
    return Signal(
        samples=samples,
        sample_rate=sample_rate,
        words=tuple(words),
        word_spans=tuple(spans),
    )


def sample_words(
    synth: SynthConfig, rng: np.random.Generator, vocabulary: Sequence[str] | None = None
) -> list[str]:
    """Sample a word sequence for one utterance."""
    pool = list(vocabulary) if vocabulary is not None else list(synth.vocabulary)
    low, high = synth.words_per_utterance
    count = int(rng.integers(low, high + 1))
    return [str(pool[int(rng.integers(0, len(pool)))]) for _ in range(count)]


def make_corpus(
    audio: AudioConfig,
    synth: SynthConfig,
    count: int,
    seed: int | None = None,
    vocabulary: Sequence[str] | None = None,
) -> list[Signal]:
    """Generate ``count`` utterances deterministically from ``seed``."""
    if count < 1:
        raise SynthesisError("count must be >= 1")
    rng = np.random.default_rng(synth.seed if seed is None else seed)
    return [
        synthesize_utterance(sample_words(synth, rng, vocabulary), audio, synth, rng)
        for _ in range(count)
    ]


def corpus_transcripts(signals: Iterable[Signal]) -> list[str]:
    """Ground-truth transcripts, one string per utterance."""
    return [signal.transcript() for signal in signals]


def corpus_duration_s(signals: Iterable[Signal]) -> float:
    """Total duration of a corpus in seconds."""
    return float(sum(signal.duration_s for signal in signals))
