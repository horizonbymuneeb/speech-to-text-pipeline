"""Synthetic speech generation: determinism, spans and signal sanity."""

from __future__ import annotations

import numpy as np
import pytest

from asr.config import AudioConfig, SynthConfig
from asr.synth import (
    CONSONANTS,
    VOWEL_FORMANTS,
    SynthesisError,
    add_noise,
    corpus_duration_s,
    corpus_transcripts,
    crossfade_concat,
    make_corpus,
    normalize_peak,
    rms,
    sample_words,
    silence,
    syllabify,
    synthesize_consonant,
    synthesize_utterance,
    synthesize_vowel,
    synthesize_word,
)

AUDIO = AudioConfig(sample_rate=8000, peak_amplitude=0.9, noise_snr_db=25.0)
SYNTH = SynthConfig(
    seed=3,
    vocabulary=("kasa", "timo", "velu", "nara"),
    words_per_utterance=(2, 3),
    f0_hz=120.0,
    gap_ms=(40.0, 80.0),
)


def test_syllabify_simple_words():
    syllables = syllabify("kasa")
    assert [(s.onset, s.vowel) for s in syllables] == [("k", "a"), ("s", "a")]
    assert [(s.onset, s.vowel) for s in syllabify("timo")] == [("t", "i"), ("m", "o")]


def test_syllabify_rejects_bad_words():
    with pytest.raises(SynthesisError, match="consonant cluster"):
        syllabify("kstra")
    with pytest.raises(SynthesisError, match="no vowel"):
        syllabify("k")
    with pytest.raises(SynthesisError, match="lowercase"):
        syllabify("Kasa")


def test_vowel_has_energy_and_expected_length():
    wave = synthesize_vowel("a", 8000, 120.0, duration_s=0.1)
    assert wave.shape == (800,)
    assert np.max(np.abs(wave)) == pytest.approx(1.0, abs=1e-5)
    assert rms(wave) > 0.05


def test_vowel_spectrum_peaks_near_first_formant():
    wave = synthesize_vowel("a", 8000, 100.0, duration_s=0.2)
    spectrum = np.abs(np.fft.rfft(wave))
    freqs = np.fft.rfftfreq(wave.size, d=1.0 / 8000)
    peak = freqs[int(np.argmax(spectrum))]
    first_formant = VOWEL_FORMANTS["a"][0]
    assert abs(peak - first_formant) < 300.0


def test_unknown_vowel_raises():
    with pytest.raises(SynthesisError, match="unknown vowel"):
        synthesize_vowel("x", 8000, 120.0)


def test_consonant_burst_is_band_limited_noise():
    rng = np.random.default_rng(0)
    burst = synthesize_consonant("s", 8000, rng)
    assert burst.size == int(round(CONSONANTS["s"][2] * 8000))
    assert np.max(np.abs(burst)) <= 0.36
    with pytest.raises(SynthesisError, match="unknown consonant"):
        synthesize_consonant("q", 8000, rng)
