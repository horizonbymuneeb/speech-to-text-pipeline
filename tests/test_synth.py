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


def test_word_concatenates_syllables():
    rng = np.random.default_rng(1)
    short = synthesize_word("kasa", 8000, 120.0, rng)
    long = synthesize_word("kasakasa", 8000, 120.0, rng)
    assert long.size > short.size
    assert rms(short) > 0


def test_crossfade_concat_handles_empty_and_single():
    assert crossfade_concat([], 8000).size == 0
    only = np.ones(100, dtype=np.float32)
    assert np.allclose(crossfade_concat([only], 8000), only)


def test_silence_and_rms():
    assert silence(0.1, 8000).size == 800
    assert np.all(silence(0.1, 8000) == 0.0)
    assert rms(np.zeros(10)) == 0.0
    assert rms(np.ones(10)) == pytest.approx(1.0)


def test_normalize_peak():
    signal = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    scaled = normalize_peak(signal, 0.5)
    assert np.max(np.abs(scaled)) == pytest.approx(0.5, rel=1e-5)
    assert np.all(normalize_peak(np.zeros(4), 0.5) == 0.0)
    with pytest.raises(SynthesisError, match="peak"):
        normalize_peak(signal, 2.0)


def test_add_noise_lowers_snr():
    rng = np.random.default_rng(0)
    clean = normalize_peak(np.sin(np.linspace(0, 20 * np.pi, 4000)), 0.5)
    noisy = add_noise(clean, 10.0, rng)
    residual = noisy - clean
    assert rms(residual) > 0.05
    assert add_noise(np.zeros(100), 10.0, rng).shape == (100,)


def test_utterance_spans_match_words():
    rng = np.random.default_rng(5)
    signal = synthesize_utterance(["kasa", "timo"], AUDIO, SYNTH, rng)
    assert signal.words == ("kasa", "timo")
    assert len(signal.word_spans) == 2
    for word, (start, end) in zip(signal.words, signal.word_spans):
        assert 0 <= start < end <= signal.num_samples
        assert signal.word_at_sample(start) == word
    assert signal.word_spans[0][1] <= signal.word_spans[1][0]


def test_utterance_amplitude_is_bounded():
    rng = np.random.default_rng(6)
    signal = synthesize_utterance(["velu", "nara", "kasa"], AUDIO, SYNTH, rng)
    assert np.max(np.abs(signal.samples)) <= 1.0
    assert signal.transcript() == "velu nara kasa"
    assert signal.duration_s == pytest.approx(signal.num_samples / 8000)


def test_utterance_requires_words():
    with pytest.raises(SynthesisError, match="at least one word"):
        synthesize_utterance([], AUDIO, SYNTH, np.random.default_rng(0))


def test_corpus_is_deterministic():
    first = make_corpus(AUDIO, SYNTH, 3, seed=11)
    second = make_corpus(AUDIO, SYNTH, 3, seed=11)
    third = make_corpus(AUDIO, SYNTH, 3, seed=12)
    assert [s.words for s in first] == [s.words for s in second]
    assert all(np.array_equal(a.samples, b.samples) for a, b in zip(first, second))
    assert any(not np.array_equal(a.samples, b.samples) for a, b in zip(first, third))


def test_corpus_word_counts_within_range():
    corpus = make_corpus(AUDIO, SYNTH, 5, seed=2)
    assert len(corpus) == 5
    for signal in corpus:
        assert 2 <= len(signal.words) <= 3
        assert all(word in SYNTH.vocabulary for word in signal.words)


def test_corpus_helpers():
    corpus = make_corpus(AUDIO, SYNTH, 2, seed=4)
    assert corpus_transcripts(corpus) == [signal.transcript() for signal in corpus]
    assert corpus_duration_s(corpus) == pytest.approx(sum(s.duration_s for s in corpus))
    with pytest.raises(SynthesisError, match="count"):
        make_corpus(AUDIO, SYNTH, 0)


def test_sample_words_is_seeded():
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    assert sample_words(SYNTH, rng_a) == sample_words(SYNTH, rng_b)
