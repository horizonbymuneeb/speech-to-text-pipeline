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
