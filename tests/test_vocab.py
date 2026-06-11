"""Vocabulary encoding and the blank-token convention."""

from __future__ import annotations

import pytest

from asr.vocab import BLANK_TOKEN, Vocabulary, VocabularyError

WORDS = ("kasa", "timo", "velu")


def test_blank_is_index_zero():
    vocab = Vocabulary.from_words(WORDS)
    assert vocab.blank_index == 0
    assert vocab.size == 4
    assert vocab.num_words == 3
    assert vocab.word_to_token("kasa") == 1


def test_encode_decode_roundtrip():
    vocab = Vocabulary.from_words(WORDS)
    tokens = vocab.encode(["kasa", "velu"])
    assert tokens == [1, 3]
    assert vocab.decode(tokens) == ["kasa", "velu"]


def test_decode_drops_blanks_but_keeps_repeats():
    vocab = Vocabulary.from_words(WORDS)
    assert vocab.decode([1, 0, 1]) == ["kasa", "kasa"]
