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


def test_unknown_word_raises():
    vocab = Vocabulary.from_words(WORDS)
    with pytest.raises(VocabularyError, match="not in the vocabulary"):
        vocab.word_to_token("nope")


def test_token_out_of_range_raises():
    vocab = Vocabulary.from_words(WORDS)
    with pytest.raises(VocabularyError, match="out of range"):
        vocab.token_to_word(99)
    with pytest.raises(VocabularyError, match="blank"):
        vocab.token_to_word(0)


def test_duplicate_words_rejected():
    with pytest.raises(VocabularyError, match="duplicate"):
        Vocabulary.from_words(("a", "a"))


def test_empty_vocabulary_rejected():
    with pytest.raises(VocabularyError, match="at least one word"):
        Vocabulary.from_words(())


def test_non_zero_blank_rejected():
    with pytest.raises(VocabularyError, match="index 0"):
        Vocabulary(words=WORDS, blank_index=1)


def test_contains_and_valid_token():
    vocab = Vocabulary.from_words(WORDS)
    assert vocab.contains("timo") is True
    assert vocab.contains("nope") is False
    assert vocab.is_valid_token(0) is True
    assert vocab.is_valid_token(3) is True
    assert vocab.is_valid_token(4) is False


def test_token_table():
    vocab = Vocabulary.from_words(WORDS)
    table = vocab.token_table()
    assert table[0] == BLANK_TOKEN
    assert table[1] == "kasa"
    assert len(table) == 4


def test_dict_roundtrip():
    vocab = Vocabulary.from_words(WORDS)
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.words == vocab.words
    assert restored.size == vocab.size


def test_from_dict_rejects_bad_payload():
    with pytest.raises(VocabularyError, match="words"):
        Vocabulary.from_dict({"words": []})
