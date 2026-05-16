"""WER/CER with hand-computed alignments."""

from __future__ import annotations

import math

import pytest

from asr.metrics import (
    OP_DEL,
    OP_EQUAL,
    OP_INS,
    OP_SUB,
    Alignment,
    EditOp,
    MetricError,
    align,
    cer_with_alignment,
    char_error_rate,
    corpus_cer,
    corpus_wer,
    edit_distance,
    error_counts,
    format_alignment,
    summarize,
    wer_with_alignment,
    word_error_rate,
)


def test_identical_sequences():
    alignment = align(["a", "b", "c"], ["a", "b", "c"])
    assert alignment.distance == 0
    assert alignment.rate == 0.0
    assert alignment.op_string() == "= = ="
    assert alignment.correct == 3


def test_single_substitution():
    alignment = align(["a", "b", "c"], ["a", "x", "c"])
    assert alignment.distance == 1
    assert alignment.substitutions == 1
    assert alignment.rate == pytest.approx(1 / 3)
    assert alignment.op_string() == "= S ="


def test_single_insertion():
    alignment = align(["a", "b"], ["a", "b", "c"])
    assert alignment.distance == 1
    assert alignment.insertions == 1
    assert alignment.deletions == 0
    assert alignment.rate == pytest.approx(0.5)
    assert alignment.op_string() == "= = I"


def test_single_deletion():
    alignment = align(["a", "b", "c"], ["a", "c"])
    assert alignment.distance == 1
    assert alignment.deletions == 1
    assert alignment.rate == pytest.approx(1 / 3)
    assert alignment.op_string() == "= D ="


def test_mixed_errors_are_counted_exactly():
    alignment = align(["a", "b", "c", "d"], ["a", "x", "c", "e", "f"])
    counts = alignment.counts()
    assert counts["substitutions"] == 2  # b->x, d->e
    assert counts["insertions"] == 1  # trailing f
    assert counts["deletions"] == 0
    assert alignment.distance == 3
    assert alignment.rate == pytest.approx(3 / 4)


def test_hand_computed_distance_matrix():
    # classic example: kitten -> sitting is 3 edits
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], ["a", "b"]) == 2
    assert edit_distance(["a", "b"], []) == 2
    assert edit_distance([], []) == 0


def test_alignment_preserves_edit_distance():
    cases = [
        (["a", "b", "c"], ["a", "b", "c"]),
        (["a", "b", "c"], ["c", "b", "a"]),
        (["kasa", "timo"], ["timo"]),
        (["x"], ["x", "x", "x"]),
    ]
    for reference, hypothesis in cases:
        assert align(reference, hypothesis).distance == edit_distance(reference, hypothesis)


def test_transposition_costs_two_edits():
    alignment = align(["a", "b"], ["b", "a"])
    assert alignment.distance == 2
    assert alignment.op_string() in {"S S", "D I", "I D"}


def test_empty_reference_conventions():
    assert word_error_rate([], []) == 0.0
    assert word_error_rate([], ["a"]) == math.inf
    assert word_error_rate(["a"], []) == pytest.approx(1.0)


def test_character_error_rate():
    assert char_error_rate("abc", "abc") == 0.0
    assert char_error_rate("abc", "axc") == pytest.approx(1 / 3)
    assert char_error_rate("abc", "ab") == pytest.approx(1 / 3)
    # one insertion over a two-character reference: denominator is the reference
    assert char_error_rate("ab", "abc") == pytest.approx(0.5)
    with pytest.raises(MetricError):
        char_error_rate(["a"], "b")


def test_cer_alignment_matches_rate():
    rate, alignment = cer_with_alignment("hello", "hallo")
    assert rate == pytest.approx(0.2)
    assert alignment.substitutions == 1


def test_wer_alignment_pair():
    rate, alignment = wer_with_alignment(["a", "b"], ["a"])
    assert rate == pytest.approx(0.5)
    assert alignment.deletions == 1


def test_corpus_wer_pools_counts_not_rates():
    # utterance 1: 1 error / 4 words; utterance 2: 1 error / 1 word
    pairs = [
        (["a", "b", "c", "d"], ["a", "b", "c", "x"]),
        (["a"], ["b"]),
    ]
    assert corpus_wer(pairs) == pytest.approx(2 / 5)
    # averaging the two rates would give (0.25 + 1.0) / 2 = 0.625
    assert corpus_wer(pairs) != pytest.approx(0.625)


def test_corpus_cer_pools_characters():
    pairs = [("abcd", "abcx"), ("a", "b")]
    assert corpus_cer(pairs) == pytest.approx(2 / 5)


def test_corpus_empty_conventions():
    assert corpus_wer([]) == 0.0
    assert corpus_wer([([], [])]) == 0.0
    assert corpus_wer([([], ["a"])]) == math.inf
