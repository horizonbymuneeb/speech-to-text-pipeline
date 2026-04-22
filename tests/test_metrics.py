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
