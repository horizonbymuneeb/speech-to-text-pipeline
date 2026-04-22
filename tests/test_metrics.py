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
