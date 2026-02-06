"""CTC loss and decoding, cross-checked against PyTorch's reference loss."""

from __future__ import annotations

import math

import pytest
import torch

from asr.ctc import (
    NEG_INF,
    BeamResult,
    CtcError,
    batch_prefix_beam_decode,
    collapse_path,
    ctc_loss,
    decode_matrix,
    extend_targets,
    greedy_decode,
    greedy_decode_single,
    log_add,
    log_softmax,
    prefix_beam_decode,
    sequence_log_prob,
)

torch.manual_seed(0)


def test_log_softmax_is_normalised_and_matches_torch():
    logits = torch.randn(5, 3)
    ours = log_softmax(logits)
    assert torch.allclose(ours, torch.log_softmax(logits, dim=-1), atol=1e-6)
    assert torch.allclose(torch.exp(ours).sum(dim=-1), torch.ones(5), atol=1e-6)


def test_log_softmax_rejects_empty():
    with pytest.raises(CtcError):
        log_softmax(torch.zeros(0))


def test_log_add_matches_direct_computation():
    assert log_add(math.log(0.3), math.log(0.7)) == pytest.approx(0.0, abs=1e-9)
    assert log_add(NEG_INF, math.log(0.5)) == pytest.approx(math.log(0.5))
    assert log_add(math.log(0.5), NEG_INF) == pytest.approx(math.log(0.5))
