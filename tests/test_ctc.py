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


def test_extend_targets_interleaves_blanks():
    targets = torch.tensor([[1, 2], [3, 0]])
    extended = extend_targets(targets, blank=0)
    assert extended.shape == (2, 5)
    assert extended[0].tolist() == [0, 1, 0, 2, 0]
    assert extended[1].tolist() == [0, 3, 0, 0, 0]


def test_ctc_loss_matches_torch_reference():
    time_steps, batch, classes, target_length = 14, 3, 5, 4
    log_probs = torch.log_softmax(torch.randn(time_steps, batch, classes), dim=-1)
    targets = torch.randint(1, classes, (batch, target_length))
    input_lengths = torch.tensor([14, 12, 14])
    target_lengths = torch.tensor([3, 4, 2])
    ours = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0)
    reference = torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )
    assert torch.isfinite(ours)
    assert torch.allclose(ours, reference, atol=1e-5)


def test_ctc_loss_single_frame_exact_value():
    log_probs = torch.log_softmax(torch.tensor([[2.0, 1.0, 0.5]]), dim=-1).unsqueeze(1)
    targets = torch.tensor([[1]])
    loss = ctc_loss(
        log_probs, targets, torch.tensor([1]), torch.tensor([1]), reduction="sum"
    )
    assert float(loss) == pytest.approx(float(-log_probs[0, 0, 1]), abs=1e-6)
