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


def test_ctc_loss_reductions_agree():
    log_probs = torch.log_softmax(torch.randn(8, 2, 4), dim=-1)
    targets = torch.tensor([[1, 2], [3, 0]])
    inputs = torch.tensor([8, 8])
    lengths = torch.tensor([2, 1])
    none = ctc_loss(log_probs, targets, inputs, lengths, reduction="none")
    total = ctc_loss(log_probs, targets, inputs, lengths, reduction="sum")
    mean = ctc_loss(log_probs, targets, inputs, lengths, reduction="mean")
    assert float(total) == pytest.approx(float(none.sum()), abs=1e-5)
    # torch convention: per-sample loss / target length, then batch mean
    expected_mean = float((none / lengths.to(none.dtype)).mean())
    assert float(mean) == pytest.approx(expected_mean, abs=1e-5)
    assert float(mean) != pytest.approx(float(none.mean()))


def test_ctc_loss_validation():
    log_probs = torch.log_softmax(torch.randn(5, 2, 3), dim=-1)
    targets = torch.tensor([[1, 2], [1, 2]])
    inputs = torch.tensor([5, 5])
    lengths = torch.tensor([2, 2])
    with pytest.raises(CtcError, match="\\(time, batch, classes\\)"):
        ctc_loss(log_probs[0], targets, inputs, lengths)
    with pytest.raises(CtcError, match="targets"):
        ctc_loss(log_probs, targets.unsqueeze(0), inputs, lengths)
    with pytest.raises(CtcError, match="batch size"):
        ctc_loss(log_probs, targets[:1], inputs, lengths)
    with pytest.raises(CtcError, match="reduction"):
        ctc_loss(log_probs, targets, inputs, lengths, reduction="median")
    with pytest.raises(CtcError, match="exceeds"):
        ctc_loss(log_probs[:, :1, :], torch.tensor([[1, 2, 1, 2, 1, 2]]), inputs[:1], torch.tensor([6]))


def test_collapse_path_rules():
    assert collapse_path([1, 1, 1]) == [1]
    assert collapse_path([1, 0, 1]) == [1, 1]
    assert collapse_path([0, 0, 0]) == []
    assert collapse_path([1, 2, 2, 0, 3]) == [1, 2, 3]


def test_greedy_decode_batch():
    log_probs = torch.full((4, 4, 3), -10.0)
    log_probs[0, 0, 1] = 10.0  # batch 0: a single token
    log_probs[0, 1, 1] = 10.0  # batch 1: the same token twice -> collapses
    log_probs[1, 1, 1] = 10.0
    log_probs[0, 3, 2] = 10.0  # batch 3: a different token
    decoded = greedy_decode(log_probs)
    assert decoded[0] == [1]
    assert decoded[1] == [1]
    assert decoded[2] == []
    assert decoded[3] == [2]


def test_greedy_decode_single_and_validation():
    matrix = torch.full((3, 3), -10.0)
    matrix[:, 2] = 10.0
    assert greedy_decode_single(matrix) == [2]
    with pytest.raises(CtcError):
        greedy_decode_single(matrix.unsqueeze(0))


def test_prefix_beam_matches_greedy_on_peaked_matrix():
    torch.manual_seed(3)
    logits = torch.randn(9, 4) * 6.0
    log_probs = torch.log_softmax(logits, dim=-1)
    beam = prefix_beam_decode(log_probs, beam_width=5)
    assert list(beam.tokens) == greedy_decode_single(log_probs)
