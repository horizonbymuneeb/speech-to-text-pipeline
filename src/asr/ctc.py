"""Connectionist Temporal Classification: loss, greedy decode, prefix beam search.

Both the loss and the decoders are implemented here rather than imported. The
loss is the standard CTC forward recursion in log space (alpha over the extended
label sequence ``[blank, l1, blank, l2, …, blank]``), vectorised across the batch
and the state axis and looped over time. ``tests/test_ctc.py`` cross-checks it
against ``torch.nn.functional.ctc_loss`` so the hand-written version is not just
self-consistent.

The decoder has two modes. Greedy takes the arg-max path, collapses repeats and
drops blanks — fast, and what streaming uses by default. Prefix beam search keeps
the ``beam_width`` most likely prefixes, tracking the blank-ending and
non-blank-ending probability of each separately, which is what makes it able to
beat greedy on ambiguous frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

NEG_INF = -1e30


class CtcError(ValueError):
    """Raised for malformed CTC inputs."""


@dataclass(frozen=True)
class BeamResult:
    """One decoded sequence and its log probability."""

    tokens: tuple[int, ...]
    score: float

    def to_list(self) -> list[int]:
        return list(self.tokens)


def log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable log-softmax."""
    if logits.numel() == 0:
        raise CtcError("cannot take log_softmax of an empty tensor")
    maximum = torch.max(logits, dim=dim, keepdim=True).values
    shifted = logits - maximum
    return shifted - torch.log(torch.sum(torch.exp(shifted), dim=dim, keepdim=True))


def log_add(a: float, b: float) -> float:
    """``log(exp(a) + exp(b))`` without overflowing."""
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    high, low = (a, b) if a > b else (b, a)
    return high + math.log1p(math.exp(low - high))


def extend_targets(targets: torch.Tensor, blank: int = 0) -> torch.Tensor:
    """Interleave blanks: ``(N, S)`` targets become ``(N, 2S+1)``."""
    if targets.ndim != 2:
        raise CtcError("targets must be 2-D (batch, target_length)")
    batch, length = targets.shape
    extended = torch.full((batch, 2 * length + 1), int(blank), dtype=torch.long)
    if length:
        extended[:, 1::2] = targets
    return extended


def _skip_mask(extended: torch.Tensor, blank: int) -> torch.Tensor:
    """States that may be reached from two steps back (a repeated label blocks it)."""
    states = extended.shape[1]
    mask = torch.zeros_like(extended, dtype=torch.bool)
    if states >= 3:
        mask[:, 2:] = (extended[:, 2:] != int(blank)) & (extended[:, 2:] != extended[:, :-2])
    return mask


def ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    reduction: str = "mean",
) -> torch.Tensor:
    """CTC negative log-likelihood.

    ``log_probs`` is ``(T, N, C)`` of log probabilities, ``targets`` is ``(N, S)``
    of token ids, and the lengths are ``(N,)`` tensors. ``reduction`` is
    ``"mean"``, ``"sum"`` or ``"none"``.

    ``"mean"`` follows PyTorch's convention rather than a plain batch average:
    each sample's loss is divided by its own target length first, then averaged.
    That keeps utterances with different transcript lengths comparable — a plain
    batch mean makes a long transcript dominate the gradient.
    """
    if log_probs.ndim != 3:
        raise CtcError("log_probs must be (time, batch, classes)")
    if targets.ndim != 2:
        raise CtcError("targets must be (batch, target_length)")
    time_steps, batch, classes = log_probs.shape
    if time_steps < 1:
        raise CtcError("need at least one time step")
    if targets.shape[0] != batch:
        raise CtcError("targets batch size does not match log_probs")
    if input_lengths.shape[0] != batch or target_lengths.shape[0] != batch:
        raise CtcError("length tensors must have one entry per batch element")
    if reduction not in ("mean", "sum", "none"):
        raise CtcError(f"unknown reduction {reduction!r}")
    if int(target_lengths.max()) > time_steps:
        raise CtcError("target length exceeds the number of time steps")
    if int(input_lengths.max()) > time_steps:
        raise CtcError("input length exceeds the number of time steps")

    extended = extend_targets(targets, blank)
    states = extended.shape[1]
    skip = _skip_mask(extended, blank)
    gathered = log_probs[:, torch.arange(batch).unsqueeze(1), extended]  # (T, N, S_ext)

    alpha = torch.full((batch, states), NEG_INF, dtype=log_probs.dtype)
    alpha[:, 0] = gathered[0, :, 0]
    if states > 1:
        alpha[:, 1] = gathered[0, :, 1]

    # A sample whose input is shorter than the padded batch must not advance
    # after its last real frame; without this mask the padded frames keep adding
    # probability mass and the loss is silently too low.
    positions = torch.arange(time_steps).unsqueeze(1)
    active = positions < input_lengths.unsqueeze(0)  # (T, N)

    for step in range(1, time_steps):
        stay = alpha
        step_back = torch.full_like(alpha, NEG_INF)
        step_back[:, 1:] = alpha[:, :-1]
        skip_back = torch.full_like(alpha, NEG_INF)
        skip_back[:, 2:] = alpha[:, :-2]
        skip_back = torch.where(skip, skip_back, torch.full_like(skip_back, NEG_INF))
        stacked = torch.stack([stay, step_back, skip_back], dim=-1)
        updated = torch.logsumexp(stacked, dim=-1) + gathered[step]
        alpha = torch.where(active[step].unsqueeze(1), updated, alpha)

    last = 2 * target_lengths
    rows = torch.arange(batch)
    final_blank = alpha[rows, last]
    final_non_blank = torch.where(
        target_lengths > 0,
        alpha[rows, torch.clamp(last - 1, min=0)],
        torch.full_like(final_blank, NEG_INF),
    )
    per_sample = -torch.logsumexp(torch.stack([final_blank, final_non_blank], dim=-1), dim=-1)

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    normaliser = torch.clamp(target_lengths, min=1).to(per_sample.dtype)
    return (per_sample / normaliser).mean()


def sequence_log_prob(
    log_probs: torch.Tensor, tokens: Sequence[int], blank: int = 0
) -> float:
    """Log probability of one exact token sequence under a ``(T, C)`` matrix."""
    if log_probs.ndim != 2:
        raise CtcError("sequence_log_prob expects a single (time, classes) matrix")
    target = torch.tensor([list(int(token) for token in tokens)], dtype=torch.long)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    inputs = torch.tensor([log_probs.shape[0]], dtype=torch.long)
    loss = ctc_loss(log_probs.unsqueeze(1), target, inputs, lengths, blank=blank, reduction="sum")
    return float(-loss)


def collapse_path(path: Sequence[int], blank: int = 0) -> list[int]:
    """Collapse a frame-level path: merge repeats, drop blanks."""
    collapsed: list[int] = []
    previous: int | None = None
    for token in path:
        token = int(token)
        if token != previous and token != int(blank):
            collapsed.append(token)
        previous = token
    return collapsed
