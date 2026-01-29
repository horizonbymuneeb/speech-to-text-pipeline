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
