"""Word and character error rates over a hand-written edit-distance alignment.

The alignment is a plain Levenshtein dynamic program with a backtrace, not a
library call, because the *ops* matter as much as the number: a scorer that only
returns a distance cannot tell you whether a system is dropping words or
hallucinating them, and that distinction is what makes WER useful. The backtrace
tie-break is fixed (substitution, then deletion, then insertion) so the same
input always produces the same op sequence — otherwise test assertions on
insertion/deletion counts would be flaky.

Error-rate conventions used here:

* ``word_error_rate([], [])`` is ``0.0``; a non-empty hypothesis against an empty
  reference is ``inf`` (every emitted word is an insertion, and the denominator
  would otherwise be zero).
* ``corpus_wer`` pools errors and reference lengths across utterances instead of
  averaging per-utterance rates; averaging lets a one-word utterance dominate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

OP_EQUAL = "equal"
OP_SUB = "sub"
OP_INS = "ins"
OP_DEL = "del"
OP_KINDS = (OP_EQUAL, OP_SUB, OP_INS, OP_DEL)

INIFINITY = float("inf")


class MetricError(ValueError):
    """Raised for inputs the scorer cannot handle."""


@dataclass(frozen=True)
class EditOp:
    """One column of an alignment."""

    kind: str
    ref_index: int | None
    hyp_index: int | None
    ref_token: str | None
    hyp_token: str | None

    def __post_init__(self) -> None:
        if self.kind not in OP_KINDS:
            raise MetricError(f"unknown op kind {self.kind!r}")
        if self.kind in (OP_EQUAL, OP_SUB) and (self.ref_token is None or self.hyp_token is None):
            raise MetricError(f"{self.kind} needs both tokens")
        if self.kind == OP_DEL and self.ref_token is None:
            raise MetricError("deletion needs a reference token")
        if self.kind == OP_INS and self.hyp_token is None:
            raise MetricError("insertion needs a hypothesis token")

    @property
    def is_error(self) -> bool:
        return self.kind != OP_EQUAL

    def symbol(self) -> str:
        """Single-character marker: ``=``, ``S``, ``D``, ``I``."""
        return {
            OP_EQUAL: "=",
            OP_SUB: "S",
            OP_DEL: "D",
            OP_INS: "I",
        }[self.kind]
