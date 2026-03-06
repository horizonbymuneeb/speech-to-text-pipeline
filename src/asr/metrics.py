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


@dataclass(frozen=True)
class Alignment:
    """An edit script plus its cost."""

    ops: tuple[EditOp, ...]
    distance: int
    ref_length: int
    hyp_length: int

    @property
    def correct(self) -> int:
        return sum(1 for op in self.ops if op.kind == OP_EQUAL)

    @property
    def substitutions(self) -> int:
        return sum(1 for op in self.ops if op.kind == OP_SUB)

    @property
    def deletions(self) -> int:
        return sum(1 for op in self.ops if op.kind == OP_DEL)

    @property
    def insertions(self) -> int:
        return sum(1 for op in self.ops if op.kind == OP_INS)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Errors per reference token (``inf`` for an empty reference)."""
        if self.ref_length == 0:
            return 0.0 if self.errors == 0 else INIFINITY
        return self.errors / float(self.ref_length)

    def counts(self) -> dict[str, int]:
        """Counts per op kind, including correct matches."""
        return {
            "correct": self.correct,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "ref_length": self.ref_length,
            "hyp_length": self.hyp_length,
        }

    def op_string(self) -> str:
        """Compact op sequence such as ``= = S I``."""
        return " ".join(op.symbol() for op in self.ops)

    def to_dict(self) -> dict[str, object]:
        return {
            "distance": self.distance,
            "rate": self.rate,
            "counts": self.counts(),
            "ops": [
                {
                    "kind": op.kind,
                    "ref_index": op.ref_index,
                    "hyp_index": op.hyp_index,
                    "ref_token": op.ref_token,
                    "hyp_token": op.hyp_token,
                }
                for op in self.ops
            ],
        }


def _as_tokens(sequence: Sequence[str]) -> list[str]:
    if isinstance(sequence, str):
        raise MetricError("pass a sequence of tokens, not a bare string")
    return [str(token) for token in sequence]


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences."""
    ref = _as_tokens(reference)
    hyp = _as_tokens(hypothesis)
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, start=1):
        current = [i]
        for j, hyp_token in enumerate(hyp, start=1):
            substitution = previous[j - 1] + (0 if ref_token == hyp_token else 1)
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]
