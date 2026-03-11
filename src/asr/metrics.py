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


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> Alignment:
    """Full edit script between two token sequences.

    Backtrace preference is substitution > deletion > insertion, which keeps the
    script stable and prefers explaining a difference as a substitution rather
    than a delete+insert pair.
    """
    ref = _as_tokens(reference)
    hyp = _as_tokens(hypothesis)
    rows, cols = len(ref), len(hyp)
    cost = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        cost[i][0] = i
    for j in range(cols + 1):
        cost[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            same = 0 if ref[i - 1] == hyp[j - 1] else 1
            cost[i][j] = min(
                cost[i - 1][j - 1] + same,
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
            )
    ops: list[EditOp] = []
    i, j = rows, cols
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = 0 if ref[i - 1] == hyp[j - 1] else 1
            if cost[i][j] == cost[i - 1][j - 1] + same:
                ops.append(
                    EditOp(
                        kind=OP_EQUAL if same == 0 else OP_SUB,
                        ref_index=i - 1,
                        hyp_index=j - 1,
                        ref_token=ref[i - 1],
                        hyp_token=hyp[j - 1],
                    )
                )
                i -= 1
                j -= 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(EditOp(OP_DEL, i - 1, None, ref[i - 1], None))
            i -= 1
            continue
        ops.append(EditOp(OP_INS, None, j - 1, None, hyp[j - 1]))
        j -= 1
    ops.reverse()
    return Alignment(
        ops=tuple(ops),
        distance=cost[rows][cols],
        ref_length=rows,
        hyp_length=cols,
    )


def word_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    """WER as ``(S + D + I) / N`` with ``N`` the reference word count."""
    return align(reference, hypothesis).rate


def wer_with_alignment(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> tuple[float, Alignment]:
    """WER together with the alignment that produced it."""
    alignment = align(reference, hypothesis)
    return alignment.rate, alignment
