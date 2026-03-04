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
