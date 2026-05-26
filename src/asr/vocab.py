"""Token vocabulary shared by the synthetic corpus, the model and the scorer.

The corpus is word-level, so the vocabulary is a list of words. Index 0 is
reserved for the CTC blank, which means a word's token id is its position in
``words`` plus one. Keeping the blank at 0 is a convention the decoder, the
model output layer and the tests all rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence
