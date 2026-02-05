"""Streaming decode: chunk the audio, carry state, stitch the tokens.

Streaming a CTC model is not just "decode 400 ms at a time": the model has no
idea what came before unless you give it context, and the seam between two
chunks is exactly where words get duplicated or dropped. Two mechanisms handle
that here.

* **Overlap.** Consecutive chunks share ``overlap_ms`` of audio, so every word
  near a boundary is seen by two windows. The tokens decoded from the shared
  region are then removed with a longest-suffix/longest-prefix match, which is
  cheap and deterministic.
* **Recurrent state.** The GRU hidden state from one chunk is passed into the
  next, so the model keeps its acoustic context even though the convolutional
  front end restarts at every chunk.

CMVN is a third, quieter trap: normalising each chunk on its own makes the
features drift between chunks. The decoder therefore fixes the mean/variance on
the first chunk and reuses those statistics for the rest of the stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch

from asr.config import DecodeConfig, FeatureConfig
from asr.ctc import batch_prefix_beam_decode, collapse_path_with_positions, decode_matrix
from asr.features import FeatureExtractor
from asr.vocab import Vocabulary

EPSILON = 1e-8


class DecodeError(ValueError):
    """Raised for impossible chunk plans or mismatched inputs."""


@dataclass(frozen=True)
class ChunkPlan:
    """Which sample ranges the stream is cut into.

    Contract: ``num_chunks`` counts *windows*, not boundaries. The first window
    starts at sample 0, every following window starts ``hop_samples`` later, and
    the last window is clipped to the end of the stream, so
    ``num_chunks == 1 + ceil((total - chunk) / hop)`` for a stream longer than one
    chunk. A stream shorter than one chunk is a single window.
    """

    total_samples: int
    chunk_samples: int
    overlap_samples: int

    def __post_init__(self) -> None:
        if self.total_samples <= 0:
            raise DecodeError("total_samples must be positive")
        if self.chunk_samples <= 0:
            raise DecodeError("chunk_samples must be positive")
        if self.overlap_samples < 0:
            raise DecodeError("overlap_samples must be non-negative")
        if self.overlap_samples >= self.chunk_samples:
            raise DecodeError("overlap_samples must be smaller than chunk_samples")

    @property
    def hop_samples(self) -> int:
        """Samples advanced between chunk starts."""
        return self.chunk_samples - self.overlap_samples

    @property
    def num_chunks(self) -> int:
        if self.total_samples <= self.chunk_samples:
            return 1
        return 1 + int(np.ceil((self.total_samples - self.chunk_samples) / self.hop_samples))

    def start(self, index: int) -> int:
        """First sample of chunk ``index``."""
        if not 0 <= index < self.num_chunks:
            raise DecodeError(f"chunk index {index} is out of range")
        return index * self.hop_samples

    def end(self, index: int) -> int:
        """One past the last sample of chunk ``index``."""
        return min(self.total_samples, self.start(index) + self.chunk_samples)

    def ranges(self) -> list[tuple[int, int]]:
        """Every ``(start, end)`` sample range in order."""
        return [(self.start(index), self.end(index)) for index in range(self.num_chunks)]

    def to_dict(self) -> dict[str, int]:
        return {
            "total_samples": self.total_samples,
            "chunk_samples": self.chunk_samples,
            "overlap_samples": self.overlap_samples,
            "num_chunks": self.num_chunks,
        }


def plan_chunks(
    total_samples: int, chunk_samples: int, overlap_samples: int
) -> ChunkPlan:
    """Build a chunk plan for a stream of ``total_samples`` samples."""
    return ChunkPlan(
        total_samples=int(total_samples),
        chunk_samples=int(chunk_samples),
        overlap_samples=int(overlap_samples),
    )


def longest_overlap(existing: Sequence[int], new: Sequence[int]) -> int:
    """Length of the longest suffix of ``existing`` that prefixes ``new``.

    This is how the seam between two chunks is found: the shared tokens are the
    ones the second chunk re-decoded from the overlapped audio.
    """
    limit = min(len(existing), len(new))
    for length in range(limit, 0, -1):
        if list(existing[len(existing) - length :]) == list(new[:length]):
            return length
    return 0


def merge_tokens(existing: Sequence[int], new: Sequence[int]) -> tuple[list[int], int]:
    """Append the non-duplicated tail of ``new``; returns ``(tokens, overlap)``."""
    overlap = longest_overlap(existing, new)
    merged = list(existing) + list(new[overlap:])
    return merged, overlap


@dataclass(frozen=True)
class DecodeResult:
    """The output of one streaming decode."""

    tokens: tuple[int, ...]
    words: tuple[str, ...]
    chunk_count: int
    chunk_tokens: tuple[int, ...]
    chunk_overlaps: tuple[int, ...]
    num_frames: int

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": list(self.tokens),
            "words": list(self.words),
            "chunk_count": self.chunk_count,
            "chunk_tokens": list(self.chunk_tokens),
            "chunk_overlaps": list(self.chunk_overlaps),
            "num_frames": self.num_frames,
        }

    def transcript(self) -> str:
        return " ".join(self.words)
