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


class AcousticModel(Protocol):
    """The slice of the model interface the decoder needs."""

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor | None = None, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...


class StreamingDecoder:
    """Decode a waveform chunk by chunk, carrying state across the seams."""

    def __init__(
        self,
        model: AcousticModel,
        extractor: FeatureExtractor,
        vocabulary: Vocabulary,
        decode_config: DecodeConfig,
        feature_config: FeatureConfig | None = None,
        sample_rate: int | None = None,
    ) -> None:
        self.model = model
        self.extractor = extractor
        self.vocabulary = vocabulary
        self.config = decode_config
        self.feature_config = feature_config or extractor.config
        self.sample_rate = sample_rate or extractor.sample_rate

    def reset(self) -> None:
        """Hook for stream-level state; chunk features are stateless here."""

    def chunk_plan(self, num_samples: int) -> ChunkPlan:
        """Chunk plan for a stream of ``num_samples`` samples."""
        return plan_chunks(
            num_samples,
            self.config.chunk_samples(self.sample_rate),
            self.config.overlap_samples(self.sample_rate),
        )

    def _features(self, samples: np.ndarray) -> np.ndarray:
        """Chunk features.

        The extractor applies CMVN from fixed corpus statistics when it has them,
        so a chunk is normalised the same way it would be inside a full
        utterance. Per-chunk normalisation would make the features drift at every
        seam, which is the streaming equivalent of a distribution shift.
        """
        return self.extractor.transform(samples)

    def overlap_output_frames(self) -> int:
        """How many model output frames correspond to the overlapped audio."""
        hop = self.extractor.hop_length
        factor = int(getattr(self.model, "downsample_factor", 2) or 1)
        overlap_features = self.config.overlap_samples(self.sample_rate) // max(1, hop)
        return int(overlap_features // max(1, factor))

    def decode_chunk(
        self, samples: np.ndarray, state: torch.Tensor | None = None
    ) -> tuple[list[int], torch.Tensor | None]:
        """Decode one chunk, returning its tokens and the new recurrent state."""
        timed, new_state = self.decode_chunk_timed(samples, state)
        return [token for token, _ in timed], new_state

    def decode_chunk_timed(
        self, samples: np.ndarray, state: torch.Tensor | None = None
    ) -> tuple[list[tuple[int, int]], torch.Tensor | None]:
        """Decode one chunk, returning ``(token, output_frame)`` pairs.

        Beam search does not expose a frame alignment, so in beam mode the frame
        index is reported as -1 and the caller falls back to identity matching.
        """
        if np.asarray(samples).size == 0:
            raise DecodeError("cannot decode an empty chunk")
        features = self._features(np.asarray(samples))
        tensor = torch.from_numpy(features).unsqueeze(0)
        with torch.no_grad():
            log_probs, _, new_state = self.model.forward(tensor, state=state)
        matrix = log_probs[:, 0, :]
        if self.config.beam_width > 1:
            results = batch_prefix_beam_decode(
                log_probs, beam_width=self.config.beam_width, blank=self.config.blank_index
            )
            return [(token, -1) for token in results[0].tokens], new_state
        path = torch.argmax(matrix, dim=-1).tolist()
        return collapse_path_with_positions(path, blank=self.config.blank_index), new_state

    def decode_samples(self, samples: np.ndarray) -> DecodeResult:
        """Decode a whole waveform in chunks."""
        signal = np.asarray(samples)
        if signal.ndim != 1:
            raise DecodeError("samples must be mono (1-D)")
        if signal.size == 0:
            raise DecodeError("cannot decode an empty signal")
        self.reset()
        plan = self.chunk_plan(signal.size)
        state: torch.Tensor | None = None
        tokens: list[int] = []
        chunk_tokens: list[int] = []
        overlaps: list[int] = []
        frames = 0
        seam = self.overlap_output_frames()
        for index in range(plan.num_chunks):
            start, end = plan.start(index), plan.end(index)
            chunk = signal[start:end]
            timed, state = self.decode_chunk_timed(chunk, state)
            chunk_tokens.append(len(timed))
            if index == 0 or any(frame < 0 for _, frame in timed):
                # no frame information (beam search): match by token identity
                plain = [token for token, _ in timed]
                tokens, overlap = merge_tokens(tokens, plain)
            else:
                # Greedy decode: drop everything emitted from the overlapped
                # audio. Dropping by frame instead of by token identity is what
                # keeps a genuine repeat ("timo timo") from being merged into
                # one token when it happens to sit on a chunk seam.
                kept = [token for token, frame in timed if frame >= seam]
                tokens.extend(kept)
                overlap = len(timed) - len(kept)
            overlaps.append(overlap)
            frames += int(self.extractor.stats(chunk).num_frames)
        words = tuple(self.vocabulary.decode(tokens))
        return DecodeResult(
            tokens=tuple(tokens),
            words=words,
            chunk_count=plan.num_chunks,
            chunk_tokens=tuple(chunk_tokens),
            chunk_overlaps=tuple(overlaps),
            num_frames=frames,
        )

    def decode_words(self, samples: np.ndarray) -> list[str]:
        """Convenience wrapper returning only the transcript words."""
        return list(self.decode_samples(samples).words)


def decode_offline(
    model: AcousticModel,
    extractor: FeatureExtractor,
    vocabulary: Vocabulary,
    samples: np.ndarray,
    blank: int = 0,
) -> list[int]:
    """Decode a whole utterance in one pass (no chunking) — the baseline."""
    features = extractor.transform(np.asarray(samples))
    tensor = torch.from_numpy(features).unsqueeze(0)
    with torch.no_grad():
        log_probs, _, _ = model.forward(tensor)
    return decode_matrix(log_probs[:, 0, :], beam_width=1, blank=blank)
