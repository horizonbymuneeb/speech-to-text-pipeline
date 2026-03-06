"""Chunk planning, overlap stitching and the streaming decoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asr.config import DecodeConfig, FeatureConfig
from asr.decode import (
    ChunkPlan,
    DecodeError,
    StreamingDecoder,
    decode_offline,
    longest_overlap,
    merge_tokens,
    plan_chunks,
)
from asr.features import FeatureExtractor
from asr.vocab import Vocabulary

SR = 8000
FEATURES = FeatureConfig(n_fft=256, n_mels=16, use_deltas=False, cmvn=False)
DECODE = DecodeConfig(chunk_ms=400.0, overlap_ms=120.0, beam_width=1, blank_index=0)


class ConstantModel:
    """Emits the same token at every frame, regardless of context."""

    def __init__(self, token: int = 1, vocab_size: int = 4) -> None:
        self.token = token
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, features, lengths=None, state=None):
        self.calls += 1
        frames = features.shape[1]
        downsampled = max(1, (frames + 1) // 2)
        logits = torch.full((downsampled, 1, self.vocab_size), -8.0)
        logits[:, :, self.token] = 8.0
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, torch.tensor([downsampled]), state


class TailModel(ConstantModel):
    """Emits one token at the *last* frame of each chunk, different per chunk.

    A token at the last frame is outside the overlapped region, so the decoder
    must append it — this is the case that proves new audio is not dropped.
    """

    def forward(self, features, lengths=None, state=None):
        self.calls += 1
        frames = features.shape[1]
        downsampled = max(2, (frames + 1) // 2)
        logits = torch.full((downsampled, 1, self.vocab_size), -8.0)
        logits[-1, :, ((self.calls - 1) % 3) + 1] = 8.0
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs, torch.tensor([downsampled]), state


@pytest.fixture()
def extractor() -> FeatureExtractor:
    return FeatureExtractor(FEATURES, SR)


@pytest.fixture()
def vocabulary() -> Vocabulary:
    return Vocabulary.from_words(("kasa", "timo", "velu"))


@pytest.fixture()
def signal() -> np.ndarray:
    rng = np.random.default_rng(0)
    time = np.arange(8000) / SR
    return (0.5 * np.sin(2 * np.pi * 200 * time) + 0.05 * rng.normal(size=8000)).astype(np.float32)


def test_chunk_plan_geometry():
    plan = plan_chunks(10000, 3200, 960)
    assert plan.hop_samples == 2240
    # windows, not boundaries: starts 0, 2240, 4480, 6720, 8960
    assert plan.num_chunks == 5
    assert plan.start(0) == 0
    assert plan.start(1) == 2240
    assert plan.end(plan.num_chunks - 1) == 10000
    ranges = plan.ranges()
    assert ranges[0][0] == 0
    assert all(end > start for start, end in ranges)
    assert ranges[-1][1] == 10000


def test_chunk_plan_single_chunk_when_short():
    plan = plan_chunks(1000, 3200, 960)
    assert plan.num_chunks == 1
    assert plan.ranges() == [(0, 1000)]


def test_chunk_plan_validation():
    with pytest.raises(DecodeError, match="total_samples"):
        plan_chunks(0, 3200, 960)
    with pytest.raises(DecodeError, match="chunk_samples"):
        plan_chunks(100, 0, 0)
    with pytest.raises(DecodeError, match="overlap"):
        plan_chunks(100, 3200, 3200)
    plan = plan_chunks(10000, 3200, 960)
    with pytest.raises(DecodeError, match="out of range"):
        plan.start(99)
    assert plan.to_dict()["num_chunks"] == 5
