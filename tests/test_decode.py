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
