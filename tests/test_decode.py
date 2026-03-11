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


def test_longest_overlap():
    assert longest_overlap([1, 2, 3], [2, 3, 4]) == 2
    assert longest_overlap([1, 2, 3], [4, 5]) == 0
    assert longest_overlap([1, 2], [1, 2, 3]) == 2
    assert longest_overlap([], [1]) == 0


def test_merge_tokens_drops_the_duplicated_seam():
    merged, overlap = merge_tokens([1, 2, 3], [3, 4])
    assert merged == [1, 2, 3, 4]
    assert overlap == 1
    merged, overlap = merge_tokens([1, 2], [3, 4])
    assert merged == [1, 2, 3, 4]
    assert overlap == 0


def test_streaming_matches_single_pass_for_context_free_model(extractor, vocabulary, signal):
    model = ConstantModel(token=2)
    decoder = StreamingDecoder(model, extractor, vocabulary, DECODE, FEATURES, SR)
    streamed = decoder.decode_samples(signal)
    offline = decode_offline(model, extractor, vocabulary, signal, blank=0)
    assert list(streamed.tokens) == offline == [2]
    assert streamed.chunk_count > 1
    assert streamed.words == ("timo",)


def test_streaming_deduplicates_the_overlap(extractor, vocabulary, signal):
    model = ConstantModel(token=1)
    decoder = StreamingDecoder(model, extractor, vocabulary, DECODE, FEATURES, SR)
    result = decoder.decode_samples(signal)
    # every chunk re-decoded the same single token; without dedup we would see
    # one token per chunk
    assert list(result.tokens) == [1]
    assert sum(result.chunk_overlaps) >= result.chunk_count - 1
    assert result.transcript() == "kasa"


def test_streaming_appends_genuinely_new_tokens(extractor, vocabulary, signal):
    model = TailModel()
    decoder = StreamingDecoder(model, extractor, vocabulary, DECODE, FEATURES, SR)
    result = decoder.decode_samples(signal)
    assert result.chunk_count >= 2
    # one tail token per chunk, none of them inside the overlapped region
    assert len(result.tokens) == result.chunk_count
    assert result.tokens[0] == 1
    assert result.num_frames > 0


def test_streaming_validation(extractor, vocabulary):
    decoder = StreamingDecoder(ConstantModel(), extractor, vocabulary, DECODE, FEATURES, SR)
    with pytest.raises(DecodeError, match="empty signal"):
        decoder.decode_samples(np.zeros(0, dtype=np.float32))
    with pytest.raises(DecodeError, match="mono"):
        decoder.decode_samples(np.zeros((2, 100), dtype=np.float32))
    with pytest.raises(DecodeError, match="empty chunk"):
        decoder.decode_chunk(np.zeros(0, dtype=np.float32))
