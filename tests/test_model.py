"""The convolutional-recurrent CTC model."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from asr.config import ModelConfig
from asr.model import (
    CHECKPOINT_VERSION,
    ModelError,
    TinyCtcModel,
    build_model,
    count_parameters,
    downsample_lengths,
    load_or_build,
    model_summary,
    validate_batch,
)

INPUT_DIM = 12
VOCAB_SIZE = 5


@pytest.fixture()
def model() -> TinyCtcModel:
    return build_model(ModelConfig(channels=16, hidden=12, dropout=0.0, seed=3), INPUT_DIM, VOCAB_SIZE)


def test_forward_shapes(model):
    features = torch.randn(2, 40, INPUT_DIM)
    log_probs, lengths, state = model.forward(features)
    assert log_probs.shape == (20, 2, VOCAB_SIZE)
    assert lengths.tolist() == [20, 20]
    assert torch.allclose(torch.exp(log_probs).sum(dim=-1), torch.ones(20, 2), atol=1e-5)
    assert state is not None


def test_forward_respects_lengths(model):
    features = torch.randn(3, 32, INPUT_DIM)
    lengths = torch.tensor([32, 20, 9])
    log_probs, out_lengths, _ = model.forward(features, lengths)
    assert out_lengths.tolist() == [16, 10, 5]
