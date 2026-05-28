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


def test_forward_rejects_bad_shapes(model):
    with pytest.raises(ModelError, match="batch, time, input_dim"):
        model.forward(torch.randn(40, INPUT_DIM))
    with pytest.raises(ModelError, match="expected 12 features"):
        model.forward(torch.randn(1, 10, INPUT_DIM + 1))
    with pytest.raises(ModelError, match="zero frames"):
        model.forward(torch.zeros(1, 0, INPUT_DIM))


def test_initialisation_is_seeded():
    first = build_model(ModelConfig(channels=16, hidden=12, seed=7), INPUT_DIM, VOCAB_SIZE)
    second = build_model(ModelConfig(channels=16, hidden=12, seed=7), INPUT_DIM, VOCAB_SIZE)
    third = build_model(ModelConfig(channels=16, hidden=12, seed=8), INPUT_DIM, VOCAB_SIZE)
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
    assert any(not torch.equal(a, b) for a, b in zip(first.parameters(), third.parameters()))


def test_parameter_count_and_summary(model):
    assert model.num_parameters() == count_parameters(model) > 1000
    summary = model_summary(model)
    assert summary["input_dim"] == INPUT_DIM
    assert summary["vocab_size"] == VOCAB_SIZE
    assert summary["downsample_factor"] == 2


def test_recurrent_state_is_carried(model):
    first = torch.randn(1, 20, INPUT_DIM)
    _, _, state = model.forward(first)
    assert state is not None
    assert state.shape == (2, 1, 12)  # (directions, batch, hidden)
    second = torch.randn(1, 20, INPUT_DIM)
    log_probs_a, _, state_a = model.forward(second, state=state)
    log_probs_b, _, _ = model.forward(second)
    assert log_probs_a.shape == log_probs_b.shape
    assert not torch.allclose(log_probs_a, log_probs_b)  # state changed the output
    assert state_a is not None


def test_save_and_load_roundtrip(model, tmp_path):
    path = tmp_path / "model.pt"
    model.save(path, {"note": "unit-test"})
    restored, metadata = TinyCtcModel.load(path)
    assert metadata["note"] == "unit-test"
    features = torch.randn(1, 24, INPUT_DIM)
    with torch.no_grad():
        a, _, _ = model.forward(features)
        b, _, _ = restored.forward(features)
    assert torch.allclose(a, b, atol=1e-6)


def test_checkpoint_carries_version_and_signature(model, tmp_path):
    path = tmp_path / "model.pt"
    model.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["version"] == CHECKPOINT_VERSION
    assert payload["signature"]["input_dim"] == INPUT_DIM
    assert payload["signature"]["channels"] == 16


def test_load_errors(tmp_path, model):
    with pytest.raises(ModelError, match="not found"):
        TinyCtcModel.load(tmp_path / "missing.pt")
    bad = tmp_path / "bad.pt"
    torch.save({"version": 1}, bad)
    with pytest.raises(ModelError, match="checkpoint"):
        TinyCtcModel.load(bad)
    wrong_version = tmp_path / "old.pt"
    model.save(wrong_version)
    payload = torch.load(wrong_version, map_location="cpu", weights_only=False)
    payload["version"] = 99
    torch.save(payload, wrong_version)
    with pytest.raises(ModelError, match="version"):
        TinyCtcModel.load(wrong_version)


def test_load_or_build_behaviour(tmp_path, model):
    config = ModelConfig(channels=16, hidden=12, seed=3)
    fresh, metadata, loaded = load_or_build(tmp_path / "nope.pt", config, INPUT_DIM, VOCAB_SIZE)
    assert loaded is False and metadata == {}
    assert fresh.input_dim == INPUT_DIM
    path = tmp_path / "saved.pt"
    model.save(path)
    restored, _, loaded = load_or_build(path, config, INPUT_DIM, VOCAB_SIZE)
    assert loaded is True
    with pytest.raises(ModelError, match="does not match"):
        load_or_build(path, config, INPUT_DIM + 2, VOCAB_SIZE)


def test_predict_tokens_returns_collapsed_tokens(model):
    features = torch.randn(30, INPUT_DIM)
    tokens = model.predict_tokens(features)
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) and 0 <= token < VOCAB_SIZE for token in tokens)
    with pytest.raises(ModelError, match="time, input_dim"):
        model.predict_tokens(torch.randn(1, 30, INPUT_DIM))
