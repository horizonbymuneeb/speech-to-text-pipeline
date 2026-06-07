"""Training loop, checkpointing and the end-to-end pipeline."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
import yaml

from asr.config import load_config
from asr.features import FeatureExtractor
from asr.pipeline import SpeechPipeline, build_parser, main
from asr.train import (
    Sample,
    TrainingError,
    collate,
    evaluate_loss,
    iterate_batches,
    load_training_result,
    make_samples,
    save_training_result,
    train_epoch,
    train_model,
)
from asr.vocab import Vocabulary


@pytest.fixture()
def prepared(small_config):
    """A tiny corpus already featurised, shared by the training tests."""
    vocabulary = Vocabulary.from_words(small_config.synth.vocabulary)
    extractor = FeatureExtractor(small_config.features, small_config.sample_rate)
    samples = make_samples(small_config, extractor, vocabulary, 8, seed=1)
    return small_config, vocabulary, extractor, samples


def test_collate_pads_and_tracks_lengths(prepared):
    _, _, _, samples = prepared
    batch = collate(samples[:3])
    assert batch.size == 3
    assert batch.features.shape[0] == 3
    assert batch.features.shape[2] == samples[0].features.shape[1]
    assert int(batch.feature_lengths.max()) == batch.features.shape[1]
    assert int(batch.target_lengths.max()) == batch.targets.shape[1]
    assert all(int(length) == len(words) for length, words in zip(batch.target_lengths, batch.words))


def test_collate_validation(prepared):
    with pytest.raises(TrainingError, match="empty batch"):
        collate([])
    _, _, _, samples = prepared
    bad = Sample(features=np.zeros((5, 3)), tokens=(1,), words=("kasa",))
    with pytest.raises(TrainingError, match="feature dimension"):
        collate([samples[0], bad])


def test_iterate_batches_is_seeded(prepared):
    _, _, _, samples = prepared
    first = iterate_batches(samples, 3, np.random.default_rng(0))
    second = iterate_batches(samples, 3, np.random.default_rng(0))
    assert [batch.size for batch in first] == [batch.size for batch in second]
    assert [batch.words for batch in first] == [batch.words for batch in second]
    unshuffled = iterate_batches(samples, 4, np.random.default_rng(0), shuffle=False)
    assert [words for batch in unshuffled for words in batch.words] == [s.words for s in samples]


def test_iterate_batches_validation(prepared):
    _, _, _, samples = prepared
    with pytest.raises(TrainingError, match="batch_size"):
        iterate_batches(samples, 0, np.random.default_rng(0))


def test_train_epoch_reduces_loss(prepared):
    config, vocabulary, _, samples = prepared
    from asr.model import build_model

    model = build_model(config.model, config.features.output_dim, vocabulary.size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    batches = iterate_batches(samples, config.train.batch_size, np.random.default_rng(0))
    first = train_epoch(model, batches, optimizer)["loss"]
    for _ in range(3):
        latest = train_epoch(model, batches, optimizer)["loss"]
    assert latest < first
    assert evaluate_loss(model, batches)["loss"] > 0


def test_train_epoch_validation(prepared):
    config, vocabulary, _, _ = prepared
    from asr.model import build_model

    model = build_model(config.model, config.features.output_dim, vocabulary.size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with pytest.raises(TrainingError, match="no batches"):
        train_epoch(model, [], optimizer)
    with pytest.raises(TrainingError, match="no batches"):
        evaluate_loss(model, [])


def test_train_model_history_and_checkpoint(small_config):
    result = train_model(small_config)
    assert len(result.history) == small_config.train.epochs
    assert result.history[-1] < result.history[0]
    assert result.parameters > 0
    assert result.cmvn is not None
    payload = result.to_dict()
    assert payload["epochs"] == small_config.train.epochs
    path = save_training_result(result, small_config)
    assert path.exists()
    restored = load_training_result(small_config)
    assert restored.vocabulary.words == result.vocabulary.words
    assert restored.history == result.history
    assert restored.cmvn is not None
    assert np.allclose(restored.cmvn[0], result.cmvn[0])


def test_training_is_reproducible(small_config):
    first = train_model(small_config)
    second = train_model(small_config)
    assert first.history == second.history
