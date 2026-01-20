"""Shared fixtures for the speech pipeline test suite."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from asr.config import load_config


@pytest.fixture(scope="session")
def config():
    """The repository's config.yaml (paths resolve next to it)."""
    return load_config("config.yaml")


@pytest.fixture()
def small_config(config, tmp_path):
    """A deliberately tiny config for training-mechanics tests (a few seconds)."""
    return replace(
        config,
        model=replace(config.model, channels=24, hidden=32),
        features=replace(config.features, n_mels=24),
        train=replace(
            config.train,
            epochs=3,
            batch_size=3,
            train_utterances=6,
            eval_utterances=2,
            checkpoint=str(tmp_path / "checkpoints" / "ctc.pt"),
        ),
        base_dir=tmp_path,
    )


@pytest.fixture(scope="session")
def learn_config(config, tmp_path_factory):
    """The shipped training recipe (same widths, LR and epoch count as config.yaml).

    Training this once per session costs about ten seconds and is what the
    end-to-end quality test needs; running it per test would blow the suite
    budget.
    """
    root = tmp_path_factory.mktemp("learning")
    return replace(
        config,
        train=replace(
            config.train,
            train_utterances=20,
            eval_utterances=4,
            checkpoint=str(root / "checkpoints" / "ctc.pt"),
        ),
        base_dir=root,
    )
