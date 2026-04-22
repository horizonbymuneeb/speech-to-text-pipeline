"""Training the tiny CTC model on the synthetic corpus.

The loop is deliberately boring: seeded dataset generation, padded batches, CTC
loss, Adam, a handful of epochs. What matters for reproducibility is that every
random choice (corpus sampling, batch order, weight init) is driven by a seed
from the config, so ``train_model`` twice with the same config produces the same
loss curve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from asr.config import PipelineConfig
from asr.ctc import ctc_loss
from asr.features import FeatureExtractor
from asr.model import TinyCtcModel, build_model, count_parameters, model_summary
from asr.synth import Signal, make_corpus
from asr.vocab import Vocabulary


class TrainingError(RuntimeError):
    """Raised when training cannot proceed."""


@dataclass(frozen=True)
class Sample:
    """One training example: features, target tokens and the words behind them."""

    features: np.ndarray
    tokens: tuple[int, ...]
    words: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise TrainingError("features must be (frames, dim)")
        if len(self.tokens) != len(self.words):
            raise TrainingError("tokens and words must line up")
        if not self.tokens:
            raise TrainingError("a sample needs at least one target token")


@dataclass(frozen=True)
class Batch:
    """A padded batch ready for the model and the CTC loss."""

    features: torch.Tensor
    feature_lengths: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    words: tuple[tuple[str, ...], ...]

    @property
    def size(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class TrainingResult:
    """Everything the pipeline needs after a training run."""

    model: TinyCtcModel
    vocabulary: Vocabulary
    history: tuple[float, ...]
    train_samples: int
    eval_samples: int
    parameters: int
    seconds: float
    cmvn: tuple[np.ndarray, np.ndarray] | None = None

    def cmvn_metadata(self) -> dict[str, list[float]] | None:
        """Global CMVN statistics in JSON-friendly form."""
        if self.cmvn is None:
            return None
        mean, std = self.cmvn
        return {"mean": [float(v) for v in mean], "std": [float(v) for v in std]}

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters,
            "train_samples": self.train_samples,
            "eval_samples": self.eval_samples,
            "epochs": len(self.history),
            "history": list(self.history),
            "final_loss": self.history[-1] if self.history else None,
            "seconds": round(self.seconds, 3),
        }
