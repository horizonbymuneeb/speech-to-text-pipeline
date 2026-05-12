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


def signals_to_samples(
    signals: Sequence[Signal], extractor: FeatureExtractor, vocabulary: Vocabulary
) -> list[Sample]:
    """Turn synthetic utterances into feature/target pairs."""
    samples: list[Sample] = []
    for signal in signals:
        features = extractor.transform(signal.samples)
        samples.append(
            Sample(
                features=features,
                tokens=tuple(signal.tokens(vocabulary)),
                words=signal.words,
            )
        )
    return samples


def make_samples(
    config: PipelineConfig,
    extractor: FeatureExtractor,
    vocabulary: Vocabulary,
    count: int,
    seed: int | None = None,
) -> list[Sample]:
    """Generate ``count`` utterances and featurise them."""
    signals = make_corpus(config.audio, config.synth, count, seed=seed)
    return signals_to_samples(signals, extractor, vocabulary)


def collate(samples: Sequence[Sample]) -> Batch:
    """Pad a list of samples into one batch (features with zeros, targets with blank)."""
    if not samples:
        raise TrainingError("cannot collate an empty batch")
    max_frames = max(sample.features.shape[0] for sample in samples)
    max_targets = max(len(sample.tokens) for sample in samples)
    dim = samples[0].features.shape[1]
    features = torch.zeros((len(samples), max_frames, dim), dtype=torch.float32)
    feature_lengths = torch.zeros(len(samples), dtype=torch.long)
    targets = torch.zeros((len(samples), max_targets), dtype=torch.long)
    target_lengths = torch.zeros(len(samples), dtype=torch.long)
    for index, sample in enumerate(samples):
        frames = sample.features.shape[0]
        if sample.features.shape[1] != dim:
            raise TrainingError("samples in a batch must share the feature dimension")
        features[index, :frames] = torch.from_numpy(sample.features)
        feature_lengths[index] = frames
        targets[index, : len(sample.tokens)] = torch.tensor(sample.tokens, dtype=torch.long)
        target_lengths[index] = len(sample.tokens)
    return Batch(
        features=features,
        feature_lengths=feature_lengths,
        targets=targets,
        target_lengths=target_lengths,
        words=tuple(sample.words for sample in samples),
    )


def iterate_batches(
    samples: Sequence[Sample], batch_size: int, rng: np.random.Generator, *, shuffle: bool = True
) -> list[Batch]:
    """Split samples into batches, optionally shuffled with a seeded generator."""
    if batch_size < 1:
        raise TrainingError("batch_size must be >= 1")
    order = rng.permutation(len(samples)) if shuffle else np.arange(len(samples))
    batches: list[Batch] = []
    for start in range(0, len(order), batch_size):
        chunk = [samples[int(index)] for index in order[start : start + batch_size]]
        batches.append(collate(chunk))
    return batches


def batch_loss(model: TinyCtcModel, batch: Batch, blank: int = 0) -> torch.Tensor:
    """CTC loss for one batch."""
    log_probs, out_lengths, _ = model.forward(batch.features, batch.feature_lengths)
    return ctc_loss(
        log_probs,
        batch.targets,
        out_lengths,
        batch.target_lengths,
        blank=blank,
        reduction="mean",
    )


def train_epoch(
    model: TinyCtcModel, batches: Sequence[Batch], optimizer: torch.optim.Optimizer, blank: int = 0
) -> dict[str, float]:
    """One pass over ``batches``; returns the mean loss and batch count."""
    if not batches:
        raise TrainingError("no batches to train on")
    model.train()
    total = 0.0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, blank)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += float(loss.detach())
    return {"loss": total / len(batches), "batches": float(len(batches))}


def evaluate_loss(
    model: TinyCtcModel, batches: Sequence[Batch], blank: int = 0
) -> dict[str, float]:
    """Mean CTC loss over ``batches`` without touching gradients."""
    if not batches:
        raise TrainingError("no batches to evaluate on")
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in batches:
            total += float(batch_loss(model, batch, blank))
    return {"loss": total / len(batches), "batches": float(len(batches))}
