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
