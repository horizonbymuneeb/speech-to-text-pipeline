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
