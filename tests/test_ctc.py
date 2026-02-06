"""CTC loss and decoding, cross-checked against PyTorch's reference loss."""

from __future__ import annotations

import math

import pytest
import torch

from asr.ctc import (
    NEG_INF,
    BeamResult,
    CtcError,
    batch_prefix_beam_decode,
    collapse_path,
    ctc_loss,
    decode_matrix,
    extend_targets,
    greedy_decode,
    greedy_decode_single,
    log_add,
    log_softmax,
    prefix_beam_decode,
    sequence_log_prob,
)

torch.manual_seed(0)
