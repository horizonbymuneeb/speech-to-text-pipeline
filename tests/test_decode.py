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
