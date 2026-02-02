"""Streaming decode: chunk the audio, carry state, stitch the tokens.

Streaming a CTC model is not just "decode 400 ms at a time": the model has no
idea what came before unless you give it context, and the seam between two
chunks is exactly where words get duplicated or dropped. Two mechanisms handle
that here.

* **Overlap.** Consecutive chunks share ``overlap_ms`` of audio, so every word
  near a boundary is seen by two windows. The tokens decoded from the shared
  region are then removed with a longest-suffix/longest-prefix match, which is
  cheap and deterministic.
* **Recurrent state.** The GRU hidden state from one chunk is passed into the
  next, so the model keeps its acoustic context even though the convolutional
  front end restarts at every chunk.

CMVN is a third, quieter trap: normalising each chunk on its own makes the
features drift between chunks. The decoder therefore fixes the mean/variance on
the first chunk and reuses those statistics for the rest of the stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import torch

from asr.config import DecodeConfig, FeatureConfig
from asr.ctc import batch_prefix_beam_decode, collapse_path_with_positions, decode_matrix
from asr.features import FeatureExtractor
from asr.vocab import Vocabulary

EPSILON = 1e-8
