"""A small convolutional-recurrent CTC acoustic model.

Shape of the network: two 1-D convolutions (time stride 2) over the feature
frames, then a bidirectional GRU, then a linear layer over the vocabulary
including the blank. That is roughly 50k parameters at the default widths — it
trains on the synthetic corpus in seconds on a laptop CPU, which is the point:
the interesting engineering in this repository is the streaming and scoring
code, not the model size.

The GRU hidden state is threaded through ``forward`` so the streaming decoder can
carry context across chunk boundaries. The convolutions keep no state; their
context comes from the decoder's overlap, which is stated in the design notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from asr.ctc import CtcError, log_softmax
from asr.config import ModelConfig
