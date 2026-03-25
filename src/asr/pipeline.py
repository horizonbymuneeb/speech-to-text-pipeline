"""End-to-end pipeline and command line entry point.

Wiring, in order: generate or accept a waveform, run the energy/ZCR detector to
drop silence, extract log-mel features, decode in overlapping chunks with the
recurrent state carried across them, then score the words against the ground
truth the generator produced. The CLI runs the same code path the tests do, so
``python -m asr.pipeline`` and ``pytest`` cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from asr.config import ConfigError, PipelineConfig, load_config
from asr.decode import DecodeResult, StreamingDecoder, decode_offline
from asr.features import FeatureExtractor
from asr.metrics import Alignment, align, summarize
from asr.model import TinyCtcModel, build_model, model_summary
from asr.synth import Signal, make_corpus
from asr.train import TrainingResult, load_training_result, save_training_result, train_model
from asr.vad import SpeechSegment, VoiceActivityDetector, trim_to_speech
from asr.vocab import Vocabulary


class PipelineError(RuntimeError):
    """Raised when the pipeline is used without a usable model."""


@dataclass(frozen=True)
class UtteranceEvaluation:
    """One decoded utterance compared against its ground truth."""

    index: int
    reference: tuple[str, ...]
    hypothesis: tuple[str, ...]
    alignment: Alignment
    segments: int
    speech_ratio: float
    chunk_count: int
    duration_s: float

    @property
    def wer(self) -> float:
        return self.alignment.rate

    @property
    def errors(self) -> int:
        return self.alignment.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "reference": list(self.reference),
            "hypothesis": list(self.hypothesis),
            "wer": self.wer,
            "errors": self.alignment.counts(),
            "segments": self.segments,
            "speech_ratio": self.speech_ratio,
            "chunks": self.chunk_count,
            "duration_s": self.duration_s,
        }
