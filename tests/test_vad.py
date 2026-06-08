"""Voice activity detection: region alignment, hangover and trimming."""

from __future__ import annotations

import numpy as np
import pytest

from asr.config import VadConfig
from asr.synth import AudioConfig, SynthConfig, synthesize_utterance
from asr.vad import (
    SpeechSegment,
    VadError,
    VoiceActivityDetector,
    apply_hangover,
    energy_db,
    frame_decisions,
    frame_energy,
    frames_to_segments,
    merge_segments,
    noise_floor_db,
    speech_ratio,
    to_db,
    trim_to_speech,
    zero_crossing_rate,
)
