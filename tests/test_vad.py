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

SR = 8000
CONFIG = VadConfig(
    frame_ms=25.0,
    hop_ms=10.0,
    energy_threshold_db=8.0,
    max_zcr=0.35,
    hangover_ms=40.0,
    min_speech_ms=50.0,
    pre_roll_ms=20.0,
)


def make_gapped_signal() -> tuple[np.ndarray, int, int]:
    """0.3 s silence, 0.4 s speech, 0.3 s silence -> known region [2400, 5600)."""
    rng = np.random.default_rng(0)
    audio = AudioConfig(sample_rate=SR, peak_amplitude=0.9, noise_snr_db=60.0)
    synth = SynthConfig(
        seed=1,
        vocabulary=("kasa",),
        words_per_utterance=(1, 1),
        f0_hz=120.0,
        gap_ms=(40.0, 40.0),
    )
    signal = synthesize_utterance(["kasa"], audio, synth, rng)
    speech = signal.samples[:3200]
    padded = np.concatenate([np.zeros(2400, dtype=np.float32), speech, np.zeros(2400, dtype=np.float32)])
    return padded, 2400, 2400 + speech.size


def test_detected_region_matches_known_gap():
    signal, expected_start, expected_end = make_gapped_signal()
    detector = VoiceActivityDetector(CONFIG, SR)
    segments = detector.detect(signal)
    assert segments, "expected one detected speech region"
    assert len(segments) == 1
    segment = segments[0]
    # onset tolerance: the frame grid plus the configured pre-roll; the offset
    # tolerance additionally allows the hangover that keeps word tails alive
    hop = CONFIG.hop_length(SR)
    start_tolerance = CONFIG.frame_length(SR) + hop * CONFIG.pre_roll_frames(SR)
    end_tolerance = CONFIG.frame_length(SR) + hop * (CONFIG.hangover_frames(SR) + 1)
    assert abs(segment.start_sample - expected_start) <= start_tolerance
    assert abs(segment.end_sample - expected_end) <= end_tolerance


def test_silence_only_yields_no_segments(silence):
    detector = VoiceActivityDetector(CONFIG, SR)
    assert detector.detect(silence) == []
    assert detector.speech_ratio(silence) == 0.0


def test_empty_signal_yields_no_segments():
    detector = VoiceActivityDetector(CONFIG, SR)
    assert detector.detect(np.zeros(0, dtype=np.float32)) == []


def test_frame_energy_and_db():
    frames = np.array([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]])
    assert np.allclose(frame_energy(frames), [1.0, 0.25, 0.0])
    assert np.allclose(to_db(np.array([1.0, 0.01])), [0.0, -20.0])
    assert np.all(np.isfinite(to_db(np.zeros(3))))


def test_zero_crossing_rate():
    alternating = np.array([[1.0, -1.0, 1.0, -1.0]])
    constant = np.array([[1.0, 1.0, 1.0, 1.0]])
    assert zero_crossing_rate(alternating)[0] == pytest.approx(1.0)
    assert zero_crossing_rate(constant)[0] == pytest.approx(0.0)
