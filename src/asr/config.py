"""Configuration objects for the speech pipeline.

``config.yaml`` holds the tunables. Everything has a default here so the YAML
only needs the keys you actually want to change, and every section is validated
on load: a typo in ``config.yaml`` fails loudly instead of silently running with
the default.

Frame-based settings are declared in milliseconds and converted to sample /
frame counts through methods that take the sample rate, because the same config
should work at 8 kHz and 16 kHz without editing numbers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a config mapping is malformed."""


DEFAULT_CONFIG: dict[str, Any] = {
    "audio": {"sample_rate": 8000, "peak_amplitude": 0.9, "noise_snr_db": 25.0},
    "synth": {
        "seed": 7,
        "vocabulary": ["kasa", "timo", "velu", "nara", "piso", "duma", "seko", "bora"],
        "words_per_utterance": [3, 5],
        "f0_hz": 120.0,
        "gap_ms": [40, 90],
    },
    "vad": {
        "frame_ms": 25.0,
        "hop_ms": 10.0,
        "energy_threshold_db": 8.0,
        "max_zcr": 0.35,
        "hangover_ms": 60.0,
        "min_speech_ms": 60.0,
        "pre_roll_ms": 30.0,
    },
    "features": {
        "frame_ms": 25.0,
        "hop_ms": 10.0,
        "n_fft": 256,
        "n_mels": 40,
        "fmin_hz": 60.0,
        "fmax_hz": 3800.0,
        "pre_emphasis": 0.97,
        "use_deltas": True,
        "cmvn": True,
    },
    "model": {"channels": 48, "hidden": 64, "dropout": 0.0, "seed": 13},
    "decode": {"chunk_ms": 400.0, "overlap_ms": 120.0, "beam_width": 1, "blank_index": 0},
    "train": {
        "epochs": 12,
        "batch_size": 4,
        "learning_rate": 0.01,
        "seed": 21,
        "checkpoint": "checkpoints/ctc.pt",
        "train_utterances": 24,
        "eval_utterances": 6,
    },
}

_SECTIONS: dict[str, dict[str, type]] = {
    "audio": {"sample_rate": int, "peak_amplitude": float, "noise_snr_db": float},
    "synth": {
        "seed": int,
        "vocabulary": list,
        "words_per_utterance": list,
        "f0_hz": float,
        "gap_ms": list,
    },
    "vad": {
        "frame_ms": float,
        "hop_ms": float,
        "energy_threshold_db": float,
        "max_zcr": float,
        "hangover_ms": float,
        "min_speech_ms": float,
        "pre_roll_ms": float,
    },
    "features": {
        "frame_ms": float,
        "hop_ms": float,
        "n_fft": int,
        "n_mels": int,
        "fmin_hz": float,
        "fmax_hz": float,
        "pre_emphasis": float,
        "use_deltas": bool,
        "cmvn": bool,
    },
    "model": {"channels": int, "hidden": int, "dropout": float, "seed": int},
    "decode": {"chunk_ms": float, "overlap_ms": float, "beam_width": int, "blank_index": int},
    "train": {
        "epochs": int,
        "batch_size": int,
        "learning_rate": float,
        "seed": int,
        "checkpoint": str,
        "train_utterances": int,
        "eval_utterances": int,
    },
}


def ms_to_samples(milliseconds: float, sample_rate: int) -> int:
    """Convert a duration in milliseconds to a whole number of samples."""
    if sample_rate <= 0:
        raise ConfigError("sample_rate must be positive")
    return int(round(milliseconds * sample_rate / 1000.0))


def ms_to_frames(milliseconds: float, hop_length: int, sample_rate: int) -> int:
    """Convert a duration in milliseconds to a number of hop frames."""
    hop_samples = ms_to_samples(milliseconds, sample_rate)
    return max(1, int(round(hop_samples / max(1, hop_length))))


@dataclass(frozen=True)
class AudioConfig:
    """Sample rate and the amplitude budget of the synthetic corpus."""

    sample_rate: int = 8000
    peak_amplitude: float = 0.9
    noise_snr_db: float = 25.0

    def __post_init__(self) -> None:
        if self.sample_rate < 4000:
            raise ConfigError("audio.sample_rate must be >= 4000")
        if not 0.0 < self.peak_amplitude <= 1.0:
            raise ConfigError("audio.peak_amplitude must be in (0, 1]")
        if self.noise_snr_db <= 0:
            raise ConfigError("audio.noise_snr_db must be positive")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AudioConfig":
        return cls(
            sample_rate=int(mapping["sample_rate"]),
            peak_amplitude=float(mapping["peak_amplitude"]),
            noise_snr_db=float(mapping["noise_snr_db"]),
        )


@dataclass(frozen=True)
class SynthConfig:
    """How the synthetic utterances are generated."""

    seed: int = 7
    vocabulary: tuple[str, ...] = (
        "kasa",
        "timo",
        "velu",
        "nara",
        "piso",
        "duma",
        "seko",
        "bora",
    )
    words_per_utterance: tuple[int, int] = (3, 5)
    f0_hz: float = 120.0
    gap_ms: tuple[float, float] = (40.0, 90.0)

    def __post_init__(self) -> None:
        if not self.vocabulary:
            raise ConfigError("synth.vocabulary must not be empty")
        if len(set(self.vocabulary)) != len(self.vocabulary):
            raise ConfigError("synth.vocabulary contains duplicates")
        for word in self.vocabulary:
            if not word or not word.isalpha() or word != word.lower():
                raise ConfigError(f"vocabulary word {word!r} must be lowercase letters")
        low, high = self.words_per_utterance
        if low < 1 or high < low:
            raise ConfigError("synth.words_per_utterance must be [low, high] with 1 <= low <= high")
        if self.f0_hz < 60 or self.f0_hz > 400:
            raise ConfigError("synth.f0_hz must be between 60 and 400")
        gap_low, gap_high = self.gap_ms
        if gap_low < 0 or gap_high < gap_low:
            raise ConfigError("synth.gap_ms must be [low, high] with 0 <= low <= high")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "SynthConfig":
        return cls(
            seed=int(mapping["seed"]),
            vocabulary=tuple(str(word) for word in mapping["vocabulary"]),
            words_per_utterance=tuple(int(v) for v in mapping["words_per_utterance"]),  # type: ignore[arg-type]
            f0_hz=float(mapping["f0_hz"]),
            gap_ms=tuple(float(v) for v in mapping["gap_ms"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class VadConfig:
    """Short-time energy / zero-crossing voice activity detection settings."""

    frame_ms: float = 25.0
    hop_ms: float = 10.0
    energy_threshold_db: float = 8.0
    max_zcr: float = 0.35
    hangover_ms: float = 60.0
    min_speech_ms: float = 60.0
    pre_roll_ms: float = 30.0

    def __post_init__(self) -> None:
        if self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ConfigError("vad frame_ms and hop_ms must be positive")
        if self.hop_ms > self.frame_ms:
            raise ConfigError("vad.hop_ms must not exceed vad.frame_ms")
        if not 0.0 <= self.max_zcr <= 1.0:
            raise ConfigError("vad.max_zcr must be in [0, 1]")
        if self.min_speech_ms < 0 or self.hangover_ms < 0 or self.pre_roll_ms < 0:
            raise ConfigError("vad durations must be non-negative")

    def frame_length(self, sample_rate: int) -> int:
        """Analysis window length in samples."""
        return ms_to_samples(self.frame_ms, sample_rate)

    def hop_length(self, sample_rate: int) -> int:
        """Hop between analysis windows in samples."""
        return ms_to_samples(self.hop_ms, sample_rate)

    def hangover_frames(self, sample_rate: int) -> int:
        """How many frames a speech decision survives after energy drops."""
        return ms_to_frames(self.hangover_ms, self.hop_length(sample_rate), sample_rate)

    def min_speech_frames(self, sample_rate: int) -> int:
        """Segments shorter than this are discarded."""
        return ms_to_frames(self.min_speech_ms, self.hop_length(sample_rate), sample_rate)

    def pre_roll_frames(self, sample_rate: int) -> int:
        """Frames of context kept before a detected onset."""
        return ms_to_frames(self.pre_roll_ms, self.hop_length(sample_rate), sample_rate)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "VadConfig":
        return cls(
            frame_ms=float(mapping["frame_ms"]),
            hop_ms=float(mapping["hop_ms"]),
            energy_threshold_db=float(mapping["energy_threshold_db"]),
            max_zcr=float(mapping["max_zcr"]),
            hangover_ms=float(mapping["hangover_ms"]),
            min_speech_ms=float(mapping["min_speech_ms"]),
            pre_roll_ms=float(mapping["pre_roll_ms"]),
        )


@dataclass(frozen=True)
class FeatureConfig:
    """Log-mel front end settings."""

    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 256
    n_mels: int = 40
    fmin_hz: float = 60.0
    fmax_hz: float = 3800.0
    pre_emphasis: float = 0.97
    use_deltas: bool = True
    cmvn: bool = True

    def __post_init__(self) -> None:
        if self.frame_ms <= 0 or self.hop_ms <= 0:
            raise ConfigError("features frame_ms and hop_ms must be positive")
        if self.hop_ms > self.frame_ms:
            raise ConfigError("features.hop_ms must not exceed features.frame_ms")
        if self.n_fft < 16 or self.n_fft & (self.n_fft - 1):
            raise ConfigError("features.n_fft must be a power of two >= 16")
        if self.n_mels < 1 or self.n_mels > self.n_fft // 2:
            raise ConfigError("features.n_mels must be between 1 and n_fft/2")
        if not 0.0 <= self.pre_emphasis < 1.0:
            raise ConfigError("features.pre_emphasis must be in [0, 1)")
        if self.fmin_hz < 0 or self.fmax_hz <= self.fmin_hz:
            raise ConfigError("features fmin_hz/fmax_hz must satisfy 0 <= fmin < fmax")

    def frame_length(self, sample_rate: int) -> int:
        """Analysis window length in samples."""
        return ms_to_samples(self.frame_ms, sample_rate)

    def hop_length(self, sample_rate: int) -> int:
        """Hop between windows in samples."""
        return ms_to_samples(self.hop_ms, sample_rate)

    @property
    def n_freq_bins(self) -> int:
        """Number of rfft bins kept by the power spectrum."""
        return self.n_fft // 2 + 1

    @property
    def output_dim(self) -> int:
        """Feature vector size after optional deltas."""
        return self.n_mels * 3 if self.use_deltas else self.n_mels

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FeatureConfig":
        return cls(
            frame_ms=float(mapping["frame_ms"]),
            hop_ms=float(mapping["hop_ms"]),
            n_fft=int(mapping["n_fft"]),
            n_mels=int(mapping["n_mels"]),
            fmin_hz=float(mapping["fmin_hz"]),
            fmax_hz=float(mapping["fmax_hz"]),
            pre_emphasis=float(mapping["pre_emphasis"]),
            use_deltas=bool(mapping["use_deltas"]),
            cmvn=bool(mapping["cmvn"]),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Acoustic model width."""

    channels: int = 48
    hidden: int = 64
    dropout: float = 0.0
    seed: int = 13

    def __post_init__(self) -> None:
        if self.channels < 8:
            raise ConfigError("model.channels must be >= 8")
        if self.hidden < 8:
            raise ConfigError("model.hidden must be >= 8")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1)")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ModelConfig":
        return cls(
            channels=int(mapping["channels"]),
            hidden=int(mapping["hidden"]),
            dropout=float(mapping["dropout"]),
            seed=int(mapping["seed"]),
        )
