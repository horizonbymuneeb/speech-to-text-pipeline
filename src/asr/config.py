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


@dataclass(frozen=True)
class DecodeConfig:
    """Streaming chunking and search settings."""

    chunk_ms: float = 400.0
    overlap_ms: float = 120.0
    beam_width: int = 1
    blank_index: int = 0

    def __post_init__(self) -> None:
        if self.chunk_ms <= 0:
            raise ConfigError("decode.chunk_ms must be positive")
        if self.overlap_ms < 0:
            raise ConfigError("decode.overlap_ms must be non-negative")
        if self.overlap_ms >= self.chunk_ms:
            raise ConfigError("decode.overlap_ms must be smaller than decode.chunk_ms")
        if self.beam_width < 1:
            raise ConfigError("decode.beam_width must be >= 1")
        if self.blank_index < 0:
            raise ConfigError("decode.blank_index must be >= 0")

    def chunk_samples(self, sample_rate: int) -> int:
        return ms_to_samples(self.chunk_ms, sample_rate)

    def overlap_samples(self, sample_rate: int) -> int:
        return ms_to_samples(self.overlap_ms, sample_rate)

    def hop_samples(self, sample_rate: int) -> int:
        return self.chunk_samples(sample_rate) - self.overlap_samples(sample_rate)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "DecodeConfig":
        return cls(
            chunk_ms=float(mapping["chunk_ms"]),
            overlap_ms=float(mapping["overlap_ms"]),
            beam_width=int(mapping["beam_width"]),
            blank_index=int(mapping["blank_index"]),
        )


@dataclass(frozen=True)
class TrainConfig:
    """Training loop settings."""

    epochs: int = 12
    batch_size: int = 4
    learning_rate: float = 0.01
    seed: int = 21
    checkpoint: str = "checkpoints/ctc.pt"
    train_utterances: int = 24
    eval_utterances: int = 6

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigError("train.epochs must be >= 1")
        if self.batch_size < 1:
            raise ConfigError("train.batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigError("train.learning_rate must be positive")
        if self.train_utterances < 1 or self.eval_utterances < 1:
            raise ConfigError("train utterance counts must be >= 1")

    def checkpoint_path(self, base_dir: Path) -> Path:
        """Absolute checkpoint path, resolved against the config directory."""
        candidate = Path(self.checkpoint).expanduser()
        return candidate if candidate.is_absolute() else (Path(base_dir) / candidate).resolve()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrainConfig":
        return cls(
            epochs=int(mapping["epochs"]),
            batch_size=int(mapping["batch_size"]),
            learning_rate=float(mapping["learning_rate"]),
            seed=int(mapping["seed"]),
            checkpoint=str(mapping["checkpoint"]),
            train_utterances=int(mapping["train_utterances"]),
            eval_utterances=int(mapping["eval_utterances"]),
        )


@dataclass(frozen=True)
class PipelineConfig:
    """The whole configuration, with paths resolved against ``base_dir``."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    base_dir: Path = field(default_factory=lambda: Path.cwd())

    @property
    def sample_rate(self) -> int:
        """Shorthand for the audio sample rate."""
        return self.audio.sample_rate

    def checkpoint_path(self) -> Path:
        """Absolute path of the model checkpoint."""
        return self.train.checkpoint_path(self.base_dir)

    def feature_dim(self) -> int:
        """Model input size implied by the feature config."""
        return self.features.output_dim

    def to_dict(self) -> dict[str, Any]:
        """Plain mapping (config + resolved base directory)."""
        return {
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "peak_amplitude": self.audio.peak_amplitude,
                "noise_snr_db": self.audio.noise_snr_db,
            },
            "synth": {
                "seed": self.synth.seed,
                "vocabulary": list(self.synth.vocabulary),
                "words_per_utterance": list(self.synth.words_per_utterance),
                "f0_hz": self.synth.f0_hz,
                "gap_ms": list(self.synth.gap_ms),
            },
            "vad": {
                "frame_ms": self.vad.frame_ms,
                "hop_ms": self.vad.hop_ms,
                "energy_threshold_db": self.vad.energy_threshold_db,
                "max_zcr": self.vad.max_zcr,
                "hangover_ms": self.vad.hangover_ms,
                "min_speech_ms": self.vad.min_speech_ms,
                "pre_roll_ms": self.vad.pre_roll_ms,
            },
            "features": {
                "frame_ms": self.features.frame_ms,
                "hop_ms": self.features.hop_ms,
                "n_fft": self.features.n_fft,
                "n_mels": self.features.n_mels,
                "fmin_hz": self.features.fmin_hz,
                "fmax_hz": self.features.fmax_hz,
                "pre_emphasis": self.features.pre_emphasis,
                "use_deltas": self.features.use_deltas,
                "cmvn": self.features.cmvn,
            },
            "model": {
                "channels": self.model.channels,
                "hidden": self.model.hidden,
                "dropout": self.model.dropout,
                "seed": self.model.seed,
            },
            "decode": {
                "chunk_ms": self.decode.chunk_ms,
                "overlap_ms": self.decode.overlap_ms,
                "beam_width": self.decode.beam_width,
                "blank_index": self.decode.blank_index,
            },
            "train": {
                "epochs": self.train.epochs,
                "batch_size": self.train.batch_size,
                "learning_rate": self.train.learning_rate,
                "seed": self.train.seed,
                "checkpoint": self.train.checkpoint,
                "train_utterances": self.train.train_utterances,
                "eval_utterances": self.train.eval_utterances,
            },
            "base_dir": str(self.base_dir),
        }

    def with_overrides(self, **overrides: Any) -> "PipelineConfig":
        """Return a copy with dataclass fields replaced (``replace`` sugar)."""
        return replace(self, **overrides)

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any], base_dir: Path | str | None = None
    ) -> "PipelineConfig":
        merged = deep_merge(DEFAULT_CONFIG, mapping)
        validate_config(merged)
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        return cls(
            audio=AudioConfig.from_mapping(merged["audio"]),
            synth=SynthConfig.from_mapping(merged["synth"]),
            vad=VadConfig.from_mapping(merged["vad"]),
            features=FeatureConfig.from_mapping(merged["features"]),
            model=ModelConfig.from_mapping(merged["model"]),
            decode=DecodeConfig.from_mapping(merged["decode"]),
            train=TrainConfig.from_mapping(merged["train"]),
            base_dir=root,
        )
