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

DOWNSAMPLE_FACTOR = 2
CHECKPOINT_VERSION = 1


class ModelError(RuntimeError):
    """Raised when a model or checkpoint is malformed."""


@dataclass(frozen=True)
class ModelSignature:
    """The shape information needed to rebuild a checkpoint."""

    input_dim: int
    vocab_size: int
    channels: int
    hidden: int
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "vocab_size": self.vocab_size,
            "channels": self.channels,
            "hidden": self.hidden,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelSignature":
        return cls(
            input_dim=int(payload["input_dim"]),
            vocab_size=int(payload["vocab_size"]),
            channels=int(payload["channels"]),
            hidden=int(payload["hidden"]),
            dropout=float(payload.get("dropout", 0.0)),
        )


class TinyCtcModel(nn.Module):
    """Conv → GRU → linear CTC model with an explicit recurrent state."""

    def __init__(
        self,
        input_dim: int,
        vocab_size: int,
        channels: int = 48,
        hidden: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ModelError("input_dim must be >= 1")
        if vocab_size < 2:
            raise ModelError("vocab_size must include at least the blank and one token")
        self.signature = ModelSignature(
            input_dim=int(input_dim),
            vocab_size=int(vocab_size),
            channels=int(channels),
            hidden=int(hidden),
            dropout=float(dropout),
        )
        self.conv1 = nn.Conv1d(input_dim, channels, kernel_size=5, stride=DOWNSAMPLE_FACTOR, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.projection = nn.Linear(2 * hidden, vocab_size)

    @property
    def input_dim(self) -> int:
        return self.signature.input_dim

    @property
    def vocab_size(self) -> int:
        return self.signature.vocab_size

    @property
    def downsample_factor(self) -> int:
        return DOWNSAMPLE_FACTOR

    def num_parameters(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def init_weights(self, seed: int | None = None) -> "TinyCtcModel":
        """Deterministic initialisation (tests rely on it for reproducibility)."""
        generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, generator=generator)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter, generator=generator)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(parameter, generator=generator)
                    else:
                        nn.init.zeros_(parameter)
        return self

    def output_length(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Frame counts after the stride-2 convolution."""
        return torch.div(input_lengths + DOWNSAMPLE_FACTOR - 1, DOWNSAMPLE_FACTOR, rounding_mode="floor")

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run the model.

        ``features`` is ``(batch, time, input_dim)``. Returns
        ``(log_probs, output_lengths, new_state)`` where ``log_probs`` is
        ``(time, batch, vocab_size)`` — the layout CTC expects.
        """
        if features.ndim != 3:
            raise ModelError("features must be (batch, time, input_dim)")
        batch, time_steps, dimension = features.shape
        if dimension != self.input_dim:
            raise ModelError(
                f"expected {self.input_dim} features per frame, got {dimension}"
            )
        if time_steps == 0:
            raise ModelError("cannot run the model on zero frames")
        hidden = features.transpose(1, 2)
        hidden = self.activation(self.conv1(hidden))
        hidden = self.activation(self.conv2(hidden))
        hidden = self.dropout(hidden.transpose(1, 2))
        hidden, new_state = self.gru(hidden, state)
        logits = self.projection(hidden)
        log_probs = log_softmax(logits, dim=-1).transpose(0, 1)
        if lengths is None:
            out_lengths = torch.full((batch,), log_probs.shape[0], dtype=torch.long)
        else:
            out_lengths = torch.clamp(self.output_length(lengths), max=log_probs.shape[0])
        return log_probs, out_lengths, new_state

    def predict_tokens(self, features: torch.Tensor, blank: int = 0) -> list[int]:
        """Greedy decode one unbatched ``(time, input_dim)`` feature matrix."""
        if features.ndim != 2:
            raise ModelError("predict_tokens expects (time, input_dim)")
        with torch.no_grad():
            log_probs, _, _ = self.forward(features.unsqueeze(0))
        from asr.ctc import greedy_decode_single

        return greedy_decode_single(log_probs[:, 0, :], blank=blank)

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write a self-describing checkpoint (signature + weights + metadata)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CHECKPOINT_VERSION,
            "signature": self.signature.to_dict(),
            "state_dict": self.state_dict(),
            "metadata": dict(metadata or {}),
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> tuple["TinyCtcModel", dict[str, Any]]:
        """Load a checkpoint; returns the model and its metadata."""
        source = Path(path)
        if not source.is_file():
            raise ModelError(f"checkpoint not found: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "signature" not in payload:
            raise ModelError(f"{source} is not a tracker model checkpoint")
        version = payload.get("version")
        if version != CHECKPOINT_VERSION:
            raise ModelError(f"unsupported checkpoint version {version!r}")
        signature = ModelSignature.from_dict(payload["signature"])
        model = cls(
            input_dim=signature.input_dim,
            vocab_size=signature.vocab_size,
            channels=signature.channels,
            hidden=signature.hidden,
            dropout=signature.dropout,
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, dict(payload.get("metadata") or {})

    def freeze(self) -> "TinyCtcModel":
        """Switch to evaluation mode and disable gradients."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


def build_model(
    config: ModelConfig,
    input_dim: int,
    vocab_size: int,
    seed: int | None = None,
) -> TinyCtcModel:
    """Construct and deterministically initialise a model from config."""
    model = TinyCtcModel(
        input_dim=input_dim,
        vocab_size=vocab_size,
        channels=config.channels,
        hidden=config.hidden,
        dropout=config.dropout,
    )
    return model.init_weights(config.seed if seed is None else seed)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count of any module."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def downsample_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Frame counts after the model's stride-2 convolution."""
    return torch.div(lengths + DOWNSAMPLE_FACTOR - 1, DOWNSAMPLE_FACTOR, rounding_mode="floor")


def model_summary(model: TinyCtcModel) -> dict[str, Any]:
    """Human readable description used by the training script's output."""
    return {
        "input_dim": model.input_dim,
        "vocab_size": model.vocab_size,
        "channels": model.signature.channels,
        "hidden": model.signature.hidden,
        "parameters": model.num_parameters(),
        "downsample_factor": model.downsample_factor,
    }


def load_or_build(
    path: str | Path,
    config: ModelConfig,
    input_dim: int,
    vocab_size: int,
) -> tuple[TinyCtcModel, dict[str, Any], bool]:
    """Load a checkpoint when it matches the requested shape, else build fresh.

    Returns ``(model, metadata, loaded)``. A checkpoint trained against a
    different vocabulary or feature size is rejected rather than silently
    producing garbage logits.
    """
    source = Path(path)
    if not source.is_file():
        return build_model(config, input_dim, vocab_size), {}, False
    model, metadata = TinyCtcModel.load(source)
    if model.input_dim != input_dim or model.vocab_size != vocab_size:
        raise ModelError(
            "checkpoint shape does not match the current features/vocabulary "
            f"(checkpoint {model.input_dim}x{model.vocab_size}, "
            f"wanted {input_dim}x{vocab_size})"
        )
    return model, metadata, True


def validate_batch(features: torch.Tensor, targets: torch.Tensor) -> None:
    """Shared shape checks for the training loop."""
    if features.ndim != 3:
        raise CtcError("features must be (batch, time, dim)")
    if targets.ndim != 2:
        raise CtcError("targets must be (batch, target_length)")
    if features.shape[0] != targets.shape[0]:
        raise CtcError("features and targets disagree on batch size")
