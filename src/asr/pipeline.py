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


@dataclass(frozen=True)
class EvaluationReport:
    """Corpus-level numbers plus the per-utterance detail."""

    evaluations: tuple[UtteranceEvaluation, ...]
    wer: float
    cer: float
    ref_words: int
    errors: int
    model_summary: dict[str, Any]
    history: tuple[float, ...]
    baseline_wer: float | None = None
    seconds: float = 0.0

    @property
    def utterances(self) -> int:
        return len(self.evaluations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterances": self.utterances,
            "wer": self.wer,
            "cer": self.cer,
            "ref_words": self.ref_words,
            "errors": self.errors,
            "baseline_wer": self.baseline_wer,
            "model": self.model_summary,
            "history": list(self.history),
            "seconds": round(self.seconds, 3),
            "utterances_detail": [evaluation.to_dict() for evaluation in self.evaluations],
        }

    def summary_line(self) -> str:
        """One-line headline used by the CLI and the README."""
        baseline = (
            f"  (untrained baseline WER {self.baseline_wer:.3f})"
            if self.baseline_wer is not None
            else ""
        )
        return (
            f"{self.utterances} utterances  WER {self.wer:.3f}  CER {self.cer:.3f}"
            f"  errors {self.errors}/{self.ref_words}{baseline}"
        )


def build_vocabulary(config: PipelineConfig) -> Vocabulary:
    """Vocabulary implied by the synthetic corpus config."""
    return Vocabulary.from_words(config.synth.vocabulary)


def build_extractor(config: PipelineConfig) -> FeatureExtractor:
    """Feature extractor for the configured sample rate."""
    return FeatureExtractor(config.features, config.sample_rate)


def build_detector(config: PipelineConfig) -> VoiceActivityDetector:
    """Voice activity detector for the configured sample rate."""
    return VoiceActivityDetector(config.vad, config.sample_rate)


class SpeechPipeline:
    """A trained model plus the front end, decoder and scorer around it."""

    def __init__(
        self,
        config: PipelineConfig,
        model: TinyCtcModel,
        vocabulary: Vocabulary,
        history: Sequence[float] = (),
        cmvn: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.vocabulary = vocabulary
        self.history = tuple(float(value) for value in history)
        self.cmvn = cmvn
        self.extractor = FeatureExtractor(config.features, config.sample_rate, stats=cmvn)
        self.detector = build_detector(config)
        self.decoder = StreamingDecoder(
            model=model,
            extractor=self.extractor,
            vocabulary=vocabulary,
            decode_config=config.decode,
            feature_config=config.features,
            sample_rate=config.sample_rate,
        )

    @classmethod
    def create(
        cls,
        config: PipelineConfig,
        *,
        force_train: bool = False,
        verbose: bool = False,
    ) -> "SpeechPipeline":
        """Load a checkpoint, or train one when it is missing or ``force_train``."""
        checkpoint = config.checkpoint_path()
        if checkpoint.is_file() and not force_train:
            result = load_training_result(config)
            if result.vocabulary.words != tuple(config.synth.vocabulary):
                raise PipelineError(
                    "checkpoint vocabulary does not match config.synth.vocabulary; "
                    "retrain with force_train=True"
                )
            return cls(config, result.model, result.vocabulary, result.history, result.cmvn)
        trained = train_model(config, verbose=verbose)
        save_training_result(trained, config)
        return cls(config, trained.model, trained.vocabulary, trained.history, trained.cmvn)

    def prepare_audio(self, samples: np.ndarray) -> tuple[np.ndarray, list[SpeechSegment]]:
        """Trim leading/trailing silence with the detector.

        Only the outer silence is removed: the gaps *between* words are kept,
        because CTC needs a blank region to separate two identical words
        ("timo timo"). Concatenating every detected segment would glue those
        words together and cost real accuracy.
        """
        signal = np.asarray(samples)
        segments = self.detector.detect(signal)
        if not segments:
            return signal, segments
        start = segments[0].start_sample
        end = segments[-1].end_sample
        trimmed = signal[start:end]
        if trimmed.size == 0:
            return signal, segments
        return trimmed, segments

    def transcribe(self, samples: np.ndarray) -> DecodeResult:
        """Decode a waveform (VAD-trimmed) in overlapping chunks."""
        audio, _ = self.prepare_audio(samples)
        return self.decoder.decode_samples(audio)

    def transcribe_offline(self, samples: np.ndarray) -> list[int]:
        """Decode a waveform in one pass — used to show what chunking costs."""
        audio, _ = self.prepare_audio(samples)
        return decode_offline(
            self.model, self.extractor, self.vocabulary, audio, blank=self.config.decode.blank_index
        )

    def evaluate(
        self,
        count: int | None = None,
        seed: int | None = None,
        *,
        include_baseline: bool = False,
    ) -> EvaluationReport:
        """Decode a freshly generated corpus and score it against ground truth."""
        started = time.perf_counter()
        utterances = count if count is not None else self.config.train.eval_utterances
        signals = make_corpus(
            self.config.audio,
            self.config.synth,
            utterances,
            seed=self.config.synth.seed + 500 if seed is None else seed,
        )
        evaluations: list[UtteranceEvaluation] = []
        for index, signal in enumerate(signals):
            audio, segments = self.prepare_audio(signal.samples)
            decoded = self.decoder.decode_samples(audio)
            alignment = align(list(signal.words), list(decoded.words))
            evaluations.append(
                UtteranceEvaluation(
                    index=index,
                    reference=signal.words,
                    hypothesis=decoded.words,
                    alignment=alignment,
                    segments=len(segments),
                    speech_ratio=float(
                        sum(segment.num_samples for segment in segments) / signal.num_samples
                    )
                    if signal.num_samples
                    else 0.0,
                    chunk_count=decoded.chunk_count,
                    duration_s=signal.duration_s,
                )
            )
        pairs = [
            (list(evaluation.reference), list(evaluation.hypothesis))
            for evaluation in evaluations
        ]
        pooled = summarize(pairs)
        baseline_wer = None
        if include_baseline:
            baseline_wer = self._baseline_wer(signals)
        return EvaluationReport(
            evaluations=tuple(evaluations),
            wer=pooled["wer"],
            cer=pooled["cer"],
            ref_words=int(pooled["ref_words"]),
            errors=int(pooled["errors"]),
            model_summary=model_summary(self.model),
            history=self.history,
            baseline_wer=baseline_wer,
            seconds=time.perf_counter() - started,
        )

    def _baseline_wer(self, signals: Sequence[Signal]) -> float:
        """WER of an untrained model with the same shape — the honest floor."""
        baseline = build_model(
            self.config.model,
            self.config.features.output_dim,
            self.vocabulary.size,
            seed=self.config.model.seed,
        )
        decoder = StreamingDecoder(
            model=baseline,
            extractor=self.extractor,
            vocabulary=self.vocabulary,
            decode_config=self.config.decode,
            feature_config=self.config.features,
            sample_rate=self.config.sample_rate,
        )
        pairs = []
        for signal in signals:
            audio, _ = self.prepare_audio(signal.samples)
            decoded = decoder.decode_samples(audio)
            pairs.append((list(signal.words), list(decoded.words)))
        return summarize(pairs)["wer"]


def build_parser() -> argparse.ArgumentParser:
    """CLI parser (separate so the tests can inspect it)."""
    parser = argparse.ArgumentParser(
        prog="asr-pipeline",
        description="Streaming CPU ASR over a synthetic speech corpus",
    )
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--train", action="store_true", help="retrain even if a checkpoint exists")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--limit", type=int, default=None, help="number of utterances to evaluate")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--quiet", action="store_true", help="only print the summary line")
    parser.add_argument("--offline", action="store_true", help="also score one-shot decoding")
    return parser


def run_pipeline(args: argparse.Namespace) -> EvaluationReport:
    """Load or train, then evaluate — the whole CLI body in one function."""
    overrides: dict[str, Any] = {}
    if args.epochs is not None:
        overrides["train"] = {"epochs": int(args.epochs)}
    config = load_config(args.config, overrides=overrides or None)
    pipeline = SpeechPipeline.create(config, force_train=args.train, verbose=not args.quiet)
    return pipeline.evaluate(count=args.limit, include_baseline=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_pipeline(args)
    except (ConfigError, PipelineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0
    if not args.quiet:
        print("model:", json.dumps(report.model_summary, sort_keys=True))
        if report.history:
            print(f"training loss: {report.history[0]:.4f} -> {report.history[-1]:.4f}")
        for evaluation in report.evaluations:
            reference = " ".join(evaluation.reference)
            hypothesis = " ".join(evaluation.hypothesis) or "<empty>"
            print(
                f"  [{evaluation.index}] wer {evaluation.wer:.2f}  "
                f"chunks {evaluation.chunk_count}  ref: {reference}  hyp: {hypothesis}"
            )
    print(report.summary_line())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
