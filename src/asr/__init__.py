"""Streaming CPU speech recognition over a synthetic corpus.

Public entry points::

    from asr import load_config, SpeechPipeline

    config = load_config("config.yaml")
    pipeline = SpeechPipeline.create(config)
    report = pipeline.evaluate(count=5)
    print(report.summary_line())
"""

from asr.config import (
    AudioConfig,
    ConfigError,
    DecodeConfig,
    FeatureConfig,
    ModelConfig,
    PipelineConfig,
    SynthConfig,
    TrainConfig,
    VadConfig,
    load_config,
    ms_to_frames,
    ms_to_samples,
)
from asr.ctc import BeamResult, ctc_loss, greedy_decode, prefix_beam_decode
from asr.decode import ChunkPlan, DecodeResult, StreamingDecoder, merge_tokens, plan_chunks
from asr.dsp import mel_filterbank, power_spectrum, pre_emphasis
from asr.features import FeatureExtractor, extract_features, log_mel_spectrogram
from asr.metrics import Alignment, EditOp, align, char_error_rate, edit_distance, summarize, word_error_rate
from asr.model import TinyCtcModel, build_model, count_parameters, model_summary
from asr.pipeline import EvaluationReport, SpeechPipeline, UtteranceEvaluation
from asr.synth import Signal, make_corpus, synthesize_utterance, synthesize_word
from asr.train import TrainingResult, load_training_result, train_model
from asr.vad import SpeechSegment, VoiceActivityDetector
from asr.vocab import Vocabulary

__all__ = [
    "Alignment",
    "AudioConfig",
    "BeamResult",
    "ChunkPlan",
    "ConfigError",
    "DecodeConfig",
    "DecodeResult",
    "EditOp",
    "EvaluationReport",
    "FeatureConfig",
    "FeatureExtractor",
    "ModelConfig",
    "PipelineConfig",
    "Signal",
    "SpeechPipeline",
    "SpeechSegment",
    "StreamingDecoder",
    "SynthConfig",
    "TinyCtcModel",
    "TrainConfig",
    "TrainingResult",
    "UtteranceEvaluation",
    "VadConfig",
    "VoiceActivityDetector",
    "Vocabulary",
    "align",
    "build_model",
    "char_error_rate",
    "count_parameters",
    "ctc_loss",
    "edit_distance",
    "extract_features",
    "greedy_decode",
    "load_config",
    "load_training_result",
    "log_mel_spectrogram",
    "make_corpus",
    "mel_filterbank",
    "merge_tokens",
    "model_summary",
    "ms_to_frames",
    "ms_to_samples",
    "plan_chunks",
    "power_spectrum",
    "pre_emphasis",
    "prefix_beam_decode",
    "summarize",
    "synthesize_utterance",
    "synthesize_word",
    "train_model",
    "word_error_rate",
]
