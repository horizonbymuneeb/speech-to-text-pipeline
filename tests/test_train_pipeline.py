"""Training loop, checkpointing and the end-to-end pipeline."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch
import yaml

from asr.config import load_config
from asr.features import FeatureExtractor
from asr.pipeline import SpeechPipeline, build_parser, main
from asr.train import (
    Sample,
    TrainingError,
    collate,
    evaluate_loss,
    iterate_batches,
    load_training_result,
    make_samples,
    save_training_result,
    train_epoch,
    train_model,
)
from asr.vocab import Vocabulary


@pytest.fixture()
def prepared(small_config):
    """A tiny corpus already featurised, shared by the training tests."""
    vocabulary = Vocabulary.from_words(small_config.synth.vocabulary)
    extractor = FeatureExtractor(small_config.features, small_config.sample_rate)
    samples = make_samples(small_config, extractor, vocabulary, 8, seed=1)
    return small_config, vocabulary, extractor, samples


def test_collate_pads_and_tracks_lengths(prepared):
    _, _, _, samples = prepared
    batch = collate(samples[:3])
    assert batch.size == 3
    assert batch.features.shape[0] == 3
    assert batch.features.shape[2] == samples[0].features.shape[1]
    assert int(batch.feature_lengths.max()) == batch.features.shape[1]
    assert int(batch.target_lengths.max()) == batch.targets.shape[1]
    assert all(int(length) == len(words) for length, words in zip(batch.target_lengths, batch.words))


def test_collate_validation(prepared):
    with pytest.raises(TrainingError, match="empty batch"):
        collate([])
    _, _, _, samples = prepared
    bad = Sample(features=np.zeros((5, 3)), tokens=(1,), words=("kasa",))
    with pytest.raises(TrainingError, match="feature dimension"):
        collate([samples[0], bad])


def test_iterate_batches_is_seeded(prepared):
    _, _, _, samples = prepared
    first = iterate_batches(samples, 3, np.random.default_rng(0))
    second = iterate_batches(samples, 3, np.random.default_rng(0))
    assert [batch.size for batch in first] == [batch.size for batch in second]
    assert [batch.words for batch in first] == [batch.words for batch in second]
    unshuffled = iterate_batches(samples, 4, np.random.default_rng(0), shuffle=False)
    assert [words for batch in unshuffled for words in batch.words] == [s.words for s in samples]


def test_iterate_batches_validation(prepared):
    _, _, _, samples = prepared
    with pytest.raises(TrainingError, match="batch_size"):
        iterate_batches(samples, 0, np.random.default_rng(0))


def test_train_epoch_reduces_loss(prepared):
    config, vocabulary, _, samples = prepared
    from asr.model import build_model

    model = build_model(config.model, config.features.output_dim, vocabulary.size)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    batches = iterate_batches(samples, config.train.batch_size, np.random.default_rng(0))
    first = train_epoch(model, batches, optimizer)["loss"]
    for _ in range(3):
        latest = train_epoch(model, batches, optimizer)["loss"]
    assert latest < first
    assert evaluate_loss(model, batches)["loss"] > 0


def test_train_epoch_validation(prepared):
    config, vocabulary, _, _ = prepared
    from asr.model import build_model

    model = build_model(config.model, config.features.output_dim, vocabulary.size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    with pytest.raises(TrainingError, match="no batches"):
        train_epoch(model, [], optimizer)
    with pytest.raises(TrainingError, match="no batches"):
        evaluate_loss(model, [])


def test_train_model_history_and_checkpoint(small_config):
    result = train_model(small_config)
    assert len(result.history) == small_config.train.epochs
    assert result.history[-1] < result.history[0]
    assert result.parameters > 0
    assert result.cmvn is not None
    payload = result.to_dict()
    assert payload["epochs"] == small_config.train.epochs
    path = save_training_result(result, small_config)
    assert path.exists()
    restored = load_training_result(small_config)
    assert restored.vocabulary.words == result.vocabulary.words
    assert restored.history == result.history
    assert restored.cmvn is not None
    assert np.allclose(restored.cmvn[0], result.cmvn[0])


def test_training_is_reproducible(small_config):
    first = train_model(small_config)
    second = train_model(small_config)
    assert first.history == second.history


def test_pipeline_learns_the_synthetic_vocabulary(trained_pipeline):
    """End to end: the trained model must actually transcribe, not just beat noise.

    Measured with the shipped recipe (18 epochs, lr 0.01): WER 0.06-0.38 across
    seeds. The 0.5 ceiling leaves room for that variance without letting a
    collapsed model pass.
    """
    report = trained_pipeline.evaluate(count=4, include_baseline=True)
    assert report.utterances == 4
    assert report.ref_words > 0
    assert report.baseline_wer is not None
    assert report.wer < report.baseline_wer
    assert report.wer < 0.5
    recognised = [evaluation for evaluation in report.evaluations if evaluation.hypothesis]
    assert len(recognised) >= 3
    assert any(
        evaluation.hypothesis == evaluation.reference for evaluation in report.evaluations
    ), "at least one utterance should be transcribed exactly"
    assert "WER" in report.summary_line()
    payload = report.to_dict()
    assert payload["utterances_detail"][0]["reference"]


def test_pipeline_loads_checkpoint_without_retraining(learn_config, trained_pipeline):
    trained = trained_pipeline
    again = SpeechPipeline.create(learn_config)
    assert again.history == trained.history
    assert again.vocabulary.words == trained.vocabulary.words
    result = again.transcribe(np.zeros(4000, dtype=np.float32))
    assert result.words == ()


def test_pipeline_prepare_audio_trims_silence(trained_pipeline):
    from asr.synth import AudioConfig, SynthConfig, synthesize_utterance

    pipeline = trained_pipeline
    audio = AudioConfig(sample_rate=8000, peak_amplitude=0.9, noise_snr_db=60.0)
    synth = SynthConfig(
        seed=1, vocabulary=("kasa",), words_per_utterance=(1, 1), f0_hz=120.0, gap_ms=(40.0, 40.0)
    )
    signal = synthesize_utterance(["kasa"], audio, synth, np.random.default_rng(0))
    padded = np.concatenate([np.zeros(2400, dtype=np.float32), signal.samples, np.zeros(2400, dtype=np.float32)])
    trimmed, segments = pipeline.prepare_audio(padded)
    assert segments
    assert trimmed.size < padded.size


def test_cli_json_output(small_config, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    payload = {
        "audio": {
            "sample_rate": small_config.sample_rate,
            "peak_amplitude": small_config.audio.peak_amplitude,
            "noise_snr_db": small_config.audio.noise_snr_db,
        },
        "synth": {
            "seed": small_config.synth.seed,
            "vocabulary": list(small_config.synth.vocabulary),
            "words_per_utterance": list(small_config.synth.words_per_utterance),
            "f0_hz": small_config.synth.f0_hz,
            "gap_ms": list(small_config.synth.gap_ms),
        },
        "vad": {
            "frame_ms": small_config.vad.frame_ms,
            "hop_ms": small_config.vad.hop_ms,
            "energy_threshold_db": small_config.vad.energy_threshold_db,
            "max_zcr": small_config.vad.max_zcr,
            "hangover_ms": small_config.vad.hangover_ms,
            "min_speech_ms": small_config.vad.min_speech_ms,
            "pre_roll_ms": small_config.vad.pre_roll_ms,
        },
        "features": {
            "frame_ms": small_config.features.frame_ms,
            "hop_ms": small_config.features.hop_ms,
            "n_fft": small_config.features.n_fft,
            "n_mels": small_config.features.n_mels,
            "fmin_hz": small_config.features.fmin_hz,
            "fmax_hz": small_config.features.fmax_hz,
            "pre_emphasis": small_config.features.pre_emphasis,
            "use_deltas": small_config.features.use_deltas,
            "cmvn": small_config.features.cmvn,
        },
        "model": {
            "channels": small_config.model.channels,
            "hidden": small_config.model.hidden,
            "dropout": small_config.model.dropout,
            "seed": small_config.model.seed,
        },
        "decode": {
            "chunk_ms": small_config.decode.chunk_ms,
            "overlap_ms": small_config.decode.overlap_ms,
            "beam_width": small_config.decode.beam_width,
            "blank_index": small_config.decode.blank_index,
        },
        "train": {
            "epochs": 2,
            "batch_size": small_config.train.batch_size,
            "learning_rate": small_config.train.learning_rate,
            "seed": small_config.train.seed,
            "checkpoint": str(tmp_path / "checkpoints" / "cli.pt"),
            "train_utterances": 6,
            "eval_utterances": 2,
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    exit_code = main(["--config", str(config_path), "--json", "--limit", "2", "--quiet"])
    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["utterances"] == 2
    assert "wer" in report and "cer" in report
    assert report["model"]["parameters"] > 0


def test_cli_parser_has_expected_flags():
    parser = build_parser()
    args = parser.parse_args(["--config", "x.yaml", "--train", "--epochs", "3", "--limit", "2", "--json"])
    assert args.train is True
    assert args.epochs == 3
    assert args.limit == 2
    assert args.json is True


def test_cli_missing_config_returns_error(tmp_path, capsys):
    exit_code = main(["--config", str(tmp_path / "absent.yaml"), "--quiet"])
    assert exit_code == 2
    assert "error:" in capsys.readouterr().err
