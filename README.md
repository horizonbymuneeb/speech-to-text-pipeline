# speech-to-text-pipeline

A streaming speech recogniser that runs on CPU with no model download and no
dataset download. The audio is generated rather than recorded: a formant
synthesiser builds short words from a consonant burst plus an additive-synthesis
vowel, so every utterance arrives with a transcript and word-level sample spans
for free. On top of that sit the pieces a real recogniser needs — an
energy/zero-crossing voice activity detector with hangover, a log-mel front end,
a small convolutional-recurrent CTC model trained from scratch on the synthetic
vocabulary, a chunked streaming decoder that carries recurrent state and stitches
overlaps, and a WER/CER scorer built on a hand-written edit-distance alignment.

Measured with the shipped configuration (8 kHz, 32 mel bands, 45,033 parameters,
18 epochs in about 12 s on two CPU cores): **WER 0.083, CER 0.088** on six held-out
utterances, with 5 of 6 transcribed exactly. One-shot decoding of the same audio
scores WER 0.000; the remaining error is a repeated word lost at a chunk seam
(see Limitations).

## What is in the box

| Module | Job |
|---|---|
| `src/asr/config.py` | Typed dataclass config, YAML loading, per-key validation, millisecond → sample/frame conversions |
| `src/asr/dsp.py` | Framing, pre-emphasis, Hann window, power spectrum, HTK mel scale, area-normalised filterbank, CMVN, deltas |
| `src/asr/synth.py` | Formant-based synthetic speech: syllables, vowels from harmonic series, consonant noise bursts, utterance assembly with word spans |
| `src/asr/vocab.py` | Word vocabulary with the CTC blank pinned at index 0; encode/decode and checkpoint serialisation |
| `src/asr/vad.py` | Short-time energy + zero-crossing detection, noise-floor estimate, hangover, segment merging and silence trimming |
| `src/asr/features.py` | Log-mel front end with optional global CMVN statistics and delta stacking; reusable extractor that caches the filterbank |
| `src/asr/ctc.py` | CTC forward loss written from scratch, greedy decode, prefix beam search, frame-positioned collapse |
| `src/asr/model.py` | Conv1d → bidirectional GRU → linear CTC model, deterministic init, checkpoint save/load, recurrent state threading |
| `src/asr/decode.py` | Chunk planning, streaming decode with state carry-over and frame-anchored overlap stitching |
| `src/asr/metrics.py` | Levenshtein alignment with an edit script, WER/CER, corpus pooling, alignment rendering |
| `src/asr/train.py` | Seeded dataset generation, padded batching, training/eval loops, global CMVN estimation, checkpointing |
| `src/asr/pipeline.py` | End-to-end wiring (VAD → features → streaming decode → scoring) and the CLI |

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .

make test                       # 180 tests, about 21 s on CPU
asr-pipeline --config config.yaml
```

Without installing, `PYTHONPATH=src python -m asr --config config.yaml` runs the
same entry point. The first run trains a model and writes
`checkpoints/ctc.pt`; later runs load it. Useful flags:

```bash
asr-pipeline --config config.yaml --train          # retrain from scratch
asr-pipeline --config config.yaml --limit 3        # fewer evaluation utterances
asr-pipeline --config config.yaml --json           # full report as JSON
asr-pipeline --config config.yaml --epochs 6       # override train.epochs
```

From Python:

```python
from asr import SpeechPipeline, load_config

config = load_config("config.yaml")
pipeline = SpeechPipeline.create(config)          # loads checkpoints/ctc.pt
report = pipeline.evaluate(count=6, include_baseline=True)
print(report.summary_line())                      # WER / CER / error counts
words = pipeline.transcribe(audio_samples).words  # streaming path
```

## Design notes

**Synthetic formant speech, not a downloaded corpus.** A recogniser needs
labelled audio, and a real corpus is both large and impossible to fetch at test
time. The generator builds each word from a consonant burst and a vowel whose
harmonic series is shaped by the first three formants of that vowel. Formants
matter: with a plain sine wave the model can win on pitch alone, whereas the
formant envelope forces the log-mel front end to carry real spectral
information. The payoff is exact ground truth and bit-identical reproducibility
from a seed; the cost is a toy vocabulary (eight CV-CV words) and audio that is
much cleaner than speech.

**Global CMVN statistics, because per-chunk normalisation drifts.** Normalising
each chunk against its own mean and variance is fine offline and wrong when
streaming: consecutive chunks end up on different scales, which is a
distribution shift at every seam. The training run therefore computes mean/std
over the whole training corpus once, stores them in the checkpoint, and every
extractor — offline or per chunk — normalises with those fixed numbers. Fixing
this took streaming WER from 0.58 to 0.29 before any other change.

**Seams are resolved by frame position, not token identity.** The obvious way to
merge two chunks is to strip the longest suffix of the accumulated output that
matches a prefix of the new chunk. That silently eats legitimate repeats: a
transcript containing "timo timo" loses one "timo" whenever the pair sits on a
seam. The decoder instead keeps the overlap purely as left context and drops the
tokens emitted from the overlapped frames, which is why "timo timo" and
"seko seko" now survive (streaming WER 0.125 → 0.083). Prefix beam search does
not expose a frame alignment, so beam mode falls back to identity matching and
keeps that limitation.

**The CTC loss is hand-written and cross-checked.** The forward recursion runs in
log space over the extended label sequence and follows PyTorch's `mean`
convention (divide each sample's loss by its target length, then average), so it
can be compared directly with `torch.nn.functional.ctc_loss`. Writing it rather
than importing it earned its keep: the comparison test caught a real bug — the
recursion advanced through padded frames for samples with a shorter
`input_length`, which inflated the loss by up to 1.5 nats per sample.

**Learning rate is not a knob you can nudge up here.** With this model, Adam at
`lr=0.01` converges to loss 0.008 in 18 epochs, while `lr=0.02` collapses to
emitting blanks forever (loss pinned around 2.5) and never recovers. The shipped
`config.yaml` is at the top of the stable range; the training tests pin the
recipe so a future tweak cannot quietly reproduce the collapsed regime.

## Tests

```bash
make test          # or: python -m pytest -q
```

**180 tests** in 11 files, about 21 s on CPU, no network and no GPU. The
interesting ones:

- `test_vad.py` — builds a signal with a known 0.3 s / 0.4 s speech / 0.3 s
  layout and asserts the detected region lands within the frame grid plus the
  configured pre-roll and hangover; also covers the ZCR veto on marginal frames,
  hangover extension, short-run rejection and silence trimming.
- `test_features.py` — asserts the log-mel shape against `frame_count`, that a
  440 Hz tone peaks in the band covering 440 Hz, that CMVN gives zero mean and
  unit variance per dimension, and that chunked features stay consistent with
  whole-utterance features when global statistics are used.
- `test_metrics.py` — hand-computed alignments for substitution, insertion and
  deletion separately, the `kitten`/`sitting` distance of 3, empty-reference
  conventions, and the fact that corpus WER pools errors over reference words
  rather than averaging per-utterance rates (2/5, not 0.625).
- `test_ctc.py` — matches `torch.nn.functional.ctc_loss` within 1e-5 across
  random shapes with variable input/target lengths, checks greedy collapsing
  (repeats merged, blanks dropped, blank-separated repeats kept), and verifies
  that the beam's reported score equals the exact CTC probability of the
  sequence it returns.
- `test_decode.py` — chunk-plan geometry, longest-suffix overlap, and a
  context-free stub model proving the stitched output equals the single-pass
  output when nothing depends on context, plus that overlap-emitted tokens are
  dropped and tail tokens kept.
- `test_model.py` — output shape `(T/2, batch, vocab)`, seeded initialisation,
  recurrent state shape and effect, checkpoint round-trip and rejection of a
  mismatched signature.
- `test_train_pipeline.py` — collation and padding, seeded batch order, loss
  decreasing within an epoch, checkpoint reproducibility, and the end-to-end
  result: the trained pipeline beats the untrained baseline and stays under a
  WER ceiling of 0.5 with at least one exact transcript.
- `test_config.py`, `test_dsp.py`, `test_synth.py`, `test_vocab.py` — validation
  errors, framing coverage, determinism of the generator, and vocabulary
  invariants.

## Limitations

- **Toy language.** Eight CV-CV words, synthetic formants, white-noise only. This
  is not a recogniser for real speech; it is a pipeline that can be measured
  honestly because its ground truth is exact.
- **Small evaluation set.** Six utterances (24 words) means the reported WER
  moves in steps of 1/24. Treat 0.083 as "2 word errors out of 24", not as a
  precise rate.
- **One repeated-word deletion remains in streaming mode.** Greedy CTC can only
  emit two identical words with a blank between them, and when that blank lands
  in an overlapped region the repeat is lost; one-shot decoding scores 0.000 on
  the same audio. Beam mode cannot fix it because it has no frame alignment.
- **No language model and no external lexicon** — the decoder is acoustic only.
- **The model is tiny** (45k parameters, two conv layers and one GRU layer). It
  needs the shipped 18 epochs to converge, and the learning rate has a narrow
  stable range (see design notes).
- **Streaming is over in-memory arrays**, not a live microphone or file stream:
  `transcribe()` takes samples and chunks them by configuration. There is no
  partial-hypothesis callback.
- **The VAD assumes a stationary noise floor per utterance** (a low percentile of
  frame energies) and does not adapt across a long non-stationary recording.
- **CPU only, single process.** No batching across streams, no ONNX export, no
  quantisation.
