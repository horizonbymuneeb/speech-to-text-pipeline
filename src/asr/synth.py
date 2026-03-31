"""Synthetic speech generation with known ground truth.

A recogniser needs labelled audio. Downloading a corpus is not an option here
(no network at test time, and a real corpus is hundreds of megabytes), so the
pipeline generates its own: each word is built from a consonant burst and a
vowel, and each vowel is additive-synthesised from a harmonic series whose
amplitude envelope is shaped by the first three formants. That produces audio
with the spectral structure a log-mel front end and a CTC model can actually
learn — a pure tone would let the model cheat on pitch alone.

Because the generator knows the word list and the exact sample span of every
word, every downstream metric has ground truth without any alignment step.
Everything is seeded: same seed, same waveform, bit for bit.
"""
