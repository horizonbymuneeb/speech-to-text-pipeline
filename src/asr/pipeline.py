"""End-to-end pipeline and command line entry point.

Wiring, in order: generate or accept a waveform, run the energy/ZCR detector to
drop silence, extract log-mel features, decode in overlapping chunks with the
recurrent state carried across them, then score the words against the ground
truth the generator produced. The CLI runs the same code path the tests do, so
``python -m asr.pipeline`` and ``pytest`` cannot drift apart.
"""
