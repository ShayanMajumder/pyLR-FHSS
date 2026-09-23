"""LR-FHSS physical-layer primitives.

Shared by the receiver pipeline and independent of it: trellis/Viterbi
decoding, GMSK symbol demodulation, header (de)interleaving and the LFSR
hop-frequency generator. Independent of the receiver pipeline and usable
without it.
"""
