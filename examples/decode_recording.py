#!/usr/bin/env python3
"""Decode the bundled recording.

    python3 examples/decode_recording.py

data/capture_39kHz_hdr3.wav is a real over-the-air LR-FHSS packet: 39.06
kHz, three header replicas, carrying "hello world". It ships already
decimated by the front end -- 2 MB at 166.667 kHz rather than 13 MB at
1 MSPS -- so it is read as samples and handed straight to decode().

The two settings below are the ones the receiver cannot work out for
itself. Bandwidth fixes the whole front end; header-replica count is not
carried in the header, and a wrong value puts the payload window a dwell
out, which decodes noise rather than failing cleanly. Grid and coding
rate come from the decoded header, so they are not settings.

Each decoded packet is also saved as a spectrogram, with every header
replica and payload fragment boxed where the receiver found it -- which
is the quickest way to see whether a capture really holds a packet, and
where a failing one falls apart. Only packets whose payload decoded are
plotted, so a plot is evidence rather than a guess.

For a different capture, edit WAV and the two settings below.
"""
import os

import numpy as np
from scipy.io import wavfile

import lrfhss

HERE = os.path.dirname(os.path.abspath(__file__))
WAV = os.path.join(HERE, 'data', 'capture_39kHz_hdr3.wav')
PLOT_DIR = os.path.join(HERE, 'plots')   # one PNG per decoded packet
BW_KHZ = 39.06           # occupied bandwidth of this recording
HDR_REPLICAS = 3         # header replicas it was transmitted with
SYNC_WORD = '12AD101B'   # RadioLib's SX126x LR-FHSS sync word

# Derives the whole front end from the bandwidth: sample rate, symbol
# length, hop dwell times, sync matched filter, CFO ramp, hop-search
# window and the payload-search budget.
lrfhss.config.retune(BW_KHZ*1e3, sync_word=SYNC_WORD, hdr_count=HDR_REPLICAS)

fs, raw = wavfile.read(WAV)
capture = raw[:, 0].astype(np.float64) + 1j*raw[:, 1].astype(np.float64)
assert abs(fs - lrfhss.config.FS) < 1, (
    'recording is %d Hz but retune(%.0f) gives %.0f Hz -- the file would be '
    'decoded at the wrong rate' % (fs, BW_KHZ*1e3, lrfhss.config.FS))

packets = lrfhss.decode(capture, lrfhss.DecodeOptions(
    plot_spectrograms=True, plot_dir=PLOT_DIR))

# `crc` is the accept gate; the rest are rejected candidates, returned
# because they are what you need when a capture will not decode.
decoded = [p for p in packets if p['crc']]

print()
for pkt in decoded:
    data = bytes(pkt['bytes'])
    print('decoded %d byte(s) at t=%.3f s: %r'
          % (len(data), pkt['t0']/lrfhss.config.FS,
             data.decode('utf-8', 'replace')))

if not decoded:
    print('no packet decoded (%d candidate(s) rejected)' % len(packets))
else:
    print('spectrogram(s) in %s' % PLOT_DIR)
