#!/usr/bin/env python3
"""Generate one LR-FHSS packet, save it, and decode it back.

Edit the settings below, then run it:

    python3 examples/encode.py

The waveform is header replicas and payload fragments, each GMSK
modulated, placed on the hop schedule the LFSR generates from HOP_SEQ_ID.
The receiver rebuilds that same schedule from the decoded header, which
is why nothing has to be told where the hops went.

Writing a .wav here uses the same interleaved float32 layout the captures
use, so the file can be fed straight to decode_recording.py.
"""
import os
import struct

import numpy as np

import lrfhss

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------- settings
PAYLOAD = b'hello world'   # what to transmit
BW_KHZ = 136.72            # occupied bandwidth; see lrfhss.encoder.BW_FIELDS
HEADER_REPLICAS = 3        # 1-4
CR = 3                     # coding rate index: 0=5/6 1=2/3 2=1/2 3=1/3
GRID = 1                   # 1 = 3.9 kHz grid, 0 = 25.4 kHz (FCC)
HOP_SEQ_ID = 77            # seed the receiver recovers from the header
CARRIER_OFF_HZ = 0.0       # offset the packet from DC, as an SDR would
NOISE = 1e-3               # AWGN amplitude; 0 for a clean waveform
LEAD_SEC = 0.15            # silence before and after the burst
OUT_WAV = os.path.join(HERE, 'generated_packet.wav')
DECODE_BACK = True         # verify by decoding what we just built
# -------------------------------------------------------------------------

# The encoder reads its rates from config, so retune before generating:
# this fixes the sample rate, symbol length and hop dwell times for both
# directions at once.
lrfhss.config.retune(BW_KHZ*1e3, hdr_count=HEADER_REPLICAS)

iq, meta = lrfhss.encode(PAYLOAD, bw_khz=BW_KHZ, header_count=HEADER_REPLICAS,
                         CR=CR, grid=GRID, hop_seq_id=HOP_SEQ_ID,
                         carrier_off_hz=CARRIER_OFF_HZ)

fs = lrfhss.config.FS
lead = np.zeros(int(LEAD_SEC*fs), complex)
buf = np.concatenate([lead, iq, lead])
if NOISE:
    rng = np.random.default_rng(0)
    buf = buf + (rng.standard_normal(len(buf)) +
                 1j*rng.standard_normal(len(buf)))*NOISE

print('packet')
print('  payload        : %r (%d bytes)' % (PAYLOAD, len(PAYLOAD)))
print('  bandwidth      : %.2f kHz   grid=%d   CR=%d' % (BW_KHZ, GRID, CR))
print('  header replicas: %d x %.1f ms' % (HEADER_REPLICAS, 233.472))
print('  payload frags  : %d x %.1f ms' % (meta['num_frags'], 102.4))
print('  hop sequence   : id=%d, %d hops spanning %.1f kHz'
      % (HOP_SEQ_ID, len(meta['hop_freqs_hz']),
         (max(meta['hop_freqs_hz'])-min(meta['hop_freqs_hz']))/1e3))
print('  waveform       : %d samples = %.3f s at %.0f Hz'
      % (len(buf), len(buf)/fs, fs))

# Interleaved float32 I/Q, 2 channels -- what load_frontend expects.
os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
inter = np.empty(2*len(buf), dtype=np.float32)
inter[0::2] = buf.real
inter[1::2] = buf.imag
data = inter.tobytes()
with open(OUT_WAV, 'wb') as f:
    f.write(b'RIFF' + struct.pack('<I', 36+len(data)) + b'WAVEfmt ')
    f.write(struct.pack('<IHHIIHH', 16, 3, 2, int(fs), int(fs)*8, 8, 32))
    f.write(b'data' + struct.pack('<I', len(data)) + data)
print('  wrote          : %s' % OUT_WAV)

if DECODE_BACK:
    print('\ndecoding it back')
    packets = lrfhss.decode(buf, lrfhss.DecodeOptions(sensitive_retry=False))
    good = [p for p in packets if p['crc']]
    for p in good:
        print('  recovered      : %r' % bytes(p['bytes']))
    if any(bytes(p['bytes']) == PAYLOAD for p in good):
        print('  round trip     : OK')
    else:
        print('  round trip     : FAILED (%d candidate(s) rejected)' % len(packets))
