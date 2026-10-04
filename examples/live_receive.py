#!/usr/bin/env python3
# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Decode the LR1120 DR8 beacon (examples/arduino/lr1120_beacon) live.

    python3 examples/live_receive.py
"""
import lrfhss
from lrfhss.sdr import LiveReceiver

# ------------------------------------------------------------ the radio
SDR = 'airspy'           # SoapySDR driver: airspy, rtlsdr, hackrf, ...
FREQ_HZ = 869.525e6      # the beacon's TEST_FREQ
CAPTURE_FS = 3e6         # Airspy: 3 or 6 MSPS only.
# Long range: LNA and mixer at maximum (lowest noise figure); VGA 10 lifts the
# noise ~25 dB over the ADC's own and keeps ~20 dB of headroom (measured).
GAIN = dict(LNA=14, MIX=12, VGA=10)
ANTENNA = None           # e.g. 'RX' on a radio with more than one
BANDWIDTH_HZ = None      # analog filter, where the radio has one
OFFSET_HZ = 97e3         # tune this far off, keeping the 137 kHz band off DC
BUFFER_S = 30            # decoder lag absorbed before a window is skipped

# ------------------------------------------------------ the transmission
DR = 8                   # 136.72 kHz, CR 1/3, 3 header replicas
REGION = 'EU868'
SYNC_WORD = '12AD101B'   # the beacon's SYNC_WORD

lrfhss.config.retune_dr(DR, region=REGION, sync_word=SYNC_WORD,
                        fs_capture=CAPTURE_FS)

with LiveReceiver(FREQ_HZ, SDR, gain=GAIN, antenna=ANTENNA,
                  bandwidth_hz=BANDWIDTH_HZ, offset_hz=OFFSET_HZ,
                  buffer_s=BUFFER_S) as radio:
    print('DR%d on %.3f MHz, Ctrl-C to stop' % (DR, FREQ_HZ/1e6))
    for n, pkt in enumerate(radio, 1):
        print('  [%6.1fs] PACKET #%d  %r  SNR %+.1f dB'
              % (radio.elapsed, n, pkt.payload, pkt.snr_db))

print(radio.summary())
