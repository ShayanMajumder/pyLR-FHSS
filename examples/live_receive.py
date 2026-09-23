#!/usr/bin/env python3
"""Decode LR-FHSS off the air, live, from an SDR.

    python3 examples/live_receive.py

Pair it with arduino/sx1262_beacon, which beacons one packet every 4 s.

The radio, the reader thread, the window overlap and the decoder process
all live in lrfhss.sdr.LiveReceiver -- see that module for why a live
receiver needs each of them. From out here it is an iterator of payloads.

Two blocks of settings: which radio to listen with, and what it is
listening for. The second one has to be right or nothing decodes --
bandwidth fixes the whole front end, and the header-replica count is not
carried in the header, so it has to be told. Grid and coding rate do come
from the decoded header, so they are not settings.
"""
import lrfhss
from lrfhss.sdr import LiveReceiver

# ------------------------------------------------------------ the radio
SDR = 'airspy'           # SoapySDR driver: airspy, rtlsdr, hackrf, ...
                         # A dict picks one radio out of several:
                         # dict(driver='airspy', serial='...').
                         # LiveReceiver.list_devices() lists what is here.
FREQ_HZ = 915e6          # what the beacon transmits on
CAPTURE_FS = 3e6         # Airspy: 3 or 6 MSPS only. An RTL-SDR takes
                         # anything up to ~3.2 MSPS, advertised or not.
GAIN = dict(LNA=8, MIX=8, VGA=8)   # a dict sets named stages, as the
                         # Airspy's LNA/MIX/VGA here. A plain number sets
                         # overall gain, which is what an RTL-SDR has: 20
                         # is about the most that does not clip a burst at
                         # bench range, 30 saturates its 8-bit ADC. None
                         # asks for the radio's own AGC.
ANTENNA = None           # e.g. 'RX' on a radio with more than one
BANDWIDTH_HZ = None      # analog filter, where the radio has one
OFFSET_HZ = 97e3         # tune this far below the signal, to keep it clear
                         # of the SDR's DC spike
QUEUE_DEPTH = 3          # windows in flight before the reader starts dropping

# ------------------------------------------------------ the transmission
BW_KHZ = 39.06           # occupied bandwidth of the packets
HDR_REPLICAS = 3         # header replicas they carry
SYNC_WORD = '12AD101B'   # RadioLib's SX126x LR-FHSS sync word
# -----------------------------------------------------------------------

lrfhss.config.retune(BW_KHZ*1e3, sync_word=SYNC_WORD, hdr_count=HDR_REPLICAS,
                     fs_capture=CAPTURE_FS)

with LiveReceiver(FREQ_HZ, SDR, gain=GAIN, antenna=ANTENNA,
                  bandwidth_hz=BANDWIDTH_HZ, offset_hz=OFFSET_HZ,
                  queue_depth=QUEUE_DEPTH) as radio:
    print('listening on %.3f MHz, Ctrl-C to stop' % (FREQ_HZ/1e6))
    for n, payload in enumerate(radio, 1):
        print('  [%6.1fs] PACKET #%d  %r' % (radio.elapsed, n, payload))

print(radio.summary())
