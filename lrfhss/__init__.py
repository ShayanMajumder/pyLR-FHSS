"""LR-FHSS blind receiver.

Decode a capture:

    import lrfhss

    packets = lrfhss.decode('capture.wav')
    for p in packets:
        if p['crc']:
            print(p['bytes'])

Generate one:

    iq, meta = lrfhss.encode(b'hello world', dr=8)

A LoRaWAN data rate names a whole configuration -- bandwidth, coding rate
and header-replica count. That last one is not carried in the header, so
for standards-compliant traffic the DR is how a receiver knows it:

    lrfhss.config.retune_dr(8)                  # EU868 DR8
    lrfhss.config.retune_dr(5, region='US915')

Much of the LR-FHSS parameter space has no DR name, so bandwidth, coding
rate and replica count can still be given directly.

Tune it for the capture first -- the defaults suit one configuration, and
every bandwidth needs its own front-end rate and hop-search window:

    lrfhss.config.FS = 1e6
    lrfhss.config.ALLBW = 900_000

Parameters MUST be set on `lrfhss.config`, not on this package: modules
read `config.FS` at call time, so `lrfhss.FS = ...` would bind a name
nothing looks at. See examples/decode_sweep.py for a full retune.

Modules follow the signal path:

    config      every retunable parameter (FS, DECIM, ALLBW, HDR_COUNT, ...)
    fft         FFT entry points (scipy.fft)
    wavio       .wav header parsing and chunked IQ reads
    frontend    decimating channeliser (load_frontend*)
    dsp         filters, tone cache, spur notching
    detect      matched filter, CFAR, packet/cluster search, fine sync
    header      header hop frequency, CFO, header decode, replica index
    payload     payload window, hop footprint, payload decode
    quality     energy/length checks
    plotting    per-packet spectrograms
    acquire     IQ + candidates, and acquiring one candidate
    arbitrate   which candidates are real packets (dedup, pruning)
    report      every console message a run emits
    pipeline    decode() -- orders the above
    encoder     encode() -- the transmit side, same primitives
    sdr         LiveReceiver -- decode off the air (optional, needs SoapySDR)
    phy/        protocol primitives, independent of the pipeline:
                  hopping  LFSR hop-frequency generator
                  fec      trellis, Viterbi, interleaving, whitening
                  gmsk     GMSK symbol demodulation
                  framing  header field layout and decode
"""
from . import config
from .config import datarate, retune_dr
from .detect import find_packets, find_packets_windowed
from .encoder import encode, encode_packet, payload_fragments
from .frontend import load_frontend, load_frontend_windowed
from .header import decode_header_at
from .payload import decode_payload_at
from .pipeline import DecodeOptions, decode
from .plotting import plot_packet_spectrogram
from .quality import check_energy_length

__all__ = [
    'config', 'datarate', 'retune_dr',
    'decode', 'DecodeOptions',
    'encode', 'encode_packet', 'payload_fragments',
    'load_frontend', 'load_frontend_windowed',
    'find_packets', 'find_packets_windowed',
    'decode_header_at', 'decode_payload_at',
    'check_energy_length',
    'plot_packet_spectrogram',
]
