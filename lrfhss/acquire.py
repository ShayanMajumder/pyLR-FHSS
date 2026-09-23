# Part of the lrfhss receiver package.
"""Getting IQ and sync-word candidates out of a capture.

Three strategies, differing only in when the samples arrive:

  `detect`            candidates from IQ already in memory
  `load_and_detect`   read the whole capture, then detect
  `stream_interleaved` decode each window's candidates while the next
                      window is still being read and detected

They all hand back the same thing -- IQ plus a cluster list -- except the
streaming one, which cannot: it never holds the full cluster set, so it
calls back per window instead.
"""
import numpy as np

from . import config as cfg
from . import report
from .detect import (_fine_sync, find_packets, find_packets_windowed,
                     find_packets_streaming_interleaved)
from .dsp import notch_spurs
from .frontend import load_frontend
from .header import decode_header_at


def detect(iq, windowed_scan=False, window_sec=0.1, hop_sec=0.05):
    """Sync-word candidates from IQ already in memory."""
    report.scanning(windowed_scan, window_sec, hop_sec)
    if windowed_scan:
        return find_packets_windowed(iq, window_sec=window_sec, hop_sec=hop_sec)
    return find_packets(iq)


def load_and_detect(fn, windowed_scan=False, window_sec=0.1, hop_sec=0.05):
    """Read the capture through the front end, then scan it."""
    report.loading()
    iq = load_frontend(fn)
    report.loaded(len(iq))
    iq = notch_spurs(iq, cfg.SPUR_FREQS)
    return iq, detect(iq, windowed_scan, window_sec, hop_sec)


def can_stream(fn, windowed_scan):
    """Whether the interleaved path applies.

    It reads the file itself, so it cannot serve a preloaded array.
    """
    return windowed_scan and str(fn).lower().endswith('.wav')


def stream_interleaved(fn, acquire, decide, max_packets=20,
                       window_sec=0.1, hop_sec=0.05):
    """Detect and decode window by window, overlapping the two.

    The alternative is running detection to completion over every window
    and only then decoding, which overlaps loading with detection but
    leaves decode as a separate second pass. Here each window's candidates
    are decoded as soon as that window is detected.

    The cost is that the batched process-pool loop cannot be used: it needs
    the full cluster set upfront to build its global correlation-ordered
    sequence and to prune across ALL clusters. With only one window's
    candidates ever in flight, that trade is the right way round.

    `acquire(iq, t0, hf)` fine-syncs and decodes a header;
    `decide(iq, acq, results)` accepts or rejects the result. Returns
    (iq, results, n_sync).
    """
    report.streaming(window_sec, hop_sec)
    results = []
    confirmed_footprints = []
    n_sync = 0
    # Mutable cell: the capture grows as chunks arrive, and a candidate's
    # header replicas and hop lookahead can reach outside its own window,
    # so decoding always sees everything read so far rather than one slice.
    grown = [None]
    chunks = []

    def on_chunk(chunk):
        chunks.append(chunk)
        grown[0] = np.concatenate(chunks)

    def on_window(w_start, w_buf, w_hits, iq_prefix_len):
        nonlocal n_sync
        for _score, t_local, hf in sorted(w_hits, reverse=True):
            if len(results) >= max_packets:
                return
            t_global = w_start + t_local
            if any(lo <= t_global <= hi for lo, hi, _ in confirmed_footprints):
                continue
            iq = grown[0]
            acq = acquire(iq, t_global, hf)
            n_sync += 1
            entry = decide(iq, acq, results)
            if entry is not None and entry['crc'] and entry['footprint'] is not None:
                confirmed_footprints.append(entry['footprint'])

    find_packets_streaming_interleaved(fn, window_sec=window_sec, hop_sec=hop_sec,
                                       on_chunk=on_chunk, on_window=on_window)
    return grown[0], results, n_sync


def acquire_one(iq, t0, hf):
    """Refine one candidate: fine sync, then decode its header.

    Returns everything a decision needs -- the refined time, frequency and
    CFO, the header (or None), and the window it was read from.
    """
    fcorr, fhf, fstart, fcfo = _fine_sync(iq, int(t0), int(hf))
    hdr, soft, hwin, hf_precise = decode_header_at(iq, fhf, fstart, fcfo)
    return dict(t0=t0, hf=hf, fcorr=fcorr, fhf=fhf, fstart=fstart, fcfo=fcfo,
                hdr=hdr, hwin=hwin, hf_precise=hf_precise)
