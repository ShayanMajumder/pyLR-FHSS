# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Getting IQ and sync-word candidates out of a capture."""
import numpy as np

from . import config as cfg
from . import report
from .detect import (_fine_sync, find_packets, find_packets_windowed,
                     find_packets_streaming_interleaved)
from .dsp import notch_spurs
from .frontend import load_frontend
from .header import _sync_carrier, decode_header_at, header_fft_peak


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
    """Whether the interleaved path applies."""
    return windowed_scan and str(fn).lower().endswith('.wav')


def stream_interleaved(fn, acquire, decide, max_packets=20,
                       window_sec=0.1, hop_sec=0.05):
    """Detect and decode window by window, overlapping the two."""
    report.streaming(window_sec, hop_sec)
    results = []
    confirmed_footprints = []
    n_sync = 0
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
    """Refine one candidate: fine sync, then decode its header."""
    fcorr, fhf, fstart, fcfo = _fine_sync(iq, int(t0), int(hf))
    win = fstart - cfg.SYNC_START_BIT*cfg.SMBL
    ws = win if win >= 0 else fstart
    f0 = header_fft_peak(iq[ws:ws + cfg.HDR_BIT_NUM*cfg.SMBL + 4*cfg.SMBL], fhf)
    est = _sync_carrier(iq, win, f0 - 400, f0 + 400)
    sync_q = est[2] if est is not None else float('inf')   # can't check: don't skip
    if sync_q < cfg.SYNC_SKIP_Q:
        return dict(t0=t0, hf=hf, fcorr=fcorr, fhf=fhf, fstart=fstart, fcfo=fcfo,
                    hdr=None, hwin=win, hf_precise=fhf, sync_q=sync_q)
    hdr, soft, hwin, hf_precise = decode_header_at(iq, fhf, fstart, fcfo)
    return dict(t0=t0, hf=hf, fcorr=fcorr, fhf=fhf, fstart=fstart, fcfo=fcfo,
                hdr=hdr, hwin=hwin, hf_precise=hf_precise, sync_q=sync_q)
