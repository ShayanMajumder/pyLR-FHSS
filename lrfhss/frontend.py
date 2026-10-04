# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT

import threading
import numpy as np
import scipy.signal as sp

from . import config as cfg
from .fft import _fft, _ifft
from .wavio import _WavChunkReader


def _channel_mask(M):
    fb = np.fft.fftfreq(M, 1/cfg.FS)
    lpf = sp.firwin(12*cfg.DECIM+1, 0.9*(cfg.FS/2)/(cfg.FS_CAPTURE/2))
    _, Hl = sp.freqz(lpf, 1, worN=2*np.pi*fb/cfg.FS_CAPTURE)
    bhp, ahp = sp.butter(4, cfg.DC_NOTCH_HZ/(cfg.FS_CAPTURE/2), 'high')
    _, Hh = sp.freqz(bhp, ahp, worN=2*np.pi*(cfg.CARRIER_OFF+fb)/cfg.FS_CAPTURE)
    return Hl*np.abs(Hh)**2


def load_frontend(fn, NB=cfg._NB, OV=cfg._OV):
    if str(fn).lower().endswith('.wav'):
        raw = _WavChunkReader(fn)
    else:
        raw = np.memmap(fn, dtype=np.float32, mode='r')
    N2 = raw.shape[0]//2
    M = NB//cfg.DECIM
    k0 = int(round(cfg.CARRIER_OFF/cfg.FS_CAPTURE*NB))
    src = (k0 + np.r_[0:M//2, -(M//2):0]) % NB
    mask = _channel_mask(M)
    hop = NB-OV; dOV = OV//cfg.DECIM
    chunk_pos = []
    pos = -OV
    while pos < N2:
        chunk_pos.append(pos)
        pos += hop

    _read_lock = threading.Lock()

    def _load_chunk(pos):
        lo = max(0, pos); hi = min(N2, pos+NB)
        with _read_lock:
            a = np.array(raw[lo*2:hi*2], copy=True)
        if cfg._HAVE_VEXT_LOCAL:
            seg = np.asarray(cfg._vext_local.deinterleave_iq_ext(
                np.ascontiguousarray(a, dtype=np.float32), NB, lo-pos))
        else:
            seg = np.zeros(NB, complex)
            seg[lo-pos:lo-pos+(hi-lo)] = (a[0::2].astype(np.float64)
                                          + 1j*a[1::2].astype(np.float64))
        X = _fft(seg)
        y = _ifft(X[src]*mask)/cfg.DECIM
        return y[dOV:]

    # Serial: the per-chunk FFT used to run across a thread pool.
    outs = [_load_chunk(p) for p in chunk_pos]
    return np.concatenate(outs)[:N2//cfg.DECIM]


def load_frontend_windowed(fn, NB=cfg._NB, OV=cfg._OV, queue_maxsize=2):
    """Generator version of load_frontend for .wav inputs: reads and
    decimates in windows with true read-ahead, instead of concatenating
    the whole capture into one array.
    """
    import threading, queue
    if not str(fn).lower().endswith('.wav'):
        yield load_frontend(fn, NB=NB, OV=OV)
        return

    raw = _WavChunkReader(fn)
    N2 = raw.shape[0]//2
    M = NB//cfg.DECIM
    k0 = int(round(cfg.CARRIER_OFF/cfg.FS_CAPTURE*NB))
    src = (k0 + np.r_[0:M//2, -(M//2):0]) % NB
    mask = _channel_mask(M)
    hop = NB-OV; dOV = OV//cfg.DECIM

    q = queue.Queue(maxsize=queue_maxsize)
    SENTINEL = object()
    exc_holder = []

    def reader():
        """I/O-only thread: seeks + reads each window's raw interleaved float32
        samples via file positioning and pushes (pos, lo, hi, raw) onto the
        queue.
        """
        try:
            pos = -OV
            while pos < N2:
                lo = max(0, pos); hi = min(N2, pos+NB)
                a = raw.read_window(lo*2, (hi-lo)*2)
                q.put((pos, lo, hi, a))
                pos += hop
        except Exception as e:
            exc_holder.append(e)
        finally:
            q.put(SENTINEL)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    total_out = N2//cfg.DECIM
    emitted = 0
    try:
        while True:
            item = q.get()
            if item is SENTINEL:
                break
            pos, lo, hi, a = item
            if cfg._HAVE_VEXT_LOCAL:
                seg = np.asarray(cfg._vext_local.deinterleave_iq_ext(
                    np.ascontiguousarray(a, dtype=np.float32), NB, lo-pos))
            else:
                seg = np.zeros(NB, complex)
                seg[lo-pos:lo-pos+(hi-lo)] = (a[0::2].astype(np.float64)
                                              + 1j*a[1::2].astype(np.float64))
            X = _fft(seg)
            y = _ifft(X[src]*mask)/cfg.DECIM
            out_chunk = y[dOV:]
            remaining = total_out - emitted
            if remaining <= 0:
                continue
            if len(out_chunk) > remaining:
                out_chunk = out_chunk[:remaining]
            emitted += len(out_chunk)
            yield out_chunk
    finally:
        t.join()
        raw.close()
        if exc_holder:
            raise exc_holder[0]
