# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT

import functools

import numpy as np
import scipy.signal as sp
from collections import OrderedDict

from . import config as cfg


_BUTTER_CACHE = {}
_ZI_CACHE = {}


@functools.lru_cache(maxsize=32)
def _hann(n):
    w = np.hanning(n)
    w.flags.writeable = False
    return w


def hann(n):
    """np.hanning(n), cached: header and payload windows come in a handful
    of lengths, and rebuilding one per FFT was ~3% of a low-SNR decode.
    """
    return _hann(int(n))


def cached_butter(N, Wn, btype='low'):
    """sp.butter() memoized."""
    key = (N, float(Wn), btype)
    sos = _BUTTER_CACHE.get(key)
    if sos is None:
        sos = sp.butter(N, Wn, btype, output='sos')
        _BUTTER_CACHE[key] = sos
    return sos


def fast_sosfiltfilt(sos, x):
    """Bit-exact drop-in for scipy.signal.sosfiltfilt on 1-D input, with
    sosfilt_zi(sos) memoized.
    """
    from scipy.signal import sosfilt, sosfilt_zi
    from scipy.signal._arraytools import odd_ext, axis_slice, axis_reverse
    sos = np.asarray(sos)
    n_sections = sos.shape[0]
    key = sos.tobytes()
    zi = _ZI_CACHE.get(key)
    if zi is None:
        zi = sosfilt_zi(sos)
        _ZI_CACHE[key] = zi
    ntaps = 2*n_sections + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    edge = ntaps*3
    x = np.asarray(x)
    if x.shape[-1] <= edge:
        return sp.sosfiltfilt(sos, x)
    if cfg._HAVE_VEXT_LOCAL and np.iscomplexobj(x) and x.ndim == 1:
        return np.asarray(cfg._vext_local.sosfiltfilt_ext(
            np.ascontiguousarray(sos, dtype=np.float64),
            np.ascontiguousarray(x, dtype=np.complex128),
            np.ascontiguousarray(zi, dtype=np.float64), int(edge)))
    ext = odd_ext(x, edge, axis=-1)
    zi_r = zi.reshape([n_sections, 2])
    x_0 = axis_slice(ext, stop=1, axis=-1)
    y, _ = sosfilt(sos, ext, axis=-1, zi=zi_r*x_0)
    y_0 = axis_slice(y, start=-1, axis=-1)
    y, _ = sosfilt(sos, axis_reverse(y, axis=-1), axis=-1, zi=zi_r*y_0)
    y = axis_reverse(y, axis=-1)
    if edge > 0:
        y = axis_slice(y, start=edge, stop=-edge, axis=-1)
    return y


_TONE_LRU = OrderedDict()
_TONE_LRU_CAP = 200
# Also capped by size: at 3 MS/s (US915 DR5/DR6) one fine-sync tone is
# 33 MB, and 200 of them filled the machine.
_TONE_LRU_BYTES = 256*2**20


_TONE_BLOCK = 512


def _block_tone(w, length):
    """exp(1j*w*n), n = 0..length-1, as one block of B phases times a per-block
    rotation: exp(1j*w*(bB + i)) = exp(1j*w*bB)*exp(1j*w*i).
    """
    B = _TONE_BLOCK
    nb = -(-length//B)
    base = np.exp(1j*w*np.arange(B))
    rot = np.exp(1j*(w*B)*np.arange(nb))
    return (rot[:, None]*base[None, :]).ravel()[:length]


def _tone(freq_hz, length):
    """exp(-2j*pi*freq_hz*n/FS) for n=0..length-1, via cos/sin written directly
    into the output's real/imag views (0.325ms vs 0.600ms for np.exp(1j*ph)
    at span length, 1.85x -- agrees with np.exp to ~7e-12, verified to leave
    find_packets' cluster set unchanged).
    """
    key = (round(float(freq_hz), 1), int(length), float(cfg.FS))
    cached = _TONE_LRU.get(key)
    if cached is not None:
        _TONE_LRU.move_to_end(key)
        return cached
    t = _block_tone(-2.0*np.pi*float(freq_hz)/cfg.FS, int(length))
    _TONE_LRU[key] = t
    _TONE_LRU.move_to_end(key)
    while len(_TONE_LRU) > 1 and (
            len(_TONE_LRU) > _TONE_LRU_CAP
            or sum(v.nbytes for v in _TONE_LRU.values()) > _TONE_LRU_BYTES):
        _TONE_LRU.popitem(last=False)
    return t


def mix_decimate_filtfilt(seg, freq_hz, cutoff_hz, D):
    """mix_and_filtfilt for a narrow output, D times cheaper: tune to baseband,
    average blocks of D samples, then the same 4th-order Butterworth at
    FS/D.
    """
    if D <= 1:
        return mix_and_filtfilt(cached_butter(4, cutoff_hz/(cfg.FS/2)), seg, freq_hz)
    n = len(seg)//D*D
    w = -2.0*np.pi*float(freq_hz)/cfg.FS
    x = (np.ascontiguousarray(seg[:n]).reshape(-1, D) @ (np.exp(1j*w*np.arange(D))/D))
    x *= np.exp(1j*(w*D)*np.arange(n//D))
    return fast_sosfiltfilt(cached_butter(4, cutoff_hz/(cfg.FS/D/2)), x)


def mix_and_filtfilt(sos, seg, freq_hz):
    """tuned = sosfiltfilt(sos, seg * exp(-2j*pi*freq_hz*n/FS)), fused."""
    return fast_sosfiltfilt(sos, seg*_tone(freq_hz, len(seg)))


def sosfilt_zi_cached(sos):
    from scipy.signal import sosfilt_zi
    return sosfilt_zi(sos)


def notch_spurs(iq, freqs):
    sos = cached_butter(2, 300/(cfg.FS/2), 'high')
    for sf in freqs:
        n = np.arange(len(iq))
        iq = iq*np.exp(-1j*2*np.pi*sf*(n/cfg.FS))
        iq = fast_sosfiltfilt(sos, iq)
        iq = iq*np.exp(1j*2*np.pi*sf*(n/cfg.FS))
    return iq
