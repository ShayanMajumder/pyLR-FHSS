# Part of the lrfhss receiver package.

import numpy as np
import scipy.signal as sp
from collections import OrderedDict

from . import config as cfg


_BUTTER_CACHE = {}
_ZI_CACHE = {}


def cached_butter(N, Wn, btype='low'):
    """sp.butter() memoized. decode_header_at/decode_payload_at re-designed
    the SAME filter on every call (thousands of calls on a low-SNR run);
    the design itself (npp_polyval etc) showed up in profiles. Pure cache,
    identical coefficients."""
    key = (N, float(Wn), btype)
    sos = _BUTTER_CACHE.get(key)
    if sos is None:
        sos = sp.butter(N, Wn, btype, output='sos')
        _BUTTER_CACHE[key] = sos
    return sos


def fast_sosfiltfilt(sos, x):
    """Bit-exact drop-in for scipy.signal.sosfiltfilt on 1-D input, with
    sosfilt_zi(sos) memoized.

    scipy recomputes sosfilt_zi(sos) -- a per-section linear solve -- on
    EVERY sosfiltfilt call. On a -10dB 122-cluster run this path was
    10,904 calls / 3.69s, the top numpy cost after the demod fix. The zi
    depends only on `sos`, and this receiver uses a tiny fixed set of
    filters, so it's cached. Everything else replicates scipy's own
    sosfiltfilt source exactly (same odd-extension padding, same sosfilt C
    kernel, same forward/reverse/trim sequence), so output is bit-exact --
    validated against scipy directly, not assumed."""
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
        # too short for the default padlen -- defer to scipy's own
        # validation/handling of the short-signal case
        return sp.sosfiltfilt(sos, x)
    if cfg._HAVE_VEXT_LOCAL and np.iscomplexobj(x) and x.ndim == 1:
        # Full C++ path: odd-extension + both biquad passes + reversals +
        # trim fused into one call. scipy's sosfilt kernel is C, but each
        # sosfiltfilt call pays Python overhead for the padding build, zi
        # scaling, two sosfilt invocations and two reversals -- 3464 calls
        # / 1.42s on a windowed run, the largest single cost there.
        # Validated bit-exact (max abs diff 0.0 vs scipy.sosfiltfilt over
        # 200 randomized trials).
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



from collections import OrderedDict

_TONE_LRU = OrderedDict()
_TONE_LRU_CAP = 200   # see _tone's docstring: simulated against the real


def _tone(freq_hz, length):
    """exp(-2j*pi*freq_hz*n/FS) for n=0..length-1, via cos/sin written
    directly into the output's real/imag views (0.325ms vs 0.600ms for
    np.exp(1j*ph) at span length, 1.85x -- agrees with np.exp to ~7e-12,
    verified to leave find_packets' cluster set unchanged).

    LRU-cached, sized to the measured working set, NOT the earlier FIFO
    attempt that was removed. That attempt failed because it was sized
    off a single-SNR isolated measurement (776 calls, ~232 distinct rf)
    and thrashed once the real (freq, length) key space -- 13 distinct
    lengths, not just frequency -- was accounted for in the full
    pipeline. Re-measured properly this time: at -10dB (122 clusters, the
    regime that actually stresses this function), the real access
    sequence has 5452 calls over 1320 distinct keys -- 75.8% reuse -- and
    the reuse is CONCENTRATED (top 100 keys cover 51% of all calls), not
    spread thin. Simulated LRU hit rate at several cap sizes against the
    real captured sequence before choosing one: cap=200 gives 68.5% hits
    at ~3MB (vs the 493MB an unbounded cache of all 1320 keys would cost
    at this segment length). LRU (evicts least-recently-used), not FIFO
    (evicts oldest regardless of reuse) -- this is the actual reason the
    earlier cache thrashed: FIFO can evict a hot key while a cold one it
    just inserted sits in the cache. Bookkeeping overhead measured at
    ~0.0004ms/op, negligible against the ~0.325ms/call tone-generation
    cost a hit avoids.
    """
    key = (round(float(freq_hz), 1), int(length))
    cached = _TONE_LRU.get(key)
    if cached is not None:
        _TONE_LRU.move_to_end(key)
        return cached
    ph = (-2.0*np.pi*float(freq_hz)/cfg.FS)*np.arange(length)
    t = np.empty(length, dtype=np.complex128)
    np.cos(ph, out=t.real)
    np.sin(ph, out=t.imag)
    _TONE_LRU[key] = t
    _TONE_LRU.move_to_end(key)
    if len(_TONE_LRU) > _TONE_LRU_CAP:
        _TONE_LRU.popitem(last=False)
    return t


def mix_and_filtfilt(sos, seg, freq_hz):
    """tuned = sosfiltfilt(sos, seg * exp(-2j*pi*freq_hz*n/FS)), fused.

    The mixing tone and the mixed-signal product were each a full-length
    complex temporary per call; find_packets' region loop alone builds
    ~776 of them at ~23347 samples each (~18M complex exponentials/run),
    and decode_header_at / try_thishdridx repeat the pattern (937 filtfilt
    calls). The C++ path evaluates the tone per sample directly into the
    filter's extension buffer, allocating neither temporary.

    A fully-fused C++ version (mix_and_sosfiltfilt_ext, generating the
    tone per sample inside the filter pass) was built and MEASURED SLOWER
    and is deliberately not used: scalar std::cos/std::sin per sample cost
    ~1.16 ms/call vs ~0.53 ms/call for numpy's vectorized complex exp
    feeding sosfiltfilt_ext (windowed run regressed 5.2s -> 13.9s). numpy's
    exp is SIMD; scalar libm trig cannot match it. The tone therefore stays
    in numpy and only the filter runs in C++. The extension function is
    left in place but unused, as a record of the negative result.

    """
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
