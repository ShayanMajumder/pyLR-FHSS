"""
lrfhss_demod.py -- GMSK symbol demod + hop extraction, ported from the repo.

Port of lrfh_demod_smbls.m: for each symbol center, measure the phase slope
across +/-lookdist samples and normalize by the expected pi/2-per-symbol slope
to produce a soft bit in ~[-cap, +cap]. Includes the DC/drift adjustment.
"""
import numpy as np

try:
    from .. import _viterbi_ext as _vext
    _HAVE_VEXT = True
except ImportError:
    _vext = None
    _HAVE_VEXT = False


def demod_symbols(sig, smpltime, lookdist, phaseslope, demodusenum=4,
                  adapt_drift=True, soft_cap=1.0):
    """
    sig         : 1D complex baseband, already tuned so the hop tone is ~0 Hz.
    smpltime    : integer sample indices of symbol centers (0-based).
    lookdist    : +/- sample span for slope estimate (LRF_cfg.lookdist).
    phaseslope  : expected phase change per sample for a '1' (LRF_cfg.phaseslope).
    Returns soft bit values (1D float).

    Highest call-count function in the pipeline (11600+ calls/run in
    profiling -- this sits at the bottom of the blind-acquisition grid
    search over freq x time x CFO). Uses the pybind11 C++ port
    (ext/viterbi_ext.cpp: demod_symbols_ext) when available -- folds the
    whole per-symbol phase measurement + np.percentile drift threshold +
    two-pass drift estimate into one call instead of ~10 chained numpy ops
    per invocation (numpy dispatch overhead was ~2.6s/run total, 0.9s of
    that in np.percentile alone). Bit-exact with the numpy fallback below
    (validated to ~1e-15, floating-point summation-order noise only -- see
    ext/test_viterbi_ext.py). Falls back to pure numpy if the extension
    isn't built.
    """
    sig = np.asarray(sig)
    smpltime = np.asarray(smpltime)
    if _HAVE_VEXT:
        # Pass the complex128 array straight through -- no .real/.imag
        # split, no ascontiguousarray. The split version cost two
        # contiguous-array materializations per call (strided .real/.imag
        # views of complex128), which profiling on a -10dB 122-cluster run
        # showed as 698,836 ascontiguousarray calls / 6.66s -- the single
        # biggest cost in the pipeline. std::complex<double> is
        # layout-compatible with numpy complex128, so C++ reads this
        # buffer in place.
        return np.asarray(_vext.demod_symbols_ext(
            np.ascontiguousarray(sig, dtype=np.complex128),
            smpltime.astype(np.int64), int(lookdist), float(phaseslope),
            bool(adapt_drift), float(soft_cap)))

    n = len(smpltime)
    valid = (smpltime - lookdist >= 0) & (smpltime + lookdist < len(sig))
    bits = np.zeros(n)
    if np.any(valid):
        c = smpltime[valid]
        phase_hi = np.angle(sig[c + lookdist])
        phase_lo = np.angle(sig[c - lookdist])
        diff = phase_hi - phase_lo
        # wrap to (-pi, pi]-ish per the repo's rule
        big = np.abs(diff) > np.pi/2
        diff = np.where(big, diff - np.sign(diff)*2*np.pi, diff)
        still_big = np.abs(diff) > np.pi/2
        diff = np.where(still_big, 0.0, diff)
        k = diff/(lookdist*2)
        bits[valid] = k/phaseslope

    # DC/drift adjustment (GMSKmodvaladj) from lrfh_demod_smbls.m
    thr = np.percentile(np.abs(bits), 80)
    useidx = np.abs(bits) < thr
    adj = 0.0
    for zzz in range(2):
        if zzz == 0:
            look0 = bits < 0
        else:
            look0 = bits >= 0
        look = look0 & useidx
        if np.any(look):
            val = np.mean(bits[look])
        else:
            val = 0.0
        frac = np.sum(look)/max(np.sum(useidx), 1)
        if zzz == 0:
            adj += frac*(val + 1)
        else:
            adj += frac*(val - 1)
    if adapt_drift:
        bits = bits - adj

    bits = np.clip(bits, -soft_cap, soft_cap)
    return bits


def tune_and_lpf(sig, freq_hz, fs, lpf_b, lpf_a):
    """Mix a hop tone at freq_hz down to 0 and zero-phase LPF (isolates the hop).
    Mirrors lrfh_lpfsig's de-rotation + filtfilt. We use a per-segment complex
    exponential (freq relative to segment center = simple e^{-j2pi f n/fs})."""
    from scipy.signal import filtfilt
    n = np.arange(len(sig))
    tuned = sig * np.exp(-1j*2*np.pi*freq_hz*(n/fs))
    return filtfilt(lpf_b, lpf_a, tuned)