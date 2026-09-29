"""
lrfhss_demod.py -- GMSK symbol demod + hop extraction, ported from the repo.

Port of lrfh_demod_smbls.m: for each symbol center, measure the phase slope
across +/-lookdist samples and normalize by the expected pi/2-per-symbol slope
to produce a soft bit in ~[-cap, +cap]. Includes the DC/drift adjustment.
"""
import numpy as np

from .. import config as cfg

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
    if _HAVE_VEXT and not cfg.DEMOD_AVG_PHASE:
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
        if cfg.DEMOD_AVG_PHASE:
            # Amplitude-weighted mean phase increment across the whole
            # +/-lookdist window (angle of the summed lag-1 products),
            # instead of the phase difference of its two end samples:
            # same quantity, but one noise dip at an end sample no longer
            # decides the bit.
            p = sig[1:]*np.conj(sig[:-1])
            cs = np.concatenate([[0], np.cumsum(p)])
            diff = np.angle(cs[c + lookdist] - cs[c - lookdist])*(2*lookdist)
        else:
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

# ---------------------------------------------------------------------------
# Phase-tracking trellis demodulator.
#
# demod_symbols above is a differential detector: each soft bit is the phase
# change between two samples a quarter symbol either side of the bit centre.
# That throws away ~4-5 dB against a detector that uses the whole waveform
# and the carrier phase. msk_trellis_llr is that detector. The waveform is
# MSK-like (+/-pi/2 per bit, smoothed at bit changes), so between two bit
# centres it depends only on the two bits either side; the carrier phase is
# unknown and drifts with residual frequency error, so it goes INTO the
# trellis, quantised to M points with a -1/0/+1 random-walk step per bit.
# State = (phase, previous bit); forward-backward gives each bit's LLR.
# Measured on DR8 payloads at exact timing: 10% PER moves from about -20.5
# to about -23 dB (125 kHz-referenced), level with LoRa SF12. It tolerates
# +/-0.23 symbol of timing error but only ~+/-5-10 Hz of frequency error,
# so callers search frequency on a fine grid (see payload._trellis_payload).
# ---------------------------------------------------------------------------
_MSK_TPL = {}


def msk_templates():
    """exp(j*(phase - phase at the previous bit centre)) over one
    centre-to-centre window, per (previous bit, bit), cut from the
    encoder's own waveform so its transition smoothing is modelled
    exactly. Returns (T[2, 2, L], L)."""
    key = (cfg.FS, cfg.SYMBOL_RATE_HZ)
    if key not in _MSK_TPL:
        from ..encoder import gen_ideal_waveform
        spb = cfg.FS/cfg.SYMBOL_RATE_HZ
        L = int(np.floor(spb))
        bits = np.random.default_rng(12345).integers(0, 2, 400)
        ph = gen_ideal_waveform(bits)
        mids = np.round((np.arange(len(bits)) + 0.5)*spb).astype(int)
        acc = {}
        for k in range(3, len(bits) - 3):
            m0 = mids[k-1]
            acc.setdefault((bits[k-1], bits[k]), []).append(ph[m0:m0+L] - ph[m0])
        T = np.empty((2, 2, L), complex)
        for (bp, b), v in acc.items():
            T[bp, b] = np.exp(1j*np.mean(v, axis=0))
        _MSK_TPL[key] = (T, L)
    return _MSK_TPL[key]


def msk_trellis_llr(hyps, first_bit=0, bw_hz=300.0, M=32, p_walk=0.2):
    """LLRs (>0 = bit 1) for several hypotheses at once.

    Each hypothesis is (tf, mids) or (tf, mids, df_hz). tf: baseband, tuned
    so the hop is ~0 Hz and low-passed to bw_hz. mids: bit-centre sample
    indices; mids[0] is the bit BEFORE the first one wanted (first_bit, or
    None if unknown), so each returns len(mids)-1 LLRs. df_hz: an extra
    frequency shift, applied only to the samples actually used -- cheaper
    than tuning and filtering the whole segment again per offset (a shift
    of tens of Hz barely moves the signal in a 300 Hz filter).
    All hypotheses must have the same len(mids). Returns [n_hyps, n_bits].

    Speed: the correlations use every D-th sample (tf is low-passed to
    ~300 Hz and sampled ~280x faster than that, so this loses nothing),
    and the forward-backward runs in the linear domain with per-step
    scaling -- multiplies and adds instead of an exp and a log per element.
    """
    T, L = msk_templates()
    spb = cfg.FS/cfg.SYMBOL_RATE_HZ
    D = cfg.TRELLIS_DECIM or max(1, int(spb//20))   # ~20 samples per symbol
    sub = np.arange(0, L, D)
    Tc = np.conj(T[:, :, sub])
    Tc4 = np.ascontiguousarray(Tc.reshape(4, -1))   # rows: (bp, b) = 00, 01, 10, 11
    H = len(hyps)
    K = len(hyps[0][1]) - 1
    S = np.zeros((H, K, 2, 2), complex)
    A = np.empty(H); sig2 = np.empty(H)
    # Hypotheses sharing a signal and a timing share one gather of the
    # samples; a frequency shift d factorises exactly as a per-bit phase
    # (at mids[k]) times a per-offset phase (at l), so it lands on the four
    # short templates and one scalar per bit rather than on every sample.
    groups = {}
    for h, hyp in enumerate(hyps):
        mids = np.asarray(hyp[1])
        groups.setdefault((id(hyp[0]), mids[:K+1].tobytes()), []).append(h)
    for hs in groups.values():
        tf, mids = hyps[hs[0]][0], np.asarray(hyps[hs[0]][1])
        idx = mids[:K, None] + sub[None, :]
        ok = idx < len(tf)
        seg = np.where(ok, tf[np.minimum(idx, len(tf)-1)], 0)
        span = tf[mids[0]:min(mids[-1], len(tf)):D]
        p_span = np.mean(np.abs(span)**2) if len(span) else None
        for h in hs:
            d = hyps[h][2] if len(hyps[h]) > 2 else 0.0
            # (bits x samples) @ (samples x 4 templates): one BLAS matmul
            if d:
                w = -2j*np.pi*d/cfg.FS
                Td = Tc4*np.exp(w*sub)[None, :]
                S[h] = (D*(seg @ Td.T)*np.exp(w*mids[:K])[:, None]).reshape(K, 2, 2)
            else:
                S[h] = (D*(seg @ Tc4.T)).reshape(K, 2, 2)
            A[h] = np.median(np.abs(S[h]).reshape(K, -1).max(1))/L
            if p_span is None:           # entirely past the end of the capture:
                A[h], sig2[h] = 0.0, 1.0     # no information, LLRs of 0
                continue
            sig2[h] = max(p_span - A[h]**2, 1e-12)
    kappa = cfg.FS/(2*bw_hz)               # correlated samples per independent one
    rot = np.exp(-2j*np.pi*np.arange(M)/M)
    G = ((2*A/(sig2*kappa))[:, None, None, None, None] *
         np.real(rot[None, None, :, None, None]*S[:, :, None, :, :]))   # [H,K,M,bp,b]
    # Linear domain, scaled per bit: E = exp(G - max over the bit's branches),
    # floored at e^-50 so a strong signal cannot underflow the forward mass
    # to zero (it only touches paths less likely than e^-50).
    E = np.exp(np.maximum(G - G.max(axis=(2, 3, 4), keepdims=True), -50.0))
    P = _msk_transitions(M, p_walk)        # [bp, b, M, M], m -> m'
    PT = np.ascontiguousarray(np.swapaxes(P, 2, 3))
    # Branch terms laid out [K, bp, b, H, M] so each step is one batched
    # matmul over all four (previous bit, bit) branches.
    Eb = np.ascontiguousarray(np.transpose(E, (1, 3, 4, 0, 2)))
    a = np.zeros((K+1, 2, H, M))           # [k, bp, h, m]: forward, by previous bit
    if first_bit is None:
        a[0] = 1.0
    else:
        a[0, first_bit] = 1.0
    for k in range(K):
        nxt = np.matmul(a[k][:, None]*Eb[k], P).sum(axis=0)    # [b, h, m]
        a[k+1] = nxt/(nxt.sum(axis=(0, 2), keepdims=True) + 1e-300)
    beta = np.ones((2, H, M))              # [b, h, m]
    llr = np.empty((H, K))
    for k in range(K-1, -1, -1):
        t = Eb[k]*np.matmul(beta[None], PT)                    # [bp, b, h, m]
        num = np.einsum('phm,pbhm->bh', a[k], t)
        llr[:, k] = np.log(num[1] + 1e-300) - np.log(num[0] + 1e-300)
        cur = t.sum(axis=1)                                    # [bp, h, m]
        beta = cur/(cur.sum(axis=(0, 2), keepdims=True) + 1e-300)
    return llr


_MSK_P = {}


def _msk_transitions(M, p_walk):
    """Phase-state transition matrices, one per (previous bit, bit): the
    modulation's phase step (+M/4, -M/4, or 0 across a bit change) followed
    by the -1/0/+1 random walk. P[bp, b][m, m'] = P(m -> m')."""
    key = (M, p_walk)
    if key not in _MSK_P:
        q4 = M//4
        step = {(1, 1): q4, (0, 0): -q4, (0, 1): 0, (1, 0): 0}
        P = np.zeros((2, 2, M, M))
        m = np.arange(M)
        for (bp, b), st in step.items():
            for d, w in ((-1, p_walk/2), (0, 1 - p_walk), (1, p_walk/2)):
                P[bp, b, m, (m + st + d) % M] += w
        _MSK_P[key] = P
    return _MSK_P[key]
