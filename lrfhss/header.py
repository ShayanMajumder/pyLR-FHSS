# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np

from . import config as cfg
from .phy.hopping import calculate_freq_from_hop_seq_id
from .phy.gmsk import demod_symbols, msk_templates, msk_trellis_llr
from .phy.framing import decode_header, deinterleave_header
from .dsp import cached_butter, hann, mix_and_filtfilt
from .fft import _fft


def _valid_bw(bw_bits):
    try:
        calculate_freq_from_hop_seq_id(1, 1, 1, list(bw_bits), [0], 1)
        return True
    except (IndexError, KeyError, ValueError):
        return False


def header_hop_freq(iq, s0):
    seg = iq[s0:s0+cfg.STAY_HDR]
    f = np.fft.fftfreq(len(seg), 1/cfg.FS)
    band = (np.abs(f) > 2e3) & (np.abs(f) < cfg.ALLBW/2)
    win = cfg._WIN_HDR if len(seg) == cfg.STAY_HDR else hann(len(seg))
    S = np.abs(_fft(seg*win)); S[~band] = 0
    return f[np.argmax(S)]


def _cfo_tone(cfo_val, smbl, n_seg):
    """exp(-1j*cfo_val/smbl*n_seg), via cos/sin into preallocated real/imag
    instead of np.exp(1j*ph).
    """
    ph = (-float(cfo_val)/smbl)*n_seg
    t = np.empty(len(n_seg), dtype=np.complex128)
    np.cos(ph, out=t.real)
    np.sin(ph, out=t.imag)
    return t


def header_fft_peak(seg, hdr_f):
    """Header burst carrier, coarse: the burst's FFT peak within 4 kHz of the
    fine-sync frequency.
    """
    if len(seg) < 2:                     # window past the capture's end
        return hdr_f
    fseg = np.fft.fftfreq(len(seg), 1/cfg.FS)
    bandm = (np.abs(fseg-hdr_f) < 4000)
    S = np.abs(_fft(seg*hann(len(seg)))); S[~bandm] = 0
    return fseg[np.argmax(S)]


def decode_header_at(iq, hdr_f, sync_start_abs, cfo):
    win_start = sync_start_abs - cfg.SYNC_START_BIT*cfg.SMBL
    if win_start < 0:
        win_start = sync_start_abs
    seg = iq[win_start:win_start + cfg.HDR_BIT_NUM*cfg.SMBL + 4*cfg.SMBL]
    sos = cached_butter(4, 300/(cfg.FS/2))
    hdr_f_precise = header_fft_peak(seg, hdr_f)
    if cfg.HEADER_TRELLIS:
        res = _decode_header_trellis(iq, win_start, hdr_f_precise)
        if res is not None:
            hdr, soft, df = res
            return hdr, soft, win_start, hdr_f_precise+df
        if not cfg.HEADER_OLD_FALLBACK:
            return None, np.array([]), win_start, hdr_f_precise
    n_seg = np.arange(len(seg))
    centers_base = cfg.LOOKDIST + np.arange(cfg.HDR_BIT_NUM)*cfg.SMBL
    best = (None, -1, None)
    cached = []
    for df in range(-300, 301, 60):
        tuned_freq = mix_and_filtfilt(sos, seg, hdr_f_precise+df)
        for cfo_adj in [cfo, 0.0, -cfo]:
            tuned2 = tuned_freq*_cfo_tone(cfo_adj, cfg.SMBL, n_seg)
            centers = centers_base[centers_base < len(tuned2)-cfg.LOOKDIST]
            soft = demod_symbols(tuned2, centers, cfg.LOOKDIST, cfg.PHASESLOPE)
            if len(soft) < cfg.HDR_BIT_NUM:
                continue
            deint = deinterleave_header(soft[:cfg.HDR_BIT_NUM])
            cached.append((deint, soft, df))
            hdr = decode_header(deint)
            if hdr is not None:
                if not (1 <= hdr['payloadlen'] <= 100) or not _valid_bw(hdr['BW']):
                    bim = np.mean(np.abs(soft) > 0.5)
                    if bim > best[1]:
                        best = (soft, bim, df)
                    continue
                return hdr, soft, win_start, hdr_f_precise+df
            bim = np.mean(np.abs(soft) > 0.5)
            if bim > best[1]:
                best = (soft, bim, df)
    for deint, soft, df in cached:
        hdr = decode_header(deint, try_backward=True)
        if hdr is not None and 1 <= hdr['payloadlen'] <= 100 and _valid_bw(hdr['BW']):
            return hdr, soft, win_start, hdr_f_precise+df
    res = _decode_header_combined(iq, seg, win_start, hdr_f_precise, cfo, sos)
    if res is not None:
        hdr, soft, df = res
        return hdr, soft, win_start, hdr_f_precise+df
    fallback_soft = best[0] if best[0] is not None else np.array([])
    fallback_f = hdr_f_precise + (best[2] if best[2] is not None else 0)
    return None, fallback_soft, win_start, fallback_f


def _smoothed_peak_hz(seg, lo_hz, hi_hz, width_hz=400.0):
    """Carrier of a ~488 Hz GMSK burst: peak of the power spectrum summed over
    width_hz.
    """
    S = np.abs(_fft(seg*hann(len(seg))))**2
    ff = np.fft.fftfreq(len(seg), 1/cfg.FS)
    o = np.argsort(ff)
    ff, S = ff[o], S[o]
    k = max(1, int(round(width_hz/(ff[1] - ff[0]))))
    Ss = np.convolve(S, np.ones(k), 'same')
    m = (ff > lo_hz) & (ff < hi_hz)
    if not np.any(m):
        return None
    return float(ff[m][np.argmax(Ss[m])])


def _decode_header_combined(iq, seg, win_start, hdr_f_precise, cfo, sos):
    """Last-resort header decode from ALL replicas, soft-combined."""
    if cfg.HDR_COUNT < 2:
        return None
    n = len(seg)
    lo, hi = -cfg.ALLBW/2 - cfg.HOP_EDGE_TOL_HZ, cfg.ALLBW/2 + cfg.HOP_EDGE_TOL_HZ
    f_lock = _smoothed_peak_hz(seg, hdr_f_precise - 4000, hdr_f_precise + 4000)
    if f_lock is None:
        return None
    others = []
    for k in range(-(cfg.HDR_COUNT-1), cfg.HDR_COUNT):
        s0 = win_start + k*cfg.STAY_HDR
        if k == 0 or s0 < 0 or s0 + n > len(iq):
            continue
        rseg = iq[s0:s0+n]
        f_k = _smoothed_peak_hz(rseg, lo, hi)
        if f_k is not None:
            others.append((rseg, f_k + (hdr_f_precise - f_lock)))
    if not others:
        return None
    n_seg = np.arange(n)
    centers = cfg.LOOKDIST + np.arange(cfg.HDR_BIT_NUM)*cfg.SMBL
    centers = centers[centers < n - cfg.LOOKDIST]
    if len(centers) < cfg.HDR_BIT_NUM:
        return None

    def soft_at(s, f, cfo_adj):
        t = mix_and_filtfilt(sos, s, f)*_cfo_tone(cfo_adj, cfg.SMBL, n_seg)
        return demod_symbols(t, centers, cfg.LOOKDIST, cfg.PHASESLOPE)[:cfg.HDR_BIT_NUM]

    for df in range(-300, 301, 60):
        for cfo_adj in [cfo, 0.0, -cfo]:
            s_lock = soft_at(seg, hdr_f_precise + df, cfo_adj)
            total = s_lock.copy()
            wsum = 1.0
            used = 0
            for rseg, f_k in others:
                s_k = soft_at(rseg, f_k + df, cfo_adj)
                den = np.linalg.norm(s_lock)*np.linalg.norm(s_k)
                w = float(np.dot(s_lock, s_k)/den) if den > 0 else 0.0
                if w > 0.1:
                    total += w*s_k
                    wsum += w
                    used += 1
            if not used:
                continue
            total /= wsum             # back to the +/-1 range decode expects
            hdr = decode_header(deinterleave_header(total))
            if hdr is not None and 1 <= hdr['payloadlen'] <= 100 and _valid_bw(hdr['BW']):
                return hdr, total, df
    return None


_SYNC_TPL = {}


def _sync_template():
    """exp(j*phase) of the 32-bit sync word from its first bit centre to its
    last, built from the encoder's own per-bit-pair waveforms
    (phy.gmsk.msk_templates), so it matches the transmitted phase exactly.
    """
    key = (cfg.FS, cfg.SYMBOL_RATE_HZ, tuple(cfg.SYNC_WORD))
    if key not in _SYNC_TPL:
        T, L = msk_templates()
        spb = cfg.FS/cfg.SYMBOL_RATE_HZ
        w = cfg.SYNC_WORD
        n = len(w) - 1
        tpl = np.zeros(int(round(n*spb)) + L, complex)
        cum = 0.0
        for k in range(1, len(w)):
            o = int(round((k - 1)*spb))
            tpl[o:o + L] = T[w[k-1], w[k]]*np.exp(1j*cum)
            cum += {(1, 1): np.pi/2, (0, 0): -np.pi/2}.get((w[k-1], w[k]), 0.0)
        _SYNC_TPL[key] = tpl[:int(round(n*spb))]
    return _SYNC_TPL[key]


_ZOOM = {}


def _zoom_fft(n, span_hz, m):
    """Cached scipy ZoomFFT: m spectrum samples across [-span/2, span/2)."""
    import scipy.signal as sps
    key = (n, round(span_hz, 6), m, cfg.FS)
    if key not in _ZOOM:
        _ZOOM[key] = sps.ZoomFFT(n, [-span_hz/2, span_hz/2], m=m, fs=cfg.FS,
                                 endpoint=False)
    return _ZOOM[key]


def _sync_carrier(iq, win_start, f_lo, f_hi):
    """Carrier and timing of a header burst from its known sync word."""
    tpl = _sync_template()
    spb = cfg.FS/cfg.SYMBOL_RATE_HZ
    n = len(tpl)
    df = cfg.FS/(1 << 17)
    zoom = cfg.SYNC_ZOOM and (f_hi - f_lo) < cfg.FS/64
    if zoom:
        df = min(df, 1.0)
        fc = 0.5*(f_lo + f_hi)
        m = int(np.ceil((f_hi - f_lo)/df))
        zf = _zoom_fft(n, m*df, m)
        ff = fc - m*df/2 + df*np.arange(m)
        conj_tpl = np.conj(tpl)*np.exp(-2j*np.pi*fc/cfg.FS*np.arange(n))
        band = np.ones(m, bool)
    else:
        nfft = 1 << 17
        ff = np.fft.fftfreq(nfft, 1/cfg.FS)
        band = (ff >= f_lo) & (ff <= f_hi)
        conj_tpl = np.conj(tpl)
    if not band.any():
        return None
    best = None
    for g in (-int(round(cfg.SMBL/3)), 0, int(round(cfg.SMBL/3))):
        a = win_start + cfg.LOOKDIST + int(round(cfg.SYNC_START_BIT*spb)) + g
        if a < 0 or a + n > len(iq):
            continue
        y = iq[a:a + n]*conj_tpl
        P = np.abs(zf(y) if zoom else np.fft.fft(y, nfft))**2
        i = np.flatnonzero(band)[np.argmax(P[band])]
        q = P[i]/(np.mean(P[band]) + 1e-30)
        if best is None or q > best[2]:
            best = (float(ff[i]), g, float(q))
    return best


def _trellis_burst_llrs(iq, s0, f_lo, f_hi):
    """Trellis LLRs for the header burst whose window starts near s0, with its
    carrier somewhere in [f_lo, f_hi]: sync-word carrier and timing
    estimate, then a small (dcfo, timing) grid around it.
    """
    est = _sync_carrier(iq, s0, f_lo, f_hi)
    if est is None:
        return None
    f, g, q = est
    spb = cfg.FS/cfg.SYMBOL_RATE_HZ
    lead = cfg.SMBL                       # room for the bit before bit 0
    n_bits = cfg.HDR_BIT_NUM
    a = s0 - lead
    if a < 0:
        return None
    seg = iq[a:a + lead + n_bits*cfg.SMBL + 4*cfg.SMBL]
    if len(seg) < lead + n_bits*cfg.SMBL:
        return None
    sos = cached_butter(4, 300/(cfg.FS/2))
    base = lead + cfg.LOOKDIST + np.round(np.arange(-1, n_bits)*spb).astype(int)
    tuned = mix_and_filtfilt(sos, seg, f)  # once; each d is a rotation in the trellis
    hyps, carriers = [], []
    for d in cfg.SYNC_DCFO_HZ:
        td = tuned if cfg.FILTER_ONCE else mix_and_filtfilt(sos, seg, f + d)
        for dg in (-int(round(cfg.SMBL/6)), 0, int(round(cfg.SMBL/6))):
            hyps.append((td, base + g + dg, d if cfg.FILTER_ONCE else 0.0))
            carriers.append(f + d)
    return msk_trellis_llr(hyps, first_bit=None), carriers, q


def _header_from_llr(llr):
    soft = np.clip(llr/(2*np.median(np.abs(llr)) + 1e-12), -1, 1)
    hdr = decode_header(deinterleave_header(soft[:cfg.HDR_BIT_NUM]))
    if hdr is not None and 1 <= hdr['payloadlen'] <= 100 and _valid_bw(hdr['BW']):
        return hdr, soft
    return None


def _decode_header_trellis(iq, win_start, hdr_f_precise, top=3):
    """Header decode with the phase-tracking trellis demodulator."""
    res = _trellis_burst_llrs(iq, win_start, hdr_f_precise - 400, hdr_f_precise + 400)
    if res is None:
        return None
    llr, carriers, q_lock = res
    order = np.argsort(-np.mean(np.abs(llr), axis=1))
    for h in order[:top]:
        got = _header_from_llr(llr[h])
        if got is not None:
            return got[0], got[1], carriers[h] - hdr_f_precise
    if cfg.HDR_COUNT < 2 or q_lock < cfg.SYNC_COMBINE_Q:
        return None
    best = order[0]
    total = llr[best].copy()
    n = cfg.HDR_BIT_NUM*cfg.SMBL
    used = 0
    for k in range(-(cfg.HDR_COUNT-1), cfg.HDR_COUNT):
        s0 = win_start + k*cfg.STAY_HDR
        if k == 0 or s0 - cfg.SMBL < 0 or s0 + n > len(iq):
            continue
        r = _trellis_burst_llrs(iq, s0, -cfg.ALLBW/2 - cfg.HOP_EDGE_TOL_HZ,
                                cfg.ALLBW/2 + cfg.HOP_EDGE_TOL_HZ)
        if r is None or r[2] < cfg.SYNC_MIN_Q:
            continue
        lk = r[0][int(np.argmax(np.mean(np.abs(r[0]), axis=1)))]
        den = np.linalg.norm(total)*np.linalg.norm(lk)
        if den > 0 and np.dot(total, lk)/den > 0.1:
            total += lk
            used += 1
    if used:
        got = _header_from_llr(total)
        if got is not None:
            return got[0], got[1], carriers[best] - hdr_f_precise
    return None


def detect_replica_index(iq, hdr, hwin, hfp):
    """Find which header replica (1-based, out of HDR_COUNT) was actually
    locked at hwin, by hypothesis-testing each candidate index against real
    signal energy at its LFSR-predicted frequencies.
    """
    payloadlen = hdr['payloadlen']; CR = hdr['CR']
    payload_length_bits = 8*(payloadlen+2)+6
    data_in_bitcount = int(np.ceil(payload_length_bits*[6/5, 3/2, 2, 3][CR]))
    num_frags = int(np.ceil(data_in_bitcount/48))
    if not _valid_bw(hdr['BW']):
        return 1
    step = 0.95367431640625
    pll = calculate_freq_from_hop_seq_id(hdr['grid'], hdr['hop'], cfg.HDR_COUNT,
                                         hdr['BW'], hdr['hopseq'], num_frags)
    if pll is None:
        return 1

    best_idx, best_score = 1, -1.0
    for cand in range(1, cfg.HDR_COUNT+1):
        addval = hfp - pll[cand-1]*step
        score = 0.0
        for k in range(cfg.HDR_COUNT):
            t0k = hwin + (k-(cand-1))*cfg.STAY_HDR
            if t0k < 0 or t0k+cfg.STAY_HDR > len(iq):
                continue
            pred_f = pll[k]*step + addval
            seg = iq[t0k:t0k+cfg.STAY_HDR]
            ff = np.fft.fftfreq(len(seg), 1/cfg.FS)
            S = np.abs(_fft(seg*cfg._WIN_HDR))**2
            bm = np.abs(ff-pred_f) < 300
            peak = S[bm].max() if np.any(bm) else 0.0
            noise = np.median(S) + 1e-30
            score += 10*np.log10(peak/noise) if peak > 0 else 0.0
        if score > best_score:
            best_score = score; best_idx = cand
    return best_idx
