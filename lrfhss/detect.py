# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT

import threading
import numpy as np
import scipy.signal as sp

from . import config as cfg
from .dsp import mix_decimate_filtfilt, notch_spurs
from .fft import _fft, _ifft
from .frontend import load_frontend_windowed


def _cfar_detect(ratio, guard=cfg.CFAR_GUARD, factor=cfg.CFAR_FACTOR,
                 floor=cfg.CFAR_FLOOR):
    """BUILT AND TESTED, NOT USED -- kept as a documented negative result."""
    n = len(ratio)
    if n == 0:
        return np.array([], dtype=int)
    idx = np.arange(n)
    # cumulative-sum based sliding reference mean (O(n), not O(n*window))
    csum = np.concatenate([[0.0], np.cumsum(ratio)])
    win = min(max(2*guard + 8, 8), n)  # reference window half-width, bounded by n
    thresholds = np.empty(n)
    for i in range(n):
        lo_ref = max(0, i - win - guard)
        hi_ref = min(n, i + win + guard + 1)
        g_lo = max(lo_ref, i - guard)
        g_hi = min(hi_ref, i + guard + 1)
        total = csum[hi_ref] - csum[lo_ref]
        excl = csum[g_hi] - csum[g_lo]
        cnt = (hi_ref - lo_ref) - (g_hi - g_lo)
        ref_mean = (total - excl) / cnt if cnt > 0 else ratio[i]
        thresholds[i] = max(ref_mean * factor, floor)
    return idx[ratio > thresholds]


PRESCREEN_NFFT = 2048          # at the 136.72 kHz rates' 166.7 kHz sample rate


def prescreen_nfft():
    """STFT size for the prescreen, scaled with the sample rate so every
    data rate gets the same 81 Hz bins and 12.3 ms windows. A fixed 2048
    gave 1.5 kHz bins and 0.7 ms windows at 3 MS/s (US915 DR5/6): three
    times the hop's width of noise per bin, a third of a symbol of signal
    per look, and most weak packets were never found."""
    return max(PRESCREEN_NFFT, int(round(PRESCREEN_NFFT*cfg.FS/(500e3/3))))


def mf_lpf_hz():
    """Low-pass cutoff around a prescreen region before the sync matched filter
    and fine sync.
    """
    err = cfg.FS/prescreen_nfft()/2 + cfg.PRESCREEN_F_ROUND_HZ/2
    return max(cfg.MF_LPF_HZ, err + 300.0)


def _mf_scores(tuned, D=1):
    """Normalized sync-word matched-filter score per candidate start, maxed
    over CFO via FFT along the sync taps.
    """
    D = max(1, int(D))
    tstep = cfg.SMBL//2
    last = len(tuned)*D - cfg.SYNC_OFF[-1] - 1
    if last <= 0:
        return np.array([], dtype=int), np.array([])
    starts = np.arange(0, last, tstep)
    idxmat = starts[:, None] + cfg.SYNC_OFF[None, :]
    if D > 1:
        idxmat = np.clip(np.rint((idxmat - (D - 1)/2)/D).astype(int), 0, len(tuned) - 1)
    S = tuned[idxmat]
    base = S/cfg.SYNC_VEC[None, :]
    energy = np.sqrt(np.sum(np.abs(S)**2, axis=1) + 1e-30)
    F = _fft(base, n=cfg.MF_NCFO, axis=1)
    mfbest = np.max(np.abs(F), axis=1)
    return starts, mfbest/(energy*np.sqrt(cfg.MF_NSYNC))


def region_decim():
    """Decimation for the region scan: the matched filter reads one sample
    per symbol after a ~450 Hz low-pass, so ~16 kHz (~33 samples a symbol)
    holds everything it uses. 10 at 167 kHz, 31 at 500 kHz, 188 at 3 MHz.
    """
    return max(1, int(round(cfg.FS/cfg.REGION_RATE_HZ)))


def find_packets_streaming_interleaved(fn, window_sec=1.0, hop_sec=0.5, queue_maxsize=2,
                                       on_chunk=None, on_window=None):
    """Same reader+assembler threading as find_packets_streaming, but instead
    of collecting all windows' hits into one list and returning after the
    whole capture is read, calls on_window(w_start, w_buf, hits, prefix_len)
    immediately after EACH window's find_packets() finishes -- so the caller
    can decode+print that window's candidates right away, while the
    assembler thread is already reading/decimating the NEXT window off disk.
    """
    import queue
    min_window = max((cfg.SYNC_OFF[-1]+1)/cfg.FS, cfg.STAY_HDR*0.6/cfg.FS)
    if window_sec < min_window*1.3:
        window_sec = min_window*1.3
    win_samp = int(window_sec*cfg.FS)
    hop_samp = int(hop_sec*cfg.FS)
    pkt_len = cfg.STAY_HDR*cfg.HDR_COUNT + cfg.STAY_DATA*8
    margin = int(cfg.STAY_HDR*0.6) + 8192
    keep_back = pkt_len + margin

    floor_state = {}
    win_idx = [0]
    win_q = queue.Queue(maxsize=2)
    WIN_SENTINEL = object()
    exc_holder = []

    def assembler():
        try:
            buf = np.zeros(0, dtype=complex)
            buf_start = 0
            for chunk in load_frontend_windowed(fn, queue_maxsize=queue_maxsize):
                chunk = notch_spurs(chunk, cfg.SPUR_FREQS)
                if on_chunk is not None:
                    on_chunk(chunk)
                buf = np.concatenate([buf, chunk])
                while len(buf) >= win_samp:
                    win_q.put((buf_start, buf[:win_samp].copy()))
                    adv = min(hop_samp, len(buf))
                    buf = buf[adv:]
                    buf_start += adv
                if len(buf) > keep_back + win_samp:
                    trim = len(buf) - (keep_back + win_samp)
                    buf = buf[trim:]
                    buf_start += trim
            if len(buf) >= cfg.STAY_HDR//4:
                win_q.put((buf_start, buf))
        except Exception as e:
            exc_holder.append(e)
        finally:
            win_q.put(WIN_SENTINEL)

    t_asm = threading.Thread(target=assembler, daemon=True)
    t_asm.start()

    prefix_len = 0
    while True:
        item = win_q.get()
        if item is WIN_SENTINEL:
            break
        w_start, w_buf = item
        hits = find_packets(w_buf, floor_state=floor_state)
        win_idx[0] += 1
        t_lo = w_start/cfg.FS; t_hi = (w_start+len(w_buf))/cfg.FS
        print('  [window %d] t=%.2f-%.2fs: %d candidate(s) this window'
             % (win_idx[0], t_lo, t_hi, len(hits)))
        prefix_len = w_start + len(w_buf)
        if on_window is not None:
            on_window(w_start, w_buf, hits, prefix_len)

    t_asm.join()
    if exc_holder:
        raise exc_holder[0]


def find_packets_streaming(fn, window_sec=1.0, hop_sec=0.5, queue_maxsize=2,
                           verbose=True, return_iq=False):
    """Producer/consumer streaming version of find_packets: never holds the
    full decimated capture in memory.
    """
    import queue
    min_window = max((cfg.SYNC_OFF[-1]+1)/cfg.FS, cfg.STAY_HDR*0.6/cfg.FS)
    if window_sec < min_window*1.3:
        window_sec = min_window*1.3
    win_samp = int(window_sec*cfg.FS)
    hop_samp = int(hop_sec*cfg.FS)
    pkt_len = cfg.STAY_HDR*cfg.HDR_COUNT + cfg.STAY_DATA*8
    margin = int(cfg.STAY_HDR*0.6) + 8192   # span margin + filter transient pad
    keep_back = pkt_len + margin

    all_hits = []
    floor_state = {}
    win_idx = 0
    kept = [] if return_iq else None

    win_q = queue.Queue(maxsize=2)
    WIN_SENTINEL = object()
    exc_holder2 = []

    def assembler():
        try:
            nonlocal_buf = np.zeros(0, dtype=complex)
            nonlocal_start = 0
            for chunk in load_frontend_windowed(fn, queue_maxsize=queue_maxsize):
                chunk = notch_spurs(chunk, cfg.SPUR_FREQS)
                if kept is not None:
                    kept.append(chunk)
                nonlocal_buf = np.concatenate([nonlocal_buf, chunk])
                while len(nonlocal_buf) >= win_samp:
                    win_q.put((nonlocal_start, nonlocal_buf[:win_samp].copy()))
                    adv = min(hop_samp, len(nonlocal_buf))
                    nonlocal_buf = nonlocal_buf[adv:]
                    nonlocal_start += adv
                if len(nonlocal_buf) > keep_back + win_samp:
                    trim = len(nonlocal_buf) - (keep_back + win_samp)
                    nonlocal_buf = nonlocal_buf[trim:]
                    nonlocal_start += trim
            if len(nonlocal_buf) >= cfg.STAY_HDR//4:
                win_q.put((nonlocal_start, nonlocal_buf))
        except Exception as e:
            exc_holder2.append(e)
        finally:
            win_q.put(WIN_SENTINEL)

    t_asm = threading.Thread(target=assembler, daemon=True)
    t_asm.start()

    while True:
        item = win_q.get()
        if item is WIN_SENTINEL:
            break
        w_start, w_buf = item
        local_hits = find_packets(w_buf, floor_state=floor_state)
        win_idx += 1
        if verbose:
            t_lo = w_start/cfg.FS; t_hi = (w_start+len(w_buf))/cfg.FS
            print('  [window %d] t=%.2f-%.2fs: %d candidate(s) this window'
                 % (win_idx, t_lo, t_hi, len(local_hits)))
        for c, t0, hf in local_hits:
            all_hits.append((c, w_start+t0, hf))

    t_asm.join()
    if exc_holder2:
        raise exc_holder2[0]

    if not all_hits:
        return ([], np.concatenate(kept)) if return_iq else []
    all_hits.sort(reverse=True)
    t_cell = max(1, pkt_len); f_cell = 3000
    buckets = {}
    deduped = []
    for c, t0, hf in all_hits:
        bt, bf = int(t0)//t_cell, int(hf)//f_cell
        is_dup = False
        for dbt in (-1, 0, 1):
            for dbf in (-1, 0, 1):
                for _, ct, cf in buckets.get((bt+dbt, bf+dbf), ()):
                    if abs(ct-t0) < pkt_len and abs(cf-hf) < 3000:
                        is_dup = True
                        break
                if is_dup:
                    break
            if is_dup:
                break
        if is_dup:
            continue
        deduped.append((c, t0, hf))
        buckets.setdefault((bt, bf), []).append((c, t0, hf))
    if return_iq:
        return deduped, np.concatenate(kept)
    return deduped


def find_packets(iq, floor_state=None):
    """Two-stage detection: cheap STFT energy screen for coarse (t,f)
    LOCALIZATION only (loose threshold, not a decision gate), then the
    sync-word matched filter runs only on those local windows.
    """
    nfft = prescreen_nfft(); hop = max(1, cfg.STAY_HDR//8)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f_axis_raw, _, Zxx = sp.stft(iq, fs=cfg.FS, window='hann', nperseg=nfft,
                                     noverlap=nfft-hop, nfft=nfft, boundary=None)
    f_axis = np.fft.fftshift(f_axis_raw)
    Zs = np.fft.fftshift(Zxx, axes=0)
    energy_map = (Zs.real*Zs.real + Zs.imag*Zs.imag).T   # [n_frames, nfft]
    n_frames = energy_map.shape[0]
    if n_frames == 0:
        return []
    band = (np.abs(f_axis) > 2000) & (np.abs(f_axis) < cfg.ALLBW/2)
    local_floor = np.median(energy_map[:, band], axis=0) + 1e-30
    if floor_state is None:
        floor = local_floor
    else:
        prev = floor_state.get('floor')
        if prev is None:
            floor = local_floor
        else:
            alpha = 0.3
            floor = alpha*local_floor + (1-alpha)*prev
        floor_state['floor'] = floor
        floor_state['n'] = floor_state.get('n', 0) + 1
    thr = floor*(10**(-30/10.0))
    bidx = np.where(band)[0]
    eb = energy_map[:, bidx]
    eb = np.where(eb >= thr, eb, 0.0)
    K = cfg.PRESCREEN_K
    if eb.shape[1] <= K:
        top_local = np.argsort(eb, axis=1)[:, ::-1]
    else:
        top_local = np.argpartition(eb, -K, axis=1)[:, -K:]      # [n_frames, K]
    top_val = np.take_along_axis(eb, top_local, axis=1)
    frame_idx, k_idx = np.where(top_val > 0)
    top_idx_full = bidx[top_local]
    region_set = set(zip((frame_idx*hop).tolist(),
                         (np.round(f_axis[top_idx_full[frame_idx, k_idx]]/cfg.PRESCREEN_F_ROUND_HZ)
                          *cfg.PRESCREEN_F_ROUND_HZ).tolist()))
    if not region_set:
        return []
    t_cell = max(1, cfg.STAY_HDR//2); f_cell = 800
    buckets = {}
    for t_, f_ in region_set:
        key = (t_ // t_cell, round(f_ / f_cell))
        if key not in buckets:
            buckets[key] = (t_, f_)
    regions = list(buckets.values())

    cut = mf_lpf_hz()
    cut_dc = max(mf_lpf_hz(), cfg.FINE_SYNC_LPF_HZ)
    D = region_decim()
    span = int(cfg.STAY_HDR*0.6)
    hits = []
    def _scan_region(args):
        rt, rf = args
        lo = max(0, int(rt - span//2)); hi = min(len(iq), int(rt + span))
        seg = iq[lo:hi]
        tuned = mix_decimate_filtfilt(
            seg, rf, cut_dc if abs(rf) < cfg.DC_GAP_REACH_HZ else cut, D)
        starts, ratio = _mf_scores(tuned, D)
        if len(starts) == 0:
            return []
        strong = np.where(ratio > cfg.MF_THRESH)[0]
        return [(float(ratio[si]), lo+int(starts[si]), rf) for si in strong]

    # Serial: this used to run the region scan across a thread pool.
    for rgn in regions:
        hits.extend(_scan_region(rgn))
    if not hits:
        return []
    hits.sort(reverse=True)
    pkt_len = cfg.STAY_HDR*cfg.HDR_COUNT + cfg.STAY_DATA*8
    clusters = []
    for c, t0, hf in hits:
        if any(abs(ct-t0) < pkt_len and abs(cf-hf) < 3000 for _, ct, cf in clusters):
            continue
        clusters.append((c, t0, hf))
    return clusters


def _corr_surface(tuned, starts, D=1, aliases=None):
    """Sync-word correlation over the CFO grid at full-rate `starts`, for the
    coarse-frequency aliases listed (default: all); tuned decimated by D
    reads the nearest decimated sample, as _mf_scores does.
    """
    idx = starts[:, None] + cfg.SYNC_OFF[None, :]
    if D > 1:
        idx = np.clip(np.rint((idx - (D - 1)/2)/D).astype(int), 0, len(tuned) - 1)
    Z = (tuned[idx] / cfg.SYNC_VEC[None, :]).T
    aliases = range(len(cfg._COARSE_HZ)) if aliases is None else aliases
    out = np.empty((len(aliases), cfg._NC, len(starts)))
    for k, ci in enumerate(aliases):
        Zc = Z*cfg._COARSE_RAMP[ci][:, None]
        out[k] = np.abs(_ifft(Zc*cfg._SGN, n=cfg._NC, axis=0)*cfg._NC)**2
    return out


def _fine_sync(iq, t0_coarse, hf_coarse, t_span=None, coarse_div=4, topM=6):
    if t_span is None:
        t_span = cfg.STAY_HDR
    guard = int(cfg.EDGE_GUARD_S*cfg.FS)
    s_lo = max(0, t0_coarse - t_span - guard)
    s_hi = min(len(iq), t0_coarse + t_span + cfg.STAY_HDR + guard)
    seg = iq[s_lo:s_hi]
    seg = seg/np.sqrt(np.mean(np.abs(seg)**2) + 1e-30)
    wide = abs(hf_coarse) < cfg.DC_GAP_REACH_HZ
    cut = max(mf_lpf_hz(), cfg.FINE_SYNC_LPF_HZ) if wide else mf_lpf_hz()
    D = region_decim() if cfg.FINE_SYNC_DECIM else 1
    tuned = mix_decimate_filtfilt(seg, hf_coarse, cut, D)
    g = -(-guard//D)
    if g and len(tuned) > 4*g:
        tuned[:g] = 0
        tuned[-g:] = 0
    lim = len(tuned)*D - cfg.SYNC_OFF[-1] - 1
    if lim <= 0:
        return (-1e18, hf_coarse, t0_coarse, 0.0)

    stA = np.arange(0, lim, max(1, cfg.SMBL//coarse_div))
    if len(stA) == 0:
        return (-1e18, hf_coarse, t0_coarse, 0.0)
    S = _corr_surface(tuned, stA, D, aliases=[cfg._COARSE_ONE])
    per_start = S.max(axis=(0, 1))
    top = np.argsort(-per_start)[:topM]
    half = max(1, cfg.SMBL//coarse_div)
    q = max(D, (cfg.SMBL//64)//D*D)
    cand = set()
    for i in top:
        s0 = stA[i]
        cand.update(range(max(0, s0-half), min(lim, s0+half), q))
    stB = np.array(sorted(cand))
    if len(stB) == 0:
        return (-1e18, hf_coarse, t0_coarse, 0.0)
    if q > D:
        S = _corr_surface(tuned, stB, D, aliases=[cfg._COARSE_ONE])
        b = int(stB[np.unravel_index(np.argmax(S), S.shape)[2]])
        stB = np.arange(max(0, b - q + D), min(lim, b + q), D)
    S = _corr_surface(tuned, stB, D)
    ci, fi, si = np.unravel_index(np.argmax(S), S.shape)
    cfo = cfg._CFO_GRID[fi]
    df = -cfo*cfg.BW/(2*np.pi)
    hf = hf_coarse + cfg._COARSE_HZ[ci] + df
    return (S[ci, fi, si], hf, s_lo+int(stB[si]), cfo)


def find_packets_windowed(iq, window_sec=0.1, hop_sec=0.05):
    """Windowed variant of find_packets: scans iq in overlapping chunks instead
    of one whole-array pass.
    """
    min_window = max((cfg.SYNC_OFF[-1]+1)/cfg.FS, cfg.STAY_HDR*0.6/cfg.FS)
    if window_sec < min_window*1.3:
        window_sec = min_window*1.3
    win_samp = int(window_sec*cfg.FS)
    hop_samp = int(hop_sec*cfg.FS)
    all_hits = []
    pos = 0
    while pos < len(iq):
        chunk = iq[pos:pos+win_samp]
        if len(chunk) < cfg.STAY_HDR//4:
            break
        for c, t0, hf in find_packets(chunk):
            all_hits.append((c, pos+t0, hf))
        pos += hop_samp
    if not all_hits:
        return []
    all_hits.sort(reverse=True)
    pkt_len = cfg.STAY_HDR*cfg.HDR_COUNT + cfg.STAY_DATA*8
    t_cell = max(1, pkt_len); f_cell = 3000
    buckets = {}
    deduped = []
    for c, t0, hf in all_hits:
        bt, bf = int(t0)//t_cell, int(hf)//f_cell
        is_dup = False
        for dbt in (-1, 0, 1):
            for dbf in (-1, 0, 1):
                for _, ct, cf in buckets.get((bt+dbt, bf+dbf), ()):
                    if abs(ct-t0) < pkt_len and abs(cf-hf) < 3000:
                        is_dup = True
                        break
                if is_dup:
                    break
            if is_dup:
                break
        if is_dup:
            continue
        deduped.append((c, t0, hf))
        buckets.setdefault((bt, bf), []).append((c, t0, hf))
    return deduped
