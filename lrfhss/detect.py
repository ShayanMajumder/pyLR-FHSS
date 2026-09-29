# Part of the lrfhss receiver package.

import os
import threading
import numpy as np
import scipy.signal as sp

from . import config as cfg
from .dsp import cached_butter, mix_and_filtfilt, notch_spurs
from .fft import _fft, _ifft
from .frontend import load_frontend_windowed


def _cfar_detect(ratio, guard=cfg.CFAR_GUARD, factor=cfg.CFAR_FACTOR,
                 floor=cfg.MF_THRESH_SENSITIVE):
    """BUILT AND TESTED, NOT USED -- kept as a documented negative result.
    find_packets() below still uses the fixed MF_THRESH. Real reason:
    swept CFAR_FACTOR from 1.6 to 2.5, result was FLAT (18 clean clusters,
    10 -10dB clusters, 1/3 recall) across the whole range -- the floor
    term was dominating everywhere, meaning the adaptive part never
    actually engaged. Root cause: at low SNR the entire ratio array sits
    near one degraded level (masking/self-normalization -- textbook CA-
    CFAR failure mode when the reference window is itself
    signal/noise-degraded, not clean background), so cell-averaging
    compares a candidate against neighbors that are ALSO already
    corrupted, which is worse than a single global threshold set once
    from the whole-capture noise floor. On clean data it also fired 7
    extra false triggers a fixed 0.62 threshold correctly rejected,
    costing ~0.5s in extra decode_header_at grid searches for zero
    recall gain (all absorbed by CRC8/CRC16, so still 3/3 correct, just
    slower). CFAR is the textbook-correct approach when the reference
    cells are genuinely independent background; this receiver's regions
    are STFT-prescreened hot spots, which is precisely the kind of
    correlated/non-i.i.d. neighborhood CFAR assumes away.

    Cell-averaging CFAR (constant false alarm rate) detector on the MF
    ratio statistic, replacing a single fixed global threshold.

    Why: MF_THRESH was one hand-tuned number applied uniformly across the
    whole capture. That's the textbook GLRT/matched-filter statistic
    (normalized correlation, amplitude-independent -- see _mf_scores'
    docstring) being compared to a threshold, which IS the
    Neyman-Pearson-optimal detector structure; there is no
    threshold-free alternative in the signal-detection literature --
    every CFAR/GLRT detector (radar, SAR-GMTI, cognitive radio) is
    "compare a statistic to a threshold," full stop. What a FIXED
    threshold gets wrong is using one number everywhere: if background
    correlation level drifts across the capture (different noise
    regions, different local spurious content), a fixed cutoff either
    misses real packets where the floor is elevated or lets through junk
    where it's depressed. CFAR fixes this by estimating the local
    reference level AT EACH CELL from its own neighbors (excluding a
    guard band around the cell itself, so a real peak's own sidelobes
    don't inflate its own reference level) and setting the threshold
    relative to that -- same false-alarm rate everywhere, not a fixed
    score.

    `floor`: a hard minimum on top of the adaptive threshold. Pure CFAR on
    a mostly-flat noise region can still fire on the highest of many
    near-equal noise cells (a real, known CFAR failure mode called
    "masking"/self-normalization when the reference window itself is
    signal-contaminated); the floor prevents that class of false
    detection regardless of local statistics, same purpose the old fixed
    MF_THRESH_SENSITIVE served, just now as a backstop instead of the
    primary gate.
    """
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


PRESCREEN_NFFT = 2048


def mf_lpf_hz():
    """Low-pass cutoff around a prescreen region before the sync matched
    filter and fine sync. MF_LPF_HZ is the floor; above it, the cutoff has
    to cover how far the region's frequency can be off -- half a prescreen
    STFT bin plus half the rounding -- plus ~300 Hz of signal half-width.
    The bin is FS/2048: 81 Hz at 166.7 kHz (DR8/DR9, so the floor rules
    there), but 1465 Hz at 3 MHz (1523/1574 kHz), where a fixed 400 Hz
    cutoff cut the signal away."""
    err = cfg.FS/PRESCREEN_NFFT/2 + cfg.PRESCREEN_F_ROUND_HZ/2
    return max(cfg.MF_LPF_HZ, err + 300.0)


def _mf_scores(tuned):
    """Normalized sync-word matched-filter score per candidate start, maxed
    over CFO via FFT along the sync taps. Range [0,1], amplitude-independent
    (unlike energy, which scales with capture gain and can't be threshold
    reliably across different SNR regions).

    NOTE on filtering: the region loop that feeds this uses sosfiltFILT
    (zero-phase, forward+backward), not a single causal sosfilt pass, even
    though this function only consumes |.| (magnitude). Tested swapping in
    one-pass sosfilt (avoids the backward pass, ~1.85x fewer filter ops):
    MEASURED WRONG -- cluster count changed 11 -> 14 with several false
    positives and altered scores/times on the real capture. The group
    delay from a causal filter shifts sample alignment against SYNC_OFF
    enough to corrupt the correlation, even though the score itself is
    magnitude-normalized. Do not "optimize" this to a single filter pass.
    """
    tstep = cfg.SMBL//2
    last = len(tuned) - cfg.SYNC_OFF[-1] - 1
    if last <= 0:
        return np.array([], dtype=int), np.array([])
    starts = np.arange(0, last, tstep)
    idxmat = starts[:, None] + cfg.SYNC_OFF[None, :]
    S = tuned[idxmat]
    base = S/cfg.SYNC_VEC[None, :]
    energy = np.sqrt(np.sum(np.abs(S)**2, axis=1) + 1e-30)
    F = _fft(base, n=cfg.MF_NCFO, axis=1)
    mfbest = np.max(np.abs(F), axis=1)
    return starts, mfbest/(energy*np.sqrt(cfg.MF_NSYNC))


def find_packets_streaming_interleaved(fn, window_sec=1.0, hop_sec=0.5, queue_maxsize=2,
                                       on_chunk=None, on_window=None):
    """Same reader+assembler threading as find_packets_streaming, but
    instead of collecting all windows' hits into one list and returning
    after the whole capture is read, calls on_window(w_start, w_buf,
    hits, prefix_len) immediately after EACH window's find_packets()
    finishes -- so the caller can decode+print that window's candidates
    right away, while the assembler thread is already reading/decimating
    the NEXT window off disk. on_chunk(chunk) is called once per raw
    decimated+notched chunk as it becomes available, before any window
    that needs it is scanned, so the caller can build up a growing iq
    array in step with what's actually been read.

    This is the actual interleaving requested: load window 1 -> detect ->
    decode+print (while window 2 loads) -> detect window 2 -> decode+print
    (while window 3 loads) -> ... rather than detect-all-windows-then-
    decode-all-clusters as a separate second pass.
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
    full decimated capture in memory. Reader thread streams+decimates the
    WAV via load_frontend_windowed (true read-ahead: seeks/reads window
    N+1 off disk while this consumer processes window N's FFT-decimate +
    STFT scan). This consumer accumulates only a bounded trailing buffer
    -- enough for one scan window plus the STAY_HDR*HDR_COUNT+STAY_DATA*8
    dedup radius (pkt_len) -- and evicts everything older once no pending
    scan window can reference it.

    Carries an EWMA noise-floor estimate (find_packets' floor_state) across
    windows so each window's STFT threshold doesn't re-estimate from
    scratch on a small, noisy sample of frames -- this is what fixes the
    false-positive gap a naive per-window floor produces (see
    find_packets' docstring for the measured before/after).

    Returns the same (score, t0, hf) tuple list as find_packets, with t0
    in the same global sample-index coordinate system as if the whole
    capture had been decimated at once.
    """
    import queue
    min_window = max((cfg.SYNC_OFF[-1]+1)/cfg.FS, cfg.STAY_HDR*0.6/cfg.FS)
    if window_sec < min_window*1.3:
        window_sec = min_window*1.3   # hard floor with margin, never silently
                                        # accept a window too small to work
    win_samp = int(window_sec*cfg.FS)
    hop_samp = int(hop_sec*cfg.FS)
    pkt_len = cfg.STAY_HDR*cfg.HDR_COUNT + cfg.STAY_DATA*8
    margin = int(cfg.STAY_HDR*0.6) + 8192   # span margin + filter transient pad
    keep_back = pkt_len + margin

    buf = np.zeros(0, dtype=complex)
    buf_start = 0   # global sample index of buf[0]
    all_hits = []
    floor_state = {}
    win_idx = 0
    kept = [] if return_iq else None

    # Real background overlap of load vs detection, not just load vs FFT.
    # The old version's `for chunk in load_frontend_windowed(...)` only let
    # the reader thread read the NEXT NB-sized chunk ahead while THIS
    # chunk's FFT-decimate ran -- but find_packets() (STFT + MF region
    # scan, the actually expensive part) still ran synchronously in this
    # same loop, blocking the generator from being asked for its next item
    # until detection finished. So detection on window N and disk-read of
    # window N+1 never overlapped -- confirmed by inspection, this was the
    # literal gap reported.
    #
    # Fix: run the load_frontend_windowed loop (reader thread + FFT-decim)
    # on its OWN background thread, which assembles scan-window-sized
    # buffers and pushes them onto a second bounded queue. This (the
    # caller's) thread pulls completed windows from that queue and runs
    # find_packets -- so while find_packets is busy on window N, the
    # assembler thread is already reading+decimating window N+1 off disk,
    # genuinely in parallel.
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
    # O(n) spatial-bucket dedup instead of O(n^2) linear scan against the
    # growing deduped list. Standard competitive-programming technique for
    # "is there a point within radius R of this one" queries: bucket space
    # into cells of size >= the radius, so any true neighbor must be in
    # one of the 3x3 (here 2D: time x freq) surrounding cells -- never
    # need to check points outside that neighborhood, since they're
    # provably farther than the dedup radius. Same pattern already used
    # for find_packets' region-list dedup (t_cell/f_cell below, mirrored).
    # At 122 candidates (a real -10dB run) this was ~14,884 comparisons
    # worst case; buckets bound it to a small constant per candidate
    # regardless of n.
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

    Why: the matched filter alone, swept over the whole capture x every
    freq bin, is correct but does a full-length filtfilt per freq bin
    (~90 bins x full capture) -- most of that work is wasted since real
    packets occupy a tiny fraction of (t,f) space. STFT energy is cheap
    (~0.2s) and only needs to not miss real bursts, which a loose
    threshold (-30dB, keep top-K=6 bins per frame so co-located packets
    at different freqs aren't lost to a single-peak-per-frame pick)
    guarantees in practice. The matched filter still does the real
    accept/reject -- this stage only prunes where to look, never what to
    accept. Confirmed zero recall loss vs the full sweep, ~5-7x faster.

    floor_state: optional dict {'floor': array or None, 'n': int} for
    carrying an EWMA noise-floor estimate ACROSS calls (used by the
    windowed/streaming scan below). Without this, each windowed call's
    np.median(energy_map, axis=0) is computed over only that window's own
    STFT frames -- far fewer than the whole capture -- which measurably
    produces a noisier, less stable floor and more false-positive region
    picks (confirmed: naive per-window floor gave 13 clusters where the
    whole-array floor gives 11, some of them false positives). Passing the
    same dict across successive windowed calls lets the floor converge via
    EWMA (alpha=0.3, first call seeds it directly) instead of each window
    re-estimating from scratch -- this is what actually fixes the
    false-positive gap while still never holding the whole capture in
    memory, since the carried state is a handful of floats, not the
    signal.
    """
    nfft = PRESCREEN_NFFT; hop = max(1, cfg.STAY_HDR//8)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f_axis_raw, _, Zxx = sp.stft(iq, fs=cfg.FS, window='hann', nperseg=nfft,
                                     noverlap=nfft-hop, nfft=nfft, boundary=None)
    f_axis = np.fft.fftshift(f_axis_raw)
    # |z|**2 without the round-trip sqrt: np.abs() takes a sqrt that the
    # **2 immediately undoes, over the full [nfft, n_frames] STFT array.
    # real**2 + imag**2 is the same quantity computed directly. Also skips
    # the fftshift on Zxx itself (a full array copy) by shifting the axis
    # once and reordering rows with the same permutation, then transposing.
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
    # Work only on the in-band columns instead of copying the full
    # [n_frames, nfft] array and zeroing/masking out-of-band columns in
    # place. Out-of-band entries were forced to 0 and so could never be
    # picked by the top-K anyway, so restricting up front is equivalent
    # and does strictly less work (smaller copy, smaller argpartition).
    # Indices are mapped back to full-spectrum bins via bidx.
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
    # O(n) grid-bucket dedup (was O(n^2) any()-scan against a growing list --
    # at ~1200 raw region points that quadratic scan alone cost ~2s). Snap to
    # a coarse (time, freq) grid matching the merge radius used before
    # (STAY_HDR//2, 800Hz) and keep the strongest-looking point per bucket.
    t_cell = max(1, cfg.STAY_HDR//2); f_cell = 800
    buckets = {}
    for t_, f_ in region_set:
        key = (t_ // t_cell, round(f_ / f_cell))
        if key not in buckets:
            buckets[key] = (t_, f_)
    regions = list(buckets.values())

    sos = cached_butter(4, mf_lpf_hz()/(cfg.FS/2))
    sos_dc = cached_butter(4, max(mf_lpf_hz(), cfg.FINE_SYNC_LPF_HZ)/(cfg.FS/2))
    span = int(cfg.STAY_HDR*0.6)
    hits = []
    # Region scan is embarrassingly parallel -- each region is an
    # independent mix + filtfilt + MF-score on its own slice of iq. This
    # only became worth threading once sosfiltfilt_ext started releasing
    # the GIL (before that every worker serialized on it and threading
    # measured zero gain); numpy's cos/sin ufuncs in _tone release the GIL
    # too, so the whole per-region body now genuinely overlaps.
    # Results are collected per-region and merged in deterministic region
    # order afterwards, so output is identical to the serial version
    # regardless of completion order.
    def _scan_region(args):
        rt, rf = args
        lo = max(0, int(rt - span//2)); hi = min(len(iq), int(rt + span))
        seg = iq[lo:hi]
        # Near DC the prescreen places no regions (|f| < 2 kHz is skipped for
        # the SDR's DC spike), so a hop there is only reachable from a region
        # at the edge of that gap -- which takes the original wide filter.
        # Narrow everywhere else. (Real capture: case077, header hop at
        # 493 Hz, found only through the region at 2050 Hz.)
        tuned = mix_and_filtfilt(sos_dc if abs(rf) < cfg.DC_GAP_REACH_HZ else sos,
                                 seg, rf)
        starts, ratio = _mf_scores(tuned)
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


def _corr_surface(tuned, starts):
    Z = (tuned[starts[:, None] + cfg.SYNC_OFF[None, :]] / cfg.SYNC_VEC[None, :]).T
    out = np.empty((len(cfg._COARSE_HZ), cfg._NC, len(starts)))
    for ci in range(len(cfg._COARSE_HZ)):
        Zc = Z*cfg._COARSE_RAMP[ci][:, None]
        out[ci] = np.abs(_ifft(Zc*cfg._SGN, n=cfg._NC, axis=0)*cfg._NC)**2
    return out


def _fine_sync(iq, t0_coarse, hf_coarse, t_span=None, coarse_div=4, topM=6):
    if t_span is None:
        t_span = cfg.STAY_HDR
    s_lo = max(0, t0_coarse - t_span)
    s_hi = min(len(iq), t0_coarse + t_span + cfg.STAY_HDR)
    seg = iq[s_lo:s_hi]
    seg = seg/np.sqrt(np.mean(np.abs(seg)**2) + 1e-30)
    # Narrow like the matched filter (it is a large part of the low-SNR
    # gain: restoring 3000 Hz everywhere cut DR8 at -22 dB from 47/48 to
    # 20/48), except next to the DC gap, where the candidate sits at the
    # gap's edge and the hop may be up to 2 kHz away (case077: 2050 Hz
    # candidate, 493 Hz hop).
    wide = abs(hf_coarse) < cfg.DC_GAP_REACH_HZ
    sos = cached_butter(4, (max(mf_lpf_hz(), cfg.FINE_SYNC_LPF_HZ) if wide
                            else mf_lpf_hz())/(cfg.FS/2))
    tuned = mix_and_filtfilt(sos, seg, hf_coarse)
    lim = len(tuned)-cfg.SYNC_OFF[-1]-1
    if lim <= 0:
        return (-1e18, hf_coarse, t0_coarse, 0.0)

    stA = np.arange(0, lim, max(1, cfg.SMBL//coarse_div))
    if len(stA) == 0:
        return (-1e18, hf_coarse, t0_coarse, 0.0)
    S = _corr_surface(tuned, stA)
    per_start = S.max(axis=(0, 1))
    top = np.argsort(-per_start)[:topM]
    half = max(1, cfg.SMBL//coarse_div)
    cand = set()
    for i in top:
        s0 = stA[i]
        cand.update(range(max(0, s0-half), min(lim, s0+half)))
    stB = np.array(sorted(cand))
    if len(stB) == 0:
        return (-1e18, hf_coarse, t0_coarse, 0.0)

    S = _corr_surface(tuned, stB)
    ci, fi, si = np.unravel_index(np.argmax(S), S.shape)
    cfo = cfg._CFO_GRID[fi]
    df = -cfo*cfg.BW/(2*np.pi)
    hf = hf_coarse + cfg._COARSE_HZ[ci] + df
    return (S[ci, fi, si], hf, s_lo+int(stB[si]), cfo)


def find_packets_windowed(iq, window_sec=0.1, hop_sec=0.05):
    """Windowed variant of find_packets: scans iq in overlapping chunks
    instead of one whole-array pass. window_sec must be large enough to
    contain a full sync word (SYNC_OFF[-1]+1 samples, ~0.0635s at this
    FS/SMBL) -- a window smaller than that can never see enough of the sync
    pattern to correlate against it, regardless of threshold. Default
    window=0.1s/hop=0.05s keeps 50% overlap (so a sync word straddling a
    chunk boundary is still fully contained in the NEXT chunk) while
    actually being big enough to work. Returns the same (score, t0, hf)
    tuple list as find_packets, deduped across chunk boundaries.
    """
    min_window = max((cfg.SYNC_OFF[-1]+1)/cfg.FS, cfg.STAY_HDR*0.6/cfg.FS)
    if window_sec < min_window*1.3:
        window_sec = min_window*1.3   # hard floor with margin, never silently
                                        # accept a window too small to work
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
    # Same O(n) spatial-bucket dedup as find_packets_streaming (see that
    # function's comment for the full rationale) -- this was the same
    # O(n^2) any()-over-growing-list pattern, same fix.
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
