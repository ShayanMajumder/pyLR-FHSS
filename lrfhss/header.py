# Part of the lrfhss receiver package.

import numpy as np

from . import config as cfg
from .phy.hopping import calculate_freq_from_hop_seq_id
from .phy.gmsk import demod_symbols
from .phy.framing import decode_header, deinterleave_header
from .dsp import cached_butter, mix_and_filtfilt
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
    win = cfg._WIN_HDR if len(seg) == cfg.STAY_HDR else np.hanning(len(seg))
    S = np.abs(_fft(seg*win)); S[~band] = 0
    return f[np.argmax(S)]


def _cfo_tone(cfo_val, smbl, n_seg):
    """exp(-1j*cfo_val/smbl*n_seg), via cos/sin into preallocated real/imag
    instead of np.exp(1j*ph). Same technique as _tone (1.66x measured at
    header-segment length, 0.0 diff vs np.exp including the cfo=0 case).
    decode_header_at calls this 3x per df (cfo, 0.0, -cfo) x 11 df values
    per cluster; profiled as ~2.08ms of a ~2.1ms 3-call block before this
    fix (the demod calls themselves cost 0.02ms combined -- the mixing
    was the real cost, not the demod)."""
    ph = (-float(cfo_val)/smbl)*n_seg
    t = np.empty(len(n_seg), dtype=np.complex128)
    np.cos(ph, out=t.real)
    np.sin(ph, out=t.imag)
    return t


def decode_header_at(iq, hdr_f, sync_start_abs, cfo):
    win_start = sync_start_abs - cfg.SYNC_START_BIT*cfg.SMBL
    if win_start < 0:
        win_start = sync_start_abs
    seg = iq[win_start:win_start + cfg.HDR_BIT_NUM*cfg.SMBL + 4*cfg.SMBL]
    sos = cached_butter(4, 300/(cfg.FS/2))   # was 500Hz; 300Hz still inside GMSK signal
                                       # bandwidth (clean-SNR decode confirmed
                                       # down to ~250-350Hz before distortion),
                                       # tighter noise rejection pushed header
                                       # decode from -14dB to -18dB in AWGN test.
    fseg = np.fft.fftfreq(len(seg), 1/cfg.FS)
    bandm = (np.abs(fseg-hdr_f) < 4000)
    S = np.abs(_fft(seg*np.hanning(len(seg)))); S[~bandm] = 0
    hdr_f_precise = fseg[np.argmax(S)]
    n_seg = np.arange(len(seg))
    centers_base = cfg.LOOKDIST + np.arange(cfg.HDR_BIT_NUM)*cfg.SMBL
    best = (None, -1, None)
    cached = []   # (deint_soft, soft, df) per combo, for the backward retry pass
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
            hdr = decode_header(deint)   # forward-only: keeps this pass's
                                          # false-positive surface identical
                                          # to before backward search existed
            if hdr is not None:
                if not (1 <= hdr['payloadlen'] <= 100) or not _valid_bw(hdr['BW']):
                    bim = np.mean(np.abs(soft) > 0.5)
                    if bim > best[1]:
                        best = (soft, bim, df)
                    continue
                # NOTE: field-mismatch is no longer printed here as a warning.
                # CRC8 alone (8 bits, and this loop tries up to 33 hdr search
                # positions x 16 Viterbi start-states = 528 checks/cluster)
                # gives ~528/256 expected false passes per cluster -- field
                # mismatch on a CRC8 pass is common and NOT evidence of a real
                # packet either way. Payload CRC16 (65536x stronger) is the
                # actual gate; mismatch bookkeeping happens there, not here.
                return hdr, soft, win_start, hdr_f_precise+df
            bim = np.mean(np.abs(soft) > 0.5)
            if bim > best[1]:
                best = (soft, bim, df)
    # Forward found nothing anywhere in the grid. SOVA-style backward retry
    # (Jung et al.): recovers real cases forward misses (validated +80% at
    # matched noise on synthetic trials), tried here ONLY as a fallback over
    # the already-demodulated combos, so it can't shadow a forward-correct
    # answer with a backward false positive found earlier in iteration order.
    for deint, soft, df in cached:
        hdr = decode_header(deint, try_backward=True)
        if hdr is not None and 1 <= hdr['payloadlen'] <= 100 and _valid_bw(hdr['BW']):
            return hdr, soft, win_start, hdr_f_precise+df
    fallback_soft = best[0] if best[0] is not None else np.array([])
    fallback_f = hdr_f_precise + (best[2] if best[2] is not None else 0)
    return None, fallback_soft, win_start, fallback_f


def detect_replica_index(iq, hdr, hwin, hfp):
    """Find which header replica (1-based, out of HDR_COUNT) was actually
    locked at hwin, by hypothesis-testing each candidate index against real
    signal energy at its LFSR-predicted frequencies.

    Why hypothesis-testing instead of STAY_HDR-spacing + hopseq matching: a
    prior version scanned outward at STAY_HDR spacing and tried decoding a
    header there, at the SAME frequency as the lock (hf_coarse unchanged).
    That's wrong: each header replica hops to a DIFFERENT frequency by
    design (that's the whole point of frequency-hopped repetition), so this
    can only ever re-confirm the one frequency already known (the lock
    itself) and will drift-reject or find nothing at the other replicas'
    real (different) frequencies -- silently defaulting to a wrong answer.

    This version breaks that circularity properly: for each candidate index
    c in 1..HDR_COUNT, compute what the LFSR predicts for ALL HDR_COUNT
    replica frequencies under the hypothesis "the lock is replica c" (uses
    calculate_freq_from_hop_seq_id, the same math decode_payload_at uses),
    then measures real FFT energy at each predicted frequency's dwell. The
    correct hypothesis lights up all HDR_COUNT slots; wrong hypotheses only
    light up the one frequency already known to be real (the lock) and read
    noise floor everywhere else. No brute-force header re-decode needed,
    just energy checks -- cheaper and more reliable than the prior approach.
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
