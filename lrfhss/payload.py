# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT

import time
import numpy as np

from . import config as cfg
from .phy.hopping import calculate_freq_from_hop_seq_id
from .phy.fec import bits_to_bytes, deinterleave_payload, dewhiten_payload, viterbi_decode_payload
from .phy.gmsk import demod_symbols, msk_trellis_llr
from .dsp import cached_butter, hann, mix_and_filtfilt
from .fft import _fft
from .header import _valid_bw, detect_replica_index


def _packet_footprint(hdr, hdr_win_start, hdr_f_measured):
    payloadlen = hdr['payloadlen']; CR = hdr['CR']
    payload_length_bits = 8*(payloadlen+2)+6
    data_in_bitcount = int(np.ceil(payload_length_bits*[6/5, 3/2, 2, 3][CR]))
    num_frags = int(np.ceil(data_in_bitcount/48))
    if not _valid_bw(hdr['BW']):
        return None
    step = 0.95367431640625
    pll = calculate_freq_from_hop_seq_id(hdr['grid'], hdr['hop'], cfg.HDR_COUNT,
                                         hdr['BW'], hdr['hopseq'], num_frags)
    t_lo = hdr_win_start - cfg.STAY_HDR*cfg.HDR_COUNT
    t_hi = hdr_win_start + cfg.STAY_HDR*cfg.HDR_COUNT + cfg.STAY_DATA*num_frags
    if pll is None:
        return None
    freqs = []
    for k in range(1, cfg.HDR_COUNT+1):
        addval = hdr_f_measured - pll[k-1]*step
        freqs.extend([p*step + addval for p in pll])
    return (t_lo, t_hi, freqs)


def _chase_payload_fallback(soft_stream, data_in_bitcount, CR, max_single=10, max_pairs=10):
    """Chase-style bit-flip retry, last resort after the dcfo/gsto grid finds
    no CRC16 pass.
    """
    if soft_stream is None:
        return None, False
    deint = deinterleave_payload(soft_stream, data_in_bitcount)
    order = np.argsort(np.abs(deint))
    flip_candidates = order[:max_single]
    for p in flip_candidates:
        cand = deint.copy(); cand[p] = -cand[p]
        info, match = viterbi_decode_payload(cand, CR=CR)
        if match:
            return info, True
    n_pair = min(max_pairs, len(flip_candidates))
    for i in range(n_pair):
        for j in range(i+1, n_pair):
            cand = deint.copy()
            cand[flip_candidates[i]] = -cand[flip_candidates[i]]
            cand[flip_candidates[j]] = -cand[flip_candidates[j]]
            info, match = viterbi_decode_payload(cand, CR=CR)
            if match:
                return info, True
    return None, False


def _common_fragment_cfo(segs, predicted_hz, max_off_hz=350.0):
    """Shared carrier offset of the payload fragments, from all of them at once.
    """
    rs = cfg.SYMBOL_RATE_HZ
    sos = cached_butter(4, (max_off_hz + rs)/(cfg.FS/2))
    lens = [len(s) for s, _ in zip(segs, predicted_hz) if len(s) >= cfg.SMBL*3]
    if not lens:
        return None
    nfft = 1 << int(np.ceil(np.log2(max(lens)*4)))
    acc = np.zeros(nfft)
    for seg, f0 in zip(segs, predicted_hz):
        if len(seg) < cfg.SMBL*3:
            continue
        b = mix_and_filtfilt(sos, seg, f0)
        acc += np.abs(np.fft.fft(b*b, nfft))**2
    ff = np.fft.fftfreq(nfft, 1/cfg.FS)
    order = np.argsort(ff)
    ff, acc = ff[order], acc[order]
    two_off = np.arange(-2*max_off_hz, 2*max_off_hz + 1e-9, 1.0)
    score = (np.interp(two_off - rs/2, ff, acc) +
             np.interp(two_off + rs/2, ff, acc))
    return float(two_off[np.argmax(score)]/2)


def _fragment_weights(seg_set, half_bw_hz=300.0):
    """Per-fragment reliability in [0, 1], for erasure-aware soft decoding."""
    w = []
    for seg, fc, _ in seg_set:
        if len(seg) < cfg.SMBL*3:
            w.append(0.0)
            continue
        S = np.abs(_fft(seg*hann(len(seg))))**2
        ff = np.fft.fftfreq(len(seg), 1/cfg.FS)
        band = np.abs(ff - fc) <= half_bw_hz
        noise = np.median(S) + 1e-30
        snr = max(S[band].sum() - band.sum()*noise, 0.0)/(band.sum()*noise)
        w.append(snr/(1.0 + snr))
    return np.array(w)


def _trellis_payload(seg_set, wts, data_in_bitcount, CR, sos):
    """Payload decode with the phase-tracking trellis demodulator
    (phy.gmsk.msk_trellis_llr), ~2.5 dB beyond the differential one.
    """
    spb = cfg.FS/cfg.SYMBOL_RATE_HZ
    dcfos = list(cfg.TRELLIS_DCFO_HZ)
    gstos = [0, -cfg.LOOKDIST, int(round(cfg.SMBL/3)), int(round(2*cfg.SMBL/3))]
    hyps = [(d, g) for d in dcfos for g in gstos]
    per_frag = []
    for fi, (seg, fc, nbits) in enumerate(seg_set):
        if len(seg) < cfg.SMBL*3:
            per_frag.append(np.zeros((len(hyps), nbits)))
            continue
        base = cfg.LOOKDIST + np.round(np.arange(nbits + 1)*spb).astype(int)
        if cfg.FILTER_ONCE:
            tuned = mix_and_filtfilt(sos, seg, fc)
            llr = msk_trellis_llr([(tuned, g + base, d) for d, g in hyps], first_bit=0)
        else:
            tuned = {d: mix_and_filtfilt(sos, seg, fc + d) for d in dcfos}
            llr = msk_trellis_llr([(tuned[d], g + base) for d, g in hyps], first_bit=0)
        per_frag.append(llr*wts[fi])
    for h in range(len(hyps)):
        s = np.concatenate([pf[h] for pf in per_frag])[:data_in_bitcount]
        scale = 2*np.median(np.abs(s)) + 1e-12
        soft = np.clip(s/scale, -1, 1)
        info, match = viterbi_decode_payload(
            deinterleave_payload(soft, data_in_bitcount), CR=CR)
        if match:
            return info, True
    return None


def _packet_slots(iq, hdr, hdr_win_start, hdr_f_measured):
    """Structured per-slot layout for spectrogram annotation: returns a list of
    dicts, one per header replica and payload fragment, each with
    (t_start_sample, t_end_sample, freq_hz, label).
    """
    payloadlen = hdr['payloadlen']; CR = hdr['CR']
    payload_length_bits = 8*(payloadlen+2)+6
    data_in_bitcount = int(np.ceil(payload_length_bits*[6/5, 3/2, 2, 3][CR]))
    num_frags = int(np.ceil(data_in_bitcount/48))
    if not _valid_bw(hdr['BW']):
        return None
    step = 0.95367431640625
    pll = calculate_freq_from_hop_seq_id(hdr['grid'], hdr['hop'], cfg.HDR_COUNT,
                                         hdr['BW'], hdr['hopseq'], num_frags)
    if pll is None:
        return None
    thishdridx = detect_replica_index(iq, hdr, hdr_win_start, hdr_f_measured)
    replica1_start = hdr_win_start - (thishdridx-1)*cfg.STAY_HDR
    addval = hdr_f_measured - pll[thishdridx-1]*step
    slots = []
    for k in range(cfg.HDR_COUNT):
        t0 = replica1_start + k*cfg.STAY_HDR
        slots.append(dict(t_start=t0, t_end=t0+cfg.STAY_HDR,
                          freq=pll[k]*step + addval, label='header %d/%d' % (k+1, cfg.HDR_COUNT)))
    pay_start = replica1_start + cfg.STAY_HDR*cfg.HDR_COUNT
    for fi in range(num_frags):
        t0 = pay_start + fi*cfg.STAY_DATA
        slots.append(dict(t_start=t0, t_end=t0+cfg.STAY_DATA,
                          freq=pll[cfg.HDR_COUNT+fi]*step + addval,
                          label='payload %d/%d' % (fi+1, num_frags)))
    return slots


def decode_payload_at(iq, hdr, hdr_win_start, hdr_f_measured):
    payloadlen = hdr['payloadlen']; CR = hdr['CR']
    payload_length_bits = 8*(payloadlen+2)+6
    data_in_bitcount = int(np.ceil(payload_length_bits*[6/5, 3/2, 2, 3][CR]))
    num_frags = int(np.ceil(data_in_bitcount/48))
    if not _valid_bw(hdr['BW']):
        return None, False
    step = 0.95367431640625
    pll = calculate_freq_from_hop_seq_id(hdr['grid'], hdr['hop'], cfg.HDR_COUNT,
                                         hdr['BW'], hdr['hopseq'], num_frags)
    sos = cached_butter(4, 300/(cfg.FS/2))   # was 500Hz, same rationale as header decode

    t_budget_start = time.monotonic()

    def try_thishdridx(thishdridx):
        if pll is None:
            return None
        dwells_to_payload = cfg.HDR_COUNT - thishdridx + 1
        pay_start = hdr_win_start + cfg.STAY_HDR*dwells_to_payload
        if pll is not None:
            addval = hdr_f_measured - pll[thishdridx-1]*step
            predicted_hop_hz = [p*step + addval for p in pll[cfg.HDR_COUNT:]]
        else:
            predicted_hop_hz = None
        segs = []
        ek_ratios = []
        for fi in range(num_frags):
            fs0 = pay_start + fi*cfg.STAY_DATA
            seg_len = cfg.STAY_DATA + int(cfg.SMBL*4)
            seg = iq[fs0 : fs0 + seg_len]
            if len(seg) < cfg.SMBL*3:
                nbits_ph = 48 if fi < num_frags-1 else (data_in_bitcount - 48*(num_frags-1))
                segs.append((seg, 0.0, nbits_ph))
                continue
            if predicted_hop_hz is not None and fi < len(predicted_hop_hz):
                f0 = predicted_hop_hz[fi]
                ff = np.fft.fftfreq(len(seg), 1/cfg.FS)
                bandm = (np.abs(ff - f0) < 600)
                S = np.abs(_fft(seg*hann(len(seg))))
                S[~bandm] = 0
                peak_val = np.max(S)
                noise_floor = np.median(np.abs(_fft(seg))) + 1e-12
                if peak_val > noise_floor * 5.0:
                    fc = ff[np.argmax(S)]
                else:
                    fc = f0
                ek_ratios.append(peak_val**2/noise_floor**2)
            else:
                f = np.fft.fftfreq(len(seg), 1/cfg.FS)
                band = (np.abs(f) > 2e3) & (np.abs(f) < cfg.ALLBW/2)
                S = np.abs(_fft(seg*hann(len(seg))))
                S[~band] = 0
                fc = f[np.argmax(S)]
            nbits = 48 if fi < num_frags-1 else (data_in_bitcount - 48*(num_frags-1))
            segs.append((seg, fc, nbits))
        best_stream = [None, -1.0]
        seg_sets = []
        n_pred = 0
        if predicted_hop_hz is not None and len(predicted_hop_hz) >= len(segs):
            seg_sets.append([(seg, predicted_hop_hz[fi], nbits)
                             for fi, (seg, _, nbits) in enumerate(segs)])
            off = _common_fragment_cfo([s for s, _, _ in segs],
                                       predicted_hop_hz[:len(segs)])
            if off is not None:
                seg_sets.append([(seg, predicted_hop_hz[fi] + off, nbits)
                                 for fi, (seg, _, nbits) in enumerate(segs)])
            n_pred = len(seg_sets)
        seg_sets.append(segs)
        for si, seg_set in enumerate(seg_sets):
            wts = _fragment_weights(seg_set)
            wts = wts/wts.max() if wts.max() > 0 else np.ones(len(seg_set))
            if cfg.PAYLOAD_TRELLIS and si < n_pred:
                res = _trellis_payload(seg_set, wts, data_in_bitcount, CR, sos)
                if res is not None:
                    return res
            if not cfg.PAYLOAD_OLD_FALLBACK:
                continue
            for gsto_step in cfg.GSTO_TIERS:
                for dcfo in range(-30, 31, 5):
                  if time.monotonic() - t_budget_start > cfg.PAYLOAD_TIME_BUDGET_S:
                      return _chase_payload_fallback(best_stream[0], data_in_bitcount, CR)
                  tfs = []
                  for seg, fc, nbits in seg_set:
                      if len(seg) < cfg.SMBL*3:
                          tfs.append((None, nbits))
                          continue
                      tf = mix_and_filtfilt(sos, seg, fc+dcfo)
                      tfs.append((tf, nbits))
                  gsto_list = np.arange(0, cfg.SMBL, gsto_step, dtype=np.int64)
                  frag_grids = []
                  for tf, nbits in tfs:
                      if tf is None:
                          frag_grids.append((None, nbits))
                          continue
                      if cfg._HAVE_VEXT_LOCAL:
                          grid, gvalid = cfg._vext_local.demod_symbols_grid(
                              np.ascontiguousarray(tf, dtype=np.complex128),
                              gsto_list, cfg.LOOKDIST, cfg.PHASESLOPE, int(nbits), int(cfg.SMBL),
                              True, 1.0)
                          frag_grids.append((np.asarray(grid), nbits))
                      else:
                          frag_grids.append((None, nbits))   # fall back per-gsto below

                  for gi, gsto in enumerate(gsto_list):
                      if time.monotonic() - t_budget_start > cfg.PAYLOAD_TIME_BUDGET_S:
                          return _chase_payload_fallback(best_stream[0], data_in_bitcount, CR)
                      parts = []
                      for fi, ((grid, nbits), (tf, _nb)) in enumerate(zip(frag_grids, tfs)):
                          if grid is not None:
                              parts.append(grid[gi]*wts[fi])
                              continue
                          if tf is None:
                              parts.append(np.zeros(nbits, dtype=float))
                              continue
                          centers = gsto + cfg.LOOKDIST + np.arange(nbits+2)*cfg.SMBL
                          centers_valid = centers[centers < len(tf)-cfg.LOOKDIST]
                          if len(centers_valid) < nbits+1:
                              parts.append(np.zeros(nbits, dtype=float))
                              continue
                          sb = demod_symbols(tf, centers_valid, cfg.LOOKDIST, cfg.PHASESLOPE)
                          parts.append(sb[1:1+nbits]*wts[fi])
                      soft_stream = np.concatenate(parts)[:data_in_bitcount]
                      deint = deinterleave_payload(soft_stream, data_in_bitcount)
                      info, match = viterbi_decode_payload(deint, CR=CR)
                      if match:
                          return info, True
                      bim = np.mean(np.abs(soft_stream) > 0.5)
                      if bim > best_stream[1]:
                          best_stream[0] = soft_stream; best_stream[1] = bim
        return _chase_payload_fallback(best_stream[0], data_in_bitcount, CR)

    order = [1, 2, 3]
    for k in order:
        if pll is not None:
            addval_check = hdr_f_measured - pll[k-1]*step
            predicted_check = [p*step + addval_check for p in pll[cfg.HDR_COUNT:]]
            if any(abs(p) > cfg.ALLBW/2 + cfg.HOP_EDGE_TOL_HZ for p in predicted_check):
                continue
        res = try_thishdridx(k)
        if res is None:
            continue
        info, match = res
        if match:
            if len(info) >= payloadlen*8:
                return bits_to_bytes(dewhiten_payload(info[:payloadlen*8], payloadlen)).tolist(), True
    return None, False
