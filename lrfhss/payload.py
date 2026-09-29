# Part of the lrfhss receiver package.

import time
import numpy as np

from . import config as cfg
from .phy.hopping import calculate_freq_from_hop_seq_id
from .phy.fec import bits_to_bytes, deinterleave_payload, dewhiten_payload, viterbi_decode_payload
from .phy.gmsk import demod_symbols, msk_trellis_llr
from .dsp import cached_butter, mix_and_filtfilt
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
    no CRC16 pass. Ported concept from the LoRa Chase decoder (chase.py):
    rank bits by reliability (here, |soft value|, low = least confident),
    flip the least-reliable ones single then in pairs, retry decode. CRC16
    gates every attempt so a wrong flip can only fail, never falsely pass.

    Validated on synthetic near-margin failures (noise level tuned so the
    raw Viterbi decode fails ~5% of the time): recovers ~12% of those
    failures at ~175ms total for 10 single + 10-choose-2 pair attempts.
    Modest, not a silver bullet -- GMSK phase-slope soft values don't
    concentrate errors as cleanly as LoRa's per-symbol FFT peak ratio does,
    so this is worth attempting only as a bounded last resort, not a
    replacement for the grid search."""
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

    Each fragment's carrier is its LFSR-predicted hop plus ONE offset that
    every fragment shares (the header-frequency measurement error -- up to
    ~240 Hz on synthetic packets, identical across fragments). Estimating
    it per fragment from the FFT peak does not work: a GMSK burst's
    spectral peak moves with its bit content, by -140..+120 Hz
    independently per fragment, which no shared +/-30 Hz dcfo nudge can
    undo. CR 1/3 absorbs the one fragment that lands badly; CR 2/3 (DR9)
    cannot, which is what capped DR9 at ~80% even at high SNR.

    Squaring removes the data: with h = 1/2 the squared signal is h = 1
    CPFSK, which carries discrete lines at 2*off +/- Rs/2 whatever the
    bits are. Each fragment is mixed to its predicted hop, low-passed,
    squared; the power spectra are summed across fragments, and the offset
    is where the PAIR of lines is strongest. Returns Hz, or None if there
    is nothing to estimate from.
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
    """Per-fragment reliability in [0, 1], for erasure-aware soft decoding.

    The demodulator turns ANY input into soft bits: a fragment that was
    wiped out (collision, fade, jammer, end of capture) still yields
    phase slopes, and noise slopes clip to +/-1 as often as real bits do.
    Fed to Viterbi at full weight, that is confident garbage -- the reason
    the receiver needed 5 of 7 DR8 fragments when CR 1/3 can in principle
    recover from 3.

    Weight = SNR/(1+SNR), the usual LLR scaling, with SNR measured as the
    excess power within +/-half_bw_hz of the fragment's carrier over the
    capture's median per-bin noise. Under plain AWGN every fragment gets
    about the same weight, and a uniform scale leaves Viterbi's decision
    unchanged; an empty fragment gets ~0 and becomes an erasure.
    """
    w = []
    for seg, fc, _ in seg_set:
        if len(seg) < cfg.SMBL*3:
            w.append(0.0)
            continue
        S = np.abs(_fft(seg*np.hanning(len(seg))))**2
        ff = np.fft.fftfreq(len(seg), 1/cfg.FS)
        band = np.abs(ff - fc) <= half_bw_hz
        noise = np.median(S) + 1e-30
        snr = max(S[band].sum() - band.sum()*noise, 0.0)/(band.sum()*noise)
        w.append(snr/(1.0 + snr))
    return np.array(w)


def _trellis_payload(seg_set, wts, data_in_bitcount, CR, sos):
    """Payload decode with the phase-tracking trellis demodulator
    (phy.gmsk.msk_trellis_llr), ~2.5 dB beyond the differential one.

    It tolerates +/-0.23 symbol of timing error but only ~+/-5-10 Hz of
    frequency error, so the shared (dcfo, gsto) search runs a fine
    frequency grid and a coarse timing one: TRELLIS_DCFO_HZ x 3 timings,
    all hypotheses for a fragment in one batched trellis call. CRC16
    gates every candidate, as in the rest of the search.
    """
    spb = cfg.FS/cfg.SYMBOL_RATE_HZ
    dcfos = list(cfg.TRELLIS_DCFO_HZ)
    gstos = [0, int(round(cfg.SMBL/3)), int(round(2*cfg.SMBL/3))]
    hyps = [(d, g) for d in dcfos for g in gstos]
    per_frag = []
    for fi, (seg, fc, nbits) in enumerate(seg_set):
        if len(seg) < cfg.SMBL*3:
            per_frag.append(np.zeros((len(hyps), nbits)))
            continue
        # Tune and filter once; each dcfo is applied by the trellis as a
        # rotation of just the samples it uses.
        # mids[0] is the fragment's leading pad bit (a known 0)
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
    """Structured per-slot layout for spectrogram annotation: returns a list
    of dicts, one per header replica and payload fragment, each with
    (t_start_sample, t_end_sample, freq_hz, label). Uses detect_replica_index
    to find which replica hdr_win_start actually is (the matched-filter
    detector does NOT reliably lock replica 1 -- confirmed empirically it
    almost always locks the middle replica), then anchors all boxes at the
    true replica-1 position so the drawn layout matches the real signal
    instead of assuming the lock is always the first replica.
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

    # Wall-clock budget for the dcfo/gsto search. Combo count alone doesn't
    # bound wall time -- Viterbi cost scales with num_frags (payload size),
    # so a garbage/false-CRC8-pass header with a large payloadlen can make
    # the SAME 570-combo grid cost 60s+ instead of ~1s. Measured worst case:
    # payloadlen=100,CR=0 garbage header -> 64.57s before this fix, all of
    # it wasted since there's no real payload there to find. Real PASS
    # cases finish well inside this budget (worst observed: 6.15s on real
    # capture); this only clips the FAIL tail.
    # monotonic, not wall clock: a machine that sleeps mid-decode must not
    # use up the budget (it did, and lost packets).
    t_budget_start = time.monotonic()

    def try_thishdridx(thishdridx):
        if pll is None:
            # no LFSR-predicted hops possible for this grid/BW config -- no
            # hop-energy evidence obtainable, and no reliable frequency
            # anchor either. Reject early rather than blind-searching.
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
                # fragment truncated -- still append a placeholder so the
                # search grid below can erase it per-attempt rather than
                # aborting this whole hdridx hypothesis (this could be the
                # RIGHT hypothesis with only its last fragment cut off by
                # end-of-capture).
                nbits_ph = 48 if fi < num_frags-1 else (data_in_bitcount - 48*(num_frags-1))
                segs.append((seg, 0.0, nbits_ph))
                continue
            if predicted_hop_hz is not None and fi < len(predicted_hop_hz):
                f0 = predicted_hop_hz[fi]
                ff = np.fft.fftfreq(len(seg), 1/cfg.FS)
                bandm = (np.abs(ff - f0) < 600)
                S = np.abs(_fft(seg*np.hanning(len(seg))))
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
                S = np.abs(_fft(seg*np.hanning(len(seg))))
                S[~band] = 0
                fc = f[np.argmax(S)]
            nbits = 48 if fi < num_frags-1 else (data_in_bitcount - 48*(num_frags-1))
            segs.append((seg, fc, nbits))
        # NOTE: a hard hop-energy reject gate used to sit here (mean Ek/noise
        # < HOP_ENERGY_MIN). Removed: validated only at high SNR where it
        # cleanly separated real (500-955) from fake (0-11.5) headers, but at
        # -10dB real packets measure Ek=2.3-26.7 -- overlapping the ORIGINAL
        # fake range and not even reliably beating a wrong hdridx guess on
        # the same real packet. It was silently rejecting every real -10dB
        # packet before the search ran. CRC16 (65536x stronger than the
        # header's own CRC8) remains the real correctness gate below.
        # SHARED carrier-offset (dcfo) scan OUTSIDE the shared symbol-timing
        # (gsto) scan. Both are single values applied to EVERY fragment --
        # the physical link has one clock and one residual CFO, not a
        # per-fragment one. The per-fragment FFT-peak `fc` above is a good
        # starting point but carries a small data-dependent spectral bias
        # that isn't identical across fragments (observed tens-to-~150Hz
        # variation); a shared dcfo nudge on top of it, CRC16-gated, absorbs
        # that residual without breaking cross-fragment bit coherence the
        # way a per-fragment-independent frequency search would.
        #
        # Measured true solutions across known-good packets: dcfo always
        # within +/-30Hz (FFT-peak bias is small and consistent), gsto spans
        # the FULL symbol range (real sub-symbol clock drift, not narrow).
        # So: tighten dcfo hard, keep gsto fine and full-range. This is a
        # real reduction in combos (5 x 114 = 570 vs the old 21 x 114 =
        # 2394), not a heuristic shortcut -- bimodality-guided refinement
        # was tried and is UNSAFE here (measured: true solutions can have
        # bimodality as low as 0.60 while wrong points hit 0.90+), so CRC16
        # stays the only acceptance test over the full remaining grid.
        best_stream = [None, -1.0]   # [soft_stream, bimodality] across the whole grid
        # Two-tier gsto grid. MEASURED at clean SNR by running the full grid
        # with early-exit disabled: 1818 of 35351 (gsto,dcfo) points pass
        # CRC16 -- 5.14%, roughly 1 in 19, NOT a single needle. So a coarse
        # first pass finds a passing point almost always and costs ~1/4 the
        # Viterbi calls; the fine pass runs only when coarse finds nothing.
        # (At -10dB the density collapses to 1/21522 -- a true needle -- which
        # is exactly why the fine fallback has to stay: the coarse tier is a
        # speed path for workable SNR, not a replacement for the full search.)
        # Starting carrier per fragment. First choice: the predicted hop plus
        # one offset shared by every fragment, estimated from all of them
        # together (_common_fragment_cfo) -- data-independent, so it does
        # not inherit the per-fragment FFT-peak bias. The per-fragment FFT
        # peaks stay as a second choice, so a packet the old search
        # decoded is still decoded; the extra pass only runs when the
        # first finds no CRC16 pass.
        seg_sets = []
        n_pred = 0
        if predicted_hop_hz is not None and len(predicted_hop_hz) >= len(segs):
            # The header carrier now comes from its sync word to ~1 Hz, and
            # header and payload share one oscillator, so the predicted
            # hops are usually right as they stand: try them first.
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
            # Erasure-aware: scale each fragment's soft bits by its measured
            # reliability, normalised so the best fragment keeps full scale.
            wts = _fragment_weights(seg_set)
            wts = wts/wts.max() if wts.max() > 0 else np.ones(len(seg_set))
            # Trellis first on the predicted-carrier sets (plain, then with
            # the shared offset): their carriers are right to a few Hz,
            # which is what the trellis needs. The differential search
            # below stays as the fallback.
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
                  # Hoist the entire gsto sweep into one C++ call per fragment
                  # (demod_symbols_grid) instead of one demod_symbols call per
                  # (fragment, gsto). Measured 343,966 demod_symbols_ext calls /
                  # 3.455s on a -10dB run for ~50 sample centers each -- almost
                  # entirely per-call pybind11 dispatch + allocation, not phase
                  # arithmetic. Same math per row (validated bit-exact, 0.0 diff
                  # over 200 randomized trials including truncated fragments),
                  # 114x fewer boundary crossings.
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

    # Cheap default order, not detect_replica_index (that costs ~2-3s/packet
    # -- reserved for plot_packet_spectrogram where accuracy over speed is
    # the right tradeoff). Empirically, _fine_sync's correlation search lands
    # on replica 1 almost always (26/26 real packets, this session) -- try 1
    # first, CRC16-gated fallback over 2,3 still covers the rare miss.
    order = [1, 2, 3]
    for k in order:
        if pll is not None:
            addval_check = hdr_f_measured - pll[k-1]*step
            predicted_check = [p*step + addval_check for p in pll[cfg.HDR_COUNT:]]
            # Tolerance: a hop can sit right at the band edge, and a carrier
            # estimate a few tens of Hz off then put it just outside -- a
            # hard edge here rejected the right guess by 8 Hz on a real
            # capture (case079). Wrong guesses miss by whole grid steps
            # (>= 3.9 kHz), so HOP_EDGE_TOL_HZ still rejects them.
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
