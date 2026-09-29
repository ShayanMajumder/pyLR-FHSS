"""
lrfhss_decode.py -- Python translation of the FSU LR-FHSS receiver decode chain.

Faithful ports of the repo's Octave/MATLAB algorithms so payloads decode
bit-exact:
  - myTrellis (64-state, rate-1/3 mother code) from lrfh_init_trellis.m
  - lrfh_deinterleaving_payload.m
  - lrfh_decode_payload.m  (soft Viterbi, CR=3 full / CR=1 punctured [1 1 0 0 1 0])
  - lrfh_crc16.m  (LUT-based)
  - lrfh_dewhitening_payload.m  (LFSR + nibble swap)

Bit conventions match the repo: 'left-msb' bit ordering throughout.
"""

import numpy as np

# Optional pybind11-accelerated Viterbi core (ext/viterbi_ext.cpp). Falls
# back to the pure-numpy per-symbol loop below if the extension isn't built
# for this platform -- same call sites, same bit-exact output either way
# (validated against each other in ext/test_viterbi_ext.py).
try:
    from .. import _viterbi_ext as _vext
    _HAVE_VEXT = True
except ImportError:
    _vext = None
    _HAVE_VEXT = False

# ---------------------------------------------------------------------------
# Trellis (from lrfh_init_trellis.m). 64 states, 2 inputs, 8 output symbols.
# nextStates[state, input] and outputs[state, input] (0-based state index).
# ---------------------------------------------------------------------------
_NEXT = np.array([[ (2*s) % 64, (2*s+1) % 64 ] for s in range(64)], dtype=int)

_OUTPUTS = np.array([
    [0,7],[3,4],[7,0],[4,3],[6,1],[5,2],[1,6],[2,5],
    [1,6],[2,5],[6,1],[5,2],[7,0],[4,3],[0,7],[3,4],
    [4,3],[7,0],[3,4],[0,7],[2,5],[1,6],[5,2],[6,1],
    [5,2],[6,1],[2,5],[1,6],[3,4],[0,7],[4,3],[7,0],
    [7,0],[4,3],[0,7],[3,4],[1,6],[2,5],[6,1],[5,2],
    [6,1],[5,2],[1,6],[2,5],[0,7],[3,4],[7,0],[4,3],
    [3,4],[0,7],[4,3],[7,0],[5,2],[6,1],[2,5],[1,6],
    [2,5],[1,6],[5,2],[6,1],[4,3],[7,0],[3,4],[0,7],
], dtype=int)

MY_TRELLIS = {'numStates': 64, 'nextStates': _NEXT, 'outputs': _OUTPUTS}

# Header trellis (16-state) from lrfh_init_trellis.m -- included for completeness.
_NEXT_H = np.array([[ (2*s) % 16, (2*s+1) % 16 ] for s in range(16)], dtype=int)
_OUTPUTS_H = np.array([
    [0,3],[1,2],[2,1],[3,0],[2,1],[3,0],[0,3],[1,2],
    [3,0],[2,1],[1,2],[0,3],[1,2],[0,3],[3,0],[2,1],
], dtype=int)
MY_TRELLIS_HEADER = {'numStates': 16, 'nextStates': _NEXT_H, 'outputs': _OUTPUTS_H}


def _int_to_bits(val, n):
    """left-msb bit expansion, matches de2bi(val, n, 'left-msb')."""
    return np.array([(val >> (n-1-i)) & 1 for i in range(n)], dtype=int)


# Precompute 3-bit left-msb patterns for each output symbol 0..7
_OUT3 = {v: _int_to_bits(v, 3) for v in range(8)}

# --- Precomputed flat branch tables for fast payload Viterbi (bit-exact) ---
_PBR_FROM = np.repeat(np.arange(64), 2)
_PBR_INP  = np.tile([0, 1], 64)
_PBR_NXT  = np.array([[MY_TRELLIS['nextStates'][s, i] for i in range(2)] for s in range(64)]).reshape(-1)
_PBR_OSY  = np.array([[MY_TRELLIS['outputs'][s, i] for i in range(2)] for s in range(64)]).reshape(-1)
_OUT3_TAB = np.array([_OUT3[v] for v in range(8)])

# Destination-sorted view of the same 128 branches, grouped [64 states x 2
# incoming branches] (every state has exactly 2 predecessors in this trellis
# -- verified). This lets the per-symbol branch-metric reduction run as a
# single vectorized argmin over axis=1 instead of a 128-iteration Python
# loop, which profiling showed was the dominant cost in the whole pipeline
# (47s of 81s on a 10-packet run, ~58%). ~5.7x faster on this stage alone,
# bit-exact (validated against the scalar-loop version on 50 random trials).
_PBR_SORT = np.argsort(_PBR_NXT, kind='stable')
_PBR_SORT_FROM = _PBR_FROM[_PBR_SORT].reshape(64, 2)
_PBR_SORT_INP  = _PBR_INP[_PBR_SORT].reshape(64, 2)
_PBR_SORT_OSY  = _PBR_OSY[_PBR_SORT].reshape(64, 2)
_ARANGE64 = np.arange(64)

# NOTE: a bidirectional (backward) Viterbi pass for the payload trellis was
# attempted and REJECTED after testing, unlike the header's backward pass
# which is real and kept. Reasoning: header's backward search helps because
# BOTH the start and end state are unknown there (16 candidates each way),
# so forward and backward searches explore genuinely different hypothesis
# space under noise -- validated empirically (+80% recovery on synthetic
# trials, see lrfhss_header.py). Payload's start state is FIXED at 0 (the
# encoder always starts there); forward already fully exploits that by
# initializing cost[0]=0 and searching only the end state via argmin. A
# backward pass fixing candidate end states and checking whether the
# recovered start lands on 0 sounds like a mirror of forward, but isn't:
# at real noise levels the TRUE end-state hypothesis's cost[0] is routinely
# NOT the global argmin over states (confirmed empirically: a controlled
# encode+corrupt+decode trial found cost[0]=12892 for the correct end
# state while the noisy trellis's argmin sat at cost=12147 on a wrong
# state), so gating on argmin==0 rejects real solutions, and gating on
# cost[0] alone (without a threshold with its own false-positive tuning
# work) isn't a free win the way header's fully-symmetric problem was.
# Swept a 300-trial forward-failure test at multiple noise levels: 0
# recoveries. Removed rather than shipped non-functional.


def deinterleave_payload(payload, data_in_bitcount):
    """Port of lrfh_deinterleaving_payload.m. payload: 1D array. Returns 1D.

    Fast path: deinterleave_payload_ext (ext/viterbi_ext.cpp) -- sequential
    index-generation loop, same non-vectorizable shape as the Viterbi DP
    (each `pos` depends on the previous iteration), small but free once
    ported (0.095s/run at 4337 calls on a real recording). Bit-exact with
    the numpy version below when len(payload) == data_in_bitcount, which
    is the invariant every real caller in this package guarantees
    (soft_stream is always truncated to exactly data_in_bitcount before
    this is called) -- validated 500/500 trials at that invariant. Falls
    back to numpy for any other shape (the reference implementation itself
    isn't bounds-safe outside that invariant either, so this isn't losing
    safety, just not fast-pathing a case nothing relies on)."""
    payload = np.asarray(payload).ravel()
    if _HAVE_VEXT and len(payload) == data_in_bitcount:
        return np.asarray(_vext.deinterleave_payload_ext(
            payload.astype(np.float64, copy=False), int(data_in_bitcount)))

    y = 0
    while y*y < data_in_bitcount:
        y += 1
    step = y
    step_v = step >> 1          # floor(bitsra(step,1))
    step = step << 1            # bitshift(step,1)
    st_idx = 0
    st_idx_init = 0
    pos = 0
    deint = np.zeros(len(payload), dtype=payload.dtype)
    # MATLAB is 1-based; here pos indexes 0-based, so we track pos as the
    # 1-based value used in MATLAB and write to deint[pos-1].
    for i in range(1, len(payload)):   # i = 2..N in MATLAB
        pos = pos + step
        if pos >= data_in_bitcount:
            st_idx = st_idx + step_v
            if st_idx >= step:
                st_idx_init += 1
                st_idx = st_idx_init
            pos = st_idx
        # MATLAB deint_payload(pos) = payload(i); pos is 1-based there.
        deint[pos-1] = payload[i]   # payload[i] is 0-based i-th (MATLAB i+1)
    # deint_payload(2:end)=deint_payload(1:end-1); deint_payload(1)=payload(1)
    deint[1:] = deint[:-1]
    deint[0] = payload[0]
    return deint


def _cal_distance_soft(bits, r):
    """cal_Distance soft: sum over bits of (bit? (255 - r_i) : r_i)-like metric.

    The repo quantizes soft values to 0..255 (128=erasure). cal_Distance with
    the soft flag computes, per bit, the cost of the hypothesized bit vs the
    received quantized value. We replicate the standard MATLAB comms soft
    metric used by vitdec 'soft': cost = sum(|hypbit*255 - r|)/... but the
    repo's own vit uses cal_Distance. We implement the equivalent: for each
    bit b and received soft r in [0,255], cost += (255-r) if b==1 else r.
    """
    r = np.asarray(r).ravel()
    b = np.asarray(bits).ravel()
    return np.sum(np.where(b == 1, 255 - r, r))


def viterbi_decode_payload(deint_payload, CR, demod_soft_val_cap=1):
    """Port of lrfh_decode_payload.m soft-Viterbi (the `if 1` branch logic,
    reimplemented as an explicit soft Viterbi to avoid depending on MATLAB's
    vitdec). Returns (decoded_info_bits, crc_match).

    deint_payload: soft values in ~[-cap, +cap] (repo demod range).
    CR: 3 (rate 1/3, no puncture) or 1 (puncture [1 1 0 0 1 0]).

    Fast path: viterbi_payload_full (ext/viterbi_ext.cpp) does the whole
    quantize -> depuncture -> cost_sym -> DP -> traceback -> CRC16 chain
    in one C++ call. Profiling after the DP-only C++ port showed the
    WRAPPER cost (depuncture loop, cost_sym np.stack, crc16's own Python
    loop, array shuffling) had become the dominant remaining cost of this
    function -- 0.898s cumulative vs 0.099s in the DP call itself on a
    real run -- so it's fused too now. Bit-exact with the numpy fallback
    below (validated 400 random trials across all CR values, 0 mismatches
    -- see ext/test_viterbi_ext.py). Falls back to the full numpy
    implementation if the extension isn't built.
    """
    deint = np.asarray(deint_payload, dtype=float).ravel()
    if _HAVE_VEXT:
        info, match = _vext.viterbi_payload_full(
            deint, int(CR), float(demod_soft_val_cap),
            _PBR_SORT_FROM, _PBR_SORT_INP, _PBR_SORT_OSY)
        return np.asarray(info), bool(match)

    # Quantize exactly as lrfh_decode_payload.m:
    #   local_mul = 128/cap - 1;  q = round(local_mul*deint) + 128
    local_mul = 128.0/demod_soft_val_cap - 1.0
    q = np.round(local_mul*deint) + 128
    q = np.clip(q, 0, 255)

    # De-puncture into mother-code rate-1/3 soft stream. Punctured positions
    # get erasure = 128 (neutral). The encoder's puncturing matrix and periods
    # (lrfh_con_encode.m): matrix = [1 1 0 0 1 0 1 0 0 0 1 0 1 0 0];
    #   CR=0 -> period 15 (rate 5/6), CR=1 -> period 6 (2/3),
    #   CR=2 -> period 3 (1/2), CR=3 -> unpunctured (1/3).
    if CR == 3:
        mother = q
    else:
        full_matrix = np.array([1,1,0,0,1,0,1,0,0,0,1,0,1,0,0])
        period = {0: 15, 1: 6, 2: 3}[CR]
        pat = full_matrix[:period]
        kept_per = int(pat.sum())
        n_kept = len(q)
        n_periods = int(np.ceil(n_kept / kept_per))
        mother = np.full(n_periods*period, 128.0)
        ki = 0
        end = 0
        for p in range(n_periods):
            for j in range(period):
                if pat[j] == 1 and ki < n_kept:
                    mother[p*period+j] = q[ki]; ki += 1
                    end = p*period + j + 1
        # The mother stream is 3 bits per trellis step, so it rarely fills
        # its last puncturing period. It ends within 2 bits of the last
        # kept one for every pattern above, so round that up to a whole
        # step. Padding to the whole period instead added phantom steps,
        # which moved where the CRC is read from the end: CR=0 failed on 4
        # payload lengths in 5, even error-free. CR=1 and CR=2 happen to
        # always end on a period.
        mother = mother[:-(-end//3)*3]

    nsyms = len(mother)//3
    INF = 1e18
    r3 = mother[:3*nsyms].reshape(nsyms, 3)
    cost_sym = np.stack([np.sum(np.where(_OUT3_TAB[v] == 1, 255 - r3, r3), axis=1)
                         for v in range(8)], axis=1)   # [nsyms, 8]

    cost = np.full(64, INF); cost[0] = 0.0
    back = np.zeros((nsyms, 64), dtype=np.int8)
    prev = np.zeros((nsyms, 64), dtype=np.int16)
    for i in range(nsyms):
        bc2 = cost[_PBR_SORT_FROM] + cost_sym[i][_PBR_SORT_OSY]   # [64, 2]
        which = np.argmin(bc2, axis=1)
        cost = bc2[_ARANGE64, which]
        back[i] = _PBR_SORT_INP[_ARANGE64, which]
        prev[i] = _PBR_SORT_FROM[_ARANGE64, which]

    # terminate at state 0 (repo uses 'term'); trace back from best state.
    end_state = int(np.argmin(cost))
    bits = np.zeros(nsyms, dtype=int)
    s = end_state
    for i in range(nsyms-1, -1, -1):
        bits[i] = back[i, s]
        s = prev[i, s]

    # decoded_pay includes info + CRC16 + 6 tail bits (term). Repo:
    #   crc = crc16(decoded(1:end-16-6)); match vs decoded(end-15-6:end-6)
    if len(bits) < 22:
        return np.array([], dtype=int), False
    info = bits[:-16-6]
    crc_rx = bits[-16-6:-6]
    crc_calc = crc16(info)
    match = np.array_equal(crc_calc, crc_rx)
    return info, match


# CRC16 LUT from lrfh_crc16.m
_CRC16_LUT = [0,30043,60086,40941,41015,54636,19073,16346,13621,16494,57219,43736,38146,57433,32692,2799,27242,7985,32988,62855,51805,48902,8427,21936,24415,10756,46569,49330,65384,35379,5598,24709,54484,41359,15970,19257,29923,440,40533,60174,57825,38074,2903,32268,16854,13453,43872,56891,48830,52197,21512,8531,7817,27602,62527,33124,35723,65232,24893,5222,11196,24295,49418,46161,56563,43432,13893,17182,31940,2463,38514,58153,59846,40093,880,30251,18929,15530,41799,54812,46745,50114,23599,10612,5806,25589,64536,35139,33708,63223,26906,7233,9115,22208,51501,48246,2087,32124,58001,38858,43024,56651,17062,14333,15634,18505,55204,41727,40229,59518,30611,712,25165,5910,35067,64928,49786,46881,10444,23959,22392,8739,48590,51349,63311,33300,7673,26786,52413,47590,9739,21328,27786,6609,34364,62311,63880,36051,4926,26213,22975,11492,45833,50770,42711,54156,19553,14650,1760,29627,60502,39181,37858,59065,31060,3087,13269,18062,55651,44088,6249,27954,62175,34692,47198,52485,21224,10163,11612,22535,51178,45745,36203,63536,26589,4742,29187,1880,39093,60910,53812,42863,14466,19929,18230,12909,44416,55515,59137,37466,3511,30956,4174,25877,64248,36771,45177,50466,23247,12180,9595,20512,53197,47766,34124,61463,28666,6817,31268,3967,37010,58825,55827,44872,12453,17918,20241,14922,42407,53500,61222,39549,1424,28875,50330,45505,11820,23415,25773,4598,36379,64320,61871,34036,6937,28226,20888,9411,47918,52853,44784,56235,17478,12573,3783,31644,58481,37162,39877,61086,29043,1064,15346,20137,53572,42015]


def crc16(decoded_bits):
    """Port of lrfh_crc16.m. Input: bit array (left-msb per byte). Returns 16 bits."""
    d = np.asarray(decoded_bits, dtype=int).ravel()
    crc = np.ones(16, dtype=int)
    nbytes = len(d)//8
    for k in range(nbytes):
        top8 = crc[:8] ^ d[k*8:k*8+8]
        pos = int(''.join(map(str, top8)), 2)   # bi2de left-msb
        lut_bits = _int_to_bits(_CRC16_LUT[pos], 16)
        crc = np.concatenate([crc[8:16], np.zeros(8, dtype=int)]) ^ lut_bits
    return crc


def dewhiten_payload(data_in_bits, nbytes):
    """Port of lrfh_dewhitening_payload.m. Input/return: bit arrays."""
    d = np.asarray(data_in_bits, dtype=int).ravel()
    out = np.zeros(nbytes*8, dtype=int)
    lfsr = np.ones(8, dtype=int)
    for idx in range(nbytes):
        b = slice(idx*8, idx*8+8)
        # nibble swap: out[1:4]=in[5:8]; out[5:8]=in[1:4]  (1-based -> here)
        seg = d[b]
        swapped = np.concatenate([seg[4:8], seg[0:4]])
        out[b] = swapped ^ lfsr
        # lfsr update (same polynomial as whitening)
        newbit = lfsr[0] ^ lfsr[2] ^ lfsr[3] ^ lfsr[4]
        lfsr = np.concatenate([lfsr[1:8], [newbit]])
    return out


def bits_to_bytes(bits):
    """left-msb bytes from a bit array whose length is a multiple of 8."""
    b = np.asarray(bits, dtype=int).ravel()
    b = b[:(len(b)//8)*8]
    return np.array([int(''.join(map(str, b[i:i+8])), 2) for i in range(0, len(b), 8)], dtype=int)