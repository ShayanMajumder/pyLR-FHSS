# Implemented with reference to the LR-FHSS-receiver MATLAB code by Jumana
# Bukhari and Zhenghao Zhang, https://github.com/jumanamirza/LR-FHSS-receiver
# That code is provided for education and academic research only.
"""Header decode primitives: deinterleaving, tail-biting Viterbi decoding
and CRC-8.
"""
import numpy as np
from .fec import MY_TRELLIS_HEADER, _int_to_bits, _HAVE_VEXT, _vext

_HDR_DEINT = np.array([1,23,45,63,6,28,50,68,11,33,55,73,15,37,59,77,19,41,2,24,46,64,7,29,51,69,12,34,56,74,16,38,60,78,20,42,3,25,47,65,8,30,52,70,13,35,57,75,17,39,61,79,21,43,4,26,48,66,9,31,53,71,14,36,58,76,18,40,62,80,22,44,5,27,49,67,10,32,54,72])

_CRC8_LUT = [0,47,94,113,188,147,226,205,87,120,9,38,235,196,181,154,174,129,240,223,18,61,76,99,249,214,167,136,69,106,27,52,115,92,45,2,207,224,145,190,36,11,122,85,152,183,198,233,221,242,131,172,97,78,63,16,138,165,212,251,54,25,104,71,230,201,184,151,90,117,4,43,177,158,239,192,13,34,83,124,72,103,22,57,244,219,170,133,31,48,65,110,163,140,253,210,149,186,203,228,41,6,119,88,194,237,156,179,126,81,32,15,59,20,101,74,135,168,217,246,108,67,50,29,208,255,142,161,227,204,189,146,95,112,1,46,180,155,234,197,8,39,86,121,77,98,19,60,241,222,175,128,26,53,68,107,166,137,248,215,144,191,206,225,44,3,114,93,199,232,153,182,123,84,37,10,62,17,96,79,130,173,220,243,105,70,55,24,213,250,139,164,5,42,91,116,185,150,231,200,82,125,12,35,238,193,176,159,171,132,245,218,23,56,73,102,252,211,162,141,64,111,30,49,118,89,40,7,202,229,148,187,33,14,127,80,157,178,195,236,216,247,134,169,100,75,58,21,143,160,209,254,51,28,109,66]


_BIT8_WEIGHTS = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.int64)


def crc8_header(bits32):
    """CRC8 over 32 header info bits."""
    if _HAVE_VEXT:
        d = np.asarray(bits32, int).ravel()
        return np.asarray(_vext.crc8_ext(d.astype(np.int32)))
    d = np.asarray(bits32, int).ravel()
    crc = np.ones(8, int)
    for k in range(4):
        top = crc ^ d[k*8:k*8+8]
        pos = int(top @ _BIT8_WEIGHTS)
        crc = _int_to_bits(_CRC8_LUT[pos], 8)
    return crc


def deinterleave_header(header_bits):
    """header_bits: 113 bits (hard or soft). temp = [bits[1:41], bits[73:]] (0-based).
    """
    h = np.asarray(header_bits)
    temp = np.concatenate([h[1:41], h[73:]])   # the coded bits around the sync word
    deint = np.zeros(80, dtype=h.dtype)
    for idx in range(80):
        deint[idx] = temp[_HDR_DEINT[idx]-1]
    return deint


def _cal_dist_soft(bits, r):
    r = np.asarray(r).ravel(); b = np.asarray(bits).ravel()
    return np.sum(np.where(b == 1, 255-r, r))


_BR_FROM = np.repeat(np.arange(16), 2)
_BR_INP  = np.tile([0, 1], 16)
_BR_NXT  = np.array([[MY_TRELLIS_HEADER['nextStates'][s, i] for i in range(2)] for s in range(16)]).reshape(-1)
_BR_OSY  = np.array([[MY_TRELLIS_HEADER['outputs'][s, i] for i in range(2)] for s in range(16)]).reshape(-1)
_OUT2_TAB = np.array([_int_to_bits(v, 2) for v in range(4)])

_BR_SORT = np.argsort(_BR_NXT, kind='stable')
_BR_SORT_FROM = _BR_FROM[_BR_SORT].reshape(16, 2)
_BR_SORT_INP  = _BR_INP[_BR_SORT].reshape(16, 2)
_BR_SORT_OSY  = _BR_OSY[_BR_SORT].reshape(16, 2)
_ARANGE16 = np.arange(16)

_REV_PRED_FROM = [[] for _ in range(16)]   # predecessor state per successor
_REV_PRED_INP = [[] for _ in range(16)]
_REV_PRED_OSYM = [[] for _ in range(16)]
for _s in range(16):
    for _b in range(2):
        _s2 = int(MY_TRELLIS_HEADER['nextStates'][_s, _b])
        _REV_PRED_FROM[_s2].append(_s)
        _REV_PRED_INP[_s2].append(_b)
        _REV_PRED_OSYM[_s2].append(int(MY_TRELLIS_HEADER['outputs'][_s, _b]))
# flatten to arrays (2 predecessors per state, fixed)
_RBR_TO = np.repeat(np.arange(16), 2)                                  # successor state
_RBR_FROM = np.array([_REV_PRED_FROM[s][k] for s in range(16) for k in range(2)])
_RBR_INP = np.array([_REV_PRED_INP[s][k] for s in range(16) for k in range(2)])
_RBR_OSY = np.array([_REV_PRED_OSYM[s][k] for s in range(16) for k in range(2)])
# already grouped by successor state (built that way above) -- just reshape
_RBR_SORT_FROM = _RBR_FROM.reshape(16, 2)
_RBR_SORT_INP  = _RBR_INP.reshape(16, 2)
_RBR_SORT_OSY  = _RBR_OSY.reshape(16, 2)


def _decode_header_backward(deint_soft):
    """Backward Viterbi: same trellis run in reverse (predecessor lookup),
    trying all 16 possible END states (mirrors forward's 16 START states,
    since the true end state is equally unknown at receive time).
    """
    q = np.round(127*np.asarray(deint_soft, float)) + 128
    q = np.clip(q, 0, 255)
    n = len(q)//2
    INF = 1e18
    r = q[:2*n].reshape(n, 2)
    cost_sym = np.stack([np.sum(np.where(_OUT2_TAB[v] == 1, 255 - r, r), axis=1)
                         for v in range(4)], axis=1)

    if _HAVE_VEXT and n > 0:
        rows = _vext.viterbi_header_backward(cost_sym, _RBR_SORT_FROM, _RBR_SORT_INP, _RBR_SORT_OSY)
        for end_state in range(16):
            bits = rows[end_state]
            if len(bits) >= 40:
                info = bits[:40]
                if np.array_equal(crc8_header(info[:32]), info[32:40]):
                    return info
        return None

    for end_state in range(16):
        cost = np.full(16, INF); cost[end_state] = 0.0
        back = np.zeros((n, 16), dtype=np.int8)
        nxtst = np.zeros((n, 16), dtype=np.int16)
        for i in range(n-1, -1, -1):
            bc2 = cost[_RBR_SORT_FROM] + cost_sym[i][_RBR_SORT_OSY]   # [16, 2]
            which = np.argmin(bc2, axis=1)
            newcost = bc2[_ARANGE16, which]
            back[i] = _RBR_SORT_INP[_ARANGE16, which]
            nxtst[i] = _RBR_SORT_FROM[_ARANGE16, which]
            cost = newcost
        start_state = int(np.argmin(cost))
        bits = np.zeros(n, int); s = start_state
        for i in range(n):
            bits[i] = back[i, s]; s = nxtst[i, s]
        if len(bits) >= 40:
            info = bits[:40]
            if np.array_equal(crc8_header(info[:32]), info[32:40]):
                return info
    return None


def decode_header(deint_soft, try_backward=False, metric='manhattan'):
    """16-state rate-1/2 Viterbi over 80 soft bits -> 40 info bits + CRC8."""
    q = np.round(127*np.asarray(deint_soft, float)) + 128
    q = np.clip(q, 0, 255)
    n = len(q)//2
    INF = 1e18
    r = q[:2*n].reshape(n, 2)
    if metric == 'euclid':
        cost_sym = np.stack([np.sum(np.where(_OUT2_TAB[v] == 1, (255-r)**2, r**2), axis=1)
                             for v in range(4)], axis=1)
    else:
        cost_sym = np.stack([np.sum(np.where(_OUT2_TAB[v] == 1, 255 - r, r), axis=1)
                             for v in range(4)], axis=1)
    best_result = None

    if _HAVE_VEXT and n > 0 and metric != 'euclid':
        rows = _vext.viterbi_header_multistart(cost_sym, _BR_SORT_FROM, _BR_SORT_INP, _BR_SORT_OSY)
        for start_state in range(16):
            bits = rows[start_state]
            if len(bits) >= 40:
                info = bits[:40]
                if np.array_equal(crc8_header(info[:32]), info[32:40]):
                    best_result = info
                    break
    else:
        for start_state in range(16):
            cost = np.full(16, INF); cost[start_state] = 0.0
            back = np.zeros((n, 16), dtype=np.int8)
            prev = np.zeros((n, 16), dtype=np.int16)
            for i in range(n):
                bc2 = cost[_BR_SORT_FROM] + cost_sym[i][_BR_SORT_OSY]   # [16, 2]
                which = np.argmin(bc2, axis=1)
                cost = bc2[_ARANGE16, which]
                back[i] = _BR_SORT_INP[_ARANGE16, which]
                prev[i] = _BR_SORT_FROM[_ARANGE16, which]
            end = int(np.argmin(cost))
            bits = np.zeros(n, int); s = end
            for i in range(n-1, -1, -1):
                bits[i] = back[i, s]; s = prev[i, s]
            if len(bits) >= 40:
                info = bits[:40]
                if np.array_equal(crc8_header(info[:32]), info[32:40]):
                    best_result = info
                    break
    if best_result is None and try_backward:
        best_result = _decode_header_backward(deint_soft)
    if best_result is None:
        return None
    info = best_result
    def b2d(b):
        return int(''.join(map(str, b)), 2)
    return dict(
        payloadlen=b2d(info[0:8]), modulation=info[8:11].tolist(),
        CR=b2d(info[11:13]), grid=int(info[13]), hop=int(info[14]),
        BW=info[15:19].tolist(), hopseq=info[19:28].tolist(),
        syncindex=int(info[28])*2+int(info[29]), CRCpass=True,
    )
