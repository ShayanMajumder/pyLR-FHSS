# Part of the lrfhss receiver package.
"""LR-FHSS transmit-side waveform generation.

Builds the waveform the receiver expects: header replicas and payload
fragments, each GMSK-modulated, placed on the LFSR hop grid. Bit-level
formatting reuses the same validated primitives the decoder uses --
trellis, CRC, whitening, interleaver from phy.fec and phy.framing -- so
what this produces is byte-compatible with what decode() reads, by
construction rather than by keeping two implementations in step.

An earlier Octave transmitter synthesised each hop as a bare CW tone:

    tone_freq = CARRIER_OFFSET_HZ + hop_freq_Hz
    bits = double(sin(2*pi*tone_freq*t) > 0)      # pure tone, thresholded

That carries no bits at all. In LR-FHSS the header and payload are
carried by GMSK phase modulation WITHIN each hop:

    y = exp(1i*phasewave) * exp(1i*2*pi*freq_Hz/fs*n)
        \\______________/
         the part the tone-only transmitter dropped

With only the carrier there is nothing for the sync correlator or the
header CRC to lock onto, which is exactly what the early backscatter
captures showed.

Timing and rates come from `config`, so retune() moves the encoder and
the decoder together.

Round trips at every bandwidth in the hop tables, at header-replica
counts 1-4, at every coding rate and at every LoRaWAN data rate.

CR=0 (5/6) was long listed here as an encoder bug that depended on
payload content. It was the decoder: its de-puncturer mishandled a
partial last puncturing period, so CR=0 failed on 4 payload lengths in 5
(see phy/fec.py). This encoder was correct throughout.
"""
import numpy as np

from . import config as cfg
from .phy.fec import MY_TRELLIS, MY_TRELLIS_HEADER, crc16, _int_to_bits
from .phy.framing import crc8_header

# Rates and dwell times are read from config at call time, not frozen
# here, so a retune() for one bandwidth applies to both directions.
PHASECHANGE = np.pi/2

# Smooth 1->0 transition shape (lrfh_init_light wave_1to0 polynomial).
#
# The reference is 241 samples long, which is a duration only because the
# original transmitter ran at one rate (FS=500000/3, SMBL=342). It is
# really "about 0.7 of a symbol", so it has to be resampled whenever SMBL
# changes: left fixed, a packet generated at 386.72 kHz (SMBL=1025) got a
# transition 3x too short for its symbols, and the receiver's matched
# filter peaked at 0.58 against a 0.62 threshold -- detected nothing,
# while the same payload at 136.72 kHz scored 0.986.
_TRANSITION_POLY = [-1.7223e-05, 0.0042, -0.0042]
_REF_TRANSITION_LEN = 241
_REF_SMBL = 342


def _transition_shape():
    n = max(3, int(round(_REF_TRANSITION_LEN*(cfg.FS/cfg.SYMBOL_RATE_HZ)/_REF_SMBL)))
    return np.polyval(_TRANSITION_POLY,
                      np.linspace(1, _REF_TRANSITION_LEN, n))

# header interleaver (lrfh_interleaving_hdr)
_HDR_INT = np.array([1,19,37,55,73,5,23,41,59,77,9,27,45,63,13,31,49,67,17,35,53,71,2,20,38,56,74,6,24,42,60,78,10,28,46,64,14,32,50,68,18,36,54,72,3,21,39,57,75,7,25,43,61,79,11,29,47,65,15,33,51,69,4,22,40,58,76,8,26,44,62,80,12,30,48,66,16,34,52,70])


def _con_encode_hdr(data, knowinit=False):
    """Port of lrfh_con_encode_hdr (rate-1/2 header code). runs twice if !knowinit."""
    runnum = 1 if knowinit else 2
    state = 0
    for _ in range(runnum):
        out = []
        state = 0 if _ == 0 else state  # matches repo: state carries across runs
        for bit in data:
            osym = MY_TRELLIS_HEADER['outputs'][state, bit]
            out.extend(_int_to_bits(osym, 2).tolist())
            state = (state*2 + bit) % 16
    return np.array(out, int)


def _interleave_hdr(header_enco):
    out = np.zeros(80, int)
    for idx in range(80):
        out[idx] = header_enco[_HDR_INT[idx]-1]
    return out


def gen_hdr_bits(decoded_header):
    """Port of lrfh_gen_hdr_bits: encode+interleave+sync insertion -> 113 bits."""
    enco = _con_encode_hdr(decoded_header, knowinit=False)
    interl = _interleave_hdr(enco)
    return np.concatenate([[0], interl[:40], cfg.SYNC_WORD, interl[40:]])


def _con_encode_payload(data, CR):
    """Port of lrfh_con_encode (rate-1/3 mother, optional puncture)."""
    state = 0
    out = []
    for bit in data:
        osym = MY_TRELLIS['outputs'][state, bit]
        out.extend(_int_to_bits(osym, 3).tolist())
        state = (state*2 + bit) % 64
    out = np.array(out, int)
    if CR != 3:
        matrix = np.array([1,1,0,0,1,0,1,0,0,0,1,0,1,0,0])
        matlen = {0:15,1:6,2:3}[CR]
        kept = []; mi = 0
        for j in range(len(out)):
            if matrix[mi]: kept.append(out[j])
            mi = (mi+1) % matlen
        out = np.array(kept, int)
    return out


def _interleave_payload(data):
    """Port of lrfh_interleaving_payload."""
    n = len(data)
    step = int(np.ceil(np.sqrt(n)))
    step_v = step >> 1
    step = step << 1
    st_idx = 1; st_idx_init = 1; pos = 1
    bits_left = n; out = np.zeros(n, int); shift = 0
    while bits_left > 0:
        in_row = min(bits_left, 48)
        for j in range(1, in_row+1):
            out[j+shift-1] = data[pos-1]
            pos += step
            if pos > n:
                st_idx += step_v
                if st_idx > step:
                    st_idx_init += 1; st_idx = st_idx_init
                pos = st_idx
        bits_left -= 48; shift += 48
    return out


def _whiten_payload(data_bits, nbytes):
    d = np.asarray(data_bits, int).ravel()
    out = np.zeros(nbytes*8, int)
    lfsr = np.ones(8, int)
    for idx in range(nbytes):
        u = d[idx*8:idx*8+8] ^ lfsr
        out[idx*8:idx*8+8] = np.concatenate([u[4:8], u[0:4]])
        newbit = lfsr[0] ^ lfsr[2] ^ lfsr[3] ^ lfsr[4]
        lfsr = np.concatenate([lfsr[1:8], [newbit]])
    return out


def gen_ideal_waveform(data_bits):
    """Port of lrfh_gen_ideal_waveform: continuous-phase GMSK phase signal.
    Returns the real phase array `modsig0`; the IQ is exp(1j*modsig0)."""
    data = np.asarray(data_bits, int).ravel()

    # Symbols are placed at the EXACT period FS/BW, so the phase ramp has
    # to advance pi/2 over that same period. Building it over the ROUNDED
    # SMBL instead is what restricted this encoder to one rate: where SMBL
    # and FS/BW disagree, each symbol reads the previous ramp at the wrong
    # point, and the error accumulates. It happened to be harmless at
    # SMBL=342 and 6148 (both ~0.46 above the true period) and broke
    # everywhere else -- 0 bit errors at 136.72 kHz against 7 to 14 per 96
    # at 183.59, 335.94 and 722.66 kHz.
    spb = cfg.FS/cfg.SYMBOL_RATE_HZ
    # Ceil, not round: the ramp must reach at least to the next symbol's
    # start. Rounding DOWN (183.59 and 722.66 kHz, where FS/BW is a whisker
    # above the integer) left a sub-sample gap every symbol that the
    # zero-fill below papered over, costing 3 and 4 bit errors per 96.
    # Overlapping instead is free -- the next symbol overwrites it.
    ramp_len = int(np.ceil(spb))
    phasesmbl = np.arange(ramp_len)*((np.pi/2)/spb)

    modlen = round(len(data)/cfg.SYMBOL_RATE_HZ*cfg.FS)
    smblmid = np.round((np.arange(len(data))+0.5)/cfg.SYMBOL_RATE_HZ*cfg.FS).astype(int)
    modsig0 = np.zeros(modlen)
    for h in range(len(data)):
        thissmbl = phasesmbl if data[h] == 1 else -phasesmbl
        thisbgn = int(round(smblmid[h] - spb/2))
        initphase = 0.0 if h == 0 else modsig0[max(0, thisbgn-1)]
        if thisbgn < 0:
            thissmbl = thissmbl[-thisbgn:]; thisbgn = 0
        avail = len(modsig0) - thisbgn
        seg = thissmbl[:avail] + initphase
        modsig0[thisbgn:thisbgn+len(seg)] = seg
    # smooth transitions at bit changes (h from 3..len-2 in MATLAB, 1-based)
    _wave_1to0 = _transition_shape()
    for h in range(2, len(data)-2):
        if data[h-1] != data[h]:
            trans = _wave_1to0 if data[h-1] == 1 else -_wave_1to0
            center = int(round(np.mean(smblmid[h-1:h+1])))
            st = center - len(trans)//2
            if st >= 0 and st+len(trans) <= len(modsig0):
                modsig0[st:st+len(trans)] = modsig0[st] + trans
    # fill zeros with previous (hold)
    z = np.where(modsig0 == 0)[0]; z = z[z != 0]
    for i in z:
        modsig0[i] = modsig0[i-1]
    return modsig0


def encode_packet(payload_bytes, hop_freq_Hz, header_count=3, CR=3,
                  grid=1, enable_hop=1, BW_field=(0,0,1,0), hop_seq_id=77,
                  syncindex=0):
    """Build the full GMSK packet IQ (header replicas + payload fragments)
    on the given per-hop frequencies. hop_freq_Hz: list length header_count+num_frags.
    Returns (iq, meta). This is the CORRECT waveform to feed the backscatter TX."""
    payload_bytes = np.asarray(payload_bytes, int).ravel()
    nbytes = len(payload_bytes)
    payload_bits = np.concatenate([_int_to_bits(b, 8) for b in payload_bytes])

    # ---- header info (32 bits) + CRC8 ----
    hdr_info = np.concatenate([
        _int_to_bits(nbytes, 8),          # payloadlen
        [0,0,0],                          # modulation
        _int_to_bits(CR, 2),              # CR
        [grid], [enable_hop],
        list(BW_field),
        _int_to_bits(hop_seq_id, 9),
        _int_to_bits(syncindex, 2),
        [0,0],                            # futureuse
    ]).astype(int)
    assert len(hdr_info) == 32
    hdr_crc = crc8_header(hdr_info)
    decoded_header = np.concatenate([hdr_info, hdr_crc])   # 40 bits
    hdr_replica_bits = gen_hdr_bits(decoded_header)         # 113 bits

    # ---- payload ----
    wh = _whiten_payload(payload_bits, nbytes)
    crc = crc16(wh)
    padded = np.concatenate([wh, crc, np.zeros(6, int)])
    enco = _con_encode_payload(padded, CR)
    interl = _interleave_payload(enco)
    num_frags = int(np.ceil(len(interl)/48))
    frags = []
    for idx in range(num_frags):
        lo = idx*48; hi = min((idx+1)*48, len(interl))
        frags.append(np.concatenate([[0], interl[lo:hi], [0]]))

    # ---- synthesize each hop's GMSK burst on its carrier, place in slots ----
    total = header_count + num_frags
    assert len(hop_freq_Hz) >= total, 'need %d hop freqs' % total
    slot_lens = [cfg.STAY_HDR]*header_count + [cfg.STAY_DATA]*num_frags
    iq = np.zeros(sum(slot_lens), dtype=complex)
    cursor = 0
    seg_bits = [hdr_replica_bits]*header_count + frags
    for si in range(total):
        modsig0 = gen_ideal_waveform(seg_bits[si])
        n = np.arange(len(modsig0))
        burst = np.exp(1j*modsig0) * np.exp(1j*2*np.pi*hop_freq_Hz[si]/cfg.FS*n)
        L = min(len(burst), slot_lens[si])
        iq[cursor:cursor+L] = burst[:L]
        cursor += slot_lens[si]

    meta = dict(header_count=header_count, num_frags=num_frags, CR=CR,
                payload_len_bytes=nbytes, hop_seq_id=hop_seq_id,
                slot_lens=slot_lens, fs=cfg.FS)
    return iq, meta


# Bandwidth field -> (bits, channel count). The decoder reads this field
# from the header and uses it to pick the hop grid, so the encoder must
# advertise the same one it hops on.
BW_FIELDS = {
    39.06: (0, 0, 0, 0), 85.94: (0, 0, 0, 1), 136.72: (0, 0, 1, 0),
    183.59: (0, 0, 1, 1), 335.94: (0, 1, 0, 0), 386.72: (0, 1, 0, 1),
    722.66: (0, 1, 1, 0), 773.44: (0, 1, 1, 1), 1523.4: (1, 0, 0, 0),
    1574.2: (1, 0, 0, 1),
}


def payload_fragments(payload_len_bytes, CR):
    """How many payload fragments a packet of this size and rate needs.

    The same arithmetic decode_payload_at uses, so callers can size a
    capture before generating one.
    """
    bits = 8*(payload_len_bytes + 2) + 6
    return int(np.ceil(int(np.ceil(bits*[6/5, 3/2, 2, 3][CR]))/48))


def encode(payload, bw_khz=None, header_count=None, CR=None, grid=1,
           hop_seq_id=77, carrier_off_hz=0.0, dr=None, region='EU868'):
    """Generate the IQ for one LR-FHSS packet.

        iq, meta = lrfhss.encode(b'hello world', dr=8)
        iq, meta = lrfhss.encode(b'hello world', bw_khz=722.66, header_count=4)

    Give either `dr` -- a LoRaWAN data rate, which fixes bandwidth, coding
    rate and header-replica count together -- or those three directly, for
    the parts of the LR-FHSS parameter space no DR names.

    Works out the hop schedule itself from `hop_seq_id`, `grid` and the
    bandwidth, using the same LFSR the receiver uses to predict it -- so
    a packet made here decodes without being told where its hops went.

    `carrier_off_hz` shifts the whole packet off DC, the way a real SDR
    capture sits offset from the tuned frequency.

    Returns (iq, meta); meta carries the hop plan and slot layout.
    """
    from .phy.hopping import FREQ_STEP_HZ, calculate_freq_from_hop_seq_id

    if dr is not None:
        spec = cfg.datarate(dr, region)
        bw_khz = spec['bw_khz'] if bw_khz is None else bw_khz
        CR = spec['cr'] if CR is None else CR
        header_count = (spec['header_replicas'] if header_count is None
                        else header_count)
    bw_khz = 136.72 if bw_khz is None else bw_khz
    CR = 3 if CR is None else CR
    header_count = 3 if header_count is None else header_count

    payload = np.frombuffer(bytes(payload), dtype=np.uint8).astype(int)
    if bw_khz not in BW_FIELDS:
        raise ValueError('bw_khz must be one of %s' % sorted(BW_FIELDS))
    bw_field = BW_FIELDS[bw_khz]

    num_frags = payload_fragments(len(payload), CR)
    pll = calculate_freq_from_hop_seq_id(
        grid, 1, header_count, list(bw_field),
        [int(b) for b in format(hop_seq_id, '09b')], num_frags)
    if pll is None:
        raise ValueError(
            'no hop sequence for grid=%d bw=%s: that combination is not in '
            'the LFSR tables (see phy/hopping.py)' % (grid, bw_khz))

    hops = [p*FREQ_STEP_HZ + carrier_off_hz for p in pll]
    iq, meta = encode_packet(payload, hops, header_count=header_count, CR=CR,
                             grid=grid, BW_field=bw_field, hop_seq_id=hop_seq_id)
    meta.update(bw_khz=bw_khz, grid=grid, hop_freqs_hz=hops,
                dr=dr, region=region if dr is not None else None)
    return iq, meta
