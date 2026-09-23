"""hop_freq.py -- port of calculate_freq_from_hop_seq_id.m (LR-FHSS LFSR hop
frequency generator). Given the decoded header's grid/BW/hop_seq_id/header_count,
compute the exact PLL-step frequencies for every header replica + payload
fragment -- the ground-truth hop schedule, rather than re-measuring each
fragment by FFT peak (error-prone when hops are dense or noisy)."""
import numpy as np

_CHANNEL_COUNT = [80, 176, 280, 376, 688, 792, 1480, 1584, 3120, 3224]
_POLY1 = [33, 45, 48, 51, 54, 57]
_POLY2 = [65, 68, 71, 72]
_POLY3 = [142, 149]


def _b2d(bits):
    return int(''.join(map(str, bits)), 2)


def _get_hop_params(grid, bw_bits, hop_seq_id_bits):
    channel_count = _CHANNEL_COUNT[_b2d(bw_bits)]
    # Floor, not true division: the grid holds floor(channel_count/nb) whole
    # slots. For the 25.39 kHz (FCC) grid most OCWs divide unevenly --
    # 1480/52 = 28.46 -- and a float never matches the integer n_grid
    # branches below, so _get_hop_params fell through to status=3 and
    # calculate_freq_from_hop_seq_id returned None for every FCC config
    # except 1523.4/1574.2 kHz (3120/52=60, 3224/52=62 divide exactly).
    # The transmitter uses the floored count (manifest ngrid=28/30/60/62).
    n_grid = channel_count//8 if grid == 1 else channel_count//52
    hop_seq_id_de = _b2d(hop_seq_id_bits)
    xoring_seed = [0]*16
    status = 3

    try:
        if n_grid in (10, 22, 28, 30, 35, 47):
            initial_state = 6
            polynomial = _POLY1[_b2d(hop_seq_id_bits[0:3])]
            xoring_seed[-6:] = hop_seq_id_bits[3:9]
            status = 0
            if hop_seq_id_de >= 384:
                status = 3
        elif n_grid in (60, 62):
            initial_state = 56
            polynomial = _POLY1[_b2d(hop_seq_id_bits[0:3])]
            xoring_seed[-6:] = hop_seq_id_bits[3:9]
            status = 0
            if hop_seq_id_de >= 384:
                status = 3
        elif n_grid in (86, 99):
            initial_state = 6
            polynomial = _POLY2[_b2d(hop_seq_id_bits[0:2])]
            xoring_seed[-7:] = hop_seq_id_bits[2:9]
            status = 0
        elif n_grid in (185, 198):
            initial_state = 6
            polynomial = _POLY3[_b2d(hop_seq_id_bits[0:1])]
            xoring_seed[-8:] = hop_seq_id_bits[1:9]
            status = 0
        elif n_grid in (390, 403):
            initial_state = 6
            polynomial = 264
            xoring_seed[-9:] = hop_seq_id_bits
            status = 0
        else:
            return 3, n_grid, None, None, None
    except IndexError:
        return 3, n_grid, None, None, None

    init_bits = [(initial_state >> (15-i)) & 1 for i in range(16)]
    poly_bits = [(polynomial >> (15-i)) & 1 for i in range(16)]
    return status, n_grid, init_bits, poly_bits, xoring_seed

def _next_state(lfsr, n_grid, poly, xor_seed):
    lfsr = list(lfsr)
    for _ in range(100000):   # cap: state space is 2^16: garbage input can
                              # cycle forever without landing hop<=n_grid
        lsb = lfsr[-1] & 1
        lfsr = [0] + lfsr[:-1]
        if lsb:
            lfsr = [a ^ b for a, b in zip(lfsr, poly)]
        hop = xor_seed if lfsr == xor_seed else [a ^ b for a, b in zip(xor_seed, lfsr)]
        if _b2d(hop) <= n_grid:
            return _b2d(hop) - 1, lfsr
    raise ValueError('LFSR did not converge -- invalid hop_seq_id/grid/BW')


def _next_freq_in_grid(enable_hop, lfsr, n_grid, poly, xor_seed, hop_seq_id_de):
    if enable_hop:
        n_i, lfsr = _next_state(lfsr, n_grid, poly, xor_seed)
    else:
        n_i = hop_seq_id_de % n_grid
    if n_i >= int(n_grid)//2:
        n_i -= int(n_grid)
    return n_i, lfsr


def _next_freq_pll(lfsr, grid, n_grid, enable_hop, poly, xor_seed, hop_seq_id_de,
                   current_hop, header_count):
    freq_table, lfsr = _next_freq_in_grid(enable_hop, lfsr, n_grid, poly, xor_seed, hop_seq_id_de)
    nb_channel_in_grid = 8 if grid == 1 else 52
    grid_offset = (1 + (int(n_grid) % 2)) * (nb_channel_in_grid // 2)
    grid_pll_steps = 4096 if grid == 1 else 26624
    freq = -freq_table*grid_pll_steps - grid_offset*512
    if enable_hop and current_hop < header_count:
        if (header_count - current_hop) % 2 == 0:
            freq += 256
    return freq, lfsr


def calculate_freq_from_hop_seq_id(grid, enable_hop, header_count, bw_bits,
                                   hop_seq_id_bits, num_frag):
    """Returns list of PLL-step frequencies, length header_count+num_frag."""
    status, n_grid, lfsr, poly, xor_seed = _get_hop_params(grid, bw_bits, hop_seq_id_bits)
    if status != 0:
        return None
    hop_seq_id_de = _b2d(hop_seq_id_bits)
    try:
        if enable_hop:
            for _ in range(4-header_count):
                _, lfsr = _next_state(lfsr, n_grid, poly, xor_seed)
        freqs = []
        current_hop = 0
        for idx in range(num_frag+header_count):
            f, lfsr = _next_freq_pll(lfsr, grid, n_grid, enable_hop, poly, xor_seed,
                                     hop_seq_id_de, current_hop, header_count)
            freqs.append(f)
            current_hop += 1
    except ValueError:
        return None
    return freqs


FREQ_STEP_HZ = 0.95367431640625


def hop_freqs_hz(hdr, num_frags, header_count=3):
    """Convenience: hop_seq_id/grid/BW from a decoded header dict -> Hz list
    (relative; caller anchors to a measured absolute header hop frequency)."""
    pll = calculate_freq_from_hop_seq_id(hdr['grid'], hdr['hop'], header_count,
                                         hdr['BW'], hdr['hopseq'], num_frags)
    if pll is None:
        return None
    return [p*FREQ_STEP_HZ for p in pll]