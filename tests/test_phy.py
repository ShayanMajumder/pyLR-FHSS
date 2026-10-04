# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Protocol primitives: CRCs, hop sequence, fragment arithmetic."""
import numpy as np
import pytest

from lrfhss.encoder import payload_fragments
from lrfhss.phy import fec, framing, hopping


def test_crc8_header_is_deterministic_and_8_bits():
    bits = np.array([int(b) for b in format(0x2A3B4C5D, '032b')])
    crc = framing.crc8_header(bits)
    assert len(crc) == 8
    assert set(np.unique(crc)) <= {0, 1}
    assert np.array_equal(crc, framing.crc8_header(bits))


def test_crc8_detects_a_single_bit_flip():
    bits = np.array([int(b) for b in format(0x2A3B4C5D, '032b')])
    ref = framing.crc8_header(bits)
    for i in (0, 7, 31):
        flipped = bits.copy()
        flipped[i] ^= 1
        assert not np.array_equal(framing.crc8_header(flipped), ref)


def test_crc16_detects_a_single_bit_flip():
    bits = np.random.default_rng(0).integers(0, 2, 64)
    ref = fec.crc16(bits)
    assert len(ref) == 16
    flipped = bits.copy()
    flipped[13] ^= 1
    assert not np.array_equal(fec.crc16(flipped), ref)


@pytest.mark.parametrize('grid,bw_bits,expect_ok', [
    (1, [0, 0, 0, 0], True),    # 39.06 kHz on the 3.9 kHz grid
    (1, [1, 0, 0, 1], True),    # 1574.2 kHz
    (0, [1, 0, 0, 0], True),    # 1523.4 kHz on the 25.4 kHz (FCC) grid
    (0, [0, 0, 0, 0], False),   # 39.06 kHz has too few FCC grid slots
])
def test_hop_sequence_validity_matches_the_grid_tables(grid, bw_bits, expect_ok):
    got = hopping.calculate_freq_from_hop_seq_id(grid, 1, 3, bw_bits, [0]*9, 8)
    assert (got is not None) == expect_ok


def test_hop_grid_size_is_floored_not_fractional():
    """1480 channels / 52 per FCC slot is 28.46, and the LFSR tables key off
    the integer 28.
    """
    for bw_bits, expect in (([0, 1, 1, 0], 28), ([0, 1, 1, 1], 30),
                            ([1, 0, 0, 0], 60), ([1, 0, 0, 1], 62)):
        _, n_grid, *_ = hopping._get_hop_params(0, bw_bits, [0]*9)
        assert n_grid == expect and isinstance(n_grid, int)


def test_hop_sequence_is_reproducible_and_sized_correctly():
    a = hopping.calculate_freq_from_hop_seq_id(1, 1, 3, [0, 0, 1, 0], [0]*9, 8)
    b = hopping.calculate_freq_from_hop_seq_id(1, 1, 3, [0, 0, 1, 0], [0]*9, 8)
    assert a == b
    assert len(a) == 3 + 8          # header replicas + payload fragments


def test_different_seeds_give_different_hop_sequences():
    bits = lambda n: [int(x) for x in format(n, '09b')]
    a = hopping.calculate_freq_from_hop_seq_id(1, 1, 3, [0, 0, 1, 0], bits(77), 8)
    b = hopping.calculate_freq_from_hop_seq_id(1, 1, 3, [0, 0, 1, 0], bits(123), 8)
    assert a != b


@pytest.mark.parametrize('nbytes,cr,frags', [(8, 1, 3), (13, 1, 4), (8, 3, 6)])
def test_fragment_count_matches_the_decoder(nbytes, cr, frags):
    assert payload_fragments(nbytes, cr) == frags


def test_longer_payloads_never_need_fewer_fragments():
    for cr in range(4):
        counts = [payload_fragments(n, cr) for n in range(1, 60)]
        assert counts == sorted(counts)


@pytest.mark.parametrize('vext', [True, False], ids=['default', 'numpy'])
@pytest.mark.parametrize('cr', [0, 1, 2, 3])
def test_fec_chain_decodes_error_free_bits_at_every_length(cr, vext, monkeypatch):
    """The encoder's coded bits straight into the decoder's deinterleaver
    and Viterbi, no waveform in between.
    """
    from lrfhss import encoder
    monkeypatch.setattr(fec, '_HAVE_VEXT', fec._HAVE_VEXT and vext)
    rng = np.random.default_rng(cr)
    for n in range(1, 41):
        wh = encoder._whiten_payload(rng.integers(0, 2, 8*n), n)
        padded = np.concatenate([wh, fec.crc16(wh), np.zeros(6, int)])
        coded = encoder._con_encode_payload(padded, cr)
        # the decoder sizes the coded stream from the header with this
        assert len(coded) == int(np.ceil(len(padded)*[6/5, 3/2, 2, 3][cr]))
        rx = fec.deinterleave_payload(
            encoder._interleave_payload(coded).astype(float), len(coded))
        info, ok = fec.viterbi_decode_payload(2*rx - 1, CR=cr)
        assert ok and np.array_equal(info, wh), 'CR=%d, %d bytes' % (cr, n)
