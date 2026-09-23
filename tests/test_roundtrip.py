"""Encode then decode: the end-to-end check that needs no recorded data.

These are the tests that would have caught most of the decode bugs this
receiver has had, because every one of them showed up as "the header
locks but the payload never verifies".
"""
import numpy as np
import pytest

import lrfhss

RNG = np.random.default_rng(0)


def transmit(payload, bw_khz=136.72, header_count=3, CR=3, grid=1,
             hop_seq_id=77, noise=1e-3, lead_sec=0.15):
    """Generate a packet and surround it with silence, as a capture would."""
    lrfhss.config.retune(bw_khz*1e3, hdr_count=header_count)
    iq, meta = lrfhss.encode(payload, bw_khz=bw_khz, header_count=header_count,
                             CR=CR, grid=grid, hop_seq_id=hop_seq_id)
    lead = np.zeros(int(lead_sec*lrfhss.config.FS), complex)
    buf = np.concatenate([lead, iq, lead])
    if noise:
        buf = buf + (RNG.standard_normal(len(buf)) +
                     1j*RNG.standard_normal(len(buf)))*noise
    return buf, meta


def payloads_of(capture):
    res = lrfhss.decode(capture, lrfhss.DecodeOptions(sensitive_retry=False))
    return [bytes(r['bytes']) for r in res if r['crc']], res


def test_round_trip_recovers_the_payload(quiet):
    msg = b'hello world'
    buf, _ = transmit(msg)
    got, _ = payloads_of(buf)
    assert msg in got


@pytest.mark.parametrize('header_count', [1, 2, 3, 4])
def test_round_trip_for_every_header_replica_count(header_count, quiet):
    """HDR_COUNT is not carried in the header. A wrong value puts the
    payload window a whole dwell out, which decodes noise that can clear
    the 16-bit CRC by chance rather than failing cleanly."""
    msg = b'replica test'
    buf, _ = transmit(msg, header_count=header_count)
    got, _ = payloads_of(buf)
    assert msg in got


@pytest.mark.parametrize('CR', [0, 1, 2, 3])
def test_round_trip_for_every_coding_rate(CR, quiet):
    msg = b'coding rate'
    buf, _ = transmit(msg, CR=CR)
    got, _ = payloads_of(buf)
    assert msg in got


@pytest.mark.parametrize('bw_khz', [39.06, 85.94, 136.72, 183.59,
                                    335.94, 386.72, 722.66, 1523.4, 1574.2])
def test_round_trip_across_bandwidths(bw_khz, quiet):
    msg = b'bandwidth'
    buf, _ = transmit(msg, bw_khz=bw_khz)
    got, _ = payloads_of(buf)
    assert msg in got


@pytest.mark.parametrize('n', [1, 11, 40])
def test_round_trip_for_various_payload_lengths(n, quiet):
    msg = bytes(range(65, 65+min(n, 26)))*(1 + n//26)
    msg = msg[:n]
    buf, _ = transmit(msg)
    got, _ = payloads_of(buf)
    assert msg in got


def test_hop_seed_is_recovered_from_the_header(quiet):
    """The receiver rebuilds the whole hop schedule from the seed in the
    header; if that were not recovered the payload could not be gathered."""
    buf, meta = transmit(b'seeded', hop_seq_id=123)
    res = lrfhss.decode(buf, lrfhss.DecodeOptions(sensitive_retry=False))
    confirmed = [r for r in res if r['crc']]
    assert confirmed
    seen = int(''.join(map(str, confirmed[0]['header']['hopseq'])), 2)
    assert seen == 123


def test_decoder_reports_the_transmitted_parameters(quiet):
    buf, meta = transmit(b'metadata', header_count=2, CR=1, grid=1)
    res = lrfhss.decode(buf, lrfhss.DecodeOptions(sensitive_retry=False))
    hdr = next(r['header'] for r in res if r['crc'])
    assert hdr['payloadlen'] == len(b'metadata')
    assert hdr['CR'] == 1
    assert hdr['grid'] == 1


def test_noise_only_input_decodes_nothing(quiet):
    """The accept gate must not invent packets out of noise."""
    lrfhss.config.retune(136_720, hdr_count=3)
    n = int(1.2*lrfhss.config.FS)
    noise = (RNG.standard_normal(n) + 1j*RNG.standard_normal(n))*1e-3
    got, _ = payloads_of(noise)
    assert got == []


def test_corrupted_payload_is_rejected(quiet):
    """Damage the payload but leave the headers intact: the header still
    locks, and the payload CRC16 must refuse it."""
    msg = b'will be broken'
    buf, meta = transmit(msg, noise=0, lead_sec=0.15)
    cfg = lrfhss.config
    # Payload slots start after the header replicas, which start after the
    # lead-in. Wipe ALL of them: the code is strong enough that damaging
    # one fragment is often corrected, which is the point of having it.
    start = int(0.15*cfg.FS) + meta['header_count']*cfg.STAY_HDR
    buf[start:start + meta['num_frags']*cfg.STAY_DATA] = 0.0
    got, _ = payloads_of(buf)
    assert msg not in got


def test_two_packets_in_one_capture_are_both_found(quiet):
    """LR-FHSS is built for concurrent senders, and the dedup rules have
    broken this before by suppressing a real packet as a duplicate."""
    a, _ = transmit(b'first', hop_seq_id=77, lead_sec=0.1, noise=0)
    b, _ = transmit(b'second', hop_seq_id=300, lead_sec=0.1, noise=0)
    gap = np.zeros(int(0.2*lrfhss.config.FS), complex)
    buf = np.concatenate([a, gap, b])
    buf += (RNG.standard_normal(len(buf)) + 1j*RNG.standard_normal(len(buf)))*1e-3
    got, _ = payloads_of(buf)
    assert b'first' in got and b'second' in got


@pytest.mark.xfail(strict=True, reason='encoder CR=0 (5/6) is content-dependent')
def test_punctured_rate_survives_arbitrary_payload_content(quiet):
    """CR=0 still depends on what the bytes are, and should not.

    Measured over 24 random payloads at essentially zero noise, after the
    BW=488.28125 fix:

        CR=0  2/24      CR=1  24/24      CR=2  23/24      CR=3  24/24

    So correcting the symbol rate fixed CR=1 (was 20/24) but left CR=0
    untouched, and CR=2 is one short of clean. A correct encoder would be
    24/24 everywhere at this noise level. Only CR=0 is asserted here
    because it is the one that fails reliably; pinning the others either
    way would make the suite flaky.

    The FEC chain itself is exonerated: feeding the encoder's punctured,
    interleaved bits straight into the decoder's deinterleaver and Viterbi
    recovers the payload at every CR, so the fault is in the waveform or
    the timing, not in puncturing.
    """
    CR = 0
    rng = np.random.default_rng(11)
    for _ in range(8):
        n = int(rng.integers(1, 30))
        msg = bytes(rng.integers(0, 256, n).tolist())
        buf, _ = transmit(msg, CR=CR)
        got, _ = payloads_of(buf)
        assert msg in got, 'CR=%d failed on %d-byte payload' % (CR, n)


@pytest.mark.parametrize('CR', [1, 3])
def test_well_coded_rates_survive_arbitrary_payload_content(CR, quiet):
    """Control for the test above: CR=1 and CR=3 are content-independent,
    which is what localises the fault rather than blaming the encoder as
    a whole. CR=2 is left out -- it is 23/24, so asserting it would
    flake."""
    rng = np.random.default_rng(11)
    for _ in range(6):
        n = int(rng.integers(1, 30))
        msg = bytes(rng.integers(0, 256, n).tolist())
        buf, _ = transmit(msg, CR=CR)
        got, _ = payloads_of(buf)
        assert msg in got
