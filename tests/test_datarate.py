# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""LoRaWAN data rates."""
import io
import contextlib

import numpy as np
import pytest

import lrfhss
from lrfhss import config as cfg

RNG = np.random.default_rng(0)


@pytest.mark.parametrize('region,dr', [('US915', 5), ('US915', 6)])
def test_round_trip_at_each_us915_datarate(region, dr):
    res = _round_trip(dr, region)
    assert b'hello world' in [bytes(r['bytes']) for r in res if r['crc']]


@pytest.mark.parametrize('region,dr,bw_khz,cr,replicas', [
    ('EU868', 8, 136.72, 3, 3),      # CR 1/3
    ('EU868', 9, 136.72, 1, 2),      # CR 2/3
    ('EU868', 10, 335.94, 3, 3),
    ('EU868', 11, 335.94, 1, 2),
    ('US915', 5, 1523.4, 3, 3),
    ('US915', 6, 1523.4, 1, 2),
    ('AU915', 5, 1523.4, 3, 3),
])
def test_datarate_table_matches_the_regional_parameters(region, dr, bw_khz,
                                                        cr, replicas):
    spec = cfg.datarate(dr, region)
    assert spec['bw_khz'] == bw_khz
    assert spec['cr'] == cr
    assert spec['header_replicas'] == replicas


def test_the_slower_rate_of_a_pair_uses_more_header_replicas():
    """DR8/DR10 are CR 1/3 with 3 replicas, DR9/DR11 CR 2/3 with 2 -- the
    robust rate spends more airtime on the header as well as the code.
    """
    for slow, fast in ((8, 9), (10, 11)):
        a, b = cfg.datarate(slow), cfg.datarate(fast)
        assert a['bw_khz'] == b['bw_khz']
        assert a['header_replicas'] > b['header_replicas']
        assert cfg.CR_RATES[a['cr']] == '1/3'
        assert cfg.CR_RATES[b['cr']] == '2/3'


def test_region_is_respected():
    assert cfg.datarate(5, 'US915')['bw_khz'] == 1523.4
    with pytest.raises(ValueError):
        cfg.datarate(5, 'EU868')          # DR5 is not LR-FHSS in EU868


def test_unknown_datarate_names_what_is_available():
    with pytest.raises(ValueError, match='DR99'):
        cfg.datarate(99)
    with pytest.raises(ValueError, match='EU868'):
        cfg.datarate(99, 'eu868')


def test_datarate_returns_a_copy():
    """Callers must not be able to edit the shared table by accident."""
    spec = cfg.datarate(8)
    spec['header_replicas'] = 99
    assert cfg.datarate(8)['header_replicas'] == 3


def test_retune_dr_sets_the_front_end_and_the_replica_count():
    spec = cfg.retune_dr(8)
    assert cfg.HDR_COUNT == spec['header_replicas'] == 3
    assert cfg.DECIM == 18 and cfg.FS == pytest.approx(3e6/18)
    cfg.retune_dr(5, region='US915')
    assert cfg.DECIM == 1 and cfg.HDR_COUNT == 3


def test_retune_dr_matches_retune_with_the_same_numbers():
    cfg.retune_dr(11)
    a = (cfg.FS, cfg.DECIM, cfg.ALLBW, cfg.HDR_COUNT, cfg.STAY_HDR)
    cfg.retune(335.94e3, hdr_count=2)
    assert (cfg.FS, cfg.DECIM, cfg.ALLBW, cfg.HDR_COUNT, cfg.STAY_HDR) == a


def _round_trip(dr, region='EU868'):
    cfg.retune_dr(dr, region=region)
    iq, meta = lrfhss.encode(b'hello world', dr=dr, region=region)
    lead = np.zeros(int(0.15*cfg.FS), complex)
    buf = np.concatenate([lead, iq, lead])
    buf += (RNG.standard_normal(len(buf)) +
            1j*RNG.standard_normal(len(buf)))*1e-3
    with contextlib.redirect_stdout(io.StringIO()):
        res = lrfhss.decode(buf, lrfhss.DecodeOptions())
    return res


@pytest.mark.parametrize('dr', [8, 9, 10, 11])
def test_round_trip_at_each_eu868_datarate(dr):
    res = _round_trip(dr)
    assert b'hello world' in [bytes(r['bytes']) for r in res if r['crc']]


@pytest.mark.parametrize('dr', [8, 9, 10, 11])
def test_decoded_header_reports_the_datarate_coding_rate(dr):
    """CR is carried in the header, so a DR-encoded packet should come
    back announcing that DR's coding rate.
    """
    res = _round_trip(dr)
    hdr = next(r['header'] for r in res if r['crc'])
    assert hdr['CR'] == cfg.datarate(dr)['cr']


def test_explicit_parameters_still_override_a_datarate():
    """Most of the LR-FHSS parameter space has no DR name, so the direct
    arguments have to keep working -- and win when both are given.
    """
    cfg.retune(722_660, hdr_count=4)
    _, meta = lrfhss.encode(b'x', dr=8, bw_khz=722.66, header_count=4, CR=0)
    assert meta['bw_khz'] == 722.66
    assert meta['header_count'] == 4
    assert meta['CR'] == 0
