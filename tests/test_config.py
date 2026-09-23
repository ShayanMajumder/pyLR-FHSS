"""config.retune(): the derived front end, and the traps around it."""
import numpy as np
import pytest

import lrfhss
from lrfhss import config as cfg


@pytest.mark.parametrize('bw_hz,decim,fs', [
    (39_060, 18, 3e6/18),
    (183_590, 9, 3e6/9),
    (386_720, 6, 3e6/6),
    (722_660, 3, 1e6),
    (1_574_200, 1, 3e6),
])
def test_decimation_is_the_largest_that_still_passes_the_signal(bw_hz, decim, fs):
    cfg.retune(bw_hz)
    assert cfg.DECIM == decim
    assert cfg.FS == pytest.approx(fs)
    assert cfg.FS >= bw_hz, 'front-end rate must span the occupied bandwidth'


def test_every_derived_value_moves_with_the_bandwidth():
    """retune() exists so these cannot be set piecemeal and go
    inconsistent -- which is how several decode bugs started."""
    cfg.retune(722_660)
    assert cfg.SMBL == round(cfg.FS/cfg.BW)
    assert cfg.LOOKDIST == round(cfg.SMBL/4)
    assert cfg.STAY_HDR == round(0.233472*cfg.FS)
    assert cfg.STAY_DATA == round(0.1024*cfg.FS)
    assert len(cfg._WIN_HDR) == cfg.STAY_HDR
    assert len(cfg.SYNC_OFF) == cfg.MF_NSYNC
    assert cfg._COARSE_RAMP.shape == (len(cfg._COARSE_HZ), cfg.MF_NSYNC)


def test_hop_window_covers_the_band_but_stays_inside_nyquist():
    for bw in (39_060, 722_660, 1_574_200):
        cfg.retune(bw)
        assert cfg.ALLBW > bw, 'window must exceed the occupied bandwidth'
        assert cfg.ALLBW <= cfg.FS, 'window cannot exceed the sampled band'


def test_hop_window_has_margin_beyond_the_outermost_hop():
    """A snug 1.01x left ~2 kHz at the edge and the outermost hops fell
    off it, so bw=39.06 kHz decoded nothing."""
    cfg.retune(39_060)
    assert cfg.ALLBW/2 - 39_060/2 > 2_000


def test_sync_word_accepts_hex_and_bits_identically():
    cfg.retune(136_720, sync_word='12AD101B')
    from_hex = cfg.SYNC_WORD.copy()
    cfg.retune(136_720, sync_word=''.join(f'{b:08b}' for b in bytes.fromhex('12AD101B')))
    assert np.array_equal(from_hex, cfg.SYNC_WORD)
    assert cfg.MF_NSYNC == 32


def test_sync_word_change_rebuilds_the_matched_filter():
    cfg.retune(136_720, sync_word='12AD101B')
    a = cfg.SYNC_VEC.copy()
    cfg.retune(136_720, sync_word='2C0F7995')
    assert not np.allclose(a, cfg.SYNC_VEC)


def test_header_count_is_only_changed_when_given():
    cfg.retune(136_720, hdr_count=4)
    assert cfg.HDR_COUNT == 4
    cfg.retune(722_660)                 # no hdr_count -> keep what was set
    assert cfg.HDR_COUNT == 4


def test_retired_cores_keyword_is_accepted_and_ignored():
    cfg.retune(136_720, cores=8)        # decoding is single-threaded now
    assert cfg.DECIM == 18


def test_payload_budget_scales_with_the_work_per_hypothesis():
    """Every hypothesis in the payload search costs ~FS, so a budget
    tuned at DECIM=18 starves DECIM=1 and truncates a real search."""
    cfg.retune(39_060)
    narrow = cfg.PAYLOAD_TIME_BUDGET_S
    cfg.retune(1_574_200)
    assert cfg.PAYLOAD_TIME_BUDGET_S > narrow


def test_setting_parameters_on_the_package_does_not_affect_decoding():
    """Modules read config.FS at call time; lrfhss.FS = ... binds a name
    nothing looks at. Worth pinning so the README stays true."""
    cfg.retune(136_720)
    lrfhss.FS = 12345
    assert cfg.FS != 12345
    del lrfhss.FS
