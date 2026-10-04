# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Decode the bundled recording."""
import io
import contextlib

import numpy as np
import pytest
from scipy.io import wavfile

import lrfhss
from conftest import DATA

CAPTURE = DATA / 'capture_39kHz_hdr3.wav'
EXPECTED = b'hello world'


@pytest.fixture(scope='module')
def capture():
    fs, raw = wavfile.read(CAPTURE)
    iq = raw[:, 0].astype(np.float64) + 1j*raw[:, 1].astype(np.float64)
    return fs, iq


def decode(iq, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return lrfhss.decode(iq, lrfhss.DecodeOptions(**kw))


def test_fixture_rate_matches_the_retuned_front_end(capture):
    """If these drift apart the capture is being decoded at the wrong
    rate, which looks like a decode bug rather than a fixture problem.
    """
    fs, _ = capture
    lrfhss.config.retune(39_060)
    assert abs(lrfhss.config.FS - fs) < 1


def test_real_capture_decodes(capture):
    fs, iq = capture
    lrfhss.config.retune(39_060, sync_word='12AD101B', hdr_count=3)
    got = [bytes(r['bytes']) for r in decode(iq) if r['crc']]
    assert EXPECTED in got


def test_real_capture_header_fields(capture):
    fs, iq = capture
    lrfhss.config.retune(39_060, sync_word='12AD101B', hdr_count=3)
    hdr = next(r['header'] for r in decode(iq) if r['crc'])
    assert hdr['payloadlen'] == len(EXPECTED)
    assert hdr['grid'] == 1          # 3.9 kHz grid (nonFCC)
    assert hdr['hop'] == 1


def test_decoding_is_deterministic(capture):
    """Results once depended on how many workers were running. Nothing is
    parallel now; this pins the property so it cannot come back.
    """
    fs, iq = capture
    runs = []
    for _ in range(2):
        lrfhss.config.retune(39_060, sync_word='12AD101B', hdr_count=3)
        runs.append([(r['t0'], r['crc'],
                      bytes(r['bytes']) if r['bytes'] else None)
                     for r in decode(iq)])
    assert runs[0] == runs[1]


def test_wrong_sync_word_finds_nothing(capture):
    """The capture is from a RadioLib SX126x (0x12AD101B). With the other
    sync word the matched filter should report no candidates at all --
    a mismatch must not degrade into a bad decode.
    """
    fs, iq = capture
    lrfhss.config.retune(39_060, sync_word='2C0F7995', hdr_count=3)
    got = [bytes(r['bytes']) for r in decode(iq) if r['crc']]
    assert EXPECTED not in got


def test_wrong_bandwidth_does_not_decode(capture):
    """Retuning to the wrong bandwidth changes the front-end rate, so the
    samples no longer mean what the receiver thinks they do.
    """
    fs, iq = capture
    lrfhss.config.retune(722_660, sync_word='12AD101B', hdr_count=3)
    got = [bytes(r['bytes']) for r in decode(iq) if r['crc']]
    assert EXPECTED not in got


def test_survives_added_noise(capture):
    """Real capture plus extra AWGN: the accept gate is CRC, so this can
    only lose the packet, never fabricate a different one.
    """
    fs, iq = capture
    lrfhss.config.retune(39_060, sync_word='12AD101B', hdr_count=3)
    rng = np.random.default_rng(0)
    noisy = iq + (rng.standard_normal(len(iq)) +
                  1j*rng.standard_normal(len(iq)))*0.01
    got = [bytes(r['bytes']) for r in decode(noisy) if r['crc']]
    assert got == [] or EXPECTED in got
