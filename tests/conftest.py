"""Shared fixtures.

The suite is self-contained: most tests generate their own signal with
the encoder, and the one real recording is bundled in examples/data. Nothing
here depends on the full capture sweep, so everything runs anywhere and
nothing is skipped.
"""
import pathlib

import pytest

import lrfhss

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'examples' / 'data'


@pytest.fixture(autouse=True)
def default_config():
    """Every test starts from a known front end.

    config is process-global and retune() mutates it, so without this a
    test that retunes would silently change the meaning of the next one.
    """
    lrfhss.config.retune(136_720, hdr_count=3)
    yield
    lrfhss.config.retune(136_720, hdr_count=3)


@pytest.fixture
def quiet(capsys):
    """The receiver is chatty; swallow it unless a test fails."""
    yield
    capsys.readouterr()
