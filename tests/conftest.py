# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Shared fixtures."""
import pathlib

import pytest

import lrfhss

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'examples' / 'data'


@pytest.fixture(autouse=True)
def default_config():
    """Every test starts from a known front end."""
    lrfhss.config.retune(136_720, hdr_count=3)
    yield
    lrfhss.config.retune(136_720, hdr_count=3)


@pytest.fixture
def quiet(capsys):
    """The receiver is chatty; swallow it unless a test fails."""
    yield
    capsys.readouterr()
