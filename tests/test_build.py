# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Checks on the installed package itself, not the receiver."""
import importlib.metadata
import os

import lrfhss


def test_version_matches_the_installed_metadata():
    assert lrfhss.__version__ == importlib.metadata.version('lrfhss')


def test_compiled_core_is_built_where_required():
    """CI and the wheel builds set LRFHSS_REQUIRE_EXT=1, so there a missing
    extension (or tests importing the uncompiled source tree) fails loudly
    instead of silently running the numpy fallback."""
    if os.environ.get('LRFHSS_REQUIRE_EXT') == '1':
        assert lrfhss.config._HAVE_VEXT_LOCAL
