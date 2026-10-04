# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""FFT entry points for the receiver."""
import scipy.fft as _sfft


def _fft(a, n=None, axis=-1):
    return _sfft.fft(a, n=n, axis=axis, workers=1)


def _ifft(a, n=None, axis=-1):
    return _sfft.ifft(a, n=n, axis=axis, workers=1)
