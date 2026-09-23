# Part of the lrfhss receiver package.
"""FFT entry points for the receiver.

Plain scipy.fft, single-threaded.

An OpenCL GPU backend lived here for a while. It worked and was
numerically exact, but it was worth only ~11% of a decode -- the FFTs are
~16% of the work, the rest being sequential C++ Viterbi and IIR filtering
that a GPU does not help -- and it aborted the interpreter when several
captures were decoded on threads. Not worth the complexity.
"""
import scipy.fft as _sfft


def _fft(a, n=None, axis=-1):
    return _sfft.fft(a, n=n, axis=axis, workers=1)


def _ifft(a, n=None, axis=-1):
    return _sfft.ifft(a, n=n, axis=axis, workers=1)
