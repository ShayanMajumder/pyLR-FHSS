# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Live capture from an SDR."""
import collections
import contextlib
import ctypes
import io
import multiprocessing as mp
import queue
import signal
import threading
import time

import numpy as np
import scipy.signal as sp

from . import config

STRICT_RATES = {'airspy'}
Packet = collections.namedtuple('Packet', 'payload snr_db')   # SNR in 125 kHz
from .pipeline import DecodeOptions, decode

try:
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_OVERFLOW
except ImportError:
    raise ImportError(
        'lrfhss.sdr needs SoapySDR, a system package with no working pip '
        'build. Either rebuild this venv with --system-site-packages, or '
        'symlink SoapySDR.py and _SoapySDR*.so from '
        '/usr/lib/python3/dist-packages into its site-packages.')


def _ring(ctx, n):
    """n complex64 samples of memory shared with forked processes."""
    return np.frombuffer(ctx.RawArray(ctypes.c_float, 2*n), np.complex64)


def _worker_start():
    """Ignore Ctrl-C (the parent takes it) and run OpenBLAS on one thread:
    its spinning threads slow the decoder ~50x on a busy CPU. Linux only."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        maps = open('/proc/self/maps').read().splitlines()
    except OSError:
        return
    for path in {m.split()[-1] for m in maps if 'openblas' in m.lower()
                 and '.so' in m}:
        with contextlib.suppress(OSError):
            lib = ctypes.CDLL(path)
            for name in ('scipy_openblas_set_num_threads64_',
                         'openblas_set_num_threads64_',
                         'openblas_set_num_threads'):
                if hasattr(lib, name):
                    getattr(lib, name)(1)


class LiveReceiver:
    """Decode LR-FHSS off the air, as an iterator of payloads.

    The SDR thread only writes samples into a shared ring, so the radio is
    never kept waiting. One process mixes and decimates that ring into a
    second one holding buffer_s seconds; another decodes overlapping windows
    from it, skipping ahead only if it falls more than buffer_s behind.
    """
    RAW_BUFFER_S = 4.0
    BLOCK = 16384                      # decimated samples made at a time

    def __init__(self, freq_hz, device='airspy', *, channel=0, gain=None,
                 antenna=None, bandwidth_hz=None, offset_hz=97e3,
                 buffer_s=30.0):
        capture_fs = config.FS_CAPTURE
        self.started = time.time()
        self.buffer_s = buffer_s
        self._stats = dict(packets=0, duplicates=0, overflows=0)
        self._recent = collections.deque(maxlen=64)   # (key, time) reported

        self.packet_s = (config.HDR_COUNT*config.STAY_HDR +
                         8*config.STAY_DATA)/config.FS
        self._window = int(2*self.packet_s*config.FS)   # decimated samples
        self._step = int(self.packet_s*config.FS)
        self._offset = offset_hz/capture_fs              # cycles per sample

        ctx = mp.get_context('fork')
        raw_block = self.BLOCK*config.DECIM
        self._raw = _ring(ctx, raw_block*int(np.ceil(
            self.RAW_BUFFER_S*capture_fs/raw_block)))
        self._iq = _ring(ctx, self.BLOCK*int(np.ceil(
            max(buffer_s, 3*self.packet_s)*config.FS/self.BLOCK)))
        self._heads = ctx.RawArray(ctypes.c_int64, 2)    # samples in each ring
        self._counts = ctx.RawArray(ctypes.c_int64, 3)   # windows, skipped, overruns
        self._decoded = ctx.Queue()
        self._workers = [ctx.Process(target=f, daemon=True)
                         for f in (self._decimate_forever, self._decode_forever)]
        for p in self._workers:
            p.start()

        try:
            self._radio, self._stream = self._open(
                device, channel, freq_hz - offset_hz, capture_fs, gain,
                antenna, bandwidth_hz)
        except Exception:
            for p in self._workers:          # do not leak them on a failed open
                p.terminate()
            raise

        self._stop = threading.Event()
        self._reader = threading.Thread(target=self._read_forever, daemon=True)
        self._reader.start()

    def _open(self, device, channel, tune_hz, capture_fs, gain, antenna,
              bandwidth_hz):
        """Open the radio and configure it, returning it and its stream."""
        args = dict(driver=device) if isinstance(device, str) else dict(device)
        try:
            radio = SoapySDR.Device(args)
        except Exception as exc:
            plugged = ', '.join(sorted(d.get('driver', '?')
                                       for d in self.list_devices())) or 'none'
            raise RuntimeError('no SDR matched %r (plugged in: %s)'
                               % (args, plugged)) from exc

        rates = list(radio.listSampleRates(SOAPY_SDR_RX, channel))
        if rates and not any(abs(r - capture_fs) < 1 for r in rates):
            listed = ', '.join('%.0f' % r for r in rates)
            if radio.getDriverKey().lower() in STRICT_RATES:
                raise ValueError(
                    'config.FS_CAPTURE is %.0f Hz, which this radio does not '
                    'offer; it has %s. Use retune(..., fs_capture=...) with '
                    'one of those.' % (capture_fs, listed))
            print('  [sdr] %.0f Hz is not advertised by this radio (%s); '
                  'asking for it anyway' % (capture_fs, listed))
        radio.setSampleRate(SOAPY_SDR_RX, channel, capture_fs)
        actual = radio.getSampleRate(SOAPY_SDR_RX, channel)
        if abs(actual - capture_fs) > 1:
            raise ValueError('asked this radio for %.0f Hz, it is running at '
                             '%.0f Hz' % (capture_fs, actual))

        radio.setFrequency(SOAPY_SDR_RX, channel, tune_hz)
        if antenna is not None:
            radio.setAntenna(SOAPY_SDR_RX, channel, antenna)
        if bandwidth_hz is not None:
            radio.setBandwidth(SOAPY_SDR_RX, channel, bandwidth_hz)
        if radio.hasGainMode(SOAPY_SDR_RX, channel):
            radio.setGainMode(SOAPY_SDR_RX, channel, gain is None)   # AGC
        if isinstance(gain, dict):
            for name, value in gain.items():
                radio.setGain(SOAPY_SDR_RX, channel, name, value)
        elif gain is not None:
            radio.setGain(SOAPY_SDR_RX, channel, gain)
        return radio, radio.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [channel])

    @staticmethod
    def list_devices():
        """Every SDR SoapySDR can see, as a list of device-argument dicts."""
        return [dict(d) for d in SoapySDR.Device.enumerate()]

    @property
    def elapsed(self):
        """Seconds since the receiver started."""
        return time.time() - self.started

    @property
    def stats(self):
        windows, skipped, overruns = self._counts
        return dict(self._stats, windows=windows, skipped=skipped,
                    overruns=overruns)

    def summary(self):
        s = self.stats
        return ('ran %.1f s: %d packet(s) from %d window(s) (%d duplicate(s) '
                'from overlapping windows removed), %d SDR overflow(s), '
                '%d buffer overrun(s), %d window(s) skipped'
                % (self.elapsed, s['packets'], s['windows'], s['duplicates'],
                   s['overflows'], s['overruns'], s['skipped']))

    def close(self):
        self._stop.set()
        self._reader.join(timeout=3)   # let it leave readStream first
        for p in self._workers:
            p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __iter__(self):
        """Yield a Packet(payload, snr_db) for each CRC-valid packet. Ctrl-C
        ends the loop."""
        try:
            while not self._stop.is_set():
                try:
                    payloads = self._decoded.get(timeout=0.5)
                except queue.Empty:
                    self._check_alive()
                    continue
                for payload, t, key, snr_db in payloads:
                    # Windows overlap by a packet length, and LR-FHSS
                    # decodes from part of a packet, so one transmission
                    # can come out of two windows.
                    if any(k == key and abs(t - tk) < self.packet_s
                           for k, tk in self._recent):
                        self._stats['duplicates'] += 1
                        continue
                    self._recent.append((key, t))
                    self._stats['packets'] += 1
                    yield Packet(payload, snr_db)
        except KeyboardInterrupt:
            return

    def _check_alive(self):
        if not self._reader.is_alive():
            raise RuntimeError('the SDR reader thread stopped')
        for p, name in zip(self._workers, ('decimator', 'decoder')):
            if not p.is_alive():
                raise RuntimeError('the %s process died (exit code %s)'
                                   % (name, p.exitcode))

    def _read_forever(self):
        """In a thread: read the radio straight into the raw ring."""
        ring, n = self._raw, len(self._raw)
        mtu = int(self._radio.getStreamMTU(self._stream))
        written = 0
        self._radio.activateStream(self._stream)
        while not self._stop.is_set():
            at = written % n
            got = self._radio.readStream(self._stream, [ring[at:]],
                                         min(mtu, n - at), timeoutUs=1000000)
            if got.ret == SOAPY_SDR_OVERFLOW:
                self._stats['overflows'] += 1     # the radio dropped samples
            if got.ret > 0:
                written += got.ret
                self._heads[0] = written
        self._radio.deactivateStream(self._stream)
        self._radio.closeStream(self._stream)

    def _decimate_forever(self):
        """In a process: mix the raw ring to 0 Hz and decimate it into the
        iq ring, keeping up with the radio. Decimated sample j is raw sample
        j*DECIM, so a stretch the radio overran is left as zeros."""
        _worker_start()
        D, raw, iq = config.DECIM, self._raw, self._iq
        B = self.BLOCK*D
        fir = sp.firwin(20*D + 1, 1/D, window=('kaiser', 5.0))   # resample_poly's
        tail = np.zeros(len(fir) - 1, np.complex64)
        tone = np.exp(-2j*np.pi*self._offset*np.arange(B)).astype(np.complex64)
        done = 0                                        # raw samples decimated
        while True:
            head = self._heads[0]
            if head - done > len(raw) - B:              # about to be overwritten
                done = self._overrun(done, (head//B - 1)*B)
                tail[:] = 0
                continue
            if head - done < B:
                time.sleep(0.01)
                continue
            phase = np.complex64(np.exp(-2j*np.pi*((self._offset*done) % 1)))
            x = raw[done % len(raw):][:B]*tone*phase
            if self._heads[0] - done > len(raw):         # overwritten as we read
                continue
            y = sp.upfirdn(fir, np.concatenate((tail, x)), 1, D)
            tail = x[-len(tail):]
            j = done//D
            iq[j % len(iq):][:self.BLOCK] = y[len(tail)//D:][:self.BLOCK]
            done += B
            self._heads[1] = j + self.BLOCK

    def _overrun(self, done, to):
        """Skip the decimator from raw sample done to to, zeroing the gap."""
        self._counts[2] += 1
        D, iq = config.DECIM, self._iq
        gap = np.arange(done//D, to//D)
        iq[gap[-len(iq):] % len(iq)] = 0
        self._heads[1] = to//D
        return to

    def _decode_forever(self):
        """In a process: decode overlapping windows of the iq ring."""
        _worker_start()
        options = DecodeOptions()
        iq, win, step = self._iq, self._window, self._step
        start = 0
        while True:
            head = self._heads[1]
            if head - start > len(iq) - step:           # buffer_s behind
                latest = (head - win)//step*step
                self._counts[1] += (latest - start)//step
                start = latest
            if head < start + win:
                time.sleep(0.05)
                continue
            x = np.take(iq, np.arange(start, start + win), mode='wrap')
            if self._heads[1] - start > len(iq):         # overwritten as we read
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    found = decode(x.astype(complex), options)
            except Exception as exc:                   # never kill the pipeline
                print('  decoder error: %s' % exc)
                found = []
            self._counts[0] += 1
            self._decoded.put([
                (bytes(p['bytes']), (start + p['t0'])/config.FS,
                 (bytes(p['bytes']), tuple((p.get('header') or {}).get('hopseq') or ())),
                 float('nan') if p.get('snr_db') is None else p['snr_db'])
                for p in found if p['crc']])
            start += step
