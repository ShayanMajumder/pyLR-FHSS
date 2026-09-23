"""Live capture from an SDR.

Optional, and deliberately not imported by `import lrfhss`: it needs
SoapySDR, a system package with no working pip build.

    lrfhss.config.retune(39_060, sync_word='12AD101B', hdr_count=3)

    with LiveReceiver(915e6) as radio:
        for payload in radio:
            print(payload)

WHY IT IS BUILT THIS WAY
------------------------
Live reception is a producer/consumer problem with three traps, and this
module is really just the three answers:

  the radio cannot be paused    a reader THREAD does nothing but read, and
                                hands finished windows to a queue

  the decoder cannot keep up    it runs in a separate PROCESS, because the
                                receiver's C++ extension never releases the
                                GIL and a decoder thread would therefore
                                stall the reader

  packets straddle windows      a window is two packets long and starts one
                                packet after the last, so every packet lands
                                whole inside some window

If the decoder does fall behind, the queue fills and the reader drops the
newest window rather than block. A transmitter repeats; the radio cannot
rewind.
"""
import contextlib
import io
import multiprocessing as mp
import queue
import signal
import threading
import time

import numpy as np
import scipy.signal as sp

from . import config

#: Drivers whose listSampleRates() is the whole truth. SoapyAirspy offers
#: only the two rates libairspy has and reports back whatever it was asked
#: for, so its list is the only thing worth believing. SoapyRTLSDR
#: advertises ten rates while the tuner takes anything from roughly 0.9 to
#: 3.2 MSPS, so refusing an unlisted rate there refuses rates that work.
STRICT_RATES = {'airspy'}
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


class LiveReceiver:
    """Decode LR-FHSS off the air, as an iterator of payloads.

    Reads the current `lrfhss.config`, including the capture rate, so
    retune() first. Iterating yields every payload that passes CRC and
    ends on Ctrl-C; `stats` and summary() say what the radio did.

    freq_hz       what the transmitter is on
    device        SoapySDR driver name, or a dict of device arguments to
                  pick one radio out of several: dict(driver='airspy',
                  serial='...'). list_devices() shows what is plugged in.
    channel       receive channel, for radios that have more than one
    gain          None for the radio's own AGC, a number for overall gain,
                  or a dict of element gains -- whose names are per radio,
                  e.g. LNA/MIX/VGA on an Airspy, TUNER on an RTL-SDR
    antenna       antenna name, where a radio has a choice
    bandwidth_hz  analog filter bandwidth, where a radio has one
    offset_hz     how far below the signal to tune, to keep it clear of the
                  SDR's DC spike
    queue_depth   windows in flight before the reader starts dropping
    """

    def __init__(self, freq_hz, device='airspy', *, channel=0, gain=None,
                 antenna=None, bandwidth_hz=None, offset_hz=97e3,
                 queue_depth=3):
        capture_fs = config.FS_CAPTURE
        self.stats = dict(packets=0, windows=0, dropped=0, overflows=0)
        self.started = time.time()

        # A packet is its header replicas plus eight payload fragments.
        # All of the windowing follows from that one number.
        self.packet_s = (config.HDR_COUNT*config.STAY_HDR +
                         8*config.STAY_DATA)/config.FS
        self._window = int(2*self.packet_s*capture_fs)   # raw samples
        self._step = int(self.packet_s*capture_fs)
        # The mixing tone, built once: every window starts from phase zero.
        # Absolute phase does not matter, as the decoder estimates carrier
        # offset and phase per packet.
        self._tone = np.exp(-2j*np.pi*offset_hz*np.arange(self._window) /
                            capture_fs).astype(np.complex64)

        # Fork the decoder BEFORE opening the radio, so it starts from a
        # clean process -- no stream handle, no driver threads -- that has
        # nonetheless inherited the configuration above.
        ctx = mp.get_context('fork')
        self._work = ctx.Queue(maxsize=queue_depth)
        self._decoded = ctx.Queue()
        self._decoder = ctx.Process(target=self._decode_forever, daemon=True)
        self._decoder.start()

        try:
            self._radio, self._stream = self._open(
                device, channel, freq_hz - offset_hz, capture_fs, gain,
                antenna, bandwidth_hz)
        except Exception:
            self._decoder.terminate()     # do not leak it on a failed open
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

        # Every dwell length and hop offset is derived from the capture
        # rate, so a radio running at a rate config does not know about
        # would decode nothing and look like a receiver bug. Check the
        # rates it advertises, because some drivers -- SoapyAirspy among
        # them -- report back whatever they were asked for.
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

    def summary(self):
        return ('ran %.1f s: %d packet(s) from %d window(s), %d window(s) '
                'dropped, %d SDR overflow(s)'
                % (self.elapsed, self.stats['packets'], self.stats['windows'],
                   self.stats['dropped'], self.stats['overflows']))

    def close(self):
        self._stop.set()
        self._reader.join(timeout=3)   # let it leave readStream first
        self._decoder.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __iter__(self):
        """Yield each payload that passes CRC. Ctrl-C ends the loop."""
        try:
            while not self._stop.is_set():
                try:
                    payloads = self._decoded.get(timeout=0.5)
                except queue.Empty:
                    continue
                for payload in payloads:
                    self.stats['packets'] += 1
                    yield payload
        except KeyboardInterrupt:
            return

    def _read_forever(self):
        """Producer, in a thread: cut the stream into overlapping windows."""
        mtu = int(self._radio.getStreamMTU(self._stream))
        buf = np.empty(mtu, np.complex64)
        held, n = [], 0                     # raw samples not yet in a window
        self._radio.activateStream(self._stream)
        while not self._stop.is_set():
            got = self._radio.readStream(self._stream, [buf], mtu,
                                         timeoutUs=1000000)
            # Only the RETURN code means overflow. SOAPY_SDR_OVERFLOW is
            # -4, so testing it against `flags` would mask off the low two
            # bits and count every read that merely carried a timestamp
            # (HAS_TIME == 4) as a lost-sample event.
            if got.ret == SOAPY_SDR_OVERFLOW:
                self.stats['overflows'] += 1      # the radio ran ahead of us
            if got.ret <= 0:                      # overflow or timeout
                continue
            held.append(buf[:got.ret].copy())
            n += got.ret
            if n < self._window:
                continue
            raw = np.concatenate(held)
            held, n = [raw[self._step:]], len(raw) - self._step   # the overlap
            try:
                self._work.put_nowait(raw[:self._window])
                self.stats['windows'] += 1
            except queue.Full:
                self.stats['dropped'] += 1
        self._radio.deactivateStream(self._stream)
        self._radio.closeStream(self._stream)

    def _decode_forever(self):
        """Consumer, in its own process.

        It mixes and decimates too. That is a quarter-second of work per
        window, and doing it here rather than in the reader is what keeps
        the reader inside readStream, where the radio needs it.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)   # the parent takes Ctrl-C
        options = DecodeOptions(sensitive_retry=False)
        while True:
            raw = self._work.get()
            iq = sp.resample_poly(raw*self._tone, 1, config.DECIM)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    found = decode(iq.astype(complex), options)
            except Exception as exc:                   # never kill the pipeline
                print('  decoder error: %s' % exc)
                continue
            self._decoded.put([bytes(p['bytes']) for p in found if p['crc']])
