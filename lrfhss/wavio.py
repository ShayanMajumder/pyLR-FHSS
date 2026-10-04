# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT

import numpy as np

from . import config as cfg


def _parse_wav_header(fn):
    """Minimal RIFF/WAVE header parser -- reads only the header chunks (a few
    dozen to ~100 bytes), never the data payload.
    """
    import struct
    with open(fn, 'rb') as f:
        riff = f.read(12)
        if riff[0:4] != b'RIFF' or riff[8:12] != b'WAVE':
            raise ValueError('not a RIFF/WAVE file: %s' % fn)
        data_offset = None
        data_size = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            cid, csize = struct.unpack('<4sI', hdr)
            if cid == b'fmt ':
                fmt_data = f.read(csize)
                (audio_format, n_channels, sample_rate, byte_rate,
                 block_align, bits_per_sample) = struct.unpack('<HHIIHH', fmt_data[:16])
                if csize % 2 == 1:
                    f.read(1)
            elif cid == b'data':
                data_offset = f.tell()
                data_size = csize
                break
            else:
                f.seek(csize + (csize % 2), 1)
        if data_offset is None:
            raise ValueError('no data chunk found in %s' % fn)
        return dict(n_channels=n_channels, sample_rate=sample_rate,
                   bits_per_sample=bits_per_sample, audio_format=audio_format,
                   data_offset=data_offset, data_size=data_size)


def _wav_to_raw_iq(fn):
    """Convert a WAV capture (IEEE-float, I/Q as 2 channels) directly to the
    raw interleaved float32 array load_frontend expects (a[0::2]=I,
    a[1::2]=Q), in memory.
    """
    from scipy.io import wavfile
    rate, data = wavfile.read(fn)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError('expected 2-channel (I/Q) WAV, got shape %s' % (data.shape,))
    if abs(rate - cfg.FS_CAPTURE) > 1:
        print('  [warn] WAV sample rate %d Hz != expected %d Hz -- results may be wrong'
              % (rate, int(cfg.FS_CAPTURE)))
    interleaved = np.empty(data.shape[0]*2, dtype=np.float32)
    interleaved[0::2] = data[:, 0]
    interleaved[1::2] = data[:, 1]
    return interleaved


class _WavChunkReader:
    """Streaming reader over a float32 2-channel WAV's data chunk."""
    def __init__(self, fn):
        info = _parse_wav_header(fn)
        if info['n_channels'] != 2:
            raise ValueError('expected 2-channel (I/Q) WAV, got %d channels' % info['n_channels'])
        if info['audio_format'] != 3 or info['bits_per_sample'] != 32:
            raise ValueError('expected IEEE float32 WAV (format=3, 32-bit), got '
                             'format=%d bits=%d -- streaming reader only supports '
                             'the float32 layout this receiver writes/expects' %
                             (info['audio_format'], info['bits_per_sample']))
        if abs(info['sample_rate'] - cfg.FS_CAPTURE) > 1:
            print('  [warn] WAV sample rate %d Hz != expected %d Hz -- results may be wrong'
                 % (info['sample_rate'], int(cfg.FS_CAPTURE)))
        self.fn = fn
        self.data_offset = info['data_offset']
        self.n_samples_total = info['data_size'] // 4   # total float32 values (both channels interleaved)
        self.shape = (self.n_samples_total,)
        self._fh = open(fn, 'rb', buffering=1024*1024)

    def read_window(self, start, n):
        """Seek to sample index `start` (in the interleaved float32 stream) and
        read exactly `n` float32 values via explicit file positioning
        (os.lseek + os.read on the persistent handle), not a fresh
        np.fromfile(path,...) open per call.
        """
        start = max(0, min(start, self.n_samples_total))
        n = max(0, min(n, self.n_samples_total - start))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        byte_offset = self.data_offset + start*4
        self._fh.seek(byte_offset, 0)
        raw_bytes = self._fh.read(n*4)
        return np.frombuffer(raw_bytes, dtype=np.float32)

    def __getitem__(self, sl):
        start = 0 if sl.start is None else sl.start
        stop = self.n_samples_total if sl.stop is None else sl.stop
        return self.read_window(start, stop - start)

    def close(self):
        self._fh.close()
