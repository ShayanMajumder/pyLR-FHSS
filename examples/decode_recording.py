#!/usr/bin/env python3
"""Decode the bundled recording -- as captured, or buried in white noise.

    python3 examples/decode_recording.py

data/capture_39kHz_hdr3.wav is a real over-the-air LR-FHSS packet: 39.06
kHz, three header replicas, carrying "hello world". It ships already
decimated by the front end -- 2 MB at 166.667 kHz rather than 13 MB at
1 MSPS -- so it is read as samples and handed straight to decode().

The two settings below are the ones the receiver cannot work out for
itself. Bandwidth fixes the whole front end; header-replica count is not
carried in the header, and a wrong value puts the payload window a dwell
out, which decodes noise rather than failing cleanly. Grid and coding
rate come from the decoded header, so they are not settings.

SNR_DB adds white noise to see how weak the packet can get. The clean
recording is decoded first to find the packet; its signal power is then
measured from the recording itself -- the energy over the packet minus
the noise floor outside it -- and just enough noise is added to reach
SNR_DB. SNR is signal power over noise power in a 125 kHz reference
bandwidth, the same measure as the curves in simulations/. NOISE_SEED
picks the noise draw.

Each decoded packet is saved as a spectrogram, with every header replica
and payload fragment boxed where the receiver found it. Only packets
whose payload decoded are plotted, so a plot is evidence rather than a
guess. With noise added, a side-by-side of the clean and noisy capture is
saved as well.

For a different capture, edit WAV and the two settings below.
"""
import contextlib
import io
import os

import numpy as np
from scipy.io import wavfile

import lrfhss

HERE = os.path.dirname(os.path.abspath(__file__))
WAV = os.path.join(HERE, 'data', 'capture_39kHz_hdr3.wav')
PLOT_DIR = os.path.join(HERE, 'plots')   # one PNG per decoded packet
BW_KHZ = 39.06           # occupied bandwidth of this recording
HDR_REPLICAS = 3         # header replicas it was transmitted with
SYNC_WORD = '12AD101B'   # RadioLib's SX126x LR-FHSS sync word
SNR_DB = None            # e.g. -22.0: add white noise down to this SNR; None = as recorded
NOISE_SEED = 0           # which noise draw


def decode(capture, plot_dir=None, quiet=False):
    opts = lrfhss.DecodeOptions(plot_spectrograms=plot_dir is not None,
                                plot_dir=plot_dir or '.')
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            return lrfhss.decode(capture, opts)
    return lrfhss.decode(capture, opts)


def packet_samples(hdr):
    """On-air length of a packet in samples: header replicas plus payload
    fragments."""
    cfg = lrfhss.config
    bits = 8*(hdr['payloadlen'] + 2) + 6
    n_frags = int(np.ceil(int(np.ceil(bits*[6/5, 3/2, 2, 3][hdr['CR']]))/48))
    return cfg.HDR_COUNT*cfg.STAY_HDR + n_frags*cfg.STAY_DATA


def add_noise(capture, packet, snr_db, seed):
    """White noise bringing `packet` (a decoded record) down to snr_db.

    Noise floor: the MEDIAN power across frequency. At any instant the
    packet occupies one narrow hop, so the median bin is noise even while
    the packet is on -- no noise-only stretch of the recording is needed
    (this one is barely longer than its packet). For complex white noise
    a periodogram bin is exponential, so mean = median/ln 2.
    Signal power: total energy minus that noise, spread over the packet's
    on-air length. Noise added: whatever density, on top of what the
    recording already has, puts the total at P_sig/(SNR * 125 kHz).
    """
    fs = lrfhss.config.FS
    n = 1024
    frames = capture[:len(capture)//n*n].reshape(-1, n)
    p_noise = np.median(np.abs(np.fft.fft(frames, axis=1))**2)/(n*np.log(2))
    e_sig = np.sum(np.abs(capture)**2) - len(capture)*p_noise
    p_sig = e_sig/packet_samples(packet['header'])
    native = 10*np.log10(p_sig/(p_noise*125e3/fs))
    n0_add = p_sig/(10**(snr_db/10)*125e3) - p_noise/fs
    if n0_add <= 0:
        raise SystemExit('the recording is already below %.1f dB SNR' % snr_db)
    rng = np.random.default_rng(seed)
    noise = np.sqrt(n0_add*fs/2)*(rng.standard_normal(len(capture)) +
                                  1j*rng.standard_normal(len(capture)))
    print('recording: %.1f dB SNR as captured; adding %.1f dB more noise for %.1f dB'
          % (native, native - snr_db, snr_db))
    return capture + noise


def side_by_side(clean, noisy, packet, path):
    """The same stretch of capture clean and noisy, each panel scaled to
    its own noise floor (the noisy one sits tens of dB higher)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fs = lrfhss.config.FS
    t_lo, t_hi, _ = packet['footprint']
    t_lo, t_hi = max(0, int(t_lo)), min(len(clean), int(t_hi))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, x, title, top in ((axes[0], clean, 'recording as captured', 60),
                              (axes[1], noisy, 'white noise added: %.0f dB SNR' % SNR_DB, 20)):
        P, f, t, _ = ax.specgram(x[t_lo:t_hi], NFFT=4096, Fs=fs, noverlap=3072)
        ax.cla()
        db = 10*np.log10(P + 1e-30)
        im = ax.pcolormesh(t + t_lo/fs, f, db - np.median(db), shading='auto',
                           cmap='viridis', vmin=0, vmax=top, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel('time (s)')
        fig.colorbar(im, ax=ax, label='dB above noise floor')
    axes[0].set_ylabel('frequency (Hz)')
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    # Derives the whole front end from the bandwidth: sample rate, symbol
    # length, hop dwell times, sync matched filter, CFO ramp, hop-search
    # window and the payload-search budget.
    lrfhss.config.retune(BW_KHZ*1e3, sync_word=SYNC_WORD, hdr_count=HDR_REPLICAS)

    fs, raw = wavfile.read(WAV)
    capture = raw[:, 0].astype(np.float64) + 1j*raw[:, 1].astype(np.float64)
    assert abs(fs - lrfhss.config.FS) < 1, (
        'recording is %d Hz but retune(%.0f) gives %.0f Hz -- the file would be '
        'decoded at the wrong rate' % (fs, BW_KHZ*1e3, lrfhss.config.FS))

    plot_dir = PLOT_DIR
    if SNR_DB is not None:
        clean = [p for p in decode(capture, quiet=True) if p['crc']]
        if not clean:
            raise SystemExit('the recording does not decode even clean')
        noisy = add_noise(capture, clean[0], SNR_DB, NOISE_SEED)
        plot_dir = os.path.join(PLOT_DIR, 'snr%+.0fdB_seed%d' % (SNR_DB, NOISE_SEED))
        os.makedirs(plot_dir, exist_ok=True)
        side_by_side(capture, noisy, clean[0], os.path.join(plot_dir, 'clean_vs_noisy.png'))
        capture = noisy

    packets = decode(capture, plot_dir, quiet=SNR_DB is not None)

    # `crc` is the accept gate; the rest are rejected candidates, returned
    # because they are what you need when a capture will not decode.
    decoded = [p for p in packets if p['crc']]

    print()
    for pkt in decoded:
        data = bytes(pkt['bytes'])
        print('decoded %d byte(s) at t=%.3f s: %r'
              % (len(data), pkt['t0']/lrfhss.config.FS,
                 data.decode('utf-8', 'replace')))

    if not decoded:
        print('no packet decoded (%d candidate(s) rejected)' % len(packets))
    else:
        print('spectrogram(s) in %s' % plot_dir)
    return bool(decoded)


if __name__ == '__main__':
    main()
