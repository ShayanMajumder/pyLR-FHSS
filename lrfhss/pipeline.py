# Part of the lrfhss receiver package.
"""Running a capture end to end.

    acquire   -> IQ + sync-word candidates
    arbitrate -> which of those candidates are real packets
    report    -> what the run printed

This module owns only the order of those steps, the worker pool and the
low-SNR retry. The decisions themselves live in arbitrate.py.
"""
import os
from dataclasses import dataclass

import numpy as np

from . import acquire
from . import arbitrate
from . import config as cfg
from . import report
from .acquire import acquire_one


@dataclass
class DecodeOptions:
    """Everything that varies between runs.

    There is no worker-count knob: decoding is single-threaded.
    """
    max_packets: int = 20
    plot_spectrograms: bool = False
    plot_dir: str = '.'
    windowed_scan: bool = False
    window_sec: float = 0.1
    hop_sec: float = 0.05
    sensitive_retry: bool = True

    def resolved(self):
        return DecodeOptions(**self.__dict__)


def _decide(iq, acq, results, opts, clusters=None, pruner=None, ci=None):
    return arbitrate.evaluate(iq, acq, results, clusters=clusters, pruner=pruner, ci=ci,
                              plot_spectrograms=opts.plot_spectrograms,
                              plot_dir=opts.plot_dir)


def _decode_clusters(iq, clusters, opts):
    """Acquire every candidate, then decide them strongest evidence first.

    Acquisition is serial; the worker pools that used to spread it over
    cores are gone.

    Ordering still matters enormously, and is the reason this is two
    passes rather than a single acquire-and-decide loop. Candidates must
    be DECIDED in order of the REFINED correlation acquisition produces,
    not the coarse matched-filter cluster score. On case013 the true
    header (fcorr=373) sorted behind a spurious same-time cluster
    (fcorr=168) by coarse score; decided in that order the spurious one
    cleared CRC8 and CRC16 by chance and the real packet was thrown away
    as its duplicate. Acquiring everything up front is what makes the
    refined score available before any decision is taken, and it gives a
    global ordering rather than the per-batch one the pooled version had.
    """
    pruner = arbitrate.ClusterPruner(len(clusters))
    results = []
    order = sorted(range(len(clusters)), key=lambda i: -clusters[i][0])

    acquired = {}
    for ci in order:
        acquired[ci] = acquire_one(iq, clusters[ci][1], clusters[ci][2])
    n_sync = len(acquired)

    for ci in sorted(order, key=lambda c: -acquired[c]['fcorr']):
        if len(results) >= opts.max_packets:
            break
        # Retirement still pays: it skips the payload decode, which is far
        # dearer than the acquisition already spent.
        if pruner.retired[ci]:
            continue
        pruner.decoded[ci] = True
        _decide(iq, acquired[ci], results, opts,
                clusters=clusters, pruner=pruner, ci=ci)
    return results, n_sync


def _retry_sensitive(fn, iq, opts):
    """Rescan once at the low-SNR tier, reusing the front-end output.

    At low SNR the real packets' matched-filter ratios fall BELOW
    MF_THRESH (measured at -10 dB on the real capture: 0.484/0.536/0.463
    against a 0.62 gate), so they never become candidates and the run
    reports nothing for detection reasons alone -- not because the payloads
    are undecodable. CRC8 and CRC16 stay the only accept gates, so this can
    add real packets but never fabricate one; the cost is search time on
    extra noise candidates, which is why it is a fallback.

    PRESCREEN_K widens too: at K=6, measured at -10 dB AWGN, one of three
    true packets is never detected at all (its energy loses the per-frame
    top-K race to noise bins); K=20 finds all three. Recall knob only.

    `iq` is handed straight back in: reloading it is pure waste, since
    load_frontend would re-read the whole capture to rebuild an array we
    already have, bit-identical.
    """
    saved_thresh, saved_k = cfg.MF_THRESH, cfg.PRESCREEN_K
    cfg.MF_THRESH = cfg.MF_THRESH_SENSITIVE
    cfg.PRESCREEN_K = cfg.PRESCREEN_K_SENSITIVE
    try:
        report.sensitive_rescan(saved_thresh)
        retry = DecodeOptions(**{**opts.__dict__, 'sensitive_retry': False})
        return decode(fn, retry, _preloaded_iq=iq)
    finally:
        cfg.MF_THRESH, cfg.PRESCREEN_K = saved_thresh, saved_k


def decode(capture, options=None, _preloaded_iq=None):
    """Decode one capture. Returns a list of candidate records.

    `capture` is a path to a .wav or raw-float32 file, or an IQ array
    already in memory (complex, at config.FS). Passing an array skips the
    front end entirely, which is what you want for a signal you just
    generated with encode() or synthesised in a test.

    Each record carries `crc`, which is the only thing that makes it a
    confirmed packet. Records with crc False are the rejected candidates,
    returned rather than dropped because they are what you need to
    diagnose a capture that failed to decode.
    """
    opts = (options or DecodeOptions()).resolved()
    fn = capture
    if _preloaded_iq is None and not isinstance(capture, (str, bytes, os.PathLike)):
        # An array: already at config.FS, so the channeliser would only
        # decimate it a second time.
        _preloaded_iq = np.ascontiguousarray(capture, dtype=complex)
        fn = None
    if _preloaded_iq is None:
        report.accel_banner()

    if _preloaded_iq is not None:
        iq = _preloaded_iq
        clusters = acquire.detect(iq, opts.windowed_scan, opts.window_sec, opts.hop_sec)
    elif acquire.can_stream(fn, opts.windowed_scan):
        iq, results, n_sync = acquire.stream_interleaved(
            fn, acquire_one, lambda i, a, r: _decide(i, a, r, opts),
            max_packets=opts.max_packets,
            window_sec=opts.window_sec, hop_sec=opts.hop_sec)
        report.summary(results, n_sync, len(results))
        return results
    else:
        iq, clusters = acquire.load_and_detect(
            fn, opts.windowed_scan, opts.window_sec, opts.hop_sec)

    if not clusters:
        report.no_clusters()
        return []
    report.clusters_found(len(clusters))

    results, n_sync = _decode_clusters(iq, clusters, opts)

    if (not any(r['crc'] for r in results)
            and opts.sensitive_retry and not opts.windowed_scan):
        return _retry_sensitive(fn, iq, opts)

    report.summary(results, n_sync, len(clusters))
    return results


def main(capture, max_packets=20,
         plot_spectrograms=False, plot_dir='.', windowed_scan=False,
         window_sec=0.1, hop_sec=0.05, sensitive_retry=True,
         _preloaded_iq=None, **_ignored):
    """Keyword-compatible wrapper around decode(); prefer decode().

    Accepts and ignores the retired n_workers / use_process_pool keywords
    so older call sites keep working.
    """
    return decode(capture, DecodeOptions(
        max_packets=max_packets,
        plot_spectrograms=plot_spectrograms, plot_dir=plot_dir,
        windowed_scan=windowed_scan, window_sec=window_sec,
        hop_sec=hop_sec, sensitive_retry=sensitive_retry),
        _preloaded_iq=_preloaded_iq)
