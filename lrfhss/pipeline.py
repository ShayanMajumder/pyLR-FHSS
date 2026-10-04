# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Running a capture end to end.

    acquire   -> IQ + sync-word candidates
    arbitrate -> which of those candidates are real packets
    report    -> what the run printed
"""
import os
from dataclasses import dataclass

import numpy as np

from . import acquire
from . import arbitrate
from . import report
from .acquire import acquire_one


@dataclass
class DecodeOptions:
    """Everything that varies between runs."""
    max_packets: int = 20
    plot_spectrograms: bool = False
    plot_dir: str = '.'
    windowed_scan: bool = False
    window_sec: float = 0.1
    hop_sec: float = 0.05

    def resolved(self):
        return DecodeOptions(**self.__dict__)


def _decide(iq, acq, results, opts, clusters=None, pruner=None, ci=None):
    return arbitrate.evaluate(iq, acq, results, clusters=clusters, pruner=pruner, ci=ci,
                              plot_spectrograms=opts.plot_spectrograms,
                              plot_dir=opts.plot_dir)


def _decode_clusters(iq, clusters, opts):
    """Acquire every candidate, then decide them strongest evidence first."""
    pruner = arbitrate.ClusterPruner(len(clusters))
    results = []
    order = sorted(range(len(clusters)), key=lambda i: -clusters[i][0])

    acquired = {}
    for ci in order:
        acquired[ci] = acquire_one(iq, clusters[ci][1], clusters[ci][2])
    n_sync = len(acquired)

    for ci in sorted(order, key=lambda c: (-acquired[c].get('sync_q', 0.0),
                                           -acquired[c]['fcorr'])):
        if len(results) >= opts.max_packets:
            break
        if pruner.retired[ci]:
            continue
        pruner.decoded[ci] = True
        _decide(iq, acquired[ci], results, opts,
                clusters=clusters, pruner=pruner, ci=ci)
    return results, n_sync


def decode(capture, options=None, _preloaded_iq=None):
    """Decode one capture. Returns a list of candidate records."""
    opts = (options or DecodeOptions()).resolved()
    fn = capture
    if _preloaded_iq is None and not isinstance(capture, (str, bytes, os.PathLike)):
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
    report.summary(results, n_sync, len(clusters))
    return results


def main(capture, max_packets=20,
         plot_spectrograms=False, plot_dir='.', windowed_scan=False,
         window_sec=0.1, hop_sec=0.05,
         _preloaded_iq=None, **_ignored):
    """Keyword-compatible wrapper around decode(); prefer decode()."""
    return decode(capture, DecodeOptions(
        max_packets=max_packets,
        plot_spectrograms=plot_spectrograms, plot_dir=plot_dir,
        windowed_scan=windowed_scan, window_sec=window_sec,
        hop_sec=hop_sec),
        _preloaded_iq=_preloaded_iq)
