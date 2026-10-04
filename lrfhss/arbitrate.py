# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Deciding which sync candidates are real packets."""
import os

from . import config as cfg
from . import report
from .payload import _packet_footprint, _packet_slots, decode_payload_at
from .plotting import plot_packet_spectrogram
from .quality import check_energy_length, packet_snr_db


def find_duplicate(results, t0):
    """The already-confirmed packet this candidate is a replica of, if any."""
    for prev in results:
        if not prev['crc']:
            continue
        if abs(t0 - prev['t0']) < cfg.STAY_HDR*cfg.HDR_COUNT:
            return prev
    return None


class ClusterPruner:
    """Tracks, per cluster, what has been submitted / decoded / retired."""

    def __init__(self, n):
        self.submitted = [False]*n
        self.decoded = [False]*n
        self.retired = [False]*n

    def next_batch(self, order, start, size):
        """The next `size` not-yet-submitted clusters, in `order`."""
        batch, j = [], start
        while j < len(order) and len(batch) < size:
            ci = order[j]
            if not self.submitted[ci]:
                batch.append(ci)
            j += 1
        return batch, j

    def mark_submitted(self, batch):
        for ci in batch:
            self.submitted[ci] = True

    def retire_predicted(self, clusters, footprint, keep):
        """Retire clusters the confirmed packet's hop schedule accounts for."""
        t_lo, t_hi, freqs = footprint
        gone = []
        for cj in range(len(clusters)):
            if cj == keep or self.decoded[cj] or self.retired[cj]:
                continue
            _, tj, fj = clusters[cj]
            if t_lo <= tj <= t_hi and any(abs(fj-p) < 1500 for p in freqs):
                self.submitted[cj] = True
                self.retired[cj] = True
                gone.append(cj)
        return gone

    def restore(self, clusters_idx):
        """Undo a retirement: the candidate that caused it failed its
        payload, so its siblings -- other locks on the same packet, often
        better aligned in time -- still deserve their turn.
        """
        for cj in clusters_idx:
            self.retired[cj] = False
            self.submitted[cj] = False


def _header_matches_known_config(hdr):
    return not (hdr['CR'] != cfg.KNOWN_CR or hdr['grid'] != cfg.KNOWN_GRID
                or hdr['hop'] != cfg.KNOWN_HOP or hdr['BW'] != cfg.KNOWN_BW
                or int(''.join(map(str, hdr['hopseq'])), 2) != cfg.KNOWN_HOPSEQ)


def evaluate(iq, acq, results, clusters=None, pruner=None, ci=None,
             plot_spectrograms=False, plot_dir='.'):
    """Decide one acquired candidate, decoding its payload if it survives."""
    hdr = acq['hdr']
    if hdr is None:
        return None

    dup_of = find_duplicate(results, acq['fstart'])
    if dup_of is not None:
        report.duplicate(acq['fstart'], acq['fhf'], dup_of['t0'])
        return None

    length_ok, pwr_snr = check_energy_length(iq, acq['hwin'], hdr)
    if not length_ok:
        report.ghost(pwr_snr)
        return None
    report.snr(pwr_snr)
    report.candidate(acq['fstart'], acq['fhf'], acq['fcorr'], hdr)

    footprint = _packet_footprint(hdr, acq['hwin'], acq['hf_precise'])
    retired_now = []
    if footprint is not None and pruner is not None and clusters is not None:
        retired_now = pruner.retire_predicted(clusters, footprint, ci)
        report.retired(len(retired_now))

    payload_bytes, crc_ok = decode_payload_at(iq, hdr, acq['hwin'], acq['hf_precise'])
    report.payload(crc_ok, payload_bytes)
    if not crc_ok and retired_now:
        pruner.restore(retired_now)

    if crc_ok:
        if not _header_matches_known_config(hdr):
            report.config_mismatch()
        if plot_spectrograms:
            os.makedirs(plot_dir, exist_ok=True)
            png = os.path.join(plot_dir, 'packet_t%.3fs.png' % (acq['fstart']/cfg.FS))
            saved = plot_packet_spectrogram(iq, hdr, acq['hwin'], acq['hf_precise'], png)
            if saved:
                report.plot_written(saved)

    snr_db = slots = None
    if crc_ok:
        slots = _packet_slots(iq, hdr, acq['hwin'], acq['hf_precise'])
        snr_db = packet_snr_db(iq, slots) if slots else None
    entry = dict(idx=len(results), t0=acq['fstart'], corr=acq['fcorr'],
                 header=hdr, bytes=payload_bytes, crc=crc_ok, footprint=footprint,
                 snr_db=snr_db, slots=slots)
    results.append(entry)
    return entry
