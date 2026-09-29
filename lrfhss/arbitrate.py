# Part of the lrfhss receiver package.
"""Deciding which sync candidates are real packets.

One physical packet shows up as several candidates -- its header replicas
sit at different frequencies, and each can clear the matched-filter
threshold on its own. Turning that candidate list into a packet list is
where this receiver has historically gone wrong, so the rules live here
rather than inline in the decode loop:

  * Strongest evidence first. Candidates are decoded in order of the
    REFINED correlation from acquisition, not the coarse cluster score.
  * A confirmed packet suppresses its own replicas, by time proximity
    (`find_duplicate`) and by predicted hop schedule (`ClusterPruner`).
  * The payload CRC16 is the accept gate -- but it is only 16 bits, so a
    garbage decode clears it roughly once in 65536 tries. Across a large
    hypothesis search that happens, and because a confirmed packet
    suppresses everything near it, ONE false pass can discard the real
    packet. Both ordering rules above exist to make the real packet win
    that race, not merely to save work.
"""
import os

from . import config as cfg
from . import report
from .payload import _packet_footprint, decode_payload_at
from .plotting import plot_packet_spectrogram
from .quality import check_energy_length


def find_duplicate(results, t0):
    """The already-confirmed packet this candidate is a replica of, if any.

    Header replicas of ONE physical packet are always within
    HDR_COUNT*STAY_HDR samples of each other in TIME -- a known constant,
    no frequency prediction needed. Matching on predicted frequency (the
    older approach, via _packet_footprint's LFSR re-prediction) missed real
    duplicates whose measured frequency didn't match the prediction closely
    enough. Time-only matching is reliable regardless of which replica
    frequency either detection landed on.
    """
    for prev in results:
        if not prev['crc']:
            continue
        if abs(t0 - prev['t0']) < cfg.STAY_HDR*cfg.HDR_COUNT:
            return prev
    return None


class ClusterPruner:
    """Tracks, per cluster, what has been submitted / decoded / retired.

    `submitted` and `retired` are deliberately separate. They used to be
    one flag, which meant the predicted-hop retirement could never prune a
    cluster sitting in the CURRENT batch -- and a batch is n_workers wide,
    so on a machine with enough cores every cluster landed in one batch and
    retirement pruned nothing. The duplicate replicas it was meant to drop
    got decoded anyway, one cleared CRC16 by chance, and the real packet
    was then discarded as its duplicate. That made the decode depend on
    core count (case044: passed at 1/2/4 workers, failed at 8/14).
    """

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
        """Retire clusters the confirmed packet's hop schedule accounts for.

        Skips the candidate itself and anything already decoded or retired
        -- but NOT merely-submitted clusters, which is the distinction that
        makes this work on same-batch siblings.
        """
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
        better aligned in time -- still deserve their turn."""
        for cj in clusters_idx:
            self.retired[cj] = False
            self.submitted[cj] = False


def _header_matches_known_config(hdr):
    return not (hdr['CR'] != cfg.KNOWN_CR or hdr['grid'] != cfg.KNOWN_GRID
                or hdr['hop'] != cfg.KNOWN_HOP or hdr['BW'] != cfg.KNOWN_BW
                or int(''.join(map(str, hdr['hopseq'])), 2) != cfg.KNOWN_HOPSEQ)


def evaluate(iq, acq, results, clusters=None, pruner=None, ci=None,
             plot_spectrograms=False, plot_dir='.'):
    """Decide one acquired candidate, decoding its payload if it survives.

    Appends to `results` and returns the new entry, or returns None if the
    candidate was rejected. `clusters`/`pruner`/`ci` enable hop-schedule
    retirement; without them the candidate is still fully evaluated, just
    with nothing to prune (the streaming path has no global cluster list).

    NOTE: a correlation-score reject gate used to sit before the energy
    check (fcorr < 60.0). It was removed: it ran AFTER header CRC8 already
    passed, and CRC8 is far stronger evidence than a raw correlation
    heuristic -- confirmed on a real recording where two CRC-valid,
    payload-CRC16-passing packets both scored fcorr in the 27-38 range and
    were being silently discarded before ever reaching payload decode.
    """
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
        # Retirement was provisional on this packet decoding. It didn't --
        # typically a lock placed badly in time -- so give the other locks
        # on the same packet their turn; which one is decided first then
        # stops deciding whether the packet is found at all.
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

    entry = dict(idx=len(results), t0=acq['fstart'], corr=acq['fcorr'],
                 header=hdr, bytes=payload_bytes, crc=crc_ok, footprint=footprint)
    results.append(entry)
    return entry
