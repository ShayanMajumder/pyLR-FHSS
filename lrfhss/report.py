# Part of the lrfhss receiver package.
"""Console output for a decode run.

Every print the pipeline emits lives here, so the decode logic reads as
decisions rather than as formatting, and so the output can be silenced or
redirected in one place instead of thirty.
"""
import numpy as np

from . import config as cfg


def accel_banner():
    """State plainly whether the C++ extension is in play.

    Every C++ path in this receiver falls back to numpy SILENTLY when the
    extension isn't built, so a run with no .so present looks identical --
    just several times slower. Make that visible.
    """
    if cfg._HAVE_VEXT_LOCAL:
        print('  [accel] C++ extension active (lrfhss._viterbi_ext)')
    else:
        print('  [accel] C++ extension NOT FOUND -- running pure-numpy fallback '
              '(several times slower). Build it with: pip install -e .')


def loading():
    print('Loading + front-end...')


def loaded(n_samples):
    print('  native samples: %d (%.2f s @ %.0f Hz)' % (n_samples, n_samples/cfg.FS, cfg.FS))


def scanning(windowed=False, window_sec=None, hop_sec=None):
    print('Matched-filter scan for sync-word candidates...')
    if windowed:
        print('  (windowed: %.3fs window, %.3fs hop)' % (window_sec, hop_sec))


def streaming(window_sec, hop_sec):
    print('Streaming detection + decode (%.1fs window / %.1fs hop, interleaved)...'
          % (window_sec, hop_sec))


def clusters_found(n):
    print('Found %d candidate cluster(s).' % n)


def no_clusters():
    print('No sync-word candidates found at all.')


def duplicate(t0, hf, prev_t0):
    print('  [dedup] t=%.3fs hf=%.0fHz matches already-confirmed packet '
          'at t=%.3fs (same physical packet, different header replica) '
          '-- skipping' % (t0/cfg.FS, hf, prev_t0/cfg.FS))


def ghost(pwr_snr):
    print('  [reject] Ghost packet. Energy footprint failed (SNR %.1f)' % pwr_snr)


def snr(pwr_snr):
    # check_energy_length can hand back 0 or a negative ratio when the
    # window it measured holds no packet -- which happens whenever the
    # caller's parameters are wrong, exactly when you are reading this
    # output. log10 of that is a RuntimeWarning and a nan in the trace.
    if not pwr_snr > 0:
        print('  Estimated packet SNR: n/a (no energy in the measured window)')
    else:
        print('  Estimated packet SNR: %.1f dB' % (10*np.log10(pwr_snr)))


def candidate(t0, hf, corr, hdr):
    print('\n--- packet at t=%.3f s, hdr_f=%.0f Hz, sync corr=%.3g ---'
          % (t0/cfg.FS, hf, corr))
    print('  header CRC8 pass (UNCONFIRMED -- 8-bit CRC, real gate is '
          'payload CRC16 below): payloadlen=%d CR=%d grid=%d hop=%d '
          'syncindex=%d hopseq=%s'
          % (hdr['payloadlen'], hdr['CR'], hdr['grid'], hdr['hop'],
             hdr.get('syncindex', 0), hdr['hopseq']))


def retired(n):
    print('  WAITING: retired %d cluster(s) via predicted hop sequence' % n)


def payload(crc_ok, payload_bytes):
    print('  PAYLOAD CRC=%s bytes=%s' % ('PASS' if crc_ok else 'FAIL', payload_bytes))


def config_mismatch():
    print('  [note] payload CRC16 passed (real gate) but header '
          'fields differ from known config -- likely a second '
          'sender/hop_seq_id, not noise.')


def plot_written(path):
    print('  [plot] wrote %s (payload-confirmed real packet)' % path)


def sensitive_rescan(old_thresh):
    print('\n==== no packets at MF_THRESH=%.2f -- rescanning at sensitive '
          'threshold %.2f (low-SNR tier) ====' % (old_thresh, cfg.MF_THRESH_SENSITIVE))


def summary(results, n_sync, n_clusters):
    ok = [r for r in results if r['crc']]
    print('\n==== SUMMARY: %d packet(s) fully decoded (payload CRC pass) ====' % len(ok))
    print('     %d sync search(es) for %d cluster(s)' % (n_sync, n_clusters))
    for r in ok:
        print('  t=%.3f s: %s' % (r['t0']/cfg.FS, r['bytes']))
    if not ok:
        print('  None. If sync corr is ~noise floor (<10), the captured signal is')
        print('  too weak or not properly GMSK-modulated for header lock.')
        print('  Verify the transmitter and improve link SNR.')
