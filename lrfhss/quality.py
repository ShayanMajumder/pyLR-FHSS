# Part of the lrfhss receiver package.

import numpy as np

from . import config as cfg


def check_energy_length(iq, hwin, hdr, snr_thresh=-20):
    payloadlen = hdr['payloadlen']
    CR = hdr['CR']
    payload_length_bits = 8*(payloadlen+2)+6
    data_in_bitcount = int(np.ceil(payload_length_bits*[6/5, 3/2, 2, 3][CR]))
    num_frags = int(np.ceil(data_in_bitcount/48))
    expected_samples = cfg.STAY_HDR * cfg.HDR_COUNT + cfg.STAY_DATA * num_frags
    sig_start = max(0, hwin - cfg.STAY_HDR)
    sig_end = min(len(iq), hwin + expected_samples)
    noise_start = max(0, sig_start - expected_samples)
    sig_pwr = np.mean(np.abs(iq[sig_start:sig_end])**2) + 1e-12
    if noise_start < sig_start:
        noise_pwr = np.mean(np.abs(iq[noise_start:sig_start])**2) + 1e-12
    else:
        # no room before the packet for a noise window (packet near capture
        # start) -- fall back to whole-capture median power as floor proxy
        noise_pwr = np.median(np.abs(iq)**2) + 1e-12
    snr_db = 10*np.log10(sig_pwr / noise_pwr)
    return snr_db >= snr_thresh, snr_db
