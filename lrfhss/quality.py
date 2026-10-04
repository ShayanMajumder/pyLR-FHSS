# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT

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
        noise_pwr = np.median(np.abs(iq)**2) + 1e-12
    snr_db = 10*np.log10(sig_pwr / noise_pwr)
    return snr_db >= snr_thresh, snr_db


def packet_snr_db(iq, slots, half_bw_hz=400.0, ref_bw_hz=125e3):
    """SNR of a decoded packet in a 125 kHz reference bandwidth (the axis the
    sensitivity curves and LoRa's packet SNR use): the power in each hop's
    channel over the noise density of the same slots, all slots pooled."""
    sig = noise = 0.0
    for s in slots:
        seg = iq[max(0, int(s['t_start'])):min(len(iq), int(s['t_end']))]
        if len(seg) < 1024:
            continue
        n = len(seg)
        P = np.abs(np.fft.fft(seg))**2/n**2               # power per bin
        f = np.fft.fftfreq(n, 1/cfg.FS)
        hop = np.abs(f - s['freq']) < half_bw_hz
        rest = ~hop & (np.abs(f) < cfg.ALLBW/2)
        per_bin = np.median(P[rest])/np.log(2)            # exponential bins
        sig += max(P[hop].sum() - hop.sum()*per_bin, 0.0)
        noise += per_bin/(cfg.FS/n)
    if sig <= 0 or noise <= 0:
        return None
    return float(10*np.log10(sig/(noise*ref_bw_hz)))
