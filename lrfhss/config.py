# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Every retunable receiver parameter, in one place."""
import os
import numpy as np


try:
    from . import _viterbi_ext as _vext_local
    _HAVE_VEXT_LOCAL = True
except ImportError:
    _vext_local = None
    _HAVE_VEXT_LOCAL = False

FS_CAPTURE   = 3e6
DSF          = 3
FS           = (500000)/DSF
DECIM        = round(FS_CAPTURE/(500e3/DSF))
CARRIER_OFF  = 97e3
DC_NOTCH_HZ  = 15e3
BW           = 488
SMBL         = round(FS/BW)
LOOKDIST     = round(SMBL/4)
PHASESLOPE   = (np.pi/2)/SMBL
STAY_HDR     = round(0.233472*FS)
STAY_DATA    = round(0.1024*FS)
HDR_COUNT    = 3
ALLBW        = 137000
PAYLOAD_TIME_BUDGET_S = 10.0
SYNC_START_BIT = 41
HDR_BIT_NUM  = 113
SPUR_FREQS = []
EDGE_GUARD_S = 0.010
# Rate the detector's region scan filters at (detect.region_decim).
REGION_RATE_HZ = 16000.0
FINE_SYNC_DECIM = False

KNOWN_CR      = 3
KNOWN_GRID    = 1
KNOWN_HOP     = 1
KNOWN_BW      = [0, 0, 1, 0]
KNOWN_HOPSEQ  = 77

_WIN_HDR = np.hanning(STAY_HDR)


SYNC_WORD = np.array([int(c) for c in '00101100000011110111100110010101'])
_ph = [0.0]
for _h in range(1, len(SYNC_WORD)):
    _ph.append(_ph[-1] if SYNC_WORD[_h] != SYNC_WORD[_h-1]
               else _ph[-1] + (np.pi/2 if SYNC_WORD[_h] == 1 else -np.pi/2))
SYNC_VEC = np.exp(1j*np.array(_ph))
SYNC_OFF = np.round(np.arange(len(SYNC_WORD))/BW*FS).astype(int)

_SYNC_K = np.arange(len(SYNC_WORD))
_NC = 120
_CFO_GRID = -np.pi + 2*np.pi*np.arange(_NC)/_NC
_SGN = ((-1.0)**_SYNC_K)[:, None]
_COARSE_HZ = np.arange(-1500, 1501, BW)
_COARSE_RAMP = np.exp(-1j*2*np.pi*np.outer(_COARSE_HZ, SYNC_OFF)/FS)
_COARSE_ONE = int(np.argmin(np.abs(_COARSE_HZ)))

_NB = 30*DECIM*4096
_OV = 90*64


MF_NSYNC = len(SYNC_WORD)
MF_NCFO = 64
MF_THRESH = 0.62   # was 0.72.
PRESCREEN_K = int(os.environ.get('LRFHSS_PRESCREEN_K', '6'))
GSTO_STEP_COARSE = 12
GSTO_STEP_FINE = 3
GSTO_TIERS = (GSTO_STEP_COARSE,)
CFAR_FLOOR = 0.55
MF_LPF_HZ = 400
FINE_SYNC_LPF_HZ = 3000
DC_GAP_REACH_HZ = 3000
PRESCREEN_F_ROUND_HZ = 50

DEMOD_AVG_PHASE = False

PAYLOAD_TRELLIS = True
HEADER_TRELLIS = True
HEADER_OLD_FALLBACK = False
PAYLOAD_OLD_FALLBACK = False
TRELLIS_DECIM = 1
FILTER_ONCE = True
SYNC_ZOOM = True
SYNC_DCFO_HZ = (-4, 0, 4)
SYNC_MIN_Q = 20.0
SYNC_COMBINE_Q = 15.0
SYNC_SKIP_Q = 10.0
TRELLIS_DCFO_HZ = tuple(range(-24, 25, 6))
CFAR_GUARD = 3
CFAR_FACTOR = 1.6    # threshold = local_ref_mean * CFAR_FACTOR.


HOP_ENERGY_MIN = 0


DECIMS = (18, 9, 6, 4, 3, 2, 1)

ALLBW_MARGIN = 1.25
HOP_EDGE_TOL_HZ = 1000.0


def pick_decim(bw_hz, fs_capture=None, margin=1.05):
    """Largest decimation whose output rate still passes the whole signal."""
    fs_capture = FS_CAPTURE if fs_capture is None else fs_capture
    for d in DECIMS:
        if fs_capture/d >= bw_hz*margin:
            return d
    return 1


def set_sync_word(hex_or_bits):
    """Rebuild the sync matched filter for a transmitter's sync word."""
    global SYNC_WORD, SYNC_VEC, MF_NSYNC, _SYNC_K, _SGN
    text = str(hex_or_bits).strip()
    if set(text) <= set('01') and len(text) >= 16:
        bits = text
    else:
        bits = ''.join(f'{b:08b}' for b in bytes.fromhex(text))
    w = np.array([int(c) for c in bits])
    phase = [0.0]
    for i in range(1, len(w)):
        phase.append(phase[-1] if w[i] != w[i-1]
                     else phase[-1] + (np.pi/2 if w[i] == 1 else -np.pi/2))
    SYNC_WORD = w
    SYNC_VEC = np.exp(1j*np.array(phase))
    MF_NSYNC = len(w)
    _SYNC_K = np.arange(len(w))
    _SGN = ((-1.0)**_SYNC_K)[:, None]


def retune(bw_hz, sync_word=None, hdr_count=None,
           fs_capture=None, allbw_margin=None, **_ignored):
    """Point the receiver at one capture's configuration."""
    global DECIM, FS, SMBL, LOOKDIST, PHASESLOPE, STAY_HDR, STAY_DATA
    global _WIN_HDR, SYNC_OFF, _COARSE_RAMP, ALLBW, HDR_COUNT
    global PAYLOAD_TIME_BUDGET_S, FS_CAPTURE

    if fs_capture is not None:
        FS_CAPTURE = float(fs_capture)
    if sync_word is not None:
        set_sync_word(sync_word)

    DECIM = pick_decim(bw_hz, FS_CAPTURE)
    FS = FS_CAPTURE/DECIM
    SMBL = round(FS/BW)
    LOOKDIST = round(SMBL/4)
    PHASESLOPE = (np.pi/2)/SMBL
    STAY_HDR = round(0.233472*FS)
    STAY_DATA = round(0.1024*FS)
    _WIN_HDR = np.hanning(STAY_HDR)
    SYNC_OFF = np.round(np.arange(MF_NSYNC)/BW*FS).astype(int)
    _COARSE_RAMP = np.exp(-1j*2*np.pi*np.outer(_COARSE_HZ, SYNC_OFF)/FS)

    margin = ALLBW_MARGIN if allbw_margin is None else allbw_margin
    ALLBW = min(round(bw_hz*margin), round(FS*0.98))

    if hdr_count:
        HDR_COUNT = int(hdr_count)

    PAYLOAD_TIME_BUDGET_S = 10.0*(18.0/DECIM)*4.0


# CR index as the header carries it: 0 -> 5/6, 1 -> 2/3, 2 -> 1/2, 3 -> 1/3
CR_RATES = {0: '5/6', 1: '2/3', 2: '1/2', 3: '1/3'}

SYMBOL_RATE_HZ = 500000/1024          # 488.28125

DATA_RATES = {
    ('EU868', 8):  dict(bw_khz=136.72, cr=3, header_replicas=3),
    ('EU868', 9):  dict(bw_khz=136.72, cr=1, header_replicas=2),
    ('EU868', 10): dict(bw_khz=335.94, cr=3, header_replicas=3),
    ('EU868', 11): dict(bw_khz=335.94, cr=1, header_replicas=2),
    ('US915', 5):  dict(bw_khz=1523.4, cr=3, header_replicas=3),
    ('US915', 6):  dict(bw_khz=1523.4, cr=1, header_replicas=2),
    ('AU915', 5):  dict(bw_khz=1523.4, cr=3, header_replicas=3),
    ('AU915', 6):  dict(bw_khz=1523.4, cr=1, header_replicas=2),
}


def datarate(dr, region='EU868'):
    """What DR `dr` means: bandwidth, coding rate and header replicas."""
    key = (region.upper(), int(dr))
    if key not in DATA_RATES:
        known = sorted(d for r, d in DATA_RATES if r == region.upper())
        raise ValueError(
            'no LR-FHSS DR%s in %s; defined there: %s'
            % (dr, region.upper(), known or 'none'))
    return dict(DATA_RATES[key])


def retune_dr(dr, region='EU868', sync_word=None, fs_capture=None):
    """Point the receiver at a LoRaWAN data rate."""
    spec = datarate(dr, region)
    retune(spec['bw_khz']*1e3, sync_word=sync_word,
           hdr_count=spec['header_replicas'], fs_capture=fs_capture)
    return spec
