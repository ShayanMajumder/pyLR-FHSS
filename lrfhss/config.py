"""Every retunable receiver parameter, in one place.

These are module-level so callers can retune
the receiver per capture by assigning to them (decode_sweep.py does this
for each bandwidth). That still works, but it MUST go through this module:
`from .config import FS` would bind a local name that a later assignment
here could never reach, so every other module imports this as `cfg` and
reads `cfg.FS` at call time.
"""
import os
import numpy as np

# scipy.fft's pocketfft beats numpy.fft even single-threaded (measured on
# this receiver's largest transform, NB=2211840: 0.0814 s/call numpy vs
# Decoding is single-threaded throughout: no worker counts, no thread or
# process pools, no GPU. Both were tried and removed -- see fftw.py.

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
# The receiver's working symbol rate. The SPEC rate is 500000/1024 =
# 488.28125 (see SYMBOL_RATE_HZ below), and at FS=166666.67 that is 341.33
# samples per symbol -- so neither 488 nor 488.28125 is exact. 488 is kept
# because it rounds SMBL to 342, which is what every other constant here
# was calibrated against: LOOKDIST, the matched-filter thresholds and the
# gsto search grid. Switching to the spec value moves SMBL to 341 and
# measurably LOSES packets on real captures -- 178 decoded against 188
# over the 198-file pkttrace set, with fewer header locks too. Correctness
# of one constant is not worth a 5% recall loss in the thing it feeds.
BW           = 488
SMBL         = round(FS/BW)
LOOKDIST     = round(SMBL/4)
PHASESLOPE   = (np.pi/2)/SMBL
STAY_HDR     = round(0.233472*FS)
STAY_DATA    = round(0.1024*FS)
HDR_COUNT    = 3
ALLBW        = 137000
# Wall-clock ceiling on the per-header dcfo/gsto payload search (see
# decode_payload_at). Module-level so a batch driver can raise it: the
# 10 s figure was measured on an UNLOADED box, and because it is wall
# clock rather than CPU time, running several decoders concurrently on
# oversubscribed cores makes real packets blow the budget and report a
# spurious CRC fail.
PAYLOAD_TIME_BUDGET_S = 10.0
SYNC_START_BIT = 41
HDR_BIT_NUM  = 113
SPUR_FREQS = []

KNOWN_CR      = 3
KNOWN_GRID    = 1
KNOWN_HOP     = 1
KNOWN_BW      = [0, 0, 1, 0]
KNOWN_HOPSEQ  = 77

# Precomputed window for the common STAY_HDR-length case. np.hanning(38912)
# costs ~1ms/call and this array is regenerated identically on every header/
# payload search iteration otherwise -- cumulative cost across a full decode
# (thousands of df/cfo/dcfo/gsto combos) is real. Fall back to a fresh
# np.hanning(len(seg)) for any segment that isn't exactly STAY_HDR long.
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

_NB = 30*DECIM*4096
_OV = 90*64


MF_NSYNC = len(SYNC_WORD)
MF_NCFO = 64
MF_THRESH = 0.62   # was 0.72. Noise-only max ~0.62-0.63 at full-capture scale
# Top-K STFT bins kept per frame when choosing where to run the matched
# filter. This is the real gate on detection cost: the -30 dB energy
# threshold below it turns out to select nothing (436 regions at every
# setting from -30 to -15 dB on a real capture), whereas K sets the region
# count almost linearly -- K=6 -> 436 regions, K=3 -> 218, K=2 -> 151.
# Override with LRFHSS_PRESCREEN_K to A/B it against recall.
PRESCREEN_K = int(os.environ.get('LRFHSS_PRESCREEN_K', '6'))
    # top-K STFT bins kept per frame in the localization prescreen. This is
    # a RECALL knob: a real burst whose energy doesn't make the top-K in
    # its frame never reaches the matched filter at all, so at low SNR
    # (where noise bins compete) a too-small K silently loses packets
    # before any decode gate sees them. Cost of raising it is a slightly
    # larger argpartition + more MF regions, not correctness -- MF score
    # and CRC8/CRC16 remain the only accept gates.
GSTO_STEP_COARSE = 12
GSTO_STEP_FINE = 3
GSTO_TIERS = (GSTO_STEP_COARSE,)
    # DEFAULT IS SPEED-FIRST: coarse gsto grid only. Measured on the real
    # capture: clean 4.12s -> 3.69s and -10dB 24.3s -> 16.3s (33% faster),
    # both still decoding the SAME packets (3/3 clean, 1/3 at -10dB).
    # Justification for coarse being safe: with early-exit disabled at clean
    # SNR, 1818/35351 grid points pass CRC16 (5.14%, ~1 in 19) -- passing
    # points are dense, not a needle, so a 4x-coarser grid still lands on
    # one. At -10dB density collapses to 1/21522, and the coarse grid still
    # happened to catch it here, but that is not guaranteed in general.
    # Set GSTO_TIERS = (GSTO_STEP_COARSE, GSTO_STEP_FINE) to restore the
    # full fine-grid fallback for maximum low-SNR recall at ~1.5x the
    # payload-search cost.
    # Payload symbol-timing search granularity. Coarse runs first; fine is
    # the fallback when coarse finds no CRC16 pass. Justified by measurement,
    # not guesswork: with early-exit disabled at clean SNR, 1818/35351 grid
    # points pass CRC16 (5.14%), so a 4x-coarser gsto grid still contains
    # passing points with overwhelming probability while doing 1/4 the
    # Viterbi work. At -10dB the pass density collapses to 1/21522, so the
    # fine tier is what actually finds those -- hence two tiers, not a
    # replacement.
PRESCREEN_K_SENSITIVE = 20
MF_THRESH_SENSITIVE = 0.55
    # Fallback floor for _cfar_detect (see below) and the sensitive-tier
    # rescan trigger in main(). Was the sole detection gate before CFAR;
    # re-measured against actual recall instead of assumed -- on the real
    # capture at -10dB AWGN, 0.45 and 0.55 detect the exact same 2/3 true
    # packets. 0.58 drops recall to 1/3, so 0.55 is the safe floor.
CFAR_GUARD = 3      # cells excluded around the cell under test (its own
                     # sync-word correlation sidelobes shouldn't count as
                     # "background" -- a real peak's neighbors are still
                     # somewhat correlated, not independent noise)
CFAR_FACTOR = 1.6    # threshold = local_ref_mean * CFAR_FACTOR. Chosen so
                     # a real header's peak score (~0.85-0.95, see the
                     # measured 0.78-0.93 range in find_packets' own
                     # docstring history) clears a noise-floor reference
                     # mean of ~0.5-0.6 (this statistic is a normalized
                     # ratio bounded in [0,1], not raw energy, so its
                     # noise-floor level is itself informative, not
                     # arbitrary).


HOP_ENERGY_MIN = 0   # non-coherent per-hop energy floor (Ek/noise ratio) for
                       # accepting a header's LFSR-predicted payload hops as
                       # real. Measured: true packets ~500-955, CRC8 false
                       # positives ~0-11.5 (16/16 tested). Wide margin both
                       # sides. See decode_payload_at.


# ---------------------------------------------------------------------
# Retuning
#
# The values above are defaults for ONE configuration. A capture at any
# other bandwidth needs a different front-end rate, and everything derived
# from that rate has to move with it -- symbol length, hop dwell times, the
# sync matched filter, the CFO ramp, the hop-search window. Getting one of
# them wrong does not raise; it decodes nothing, or worse, decodes garbage
# that clears the 16-bit payload CRC by chance. `retune()` derives the
# whole set from the bandwidth so callers cannot set half of it.
# ---------------------------------------------------------------------

# Decimations of a 3 MSPS capture that divide load_frontend's block and
# overlap sizes exactly.
DECIMS = (18, 9, 6, 4, 3, 2, 1)

# Hop-window width as a multiple of the occupied bandwidth. The window must
# hold the hop grid AND the fragment's own width AND residual CFO: a snug
# 1.01x leaves ~2 kHz at the edge and the outermost hops fall off it
# (bw=39.06 kHz failed at 1.01x, decodes at every factor from 1.05x up).
ALLBW_MARGIN = 1.25


def pick_decim(bw_hz, fs_capture=None, margin=1.05):
    """Largest decimation whose output rate still passes the whole signal."""
    fs_capture = FS_CAPTURE if fs_capture is None else fs_capture
    for d in DECIMS:
        if fs_capture/d >= bw_hz*margin:
            return d
    return 1


def set_sync_word(hex_or_bits):
    """Rebuild the sync matched filter for a transmitter's sync word.

    Accepts hex ('12AD101B') or a bit string. The default suits one
    transmitter; RadioLib's SX126x LR-FHSS uses 0x12AD101B, and a mismatch
    reports no sync-word candidates at all rather than a bad decode.
    """
    global SYNC_WORD, SYNC_VEC, MF_NSYNC, SYNC_OFF, _SYNC_K, _SGN
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
    """Point the receiver at one capture's configuration.

    bw_hz       occupied bandwidth, e.g. 722_660 for the 722.66 kHz config
    sync_word   transmitter's sync word, hex or bits (optional)
    hdr_count   header replicas actually transmitted, 1..4. NOT recoverable
                from the header, and a wrong value puts the payload window
                a whole dwell out.

    A retired `cores` keyword is accepted and ignored: decoding is
    single-threaded.
    """
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
    # Clamp inside Nyquist so the gate stays meaningful instead of
    # degenerating to "the whole band" on the widest captures.
    ALLBW = min(round(bw_hz*margin), round(FS*0.98))

    if hdr_count:
        HDR_COUNT = int(hdr_count)

    # The 10 s default was measured at DECIM=18 (FS=167 kHz). Every
    # hypothesis in the payload search costs ~FS, so at DECIM=1 the same
    # search is ~18x dearer and a real packet blows 10 s mid-search,
    # leaving a half-finished guess that can clear CRC16 by chance.
    # Scaled generously now that the host side is serial.
    PAYLOAD_TIME_BUDGET_S = 10.0*(18.0/DECIM)*4.0


# ---------------------------------------------------------------------
# LoRaWAN data rates
#
# A DR index names a whole LR-FHSS configuration: occupied bandwidth,
# coding rate AND header-replica count. That last one is the reason this
# table earns its place -- the replica count is not carried in the header,
# so a receiver has to be told it, and for standards-compliant traffic the
# DR is how you know.
#
# Bandwidths are the nominal names from the regional parameters ("137 kHz",
# "1.523 MHz"); the actual occupied widths are the channel-count multiples
# the hop tables use (136.72, 1523.4 kHz), which is what goes to retune().
#
# Sources: LoRa Alliance RP2 regional parameters, via
# thethingsnetwork.org/docs/lorawan/regional-parameters/{eu868,us915}.
# Replica counts: DR8/DR10 use CR 1/3 with 3 header repetitions, DR9/DR11
# use CR 2/3 with 2 (LR-FHSS: Overview and Performance Analysis,
# arxiv.org/pdf/2010.00491).
#
# Only the two lowest rates of each width are defined; the rest of the
# LR-FHSS parameter space is legal on the air but has no DR name, which is
# why bandwidth / CR / hdr_count can still be given directly.
# ---------------------------------------------------------------------

# CR index as the header carries it: 0 -> 5/6, 1 -> 2/3, 2 -> 1/2, 3 -> 1/3
CR_RATES = {0: '5/6', 1: '2/3', 2: '1/2', 3: '1/3'}

# The true LR-FHSS symbol rate, 500000/1024. Used where the exact figure
# matters rather than the receiver's calibrated approximation: the encoder
# sizes its dwells from this, because a 50-bit payload fragment has to
# land in exactly 102.4 ms. At BW=488 it came to 102.459 ms and the
# encoder truncated the tail of every fragment.
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
    """What DR `dr` means: bandwidth, coding rate and header replicas.

        >>> datarate(8)['header_replicas']
        3
    """
    key = (region.upper(), int(dr))
    if key not in DATA_RATES:
        known = sorted(d for r, d in DATA_RATES if r == region.upper())
        raise ValueError(
            'no LR-FHSS DR%s in %s; defined there: %s'
            % (dr, region.upper(), known or 'none'))
    return dict(DATA_RATES[key])


def retune_dr(dr, region='EU868', sync_word=None, fs_capture=None):
    """Point the receiver at a LoRaWAN data rate.

        lrfhss.config.retune_dr(8)                  # EU868 DR8
        lrfhss.config.retune_dr(5, region='US915')

    Shorthand for retune() with the bandwidth and header-replica count the
    DR implies. Returns the resolved parameters, so the coding rate is
    available for encoding (the decoder reads it from the header).
    """
    spec = datarate(dr, region)
    retune(spec['bw_khz']*1e3, sync_word=sync_word,
           hdr_count=spec['header_replicas'], fs_capture=fs_capture)
    return spec
