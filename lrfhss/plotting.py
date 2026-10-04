# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT

from . import config as cfg
from .payload import _packet_slots


def plot_packet_spectrogram(iq, hdr, hwin, hfp, out_path, margin_sec=0.15):
    """Save a spectrogram of one packet with header/payload slots boxed and
    labeled.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches

    slots = _packet_slots(iq, hdr, hwin, hfp)
    if slots is None:
        return None
    t_lo = min(s['t_start'] for s in slots) - int(margin_sec*cfg.FS)
    t_hi = max(s['t_end'] for s in slots) + int(margin_sec*cfg.FS)
    t_lo = max(0, t_lo); t_hi = min(len(iq), t_hi)
    seg = iq[t_lo:t_hi]
    if len(seg) < 256:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    nfft = 512
    ax.specgram(seg, NFFT=nfft, Fs=cfg.FS, noverlap=nfft//2,
               scale='dB', cmap='viridis',
               xextent=(0, len(seg)/cfg.FS))
    ax.set_xlabel('Time (s, relative to packet window start)', fontsize=24)
    ax.set_ylabel('Frequency (Hz)', fontsize=24)
    ax.tick_params(labelsize=20)

    box_h = 2000
    for s in slots:
        rel_t0 = (s['t_start']-t_lo)/cfg.FS
        rel_t1 = (s['t_end']-t_lo)/cfg.FS
        is_header = s['label'].startswith('header')
        color = 'red' if is_header else 'cyan'
        ax.add_patch(matplotlib.patches.Rectangle(
            (rel_t0, s['freq']-box_h), rel_t1-rel_t0, 2*box_h,
            fill=False, edgecolor=color, linewidth=2.0))
        ax.text(rel_t0, s['freq'] + (box_h+500 if is_header else -box_h-500),
                s['label'], color='red' if is_header else 'orange',
                fontsize=17, fontweight='bold',
                va='bottom' if is_header else 'top')

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
