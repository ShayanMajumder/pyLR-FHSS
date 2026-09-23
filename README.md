# lrfhss

A blind LR-FHSS receiver. Given raw IQ, it finds packets without being
told where they are: matched-filter sync search, header decode, then the
LFSR-predicted hop schedule to gather and decode the payload.

![One decoded packet](examples/plots/packet_spectrogram.png)

The bundled recording, decoded and drawn by `examples/decode_recording.py`:
three header replicas (red), then three payload fragments (cyan), each
boxed where the receiver found it. Nothing here was told where to look;
the hop pattern is predicted from the header and confirmed by CRC.

## Install

Python 3.9+, and a C++ compiler for the accelerated core. From this
directory:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

`-e` (editable) installs the package by pointing at this directory rather
than copying it, so edits to `lrfhss/` take effect without reinstalling.
The extension is different: it is compiled, so **re-run `pip install -e .`
after changing anything in `src/`**.

Optional extras:

```bash
pip install -e ".[test]"     # pytest, to run the suite
pip install -e ".[plots]"    # matplotlib, for the spectrograms
```

Live reception from an SDR (`lrfhss.sdr`) needs
SoapySDR. It is not a pip extra because it is not on PyPI: the Python
bindings are built against the system library, so they come from your
package manager, and `pip install soapysdr` will fail.

```bash
sudo apt install soapysdr-tools python3-soapysdr
sudo apt install soapysdr-module-airspy    # or -rtlsdr, or -hackrf
```

Those land in the system `site-packages`, so a virtualenv only sees them if
you ask it to. Create it with `python3 -m venv --system-site-packages venv`,
or symlink `SoapySDR.py` and `_SoapySDR*.so` from
`/usr/lib/python3/dist-packages` into `venv/lib/python3.*/site-packages`.
Check the radio is visible before blaming the decoder:

```bash
SoapySDRUtil --find
```

### The C++ core

That same `pip install -e .` compiles `src/*.cpp` into the package as
`lrfhss._viterbi_ext`. The hot paths use it automatically: the Viterbi
payload search, the IIR filters, GMSK demodulation, deinterleaving and the
CRCs.

It is an accelerator, not a requirement. Every one of those paths falls
back to numpy, bit-for-bit identical, if the extension is missing; the
decode is simply several times slower. The fallback is silent, so check
which one you are on:

```bash
python3 -c "import lrfhss; print('C++ core:', 'active' if lrfhss.config._HAVE_VEXT_LOCAL else 'MISSING')"
```

Decoding from a file path also prints an `[accel]` line at the start. If it
says `MISSING`, the compile failed. Re-run `pip install -e .` and read the
build output; the usual cause is no compiler (`sudo apt install build-essential`).

Tuning the build is deliberate, not automatic: `NATIVE=1 pip install -e .`
adds `-march=native`, which helps the vectorised paths a little and *hurts*
the scalar filter recurrence, and ties the binary to the CPU that built it.
Measure before using it. `setup.py` has the numbers.

## Decode a capture

```python
import lrfhss

# The defaults suit one configuration. Point the front end at this
# capture first: everything derives from the occupied bandwidth.
lrfhss.config.retune(722_660, sync_word='12AD101B', hdr_count=4)

for packet in lrfhss.decode('capture.wav'):
    if packet['crc']:
        print(bytes(packet['bytes']))
```

Two parameters the receiver cannot work out for itself:

- **bandwidth** sets the front-end rate, symbol length, hop dwell times
  and the hop-search window.
- **`hdr_count`** is how many header replicas were transmitted (1-4). It is
  not carried in the header, and a wrong value puts the payload window a
  whole dwell out. That does not fail cleanly: it decodes noise, which can
  clear the 16-bit payload CRC by chance.

Grid and coding rate *are* carried in the header, so they are not settings.

### Or name a LoRaWAN data rate

A DR fixes all of it at once, including the replica count, which is why
this is the right way in for standards-compliant traffic:

```python
lrfhss.config.retune_dr(8)                  # EU868 DR8
lrfhss.config.retune_dr(5, region='US915')
```

| region | DR | bandwidth | coding rate | header replicas |
|---|---|---|---|---|
| EU868 | 8 | 137 kHz | 1/3 | 3 |
| EU868 | 9 | 137 kHz | 2/3 | 2 |
| EU868 | 10 | 336 kHz | 1/3 | 3 |
| EU868 | 11 | 336 kHz | 2/3 | 2 |
| US915 / AU915 | 5 | 1523 kHz | 1/3 | 3 |
| US915 / AU915 | 6 | 1523 kHz | 2/3 | 2 |

Only these are defined. The rest of the LR-FHSS parameter space is legal
on the air but has no DR name, so bandwidth, coding rate and replica count
can still be given directly, which is what the capture sweep uses.

Parameters must be set on `lrfhss.config`, never on the package:
modules read `config.FS` at call time, so `lrfhss.FS = ...` binds a name
nothing looks at.

`decode()` returns every candidate it considered, not just the good ones.
`packet['crc']` is the accept gate; the rejects are returned because they
are what you need when a capture will not decode.

## Generate a packet

```python
iq, meta = lrfhss.encode(b'hello world', dr=8)                    # by DR
iq, meta = lrfhss.encode(b'hello world', bw_khz=722.66, CR=0)     # or directly
```

The encoder reuses the decoder's own trellis, CRC, whitening and
interleaver, so what it produces is byte-compatible by construction.
`examples/encode.py` writes a .wav and decodes it back.

Round trips at every bandwidth in the hop tables, at header-replica counts
1–4, and at every LoRaWAN data rate.

Known bug, pinned by a strict xfail: CR=0 (5/6) depends on payload
content: 2 of 24 random payloads round trip, against 24/24 for CR=1 and
CR=3 and 23/24 for CR=2. Generation only; decoding real CR=0 captures is
unaffected. See `lrfhss/encoder.py`.

## Tests

```bash
pip install -e ".[test]"
pytest              # 88 tests, ~55 s
```

Self-contained: nothing is skipped and nothing needs external data. Most
tests generate their own signal with the encoder; the rest decode
`examples/data/capture_39kHz_hdr3.wav`, a real 39.06 kHz over-the-air packet
kept small (2 MB rather than 62 MB) by committing it already decimated
to 166.667 kHz. `retune(39_060)` derives exactly that rate, so the
samples go straight to `decode()`.

## Layout

```
lrfhss/
  config       every retunable parameter, retune(), DR tables
  fft          FFT entry points
  wavio        .wav parsing, chunked IQ reads
  frontend     decimating channeliser
  dsp          filters, tone cache, spur notching
  detect       matched filter, CFAR, packet search, fine sync
  header       hop frequency, CFO, header decode, replica index
  payload      payload window, hop footprint, payload decode
  quality      energy/length checks
  plotting     per-packet spectrograms
  acquire      IQ + candidates; acquiring one candidate
  arbitrate    which candidates are real packets
  report       console output
  pipeline     decode() -- orders the above
  encoder      encode() -- the transmit side
  phy/         protocol primitives, independent of the pipeline
    hopping    LFSR hop-frequency generator
    fec        trellis, Viterbi, interleaving, whitening
    gmsk       GMSK symbol demodulation
    framing    header field layout and decode
src/           C++ core, one file per role
examples/      runnable scripts; edit the settings block at the top
```

`arbitrate` is the one to read first if you are changing behaviour. One
physical packet appears as several candidates (its header replicas sit on
different frequencies and each can clear the matched filter), and deciding
which are real is where this receiver has historically gone wrong. The
rules, and why they are the way they are, are in that module.

## Examples

```bash
python3 examples/decode_recording.py   # the bundled recording, no setup
python3 examples/encode.py             # generate a packet and decode it back
python3 examples/live_receive.py       # decode off an SDR, live
```

Each is a flat script: edit the settings block at the top and run it.
`decode_recording.py` needs nothing but the repo. `encode.py` needs no
radio either, since it decodes back what it just generated.
`live_receive.py` needs an SDR and SoapySDR, and drives
`lrfhss.sdr.LiveReceiver`, which owns the radio, the reader thread and the
windowing.

To have something to receive, [`examples/arduino/`](examples/arduino) has
two transmitter sketches for an STM32 + SX1262 or LR1120, each beaconing
one LR-FHSS packet every few seconds at the settings `live_receive.py`
expects. Its README covers wiring, flashing and the per-radio traps.
