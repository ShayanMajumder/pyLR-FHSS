# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0] - 2026-10-09

First public release.

- Blind LR-FHSS receiver: decodes all six EU868 and US915 LR-FHSS data rates
  (DR8 to DR11, DR5 and DR6) without knowing a packet's timing, carrier
  frequency or hopping sequence.
- Standard-compliant encoder that reuses the receiver's trellis, CRC,
  whitening and interleaver.
- Live reception from an SDR through SoapySDR (`lrfhss.sdr`).
- Optional C++ core (`lrfhss._viterbi_ext`), bit-exact with the numpy
  implementation, shipped in the binary wheels.

[Unreleased]: https://github.com/ShayanMajumder/LR-FHSS/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ShayanMajumder/LR-FHSS/releases/tag/v1.0.0
