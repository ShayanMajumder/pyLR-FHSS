// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#include "common.hpp"
#include "crc.hpp"
#include "viterbi.hpp"
#include "demod.hpp"
#include "deinterleave.hpp"
#include "filters.hpp"

PYBIND11_MODULE(_viterbi_ext, m) {
    m.doc() = "Sequential Viterbi DP cores for LR-FHSS payload/header decode";
    m.def("viterbi_payload", &viterbi_payload,
         "64-state forward Viterbi, start state 0, best end state");
    m.def("viterbi_header_multistart", &viterbi_header_multistart,
         "16-state forward Viterbi over all start states, one row per start");
    m.def("viterbi_header_backward", &viterbi_header_backward,
         "16-state backward (reverse trellis) Viterbi over all end states");
    m.def("demod_symbols_grid", &demod_symbols_grid,
         "Batch demod_symbols over a whole gsto sweep in one call");
    m.def("demod_symbols_ext", &demod_symbols_ext,
         "GMSK symbol demod + drift adjust, full C++ port of demod_symbols");
    m.def("crc16_ext", &crc16_ext, "CRC16 LUT walk, exact port of crc16()");
    m.def("viterbi_payload_full", &viterbi_payload_full,
         "Fused quantize+depuncture+DP+traceback+CRC16 payload decode, one call");
    m.def("deinterleave_payload_ext", &deinterleave_payload_ext,
         "Sequential index-generation deinterleave, exact port of deinterleave_payload()");
    m.def("crc8_ext", &crc8_ext,
         "CRC8 over 32 header info bits, exact port of crc8_header()");
    m.def("deinterleave_iq_ext", &deinterleave_iq_ext,
         "Interleaved float32 I/Q -> zero-padded complex128 buffer, one pass");
    m.def("mix_and_sosfiltfilt_ext", &mix_and_sosfiltfilt_ext,
         "Fused mix-by-tone + sosfiltfilt, no intermediate temporaries");
    m.def("sosfiltfilt_ext", &sosfiltfilt_ext,
         "Full C++ sosfiltfilt (odd-ext + fwd/rev biquad cascade) for 1-D complex input");
}
