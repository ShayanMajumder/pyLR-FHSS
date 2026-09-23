// viterbi_ext.cpp -- pybind11 C++ cores for the two hot sequential Viterbi
// loops in lrfhss_decode.py / lrfhss_header.py.
//
// These replace ONLY the per-symbol "for i in range(n): argmin(...)"
// Python loops. All trellis-shape tables (sorted-branch predecessor lists,
// output-symbol bit patterns, CRC LUTs) are computed once in Python (as the
// existing modules already do) and passed in as plain numpy arrays, so
// there is no duplicated/hand-transcribed trellis data here -- if the
// Python trellis constants ever change, these functions pick it up
// automatically with no C++ edits. This keeps the C++ side bit-exact by
// construction rather than by manual sync.
//
// Two entry points:
//   viterbi_payload(cost_sym, sort_from, sort_inp, sort_osy)
//       64-state rate-1/3 payload trellis, cost[0]=0 start, best-end trace.
//   viterbi_header(cost_sym, sort_from, sort_inp, sort_osy)
//       16-state rate-1/2 header trellis, ONE start state (the Python side
//       loops this 16x over start states -- see viterbi_header_multistart
//       below which does all 16 in one C++ call to cut per-call Python/pybind
//       overhead 16x).
//   viterbi_header_multistart(cost_sym, sort_from, sort_inp, sort_osy)
//       runs all 16 start states, returns the winning info-bit array for
//       each (caller still does the CRC8 check + early-exit in Python, but
//       the expensive DP inner loop -- the actual hot path -- is now C++
//       for every state).
//   viterbi_header_backward(cost_sym, rsort_from, rsort_inp, rsort_osy)
//       mirrors _decode_header_backward: all 16 END states, predecessor
//       (reverse) trellis, forward-in-i but backward-in-trellis-direction
//       DP exactly as the Python version does it.
//
// All cost math, branch selection (np.argmin over axis=1, ties broken by
// first-index like numpy), and traceback exactly mirror the numpy
// reference implementations bit-for-bit (verified against them before
// swapping call sites).

// NOTE: this file now holds only the pybind11 module definition. The
// implementations live beside it, grouped by role: crc.cpp, viterbi_dp.cpp,
// viterbi.cpp, demod.cpp, deinterleave.cpp, filters.cpp.

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
