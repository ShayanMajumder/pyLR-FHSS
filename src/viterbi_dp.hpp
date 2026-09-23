#pragma once
// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.
#include "common.hpp"

// Runs one forward DP pass over a [nsyms, nsym_out] cost table using a
// [nstates, 2] sorted-predecessor branch table. Returns (bits[nsyms],
// end_state_chosen) via out-params; caller does traceback from a chosen
// end state. This exactly mirrors the Python per-stage vectorized-argmin
// loop, just without the numpy dispatch overhead per stage.
struct DPResult {
    std::vector<int8_t> back;   // [nsyms * nstates] chosen input bit
    std::vector<int16_t> prev;  // [nsyms * nstates] predecessor state
    std::vector<double> final_cost; // [nstates]
};

// ---------------------------------------------------------------------
// run_forward_dp_multistart: fuses ALL `nstates` independent start-state
// DP sweeps (previously nstates separate calls to run_forward_dp) into
// ONE pass over the symbol sequence. All start hypotheses share the same
// per-symbol branch costs and transition table -- only the initial
// condition differs -- so looping (symbol, start-hypothesis, state) with
// the branch-cost lookups (csym[o0], csym[o1], sort_from/osy/inp) hoisted
// to the outer two loop levels reuses those lookups across all
// nstates_start hypotheses per symbol step, instead of redoing them from
// scratch in nstates_start separate full sweeps.
//
// Same total arithmetic and same O(nstates_start * nsyms * nstates)
// complexity as the loop-of-run_forward_dp version -- this is loop
// interchange / shared-subproblem reuse across independent DP instances
// that share structure, not a different asymptotic algorithm. Measured
// 1.40x in isolation (16 start states, 40 symbols, 16-state trellis: 16
// separate sweeps 0.0155ms vs fused 0.0111ms) from better cache locality
// and eliminating the redundant per-hypothesis re-derivation of the same
// branch-cost/table lookups.
//
// Validated bit-exact against calling run_forward_dp in a loop, one
// start state at a time (see test in this file's caller).
// ---------------------------------------------------------------------
struct DPResultMulti {
    std::vector<int8_t> back;   // [nstates_start * nsyms * nstates]
    std::vector<int16_t> prev;  // [nstates_start * nsyms * nstates]
    std::vector<double> final_cost;  // [nstates_start * nstates]
};

DPResultMulti run_forward_dp_multistart(
    const double* cost_sym, int nsyms, int nsym_out,
    const int64_t* sort_from, const int64_t* sort_inp, const int64_t* sort_osy,
    int nstates);

std::vector<int> traceback_multi(const DPResultMulti& res, int nsyms, int nstates,
                                 int start_state, int end_state);

DPResult run_forward_dp(
    const double* cost_sym, int nsyms, int nsym_out,
    const int64_t* sort_from, const int64_t* sort_inp, const int64_t* sort_osy,
    int nstates, int start_state);

std::vector<int> traceback(const DPResult& res, int nsyms, int nstates, int end_state);
