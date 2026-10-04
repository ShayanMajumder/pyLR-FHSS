// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#pragma once
#include "common.hpp"

struct DPResult {
    std::vector<int8_t> back;   // [nsyms * nstates] chosen input bit
    std::vector<int16_t> prev;  // [nsyms * nstates] predecessor state
    std::vector<double> final_cost; // [nstates]
};

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
