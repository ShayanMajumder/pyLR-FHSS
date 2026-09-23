// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.
#include "viterbi_dp.hpp"

DPResultMulti run_forward_dp_multistart(
    const double* cost_sym, int nsyms, int nsym_out,
    const int64_t* sort_from, const int64_t* sort_inp, const int64_t* sort_osy,
    int nstates)
{
    DPResultMulti res;
    size_t plane = (size_t)nsyms * nstates;
    res.back.assign((size_t)nstates * plane, 0);
    res.prev.assign((size_t)nstates * plane, 0);

    std::vector<double> costM((size_t)nstates * nstates);
    for (int st = 0; st < nstates; ++st)
        for (int s = 0; s < nstates; ++s)
            costM[(size_t)st*nstates + s] = (s == st) ? 0.0 : INF;
    std::vector<double> newM((size_t)nstates * nstates);

    for (int i = 0; i < nsyms; ++i) {
        const double* csym = cost_sym + (size_t)i * nsym_out;
        for (int s = 0; s < nstates; ++s) {
            int64_t f0 = sort_from[s*2+0], f1 = sort_from[s*2+1];
            int64_t o0 = sort_osy[s*2+0], o1 = sort_osy[s*2+1];
            double b0 = csym[o0], b1 = csym[o1];
            int8_t inp0 = (int8_t)sort_inp[s*2+0], inp1 = (int8_t)sort_inp[s*2+1];
            for (int st = 0; st < nstates; ++st) {
                double c0 = costM[(size_t)st*nstates + f0] + b0;
                double c1 = costM[(size_t)st*nstates + f1] + b1;
                int which = (c1 < c0) ? 1 : 0;
                newM[(size_t)st*nstates + s] = which ? c1 : c0;
                size_t idx = (size_t)st*plane + (size_t)i*nstates + s;
                res.back[idx] = which ? inp1 : inp0;
                res.prev[idx] = which ? (int16_t)f1 : (int16_t)f0;
            }
        }
        costM.swap(newM);
    }
    res.final_cost = costM;
    return res;
}

std::vector<int> traceback_multi(const DPResultMulti& res, int nsyms, int nstates,
                                        int start_state, int end_state) {
    std::vector<int> bits(nsyms);
    size_t plane = (size_t)nsyms * nstates;
    size_t base = (size_t)start_state * plane;
    int s = end_state;
    for (int i = nsyms - 1; i >= 0; --i) {
        bits[i] = res.back[base + (size_t)i*nstates + s];
        s = res.prev[base + (size_t)i*nstates + s];
    }
    return bits;
}

DPResult run_forward_dp(
    const double* cost_sym, int nsyms, int nsym_out,
    const int64_t* sort_from, const int64_t* sort_inp, const int64_t* sort_osy,
    int nstates, int start_state)
{
    DPResult res;
    res.back.assign((size_t)nsyms * nstates, 0);
    res.prev.assign((size_t)nsyms * nstates, 0);
    std::vector<double> cost(nstates, INF);
    cost[start_state] = 0.0;
    std::vector<double> newcost(nstates);

    for (int i = 0; i < nsyms; ++i) {
        const double* csym = cost_sym + (size_t)i * nsym_out;
        for (int s = 0; s < nstates; ++s) {
            // two predecessor branches for state s
            int64_t f0 = sort_from[s * 2 + 0], f1 = sort_from[s * 2 + 1];
            int64_t o0 = sort_osy[s * 2 + 0], o1 = sort_osy[s * 2 + 1];
            double c0 = cost[f0] + csym[o0];
            double c1 = cost[f1] + csym[o1];
            int which = (c1 < c0) ? 1 : 0;   // argmin, ties -> index 0 (matches np.argmin)
            newcost[s] = which ? c1 : c0;
            res.back[(size_t)i * nstates + s] = (int8_t)sort_inp[s * 2 + which];
            res.prev[(size_t)i * nstates + s] = (int16_t)(which ? f1 : f0);
        }
        cost.swap(newcost);
    }
    res.final_cost = cost;
    return res;
}

std::vector<int> traceback(const DPResult& res, int nsyms, int nstates, int end_state) {
    std::vector<int> bits(nsyms);
    int s = end_state;
    for (int i = nsyms - 1; i >= 0; --i) {
        bits[i] = res.back[(size_t)i * nstates + s];
        s = res.prev[(size_t)i * nstates + s];
    }
    return bits;
}
