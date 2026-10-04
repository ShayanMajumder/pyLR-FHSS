// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#include "viterbi.hpp"
#include "viterbi_dp.hpp"
#include "crc.hpp"

py::array_t<int> viterbi_payload(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost_sym,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_from,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_inp,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_osy)
{
    auto cs = cost_sym.unchecked<2>();
    int nsyms = (int)cs.shape(0);
    int nsym_out = (int)cs.shape(1);
    int nstates = (int)sort_from.shape(0);

    if (nsyms == 0) return py::array_t<int>(0);

    DPResult res = run_forward_dp(cost_sym.data(), nsyms, nsym_out,
                                  sort_from.data(), sort_inp.data(), sort_osy.data(),
                                  nstates, /*start_state=*/0);
    int end_state = 0;
    double best = std::numeric_limits<double>::infinity();
    for (int s = 0; s < nstates; ++s) {
        if (res.final_cost[s] < best) { best = res.final_cost[s]; end_state = s; }
    }
    std::vector<int> bits = traceback(res, nsyms, nstates, end_state);

    py::array_t<int> out(nsyms);
    auto o = out.mutable_unchecked<1>();
    for (int i = 0; i < nsyms; ++i) o(i) = bits[i];
    return out;
}

py::array_t<int> viterbi_header_multistart(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost_sym,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_from,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_inp,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_osy)
{
    auto cs = cost_sym.unchecked<2>();
    int nsyms = (int)cs.shape(0);
    int nsym_out = (int)cs.shape(1);
    int nstates = (int)sort_from.shape(0);

    py::array_t<int> out({nstates, nsyms});
    auto o = out.mutable_unchecked<2>();
    if (nsyms == 0) return out;

    DPResultMulti res = run_forward_dp_multistart(cost_sym.data(), nsyms, nsym_out,
                                                   sort_from.data(), sort_inp.data(),
                                                   sort_osy.data(), nstates);
    for (int start = 0; start < nstates; ++start) {
        int end_state = 0;
        double best = std::numeric_limits<double>::infinity();
        for (int s = 0; s < nstates; ++s) {
            double c = res.final_cost[(size_t)start*nstates + s];
            if (c < best) { best = c; end_state = s; }
        }
        std::vector<int> bits = traceback_multi(res, nsyms, nstates, start, end_state);
        for (int i = 0; i < nsyms; ++i) o(start, i) = bits[i];
    }
    return out;
}

py::array_t<int> viterbi_header_backward(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost_sym,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rsort_from,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rsort_inp,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rsort_osy)
{
    auto cs = cost_sym.unchecked<2>();
    int nsyms = (int)cs.shape(0);
    int nsym_out = (int)cs.shape(1);
    int nstates = (int)rsort_from.shape(0);

    py::array_t<int> out({nstates, nsyms});
    auto o = out.mutable_unchecked<2>();
    if (nsyms == 0) return out;

    const double* csym_base = cost_sym.data();
    const int64_t* rf = rsort_from.data();
    const int64_t* ri = rsort_inp.data();
    const int64_t* ro = rsort_osy.data();

    for (int end_state = 0; end_state < nstates; ++end_state) {
        std::vector<double> cost(nstates, INF);
        cost[end_state] = 0.0;
        std::vector<int8_t> back((size_t)nsyms * nstates, 0);
        std::vector<int16_t> nxtst((size_t)nsyms * nstates, 0);
        std::vector<double> newcost(nstates);

        for (int i = nsyms - 1; i >= 0; --i) {
            const double* csym = csym_base + (size_t)i * nsym_out;
            for (int s = 0; s < nstates; ++s) {
                int64_t f0 = rf[s * 2 + 0], f1 = rf[s * 2 + 1];
                int64_t o0 = ro[s * 2 + 0], o1 = ro[s * 2 + 1];
                double c0 = cost[f0] + csym[o0];
                double c1 = cost[f1] + csym[o1];
                int which = (c1 < c0) ? 1 : 0;
                newcost[s] = which ? c1 : c0;
                back[(size_t)i * nstates + s] = (int8_t)ri[s * 2 + which];
                nxtst[(size_t)i * nstates + s] = (int16_t)(which ? f1 : f0);
            }
            cost.swap(newcost);
        }
        int start_state = 0;
        double best = std::numeric_limits<double>::infinity();
        for (int s = 0; s < nstates; ++s) {
            if (cost[s] < best) { best = cost[s]; start_state = s; }
        }
        int s = start_state;
        for (int i = 0; i < nsyms; ++i) {
            o(end_state, i) = back[(size_t)i * nstates + s];
            s = nxtst[(size_t)i * nstates + s];
        }
    }
    return out;
}

py::tuple viterbi_payload_full(
    py::array_t<double, py::array::c_style | py::array::forcecast> deint_payload,
    int CR, double demod_soft_val_cap,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_from,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_inp,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_osy)
{
    const double* dp_ptr = deint_payload.data();
    int ndeint = (int)deint_payload.shape(0);
    const int64_t* sf = sort_from.data();
    const int64_t* si = sort_inp.data();
    const int64_t* so_ = sort_osy.data();

    std::vector<int> bits_out;
    int info_len = 0;
    bool match = false;
    bool too_short = false;

    {
        py::gil_scoped_release release;

        double local_mul = 128.0 / demod_soft_val_cap - 1.0;
        std::vector<double> q(ndeint);
        for (int i = 0; i < ndeint; ++i) {
            double v = std::round(local_mul * dp_ptr[i]) + 128.0;
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            q[i] = v;
        }

        std::vector<double> mother;
        if (CR == 3) {
            mother = q;
        } else {
            static const int full_matrix[15] = {1,1,0,0,1,0,1,0,0,0,1,0,1,0,0};
            int period = (CR == 0) ? 15 : (CR == 1) ? 6 : 3;
            int kept_per = 0;
            for (int j = 0; j < period; ++j) kept_per += full_matrix[j];
            int n_kept = ndeint;
            int n_periods = (n_kept + kept_per - 1) / kept_per;
            mother.assign((size_t)n_periods * period, 128.0);
            int ki = 0;
            int end = 0;
            for (int p = 0; p < n_periods; ++p) {
                for (int j = 0; j < period; ++j) {
                    if (full_matrix[j] == 1 && ki < n_kept) {
                        mother[(size_t)p*period + j] = q[ki]; ki++;
                        end = p*period + j + 1;
                    }
                }
            }
            int keep_len = (end + 2) / 3 * 3;
            mother.resize(keep_len);
        }

        int nsyms = (int)mother.size() / 3;
        if (nsyms == 0) {
            too_short = true;
        } else {
            std::vector<double> cost_sym((size_t)nsyms * 8);
            for (int i = 0; i < nsyms; ++i) {
                double r0 = mother[i*3+0], r1 = mother[i*3+1], r2 = mother[i*3+2];
                for (int v = 0; v < 8; ++v) {
                    double c = 0.0;
                    c += ((v >> 2) & 1) ? (255.0 - r0) : r0;
                    c += ((v >> 1) & 1) ? (255.0 - r1) : r1;
                    c += ((v >> 0) & 1) ? (255.0 - r2) : r2;
                    cost_sym[(size_t)i*8 + v] = c;
                }
            }

            DPResult res = run_forward_dp(cost_sym.data(), nsyms, 8,
                                          sf, si, so_, 64, 0);
            int end_state = 0;
            double best = std::numeric_limits<double>::infinity();
            for (int s = 0; s < 64; ++s) {
                if (res.final_cost[s] < best) { best = res.final_cost[s]; end_state = s; }
            }
            std::vector<int> bits = traceback(res, nsyms, 64, end_state);

            if ((int)bits.size() < 22) {
                too_short = true;
            } else {
                info_len = (int)bits.size() - 16 - 6;
                int out_bits[16];
                crc16_bits(bits.data(), info_len, out_bits);
                match = true;
                for (int i = 0; i < 16; ++i) {
                    if (out_bits[i] != bits[info_len + i]) { match = false; break; }
                }
                bits_out.swap(bits);
            }
        }
    }   // GIL reacquired here

    if (too_short) return py::make_tuple(py::array_t<int>(0), false);

    py::array_t<int> info_arr(info_len);
    auto ia = info_arr.mutable_unchecked<1>();
    for (int i = 0; i < info_len; ++i) ia(i) = bits_out[i];
    return py::make_tuple(info_arr, match);
}
