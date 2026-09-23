// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.
#include "viterbi.hpp"
#include "viterbi_dp.hpp"
#include "crc.hpp"

// ---------------------------------------------------------------------
// viterbi_payload: 64-state, single start state (state 0), best end state.
// ---------------------------------------------------------------------
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

// ---------------------------------------------------------------------
// viterbi_header_multistart: 16-state, tries all `nstates` start states
// (mirrors the Python for start_state in range(16) loop). Returns a
// [nstates, nsyms] int array -- row k is the decoded bit sequence for
// start_state=k, end_state=argmin(cost) for that run. Caller (Python)
// still does the CRC8 check per row and stops at the first pass, exactly
// as before -- only the DP inner loop moves to C++.
// ---------------------------------------------------------------------
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

// ---------------------------------------------------------------------
// viterbi_header_backward: mirrors _decode_header_backward -- runs the
// REVERSE-direction trellis (predecessor lookup, i from n-1 down to 0)
// for all `nstates` END states. rsort_from/inp/osy are the reverse
// (predecessor) sorted tables, grouped by SUCCESSOR state exactly as
// _RBR_SORT_* is in Python. Returns [nstates, nsyms] like multistart,
// one row per end_state, with the traceback direction reversed to match
// the Python backward decoder's forward-order bit assembly.
// ---------------------------------------------------------------------
py::array_t<int> viterbi_header_backward(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost_sym,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rsort_from,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rsort_inp,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rsort_osy)
{
    // NOT fused like viterbi_header_multistart above. That fusion is
    // worth it there because forward multistart runs on EVERY grid point
    // (528 times/cluster -- see decode_header_at's df/cfo sweep). Backward
    // only runs as an opt-in fallback when forward finds nothing at all
    // across the whole grid -- a rare path, not the hot one -- so fusing
    // this too would add real risk (another full DP restructure to
    // re-validate) for a win that rarely executes. Left as the original
    // per-end-state loop; revisit if backward-path profiling ever shows
    // otherwise.
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

// ---------------------------------------------------------------------
// viterbi_payload_full: fused end-to-end payload decode. Takes the raw
// deinterleaved soft array + CR straight from deinterleave_payload's
// output, does quantize -> depuncture -> cost_sym -> 64-state Viterbi DP
// -> traceback -> CRC16 check, all in one C++ call. Was previously 4-5
// separate Python/numpy stages per call (quantize, depuncture loop for
// punctured CRs, cost_sym build via np.stack, viterbi_payload call,
// crc16 call) -- profiling showed viterbi_decode_payload's WRAPPER cost
// (deinterleave_payload + crc16 + array shuffling) exceeded the DP
// itself once the DP moved to C++ (0.898s cumulative vs 0.099s pure DP
// on a real run). This collapses the whole function to one boundary
// crossing. Bit-exact with lrfhss_decode.py's viterbi_decode_payload
// (validated in ext/test_viterbi_ext.py).
//
// Returns: (info_bits[nsyms-16-6] or empty, match: bool) via a 2-tuple.
// ---------------------------------------------------------------------
py::tuple viterbi_payload_full(
    py::array_t<double, py::array::c_style | py::array::forcecast> deint_payload,
    int CR, double demod_soft_val_cap,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_from,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_inp,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_osy)
{
    // GIL handling: pybind11 holds the GIL for the whole call by default,
    // which serializes every worker thread in Python's ThreadPoolExecutor
    // and is why threading measured ZERO speedup on this pipeline before.
    // Raw pointers are taken while the GIL is held, then it's released for
    // the pure-C++ compute (no Python object access inside), and
    // reacquired implicitly on scope exit before the result arrays are
    // built. This is what actually lets --workers N parallelize on a
    // multi-core machine.
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
            for (int p = 0; p < n_periods; ++p) {
                for (int j = 0; j < period; ++j) {
                    if (full_matrix[j] == 1 && ki < n_kept) {
                        mother[(size_t)p*period + j] = q[ki]; ki++;
                    }
                }
            }
            int keep_len = ((int)mother.size() / 3) * 3;
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
