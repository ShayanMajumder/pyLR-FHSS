// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.
#include "deinterleave.hpp"

// NOTE: a bidirectional (backward) Viterbi pass for the 64-state payload
// trellis was implemented and tested here, then removed. Unlike the
// header's backward pass (kept -- see lrfhss_viterbi_ext's header
// functions above, real win, validated +80% recovery), payload's start
// state is FIXED at 0 by the encoder, so forward already fully exploits
// the one constraint that makes a bidirectional search useful. A
// controlled encode+corrupt+decode test confirmed the natural backward
// formulation (fix candidate end states, walk back, gate on recovered
// start==0) rejects real solutions at real noise levels -- the true end
// state's cost[0] is routinely not the global argmin under noise. Swept
// 300 forward-failure trials at multiple noise levels: 0 recoveries.
// See lrfhss_decode.py's equivalent comment for the full writeup.

// ---------------------------------------------------------------------
// deinterleave_payload_ext: exact port of lrfhss_decode.py's
// deinterleave_payload. Sequential index-generation loop (each `pos`
// depends on the previous iteration's state) -- same non-vectorizable
// shape as the Viterbi DP loops, small (0.095s/run at 4337 calls on a
// real recording) but free once ported. Bit-exact with the Python
// version (validated in the same trial harness as the other functions).
// ---------------------------------------------------------------------
py::array_t<double> deinterleave_payload_ext(
    py::array_t<double, py::array::c_style | py::array::forcecast> payload,
    int data_in_bitcount)
{
    auto p = payload.unchecked<1>();
    int n = (int)p.shape(0);
    py::array_t<double> out(n);
    auto o = out.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) o(i) = 0.0;
    if (n == 0) return out;

    int y = 0;
    while (y*y < data_in_bitcount) y++;
    int step = y;
    int step_v = step >> 1;
    step = step << 1;
    int st_idx = 0;
    int st_idx_init = 0;
    long long pos = 0;   // matches Python's 1-based pos tracking

    for (int i = 1; i < n; ++i) {
        pos = pos + step;
        if (pos >= data_in_bitcount) {
            st_idx = st_idx + step_v;
            if (st_idx >= step) {
                st_idx_init += 1;
                st_idx = st_idx_init;
            }
            pos = st_idx;
        }
        o(pos - 1) = p(i);
    }
    // deint_payload(2:end)=deint_payload(1:end-1); deint_payload(1)=payload(1)
    for (int i = n - 1; i >= 1; --i) o(i) = o(i - 1);
    o(0) = p(0);
    return out;
}

// ---------------------------------------------------------------------
// deinterleave_iq_ext: interleaved float32 [I0,Q0,I1,Q1,...] -> complex128
// array, placed into a zero-padded output buffer at a given offset. Fuses
// what load_frontend's inner loop did as a[0::2].astype(float64) +
// 1j*a[1::2].astype(float64) followed by a seg[...]= assignment -- three
// separate numpy ops (two strided-slice float64 copies + one complex
// assembly) per window, called once per NB-sized chunk. Single C++ pass,
// no strided intermediate copies.
// ---------------------------------------------------------------------
py::array_t<std::complex<double>> deinterleave_iq_ext(
    py::array_t<float, py::array::c_style | py::array::forcecast> a,
    int out_len, int dst_offset)
{
    auto ap = a.unchecked<1>();
    int64_t n_iq = (int64_t)ap.shape(0) / 2;   // number of complex samples in `a`

    py::array_t<std::complex<double>> out(out_len);
    // Write through a plain double* view (complex128 == two contiguous
    // doubles) instead of constructing std::complex per element. This is
    // just a float32->float64 convert + interleaved copy, which the
    // compiler auto-vectorizes as a flat pointer loop but generally will
    // not through per-element std::complex construction with an in-loop
    // bounds branch. Memory-bound (~35MB of complex output per NB chunk),
    // so the branch hoisting matters as much as the vectorization.
    double* op = reinterpret_cast<double*>(out.mutable_data());
    for (int64_t i = 0; i < 2*(int64_t)out_len; ++i) op[i] = 0.0;

    const float* ap_raw = a.data();
    int64_t lo_i = 0, hi_i = n_iq;
    if (dst_offset < 0) lo_i = -(int64_t)dst_offset;
    if ((int64_t)dst_offset + hi_i > (int64_t)out_len) hi_i = (int64_t)out_len - dst_offset;
    for (int64_t i = lo_i; i < hi_i; ++i) {
        int64_t dst = (int64_t)dst_offset + i;
        op[2*dst]     = (double)ap_raw[2*i];
        op[2*dst + 1] = (double)ap_raw[2*i + 1];
    }
    return out;
}
