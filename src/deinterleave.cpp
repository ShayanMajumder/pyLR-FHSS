// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#include "deinterleave.hpp"

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

py::array_t<std::complex<double>> deinterleave_iq_ext(
    py::array_t<float, py::array::c_style | py::array::forcecast> a,
    int out_len, int dst_offset)
{
    auto ap = a.unchecked<1>();
    int64_t n_iq = (int64_t)ap.shape(0) / 2;   // number of complex samples in `a`

    py::array_t<std::complex<double>> out(out_len);
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
