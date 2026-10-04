// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#include "filters.hpp"

static void sos_forward(const double* sos, int n_sections,
                        std::complex<double>* x, int n,
                        const std::complex<double>* zi_scaled)
{
    std::vector<std::complex<double> > z(2*n_sections);
    for (int s = 0; s < n_sections; ++s) {
        z[2*s+0] = zi_scaled[2*s+0];
        z[2*s+1] = zi_scaled[2*s+1];
    }
    for (int i = 0; i < n; ++i) {
        std::complex<double> x_n = x[i];
        for (int s = 0; s < n_sections; ++s) {
            double b0 = sos[s*6+0], b1 = sos[s*6+1], b2 = sos[s*6+2];
            double a1 = sos[s*6+4], a2 = sos[s*6+5];
            std::complex<double> x_cur = b0*x_n + z[2*s+0];
            z[2*s+0] = b1*x_n - a1*x_cur + z[2*s+1];
            z[2*s+1] = b2*x_n - a2*x_cur;
            x_n = x_cur;
        }
        x[i] = x_n;
    }
}

py::array_t<std::complex<double> > sosfiltfilt_ext(
    py::array_t<double, py::array::c_style | py::array::forcecast> sos_arr,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> x_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> zi_arr,
    int edge)
{
    auto so = sos_arr.unchecked<2>();
    int n_sections = (int)so.shape(0);
    auto xa = x_arr.unchecked<1>();
    int n = (int)xa.shape(0);

    std::vector<double> sos(n_sections*6);
    for (int s = 0; s < n_sections; ++s)
        for (int k = 0; k < 6; ++k) sos[s*6+k] = so(s, k);

    py::array_t<std::complex<double> > out(n);
    double* op = reinterpret_cast<double*>(out.mutable_data());
    const double* xr0 = reinterpret_cast<const double*>(x_arr.data());
    const double* zir = zi_arr.data();
    {
    py::gil_scoped_release release;

    // ---- odd extension ----
    int next = n + 2*edge;
    std::vector<std::complex<double> > ext(next);
    const double* xr = xr0;
    double* er = reinterpret_cast<double*>(ext.data());
    double x0r = xr[0], x0i = xr[1];
    double xlr = xr[2*(n-1)], xli = xr[2*(n-1)+1];
    for (int i = 0; i < edge; ++i) {
        int64_t li = edge - i;
        er[2*i]     = 2.0*x0r - xr[2*li];
        er[2*i + 1] = 2.0*x0i - xr[2*li + 1];
        int64_t ri = n - 2 - i;
        er[2*(edge+n+i)]     = 2.0*xlr - xr[2*ri];
        er[2*(edge+n+i) + 1] = 2.0*xli - xr[2*ri + 1];
    }
    for (int i = 0; i < n; ++i) {
        er[2*(edge+i)]     = xr[2*i];
        er[2*(edge+i) + 1] = xr[2*i + 1];
    }

    // ---- forward pass, zi scaled by ext[0] ----
    std::vector<std::complex<double> > zs(2*n_sections);
    std::complex<double> e0 = ext[0];
    for (int s = 0; s < n_sections; ++s) {
        zs[2*s+0] = zir[s*2+0]*e0;
        zs[2*s+1] = zir[s*2+1]*e0;
    }
    sos_forward(sos.data(), n_sections, ext.data(), next, zs.data());

    // ---- reverse, second pass, zi scaled by that pass's first sample ----
    std::reverse(ext.begin(), ext.end());
    std::complex<double> y0 = ext[0];
    for (int s = 0; s < n_sections; ++s) {
        zs[2*s+0] = zir[s*2+0]*y0;
        zs[2*s+1] = zir[s*2+1]*y0;
    }
    sos_forward(sos.data(), n_sections, ext.data(), next, zs.data());
    std::reverse(ext.begin(), ext.end());

    const double* er_out = reinterpret_cast<const double*>(ext.data());
    for (int i = 0; i < n; ++i) {
        op[2*i]     = er_out[2*(edge+i)];
        op[2*i + 1] = er_out[2*(edge+i) + 1];
    }
    }   // GIL reacquired here
    return out;
}

py::array_t<std::complex<double> > mix_and_sosfiltfilt_ext(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> sig_arr,
    double freq_hz, double fs,
    py::array_t<double, py::array::c_style | py::array::forcecast> sos_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> zi_arr,
    int edge)
{
    auto sg = sig_arr.unchecked<1>();
    int n = (int)sg.shape(0);
    auto so = sos_arr.unchecked<2>();
    int n_sections = (int)so.shape(0);
    auto zia = zi_arr.unchecked<2>();

    std::vector<double> sos(n_sections*6);
    for (int s = 0; s < n_sections; ++s)
        for (int k = 0; k < 6; ++k) sos[s*6+k] = so(s, k);

    const double TWO_PI = 6.283185307179586476925286766559;
    // mixed signal (no temporary handed back to Python)
    std::vector<std::complex<double> > x(n);
    for (int i = 0; i < n; ++i) {
        double ph = -TWO_PI * freq_hz * ((double)i / fs);
        x[i] = sg(i) * std::complex<double>(std::cos(ph), std::sin(ph));
    }

    if (n <= edge || edge < 1) {
        py::array_t<std::complex<double> > out(n);
        auto o = out.mutable_unchecked<1>();
        for (int i = 0; i < n; ++i) o(i) = x[i];
        return out;   // caller falls back / signal too short to pad
    }

    int next = n + 2*edge;
    std::vector<std::complex<double> > ext(next);
    std::complex<double> x0 = x[0], xl = x[n-1];
    for (int i = 0; i < edge; ++i) {
        ext[i] = 2.0*x0 - x[edge - i];
        ext[edge + n + i] = 2.0*xl - x[n - 2 - i];
    }
    for (int i = 0; i < n; ++i) ext[edge + i] = x[i];

    std::vector<std::complex<double> > zs(2*n_sections);
    std::complex<double> e0 = ext[0];
    for (int s = 0; s < n_sections; ++s) {
        zs[2*s+0] = zia(s, 0)*e0;
        zs[2*s+1] = zia(s, 1)*e0;
    }
    sos_forward(sos.data(), n_sections, ext.data(), next, zs.data());

    std::reverse(ext.begin(), ext.end());
    std::complex<double> y0 = ext[0];
    for (int s = 0; s < n_sections; ++s) {
        zs[2*s+0] = zia(s, 0)*y0;
        zs[2*s+1] = zia(s, 1)*y0;
    }
    sos_forward(sos.data(), n_sections, ext.data(), next, zs.data());
    std::reverse(ext.begin(), ext.end());

    py::array_t<std::complex<double> > out(n);
    auto o = out.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) o(i) = ext[edge + i];
    return out;
}
