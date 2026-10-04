// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#include "demod.hpp"

py::array_t<double> demod_symbols_ext(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> sig,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> smpltime,
    int lookdist, double phaseslope, bool adapt_drift, double soft_cap)
{
    auto sg = sig.unchecked<1>();
    auto st = smpltime.unchecked<1>();
    int64_t nsig = (int64_t)sg.shape(0);
    int n = (int)st.shape(0);

    std::vector<double> bits(n, 0.0);
    const double PI = 3.14159265358979323846;
    const double HALF_PI = PI / 2.0;
    const double TWO_PI = 2.0 * PI;

    for (int i = 0; i < n; ++i) {
        int64_t c = st(i);
        int64_t lo = c - lookdist, hi = c + lookdist;
        if (lo < 0 || hi >= nsig) continue;   // matches Python's `valid` mask -> bits stays 0
        std::complex<double> vh = sg(hi), vl = sg(lo);
        double phase_hi = std::atan2(vh.imag(), vh.real());
        double phase_lo = std::atan2(vl.imag(), vl.real());
        double diff = phase_hi - phase_lo;
        if (std::fabs(diff) > HALF_PI) {
            double sgn = (diff > 0) ? 1.0 : ((diff < 0) ? -1.0 : 0.0);
            diff = diff - sgn * TWO_PI;
        }
        if (std::fabs(diff) > HALF_PI) diff = 0.0;
        double k = diff / (lookdist * 2.0);
        bits[i] = k / phaseslope;
    }

    std::vector<double> absbits(n);
    for (int i = 0; i < n; ++i) absbits[i] = std::fabs(bits[i]);
    double thr = 0.0;
    if (n > 0) {
        std::vector<double> sorted_abs = absbits;
        double idx = (n - 1) * 80.0 / 100.0;
        int ilo = (int)std::floor(idx);
        int ihi = (int)std::ceil(idx);
        std::nth_element(sorted_abs.begin(), sorted_abs.begin()+ilo, sorted_abs.end());
        double vlo = sorted_abs[ilo];
        double vhi;
        if (ihi == ilo) {
            vhi = vlo;
        } else {
            std::nth_element(sorted_abs.begin()+ilo+1, sorted_abs.begin()+ihi, sorted_abs.end());
            vhi = sorted_abs[ihi];
        }
        double frac = idx - ilo;
        thr = vlo + (vhi - vlo) * frac;
    }

    std::vector<uint8_t> useidx(n);
    int nuse = 0;
    for (int i = 0; i < n; ++i) {
        useidx[i] = (absbits[i] < thr) ? 1 : 0;
        if (useidx[i]) nuse++;
    }
    double adj = 0.0;
    for (int zzz = 0; zzz < 2; ++zzz) {
        double sum = 0.0; int cnt = 0; int lookcnt = 0;
        for (int i = 0; i < n; ++i) {
            bool look0 = (zzz == 0) ? (bits[i] < 0) : (bits[i] >= 0);
            bool look = look0 && useidx[i];
            if (look) { sum += bits[i]; cnt++; lookcnt++; }
        }
        double val = (cnt > 0) ? (sum / cnt) : 0.0;
        double frac = (double)lookcnt / (double)std::max(nuse, 1);
        adj += (zzz == 0) ? frac * (val + 1.0) : frac * (val - 1.0);
    }

    py::array_t<double> out(n);
    auto o = out.mutable_unchecked<1>();
    for (int i = 0; i < n; ++i) {
        double v = bits[i];
        if (adapt_drift) v -= adj;
        if (v > soft_cap) v = soft_cap;
        if (v < -soft_cap) v = -soft_cap;
        o(i) = v;
    }
    return out;
}

py::tuple demod_symbols_grid(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> sig,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> gstos,
    int lookdist, double phaseslope, int nbits, int smbl,
    bool adapt_drift, double soft_cap)
{
    int64_t nsig = (int64_t)sig.shape(0);
    int n_gsto = (int)gstos.shape(0);

    py::array_t<double> out({n_gsto, nbits});
    py::array_t<int> valid(n_gsto);
    const std::complex<double>* sg_p = sig.data();
    const int64_t* gs_p = gstos.data();
    double* o_p = out.mutable_data();
    int* v_p = valid.mutable_data();
    {
    py::gil_scoped_release release;

    const double PI = 3.14159265358979323846;
    const double HALF_PI = PI / 2.0;
    const double TWO_PI = 2.0 * PI;

    // caller demods nbits+2 centers then slices [1 : 1+nbits]
    int ncent = nbits + 2;
    std::vector<double> bits(ncent);
    std::vector<double> absbits(ncent);
    std::vector<double> sorted_abs(ncent);
    std::vector<uint8_t> useidx(ncent);

    for (int g = 0; g < n_gsto; ++g) {
        int64_t gsto = gs_p[g];
        // count valid centers exactly as the Python caller does
        int n_valid = 0;
        for (int i = 0; i < ncent; ++i) {
            int64_t c = gsto + lookdist + (int64_t)i * smbl;
            if (c < nsig - lookdist) n_valid++;
        }
        if (n_valid < nbits + 1) {
            v_p[g] = 0;
            for (int b = 0; b < nbits; ++b) o_p[(size_t)g*nbits + b] = 0.0;
            continue;
        }
        v_p[g] = 1;

        int m = n_valid;   // number of centers actually demodulated
        for (int i = 0; i < m; ++i) {
            int64_t c = gsto + lookdist + (int64_t)i * smbl;
            int64_t lo = c - lookdist, hi = c + lookdist;
            double val = 0.0;
            if (lo >= 0 && hi < nsig) {
                std::complex<double> vh = sg_p[hi], vl = sg_p[lo];
                double phase_hi = std::atan2(vh.imag(), vh.real());
                double phase_lo = std::atan2(vl.imag(), vl.real());
                double diff = phase_hi - phase_lo;
                if (std::fabs(diff) > HALF_PI) {
                    double sgn = (diff > 0) ? 1.0 : ((diff < 0) ? -1.0 : 0.0);
                    diff = diff - sgn * TWO_PI;
                }
                if (std::fabs(diff) > HALF_PI) diff = 0.0;
                val = (diff / (lookdist * 2.0)) / phaseslope;
            }
            bits[i] = val;
        }

        for (int i = 0; i < m; ++i) absbits[i] = std::fabs(bits[i]);
        sorted_abs.assign(absbits.begin(), absbits.begin() + m);
        double idx = (m - 1) * 80.0 / 100.0;
        int ilo = (int)std::floor(idx);
        int ihi = (int)std::ceil(idx);
        std::nth_element(sorted_abs.begin(), sorted_abs.begin()+ilo, sorted_abs.begin()+m);
        double vlo = sorted_abs[ilo];
        double vhi;
        if (ihi == ilo) {
            vhi = vlo;
        } else {
            std::nth_element(sorted_abs.begin()+ilo+1, sorted_abs.begin()+ihi, sorted_abs.begin()+m);
            vhi = sorted_abs[ihi];
        }
        double frac = idx - ilo;
        double thr = vlo + (vhi - vlo) * frac;

        int nuse = 0;
        for (int i = 0; i < m; ++i) {
            useidx[i] = (absbits[i] < thr) ? 1 : 0;
            if (useidx[i]) nuse++;
        }
        double adj = 0.0;
        for (int zzz = 0; zzz < 2; ++zzz) {
            double sum = 0.0; int cnt = 0;
            for (int i = 0; i < m; ++i) {
                bool look0 = (zzz == 0) ? (bits[i] < 0) : (bits[i] >= 0);
                if (look0 && useidx[i]) { sum += bits[i]; cnt++; }
            }
            double mean = (cnt > 0) ? (sum / cnt) : 0.0;
            double fr = (double)cnt / (double)std::max(nuse, 1);
            adj += (zzz == 0) ? fr * (mean + 1.0) : fr * (mean - 1.0);
        }

        // caller takes sb[1 : 1+nbits]
        for (int b = 0; b < nbits; ++b) {
            int i = b + 1;
            double val = (i < m) ? bits[i] : 0.0;
            if (adapt_drift) val -= adj;
            if (val > soft_cap) val = soft_cap;
            if (val < -soft_cap) val = -soft_cap;
            o_p[(size_t)g*nbits + b] = val;
        }
    }
    }   // GIL reacquired here
    return py::make_tuple(out, valid);
}
