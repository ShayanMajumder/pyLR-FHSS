// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.
#include "demod.hpp"

// ---------------------------------------------------------------------
// demod_symbols: full port of lrfhss_demod.py's demod_symbols. Folds the
// per-symbol phase-slope measurement, wrap rule, percentile-based drift
// threshold, two-pass drift estimate, and final clip into one C++ call.
// Was 11617 calls/run in profiling, 2.58s total with 0.92s of that pure
// np.percentile overhead (full sort dispatch via numpy's Python-level
// quantile machinery) plus per-call numpy op dispatch for angle/where/
// sign/mean/sum on small arrays -- none of it a true sequential
// dependency, but chaining ~10 numpy calls per invocation at this call
// count adds up. Matches numpy's percentile(..., 80) EXACTLY: same
// linear-interpolation formula (idx=(n-1)*p/100, interp between
// sorted[floor(idx)] and sorted[ceil(idx)]), validated against the numpy
// reference (see ext/test_viterbi_ext.py).
//
// sig: complex128 array passed DIRECTLY (no re/im split). Passing the
// complex array as-is avoids the two np.ascontiguousarray copies the
// split version forced -- .real/.imag of a complex128 array are strided
// views, so handing them to C++ required materializing two contiguous
// float64 arrays on EVERY call. At this function's real call volume
// (343,966 calls on a -10dB 122-cluster run) that copy tax measured
// 698,836 ascontiguousarray calls / 6.66s -- the single largest cost in
// the whole pipeline, larger than sosfilt or the Viterbi. std::complex
// is layout-compatible with numpy complex128, so this binding reads the
// caller's buffer in place with no copy at all.
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

    // percentile(|bits|, 80), numpy 'linear' method, bit-exact formula.
    // Was a full std::sort (O(n log n)) just to read off two order
    // statistics (ranks ilo, ihi). std::nth_element is O(n) average --
    // the competitive-programming substitution for "I need the k-th
    // smallest element" when the full sorted order isn't needed. Two
    // nth_element calls (ilo, then ihi restricted to the remaining
    // partition v[ilo+1:]) still total O(n), strictly less work than one
    // O(n log n) sort. Validated bit-exact against the sort-based version
    // over 2000 randomized trials (max diff 0); measured 2.18x-3.85x at
    // n=40..500 (the growing gap matches the O(n log n) vs O(n) theory --
    // benefit increases with n, as expected).
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

// ---------------------------------------------------------------------
// demod_symbols_grid: batch demod_symbols over a whole gsto (symbol-timing
// offset) sweep in ONE call, instead of one pybind11 call per gsto.
//
// try_thishdridx's payload search runs
//     for dcfo: for gsto in range(0, SMBL, 3): for each fragment: demod
// which measured 343,966 demod_symbols_ext calls / 3.455s on a -10dB run
// -- at ~50 sample centers per call, that is dominated by per-call
// dispatch + array allocation, not by the phase-slope arithmetic. This
// hoists the gsto loop into C++ so the whole 114-value sweep for one
// (fragment, dcfo) pair costs a single crossing.
//
// Math per gsto row is IDENTICAL to demod_symbols_ext (same centers rule
// gsto + lookdist + i*smbl, same wrap rule, same percentile-80 drift
// threshold and two-pass adjust, same clip), so output is bit-for-bit the
// same as looping the scalar version -- validated against it directly.
//
// Returns [n_gsto, nbits] soft values. A row whose centers don't fit in
// the signal is returned as all-zeros and flagged 0 in `valid`, matching
// the Python caller's own "parts.append(np.zeros(nbits))" fallback.
// ---------------------------------------------------------------------
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
    // Raw pointers taken under the GIL; compute below runs with it
    // released so worker threads parallelize (see viterbi_payload_full).
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

        // percentile(|bits|, 80), numpy 'linear' method (same fix as
        // demod_symbols_ext: nth_element instead of full sort -- see that
        // site's comment for validation/benchmark detail). This call site
        // runs inside the gsto sweep (up to 114x per fragment per
        // hypothesis), higher volume than the header path, so the O(n) vs
        // O(n log n) gap matters more here.
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
