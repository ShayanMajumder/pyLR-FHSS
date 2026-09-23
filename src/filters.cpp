// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.
#include "filters.hpp"

// ---------------------------------------------------------------------
// sosfiltfilt_ext: full C++ port of scipy.signal.sosfiltfilt for 1-D
// complex128 input with real second-order-section coefficients.
//
// Profiled as the single biggest remaining cost in the windowed pipeline
// (scipy sosfilt: 3464 calls / 1.42s of a 6.7s run). scipy's sosfilt
// kernel is C, but each sosfiltfilt call pays Python-level overhead for
// odd-extension construction, zi scaling, two separate sosfilt
// invocations, and two array reversals -- all of which fuse into one
// pass here.
//
// Replicates scipy exactly:
//   * odd extension: left = 2*x[0] - x[n:0:-1],
//                    right = 2*x[-1] - x[-2:-(n+2):-1]
//   * Direct Form II transposed biquad cascade, matching
//     scipy/signal/_sosfilt.pyx: per sample, per section:
//        x_cur    = b0*x_n + z0
//        z0       = b1*x_n - a1*x_cur + z1
//        z1       = b2*x_n - a2*x_cur
//        x_n      = x_cur
//     (a0 is assumed normalized to 1, exactly as scipy's kernel does --
//      sos[:,3] is not read)
//   * forward pass with zi scaled by ext[0], reverse, second pass with zi
//     scaled by that pass's first sample, reverse back, trim `edge`.
//
// zi (the steady-state initial conditions from sosfilt_zi) is passed in
// from Python where it's already cached per-sos -- recomputing it is a
// per-section linear solve and has nothing to do with the sample loop.
// ---------------------------------------------------------------------
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
    auto zia = zi_arr.unchecked<2>();   // [n_sections, 2]

    std::vector<double> sos(n_sections*6);
    for (int s = 0; s < n_sections; ++s)
        for (int k = 0; k < 6; ++k) sos[s*6+k] = so(s, k);

    // Output allocated with the GIL held; the filter itself runs with it
    // released so Python worker threads actually run in parallel (see the
    // GIL note in viterbi_payload_full -- this is the enabler for
    // --workers N on a multi-core machine).
    py::array_t<std::complex<double> > out(n);
    double* op = reinterpret_cast<double*>(out.mutable_data());
    const double* xr0 = reinterpret_cast<const double*>(x_arr.data());
    const double* zir = zi_arr.data();
    {
    py::gil_scoped_release release;

    // ---- odd extension ----
    int next = n + 2*edge;
    std::vector<std::complex<double> > ext(next);
    // Same vectorization fix as deinterleave_iq_ext: write through a flat
    // double* view of the complex buffer instead of per-element
    // std::complex construction, which GCC's vectorizer reports as
    // unvectorizable ("more than one data ref in stmt") through the
    // std::complex assignment operator. This loop is pure elementwise
    // copy/arithmetic with no cross-iteration dependency, unlike
    // sos_forward's genuine IIR recurrence below, which cannot vectorize
    // regardless.
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

    // ---- trim the extension ---- (still inside the GIL-released scope;
    // `op` points at the output buffer allocated before the release)
    const double* er_out = reinterpret_cast<const double*>(ext.data());
    for (int i = 0; i < n; ++i) {
        op[2*i]     = er_out[2*(edge+i)];
        op[2*i + 1] = er_out[2*(edge+i) + 1];
    }
    }   // GIL reacquired here
    return out;
}

// ---------------------------------------------------------------------
// mix_and_sosfiltfilt_ext: fuses the receiver's ubiquitous
//     tuned = sosfiltfilt(sos, seg * exp(-2j*pi*f*n/fs))
// into a single pass. The mixing tone was being materialized as a full
// complex temporary per call -- in find_packets' region loop alone that
// is ~776 regions x ~23347 samples = ~18M complex exponentials per run,
// plus a second full-length temporary for the product, and the same
// pattern repeats in decode_header_at / try_thishdridx (937 filtfilt
// calls). Here the tone is evaluated per sample straight into the
// extension buffer, so neither temporary is ever allocated.
//
// Uses std::exp per sample (NOT an incremental phase recurrence): a
// recurrence would accumulate phase drift over 20k+ samples and diverge
// from numpy's elementwise exp, which is what the rest of the receiver
// was tuned against. Filtering is the same Direct Form II transposed
// cascade as sosfiltfilt_ext.
// ---------------------------------------------------------------------
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
