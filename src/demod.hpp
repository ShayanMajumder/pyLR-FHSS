#pragma once
// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.
#include "common.hpp"

py::array_t<double> demod_symbols_ext(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> sig,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> smpltime,
    int lookdist, double phaseslope, bool adapt_drift, double soft_cap);

py::tuple demod_symbols_grid(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> sig,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> gstos,
    int lookdist, double phaseslope, int nbits, int smbl,
    bool adapt_drift, double soft_cap);
