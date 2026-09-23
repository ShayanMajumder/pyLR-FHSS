#pragma once
// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <complex>
#include <pybind11/complex.h>

namespace py = pybind11;

// Was `static const double INF` in the single-file build. `inline constexpr`
// keeps one definition across all translation units instead of a private
// copy per object file.
inline constexpr double INF = 1e18;
