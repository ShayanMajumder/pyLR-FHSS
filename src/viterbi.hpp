#pragma once
// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.
#include "common.hpp"

py::array_t<int> viterbi_payload(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost_sym,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_from,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_inp,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_osy);

py::array_t<int> viterbi_header_multistart(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost_sym,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_from,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_inp,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_osy);

py::array_t<int> viterbi_header_backward(
    py::array_t<double, py::array::c_style | py::array::forcecast> cost_sym,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rsort_from,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rsort_inp,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> rsort_osy);

py::tuple viterbi_payload_full(
    py::array_t<double, py::array::c_style | py::array::forcecast> deint_payload,
    int CR, double demod_soft_val_cap,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_from,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_inp,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> sort_osy);
