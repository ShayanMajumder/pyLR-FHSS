#pragma once
// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.
#include "common.hpp"

py::array_t<double> deinterleave_payload_ext(
    py::array_t<double, py::array::c_style | py::array::forcecast> payload,
    int data_in_bitcount);

py::array_t<std::complex<double>> deinterleave_iq_ext(
    py::array_t<float, py::array::c_style | py::array::forcecast> a,
    int out_len, int dst_offset);
