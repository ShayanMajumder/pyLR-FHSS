#pragma once
// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.
#include "common.hpp"

py::array_t<std::complex<double> > sosfiltfilt_ext(
    py::array_t<double, py::array::c_style | py::array::forcecast> sos_arr,
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> x_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> zi_arr,
    int edge);

py::array_t<std::complex<double> > mix_and_sosfiltfilt_ext(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> sig_arr,
    double freq_hz, double fs,
    py::array_t<double, py::array::c_style | py::array::forcecast> sos_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> zi_arr,
    int edge);
