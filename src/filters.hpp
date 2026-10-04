// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#pragma once
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
