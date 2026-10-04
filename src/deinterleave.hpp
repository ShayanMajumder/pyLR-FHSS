// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#pragma once
#include "common.hpp"

py::array_t<double> deinterleave_payload_ext(
    py::array_t<double, py::array::c_style | py::array::forcecast> payload,
    int data_in_bitcount);

py::array_t<std::complex<double>> deinterleave_iq_ext(
    py::array_t<float, py::array::c_style | py::array::forcecast> a,
    int out_len, int dst_offset);
