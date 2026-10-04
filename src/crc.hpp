// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#pragma once
#include "common.hpp"

py::array_t<int> crc8_ext(
    py::array_t<int, py::array::c_style | py::array::forcecast> bits32);

void crc16_bits(const int* d, int ndbits, int out_bits[16]);

py::array_t<int> crc16_ext(
    py::array_t<int, py::array::c_style | py::array::forcecast> decoded_bits);
