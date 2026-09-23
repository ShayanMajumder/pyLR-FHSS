#pragma once
// Part of the lrfhss_viterbi_ext pybind11 extension.
// Split out of the former single-file viterbi_ext.cpp; the code below is
// unchanged apart from the includes and linkage needed to compile
// separately. See module.cpp for the module-level documentation.
#include "common.hpp"

py::array_t<int> crc8_ext(
    py::array_t<int, py::array::c_style | py::array::forcecast> bits32);

// Shared with viterbi.cpp (viterbi_payload_full checks CRC16 per candidate),
// so this can no longer be static.
void crc16_bits(const int* d, int ndbits, int out_bits[16]);

py::array_t<int> crc16_ext(
    py::array_t<int, py::array::c_style | py::array::forcecast> decoded_bits);
