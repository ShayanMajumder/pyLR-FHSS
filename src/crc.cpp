// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#include "crc.hpp"
#include "crc8_lut.inc"
#include "crc16_lut.inc"

py::array_t<int> crc8_ext(
    py::array_t<int, py::array::c_style | py::array::forcecast> bits32)
{
    const int* d = bits32.data();
    int n = (int)bits32.shape(0);   // expected 32, but don't assume
    uint8_t crc = 0xFF;   // np.ones(8,int) packed MSB-first = 0b11111111
    int rounds = n / 8;
    for (int k = 0; k < rounds; ++k) {
        uint8_t top = 0;
        for (int b = 0; b < 8; ++b) {
            int crc_bit = (crc >> (7-b)) & 1;
            int d_bit = d[k*8+b] & 1;
            top = (uint8_t)((top << 1) | (crc_bit ^ d_bit));
        }
        crc = CRC8_LUT[top];
    }
    py::array_t<int> out(8);
    auto o = out.mutable_unchecked<1>();
    for (int i = 0; i < 8; ++i) o(i) = (crc >> (7-i)) & 1;
    return out;
}

void crc16_bits(const int* d, int ndbits, int out_bits[16]) {
    uint16_t crc = 0xFFFF;   // np.ones(16) as an MSB-first bit array == 0xFFFF
    int nbytes = ndbits / 8;
    for (int k = 0; k < nbytes; ++k) {
        uint8_t byte = 0;
        for (int b = 0; b < 8; ++b) byte = (byte << 1) | (uint8_t)(d[k*8+b] & 1);
        uint8_t top8 = (uint8_t)((crc >> 8) & 0xFF) ^ byte;
        uint16_t lut = CRC16_LUT[top8];
        uint16_t low8 = (uint16_t)(crc & 0xFF);
        uint16_t shifted = (uint16_t)(low8 << 8);   // [crc(8:16), 0,0,0,0,0,0,0,0]
        crc = shifted ^ lut;
    }
    for (int i = 0; i < 16; ++i) out_bits[i] = (crc >> (15 - i)) & 1;
}

py::array_t<int> crc16_ext(
    py::array_t<int, py::array::c_style | py::array::forcecast> decoded_bits)
{
    const int* dv = decoded_bits.data();
    int n = (int)decoded_bits.shape(0);
    int out_bits[16];
    crc16_bits(dv, n, out_bits);
    py::array_t<int> out(16);
    auto o = out.mutable_unchecked<1>();
    for (int i = 0; i < 16; ++i) o(i) = out_bits[i];
    return out;
}
