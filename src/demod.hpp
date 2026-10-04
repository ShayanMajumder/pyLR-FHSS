// Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
// SPDX-License-Identifier: MIT
#pragma once
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
