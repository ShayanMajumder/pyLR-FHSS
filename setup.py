# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Builds lrfhss._viterbi_ext.

The extension is an accelerator: without it every routine falls back to
numpy, bit-for-bit identical. So a failed compile only warns, unless the
build asks for it explicitly:

    LRFHSS_REQUIRE_EXT=1   fail the install if the extension does not build
                           (set by CI and the wheel builds)
    LRFHSS_NO_EXT=1        do not build it at all (tests the numpy fallback)
    NATIVE=1               add -march=native (local builds only, never wheels)
"""
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

REQUIRE_EXT = os.environ.get("LRFHSS_REQUIRE_EXT", "0") == "1"
NO_EXT = os.environ.get("LRFHSS_NO_EXT", "0") == "1"
NATIVE = os.environ.get("NATIVE", "0") == "1"

SOURCES = [
    "src/crc.cpp",
    "src/deinterleave.cpp",
    "src/demod.cpp",
    "src/filters.cpp",
    "src/module.cpp",
    "src/viterbi.cpp",
    "src/viterbi_dp.cpp",
]


class BuildExt(build_ext):
    """Optimisation flags the compiler actually understands."""

    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            flags = ["/O2"]
        else:
            flags = ["-O3", "-funroll-loops"]
            if NATIVE:
                flags.append("-march=native")
        for ext in self.extensions:
            ext.extra_compile_args += flags
        super().build_extensions()


ext_modules = [] if NO_EXT else [
    Pybind11Extension(
        "lrfhss._viterbi_ext",
        sorted(SOURCES),
        include_dirs=["src"],
        cxx_std=17,
        optional=not REQUIRE_EXT,
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExt})
