# Copyright (c) 2026 Shayan Majumder <shayan.majumder2@gmail.com>
# SPDX-License-Identifier: MIT
"""Builds lrfhss._viterbi_ext."""
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

FLAGS = ["-O3", "-funroll-loops"]
if os.environ.get("NATIVE", "0") == "1":
    FLAGS.append("-march=native")

SOURCES = [
    "src/crc.cpp",
    "src/deinterleave.cpp",
    "src/demod.cpp",
    "src/filters.cpp",
    "src/module.cpp",
    "src/viterbi.cpp",
    "src/viterbi_dp.cpp",
]

setup(
    ext_modules=[
        Pybind11Extension(
            "lrfhss._viterbi_ext",
            sorted(SOURCES),
            include_dirs=["src"],
            cxx_std=17,
            extra_compile_args=FLAGS,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
