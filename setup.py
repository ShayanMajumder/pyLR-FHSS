"""Builds lrfhss._viterbi_ext.

The metadata lives in pyproject.toml; this file exists only because the
extension needs pybind11's include path resolved at build time.

Portable flags by default. NATIVE=1 adds -march=native, which is NOT
uniformly faster here and was measured, not assumed: on the scalar IIR
recurrence in sos_forward it was ~60-70% SLOWER than plain -O3 (0.29 ms
vs 0.17-0.21 ms, repeatable) because AVX-512/AVX2 codegen does not help a
loop that cannot use wide vectors, while on the auto-vectorised
elementwise paths it was a small real win (~2%). It also ties the binary
to the building machine's ISA, so a NATIVE=1 build will die with SIGILL
on a CPU without those instructions. Benchmark your own workload before
choosing it.
"""
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
