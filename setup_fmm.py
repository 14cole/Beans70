"""
Build the Cython near-field extension for the FMM module.

Usage:
    pip install cython numpy
    python setup_fmm.py build_ext --inplace

Or one-liner (no setup.py needed):
    cythonize -i fmm_near_cy.pyx
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "fmm_near_cy",
        sources=["fmm_near_cy.pyx"],
        include_dirs=[np.get_include()],
        libraries=["m"],            # libm for j0/y0/j1/y1
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="fmm_near_cy",
    ext_modules=cythonize(extensions, compiler_directives={
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
    }),
)
