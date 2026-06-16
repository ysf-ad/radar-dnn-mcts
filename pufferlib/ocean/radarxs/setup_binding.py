import os
import sys

import numpy as np
from setuptools import Extension, setup


here = os.path.dirname(os.path.abspath(__file__))
openmp_args = ["/openmp"] if sys.platform == "win32" else ["-fopenmp"]

ext = Extension(
    name="binding",
    sources=[os.path.join(here, "binding.c")],
    include_dirs=[np.get_include(), here, os.path.normpath(os.path.join(here, ".."))],
    define_macros=[("NO_RAYLIB", "1")],
    extra_compile_args=openmp_args,
    extra_link_args=[] if sys.platform == "win32" else openmp_args,
)

setup(
    name="radarxs_binding",
    ext_modules=[ext],
)
