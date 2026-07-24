import os
from setuptools import setup, Extension
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

ext = Extension(
    name='binding',
    sources=[os.path.join(here, 'binding.c')],
    depends=[
        os.path.join(here, 'radarxs.h'),
        os.path.normpath(os.path.join(here, '..', 'env_binding.h')),
    ],
    include_dirs=[np.get_include(), here, os.path.normpath(os.path.join(here, '..'))],
    define_macros=[('NO_RAYLIB', '1')],
)

setup(
    name='radarxs_binding',
    ext_modules=[ext],
)
