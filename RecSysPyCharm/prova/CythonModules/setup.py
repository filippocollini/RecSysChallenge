from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os

setup(
    ext_modules=cythonize([os.path.dirname(os.path.realpath(__file__)) + "/*.pyx"]),
    include_dirs=[numpy.get_include()]
)
