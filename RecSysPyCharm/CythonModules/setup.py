from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("/home/alle/GitHub/RecSysChallenge/RecSysPyCharm/CythonModules/MF_BPR_Cython_Epoch.pyx"),
    include_dirs=[numpy.get_include()]
)
