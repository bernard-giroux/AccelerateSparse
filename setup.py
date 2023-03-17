from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [Extension("pyAccelerate.sparse_solve",
                        sources=["pyAccelerate/sparse_solve.pyx"],
                        # libraries=["Accelerate"],
                        include_dirs=[np.get_include()], #"/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers"],
                        extra_compile_args = ['-framework','Accelerate'],
                        extra_link_args = ['-Wl,-framework','-Wl,Accelerate'],
                        language="c++")]

setup(
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[np.get_include()]
)

