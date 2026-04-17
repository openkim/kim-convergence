"""Setuptools based setup module."""

from setuptools import setup, Extension

import versioneer


class get_numpy_include:
    """Defer numpy import until build time."""
    def __str__(self):
        import numpy
        return numpy.get_include()


# Single-threaded FFT extension (optional - build fails gracefully)
ext_modules = [
    Extension(
        'kim_convergence.stats._stfft._stfft_core',
        sources=['kim_convergence/stats/_stfft/stfft_wrapper.cpp'],
        include_dirs=[
            get_numpy_include(),
            'kim_convergence/stats/_stfft',
        ],
        extra_compile_args=[
            '-std=c++11',
            '-O3',
            '-fno-openmp',
        ],
        extra_link_args=['-fno-openmp'],
        language='c++',
    )
]


setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=ext_modules,
)
