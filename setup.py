from setuptools import setup, Extension
import pybind11
from mpi4py import get_include as mpi4py_get_include
from mpi4py import get_config as mpi4py_get_config
import pkgconfig
import os
import os.path
import sys

thallium = pkgconfig.parse('thallium')

extra_compile_args=['-std=c++14']
if sys.platform == 'darwin':
    extra_compile_args.append('-mmacosx-version-min=10.9')

s4m_ext = Extension('_s4m',
        ['s4m/src/s4m.cpp'],
        libraries=thallium['libraries'] + ['stdc++', 'mpi'],
        library_dirs=thallium['library_dirs'],
        include_dirs=thallium['include_dirs'] + [pybind11.get_include(), mpi4py_get_include()],
        language='c++14',
        extra_compile_args=extra_compile_args,
        depends=[])

setup(name='s4m',
      version='0.1',
      author='Matthieu Dorier',
      description='''Mochi-based python library to allow processes to exchange data''',
      ext_modules=[s4m_ext],
      packages=['s4m']
)
