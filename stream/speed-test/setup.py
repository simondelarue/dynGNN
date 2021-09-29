# -*- coding: utf-8 -*-

from setuptools import setup
from Cython.Build import cythonize

'''setup(
    ext_modules = cythonize(
        ["vecmul_cy.pyx",
         "vecmul_py.py"],
        annotate=True)
)'''

setup(
    ext_modules = cythonize(
        'mydict.pyx',
        annotate=True
    )
)

# Compile with command line below : 
# python3 setup.py build_ext --inplace