from setuptools import setup
from Cython.Build import cythonize


'''This file is used to compile *.pyx files (cython),
    with normal python code (basically code in a normal *), 
    to create a .c file'''

'''to compile the files, run in the terminal, in the directory of this file (--build-lib build is optional to put the linking files in a separate folder):
   
   python cython_setup.py  build_ext --build-lib build 
   
   '''


setup(
    ext_modules = cythonize("grid_functions/grid_functions_3D_Cython.pyx")
)

