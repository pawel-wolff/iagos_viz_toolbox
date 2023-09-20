#!/usr/bin/env python

from setuptools import setup

setup(
    name='iagos-viz-toolbox',
    version='0.1',
    description='Toolbox with visualization routines used in the IAGOS project',
    author='Pawel Wolff',
    author_email='pawel.wolff@aero.obs-mip.fr',
    packages=[
        'iagos_viz',
    ],
    install_requires=[
        'matplotlib',
        'cartopy',
        'numpy',
        'pandas',
        'xarray',
        'scipy',
        'netcdf4',
        'h5netcdf',
        'dask',
        'xarray-extras'
    ],
)
