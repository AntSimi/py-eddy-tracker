# -*- coding: utf-8 -*-
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext as cython_build_ext
import numpy

setup(
    name="pyEddyTracker",
    version='2.0.3',
    description="Py-Eddy-Tracker libraries",
    classifiers=['Development Status :: Alpha',
                 'Topic :: Eddy',
                 'Programming Language :: Python'],
    keywords='eddy science',
    author='emason',
    author_email='emason@imedea.uib-csic.es',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    scripts=[
        'src/scripts/EddyIdentification',
        'src/scripts/EddyTracking',
        'src/scripts/EddyFinalTracking',
        'src/scripts/EddyMergeCorrespondances',
        # 'src/scripts/EddyTrackingFull'
        ],
    zip_safe=False,
    cmdclass={
        'build_ext': cython_build_ext,
    },
    ext_modules=[Extension("py_eddy_tracker.tools",
                           ["src/py_eddy_tracker/tools.pyx"],
                           include_dirs=[numpy.get_include()])],
    setup_requires=[
        'numpy>=1.8'],
    install_requires=[
        'numpy>=1.9',
        # Bug with 1.5.1 (slow memory leak)
        'matplotlib==1.4.3',
        'scipy>=0.15.1',
        'netCDF4>=1.1.0',
        'shapely',
        'pyyaml',
        'pyproj',
        ],
    )
