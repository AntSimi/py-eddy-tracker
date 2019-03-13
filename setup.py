# -*- coding: utf-8 -*-
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext as cython_build_ext
import numpy

setup(
    name="pyEddyTracker",
    version='3.0.0',
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
        'src/scripts/EddyId',
        'src/scripts/EddyIdentification',
        'src/scripts/EddyTracking',
        'src/scripts/EddyFinalTracking',
        'src/scripts/EddyMergeCorrespondances',
        # 'src/scripts/EddyTrackingFull'
        ],
    zip_safe=False,
    cmdclass=dict(
        build_ext=cython_build_ext,
        ),
    entry_points=dict(console_scripts=[
        'EddyUpdate = py_eddy_tracker.update.__init__:eddy_update',
        ]),
    ext_modules=[Extension("py_eddy_tracker.tools",
                           ["src/py_eddy_tracker/tools.pyx"],
                           include_dirs=[numpy.get_include()])],
    package_data={'py_eddy_tracker.featured_tracking': ['*.nc', ]},
    setup_requires=[
        'numpy>=1.14'],
    install_requires=[
        'numpy>=1.14',
        # Bug with 1.5.1 (slow memory leak)
        'matplotlib>=2.0.0',
        'scipy>=0.15.1',
        'netCDF4>=1.1.0',
        'opencv-python',
        'shapely',
        'pyyaml',
        'pyproj',
        'pint',
        'numba',
        ],
    )
