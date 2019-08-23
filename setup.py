# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="pyEddyTracker",
    version='3.0.0',
    description="Py-Eddy-Tracker libraries",
    classifiers=['Development Status :: 3 - Alpha',
                 'Topic :: Scientific/Engineering :: Physics',
                 'Programming Language :: Python'],
    keywords='eddy science',
    author='emason',
    author_email='emason@imedea.uib-csic.es',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    scripts=[
        'src/scripts/GridFiltering',
        'src/scripts/EddyId',
        'src/scripts/EddyTracking',
        'src/scripts/EddyFinalTracking',
        'src/scripts/EddyMergeCorrespondances',
        ],
    zip_safe=False,
    entry_points=dict(console_scripts=[
        'EddyUpdate = py_eddy_tracker.update.__init__:eddy_update',
        ]),
    package_data={
        'py_eddy_tracker.featured_tracking': ['*.nc'],
        'py_eddy_tracker': ['data/*.nc'],
    },
    setup_requires=[
        'numpy>=1.14'],
    install_requires=[
        'numpy>=1.14',
        # Bug with 1.5.1 (slow memory leak)
        # 'matplotlib>=2.0.0',
        'scipy>=0.15.1',
        'netCDF4>=1.1.0',
        'opencv-python',
        'shapely',
        'pyyaml',
        'pyproj',
        'pint',
        'polygon3',
        'numba',
        ],
    )
