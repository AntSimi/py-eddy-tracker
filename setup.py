# -*- coding: utf-8 -*-
import versioneer
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = fh.read().split("\n")

setup(
    name="pyEddyTracker",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Py-Eddy-Tracker libraries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python",
    ],
    keywords="eddy science, eddy tracking, eddy tracker",
    author="emason & adelepoulle",
    author_email="emason@imedea.uib-csic.es",
    packages=find_packages("src"),
    package_dir={"": "src"},
    scripts=[
        "src/scripts/EddySubSetter",
        "src/scripts/EddyTranslate",
        "src/scripts/EddyTracking",
        "src/scripts/EddyFinalTracking",
        "src/scripts/EddyMergeCorrespondances",
        "src/scripts/GUIEddy",
    ],
    zip_safe=False,
    entry_points=dict(
        console_scripts=[
            # grid
            "GridFiltering = py_eddy_tracker.appli.grid:grid_filtering",
            "EddyId = py_eddy_tracker.appli.grid:eddy_id",
            # eddies
            "MergeEddies = py_eddy_tracker.appli.eddies:merge_eddies",
            # network
            "EddyNetworkGroup = py_eddy_tracker.appli.network:build_network",
            "EddyNetworkBuildPath = py_eddy_tracker.appli.network:divide_network",
            # anim/gui
            "EddyAnim = py_eddy_tracker.appli.gui:anim",
            # misc
            "ZarrDump = py_eddy_tracker.appli.misc:zarrdump",
        ]
    ),
    package_data={
        "py_eddy_tracker.featured_tracking": ["*.nc"],
        "py_eddy_tracker": ["data/*.nc"],
    },
    install_requires=requirements,
)
