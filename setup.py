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
    author="emason",
    author_email="emason@imedea.uib-csic.es",
    packages=find_packages("src"),
    package_dir={"": "src"},
    scripts=[
        "src/scripts/GridFiltering",
        "src/scripts/EddyId",
        "src/scripts/EddySubSetter",
        "src/scripts/EddyTranslate",
        "src/scripts/EddyTracking",
        "src/scripts/EddyFinalTracking",
        "src/scripts/EddyMergeCorrespondances",
        "src/scripts/ZarrDump",
        "src/scripts/GUIEddy",
    ],
    zip_safe=False,
    entry_points=dict(
        console_scripts=[
            "MergeEddies = py_eddy_tracker.appli:merge_eddies",
            "EddyNetworkGroup = py_eddy_tracker.network:build_network",
            "EddyAnim = py_eddy_tracker.appli:anim"
        ]
    ),
    package_data={
        "py_eddy_tracker.featured_tracking": ["*.nc"],
        "py_eddy_tracker": ["data/*.nc"],
    },
    install_requires=requirements,
)
