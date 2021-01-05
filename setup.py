# -*- coding: utf-8 -*-
import versioneer
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = fh.read().split("\n")

setup(
    name="pyEddyTracker",
    python_requires=">=3.7",
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
        "src/scripts/EddyFinalTracking",
        "src/scripts/EddyMergeCorrespondances",
    ],
    zip_safe=False,
    entry_points=dict(
        console_scripts=[
            # grid
            "GridFiltering = py_eddy_tracker.appli.grid:grid_filtering",
            "EddyId = py_eddy_tracker.appli.grid:eddy_id",
            # eddies
            "MergeEddies = py_eddy_tracker.appli.eddies:merge_eddies",
            "EddyFrequency = py_eddy_tracker.appli.eddies:get_frequency_grid",
            "EddyInfos = py_eddy_tracker.appli.eddies:display_infos",
            "EddyCircle = py_eddy_tracker.appli.eddies:eddies_add_circle",
            "EddyTracking = py_eddy_tracker.appli.eddies:eddies_tracking",
            "EddyQuickCompare = py_eddy_tracker.appli.eddies:quick_compare",
            # network
            "EddyNetworkGroup = py_eddy_tracker.appli.network:build_network",
            "EddyNetworkBuildPath = py_eddy_tracker.appli.network:divide_network",
            # anim/gui
            "EddyAnim = py_eddy_tracker.appli.gui:anim",
            "GUIEddy = py_eddy_tracker.appli.gui:guieddy",
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
