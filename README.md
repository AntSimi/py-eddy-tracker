[![PyPI version](https://badge.fury.io/py/pyEddyTracker.svg)](https://badge.fury.io/py/pyEddyTracker) [![Documentation Status](https://readthedocs.org/projects/py-eddy-tracker/badge/?version=stable)](https://py-eddy-tracker.readthedocs.io/en/stable/?badge=stable) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AntSimi/py-eddy-tracker/master?urlpath=lab/tree/notebooks/python_module/)

# README #

### Method ###

Method was described in :

[Mason, E., A. Pascual, and J. C. McWilliams, 2014: A new sea surface height–based code for oceanic mesoscale eddy tracking.](https://doi.org/10.1175/JTECH-D-14-00019.1)

### Use case ###

Method is used in :
 
[Mason, E., A. Pascual, P. Gaube, S.Ruiz, J. Pelegrí, A. Delepoulle, 2017: Subregional characterization of mesoscale eddies across the Brazil-Malvinas Confluence](https://doi.org/10.1002/2016JC012611)

### How do I get set up? ###

To avoid problems with installation, use of the virtualenv Python virtual environment is recommended.

Then use pip to install all dependencies (numpy, scipy, matplotlib, netCDF4, ...), e.g.:

```bash
pip install numpy scipy netCDF4 matplotlib opencv-python pyyaml pint polygon3
```

Then run the following to install the eddy tracker:

```bash
python setup.py install
```
### Tools gallery ###
Several examples based on py eddy tracker module are [here](https://py-eddy-tracker.readthedocs.io/en/latest/python_module/index.html).

![](https://py-eddy-tracker.readthedocs.io/en/latest/_static/logo.png)

### Quick use ###

```bash
EddyId share/nrt_global_allsat_phy_l4_20190223_20190226.nc 20190223 adt ugos vgos longitude latitude ./ -v INFO
```

for identification, followed by:

```bash
EddyTracking tracking.yaml
```

for tracking (Edit the corresponding yaml files and then run the code).
