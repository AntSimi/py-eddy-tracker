# README #

### Method ###

Method was described in :

[Mason, E., A. Pascual, and J. C. McWilliams, 2014: A new sea surface height–based code for oceanic mesoscale eddy tracking.](https://doi.org/10.1002/2016JC012611)

### Use case ###

Method is used in :
 
[Mason, E., A. Pascual, P. Gaube, S.Ruiz, J. Pelegrí, A. Delepoulle, 2017: Subregional characterization of mesoscale eddies across the Brazil-Malvinas Confluence](https://doi.org/10.1002/2016JC012611)

### How do I get set up? ###

To avoid problems with installation, use of the virtualenv Python virtual environment is recommended.

Then use pip to install all dependencies (numpy, scipy, matplotlib, netCDF4, cython, pyproj, Shapely, ...), e.g.:

```bash
pip install cython numpy matplotlib scipy netCDF4 shapely pyproj
```

Then run the following to install the eddy tracker:

```bash
python setup.py install
```

Two executables are now available in your PATH: EddyIdentification and EddyTracking

Edit the corresponding yaml files and then run the code, e.g.:

```bash
EddyIdentification eddy_identification.yaml
```

for identification, followed by:

```bash
EddyTracking tracking.yaml
```

for tracking.


