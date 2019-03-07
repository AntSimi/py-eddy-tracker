# README #

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


# Py Eddy Tracker module #

### Grid manipulation ###

Loading grid
```python
from py_eddy_tracker.dataset.grid import RegularGridDataset
h = RegularGridDataset('share/nrt_global_allsat_phy_l4_20190223_20190226.nc', 'longitude', 'latitude')
```

Plotting grid
```python
fig = plt.figure(figsize=(14, 12))
ax = fig.add_axes([.02, .51, .9, .45])
ax.set_title('ADT (m)')
ax.set_ylim(-75, 75)
ax.set_aspect('equal')
m=ax.pcolormesh(h.x_bounds, h.y_bounds, h.grid('adt').T.copy(), vmin=-1, vmax=1, cmap='coolwarm')
ax.grid(True)
plt.colorbar(m, cax=fig.add_axes([.94, .51, .01, .45]))
```

Filtering
```python
h.bessel_high_filter('adt', 500, order=3)
```

Add second plot
```python
ax = fig.add_axes([.03, .02, .9, .45])
ax.set_title('ADT Filtered (m)')
ax.set_aspect('equal')
ax.set_ylim(-75, 75)
m=ax.pcolormesh(h.x_bounds, h.y_bounds, h.grid('adt').T, vmin=-.1, vmax=.1, cmap='coolwarm')
ax.grid(True)
plt.colorbar(m, cax=fig.add_axes([.94, .02, .01, .45]))
fig.savefig('share/png/filter.png')
```

![signal filtering](share/png/filter.png)

