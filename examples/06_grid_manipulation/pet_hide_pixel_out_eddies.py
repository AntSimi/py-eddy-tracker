"""
Select pixel in eddies
======================

"""

from matplotlib import pyplot as plt
from matplotlib.path import Path
from numpy import ones

from py_eddy_tracker import data
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker.observations.observation import EddiesObservations
from py_eddy_tracker.poly import create_vertice

# %%
# Load an eddy file which contains contours
a = EddiesObservations.load_file(data.get_path("Anticyclonic_20190223.nc"))

# %%
# Load a grid where we want found pixels in eddies or out
g = RegularGridDataset(
    data.get_path("nrt_global_allsat_phy_l4_20190223_20190226.nc"),
    "longitude",
    "latitude",
)

# %%
# For each contours, we will get pixels indice in contour.
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes((0.05, 0.05, 0.9, 0.9))
ax.set_aspect("equal")
ax.set_xlim(10, 70)
ax.set_ylim(-50, -25)
# We will used the outter contour
x_name, y_name = a.intern(False)
adt = g.grid("adt")
mask = ones(adt.shape, dtype="bool")
for eddy in a:
    i, j = Path(create_vertice(eddy[x_name], eddy[y_name])).pixels_in(g)
    mask[i, j] = False
adt.mask[:] += ~mask
g.display(ax, "adt")
a.display(ax, label="Anticyclonic", color="g", lw=1, extern_only=True)
# %%
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes((0.05, 0.05, 0.9, 0.9))
ax.set_aspect("equal")
ax.set_xlim(10, 70)
ax.set_ylim(-50, -25)
adt.mask[:] = mask
g.display(ax, "adt")
a.display(ax, label="Anticyclonic", color="g", lw=1, extern_only=True)
