"""
Eddy detection Gulf stream
==========================

Script will detect eddies on adt field, and compute u,v with method add_uv(which could use, only if equator is avoid)

Figures will show different step to detect eddies.

"""
from datetime import datetime
from matplotlib import pyplot as plt
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker import data
from py_eddy_tracker.eddy_feature import Contours


# %%
def start_axes(title):
    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([0.03, 0.03, 0.90, 0.94])
    ax.set_xlim(279, 304), ax.set_ylim(29, 44)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def update_axes(ax, mappable=None):
    ax.grid()
    if mappable:
        plt.colorbar(mappable, cax=ax.figure.add_axes([0.95, 0.05, 0.01, 0.9]))


# %%
# Load Input grid, ADT will be used to detect eddies
margin = 30
g = RegularGridDataset(
    data.get_path("nrt_global_allsat_phy_l4_20190223_20190226.nc"),
    "longitude",
    "latitude",
    # Manual area subset
    indexs=dict(
        longitude=slice(1116 - margin, 1216 + margin),
        latitude=slice(476 - margin, 536 + margin),
    ),
)

ax = start_axes("ADT (m)")
m = g.display(ax, "adt", vmin=-0.15, vmax=1)
# Draw line on the gulf stream front
great_current = Contours(g.x_c, g.y_c, g.grid("adt"), levels=(0.35,), keep_unclose=True)
great_current.display(ax, color="k")
update_axes(ax, m)

# %%
# Get u/v
# -------
# U/V are deduced from ADT, this algortihm are not usable around equator (~+- 2°)
g.add_uv("adt")

# %%
# Pre-processings
# ---------------
# Apply high filter to remove long scale to highlight mesoscale
g.bessel_high_filter("adt", 700)
ax = start_axes("ADT (m) filtered (700km)")
m = g.display(ax, "adt", vmin=-0.25, vmax=0.25)
great_current.display(ax, color="k")
update_axes(ax, m)

# %%
# Identification
# --------------
# run identification with slice of 2 mm
date = datetime(2016, 5, 15)
a, c = g.eddy_identification("adt", "u", "v", date, 0.002)

# %%
# All closed contour found in this input grid (Display only 1 contour every 5)
ax = start_axes("ADT closed contour (only 1 / 5 levels)")
g.contours.display(ax, step=5, lw=1)
great_current.display(ax, color="k")
update_axes(ax)

# %%
# Contours include in eddies
ax = start_axes("ADT contour used as eddies")
g.contours.display(ax, only_used=True, lw=0.25)
great_current.display(ax, color="k")
update_axes(ax)

# %%
# Contours reject from several origin (shape error to high, several extremum in contour, ...)
ax = start_axes("ADT contour reject")
g.contours.display(ax, only_unused=True, lw=0.25)
great_current.display(ax, color="k")
update_axes(ax)

# %%
# Contours closed which contains several eddies
ax = start_axes("ADT contour reject but which contain eddies")
g.contours.label_contour_unused_which_contain_eddies(a)
g.contours.label_contour_unused_which_contain_eddies(c)
g.contours.display(
    ax, only_contain_eddies=True, color="k", lw=1, label="Could be interaction contour"
)
a.display(ax, color="r", linewidth=0.5, label="Anticyclonic", ref=-10)
c.display(ax, color="b", linewidth=0.5, label="Cyclonic", ref=-10)
ax.legend()
update_axes(ax)

# %%
# Output
# ------
# Display detected eddies, dashed lines represent effective contour
# and solid lines represent contour of maximum of speed. See figure 1 of https://doi.org/10.1175/JTECH-D-14-00019.1

ax = start_axes("Eddies detected")
a.display(ax, color="r", linewidth=0.5, label="Anticyclonic", ref=-10)
c.display(ax, color="b", linewidth=0.5, label="Cyclonic", ref=-10)
ax.legend()
great_current.display(ax, color="k")
update_axes(ax)


# %%
# Display speed radius of eddies detected
ax = start_axes("Eddies speed radius (km)")
a.filled(ax, "radius_e", vmin=10, vmax=150, cmap="magma_r", factor=0.001, lut=14)
m = c.filled(ax, "radius_e", vmin=10, vmax=150, cmap="magma_r", factor=0.001, lut=14)
great_current.display(ax, color="k")
update_axes(ax, m)
