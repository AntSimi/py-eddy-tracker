"""
Grid filtering in PET
=====================

How filter work in py eddy tracker. This implementation maybe doesn't respect state art, but ...

We code a specific filter in order to filter grid with same wavelength at each pixel.
"""

from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker import data
from matplotlib import pyplot as plt


def start_axes(title):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_axes([0.03, 0.03, 0.90, 0.94])
    ax.set_xlim(-6, 36.5), ax.set_ylim(30, 46)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def update_axes(ax, mappable=None):
    ax.grid()
    if mappable:
        plt.colorbar(m, cax=ax.figure.add_axes([0.95, 0.05, 0.01, 0.9]))


# %%
# All information will be for regular grid
g = RegularGridDataset(
    data.get_path("dt_med_allsat_phy_l4_20160515_20190101.nc"), "longitude", "latitude",
)
# %%
# Kernel
# ------
# Shape of kernel will increase in x when latitude increase
fig = plt.figure(figsize=(12, 8))
for i, latitude in enumerate((15, 35, 55, 75)):
    k = g.kernel_bessel(latitude, 500, order=3).T
    ax0 = plt.subplot(
        2,
        2,
        i + 1,
        title=f"Kernel at {latitude}° of latitude\nfor 1/8° grid, shape : {k.shape}",
        aspect="equal",
    )
    m = ax0.pcolormesh(k, vmin=-0.5, vmax=2, cmap="viridis_r")
plt.colorbar(m, cax=fig.add_axes((0.92, 0.05, 0.01, 0.9)))

# %%
# Kernel applying
# ---------------
# Original grid
ax = start_axes("ADT")
m = g.display(ax, "adt", vmin=-0.15, vmax=0.15)
update_axes(ax, m)

# %%
# We will select wavelength of 300 km
#
# Low frequency
ax = start_axes("ADT low frequency")
g.copy("adt", "adt_low")
g.bessel_low_filter("adt_low", 300, order=3)
m = g.display(ax, "adt_low", vmin=-0.15, vmax=0.15)
update_axes(ax, m)

# %%
# High frequency
ax = start_axes("ADT high frequency")
g.copy("adt", "adt_high")
g.bessel_high_filter("adt_high", 300, order=3)
m = g.display(ax, "adt_high", vmin=-0.15, vmax=0.15)
update_axes(ax, m)

# %%
# Clues
# -----
# wavelength : 80km

g.copy("adt", "adt_high_80")
g.bessel_high_filter("adt_high_80", 80, order=3)
g.copy("adt", "adt_low_80")
g.bessel_low_filter("adt_low_80", 80, order=3)

area = dict(llcrnrlon=11.75, urcrnrlon=21, llcrnrlat=33, urcrnrlat=36.75)

# %%
# Spectrum
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Spectrum")
ax.set_xlabel("km")

lon_spec, lat_spec = g.spectrum_lonlat("adt", area=area)
mappable = ax.loglog(*lat_spec, label="lat raw")[0]
ax.loglog(*lon_spec, label="lon raw", color=mappable.get_color(), linestyle="--")

lon_spec, lat_spec = g.spectrum_lonlat("adt_high_80", area=area)
mappable = ax.loglog(*lat_spec, label="lat high")[0]
ax.loglog(*lon_spec, label="lon high", color=mappable.get_color(), linestyle="--")

lon_spec, lat_spec = g.spectrum_lonlat("adt_low_80", area=area)
mappable = ax.loglog(*lat_spec, label="lat low")[0]
ax.loglog(*lon_spec, label="lon low", color=mappable.get_color(), linestyle="--")

ax.set_xlim(10, 1000)
ax.set_ylim(1e-6, 1)
ax.set_xscale("log")
ax.legend()
ax.grid()

# %%
# Spectrum ratio
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.set_title("Spectrum ratio")
ax.set_xlabel("km")

lon_spec, lat_spec = g.spectrum_lonlat(
    "adt_high_80", area=area, ref=g, ref_grid_name="adt"
)
mappable = ax.plot(*lat_spec, label="lat high")[0]
ax.plot(*lon_spec, label="lon high", color=mappable.get_color(), linestyle="--")

lon_spec, lat_spec = g.spectrum_lonlat(
    "adt_low_80", area=area, ref=g, ref_grid_name="adt"
)
mappable = ax.plot(*lat_spec, label="lat low")[0]
ax.plot(*lon_spec, label="lon low", color=mappable.get_color(), linestyle="--")

ax.set_xlim(10, 1000)
ax.set_ylim(0, 1)
ax.set_xscale("log")
ax.legend()
ax.grid()

# %%
# Old filter
# ----------
# To do ...

