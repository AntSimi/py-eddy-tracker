"""
Eddy detection
==============

Script will detect eddies on adt field, and compute u,v with method add_uv(which could use, only if equator is avoid)

"""
from datetime import datetime
from matplotlib import pyplot as plt
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker import data


def start_axes(title):
    fig = plt.figure(figsize=(12.5, 5))
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(-5, 37)
    ax.set_ylim(30, 46)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def update_axes(ax, mappable=None):
    ax.grid()
    if mappable:
        plt.colorbar(m, cax=ax.figure.add_axes([0.95, 0.05, 0.01, 0.9]))


g = RegularGridDataset(
    data.get_path("dt_med_allsat_phy_l4_20160515_20190101.nc"), "longitude", "latitude"
)

ax = start_axes("ADT(m)")
m = g.display(ax, "adt", vmin=-0.15, vmax=0.15)
update_axes(ax, m)

g.add_uv("adt")
g.bessel_high_filter("adt", 500, order=2)

ax = start_axes("ADT (m) filtered(500km, order 2)")
m = g.display(ax, "adt", vmin=-0.15, vmax=0.15)
update_axes(ax, m)

date = datetime(2016, 5, 15)
a, c = g.eddy_identification("adt", "u", "v", date, 0.002)

ax = start_axes("ADT closed contour (only 1 / 4 levels)")
g.contours.display(ax, step=4)
update_axes(ax)

ax = start_axes("ADT contour used as eddies")
g.contours.display(ax, only_used=True)
update_axes(ax)

ax = start_axes("ADT contour reject")
g.contours.display(ax, only_unused=True)
update_axes(ax)

ax = start_axes("ADT contour reject but which contain eddies")
g.contours.label_contour_unused_which_contain_eddies(a)
g.contours.label_contour_unused_which_contain_eddies(c)
g.contours.display(ax, only_contain_eddies=True, color="k", lw=1, label="Could be interaction contour")
a.display(ax, color="b", linewidth=0.5, label="Anticyclonic", ref=-10)
c.display(ax, color="r", linewidth=0.5, label="Cyclonic", ref=-10)
ax.legend()
update_axes(ax)

ax = start_axes("Eddies detected")
a.display(ax, color="b", linewidth=0.5, label="Anticyclonic", ref=-10)
c.display(ax, color="r", linewidth=0.5, label="Cyclonic", ref=-10)
ax.legend()
update_axes(ax)

ax = start_axes("Eddies speed radius (km)")
a.scatter(ax, "radius_s", vmin=10, vmax=50, s=80, ref=-10, cmap='jet', factor=0.001)
m = c.scatter(ax, "radius_s", vmin=10, vmax=50, s=80, ref=-10, cmap='jet', factor=0.001)
update_axes(ax, m)
