"""
Eddy detection and filter
=========================

"""
from datetime import datetime
from matplotlib import pyplot as plt
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker import data
from py_eddy_tracker.generic import reverse_index
from numpy import arange


# %%
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
        plt.colorbar(mappable, cax=ax.figure.add_axes([0.95, 0.05, 0.01, 0.9]))


# %%
# Load Input grid, ADT will be used to detect eddies

g = RegularGridDataset(
    data.get_path("dt_med_allsat_phy_l4_20160515_20190101.nc"), "longitude", "latitude",
)
g.add_uv("adt")
g.copy("adt", "adt_high")
wavelength = 400
g.bessel_high_filter("adt_high", wavelength)
date = datetime(2016, 5, 15)

# %%
# Run algorithm of detection
a_f, c_f = g.eddy_identification("adt_high", "u", "v", date, 0.002)
merge_f = a_f.merge(c_f)
a_r, c_r = g.eddy_identification("adt", "u", "v", date, 0.002)
merge_r = a_r.merge(c_r)

# %%
# Display detection
ax = start_axes("Eddies detected over ADT")
m = g.display(ax, "adt", vmin=-0.15, vmax=0.15)
merge_f.display(ax, lw=0.5, label="Eddy from filtered grid", ref=-10, color="k")
merge_r.display(ax, lw=0.5, label="Eddy from raw grid", ref=-10, color="r")
ax.legend()
update_axes(ax, m)

# %%
# Parameters distribution
# -----------------------
fig = plt.figure(figsize=(12, 5))
ax_a = plt.subplot(121, xlabel="amplitdue(cm)")
ax_r = plt.subplot(122, xlabel="speed radius (km)")
ax_a.hist(
    merge_f["amplitude"] * 100,
    bins=arange(0.0005, 100, 1),
    label="Eddy from filtered grid",
    histtype="step",
)
ax_a.hist(
    merge_r["amplitude"] * 100,
    bins=arange(0.0005, 100, 1),
    label="Eddy from raw grid",
    histtype="step",
)
ax_a.set_xlim(0, 10)
ax_r.hist(merge_f["radius_s"] / 1000.0, bins=arange(0, 300, 5), histtype="step")
ax_r.hist(merge_r["radius_s"] / 1000.0, bins=arange(0, 300, 5), histtype="step")
ax_r.set_xlim(0, 100)
ax_a.legend()

# %%
# Match detection and compare
# ---------------------------

i, j, c = merge_f.match(merge_r)
m = c > 0.1
i_, j_ = i[m], j[m]


# %%
# where is lonely eddies
kwargs_f = dict(lw=1.5, label="Lonely eddy from filtered grid", ref=-10, color="k")
kwargs_r = dict(lw=1.5, label="Lonely eddy from raw grid", ref=-10, color="r")
ax = start_axes("Eddies with no match, over filtered ADT")
mappable = g.display(ax, "adt_high", vmin=-0.15, vmax=0.15)
merge_f.index(reverse_index(i_, len(merge_f))).display(ax, **kwargs_f)
merge_r.index(reverse_index(j_, len(merge_r))).display(ax, **kwargs_r)
ax.legend()
update_axes(ax, mappable)

ax = start_axes("Eddies with no match, over filtered ADT (zoom)")
ax.set_xlim(25, 36), ax.set_ylim(31, 35.25)
mappable = g.display(ax, "adt_high", vmin=-0.15, vmax=0.15)
u, v = g.grid("u").T, g.grid("v").T
ax.quiver(g.x_c, g.y_c, u, v, scale=10, pivot="mid", color="gray")
merge_f.index(reverse_index(i_, len(merge_f))).display(ax, **kwargs_f)
merge_r.index(reverse_index(j_, len(merge_r))).display(ax, **kwargs_r)
ax.legend()
update_axes(ax, mappable)

# %%
fig = plt.figure(figsize=(12, 12))
fig.suptitle(f"Scatter plot ({m.sum()} matches)")

for i, (label, field, factor, stop) in enumerate(
    (
        ("speed radius (km)", "radius_s", 0.001, 80),
        ("outter radius (km)", "radius_e", 0.001, 120),
        ("amplitude (cm)", "amplitude", 100, 25),
        ("speed max (cm/s)", "speed_average", 100, 25),
    )
):
    ax = fig.add_subplot(
        2, 2, i + 1, xlabel="filtered grid", ylabel="raw grid", title=label
    )
    ax.plot(merge_f[field][i_] * factor, merge_r[field][j_] * factor, ".")
    ax.set_aspect("equal"), ax.grid()
    ax.plot((0, 1000), (0, 1000), "r")
    ax.set_xlim(0, stop), ax.set_ylim(0, stop)
