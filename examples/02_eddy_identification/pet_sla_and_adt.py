"""
Eddy detection on SLA and ADT
=============================

"""
from datetime import datetime
from matplotlib import pyplot as plt
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker import data


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
g.add_uv("adt", "ugos", "vgos")
g.add_uv("sla", "ugosa", "vgosa")
wavelength = 400
g.copy("adt", "adt_raw")
g.copy("sla", "sla_raw")
g.bessel_high_filter("adt", wavelength)
g.bessel_high_filter("sla", wavelength)
date = datetime(2016, 5, 15)

# %%
kwargs_a_adt = dict(lw=0.5, label="Anticyclonic ADT", ref=-10, color="k")
kwargs_c_adt = dict(lw=0.5, label="Cyclonic ADT", ref=-10, color="r")
kwargs_a_sla = dict(lw=0.5, label="Anticyclonic SLA", ref=-10, color="g")
kwargs_c_sla = dict(lw=0.5, label="Cyclonic SLA", ref=-10, color="b")

# %%
# Run algorithm of detection
a_adt, c_adt = g.eddy_identification("adt", "ugos", "vgos", date, 0.002)
a_sla, c_sla = g.eddy_identification("sla", "ugosa", "vgosa", date, 0.002)

# %%
# over filtered
ax = start_axes(f"ADT (m) filtered ({wavelength}km)")
m = g.display(ax, "adt", vmin=-0.15, vmax=0.15)
a_adt.display(ax, **kwargs_a_adt), c_adt.display(ax, **kwargs_c_adt)
ax.legend(), update_axes(ax, m)

ax = start_axes(f"SLA (m) filtered ({wavelength}km)")
m = g.display(ax, "sla", vmin=-0.15, vmax=0.15)
a_sla.display(ax, **kwargs_a_sla), c_sla.display(ax, **kwargs_c_sla)
ax.legend(), update_axes(ax, m)

# %%
# over raw
ax = start_axes("ADT (m)")
m = g.display(ax, "adt_raw", vmin=-0.15, vmax=0.15)
a_adt.display(ax, **kwargs_a_adt), c_adt.display(ax, **kwargs_c_adt)
ax.legend(), update_axes(ax, m)

ax = start_axes("SLA (m)")
m = g.display(ax, "sla_raw", vmin=-0.15, vmax=0.15)
a_sla.display(ax, **kwargs_a_sla), c_sla.display(ax, **kwargs_c_sla)
ax.legend(), update_axes(ax, m)

# %%
# Display detection
ax = start_axes("Eddies detected")
a_adt.display(ax, **kwargs_a_adt)
a_sla.display(ax, **kwargs_a_sla)
c_adt.display(ax, **kwargs_c_adt)
c_sla.display(ax, **kwargs_c_sla)
ax.legend()
update_axes(ax)

# %%
# Match
# -----------------------
# Where cyclone meet anticyclone
i_c_adt, i_a_sla, c = c_adt.match(a_sla, cmin=0.01)
i_a_adt, i_c_sla, c = a_adt.match(c_sla, cmin=0.01)

ax = start_axes("Cyclone share area with anticyclone")
a_adt.index(i_a_adt).display(ax, **kwargs_a_adt)
c_adt.index(i_c_adt).display(ax, **kwargs_c_adt)
a_sla.index(i_a_sla).display(ax, **kwargs_a_sla)
c_sla.index(i_c_sla).display(ax, **kwargs_c_sla)
ax.legend()
update_axes(ax)


# %%
# Scatter plot
# ------------
i_a_adt, i_a_sla, c = a_adt.match(a_sla, cmin=0.1)
i_c_adt, i_c_sla, c = c_adt.match(c_sla, cmin=0.1)

# %%
# where is lonely eddies
ax = start_axes("Eddies with no match")
a_adt.index(i_a_adt, reverse=True).display(ax, **kwargs_a_adt)
c_adt.index(i_c_adt, reverse=True).display(ax, **kwargs_c_adt)
a_sla.index(i_a_sla, reverse=True).display(ax, **kwargs_a_sla)
c_sla.index(i_c_sla, reverse=True).display(ax, **kwargs_c_sla)
ax.legend()
update_axes(ax)

# %%
fig = plt.figure(figsize=(12, 12))
fig.suptitle(f"Scatter plot (A : {i_a_adt.shape[0]}, C : {i_c_adt.shape[0]} matches)")

for i, (label, field, factor, stop) in enumerate(
    (
        ("speed radius (km)", "radius_s", 0.001, 80),
        ("outter radius (km)", "radius_e", 0.001, 120),
        ("amplitude (cm)", "amplitude", 100, 25),
        ("speed max (cm/s)", "speed_average", 100, 25),
    )
):
    ax = fig.add_subplot(2, 2, i + 1, title=label)
    ax.set_xlabel("Absolute Dynamic Topography")
    ax.set_ylabel("Sea Level Anomaly")

    ax.plot(
        a_adt[field][i_a_adt] * factor,
        a_sla[field][i_a_sla] * factor,
        "r.",
        label="Anticyclonic",
    )
    ax.plot(
        c_adt[field][i_c_adt] * factor,
        c_sla[field][i_c_sla] * factor,
        "b.",
        label="Cyclonic",
    )
    ax.set_aspect("equal"), ax.grid()
    ax.plot((0, 1000), (0, 1000), "g")
    ax.set_xlim(0, stop), ax.set_ylim(0, stop)
    ax.legend()
