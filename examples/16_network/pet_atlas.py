"""
Network Analysis
================
"""
from matplotlib import pyplot as plt
from numpy import ma

import py_eddy_tracker.gui
from py_eddy_tracker.observations.network import NetworkObservations

n = NetworkObservations.load_file(
    "/data/adelepoulle/work/Eddies/20201217_network_build/tracking/med/Anticyclonic_seg.nc"
)

# %%
# Parameters
step = 1 / 10.0
bins = ((-10, 37, step), (30, 46, step))
kw_time = dict(cmap="terrain_r", factor=100.0 / n.nb_days, name="count")
kw_ratio = dict(cmap=plt.get_cmap("magma_r", 10))


# %%
# Functions
def start_axes(title):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_axes([0.03, 0.03, 0.90, 0.94], projection="full_axes")
    ax.set_xlim(-6, 36.5), ax.set_ylim(30, 46)
    ax.set_aspect("equal")
    ax.set_title(title, weight="bold")
    return ax


def update_axes(ax, mappable=None):
    ax.grid()
    ax.update_env()
    if mappable:
        return plt.colorbar(mappable, cax=ax.figure.add_axes([0.94, 0.05, 0.01, 0.9]))


# %%
# All
# ---
# Display the % of time each pixel (1/10°) is within an anticyclonic network
ax = start_axes("")
g_all = n.grid_count(bins)
m = g_all.display(ax, **kw_time, vmin=0, vmax=75)
update_axes(ax, m).set_label("Pixel used in % of time")

# %%
# Network longer than 10 days
# ---------------------------
# Display the % of time each pixel (1/10°) is within an anticyclonic network
# which total lifetime in longer than 10 days
ax = start_axes("")
n10 = n.longer_than(10)
g_10 = n10.grid_count(bins)
m = g_10.display(ax, **kw_time, vmin=0, vmax=75)
update_axes(ax, m).set_label("Pixel used in % of time")

# %%
# Display the ratio between the short and total presence.
#
# Light = mostly short networks
ax = start_axes("")
m = g_10.display(
    ax,
    **kw_ratio,
    vmin=50,
    vmax=100,
    name=g_10.vars["count"] * 100.0 / g_all.vars["count"]
)
update_axes(ax, m).set_label("Pixel used in % all atlas")
# %%
# Network longer than 20 days
# ---------------------------
# Display the % of time each pixel (1/10°) is within an anticyclonic network
# which total lifetime is longer than 20 days
ax = start_axes("")
n20 = n.longer_than(20)
g_20 = n20.grid_count(bins)
m = g_20.display(ax, **kw_time, vmin=0, vmax=75)
update_axes(ax, m).set_label("Pixel used in % of time")

# %%
# Display the ratio between the short and total presence.
#
# Light = mostly short networks
ax = start_axes("")
m = g_20.display(
    ax,
    **kw_ratio,
    vmin=50,
    vmax=100,
    name=g_20.vars["count"] * 100.0 / g_all.vars["count"]
)
update_axes(ax, m).set_label("Pixel used in % all atlas")

# %%
# Display the ratio between the short and total presence.
#
# Light = mostly short networks
#
# Networks shorter than 365 days are masked
ax = start_axes("")
m = g_20.display(
    ax,
    **kw_ratio,
    vmin=50,
    vmax=100,
    name=ma.array(
        g_20.vars["count"] * 100.0 / g_all.vars["count"], mask=g_all.vars["count"] < 365
    )
)
update_axes(ax, m).set_label("Pixel used in % all atlas")
# %%
# Display the ratio between the short and total presence.
#
# Networks longer than 365 days are masked
#
# # -> Coastal areas are mostly populated by short networks
ax = start_axes("")
m = g_20.display(
    ax,
    **kw_ratio,
    vmin=50,
    vmax=100,
    name=ma.array(
        g_20.vars["count"] * 100.0 / g_all.vars["count"],
        mask=g_all.vars["count"] >= 365,
    )
)
update_axes(ax, m).set_label("Pixel used in % all atlas")


# %%
# All merging
# -----------
# Display the occurence of merging events
ax = start_axes("")
g_all_merging = n.merging_event().grid_count(bins)
m = g_all_merging.display(ax, **kw_time, vmin=0, vmax=1)
update_axes(ax, m).set_label("Pixel used in % of time")

# %%
# Ratio merging events / eddy presence
ax = start_axes("")
m = g_all_merging.display(
    ax,
    **kw_ratio,
    vmin=0,
    vmax=1,
    name=g_all_merging.vars["count"] * 100.0 / g_all.vars["count"]
)
update_axes(ax, m).set_label("Pixel used in % all atlas")

# %%
# Merging in networks longer than 10 days
# ---------------------------------------
ax = start_axes("")
g_10_merging = n10.merging_event().grid_count(bins)
m = g_10_merging.display(ax, **kw_time, vmin=0, vmax=1)
update_axes(ax, m).set_label("Pixel used in % of time")
# %%
ax = start_axes("")
m = g_10_merging.display(
    ax,
    **kw_ratio,
    vmin=0,
    vmax=2,
    name=ma.array(
        g_10_merging.vars["count"] * 100.0 / g_10.vars["count"],
        mask=g_10.vars["count"] < 365,
    )
)
update_axes(ax, m).set_label("Pixel used in % all atlas")
