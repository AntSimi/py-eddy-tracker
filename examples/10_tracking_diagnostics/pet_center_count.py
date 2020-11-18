"""
Count center
============

"""
import py_eddy_tracker_sample
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# %%
# Load an experimental med atlas over a period of 26 years (1993-2019)
a = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Anticyclonic.zarr")
)
c = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Cyclonic.zarr")
)

# Parameters
t0, t1 = a.period
step = 0.125
bins = ((-10, 37, step), (30, 46, step))
kwargs_pcolormesh = dict(
    cmap="terrain_r", vmin=0, vmax=2, factor=1 / (step ** 2 * (t1 - t0)), name="count"
)


# %%
# Plot
fig = plt.figure(figsize=(12, 18.5))
ax_a = fig.add_axes([0.03, 0.75, 0.90, 0.25])
ax_a.set_title("Anticyclonic center frequency")
ax_c = fig.add_axes([0.03, 0.5, 0.90, 0.25])
ax_c.set_title("Cyclonic center frequency")
ax_all = fig.add_axes([0.03, 0.25, 0.90, 0.25])
ax_all.set_title("All eddies center frequency")
ax_ratio = fig.add_axes([0.03, 0.0, 0.90, 0.25])
ax_ratio.set_title("Ratio cyclonic / Anticyclonic")

# Count pixel used for each center
g_a = a.grid_count(bins, intern=True, center=True)
g_a.display(ax_a, **kwargs_pcolormesh)
g_c = c.grid_count(bins, intern=True, center=True)
g_c.display(ax_c, **kwargs_pcolormesh)
# Compute a ratio Cyclonic / Anticyclonic
ratio = g_c.vars["count"] / g_a.vars["count"]

# Mask manipulation to be able to sum the 2 grids
m_c = g_c.vars["count"].mask
m = m_c & g_a.vars["count"].mask
g_c.vars["count"][m_c] = 0
g_c.vars["count"] += g_a.vars["count"]
g_c.vars["count"].mask = m

m = g_c.display(ax_all, **kwargs_pcolormesh)
cb = plt.colorbar(m, cax=fig.add_axes([0.94, 0.27, 0.01, 0.7]))
cb.set_label("Eddies by 1°^2 by day")

g_c.vars["count"] = ratio
m = g_c.display(
    ax_ratio, name="count", vmin=0.1, vmax=10, norm=LogNorm(), cmap="coolwarm_r"
)
plt.colorbar(m, cax=fig.add_axes([0.94, 0.02, 0.01, 0.2]))

for ax in (ax_a, ax_c, ax_all, ax_ratio):
    ax.set_aspect("equal")
    ax.set_xlim(-6, 36.5), ax.set_ylim(30, 46)
    ax.grid()

# %%
# Count Anticyclones as a function of lifetime
# --------------------------------------------
# Count at the center's position

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(211)
g_a = a.grid_count(bins, center=True, filter=a.lifetime >= 30)
m = g_a.display(ax, **kwargs_pcolormesh)
ax.set_aspect("equal")
ax.set_title("Anticyclones with lifetime >= 30 days")
cb = plt.colorbar(m, cax=fig.add_axes([0.94, 0.1, 0.015, 0.8]))
ax = fig.add_subplot(212)
g_a = a.grid_count(bins, center=True, filter=a.lifetime < 30)
m = g_a.display(ax, **kwargs_pcolormesh)
ax.set_aspect("equal")
ax.set_title("Anticyclones with lifetime < 30 days")
