"""
Count pixel used
================

Do Geo stat with frequency and compare with center count
method: :ref:`sphx_glr_python_module_10_tracking_diagnostics_pet_center_count.py`
"""
import py_eddy_tracker_sample
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# %%
# Load an experimental med atlas over a period of 26 years (1993-2019)
a = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_demo_path(
        "eddies_med_adt_allsat_dt2018/Anticyclonic.zarr"
    )
)
c = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_demo_path("eddies_med_adt_allsat_dt2018/Cyclonic.zarr")
)

# %%
# Parameters
step = 0.125
bins = ((-10, 37, step), (30, 46, step))
kwargs_pcolormesh = dict(
    cmap="terrain_r", vmin=0, vmax=0.75, factor=1 / a.nb_days, name="count"
)


# %%
# Plot
fig = plt.figure(figsize=(12, 18.5))
ax_a = fig.add_axes([0.03, 0.75, 0.90, 0.25])
ax_a.set_title("Anticyclonic frequency")
ax_c = fig.add_axes([0.03, 0.5, 0.90, 0.25])
ax_c.set_title("Cyclonic frequency")
ax_all = fig.add_axes([0.03, 0.25, 0.90, 0.25])
ax_all.set_title("All eddies frequency")
ax_ratio = fig.add_axes([0.03, 0.0, 0.90, 0.25])
ax_ratio.set_title("Ratio cyclonic / Anticyclonic")

# Count pixel used for each contour
g_a = a.grid_count(bins, intern=True)
g_a.display(ax_a, **kwargs_pcolormesh)
g_c = c.grid_count(bins, intern=True)
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
plt.colorbar(m, cax=fig.add_axes([0.95, 0.27, 0.01, 0.7]))

g_c.vars["count"] = ratio
m = g_c.display(
    ax_ratio, name="count", norm=LogNorm(vmin=0.1, vmax=10), cmap="coolwarm_r"
)
plt.colorbar(m, cax=fig.add_axes([0.95, 0.02, 0.01, 0.2]))

for ax in (ax_a, ax_c, ax_all, ax_ratio):
    ax.set_aspect("equal")
    ax.set_xlim(-6, 36.5), ax.set_ylim(30, 46)
    ax.grid()

# %%
# Count Anticyclones as a function of lifetime
# --------------------------------------------
fig = plt.figure(figsize=(12, 10))
mask = a.lifetime >= 60
ax_long = fig.add_axes([0.03, 0.53, 0.90, 0.45])
g_a = a.grid_count(bins, intern=True, filter=mask)
g_a.display(ax_long, **kwargs_pcolormesh)
ax_long.set_title(f"Anticyclones with lifetime >= 60 days ({mask.sum()} Obs)")
ax_short = fig.add_axes([0.03, 0.03, 0.90, 0.45])
g_a = a.grid_count(bins, intern=True, filter=~mask)
m = g_a.display(ax_short, **kwargs_pcolormesh)
ax_short.set_title(f"Anticyclones with lifetime < 60 days ({(~mask).sum()} Obs)")
for ax in (ax_short, ax_long):
    ax.set_aspect("equal"), ax.grid()
    ax.set_xlim(-6, 36.5), ax.set_ylim(30, 46)
cb = plt.colorbar(m, cax=fig.add_axes([0.94, 0.05, 0.015, 0.9]))
