"""
Count pixel used
======================

"""

from matplotlib import pyplot as plt
from py_eddy_tracker.observations.tracking import TrackEddiesObservations
import py_eddy_tracker_sample


a = TrackEddiesObservations.load_file(py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Anticyclonic.zarr"))
c = TrackEddiesObservations.load_file(py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Cyclonic.zarr"))

# Plot
fig = plt.figure(figsize=(15, 20))
ax_a = fig.add_subplot(311)
ax_a.set_title('Anticyclonic frequency')
ax_c = fig.add_subplot(312)
ax_c.set_title('Cyclonic frequency')
ax_all = fig.add_subplot(313)
ax_all.set_title('All eddies frequency')

step = 0.1
kwargs_pcolormesh = dict(cmap='terrain_r', vmin=0, vmax=.75)
g_a = a.grid_count(((-5, 37, step), (30, 46, step)), intern=True)
t0, t1 = a.period
g_a.vars["count"] = g_a.vars["count"] / (t1 - t0)
m = g_a.display(ax_a, name="count", **kwargs_pcolormesh)

g_c = c.grid_count(((-5, 37, step), (30, 46, step)), intern=True)
t0, t1 = c.period
g_c.vars["count"] = g_c.vars["count"] / (t1 - t0)
m = g_c.display(ax_c, name="count", **kwargs_pcolormesh)

m_c = g_c.vars["count"].mask
m = m_c & g_a.vars["count"].mask
g_c.vars["count"][m_c] = 0
g_c.vars["count"] += g_a.vars["count"]
g_c.vars["count"].mask = m

m = g_c.display(ax_all, name="count", **kwargs_pcolormesh)

for ax in (ax_a, ax_c, ax_all):
    ax.set_aspect("equal")
    ax.set_xlim(-5, 37)
    ax.set_ylim(30, 46)
    ax.grid()

plt.colorbar(m, cax=fig.add_axes([0.95, 0.05, 0.01, 0.9]))
