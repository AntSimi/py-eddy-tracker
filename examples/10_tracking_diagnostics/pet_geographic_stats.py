"""
Geographical statistics
=======================

"""

from matplotlib import pyplot as plt
from py_eddy_tracker.observations.tracking import TrackEddiesObservations
import py_eddy_tracker_sample


a = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Anticyclonic.zarr")
)
c = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Cyclonic.zarr")
)
a = a.merge(c)

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)
ax.set_xlim(-5, 37)
ax.set_ylim(30, 46)
ax.set_aspect("equal")
step = 0.1
g = a.grid_stat(((-5, 37, step), (30, 46, step)), "amplitude")
m = g.display(ax, name="amplitude", vmin=0, vmax=0.2)
ax.grid()
cb = plt.colorbar(m, cax=fig.add_axes([0.92, 0.05, 0.01, 0.9]))
cb.set_label("Amplitude (m)")
ax.set_title("Amplitude mean by box of %s°" % step)

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)
ax.set_xlim(-5, 37)
ax.set_ylim(30, 46)
ax.set_aspect("equal")
step = 0.1
g = a.grid_stat(((-5, 37, step), (30, 46, step)), "radius_s")
m = g.display(ax, name="radius_s", vmin=10, vmax=50, factor=0.001)
ax.grid()
cb = plt.colorbar(m, cax=fig.add_axes([0.92, 0.05, 0.01, 0.9]))
cb.set_label("Speed radius (km)")
ax.set_title("Speed radius mean by box of %s°" % step)

fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(111)
ax.set_xlim(-5, 37)
ax.set_ylim(30, 46)
ax.set_aspect("equal")
step = 0.1
g = a.grid_stat(((-5, 37, step), (30, 46, step)), "virtual")
g.vars["virtual"] *= 100
m = g.display(ax, name="virtual", vmin=0, vmax=15)
ax.grid()
cb = plt.colorbar(m, cax=fig.add_axes([0.92, 0.05, 0.01, 0.9]))
cb.set_label("Percent of virtual (%)")
ax.set_title("Percent of virtual by box of %s°" % step)
