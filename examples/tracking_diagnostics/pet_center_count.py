"""
Count center
======================

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
# Plot
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111)
ax.set_xlim(-5, 37)
ax.set_ylim(30, 46)
step = 0.1
t0, t1 = a.period
g = a.grid_count(((-5, 37, step), (30, 46, step)), center=True)
g.vars["count"] = g.vars["count"] / (step ** 2 * (t1 - t0))
m = g.display(ax, name="count", vmin=0, vmax=2)
ax.grid()
cb = plt.colorbar(m, cax=fig.add_axes([0.95, 0.05, 0.01, 0.9]))
cb.set_label("Eddies by 1°^2 by day")
