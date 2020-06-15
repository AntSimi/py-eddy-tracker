"""
Display Tracks
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

a = a.extract_with_length((7 * 20, -1))
c = c.extract_with_length((7 * 20, -1))
a.median_filter(1, 'time', 'lon').loess_filter(5, 'time', 'lon')
a.median_filter(1, 'time', 'lat').loess_filter(5, 'time', 'lat')
c.median_filter(1, 'time', 'lon').loess_filter(5, 'time', 'lon')
c.median_filter(1, 'time', 'lat').loess_filter(5, 'time', 'lat')


# Plot
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
ax.set_aspect("equal")
ax.set_xlim(-5, 37)
ax.set_ylim(30, 46)
a.plot(ax, ref=-10, label="Anticyclonic", color="r", lw=0.1)
c.plot(ax, ref=-10, label="Cyclonic", color="b", lw=0.1)
ax.legend()
