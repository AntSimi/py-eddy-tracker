"""
One Track
===================

"""
from matplotlib import pyplot as plt
from py_eddy_tracker.observations.tracking import TrackEddiesObservations
import py_eddy_tracker_sample

a = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Anticyclonic.zarr")
)

eddy = a.extract_ids([9672])
eddy_f = a.extract_ids([9672])
eddy_f.median_filter(1, "time", "lon").loess_filter(5, "time", "lon")
eddy_f.median_filter(1, "time", "lat").loess_filter(5, "time", "lat")
fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes((0.05, 0.1, 0.9, 0.9))
ax.set_xlim(17.5, 22)
ax.set_ylim(35, 36.5)
ax.set_aspect("equal")
ax.grid()
eddy.plot(ax, color="r", lw=0.5)
eddy_f.plot(ax, color="g", lw=1)
