"""
Display Tracks
======================

"""

import py_eddy_tracker_sample
from matplotlib import pyplot as plt

from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# %%
# Load experimental atlas
a = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Anticyclonic.zarr")
)
c = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Cyclonic.zarr")
)
print(a)

# %%
# keep only eddies longer than 20 weeks, use -1 to have no upper limit
a = a.extract_with_length((7 * 20, -1))
c = c.extract_with_length((7 * 20, -1))
print(a)

# %%
# Position filtering for nice display
a.position_filter(median_half_window=1, loess_half_window=5)
c.position_filter(median_half_window=1, loess_half_window=5)

# %%
# Plot
fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes((0.05, 0.1, 0.9, 0.9))
ax.set_aspect("equal")
ax.set_xlim(-6, 36.5), ax.set_ylim(30, 46)
a.plot(ax, ref=-10, label="Anticyclonic", color="r", lw=0.1)
c.plot(ax, ref=-10, label="Cyclonic", color="b", lw=0.1)
ax.legend()
ax.grid()
