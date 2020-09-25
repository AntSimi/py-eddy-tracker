"""
Propagation Histogram
=====================

"""
from matplotlib import pyplot as plt
from py_eddy_tracker.observations.tracking import TrackEddiesObservations
from py_eddy_tracker.generic import cumsum_by_track
import py_eddy_tracker_sample
from numpy import arange

# %%
# Load an experimental med atlas over a period of 26 years (1993-2019)
a = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Anticyclonic.zarr")
)
c = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Cyclonic.zarr")
)

# %%
# Filtering position to remove noisy position
a.position_filter(median_half_window=1, loess_half_window=5)
c.position_filter(median_half_window=1, loess_half_window=5)

# %%
# Compute curvilign distance
d_a = cumsum_by_track(a.distance_to_next(), a.tracks) / 1000.0
d_c = cumsum_by_track(c.distance_to_next(), c.tracks) / 1000.0

# %%
# Plot
fig = plt.figure()
ax_propagation = fig.add_axes([0.05, 0.55, 0.4, 0.4])
ax_cum_propagation = fig.add_axes([0.55, 0.55, 0.4, 0.4])
ax_ratio_propagation = fig.add_axes([0.05, 0.05, 0.4, 0.4])
ax_ratio_cum_propagation = fig.add_axes([0.55, 0.05, 0.4, 0.4])

bins = arange(0, 1500, 10)
cum_a, bins, _ = ax_cum_propagation.hist(
    d_a, histtype="step", bins=bins, label="Anticyclonic", color="r"
)
cum_c, bins, _ = ax_cum_propagation.hist(
    d_c, histtype="step", bins=bins, label="Cyclonic", color="b"
)

x = (bins[1:] + bins[:-1]) / 2.0
ax_ratio_cum_propagation.plot(x, cum_c / cum_a)

nb_a, nb_c = cum_a[:-1] - cum_a[1:], cum_c[:-1] - cum_c[1:]
ax_propagation.plot(x[1:], nb_a, label="Anticyclonic", color="r")
ax_propagation.plot(x[1:], nb_c, label="Cyclonic", color="b")

ax_ratio_propagation.plot(x[1:], nb_c / nb_a)

for ax in (
    ax_propagation,
    ax_cum_propagation,
    ax_ratio_cum_propagation,
    ax_ratio_propagation,
):
    ax.set_xlim(0, 1000)
    if ax in (ax_propagation, ax_cum_propagation):
        ax.set_ylim(1, None)
        ax.set_yscale("log")
        ax.legend()
    else:
        ax.set_ylim(0, 2)
        ax.set_ylabel("Ratio Cyclonic/Anticyclonic")
    ax.set_xlabel("Propagation (km)")
    ax.grid()
