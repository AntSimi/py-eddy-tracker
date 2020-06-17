"""
Lifetime Histogram
===================

"""
from matplotlib import pyplot as plt
from py_eddy_tracker.observations.tracking import TrackEddiesObservations
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
# Plot
fig = plt.figure()
ax_lifetime = fig.add_axes([0.05, 0.55, 0.4, 0.4])
ax_cum_lifetime = fig.add_axes([0.55, 0.55, 0.4, 0.4])
ax_ratio_lifetime = fig.add_axes([0.05, 0.05, 0.4, 0.4])
ax_ratio_cum_lifetime = fig.add_axes([0.55, 0.05, 0.4, 0.4])

cum_a, bins, _ = ax_cum_lifetime.hist(
    a["n"], histtype="step", bins=arange(0, 800, 1), label="Anticyclonic", color="r"
)
cum_c, bins, _ = ax_cum_lifetime.hist(
    c["n"], histtype="step", bins=arange(0, 800, 1), label="Cyclonic", color="b"
)

x = (bins[1:] + bins[:-1]) / 2.0
ax_ratio_cum_lifetime.plot(x, cum_c / cum_a)

nb_a, nb_c = cum_a[:-1] - cum_a[1:], cum_c[:-1] - cum_c[1:]
ax_lifetime.plot(x[1:], nb_a, label="Anticyclonic", color="r")
ax_lifetime.plot(x[1:], nb_c, label="Cyclonic", color="b")

ax_ratio_lifetime.plot(x[1:], nb_c / nb_a)

for ax in (ax_lifetime, ax_cum_lifetime, ax_ratio_cum_lifetime, ax_ratio_lifetime):
    ax.set_xlim(0, 365)
    if ax in (ax_lifetime, ax_cum_lifetime):
        ax.set_ylim(1, None)
        ax.set_yscale("log")
        ax.legend()
    else:
        ax.set_ylim(0, 2)
        ax.set_ylabel("Ratio Cyclonic/Anticyclonic")
    ax.set_xlabel("Lifetime (days)")
    ax.grid()
