"""
[draft] Loopers vs eddies from altimetry
========================================

All loopers datas used in this example are a subset from the dataset describe in this article
[Lumpkin, R. : Global characteristics of coherent vortices from surface drifter trajectories](https://doi.org/10.1002/2015JC011435)


"""

import re

import numpy as np
import py_eddy_tracker_sample
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from py_eddy_tracker import data
from py_eddy_tracker.appli.gui import Anim
from py_eddy_tracker.observations.tracking import TrackEddiesObservations


# %%
class VideoAnimation(FuncAnimation):
    def _repr_html_(self, *args, **kwargs):
        """To get video in html and have a player"""
        content = self.to_html5_video()
        return re.sub(
            r'width="[0-9]*"\sheight="[0-9]*"', 'width="100%" height="100%"', content
        )

    def save(self, *args, **kwargs):
        if args[0].endswith("gif"):
            # In this case gif is use to create thumbnail which are not use but consume same time than video
            # So we create an empty file, to save time
            with open(args[0], "w") as _:
                pass
            return
        return super().save(*args, **kwargs)


def start_axes(title):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_axes([0.03, 0.03, 0.90, 0.94], aspect="equal")
    ax.set_xlim(-6, 36.5), ax.set_ylim(30, 46)
    ax.set_title(title, weight="bold")
    return ax


def update_axes(ax, mappable=None):
    ax.grid()
    if mappable:
        plt.colorbar(mappable, cax=ax.figure.add_axes([0.94, 0.05, 0.01, 0.9]))


# %%
# Load eddies dataset
cyclonic_eddies = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_demo_path("eddies_med_adt_allsat_dt2018/Cyclonic.zarr")
)
anticyclonic_eddies = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_demo_path(
        "eddies_med_adt_allsat_dt2018/Anticyclonic.zarr"
    )
)

# %%
# Load loopers dataset
loopers_med = TrackEddiesObservations.load_file(
    data.get_demo_path("loopers_lumpkin_med.nc")
)

# %%
#
ax = start_axes("All drifter available in med from Lumpkin dataset")
loopers_med.plot(ax, lw=0.5, color="r", ref=-10)
update_axes(ax)

# %%
# Only long period drifter
long_drifter = loopers_med.extract_with_length((400, -1))
ax = start_axes("Long period drifter")
long_drifter.plot(ax, lw=0.5, color="r", ref=-10)
update_axes(ax)
print(np.unique(long_drifter.tracks))

# %%
drifter_1 = long_drifter.extract_ids((3588,))
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111, aspect="equal")
ax.grid()
drifter_1.plot(ax, lw=0.5)
# %%
drifter_1.close_tracks(
    anticyclonic_eddies, method="close_center", delta=0.5, nb_obs_min=25
).tracks

# %%
# Animation which show drifter with colocated eddies
ed = anticyclonic_eddies.extract_ids((4738, 4836, 4910,))
x, y, t = drifter_1.lon, drifter_1.lat, drifter_1.time


def update(frame):
    # We display last 5 days of loopers trajectory
    m = (t < frame) * (t > (frame - 5))
    a.func_animation(frame)
    line.set_data(x[m], y[m])


a = Anim(ed, intern=True, figsize=(8, 8), cmap="magma_r", nb_step=10, dpi=60)
line = a.ax.plot([], [], "r", lw=4, zorder=100)[0]
kwargs = dict(frames=np.arange(*a.period, 1), interval=100)
a.fig.suptitle("")
_ = VideoAnimation(a.fig, update, **kwargs)

# %%
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.plot(drifter_1.time, drifter_1.radius_s / 1e3, label="loopers")
drifter_1.median_filter(7, "time", "radius_s", inplace=True)
drifter_1.loess_filter(15, "time", "radius_s", inplace=True)
ax.plot(
    drifter_1.time,
    drifter_1.radius_s / 1e3,
    label="loopers (filtered half window 15 days)",
)
ax.plot(ed.time, ed.radius_s / 1e3, label="altimetry")
ed.median_filter(7, "time", "radius_s", inplace=True)
ed.loess_filter(15, "time", "radius_s", inplace=True)
ax.plot(ed.time, ed.radius_s / 1e3, label="altimetry (filtered half window 15 days)")
ax.set_ylabel("radius(km)"), ax.set_ylim(0, 100)
ax.legend()
ax.grid()
_ = ax.set_title("Radius from loopers and altimeter")
