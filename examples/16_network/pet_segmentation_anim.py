"""
Network segmentation process
============================
"""

import re

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from numpy import ones

import py_eddy_tracker.gui
from py_eddy_tracker import data
from py_eddy_tracker.observations.tracking import TrackEddiesObservations


# %%
class VideoAnimation(FuncAnimation):
    def _repr_html_(self, *args, **kwargs):
        """To get video in html and have a player"""
        content = self.to_html5_video()
        return re.sub(
            'width="[0-9]*"\sheight="[0-9]*"', 'width="100%" height="100%"', content
        )

    def save(self, *args, **kwargs):
        if args[0].endswith("gif"):
            # In this case gif is use to create thumbnail which are not use but consume same time than video
            # So we create an empty file, to save time
            with open(args[0], "w") as _:
                pass
            return
        return super().save(*args, **kwargs)


# %%
# Overlaod of class to pick up
TRACKS = list()
INDICES = list()


class MyTrack(TrackEddiesObservations):
    @staticmethod
    def get_next_obs(i_current, ids, x, y, time_s, time_e, time_ref, window, **kwargs):
        TRACKS.append(ids["track"].copy())
        INDICES.append(i_current)
        return TrackEddiesObservations.get_next_obs(
            i_current, ids, x, y, time_s, time_e, time_ref, window, **kwargs
        )


# %%
# Load data
# ---------
# Load data where observations are put in same network but no segmentation
e = MyTrack.load_file(data.get_path("c568803.nc"))
# FIXME : Must be rewrote
e.lon[:] = (e.lon + 180) % 360 - 180
e.contour_lon_e[:] = ((e.contour_lon_e.T - e.lon + 180) % 360 - 180 + e.lon).T
e.contour_lon_s[:] = ((e.contour_lon_s.T - e.lon + 180) % 360 - 180 + e.lon).T
# %%
# Do segmentation
# ---------------
# Segmentation based on maximum overlap, temporal window for candidates = 5 days
matrix = e.split_network(intern=False, window=5)


# %%
# Anim
# ----
def update(i_frame):
    tr = TRACKS[i_frame]
    mappable_tracks.set_array(tr)
    s = 40 * ones(tr.shape)
    s[tr == 0] = 4
    mappable_tracks.set_sizes(s)

    indices_frames = INDICES[i_frame]
    mappable_CONTOUR.set_data(e.contour_lon_e[indices_frames], e.contour_lat_e[indices_frames],)
    mappable_CONTOUR.set_color(cmap.colors[tr[indices_frames] % len(cmap.colors)])
    return (mappable_tracks,)


fig = plt.figure(figsize=(15, 8), dpi=60)
ax = fig.add_axes([0.04, 0.06, 0.94, 0.88], projection="full_axes")
ax.set_title(f"{len(e)} observations to segment")
ax.set_xlim(-13, 20), ax.set_ylim(-36.5, -20), ax.grid()
vmax = TRACKS[-1].max()
cmap = ListedColormap(["gray", *e.COLORS[:-1]], name="from_list", N=vmax)
mappable_tracks = ax.scatter(
    e.lon, e.lat, c=TRACKS[0], cmap=cmap, vmin=0, vmax=vmax, s=20
)
mappable_CONTOUR = ax.plot(
    e.contour_lon_e[INDICES[0]], e.contour_lat_e[INDICES[0]], color=cmap.colors[0]
)[0]
ani = VideoAnimation(
    fig, update, frames=range(1, len(TRACKS), 4), interval=125, blit=True
)