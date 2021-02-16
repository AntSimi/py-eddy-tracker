"""
Grid advection
==============

Dummy advection which use only static geostrophic current, which didn't resolve the complex circulation of the ocean.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import py_eddy_tracker.gui
from py_eddy_tracker import data
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker.observations.observation import EddiesObservations

# %%
# Load Input grid, ADT is used to detect eddies
g = RegularGridDataset(
    data.get_path("dt_med_allsat_phy_l4_20160515_20190101.nc"), "longitude", "latitude"
)
# Compute u/v from height
g.add_uv("adt")

# %%
# Load detection files
a = EddiesObservations.load_file(data.get_path("Anticyclonic_20160515.nc"))
c = EddiesObservations.load_file(data.get_path("Cyclonic_20160515.nc"))


# %%
# Quiver from u/v with eddies
fig = plt.figure(figsize=(10, 5))
ax = fig.add_axes([0, 0, 1, 1], projection="full_axes")
ax.set_xlim(19, 30), ax.set_ylim(31, 36.5), ax.grid()
x, y = np.meshgrid(g.x_c, g.y_c)
a.filled(ax, facecolors="r", alpha=0.1), c.filled(ax, facecolors="b", alpha=0.1)
_ = ax.quiver(x.T, y.T, g.grid("u"), g.grid("v"), scale=20)


# %%
class VideoAnimation(FuncAnimation):
    def _repr_html_(self, *args, **kwargs):
        """To get video in html and have a player"""
        return self.to_html5_video()

    def save(self, *args, **kwargs):
        if args[0].endswith("gif"):
            # In this case gif is use to create thumbnail which are not use but consume same time than video
            # So we create an empty file, to save time
            with open(args[0], "w") as _:
                pass
            return
        return super().save(*args, **kwargs)


# %%
# Anim
# ----
# Particules positions
x, y = np.meshgrid(np.arange(13, 36, 0.125), np.arange(28, 40, 0.125))
x, y = x.reshape(-1), y.reshape(-1)
# Remove all original position that we can't advect at first place
m = ~np.isnan(g.interp("u", x, y))
x, y = x[m], y[m]

# Movie properties
kwargs = dict(frames=np.arange(51), interval=90)
kw_p = dict(nb_step=2, time_step=21600)
frame_t = kw_p["nb_step"] * kw_p["time_step"] / 86400.0


def anim_ax(generator, **kw):
    t = 0
    for _ in range(4):
        generator.__next__()
        t += frame_t

    fig = plt.figure(figsize=(10, 5), dpi=64)
    ax = fig.add_axes([0, 0, 1, 1], projection="full_axes")
    ax.set_xlim(19, 30), ax.set_ylim(31, 36.5), ax.grid()
    a.filled(ax, facecolors="r", alpha=0.1), c.filled(ax, facecolors="b", alpha=0.1)
    line = ax.plot([], [], "k", **kw)[0]
    return fig, ax.text(21, 32.1, ""), line, t


def update(i_frame, t_step):
    global t
    x, y = p.__next__()
    t += t_step
    l.set_data(x, y)
    txt.set_text(f"T0 + {t:.1f} days")


# %%
# Filament forward
# ^^^^^^^^^^^^^^^^
p = g.filament(x, y, "u", "v", **kw_p, filament_size=4, rk4=True)
fig, txt, l, t = anim_ax(p, lw=0.5)
ani = VideoAnimation(fig, update, **kwargs, fargs=(frame_t,))

# %%
# Filament backward
# ^^^^^^^^^^^^^^^^^
p = g.filament(x, y, "u", "v", **kw_p, filament_size=4, backward=True, rk4=True)
fig, txt, l, t = anim_ax(p, lw=0.5)
ani = VideoAnimation(fig, update, **kwargs, fargs=(-frame_t,))

# %%
# Particule forward
# ^^^^^^^^^^^^^^^^^
p = g.advect(x, y, "u", "v", **kw_p, rk4=True)
fig, txt, l, t = anim_ax(p, ls="", marker=".", markersize=1)
ani = VideoAnimation(fig, update, **kwargs, fargs=(frame_t,))
