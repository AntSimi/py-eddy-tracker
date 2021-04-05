"""
Time advection
==============

Example which use CMEMS surface current with a Runge-Kutta 4 algorithm to advect particles.
"""
# sphinx_gallery_thumbnail_number = 2
import re
from datetime import datetime, timedelta

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import arange, isnan, meshgrid, ones

from py_eddy_tracker import start_logger
from py_eddy_tracker.data import get_demo_path
from py_eddy_tracker.dataset.grid import GridCollection
from py_eddy_tracker.gui import GUI_AXES

start_logger().setLevel("ERROR")


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


# %%
# Data
# ----
# Load Input time grid ADT
c = GridCollection.from_netcdf_cube(
    get_demo_path("dt_med_allsat_phy_l4_2005T2.nc"),
    "longitude",
    "latitude",
    "time",
    # To create U/V variable
    heigth="adt",
)

# %%
# Anim
# ----
# Particles setup
step_p = 1 / 8
x, y = meshgrid(arange(13, 36, step_p), arange(28, 40, step_p))
x, y = x.reshape(-1), y.reshape(-1)
# Remove all original position that we can't advect at first place
t0 = 20181
m = ~isnan(c[t0].interp("u", x, y))
x0, y0 = x[m], y[m]
x, y = x0.copy(), y0.copy()


# %%
# Function
def anim_ax(**kw):
    fig = plt.figure(figsize=(10, 5), dpi=55)
    axes = fig.add_axes([0, 0, 1, 1], projection=GUI_AXES)
    axes.set_xlim(19, 30), axes.set_ylim(31, 36.5), axes.grid()
    line = axes.plot([], [], "k", **kw)[0]
    return fig, axes.text(21, 32.1, ""), line


def update(_):
    tt, xt, yt = f.__next__()
    mappable.set_data(xt, yt)
    d = timedelta(tt / 86400.0) + datetime(1950, 1, 1)
    txt.set_text(f"{d:%Y/%m/%d-%H}")


# %%
f = c.filament(x, y, "u", "v", t_init=t0, nb_step=2, time_step=21600, filament_size=3)
fig, txt, mappable = anim_ax(lw=0.5)
ani = VideoAnimation(fig, update, frames=arange(160), interval=100)


# %%
# Particules stat
# ---------------
# Time_step settings
# ^^^^^^^^^^^^^^^^^^
# Dummy experiment to test advection precision, we run particles 50 days forward and backward with different time step
# and we measure distance between new positions and original positions.
fig = plt.figure()
ax = fig.add_subplot(111)
kw = dict(
    bins=arange(0, 50, 0.002),
    cumulative=True,
    weights=ones(x0.shape) / x0.shape[0] * 100.0,
    histtype="step",
)
kw_p = dict(u_name="u", v_name="v", nb_step=1)
for time_step in (10800, 21600, 43200, 86400):
    x, y = x0.copy(), y0.copy()
    nb = int(30 * 86400 / time_step)
    # Go forward
    p = c.advect(x, y, time_step=time_step, t_init=20181.5, **kw_p)
    for i in range(nb):
        t_, _, _ = p.__next__()
    # Go backward
    p = c.advect(x, y, time_step=time_step, backward=True, t_init=t_ / 86400.0, **kw_p)
    for i in range(nb):
        t_, _, _ = p.__next__()
    d = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
    ax.hist(d, **kw, label=f"{86400. / time_step:.0f} time step by day")
ax.set_xlim(0, 0.25), ax.set_ylim(0, 100), ax.legend(loc="lower right"), ax.grid()
ax.set_title("Distance after 50 days forward and 50 days backward")
ax.set_xlabel("Distance between original position and final position (in degrees)")
_ = ax.set_ylabel("Percent of particles with distance lesser than")

# %%
# Time duration
# ^^^^^^^^^^^^^
# We keep same time_step but change time duration
fig = plt.figure()
ax = fig.add_subplot(111)
time_step = 10800
for duration in (10, 40, 80):
    x, y = x0.copy(), y0.copy()
    nb = int(duration * 86400 / time_step)
    # Go forward
    p = c.advect(x, y, time_step=time_step, t_init=20181.5, **kw_p)
    for i in range(nb):
        t_, _, _ = p.__next__()
    # Go backward
    p = c.advect(x, y, time_step=time_step, backward=True, t_init=t_ / 86400.0, **kw_p)
    for i in range(nb):
        t_, _, _ = p.__next__()
    d = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
    ax.hist(d, **kw, label=f"Time duration {duration} days")
ax.set_xlim(0, 0.25), ax.set_ylim(0, 100), ax.legend(loc="lower right"), ax.grid()
ax.set_title(
    "Distance after N days forward and N days backward\nwith a time step of 1/8 days"
)
ax.set_xlabel("Distance between original position and final position (in degrees)")
_ = ax.set_ylabel("Percent of particles with distance lesser than ")
