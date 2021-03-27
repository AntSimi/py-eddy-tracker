"""
Follow particle
===============

"""
import re

from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from numba import types as nb_types
from numpy import arange, meshgrid, ones, unique, where, zeros

from py_eddy_tracker import start_logger
from py_eddy_tracker.appli.gui import Anim
from py_eddy_tracker.data import get_path
from py_eddy_tracker.dataset.grid import GridCollection
from py_eddy_tracker.observations.network import NetworkObservations
from py_eddy_tracker.poly import group_obs

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
n = NetworkObservations.load_file(get_path("network_med.nc")).network(651)
n = n.extract_with_mask((n.time >= 20180) * (n.time <= 20269))
n = n.remove_dead_end(nobs=0, ndays=10)
n.numbering_segment()
c = GridCollection.from_netcdf_cube(
    get_path("dt_med_allsat_phy_l4_2005T2.nc"),
    "longitude",
    "latitude",
    "time",
    heigth="adt",
)

# %%
# Schema
# ------
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
_ = n.display_timeline(ax, field="longitude", marker="+", lw=2, markersize=5)

# %%
# Animation
# ---------
# Particle settings
t_snapshot = 20200
step = 1 / 50.0
x, y = meshgrid(arange(20, 36, step), arange(30, 46, step))
N = 6
x_f, y_f = x[::N, ::N].copy(), y[::N, ::N].copy()
x, y = x.reshape(-1), y.reshape(-1)
x_f, y_f = x_f.reshape(-1), y_f.reshape(-1)
n_ = n.extract_with_mask(n.time == t_snapshot)
index = n_.contains(x, y, intern=True)
m = index != -1
index = n_.segment[index[m]]
index_ = unique(index)
x, y = x[m], y[m]
m = ~n_.inside(x_f, y_f, intern=True)
x_f, y_f = x_f[m], y_f[m]

# %%
# Animation
cmap = colors.ListedColormap(list(n.COLORS), name="from_list", N=n.segment.max() + 1)
a = Anim(
    n,
    intern=False,
    figsize=(12, 6),
    nb_step=1,
    dpi=60,
    field_color="segment",
    field_txt="segment",
    cmap=cmap,
)
a.fig.suptitle(""), a.ax.set_xlim(24, 36), a.ax.set_ylim(30, 36)
a.txt.set_position((25, 31))

step = 0.25
kw_p = dict(nb_step=2, time_step=86400 * step * 0.5, t_init=t_snapshot - 2 * step)

mappables = dict()
particules = c.advect(x, y, "u", "v", **kw_p)
filament = c.filament(x_f, y_f, "u", "v", **kw_p, filament_size=3)
kw = dict(ls="", marker=".", markersize=0.25)
for k in index_:
    m = k == index
    mappables[k] = a.ax.plot([], [], color=cmap(k), **kw)[0]
m_filament = a.ax.plot([], [], lw=0.25, color="gray")[0]


def update(frame):
    tt, xt, yt = particules.__next__()
    for k, mappable in mappables.items():
        m = index == k
        mappable.set_data(xt[m], yt[m])
    tt, xt, yt = filament.__next__()
    m_filament.set_data(xt, yt)
    if frame % 1 == 0:
        a.func_animation(frame)


ani = VideoAnimation(a.fig, update, frames=arange(20200, 20269, step), interval=200)


# %%
# In which observations are the particle
# --------------------------------------
def advect(x, y, c, t0, delta_t):
    """
    Advect particle from t0 to t0 + delta_t, with data cube.
    """
    kw = dict(nb_step=6, time_step=86400 / 6)
    if delta_t < 0:
        kw["backward"] = True
        delta_t = -delta_t
    p = c.advect(x, y, "u", "v", t_init=t0, **kw)
    for _ in range(delta_t):
        t, x, y = p.__next__()
    return t, x, y


def particle_candidate(x, y, c, eddies, t_start, i_target, pct, **kwargs):
    # Obs from initial time
    m_start = eddies.time == t_start
    e = eddies.extract_with_mask(m_start)
    # to be able to get global index
    translate_start = where(m_start)[0]
    # Identify particle in eddies(only in core)
    i_start = e.contains(x, y, intern=True)
    m = i_start != -1
    x, y, i_start = x[m], y[m], i_start[m]
    # Advect
    t_end, x, y = advect(x, y, c, t_start, **kwargs)
    # eddies at last date
    m_end = eddies.time == t_end / 86400
    e_end = eddies.extract_with_mask(m_end)
    # to be able to get global index
    translate_end = where(m_end)[0]
    # Id eddies for each alive particle(in core and extern)
    i_end = e_end.contains(x, y)
    # compute matrix and filled target array
    get_matrix(i_start, i_end, translate_start, translate_end, i_target, pct)


@njit(cache=True)
def get_matrix(i_start, i_end, translate_start, translate_end, i_target, pct):
    nb_start, nb_end = translate_start.size, translate_end.size
    # Matrix which will store count for every couple
    count = zeros((nb_start, nb_end), dtype=nb_types.int32)
    # Number of particle in each origin observation
    ref = zeros(nb_start, dtype=nb_types.int32)
    # For each particle
    for i in range(i_start.size):
        i_end_ = i_end[i]
        i_start_ = i_start[i]
        if i_end_ != -1:
            count[i_start_, i_end_] += 1
        ref[i_start_] += 1
    for i in range(nb_start):
        for j in range(nb_end):
            pct_ = count[i, j]
            # If there are particle from i to j
            if pct_ != 0:
                # Get percent
                pct_ = pct_ / ref[i] * 100.0
                # Get indices in full dataset
                i_, j_ = translate_start[i], translate_end[j]
                pct_0 = pct[i_, 0]
                if pct_ > pct_0:
                    pct[i_, 1] = pct_0
                    pct[i_, 0] = pct_
                    i_target[i_, 1] = i_target[i_, 0]
                    i_target[i_, 0] = j_
                elif pct_ > pct[i_, 1]:
                    pct[i_, 1] = pct_
                    i_target[i_, 1] = j_
    return i_target, pct


# %%
# Particle advection
# ^^^^^^^^^^^^^^^^^^
step = 1 / 60.0

x, y = meshgrid(arange(24, 36, step), arange(31, 36, step))
x0, y0 = x.reshape(-1), y.reshape(-1)
# Pre-order to speed up
_, i = group_obs(x0, y0, 1, 360)
x0, y0 = x0[i], y0[i]

t_start, t_end = n.period
dt = 14

shape = (n.obs.size, 2)
# Forward run
i_target_f, pct_target_f = -ones(shape, dtype="i4"), zeros(shape, dtype="i1")
for t in range(t_start, t_end - dt):
    particle_candidate(x0, y0, c, n, t, i_target_f, pct_target_f, delta_t=dt)

# Backward run
i_target_b, pct_target_b = -ones(shape, dtype="i4"), zeros(shape, dtype="i1")
for t in range(t_start + dt, t_end):
    particle_candidate(x0, y0, c, n, t, i_target_b, pct_target_b, delta_t=-dt)

# %%
fig = plt.figure(figsize=(10, 10))
ax_1st_b = fig.add_axes([0.05, 0.52, 0.45, 0.45])
ax_2nd_b = fig.add_axes([0.05, 0.05, 0.45, 0.45])
ax_1st_f = fig.add_axes([0.52, 0.52, 0.45, 0.45])
ax_2nd_f = fig.add_axes([0.52, 0.05, 0.45, 0.45])
ax_1st_b.set_title("Backward advection for each time step")
ax_1st_f.set_title("Forward advection for each time step")


def color_alpha(target, pct, vmin=5, vmax=80):
    color = cmap(n.segment[target])
    # We will hide under 5 % and from 80% to 100 % it will be 1
    alpha = (pct - vmin) / (vmax - vmin)
    alpha[alpha < 0] = 0
    alpha[alpha > 1] = 1
    color[:, 3] = alpha
    return color


kw = dict(
    name=None, yfield="longitude", event=False, zorder=-100, s=(n.speed_area / 20e6)
)
n.scatter_timeline(ax_1st_b, c=color_alpha(i_target_b.T[0], pct_target_b.T[0]), **kw)
n.scatter_timeline(ax_2nd_b, c=color_alpha(i_target_b.T[1], pct_target_b.T[1]), **kw)
n.scatter_timeline(ax_1st_f, c=color_alpha(i_target_f.T[0], pct_target_f.T[0]), **kw)
n.scatter_timeline(ax_2nd_f, c=color_alpha(i_target_f.T[1], pct_target_f.T[1]), **kw)
for ax in (ax_1st_b, ax_2nd_b, ax_1st_f, ax_2nd_f):
    n.display_timeline(ax, field="longitude", marker="+", lw=2, markersize=5)
    ax.grid()