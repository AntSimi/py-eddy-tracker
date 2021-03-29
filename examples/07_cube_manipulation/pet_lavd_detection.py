"""
LAVD detection and geometric detection
======================================

Naive method to reproduce LAVD(Lagrangian-Averaged Vorticity deviation).
In the current example we didn't remove a mean vorticity.

Method are described here:

    - Abernathey, Ryan, and George Haller. "Transport by Lagrangian Vortices in the Eastern Pacific",
      Journal of Physical Oceanography 48, 3 (2018): 667-685, accessed Feb 16, 2021,
      https://doi.org/10.1175/JPO-D-17-0102.1
    - `Transport by Coherent Lagrangian Vortices`_,
      R. Abernathey, Sinha A., Tarshish N., Liu T., Zhang C., Haller G., 2019,
      Talk a t the Sources and Sinks of Ocean Mesoscale Eddy Energy CLIVAR Workshop

.. _Transport by Coherent Lagrangian Vortices:
    https://usclivar.org/sites/default/files/meetings/2019/presentations/Aberernathey_CLIVAR.pdf

"""
from datetime import datetime

from matplotlib import pyplot as plt
from numpy import arange, isnan, ma, meshgrid, zeros

import py_eddy_tracker.gui
from py_eddy_tracker import start_logger
from py_eddy_tracker.data import get_demo_path
from py_eddy_tracker.dataset.grid import GridCollection, RegularGridDataset

start_logger().setLevel("ERROR")


# %%
class LAVDGrid(RegularGridDataset):
    def init_speed_coef(self, uname="u", vname="v"):
        """Hack to be able to identify eddy with LAVD field"""
        self._speed_ev = self.grid("lavd")

    @classmethod
    def from_(cls, x, y, z):
        z.mask += isnan(z.data)
        datas = dict(lavd=z, lon=x, lat=y)
        return cls.with_array(coordinates=("lon", "lat"), datas=datas, centered=True)


# %%
def start_ax(title="", dpi=90):
    fig = plt.figure(figsize=(12, 5), dpi=dpi)
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.9], projection="full_axes")
    ax.set_xlim(-6, 36), ax.set_ylim(31, 45)
    ax.set_title(title)
    return fig, ax, ax.text(3, 32, "", fontsize=20)


def update_axes(ax, mappable=None):
    ax.grid()
    if mappable:
        cb = plt.colorbar(
            mappable,
            cax=ax.figure.add_axes([0.05, 0.1, 0.9, 0.01]),
            orientation="horizontal",
        )
        cb.set_label("LAVD at initial position")
        return cb


kw_lavd = dict(vmin=0, vmax=2e-5, cmap="viridis")

# %%
# Data
# ----

# Load data cube of 3 month
c = GridCollection.from_netcdf_cube(
    get_demo_path("dt_med_allsat_phy_l4_2005T2.nc"),
    "longitude",
    "latitude",
    "time",
    heigth="adt",
)

# Add vorticity at each time step
for g in c:
    u_y = g.compute_stencil(g.grid("u"), vertical=True)
    v_x = g.compute_stencil(g.grid("v"))
    g.vars["vort"] = v_x - u_y

# %%
# Particles
# ---------

# Time properties, for example with advection only 25 days
nb_days, step_by_day = 25, 6
nb_time = step_by_day * nb_days
kw_p = dict(nb_step=1, time_step=86400 / step_by_day)
t0 = 20236
t0_grid = c[t0]
# Geographic properties, we use a coarser resolution for time consuming reasons
step = 1 / 32.0
x_g, y_g = arange(-6, 36, step), arange(30, 46, step)
x0, y0 = meshgrid(x_g, y_g)
original_shape = x0.shape
x0, y0 = x0.reshape(-1), y0.reshape(-1)
# Get all particles in defined area
m = ~isnan(t0_grid.interp("vort", x0, y0))
x0, y0 = x0[m], y0[m]
print(f"{x0.size} particles advected")
# Gridded mask
m = m.reshape(original_shape)

# %%
# LAVD forward (dynamic field)
# ----------------------------
lavd = zeros(original_shape)
lavd_ = lavd[m]
p = c.advect(x0.copy(), y0.copy(), "u", "v", t_init=t0, **kw_p)
for _ in range(nb_time):
    t, x, y = p.__next__()
    lavd_ += abs(c.interp("vort", t / 86400.0, x, y))
lavd[m] = lavd_ / nb_time
# Put LAVD result in a standard py eddy tracker grid
lavd_forward = LAVDGrid.from_(x_g, y_g, ma.array(lavd, mask=~m).T)
# Display
fig, ax, _ = start_ax("LAVD with a forward advection")
mappable = lavd_forward.display(ax, "lavd", **kw_lavd)
_ = update_axes(ax, mappable)

# %%
# LAVD backward (dynamic field)
# -----------------------------
lavd = zeros(original_shape)
lavd_ = lavd[m]
p = c.advect(x0.copy(), y0.copy(), "u", "v", t_init=t0, backward=True, **kw_p)
for i in range(nb_time):
    t, x, y = p.__next__()
    lavd_ += abs(c.interp("vort", t / 86400.0, x, y))
lavd[m] = lavd_ / nb_time
# Put LAVD result in a standard py eddy tracker grid
lavd_backward = LAVDGrid.from_(x_g, y_g, ma.array(lavd, mask=~m).T)
# Display
fig, ax, _ = start_ax("LAVD with a backward advection")
mappable = lavd_backward.display(ax, "lavd", **kw_lavd)
_ = update_axes(ax, mappable)

# %%
# LAVD forward (static field)
# ---------------------------
lavd = zeros(original_shape)
lavd_ = lavd[m]
p = t0_grid.advect(x0.copy(), y0.copy(), "u", "v", **kw_p)
for _ in range(nb_time):
    x, y = p.__next__()
    lavd_ += abs(t0_grid.interp("vort", x, y))
lavd[m] = lavd_ / nb_time
# Put LAVD result in a standard py eddy tracker grid
lavd_forward_static = LAVDGrid.from_(x_g, y_g, ma.array(lavd, mask=~m).T)
# Display
fig, ax, _ = start_ax("LAVD with a forward advection on a static velocity field")
mappable = lavd_forward_static.display(ax, "lavd", **kw_lavd)
_ = update_axes(ax, mappable)

# %%
# LAVD backward (static field)
# ----------------------------
lavd = zeros(original_shape)
lavd_ = lavd[m]
p = t0_grid.advect(x0.copy(), y0.copy(), "u", "v", backward=True, **kw_p)
for i in range(nb_time):
    x, y = p.__next__()
    lavd_ += abs(t0_grid.interp("vort", x, y))
lavd[m] = lavd_ / nb_time
# Put LAVD result in a standard py eddy tracker grid
lavd_backward_static = LAVDGrid.from_(x_g, y_g, ma.array(lavd, mask=~m).T)
# Display
fig, ax, _ = start_ax("LAVD with a backward advection on a static velocity field")
mappable = lavd_backward_static.display(ax, "lavd", **kw_lavd)
_ = update_axes(ax, mappable)

# %%
# Contour detection
# -----------------
# To extract contour from LAVD grid, we will used method design for SSH, with some hacks and adapted options.
# It will produce false amplitude and speed.
kw_ident = dict(
    force_speed_unit="m/s",
    force_height_unit="m",
    pixel_limit=(40, 200000),
    date=datetime(2005, 5, 18),
    uname=None,
    vname=None,
    grid_height="lavd",
    shape_error=70,
    step=1e-6,
)
fig, ax, _ = start_ax("Detection of eddies with several method")
t0_grid.bessel_high_filter("adt", 700)
a, c = t0_grid.eddy_identification(
    "adt", "u", "v", kw_ident["date"], step=0.002, shape_error=70
)
kw_ed = dict(ax=ax, intern=True, ref=-10)
a.filled(
    facecolors="#FFEFCD", label="Anticyclonic SSH detection {nb_obs} eddies", **kw_ed
)
c.filled(facecolors="#DEDEDE", label="Cyclonic SSH detection {nb_obs} eddies", **kw_ed)
kw_cont = dict(ax=ax, extern_only=True, ls="-", ref=-10)
forward, _ = lavd_forward.eddy_identification(**kw_ident)
forward.display(label="LAVD forward {nb_obs} eddies", color="g", **kw_cont)
backward, _ = lavd_backward.eddy_identification(**kw_ident)
backward.display(label="LAVD backward {nb_obs} eddies", color="r", **kw_cont)
forward, _ = lavd_forward_static.eddy_identification(**kw_ident)
forward.display(label="LAVD forward static {nb_obs} eddies", color="cyan", **kw_cont)
backward, _ = lavd_backward_static.eddy_identification(**kw_ident)
backward.display(
    label="LAVD backward static {nb_obs} eddies", color="orange", **kw_cont
)
ax.legend()
update_axes(ax)
