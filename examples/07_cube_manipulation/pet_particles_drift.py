"""
Build path of particle drifting
===============================

"""

from matplotlib import pyplot as plt
from numpy import arange, meshgrid

from py_eddy_tracker import start_logger
from py_eddy_tracker.data import get_demo_path
from py_eddy_tracker.dataset.grid import GridCollection

start_logger().setLevel("ERROR")

# %%
# Load data cube
c = GridCollection.from_netcdf_cube(
    get_demo_path("dt_med_allsat_phy_l4_2005T2.nc"),
    "longitude",
    "latitude",
    "time",
    unset=True,
)

# %%
# Advection properties
nb_days, step_by_day = 10, 6
nb_time = step_by_day * nb_days
kw_p = dict(nb_step=1, time_step=86400 / step_by_day)
t0 = 20210

# %%
# Get paths
x0, y0 = meshgrid(arange(32, 35, 0.5), arange(32.5, 34.5, 0.5))
x0, y0 = x0.reshape(-1), y0.reshape(-1)
t, x, y = c.path(x0, y0, h_name="adt", t_init=t0, **kw_p, nb_time=nb_time)

# %%
# Plot paths
ax = plt.figure(figsize=(9, 6)).add_subplot(111, aspect="equal")
ax.plot(x0, y0, "k.", ms=20)
ax.plot(x, y, lw=3)
ax.set_title("10 days particle paths")
ax.set_xlim(31, 35), ax.set_ylim(32, 34.5)
ax.grid()
