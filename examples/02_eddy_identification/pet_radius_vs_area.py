"""
Radius vs area
==============

"""
from matplotlib import pyplot as plt
from numpy import array, pi

from py_eddy_tracker import data
from py_eddy_tracker.generic import coordinates_to_local
from py_eddy_tracker.observations.observation import EddiesObservations
from py_eddy_tracker.poly import poly_area

# %%
# Load detection files
a = EddiesObservations.load_file(data.get_demo_path("Anticyclonic_20190223.nc"))
areas = list()
# For each contour area will be compute in local reference
for i in a:
    x, y = coordinates_to_local(
        i["contour_lon_s"], i["contour_lat_s"], i["lon"], i["lat"]
    )
    areas.append(poly_area(x, y))
areas = array(areas)

# %%
# Radius provided by eddy detection is computed with :func:`~py_eddy_tracker.poly.fit_circle` method.
# This radius will be compared with an equivalent radius deduced from polygon area.
ax = plt.subplot(111)
ax.set_aspect("equal")
ax.grid()
ax.set_xlabel("Speed radius computed with fit_circle")
ax.set_ylabel("Radius deduced from area\nof contour_lon_s/contour_lat_s")
ax.set_title("Area vs radius")
ax.plot(a["radius_s"] / 1000.0, (areas / pi) ** 0.5 / 1000.0, ".")
ax.plot((0, 250), (0, 250), "r")

# %%
# Fit circle give a radius bigger than polygon area

# %%
# When error is tiny, radius are very close.
ax = plt.subplot(111)
ax.grid()
ax.set_xlabel("Radius ratio")
ax.set_ylabel("Shape error")
ax.set_title("err = f(radius_ratio)")
ax.plot(a["radius_s"] / (areas / pi) ** 0.5, a["shape_error_s"], ".")
