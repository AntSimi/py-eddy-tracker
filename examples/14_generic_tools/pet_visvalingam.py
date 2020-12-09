"""
Visvalingam algorithm
=====================
"""
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from numba import njit
from numpy import array, empty

from py_eddy_tracker import data
from py_eddy_tracker.generic import uniform_resample
from py_eddy_tracker.observations.observation import EddiesObservations
from py_eddy_tracker.poly import vertice_overlap, visvalingam


@njit(cache=True)
def visvalingam_polys(x, y, nb_pt):
    nb = x.shape[0]
    x_new = empty((nb, nb_pt), dtype=x.dtype)
    y_new = empty((nb, nb_pt), dtype=y.dtype)
    for i in range(nb):
        x_new[i], y_new[i] = visvalingam(x[i], y[i], nb_pt)
    return x_new, y_new


@njit(cache=True)
def uniform_resample_polys(x, y, nb_pt):
    nb = x.shape[0]
    x_new = empty((nb, nb_pt), dtype=x.dtype)
    y_new = empty((nb, nb_pt), dtype=y.dtype)
    for i in range(nb):
        x_new[i], y_new[i] = uniform_resample(x[i], y[i], fixed_size=nb_pt)
    return x_new, y_new


def update_line(num):
    nb = 50 - num - 20
    x_v, y_v = visvalingam_polys(a.contour_lon_e, a.contour_lat_e, nb)
    for i, (x_, y_) in enumerate(zip(x_v, y_v)):
        lines_v[i].set_data(x_, y_)
    x_u, y_u = uniform_resample_polys(a.contour_lon_e, a.contour_lat_e, nb)
    for i, (x_, y_) in enumerate(zip(x_u, y_u)):
        lines_u[i].set_data(x_, y_)
    scores_v = vertice_overlap(a.contour_lon_e, a.contour_lat_e, x_v, y_v) * 100.0
    scores_u = vertice_overlap(a.contour_lon_e, a.contour_lat_e, x_u, y_u) * 100.0
    for i, (s_v, s_u) in enumerate(zip(scores_v, scores_u)):
        texts[i].set_text(f"Score uniform {s_u:.1f} %\nScore visvalingam {s_v:.1f} %")
    title.set_text(f"{nb} points by contour in place of 50")
    return (title, *lines_u, *lines_v, *texts)


# %%
# Load detection files
a = EddiesObservations.load_file(data.get_path("Anticyclonic_20190223.nc"))
a = a.extract_with_mask((abs(a.lat) < 66) * (abs(a.radius_e) > 80e3))

nb_pt = 10
x_v, y_v = visvalingam_polys(a.contour_lon_e, a.contour_lat_e, nb_pt)
x_u, y_u = uniform_resample_polys(a.contour_lon_e, a.contour_lat_e, nb_pt)
scores_v = vertice_overlap(a.contour_lon_e, a.contour_lat_e, x_v, y_v) * 100.0
scores_u = vertice_overlap(a.contour_lon_e, a.contour_lat_e, x_u, y_u) * 100.0
d_6 = scores_v - scores_u
nb_pt = 18
x_v, y_v = visvalingam_polys(a.contour_lon_e, a.contour_lat_e, nb_pt)
x_u, y_u = uniform_resample_polys(a.contour_lon_e, a.contour_lat_e, nb_pt)
scores_v = vertice_overlap(a.contour_lon_e, a.contour_lat_e, x_v, y_v) * 100.0
scores_u = vertice_overlap(a.contour_lon_e, a.contour_lat_e, x_u, y_u) * 100.0
d_12 = scores_v - scores_u
a = a.index(array((d_6.argmin(), d_6.argmax(), d_12.argmin(), d_12.argmax())))


# %%
fig = plt.figure()
axs = [
    fig.add_subplot(221),
    fig.add_subplot(222),
    fig.add_subplot(223),
    fig.add_subplot(224),
]
lines_u, lines_v, texts, score_text = list(), list(), list(), list()
for i, obs in enumerate(a):
    axs[i].set_aspect("equal")
    axs[i].grid()
    axs[i].set_xticklabels([]), axs[i].set_yticklabels([])
    axs[i].plot(
        obs["contour_lon_e"], obs["contour_lat_e"], "r", lw=6, label="Original contour"
    )
    lines_v.append(axs[i].plot([], [], color="limegreen", lw=4, label="visvalingam")[0])
    lines_u.append(
        axs[i].plot([], [], color="black", lw=2, label="uniform resampling")[0]
    )
    texts.append(axs[i].set_title("", fontsize=8))
axs[0].legend(fontsize=8)
title = fig.suptitle("")
ani = animation.FuncAnimation(fig, update_line, 27)
