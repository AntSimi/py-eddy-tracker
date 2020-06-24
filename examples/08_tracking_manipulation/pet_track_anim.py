"""
Track animation
===============

Run in a terminal this script, which allow to watch eddy evolution

"""
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from numpy import arange
from py_eddy_tracker.observations.tracking import TrackEddiesObservations
from py_eddy_tracker.poly import create_vertice
import py_eddy_tracker_sample

# %%
# Load experimental atlas, and we select one eddy
a = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Anticyclonic.zarr")
)
eddy = a.extract_ids([9672])
t0, t1 = eddy.period
t = eddy.time
x = eddy["contour_lon_s"]
y = eddy["contour_lat_s"]

# %%
# General value
T = 25.0
cmap = plt.get_cmap("viridis")
COLORS = cmap(arange(T + 1) / T)


# %%
# plot
fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes((0.05, 0.05, 0.9, 0.9))
ax.set_xlim(16.5, 23)
ax.set_ylim(34.5, 37)
ax.set_aspect("equal")
ax.grid()
# init mappable
txt = ax.text(16.6, 36.8, "", zorder=10)
segs = list()
c = LineCollection([], zorder=1)
ax.add_collection(c)

fig.canvas.draw()
plt.show(block=False)
# save background for future bliting
bg_cache = fig.canvas.copy_from_bbox(ax.bbox)
# display contour every 2 day
for t_ in range(t0, t1 + 1, 2):
    fig.canvas.restore_region(bg_cache)
    # select contour for this time step
    m = t == t_
    segs.append(create_vertice(x[m][0], y[m][0]))
    c.set_paths(segs)
    c.set_color(COLORS[-len(segs) :])
    txt.set_text(f"{t0} -> {t_} -> {t1}")
    ax.draw_artist(c)
    ax.draw_artist(txt)
    # Remove first segment to keep only T contour
    if len(segs) > T:
        segs.pop(0)
    # paint updated artist
    fig.canvas.blit(ax.bbox)

    fig.canvas.start_event_loop(1e-10)
