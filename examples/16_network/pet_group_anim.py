"""
Network group process
=====================
"""
# sphinx_gallery_thumbnail_number = 2
import re
from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from numba import njit
from numpy import arange, array, empty, ones

from py_eddy_tracker import data
from py_eddy_tracker.generic import flatten_line_matrix
from py_eddy_tracker.observations.network import Network
from py_eddy_tracker.observations.observation import EddiesObservations


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
NETWORK_GROUPS = list()


@njit(cache=True)
def apply_replace(x, x0, x1):
    nb = x.shape[0]
    for i in range(nb):
        if x[i] == x0:
            x[i] = x1


# %%
# Modified class to catch group process at each step in order to illustrate processing
class MyNetwork(Network):
    def get_group_array(self, results, nb_obs):
        """With a loop on all pair of index, we will label each obs with a group
        number
        """
        nb_obs = array(nb_obs, dtype="u4")
        day_start = nb_obs.cumsum() - nb_obs
        gr = empty(nb_obs.sum(), dtype="u4")
        gr[:] = self.NOGROUP

        id_free = 1
        for i, j, ii, ij in results:
            gr_i = gr[slice(day_start[i], day_start[i] + nb_obs[i])]
            gr_j = gr[slice(day_start[j], day_start[j] + nb_obs[j])]
            # obs with no groups
            m = (gr_i[ii] == self.NOGROUP) * (gr_j[ij] == self.NOGROUP)
            nb_new = m.sum()
            gr_i[ii[m]] = gr_j[ij[m]] = arange(id_free, id_free + nb_new)
            id_free += nb_new
            # associate obs with no group with obs with group
            m = (gr_i[ii] != self.NOGROUP) * (gr_j[ij] == self.NOGROUP)
            gr_j[ij[m]] = gr_i[ii[m]]
            m = (gr_i[ii] == self.NOGROUP) * (gr_j[ij] != self.NOGROUP)
            gr_i[ii[m]] = gr_j[ij[m]]
            # case where 2 obs have a different group
            m = gr_i[ii] != gr_j[ij]
            if m.any():
                # Merge of group, ref over etu
                for i_, j_ in zip(ii[m], ij[m]):
                    g0, g1 = gr_i[i_], gr_j[j_]
                    apply_replace(gr, g0, g1)
            NETWORK_GROUPS.append((i, j, gr.copy()))
        return gr


# %%
# Movie period
t0 = (datetime(2005, 5, 1) - datetime(1950, 1, 1)).days
t1 = (datetime(2005, 6, 1) - datetime(1950, 1, 1)).days

# %%
# Get data from period and area
e = EddiesObservations.load_file(data.get_path("network_med.nc"))
e = e.extract_with_mask((e.time >= t0) * (e.time < t1)).extract_with_area(
    dict(llcrnrlon=25, urcrnrlon=35, llcrnrlat=31, urcrnrlat=37.5)
)
# %%
# Reproduce individual daily identification(for demonstration)
EDDIES_BY_DAYS = list()
for i, b0, b1 in e.iter_on("time"):
    EDDIES_BY_DAYS.append(e.index(i))
# need for display
e = EddiesObservations.concatenate(EDDIES_BY_DAYS)

# %%
# Run network building group to intercept every step
n = MyNetwork.from_eddiesobservations(EDDIES_BY_DAYS, window=7)
_ = n.group_observations(minimal_area=True)


# %%
def update(frame):
    i_current, i_match, gr = NETWORK_GROUPS[frame]
    current = EDDIES_BY_DAYS[i_current]
    x = flatten_line_matrix(current.contour_lon_e)
    y = flatten_line_matrix(current.contour_lat_e)
    current_contour.set_data(x, y)
    match = EDDIES_BY_DAYS[i_match]
    x = flatten_line_matrix(match.contour_lon_e)
    y = flatten_line_matrix(match.contour_lat_e)
    matched_contour.set_data(x, y)
    groups.set_array(gr)
    txt.set_text(f"Day {i_current} match with day {i_match}")
    s = 80 * ones(gr.shape)
    s[gr == 0] = 4
    groups.set_sizes(s)


# %%
# Anim
# ----
fig = plt.figure(figsize=(16, 9), dpi=50)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_aspect("equal"), ax.grid(), ax.set_xlim(26, 34), ax.set_ylim(31, 35.5)
cmap = ListedColormap(["gray", *e.COLORS[:-1]], name="from_list", N=30)
kw_s = dict(cmap=cmap, vmin=0, vmax=30)
groups = ax.scatter(e.lon, e.lat, c=NETWORK_GROUPS[0][2], **kw_s)
current_contour = ax.plot([], [], "k", lw=2, label="Current contour")[0]
matched_contour = ax.plot([], [], "r", lw=1, ls="--", label="Candidate contour")[0]
txt = ax.text(29, 35, "", fontsize=25)
ax.legend(fontsize=25)
ani = VideoAnimation(fig, update, frames=len(NETWORK_GROUPS), interval=220)

# %%
# Final Result
# ------------
fig = plt.figure(figsize=(16, 9))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_aspect("equal"), ax.grid(), ax.set_xlim(26, 34), ax.set_ylim(31, 35.5)
_ = ax.scatter(e.lon, e.lat, c=NETWORK_GROUPS[-1][2], **kw_s)
