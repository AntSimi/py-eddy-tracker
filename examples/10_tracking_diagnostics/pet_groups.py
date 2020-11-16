"""
Groups distribution
==================

"""
import py_eddy_tracker_sample
from matplotlib import pyplot as plt
from numpy import arange, ones, percentile

from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# %%
# Load an experimental med atlas over a period of 26 years (1993-2019)
a = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Anticyclonic.zarr")
)

# %%
# Group distribution
groups = dict()
bins_time = [10, 20, 30, 60, 90, 180, 360, 100000]
for t0, t1 in zip(bins_time[:-1], bins_time[1:]):
    groups[f"lifetime_{t0}_{t1}"] = lambda dataset, t0=t0, t1=t1: (
        dataset.lifetime >= t0
    ) * (dataset.lifetime < t1)
bins_percentile = arange(0, 100.0001, 5)


# %%
# Function to build stats
def stats_compilation(dataset, groups, field, bins, filter=None):
    datas = dict(ref=dataset.bins_stat(field, bins=bins, mask=filter)[1], y=dict())
    for k, index in groups.items():
        i = dataset.merge_indexs(filter, index)
        x, datas["y"][k] = dataset.bins_stat(field, bins=bins, mask=i)
    datas["x"], datas["bins"] = x, bins
    return datas


def plot_stats(ax, bins, x, y, ref, box=False, cmap=None, percentiles=None, **kw):
    base, ref = ones(x.shape) * 100.0, ref / 100.0
    x = arange(bins.shape[0]).repeat(2)[1:-1] if box else x
    y0 = base
    if cmap is not None:
        cmap, nb_groups = plt.get_cmap(cmap), len(y)
    keys = tuple(y.keys())
    for i, k in enumerate(keys[::-1]):
        y1 = y0 - y[k] / ref
        args = (y0.repeat(2), y1.repeat(2)) if box else (y0, y1)
        if cmap is not None:
            kw["color"] = cmap(1 - i / (nb_groups - 1))
        ax.fill_between(x, *args, label=k, **kw)
        y0 = y1
    if percentiles:
        for b in bins:
            ax.axvline(b, **percentiles)


# %%
# Speed radius by track period
stats = stats_compilation(
    a, groups, "radius_s", percentile(a.radius_s, bins_percentile)
)
fig = plt.figure()
ax = fig.add_subplot(111)
plot_stats(ax, **stats, cmap="magma", percentiles=dict(color="gray", ls="-.", lw=0.4))
ax.set_xlabel("Speed radius (m)"), ax.set_ylabel("% of class"), ax.set_ylim(0, 100)
ax.grid(), ax.legend()

# %%
# Amplitude by track period
stats = stats_compilation(
    a, groups, "amplitude", percentile(a.amplitude, bins_percentile)
)
fig = plt.figure()
ax = fig.add_subplot(111)
plot_stats(ax, **stats, cmap="magma")
ax.set_xlabel("Amplitude (m)"), ax.set_ylabel("% of class"), ax.set_ylim(0, 100)
ax.grid(), ax.legend()
