"""
Network basic manipulation
==========================
"""

from matplotlib import pyplot as plt

from py_eddy_tracker import data
from py_eddy_tracker.observations.network import NetworkObservations
from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# %%
# Load data
# ---------
# Load data where observations are put in same network but no segmentation
e = TrackEddiesObservations.load_file(data.get_path("c568803.nc"))
# FIXME : Must be rewrote
e.lon[:] = (e.lon + 180) % 360 - 180
e.contour_lon_e[:] = ((e.contour_lon_e.T - e.lon + 180) % 360 - 180 + e.lon).T
e.contour_lon_s[:] = ((e.contour_lon_s.T - e.lon + 180) % 360 - 180 + e.lon).T
##############
n = NetworkObservations.from_split_network(e, e.split_network(intern=False, window=5))

# %%
# Timeline
# --------

# %%
# Display timeline
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
_ = n.display_timeline(ax)

# %%
# Display timeline without event
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
_ = n.display_timeline(ax, event=False)

# %%
# Timeline by latitude mean
# -------------------------
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
_ = n.display_timeline(ax, field="latitude")

# %%
# Timeline by radius mean
# -----------------------
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
_ = n.display_timeline(ax, field="radius_e")

# %%
# Timeline by latitude
# --------------------
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.05, 0.92, 0.92])
_ = n.display_timeline(ax, field="lat", method="all")

# %%
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.05, 0.92, 0.92])
n_copy = n.copy()
n_copy.median_filter(15, "time", "latitude")
_ = n_copy.display_timeline(ax, field="lat", method="all")

# %%
# Parameters timeline
# -------------------
kw = dict(s=25, cmap="Spectral_r", zorder=10)
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.04, 0.54, 0.90, 0.44])
m = n.scatter_timeline(ax, "radius_e", factor=1e-3, vmin=50, vmax=150, **kw)
cb = plt.colorbar(
    m["scatter"], cax=fig.add_axes([0.95, 0.54, 0.01, 0.44]), orientation="vertical"
)
cb.set_label("Effective radius (km)")

ax = fig.add_axes([0.04, 0.04, 0.90, 0.44])
m = n.scatter_timeline(ax, "amplitude", factor=100, vmin=0, vmax=15, **kw)
cb = plt.colorbar(
    m["scatter"], cax=fig.add_axes([0.95, 0.04, 0.01, 0.44]), orientation="vertical"
)
cb.set_label("Amplitude (cm)")

# %%
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.06, 0.90, 0.88])
m = n.scatter_timeline(ax, "speed_average", factor=100, vmin=0, vmax=40, **kw)
cb = plt.colorbar(
    m["scatter"], cax=fig.add_axes([0.95, 0.04, 0.01, 0.92]), orientation="vertical"
)
cb.set_label("Maximum speed (cm/s)")

# %%
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.06, 0.90, 0.88])
m = n.scatter_timeline(ax, "radius_s", factor=1e-3, vmin=20, vmax=100, **kw)
cb = plt.colorbar(
    m["scatter"], cax=fig.add_axes([0.95, 0.04, 0.01, 0.92]), orientation="vertical"
)
cb.set_label("Speed radius (km)")

# %%
# Remove dead branch
# ------------------
# Remove all tiny segment with less than N obs which didn't join two segments
#
# .. warning::
#     Must be explore, no solution to solve all the case

n_clean = n.remove_dead_branch(nobs=51)
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.04, 0.54, 0.90, 0.40])
ax.set_title(f"Original network ({n.infos()})")
n.display_timeline(ax)
ax = fig.add_axes([0.04, 0.04, 0.90, 0.40])
ax.set_title(f"Clean network ({n_clean.infos()})")
_ = n_clean.display_timeline(ax)

# %%
# Keep close relative
# -------------------
# First choose an observation in the network
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.06, 0.90, 0.88])
n.display_timeline(ax)
i = 1100
obs_args = n.time[i], n.segment[i]
obs_kw = dict(color="black", markersize=30, marker=".")
_ = ax.plot(*obs_args, **obs_kw)

# %%
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.06, 0.90, 0.88])
m = n.scatter_timeline(
    ax, n.obs_relative_order(i), vmin=-1.5, vmax=6.5, cmap=plt.get_cmap("jet", 8), s=10
)
ax.plot(*obs_args, **obs_kw)
cb = plt.colorbar(
    m["scatter"], cax=fig.add_axes([0.95, 0.04, 0.01, 0.92]), orientation="vertical"
)
cb.set_label("Relative order")
# %%
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.06, 0.90, 0.88])
close_to_i1 = n.relative(i, order=1)
ax.set_title(f"Close segments ({close_to_i1.infos()})")
_ = close_to_i1.display_timeline(ax)
# %%
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.06, 0.90, 0.88])
close_to_i2 = n.relative(i, order=2)
ax.set_title(f"Close segments ({close_to_i2.infos()})")
_ = close_to_i2.display_timeline(ax)
# %%
fig = plt.figure(figsize=(15, 5))
ax = fig.add_axes([0.04, 0.06, 0.90, 0.88])
close_to_i3 = n.relative(i, order=3)
ax.set_title(f"Close segments ({close_to_i3.infos()})")
_ = close_to_i3.display_timeline(ax)

# %%
# Display track on map
# --------------------
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.04, 0.06, 0.94, 0.88], projection="full_axes")
close_to_i2.plot(ax)
ax.set_xlim(-13, 20), ax.set_ylim(-36.5, -20), ax.grid()
ax = fig.add_axes([0.08, 0.67, 0.55, 0.3])
_ = close_to_i2.display_timeline(ax, field="latitude")


# %%
# Get merging event
# -----------------
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.04, 0.06, 0.90, 0.88], projection="full_axes")
merging = close_to_i2.merging_event()
merging.display(ax)
ax.set_xlim(-10, 20), ax.set_ylim(-37, -21), ax.grid()
merging

# %%
# Get spliting event
# ------------------
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.04, 0.06, 0.90, 0.88], projection="full_axes")
spliting = close_to_i2.spliting_event()
spliting.display(ax)
ax.set_xlim(-10, 20), ax.set_ylim(-37, -21), ax.grid()
spliting

# %%
# Get birth event
# ------------------
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.04, 0.06, 0.90, 0.88], projection="full_axes")
birth = close_to_i2.birth_event()
birth.display(ax)
ax.set_xlim(-10, 20), ax.set_ylim(-37, -21), ax.grid()
birth

# %%
# Get death event
# ------------------
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.04, 0.06, 0.90, 0.88], projection="full_axes")
death = close_to_i2.death_event()
death.display(ax)
ax.set_xlim(-10, 20), ax.set_ylim(-37, -21), ax.grid()
death
