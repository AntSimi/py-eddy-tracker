"""
Display identification
======================

"""

from matplotlib import pyplot as plt
from py_eddy_tracker.observations.observation import EddiesObservations
from py_eddy_tracker import data

# %%
# Load detection files
a = EddiesObservations.load_file(data.get_path("Anticyclonic_20190223.nc"))
c = EddiesObservations.load_file(data.get_path("Cyclonic_20190223.nc"))

# %%
# filled contour with amplitude field
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.03, 0.03, 0.90, 0.94])
ax.set_aspect("equal")
ax.set_xlim(0, 140)
ax.set_ylim(-80, 0)
kwargs = dict(extern_only=True, color="k", lw=1)
a.display(ax, **kwargs), c.display(ax, **kwargs)
a.filled(ax, "amplitude", cmap="magma_r", vmin=0, vmax=.5)
m = c.filled(ax, "amplitude", cmap="magma_r", vmin=0, vmax=.5)
colorbar = plt.colorbar(m, cax=ax.figure.add_axes([0.95, 0.05, 0.01, 0.9]))
colorbar.set_label('Amplitude (m)')

plt.show()

# %%
# draw contour
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
ax.set_aspect("equal")
ax.set_xlim(0, 360)
ax.set_ylim(-80, 80)
a.display(ax, label="Anticyclonic", color="r", lw=1)
c.display(ax, label="Cyclonic", color="b", lw=1)
ax.legend(loc="upper right")

# %%
# Get general informations
print(a)
# %%
print(c)
