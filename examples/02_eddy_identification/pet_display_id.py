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
# filled contour
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.03, 0.03, 0.90, 0.94])
ax.set_aspect("equal")
ax.set_xlim(0, 140)
ax.set_ylim(-80, 0)
a.display(ax, label="Anticyclonic", color="r", lw=1)
m = a.filled(ax, "amplitude", cmap="magma_r", vmin=0, vmax=.5)
plt.colorbar(m, cax=ax.figure.add_axes([0.95, 0.05, 0.01, 0.9]))
plt.show()
# %%
# Get general informations
print(a)
# %%
print(c)
