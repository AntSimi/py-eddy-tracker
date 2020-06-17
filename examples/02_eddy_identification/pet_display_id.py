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
# Plot
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
ax.set_aspect("equal")
ax.set_xlim(0, 360)
ax.set_ylim(-80, 80)
a.display(ax, label="Anticyclonic", color="r", lw=1)
c.display(ax, label="Cyclonic", color="b", lw=1)
ax.legend(loc="upper right")
