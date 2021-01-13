"""
Network basic manipulation
==========================
"""

from matplotlib import pyplot as plt

import py_eddy_tracker.gui
from py_eddy_tracker import data
from py_eddy_tracker.observations.network import NetworkObservations
from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# %%
# Load data
# ---------
# Load data where observations are put in same network but no segmentation
e = TrackEddiesObservations.load_file(data.get_path("c568803.nc"))
n = NetworkObservations.from_split_network(e, e.split_network(intern=False, window=5))

# %%
# Timeline
# --------

# %%
# Display timeline
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
n.display_timeline(ax)

# %%
# Display timeline with event
fig = plt.figure(figsize=(15, 8))
ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
n.display_timeline(ax, event=True)
plt.show()

# %%
# Keep close relative
# -------------------
# First choose an observation in the network
