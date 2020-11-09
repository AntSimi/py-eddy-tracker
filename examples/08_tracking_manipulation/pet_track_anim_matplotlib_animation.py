"""
Track animation with standard matplotlib
========================================

Run in a terminal this script, which allow to watch eddy evolution

"""
import py_eddy_tracker_sample
from matplotlib.animation import FuncAnimation
from numpy import arange

from py_eddy_tracker.appli.gui import Anim
from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# %%
# Load experimental atlas, and we select one eddy
a = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Anticyclonic.zarr")
)
eddy = a.extract_ids([9672])

# %%
# Run animation
a = Anim(eddy, intern=True, figsize=(8, 3.5), cmap="magma_r", nb_step=6)
a.txt.set_position((17, 34.6))
a.ax.set_xlim(16.5, 23)
a.ax.set_ylim(34.5, 37)

# arguments to get full animation
# kwargs = dict(frames=arange(*a.period), interval=50)
# arguments to reduce compute cost for doucmentation, we display only every 10 days
kwargs = dict(frames=arange(*a.period)[200:800:10], save_count=60, interval=200)

ani = FuncAnimation(a.fig, a.func_animation, **kwargs)
