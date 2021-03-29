"""
Track animation
===============

Run in a terminal this script, which allow to watch eddy evolution

"""
import py_eddy_tracker_sample

from py_eddy_tracker.appli.gui import Anim
from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# %%
# Load experimental atlas, and we select one eddy
a = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_demo_path(
        "eddies_med_adt_allsat_dt2018/Anticyclonic.zarr"
    )
)
# We get only 300 first step to save time of documentation builder
eddy = a.extract_ids([9672]).index(slice(0, 300))

# %%
# Run animation
# Key shortcut :
#  * Escape => exit
#  * SpaceBar => pause
#  * left arrow => t - 1
#  * right arrow => t + 1
#  * \+ => speed increase of 10 %
#  * \- => speed decrease of 10 %
a = Anim(eddy, sleep_event=1e-10, intern=True, figsize=(8, 3.5), cmap="viridis")
a.txt.set_position((17, 34.6))
a.ax.set_xlim(16.5, 23)
a.ax.set_ylim(34.5, 37)
a.show(infinity_loop=False)
