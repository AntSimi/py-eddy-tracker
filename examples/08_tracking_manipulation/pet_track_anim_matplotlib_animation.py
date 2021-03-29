"""
Track animation with standard matplotlib
========================================

Run in a terminal this script, which allow to watch eddy evolution

"""
import re

import py_eddy_tracker_sample
from matplotlib.animation import FuncAnimation
from numpy import arange

from py_eddy_tracker.appli.gui import Anim
from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# sphinx_gallery_thumbnail_path = '_static/no_image.png'


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
# Load experimental atlas, and we select one eddy
a = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_demo_path(
        "eddies_med_adt_allsat_dt2018/Anticyclonic.zarr"
    )
)
eddy = a.extract_ids([9672])

# %%
# Run animation
a = Anim(eddy, intern=True, figsize=(8, 3.5), cmap="magma_r", nb_step=5, dpi=50)
a.txt.set_position((17, 34.6))
a.ax.set_xlim(16.5, 23)
a.ax.set_ylim(34.5, 37)

# arguments to get full animation
kwargs = dict(frames=arange(*a.period)[300:800], interval=90)

ani = VideoAnimation(a.fig, a.func_animation, **kwargs)
