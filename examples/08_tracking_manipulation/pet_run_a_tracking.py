"""
Track in python
===============

This example didn't replace EddyTracking, we remove check that application do and also postprocessing step.
"""

# %%

from py_eddy_tracker.data import get_remote_sample
from py_eddy_tracker.tracking import Correspondances
from py_eddy_tracker.featured_tracking.area_tracker import AreaTracker
from numpy import where, empty
from py_eddy_tracker.gui import GUI


# %%
# Function to have track with contiguous longitude
def wrap_longitude(eddies):
    lon = eddies.longitude
    first = where(eddies.obs["n"] == 0)[0]
    nb_obs = empty(first.shape, dtype="u4")
    nb_obs[:-1] = first[1:] - first[:-1]
    nb_obs[-1] = lon.shape[0] - first[-1]
    lon0 = (lon[first] - 180).repeat(nb_obs)
    lon[:] = (lon - lon0) % 360 + lon0


# %%
# Get remote data, we will keep only 180 first days
file_objects = get_remote_sample(
    "eddies_med_adt_allsat_dt2018/Anticyclonic_2010_2011_2012"
)[:180]

# %%
# We run a traking with a tracker which use contour overlap
c = Correspondances(datasets=file_objects, class_method=AreaTracker)
c.track()
c.prepare_merging()
# We have now an eddy object
eddies_area_tracker = c.merge(raw_data=False)
wrap_longitude(eddies_area_tracker)

# %%
# We run a traking with default tracker
c = Correspondances(datasets=file_objects)
c.track()
c.prepare_merging()
eddies_default_tracker = c.merge(raw_data=False)
wrap_longitude(eddies_default_tracker)

# %%
# Start GUI to compare tracking
g = GUI(
    Acyc_area_tracker=eddies_area_tracker, Acyc_default_tracker=eddies_default_tracker
)
g.now = 22000
g.bbox = 0, 9, 36, 40
g.adjust()
g.show()

# %%
# Start GUI with area tracker
g = GUI(Acyc_area_tracker=eddies_area_tracker)
g.now = 22000
g.bbox = 0, 9, 36, 40
g.adjust()
g.show()

# %%
# Start GUI with default one
g = GUI(Acyc_default_tracker=eddies_default_tracker)
g.now = 22000
g.bbox = 0, 9, 36, 40
g.adjust()
g.show()
