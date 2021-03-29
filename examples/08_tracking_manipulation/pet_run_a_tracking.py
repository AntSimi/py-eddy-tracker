"""
Track in python
===============

This example didn't replace EddyTracking, we remove check that application do and also postprocessing step.
"""

# %%

from py_eddy_tracker.data import get_remote_demo_sample
from py_eddy_tracker.featured_tracking.area_tracker import AreaTracker
from py_eddy_tracker.gui import GUI
from py_eddy_tracker.tracking import Correspondances

# %%
# Get remote data, we will keep only 180 first days,
# `get_remote_demo_sample` function is only to get demo dataset, in your own case give a list of identification filename
# and don't mix cyclonic and anticyclonic files.
file_objects = get_remote_demo_sample(
    "eddies_med_adt_allsat_dt2018/Anticyclonic_2010_2011_2012"
)[:180]

# %%
# We run a traking with a tracker which use contour overlap
c = Correspondances(datasets=file_objects, class_method=AreaTracker, virtual=3)
c.track()
c.prepare_merging()
# We have now an eddy object
eddies_area_tracker = c.merge(raw_data=False)
eddies_area_tracker.virtual[:] = eddies_area_tracker.time == 0
eddies_area_tracker.filled_by_interpolation(eddies_area_tracker.virtual == 1)

# %%
# We run a traking with default tracker
c = Correspondances(datasets=file_objects, virtual=3)
c.track()
c.prepare_merging()
eddies_default_tracker = c.merge(raw_data=False)
eddies_default_tracker.virtual[:] = eddies_default_tracker.time == 0
eddies_default_tracker.filled_by_interpolation(eddies_default_tracker.virtual == 1)

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
