"""
How data is stored
==================

General information about eddies storage.

All files have the same structure, with more or less fields and possible different order.

There are 3 class of files:

- **Eddies collections** : contain a list of eddies without link between them
- **Track eddies collections** :
  manage eddies associated in trajectories, the ```track``` field allows to separate each trajectory
- **Network eddies collections** :
  manage eddies associated in networks, the ```track``` and ```segment``` fields allow to separate observations
"""

import py_eddy_tracker_sample

from py_eddy_tracker.data import get_demo_path, get_remote_demo_sample
from py_eddy_tracker.observations.network import NetworkObservations
from py_eddy_tracker.observations.observation import EddiesObservations, Table
from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# %%
# Eddies can be stored in 2 formats with the same structure:
#
# - zarr (https://zarr.readthedocs.io/en/stable/), which allow efficiency in IO,...
# - NetCDF4 (https://unidata.github.io/netcdf4-python/), well-known format
#
# Each field are stored in column, each row corresponds at 1 observation,
# array field like contour/profile are 2D column.

# %%
# Eddies files (zarr or netcdf) can be loaded with ```load_file``` method:
eddies_collections = EddiesObservations.load_file(get_demo_path("Cyclonic_20160515.nc"))
eddies_collections.field_table()
# offset and scale_factor are used only when data is stored in zarr or netCDF4

# %%
# Field access
# ------------
# To access the total field, here ```amplitude```
eddies_collections.amplitude

# To access only a specific part of the field
eddies_collections.amplitude[4:15]

# %%
# Data matrix is a numpy ndarray
eddies_collections.obs
# %%
eddies_collections.obs.dtype


# %%
# Contour storage
# ---------------
# All contours are stored on the same number of points, and are resampled if needed with an algorithm to be stored as objects

# %%
# Trajectories
# ------------
# Tracks eddies collections add several fields :
#
# - **track** : Trajectory number
# - **observation_flag** : Flag indicating if the value is interpolated between two observations or not
#   (0: observed eddy, 1: interpolated eddy)"
# - **observation_number** : Eddy temporal index in a trajectory, days starting at the eddy first detection
# - **cost_association** : result of the cost function to associate the eddy with the next observation
eddies_tracks = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_demo_path("eddies_med_adt_allsat_dt2018/Cyclonic.zarr")
)
# In this example some fields are removed (effective_contour_longitude,...) in order to save time for doc building
eddies_tracks.field_table()

# %%
# Networks
# --------
# Network files use some specific fields :
#
# - track :  ID of network (ID 0 correspond to lonely eddies)
# - segment :  ID of a segment within a network (from 1 to N)
# - previous_obs : Index of the previous observation in the full dataset,
#   if -1 there are no previous observation (the segment starts)
# - next_obs : Index of the next observation in the full dataset, if -1 there are no next observation (the segment ends)
# - previous_cost : Result of the cost function (1 is a good association, 0 is bad) with previous observation
# - next_cost : Result of the cost function (1 is a good association, 0 is bad) with next observation
eddies_network = NetworkObservations.load_file(
    get_remote_demo_sample(
        "eddies_med_adt_allsat_dt2018_err70_filt500_order1/Anticyclonic_network.nc"
    )
)
eddies_network.field_table()

# %%
sl = slice(70, 100)
Table(
    eddies_network.network(651).obs[sl][
        [
            "time",
            "track",
            "segment",
            "previous_obs",
            "previous_cost",
            "next_obs",
            "next_cost",
        ]
    ]
)

# %%
# Networks are ordered by increasing network number (`track`), then increasing segment number, then increasing time
