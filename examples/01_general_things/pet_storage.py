"""
How data is stored
==================

General information about eddies storage.

All eddies files have same structure with more or less field and a way of ordering.

There are 3 class of files:

- Eddies collections which contains a list of eddies without link between observations
- Track eddies collections which manage eddies when there are merged in trajectory
  (track field allow to separate each track)
- Network eddies collections which manage eddies when there are merged in network
  (track/segment field allow to separate observations)
"""

import py_eddy_tracker_sample

from py_eddy_tracker.data import get_path, get_remote_sample
from py_eddy_tracker.observations.network import NetworkObservations
from py_eddy_tracker.observations.observation import EddiesObservations, Table
from py_eddy_tracker.observations.tracking import TrackEddiesObservations

# %%
# Eddies could be store in 2 formats with same structures:
#
# - zarr (https://zarr.readthedocs.io/en/stable/), which allow efficiency in IO,...
# - NetCDF4 (https://unidata.github.io/netcdf4-python/), well-known format
#
# Each field are stored in column, each row corresponds at 1 observation,
# array field like contour/profile are 2D column.

# %%
# Eddies files (zarr or netcdf) could be loaded with `load_file` method:
eddies_collections = EddiesObservations.load_file(get_path("Cyclonic_20160515.nc"))
eddies_collections.field_table()
# offset and scale_factor are used only when data is stored in zarr or netCDF4

# %%
# Field access
# ------------
eddies_collections.amplitude

# %%
# Data matrix is a numpy ndarray
eddies_collections.obs
# %%
eddies_collections.obs.dtype


# %%
# Contour storage
# ---------------
# Contour are stored to fixed size for all, contour are resample with an algorithm before to be store in object


# %%
# Tracks
# ------
# Tracks add several field like:
#
# - track : ID which allow to identify path
# - observation_flag : True if it's an observation to filled a missing detection
# - observation_number : Age of eddies
# - cost_association : result of cost function which allow to associate the observation with eddy path
eddies_tracks = TrackEddiesObservations.load_file(
    py_eddy_tracker_sample.get_path("eddies_med_adt_allsat_dt2018/Cyclonic.zarr")
)
# In this example some field are removed like effective_contour_longitude, ... in order to save time for doc building
eddies_tracks.field_table()

# %%
# Network
# -------
# Network files use some specific field:
#
# - track :  ID of network (ID 0 are for lonely eddies/trash)
# - segment :  ID of path in network (from 0 to N)
# - previous_obs : Index of the previous observation in the full dataset, if -1 there are no previous observation
# - next_obs : Index of the next observation in the full dataset, if -1 there are no next observation
# - previous_cost : Result of cost_function (1 good <> 0 bad) with previous observation
# - next_cost : Result of cost_function (1 good <> 0 bad) with next observation
eddies_network = NetworkObservations.load_file(
    get_remote_sample(
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
