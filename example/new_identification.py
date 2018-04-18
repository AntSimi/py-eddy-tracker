from netCDF4 import Dataset
from py_eddy_tracker.dataset.grid import RegularGridDataset, UnRegularGridDataset
import logging
logging.basicConfig(level=logging.DEBUG)

# h = UnRegularGridDataset('NEATL36_1d_grid2D_20140515-20140515_short.nc', 'nav_lon', 'nav_lat')
# h.high_filter('sossheig', 20, 10)
# h.add_uv('sossheig')
# h.write('unregular.nc')

h = RegularGridDataset('/data/adelepoulle/Test/Test_eddy/20180417_eddy_tracker_validation_object_oriented/nrt_global_allsat_phy_l4_20180409_20180415.nc', 'longitude', 'latitude')
h.high_filter('sla', 10, 5)
anticyclonic, cyclonic = h.eddy_identification('sla', 'ugos', 'vgos', 0.0025)
print(len(anticyclonic))
print(len(cyclonic))
with Dataset('/tmp/a.nc', 'w') as h:
    anticyclonic.to_netcdf(h)
with Dataset('/tmp/c.nc', 'w') as h:
    cyclonic.to_netcdf(h)

# h = UnRegularGridDataset('/tmp/t.nc', 'nav_lon', 'nav_lat')
# eddies = h.eddy_identification('sossheig', step=0.005)
