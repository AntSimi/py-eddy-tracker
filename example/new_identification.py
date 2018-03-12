from py_eddy_tracker.dataset.grid import RegularGridDataset, UnRegularGridDataset
import logging
logging.basicConfig(level=logging.DEBUG)

# h = UnRegularGridDataset('NEATL36_1d_grid2D_20140515-20140515_short.nc', 'nav_lon', 'nav_lat')
# h.high_filter('sossheig', 20, 10)
# h.add_uv('sossheig')
# h.write('unregular.nc')

# h = RegularGridDataset('msla_h_20170425T000000_20170425T000000.nc', 'NbLongitudes', 'NbLatitudes')
# h.high_filter('Grid_0001', 10, 5)
# h.eddy_identification('Grid_0001', 0.5)

h = UnRegularGridDataset('/home/tildou/ant_work/data/unregular.nc', 'nav_lon', 'nav_lat')
h.eddy_identification('sossheig')

# h = RegularGridDataset('madt_h_20170425T000000_20170425T000000.nc', 'NbLongitudes', 'NbLatitudes')

# h.add_uv('Grid_0001')
# h.write('regular.nc')

