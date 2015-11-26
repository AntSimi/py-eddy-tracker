# -*- coding: utf-8 -*-
from netCDF4 import Dataset
from . import VAR_DESCR
import logging


class GlobalTracking(object):
    """
    """

    def __init__(self, eddy, date):
        self.date = date
        self.eddy = eddy

    @property
    def sign_type(self):
        return self.eddy.sign_type

    def create_variable(self, handler_nc, kwargs_variable,
                        attr_variable, data, scale_factor=None):
        var = handler_nc.createVariable(
            zlib=True,
            complevel=1,
            **kwargs_variable)
        for attr, attr_value in attr_variable.iteritems():
            var.setncattr(attr, attr_value)
            var[:] = data
            var.set_auto_maskandscale(False)
            if scale_factor is not None:
                var.scale_factor = scale_factor
            
        try:
            var.setncattr('min', var[:].min())
            var.setncattr('max', var[:].max())
        except ValueError:
            logging.warn('Data is empty')

    def write_netcdf(self):
        """Write a netcdf with eddy
        Write eddy property data to tracks.
        """
        eddy_size = len(self.eddy.tmp_observations)
        print dir(self)
        exit()
        filename = '%s_%s.nc' % (self.sign_type, self.date.strftime('%Y%m%d'))
        with Dataset(filename, 'w', format='NETCDF4') as h_nc:
            logging.info('Create intermediary file %s', filename)
            # Create dimensions
            logging.debug('Create Dimensions "Nobs" : %d', eddy_size)
            h_nc.createDimension('Nobs', eddy_size)
            # Iter on variables to create:
            for name, _ in self.eddy.tmp_observations.dtype:
                logging.debug('Create Variable %s', VAR_DESCR[name]['nc_name'])
                self.create_variable(
                    h_nc,
                    dict(varname=VAR_DESCR[name]['nc_name'],
                         datatype=VAR_DESCR[name]['nc_type'],
                         dimensions=VAR_DESCR[name]['nc_dims']),
                    VAR_DESCR[name]['nc_attr'],
                    self.eddy.tmp_observations.obs[name],
                    scale_factor=None if 'scale_factor' not in VAR_DESCR[name] else VAR_DESCR[name]['scale_factor'])

            # Add cyclonic information
            self.create_variable(
                h_nc,
                dict(varname=VAR_DESCR['type_cyc']['nc_name'],
                     datatype=VAR_DESCR['type_cyc']['nc_type'],
                     dimensions=VAR_DESCR['type_cyc']['nc_dims']),
                VAR_DESCR['type_cyc']['nc_attr'],
                -1 if self.sign_type == 'Cyclonic' else 1)

    def read_tracks(self):
        """
        Read and sort the property data for returning to the
        Eddy object
        """
#         with Dataset('Anticyclonic_20140312.nc') as nc:
#             tracklengths = nc.variables['track_lengths'][:]
#             for t, tracklen in enumerate(tracklengths):

#                 varname = 'track_%s' % np.str(t).zfill(4)
#                 track = nc.variables[varname][:]
#                 dayzero = track[0].astype(bool)
#                 saved2nc = track[1].astype(bool)
#                 save_extras = track[2].astype(bool)

#                 inds = np.arange(3, 3 + (tracklen * 8), tracklen)
#                 lon = track[inds[0]:inds[1]]
#                 lat = track[inds[1]:inds[2]]
#                 amplitude = track[inds[2]:inds[3]]
#                 radius_s = track[inds[3]:inds[4]]
#                 radius_e = track[inds[4]:inds[5]]
#                 uavg = track[inds[5]:inds[6]]
#                 teke = track[inds[6]:inds[7]]
#                 ocean_time = track[inds[7]:]

#             properties = nc.variables['track_0000'][:]
