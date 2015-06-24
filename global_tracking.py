# %run global_tracking.py

from netCDF4 import Dataset
import numpy as np
import logging


class GlobalTracking(object):
    """

    """

    VAR_DESCR = dict(
        time=dict(
            attr_name='new_time_tmp',
            nc_name='j1',
            nc_type='int32',
            nc_dims=('Nobs',),
            nc_attr=dict(
                long_name='Julian date',
                units='days',
                description='date of this observation',
                reference=2448623,
                reference_description="Julian date on Jan 1, 1992",
                )
            ),
        type_cyc=dict(
            attr_name=None,
            nc_name='cyc',
            nc_type='byte',
            nc_dims=('Nobs',),
            nc_attr=dict(
                long_name='cyclonic',
                units='boolean',
                description='cyclonic -1; anti-cyclonic +1',
                )
            ),
        lon=dict(
            attr_name='new_lon_tmp',
            nc_name='lon',
            nc_type='float32',
            nc_dims=('Nobs',),
            nc_attr=dict(
                units='deg. longitude',
                )
            ),
        lat=dict(
            attr_name='new_lat_tmp',
            nc_name='lat',
            nc_type='float32',
            nc_dims=('Nobs',),
            nc_attr=dict(
                units='deg. latitude',
                )
            ),
        amplitude=dict(
            attr_name='new_amp_tmp',
            nc_name='A',
            nc_type='float32',
            nc_dims=('Nobs',),
            nc_attr=dict(
                long_name='amplitude',
                units='cm',
                description='magnitude of the height difference between the '
                            'extremum of SSH within the eddy and the SSH '
                            'around the contour defining the eddy perimeter',
                )
            ),
        rayon=dict(
            attr_name=None,
            nc_name='L',
            nc_type='float32',
            nc_dims=('Nobs',),
            nc_attr=dict(
                long_name='speed radius scale',
                units='km',
                description='radius of a circle whose area is equal to that '
                            'enclosed by the contour of maximum circum-average'
                            ' speed',
                )
            ),
        speed_radius=dict(
            attr_name='new_uavg_tmp',
            nc_name='U',
            nc_type='float32',
            nc_dims=('Nobs',),
            nc_attr=dict(
                long_name='maximum circum-averaged speed',
                units='cm/sec',
                description='average speed of the contour defining the radius '
                            'scale L',
                )
            ),
        eke=dict(
            attr_name='new_teke_tmp',
            nc_name='Teke',
            nc_type='float32',
            nc_dims=('Nobs',),
            nc_attr=dict(
                long_name='sum EKE within contour Ceff',
                units='m^2/sec^2',
                description='sum of eddy kinetic energy within contour '
                            'defining the effective radius',
                )
            ),
        radius_e=dict(
            attr_name='new_radii_e_tmp',
            nc_name='radius_e',
            nc_type='float32',
            nc_dims=('Nobs',),
            nc_attr=dict(
                long_name='effective radius scale',
                units='km',
                description='effective eddy radius',
                )
            ),
        )

    def __init__(self, eddy, ymd_str):
        """
        """
        self.ymd_str = ymd_str
        self.eddy = eddy

    @property
    def sign_type(self):
        return self.eddy.SIGN_TYPE

    def create_netcdf(self):
        """
        """
        eddy_size = None
        for key in self.VAR_DESCR:
            attr_name = self.VAR_DESCR[key]['attr_name']
            if attr_name is not None and hasattr(self.eddy, attr_name):
                eddy_size = len(getattr(self.eddy, attr_name))
                break

        filename = '%s_%s.nc' % (self.sign_type, self.ymd_str)
        with Dataset(filename, 'w', format='NETCDF4') as h_nc:
            logging.info('Create intermediary file %s', filename)
            # Create dimensions
            logging.debug('Create Dimensions "Nobs"')
            h_nc.createDimension('Nobs', eddy_size)
            # Iter on variables to create:
            for key, value in self.VAR_DESCR.iteritems():
                attr_name = value['attr_name']
                if attr_name is None or not hasattr(self.eddy, attr_name):
                    continue
                logging.debug('Create Variable %s',
                              value['nc_name'])
                var = h_nc.createVariable(
                    varname=value['nc_name'],
                    datatype=value['nc_type'],
                    dimensions=value['nc_dims'],
                    zlib=True,
                    complevel=1,
                    )
                for attr, attr_value in value['nc_attr'].iteritems():
                    var.setncattr(attr, attr_value)

                var[:] = getattr(self.eddy, attr_name)

    def write_tracks(self):
        """
        """
#         t = 0
#         print self.old_lon
#         for track in self.tracklist:
#             tracklen = len(track.lon)
#             if track.alive:
#                 properties = np.hstack((
#                     track.dayzero, track.saved2nc,
#                     track.save_extras, track.lon, track.lat,
#                     track.amplitude, track.radius_s,
#                     track.radius_e, track.uavg,
#                     track.teke, track.ocean_time))
# 
#                 with Dataset('%s_%s.nc' % (self.sign_type,
#                                            self.ymd_str), 'a') as nc:
#                     varname = 'track_%s' % np.str(t).zfill(4)
#                     nc.variables[varname][:] = properties
#                     nc.variables['track_lengths'][t] = tracklen
#                 t += 1

    def read_tracks(self):
        """
        Read and sort the property data for returning to the
        Eddy object
        """
        with Dataset('Anticyclonic_20140312.nc') as nc:
            tracklengths = nc.variables['track_lengths'][:]
            for t, tracklen in enumerate(tracklengths):

                varname = 'track_%s' % np.str(t).zfill(4)
                track = nc.variables[varname][:]
                dayzero = track[0].astype(bool)
                saved2nc = track[1].astype(bool)
                save_extras = track[2].astype(bool)

                inds = np.arange(3, 3 + (tracklen * 8), tracklen)
                lon = track[inds[0]:inds[1]]
                lat = track[inds[1]:inds[2]]
                amplitude = track[inds[2]:inds[3]]
                radius_s = track[inds[3]:inds[4]]
                radius_e = track[inds[4]:inds[5]]
                uavg = track[inds[5]:inds[6]]
                teke = track[inds[6]:inds[7]]
                ocean_time = track[inds[7]:]

            properties = nc.variables['track_0000'][:]
