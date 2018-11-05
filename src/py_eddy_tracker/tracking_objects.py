# -*- coding: utf-8 -*-

"""
===========================================================================
This file is part of py-eddy-tracker.

    py-eddy-tracker is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    py-eddy-tracker is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with py-eddy-tracker.  If not, see <http://www.gnu.org/licenses/>.

Copyright (c) 2014-2017 by Evan Mason and Antoine Delepoulle
Email: emason@imedea.uib-csic.es
===========================================================================


tracking_objects.py

Version 3.0.0


===========================================================================


"""
from netCDF4 import Dataset
from . import VAR_DESCR
from numpy import arange, int_, interp, float64, array
import logging
from .observations import EddiesObservations


def nearest(lon_pt, lat_pt, lon2d, lat2d):
    """
    Return the nearest i, j point to a given lon, lat point
    in a lat/lon grid
    """
    try:
        i_x = int_(interp(lon_pt,
                          lon2d,
                          arange(len(lon2d)),
                          left=0, right=-1))
        i_y = int_(interp(lat_pt,
                          lat2d,
                          arange(len(lat2d)),
                          left=0, right=-1))
    except ValueError:
        logging.error('%s, %s', lat2d, lat_pt)
        raise ValueError()
    return i_x, i_y


class IdentificationList(object):
    """
    Class that holds list of eddy identify:
    """

    def __init__(self, sign_type, grd, date, **kwargs):
        """
        Initialise the list 'tracklist'
        """
        self._grd = grd
        self.date = date
        self.sign_type = sign_type

        self.diagnostic_type = kwargs.get('DIAGNOSTIC_TYPE', 'SLA')
        self.the_domain = kwargs.get('THE_DOMAIN', 'Regional')
        self.track_extra_variables = kwargs.get('TRACK_EXTRA_VARIABLES', [])
        if self.track_extra_variables is None:
            self.track_extra_variables = []
        array_properties = kwargs.get('TRACK_ARRAY_VARIABLES', dict())
        if array_properties is None:
            array_properties = dict()
        self.track_array_variables_sampling = array_properties.get('NB_SAMPLES', 0)
        self.track_array_variables = array_properties.get('VARIABLES', [])
        self.smoothing = kwargs.get('SMOOTHING', True)
        self.max_local_extrema = kwargs.get('MAX_LOCAL_EXTREMA', 1)
        self.interp_method = kwargs.get('INTERP_METHOD', 'RectBivariate')
        # NOTE: '.copy()' suffix is essential here
        self.contour_parameter = kwargs.get('CONTOUR_PARAMETER').copy()
        self.interval = self.contour_parameter[1] - self.contour_parameter[0]
        if 'Cyclonic' in sign_type:
            self.contour_parameter *= -1
        self.shape_error = kwargs.get('SHAPE_ERROR', 55.)
        self.radmin = float64(kwargs.get('RADMIN', 0.4))
        self.radmax = float64(kwargs.get('RADMAX', 4.461))
        self.ampmin = float64(kwargs.get('AMPMIN', 1.))
        self.ampmax = float64(kwargs.get('AMPMAX', 150.))
        self.evolve_amp_min = float64(kwargs.get('EVOLVE_AMP_MIN', .0005))
        self.evolve_amp_max = float64(kwargs.get('EVOLVE_AMP_MAX', 500))
        self.evolve_area_min = float64(kwargs.get('EVOLVE_AREA_MIN', .0005))
        self.evolve_area_max = float64(kwargs.get('EVOLVE_AREA_MAX', 500))
        self.pixel_threshold = [kwargs.get('PIXMIN'), kwargs.get('PIXMAX')]

        self.points = array([grd.lon.ravel(), grd.lat.ravel()]).T

        self.sla = None

        self.observations = EddiesObservations(
            track_extra_variables=self.track_extra_variables,
            track_array_variables=self.track_array_variables_sampling,
            array_variables=self.track_array_variables
        )

        self.index = 0  # counter
        self.pad = 2

    @property
    def fillval(self):
        return self._grd.fillval

    def update_eddy_properties(self, properties):
        """
        Append new variable values to track arrays
        """
        self.observations += properties

    def set_global_attr_netcdf(self, h_nc):
        h_nc.title = self.sign_type + ' eddy tracks'
        h_nc.grid_filename = self.grd.grid_filename
        h_nc.grid_date = str(self.grd.grid_date)
        #~ h_nc.product = self.product

        h_nc.contour_parameter = self.contour_parameter
        h_nc.shape_error = self.shape_error
        h_nc.pixel_threshold = self.pixel_threshold

        if self.smoothing in locals():
            h_nc.smoothing = self.smoothing
            h_nc.SMOOTH_FAC = self.SMOOTH_FAC
        else:
            h_nc.smoothing = 'None'

        h_nc.evolve_amp_min = self.evolve_amp_min
        h_nc.evolve_amp_max = self.evolve_amp_max
        h_nc.evolve_area_min = self.evolve_area_min
        h_nc.evolve_area_max = self.evolve_area_max

        h_nc.llcrnrlon = self.grd.lonmin
        h_nc.urcrnrlon = self.grd.lonmax
        h_nc.llcrnrlat = self.grd.latmin
        h_nc.urcrnrlat = self.grd.latmax

    @staticmethod
    def create_variable(handler_nc, kwargs_variable,
                        attr_variable, data, scale_factor=None, add_offset=None):
        var = handler_nc.createVariable(
            zlib=True,
            complevel=1,
            **kwargs_variable)
        for attr, attr_value in attr_variable.items():
            var.setncattr(attr, attr_value)
        if scale_factor is not None:
            var.scale_factor = scale_factor
            if add_offset is not None:
                var.add_offset = add_offset
            else:
                var.add_offset = 0
        var[:] = data
        try:
            var.setncattr('min', var[:].min())
            var.setncattr('max', var[:].max())
        except ValueError:
            logging.warning('Data is empty')

    def write_netcdf(self, path='./'):
        """Write a netcdf with eddy obs
        """
        eddy_size = len(self.observations)
        filename = '%s/%s_%s.nc' % (
            path, self.sign_type, self.date.strftime('%Y%m%d'))
        with Dataset(filename, 'w', format='NETCDF4') as h_nc:
            logging.info('Create intermediary file %s', filename)
            # Create dimensions
            logging.debug('Create Dimensions "Nobs" : %d', eddy_size)
            h_nc.createDimension('Nobs', eddy_size)
            if self.track_array_variables_sampling != 0:
                h_nc.createDimension('NbSample', self.track_array_variables_sampling)
            # Iter on variables to create:
            for dtype in self.observations.dtype:
                name = dtype[0]
                logging.debug('Create Variable %s', VAR_DESCR[name]['nc_name'])
                self.create_variable(
                    h_nc,
                    dict(varname=VAR_DESCR[name]['nc_name'],
                         datatype=VAR_DESCR[name]['output_type'],
                         dimensions=VAR_DESCR[name]['nc_dims']),
                    VAR_DESCR[name]['nc_attr'],
                    self.observations.obs[name],
                    scale_factor=VAR_DESCR[name].get('scale_factor', None),
                    add_offset=VAR_DESCR[name].get('add_offset', None)
                )

            # Add cyclonic information
            self.create_variable(
                h_nc,
                dict(varname=VAR_DESCR['type_cyc']['nc_name'],
                     datatype=VAR_DESCR['type_cyc']['nc_type'],
                     dimensions=VAR_DESCR['type_cyc']['nc_dims']),
                VAR_DESCR['type_cyc']['nc_attr'],
                -1 if self.sign_type == 'Cyclonic' else 1)
            # Global attr
            h_nc.title = self.sign_type + ' eddy tracks'
            self.set_global_attr_netcdf(h_nc)

    def set_bounds(self, contlon, contlat, grd):
        """
        Get indices to a bounding box around the eddy
        WARNING won't work for a rotated grid
        """
        lonmin, lonmax = contlon.min(), contlon.max()
        latmin, latmax = contlat.min(), contlat.max()

        self.imin, self.jmin = nearest(lonmin, latmin,
                                       grd.lon[0], grd.lat[:, 0])
        self.imax, self.jmax = nearest(lonmax, latmax,
                                       grd.lon[0], grd.lat[:, 0])

        # For indexing the mins must not be less than zero
        self.imin = max(self.imin - self.pad, 0)
        self.jmin = max(self.jmin - self.pad, 0)
        self.imax += self.pad + 1
        self.jmax += self.pad + 1
        return self

    @property
    def slice_i(self):
        return slice(self.imin, self.imax)

    @property
    def slice_j(self):
        return slice(self.jmin, self.jmax)

    @property
    def bounds(self):
        return self.imin, self.imax, self.jmin, self.jmax

    def set_mask_eff(self, contour, grd):
        """
        Set points within bounding box around eddy and calculate
        mask for effective contour
        """
        self.points = array([grd.lon[self.slice_j,
                                     self.slice_i].ravel(),
                             grd.lat[self.slice_j,
                                     self.slice_i].ravel()]).T
        # NOTE: Path.contains_points requires matplotlib 1.2 or higher
        self.mask_eff_1d = contour.contains_points(self.points)
        self.mask_eff_sum = self.mask_eff_1d.sum()

    def reshape_mask_eff(self, grd):
        """
        """
        shape = grd.lon[self.jmin:self.jmax, self.imin:self.imax].shape
        self.mask_eff = self.mask_eff_1d.reshape(shape)

    def check_pixel_count(self, nb_valid_pixel):
        return self.pixel_threshold[0] <= nb_valid_pixel <= self.pixel_threshold[1]
