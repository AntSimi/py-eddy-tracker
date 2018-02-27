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

aviso.py

Version 3.0.0

===========================================================================
"""
from matplotlib.dates import date2num
from scipy import ndimage
from scipy import spatial
from dateutil import parser
from numpy import meshgrid, zeros, array, where, ma, argmin, vstack, ones, \
    newaxis, sqrt, diff, r_, arange
from scipy.interpolate import interp1d
import logging
from netCDF4 import Dataset

from . import BaseData


class AvisoGrid(BaseData):
    """
    Class to satisfy the need of the eddy tracker
    to have a grid class
    """
    KNOWN_UNITS = dict(
        m=100.,
        cm=1.,
    )
    __slots__ = (
        'lonmin',
        'lonmax',
        'latmin',
        'latmax',
        'lon_name',
        'lat_name',
        'grid_name',
        '_lon',
        '_lat',
        'fillval',
        '_angle',
        'sla_coeffs',
        'xinterp',
        'yinterp',
        'uspd_coeffs',
        '__lon',
        '__lat',
        '__lonpad',
        '__latpad',
        'labels',
    )

    def __init__(self, aviso_file, the_domain,
                 lonmin, lonmax, latmin, latmax, grid_name, lon_name,
                 lat_name, with_pad=True):
        """
        Initialise the grid object
        """
        super(AvisoGrid, self).__init__()
        logging.info('Initialising the *AVISO_grid*')
        self.grid_filename = aviso_file

        self.lon_name = lon_name
        self.lat_name = lat_name
        self.grid_name = grid_name

        self._lon = self.read_nc(self.lon_name)
        self._lat = self.read_nc(self.lat_name)
        if the_domain is None:
            self.domain = 'Automatic Domain'
            dlon = abs(self._lon[1] - self._lon[0])
            dlat = abs(self._lat[1] - self._lat[0])
            self.lonmin = float(self._lon.min()) + dlon * 2
            self.lonmax = float(self._lon.max()) - dlon * 2
            self.latmin = float(self._lat.min()) + dlat * 2
            self.latmax = float(self._lat.max()) - dlat * 2
            if ((self._lon[-1] + dlon) % 360) == self._lon[0]:
                self.domain = 'Global'
                self.lonmin = -100.
                self.lonmax = 290.
                self.latmin = -80.
                self.latmax = 80.
        else:
            self.domain = the_domain
            self.lonmin = float(lonmin)
            self.lonmax = float(lonmax)
            self.latmin = float(latmin)
            self.latmax = float(latmax)

        self.fillval = self.read_nc_att(self.grid_name, '_FillValue')

        if self.lonmin < 0 and self.lonmax <= 0:
            self._lon -= 360.
        self._lon, self._lat = meshgrid(self._lon, self._lat)
        self._angle = zeros(self._lon.shape)

        if 'MedSea' in self.domain:
            self._lon -= 360.

        # zero_crossing, used for handling a longitude range that
        # crosses zero degree meridian
        if self.lonmin < 0 <= self.lonmax and 'MedSea' not in self.domain:
            if self._lon.min() < self.lonmax < self._lon.max() and self._lon.min() < self.lonmin < self._lon.max():
                pass
            else:
                self.zero_crossing = True

        self.sla_coeffs = None
        self.uspd_coeffs = None

        self.__lon, self.__lat = None, None
        self.__lonpad, self.__latpad = None, None
        self.set_initial_indices()
        self.set_index_padding()
        self.set_basemap(with_pad=with_pad)
        self.get_aviso_f_pm_pn()
        self.set_u_v_eke()
        self.shape = self.lon.shape

        # self.init_pos_interpolator()

    def init_pos_interpolator(self):
        self.xinterp = interp1d(self.lon[0].copy(), arange(self.lon.shape[1]), assume_sorted=True, copy=False,
                                fill_value=(0, -1), bounds_error=False, kind='nearest')
        self.yinterp = interp1d(self.lat[:, 0].copy(), arange(self.lon.shape[0]), assume_sorted=True, copy=False,
                                fill_value=(0, -1), bounds_error=False, kind='nearest')

    def nearest_indice(self, lon, lat):
        return self.xinterp(lon), self.yinterp(lat)

    def set_filename(self, file_name):
        self.grid_filename = file_name

    def get_aviso_data(self, aviso_file, dimensions=None):
        """
        Read nc data from AVISO file
        """
        if dimensions is None:
            dimensions = dict()
        self.grid_filename = aviso_file
        units = self.read_nc_att(self.grid_name, 'units')
        if units not in self.KNOWN_UNITS:
            raise Exception('Unknown units : %s' % units)

        with Dataset(self.grid_filename.decode('utf-8')) as h_nc:
            grid_dims = array(h_nc.variables[self.grid_name].dimensions)
            lat_dim = h_nc.variables[self.lat_name].dimensions[0]
            lon_dim = h_nc.variables[self.lon_name].dimensions[0]

        i_list = []
        transpose = False
        if where(grid_dims == lat_dim)[0][0] > where(grid_dims == lon_dim)[0][0]:
            transpose = True
        for grid_dim in grid_dims:
            if grid_dim == lat_dim:
                i_list.append(self.view_pad[0])
            elif grid_dim == lon_dim:
                i_list.append(self.view_pad[1])
            else:
                i_list.append(
                    dimensions[grid_dim] if grid_dim in dimensions.keys() else 0)

        zeta = self.read_nc(self.grid_name, indices=i_list)
        if transpose:
            zeta = zeta.T

        zeta *= self.KNOWN_UNITS[units]  # units to cm
        if not hasattr(zeta, 'mask'):
            zeta = ma.array(zeta)
        return zeta

    def set_mask(self, sla):
        """
        """
        if sla.mask.size == 1:  # all sea points
            self.mask = None
        else:
            self.mask = sla.mask.copy()
            if 'Global' in self.domain:
                # Close Drake Passage
                minus70 = argmin(abs(self.lonpad[0] + 70))
                self.mask[:125, minus70] = True

                # Mask all unwanted regions (Caspian Sea, etc)
                self.labels = ndimage.label(~self.mask)[0]

                # Set to known sea point
                plus200 = argmin(abs(self.lonpad[0] - 200))
                plus9 = argmin(abs(self.latpad[:, 0] - 9))
                sea_label = self.labels[plus9, plus200]
                self.mask += self.labels != sea_label

    @property
    def lon(self):
        if self.__lon is None:
            # It must be an 1D array and not an 2d array ?
            self.__lon = self._lon[self.view]
            if self.zero_crossing:
                self.__lon[:, :self._lon.shape[1] - self.slice_i.stop] -= 360
        return self.__lon

    @property
    def lat(self):
        # It must be an 1D array and not an 2d array ?
        if self.__lat is None:
            self.__lat = self._lat[self.view]
        return self.__lat

    @property
    def lonpad(self):
        if self.__lonpad is None:
            self.__lonpad = self._lon[self.view_pad]
            if self.zero_crossing:
                self.__lonpad[:, :self._lon.shape[1] - self.slice_i_pad.stop
                ] -= 360
        return self.__lonpad

    @property
    def latpad(self):
        if self.__latpad is None:
            self.__latpad = self._lat[self.view_pad]
        return self.__latpad

    @property
    def angle(self):
        return self._angle[self.view]

    @property
    def umask(self):  # Mask at U points
        return self._umask

    @property
    def vmask(self):  # Mask at V points
        return self._vmask

    @property
    def f_coriolis(self):  # Coriolis
        return self._f_val

    @property
    def gof(self):  # Gravity / Coriolis
        return self._gof

    @property
    def d_x(self):  # Grid spacing along X direction
        return self._dx

    @property
    def d_y(self):  # Grid spacing along Y direction
        return self._dy

    @property
    def p_m(self):  # Reciprocal of d_x
        return self._pm

    @property
    def p_n(self):  # Reciprocal of d_y
        return self._pn

    @property
    def resolution(self):
        return sqrt(diff(self.lon[1:], axis=1) *
                    diff(self.lat[:, 1:], axis=0)).mean()

    @property
    def boundary(self):
        """
        Return lon, lat of perimeter around a ROMS grid
        Input:
          indices to get boundary of specified subgrid
        Returns:
          lon/lat boundary points
        """
        lon = r_[(self.lon[:, 0], self.lon[-1],
                  self.lon[::-1, -1], self.lon[0, ::-1])]
        lat = r_[(self.lat[:, 0], self.lat[-1],
                  self.lat[::-1, -1], self.lat[0, ::-1])]
        return lon, lat
