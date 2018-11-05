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

Copyright (c) 2014-2017 by Antoine Delepoulle and Evan Mason
Email: emason@imedea.uib-csic.es
===========================================================================

__init__.py

Version 3.0.0

===========================================================================
"""

from netCDF4 import Dataset
from scipy import interpolate
from scipy import spatial
from pyproj import Proj
from numpy import unique, array, unravel_index, r_, floor, interp, arange, \
    sin, cos, deg2rad, arctan2, sqrt, pi, zeros, reciprocal, ma, empty, \
    concatenate, bytes_
import logging
from ..tracking_objects import nearest
from re import compile as re_compile
from os.path import join as join_path
from datetime import datetime
from glob import glob


def browse_dataset_in(data_dir, files_model, date_regexp, date_model,
                      start_date=None, end_date=None, sub_sampling_step=1,
                      files=None):
    if files is not None:
        pattern_regexp = re_compile('.*/' + date_regexp)
        filenames = bytes_(files)
    else:
        pattern_regexp = re_compile('.*/' + date_regexp)
        full_path = join_path(data_dir, files_model)
        logging.info('Search files : %s', full_path)
        filenames = bytes_(glob(full_path))

    dataset_list = empty(len(filenames),
                         dtype=[('filename', 'S500'),
                                ('date', 'datetime64[D]'),
                                ])
    dataset_list['filename'] = filenames

    logging.info('%s grids available', dataset_list.shape[0])
    mode_attrs = False
    if '(' not in date_regexp:
        logging.debug('Attrs date : %s', date_regexp)
        mode_attrs = date_regexp.strip().split(':')
    else:
        logging.debug('Pattern date : %s', date_regexp)

    for item in dataset_list:
        str_date = None
        if mode_attrs:
            with Dataset(item['filename'].decode("utf-8")) as h:
                if len(mode_attrs) == 1:
                    str_date = getattr(h, mode_attrs[0])
                else:
                    str_date = getattr(h.variables[mode_attrs[0]], mode_attrs[1])
        else:
            result = pattern_regexp.match(str(item['filename']))
            if result:
                str_date = result.groups()[0]

        if str_date is not None:
            item['date'] = datetime.strptime(str_date, date_model).date()

    dataset_list.sort(order=['date', 'filename'])

    steps = unique(dataset_list['date'][1:] - dataset_list['date'][:-1])
    if len(steps) > 1:
        raise Exception('Several days steps in grid dataset %s' % steps)

    if sub_sampling_step != 1:
        logging.info('Grid subsampling %d', sub_sampling_step)
        dataset_list = dataset_list[::sub_sampling_step]

    if start_date is not None or end_date is not None:
        logging.info('Available grid from %s to %s',
                     dataset_list[0]['date'],
                     dataset_list[-1]['date'])
        logging.info('Filtering grid by time %s, %s', start_date, end_date)
        mask = (dataset_list['date'] >= start_date) * (
            dataset_list['date'] <= end_date)

        dataset_list = dataset_list[mask]
    return dataset_list


class BaseData(object):
    """
    Base object

    Methods defined here are grouped into categories based on input data
    source, i.e., AVISO, ROMS, etc.  To introduce a new data source new methods
    can be introduced here.

    METHODS:
      Common: read_nc
              read_nc_att
              set_initial_indices
              set_index_padding
              haversine_dist
              half_interp
      AVISO:  get_aviso_f_pm_pn
      ROMS:   get_ROMS_f_pm_pn

    """

    __slots__ = (
        'zero_crossing',
        'slice_i',
        'slice_j',
        'slice_i_pad',
        'slice_j_pad',
        'slice_i_unpad',
        'slice_j_unpad',
        'eke',
        'u_val',
        'upad',
        'v_val',
        'vpad',
        '_uspd',
        'm_val',
        'm_x',
        'm_y',
        '_lon',
        '_lat',
        '_f_val',
        '_gof',
        '_pm',
        '_pn',
        '_dx',
        '_dy',
        '_umask',
        '_vmask',
        'grid_filename',
        'grid_date',
        'domain',
        'pad',
        'shape',
        'mask',
    )

    GRAVITY = 9.81
    earth_radius = 6371315.0

    def __init__(self):
        """
        Set some constants
          *zero_crossing*:
            Boolean, *True* if the_domain crosses 0 degree meridian
        """
        self.zero_crossing = False
        self.slice_i = slice(None)
        self.slice_j = slice(None)
        self.slice_i_pad = slice(None)
        self.slice_j_pad = slice(None)
        self.slice_i_unpad = slice(None)
        self.slice_j_unpad = slice(None)
        self.eke = None
        self.u_val, self.upad = None, None
        self.v_val, self.vpad = None, None
        self._uspd = None
        self.m_val = None
        self.m_x = None
        self.m_y = None
        self._lon = None
        self._lat = None
        self._f_val = None
        self._gof = None
        self._pm = None
        self._pn = None
        self._dx = None
        self._dy = None
        self._umask = None
        self._vmask = None
        self.grid_filename = None
        self.grid_date = None

    def read_nc(self, varname, indices=slice(None)):
        """
        Read data from nectdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          indices : slice
        """
        with Dataset(self.grid_filename.decode("utf-8")) as h_nc:
            return h_nc.variables[varname][indices]

    @property
    def view(self):
        return (self.slice_j,
                self.slice_i if not self.zero_crossing else
                self.create_index_inverse(self.slice_i, self._lon.shape[1]))

    @property
    def view_pad(self):
        return (self.slice_j_pad,
                self.slice_i_pad if not self.zero_crossing else
                self.create_index_inverse(self.slice_i_pad, self._lon.shape[1]))

    @property
    def view_unpad(self):
        return (self.slice_j_unpad,
                self.slice_i_unpad)

    def read_nc_att(self, varname, att):
        """
        Read data attribute from nectdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          att : string of attribute, eg. 'valid_range'
        """
        with Dataset(self.grid_filename.decode("utf-8")) as h_nc:
            return getattr(h_nc.variables[varname], att)

    @property
    def is_regular(self):
        steps_lon = unique(self._lon[0, 1:] - self._lon[0, :-1])
        steps_lat = unique(self._lat[1:, 0] - self._lat[:-1, 0])
        return len(steps_lon) == 1 and len(steps_lat) == 1 and \
               steps_lon[0] != 0. and steps_lat[0] != 0.

    def set_initial_indices(self):
        """
        Set indices for desired domain
        """
        lonmin, lonmax = self.lonmin, self.lonmax
        latmin, latmax = self.latmin, self.latmax

        logging.info('Setting initial indices to *%s* domain', self.domain)
        logging.info('lonmin = %s, lonmax = %s, latmin = %s, latmax = %s',
                     lonmin, lonmax, latmin, latmax)
        latmin_offset = latmin + (0.5 * (latmax - latmin))
        i_0, _ = self.nearest_point(lonmin, latmin_offset)
        i_1, _ = self.nearest_point(lonmax, latmin_offset)
        lonmin_offset = lonmin + (0.5 * (lonmax - lonmin))
        _, j_0 = self.nearest_point(lonmin_offset, latmin)
        _, j_1 = self.nearest_point(lonmin_offset, latmax)
        self.slice_j = slice(j_0, j_1)

        def kdt(lon, lat, limits, k=4):
            """
            Make kde tree for indices if domain crosses zero meridian
            
            Don't use cKDTree for regular grid
            """
            ppoints = array([lon.ravel(), lat.ravel()]).T
            ptree = spatial.cKDTree(ppoints)
            pindices = ptree.query(limits, k=k)[1]
            iind, jind = array([], dtype=int), array([], dtype=int)
            for pind in pindices.ravel():
                j, i = unravel_index(pind, lon.shape)
                iind = r_[iind, i]
                jind = r_[jind, j]
            return iind, jind

        if 'AvisoGrid' in self.__class__.__name__:

            if self.zero_crossing:
                """
                Used for a zero crossing, e.g., across Agulhas region
                """
                if self.is_regular:
                    i_1 = int(floor(interp((lonmin - 0.5) % 360,
                                           self._lon[0],
                                           arange(len(self._lon[0])))))
                    i_0 = int(floor(interp((lonmax + 0.5) % 360,
                                           self._lon[0],
                                           arange(len(self._lon[0])))
                                    ) + 1)
                else:
                    def half_limits(lon, lat):
                        return array([[lon.min(), lon.max(),
                                       lon.max(), lon.min()],
                                      [lat.min(), lat.min(),
                                       lat.max(), lat.max()]]).T

                    # Get bounds for right part of grid
                    lat = self._lat[self._lon >= 360 + lonmin - 0.5]
                    lon = self._lon[self._lon >= 360 + lonmin - 0.5]
                    limits = half_limits(lon, lat)
                    iind, _ = kdt(self._lon, self._lat, limits)
                    i_1 = iind.min()
                    # Get bounds for left part of grid
                    lat = self._lat[self._lon <= lonmax + 0.5]
                    lon = self._lon[self._lon <= lonmax + 0.5]
                    limits = half_limits(lon, lat)
                    iind, _ = kdt(self._lon, self._lat, limits)
                    i_0 = iind.max()
        self.slice_i = slice(i_0, i_1)

    def set_index_padding(self, pad=2):
        """
        Set start and end indices for temporary padding and later unpadding
        around 2d variables.
        Padded matrices are needed only for geostrophic velocity computation.
        """
        logging.info('\tSetting padding indices with PAD = %s', pad)

        self.pad = pad

        def get_str(start, pad):
            """
            Get start indices for pad
            Returns:
              pad_start   - index to add pad
            """
            return max(0, start - pad)

        def get_end(theend, shape, pad):
            """
            Get end indices for pad
            Returns:
              pad_end   - index to add pad
            """
            if theend is None:
                pad_end = None
            else:
                pad_end = min(shape, theend + pad)
            return pad_end

        jp0 = get_str(self.slice_j.start, pad)
        jp1 = get_end(self.slice_j.stop, self._lon.shape[0], pad)
        self.slice_j_unpad = slice(pad, -pad)
        self.slice_j_pad = slice(jp0, jp1)
        if self.zero_crossing:
            pad = -pad
        ip0 = get_str(self.slice_i.start, pad)
        ip1 = get_end(self.slice_i.stop, self._lon.shape[1], pad)
        self.slice_i_unpad = slice(abs(pad), -abs(pad))
        self.slice_i_pad = slice(ip0, ip1)

    @staticmethod
    def haversine_dist(lon1, lat1, lon2, lat2):
        """
        TO DO: change to use f2py version
        Haversine formula to calculate distance between two lon/lat points
        Uses mean earth radius in metres (from ROMS scalars.h) = 6371315.0
        Input:
          lon1, lat1, lon2, lat2
        Return:
          distance (m)
        """
        sin_dlat = sin(deg2rad(lat2 - lat1) * 0.5)
        sin_dlon = sin(deg2rad(lon2 - lon1) * 0.5)
        cos_lat1 = cos(deg2rad(lat1))
        cos_lat2 = cos(deg2rad(lat2))
        a_val = sin_dlon ** 2 * cos_lat1 * cos_lat2 + sin_dlat ** 2
        c_val = 2 * arctan2(sqrt(a_val), sqrt(1 - a_val))
        return 6371315.0 * c_val  # Return the distance

    def nearest_point(self, lon, lat):
        """
        Get indices to a point (lon, lat) in the grid
        """
        return nearest(lon, lat, self._lon[0], self._lat[:, 0])

    @staticmethod
    def half_interp(h_one, h_two):
        """
        Speed up frequent operations of type 0.5 * (arr[:-1] + arr[1:])
        """
        return (h_one + h_two) * 0.5

    def get_aviso_f_pm_pn(self):
        """
        Padded matrices are used here because Coriolis (f), p_m and p_n
        are needed for the geostrophic velocity computation in
        method getSurfGeostrVel()
        NOTE: this should serve for ROMS too
        """
        logging.info('--- Computing Coriolis (f), d_x(p_m),'
                     'd_y (p_n) for padded grid')
        # Get GRAVITY / Coriolis
        self._gof = sin(deg2rad(self.latpad)) * 4. * pi / 86400.
        self._f_val = self._gof.copy()
        self._gof = self.GRAVITY / self._gof

        lonu = self.half_interp(self.lonpad[:, :-1], self.lonpad[:, 1:])
        latu = self.half_interp(self.latpad[:, :-1], self.latpad[:, 1:])
        lonv = self.half_interp(self.lonpad[:-1], self.lonpad[1:])
        latv = self.half_interp(self.latpad[:-1], self.latpad[1:])

        # Get p_m and p_n
        p_m = zeros(self.lonpad.shape)
        p_m[:, 1:-1] = self.haversine_dist(lonu[:, :-1], latu[:, :-1],
                                           lonu[:, 1:], latu[:, 1:])
        p_m[:, 0] = p_m[:, 1]
        p_m[:, -1] = p_m[:, -2]
        self._dx = p_m
        self._pm = reciprocal(p_m)

        p_n = zeros(self.lonpad.shape)
        p_n[1:-1] = self.haversine_dist(lonv[:-1], latv[:-1],
                                        lonv[1:], latv[1:])
        p_n[0] = p_n[1]
        p_n[-1] = p_n[-2]
        self._dy = p_n
        self._pn = reciprocal(p_n)

    def u2rho_2d(self, uu_in):
        """
        Convert a 2D field at u_val points to a field at rho points
        """

        def uu2ur(uu_in, m_p, l_p):
            u_out = zeros((m_p, l_p))
            u_out[:, 1:-1] = self.half_interp(uu_in[:, :-1], uu_in[:, 1:])
            u_out[:, 0] = u_out[:, 1]
            u_out[:, -1] = u_out[:, -2]
            return u_out.squeeze()

        mshp, lshp = uu_in.shape
        return uu2ur(uu_in, mshp, lshp + 1)

    def v2rho_2d(self, vv_in):
        # Convert a 2D field at v_val points to a field at rho points
        def vv2vr(vv_in, m_p, l_p):
            v_out = zeros((m_p, l_p))
            v_out[1:-1] = self.half_interp(vv_in[:-1], vv_in[1:])
            v_out[0] = v_out[1]
            v_out[-1] = v_out[-2]
            return v_out.squeeze()

        mshp, lshp = vv_in.shape
        return vv2vr(vv_in, mshp + 1, lshp)

    def uvmask(self):
        """
        Get mask at U and V points
        """
        logging.info('--- Computing umask and vmask for padded grid')
        if getattr(self, 'mask', None) is not None:
            self._umask = self.mask[:, :-1] * self.mask[:, 1:]
            self._vmask = self.mask[:-1] * self.mask[1:]

    def set_basemap(self, with_pad=True):
        """
        Use Basemap to make a landmask
        Format is 1 == ocean, 0 == land
        """
        logging.info('Computing Basemap')
        # Create Basemap instance for Mercator projection.
        self.m_val = Proj(
            proj='merc',
            llcrnrlon=self.lonmin - 1,
            urcrnrlon=self.lonmax + 1,
            llcrnrlat=self.latmin - 1,
            urcrnrlat=self.latmax + 1,
            lat_ts=0.5 * (self.latmin + self.latmax)
        )

        if with_pad:
            x_val, y_val = self.m_val(self.lonpad, self.latpad)
        else:
            x_val, y_val = self.m_val(self.lon, self.lat)
        self.m_x, self.m_y = x_val, y_val

    def set_geostrophic_velocity(self, zeta):
        """
        Set u_val and v_val geostrophic velocity at
        surface from variables f, zeta, p_m, p_n...
        Note: output at rho points
        """
        zeta1, zeta2 = zeta.data[1:].view(), zeta.data[:-1].view()
        pn1, pn2 = self.p_n[1:].view(), self.p_n[:-1].view()
        self.upad[:] = self.v2rho_2d(
            ma.array((zeta1 - zeta2) * 0.5 * (pn1 + pn2), mask=self.vmask))
        self.upad *= -self.gof

        zeta1, zeta2 = zeta.data[:, 1:].view(), zeta.data[:, :-1].view()
        pm1, pm2 = self.p_m[:, 1:].view(), self.p_m[:, :-1].view()
        self.vpad[:] = self.u2rho_2d(
            ma.array((zeta1 - zeta2) * 0.5 * (pm1 + pm2), mask=self.umask))
        self.vpad *= self.gof

    def set_u_v_eke(self, pad=2):
        """
        Set empty arrays for u_val, v_val, upad, vpad
        """
        j_size = self.slice_j_pad.stop - self.slice_j_pad.start
        if self.zero_crossing:
            u_1 = empty((j_size, self.slice_i_pad.start))
            u_0 = empty((j_size, self._lon.shape[1] - self.slice_i_pad.stop))
            self.upad = ma.concatenate((u_0, u_1), axis=1)
        else:
            self.upad = empty((j_size,
                               self.slice_i_pad.stop - self.slice_i_pad.start))
        self.vpad = empty(self.upad.shape)

    def get_eke(self):
        """
        Set EKE; also sets u_val/v_val from upad/vpad
        """
        self.u_val = self.upad[self.view_unpad]
        self.v_val = self.vpad[self.view_unpad]
        self.eke = (self.u_val ** 2 + self.v_val ** 2) * 0.5

    @property
    def uspd(self):
        """Get scalar speed
        """
        uspd = (self.u_val ** 2 + self.v_val ** 2) ** .5
        if self.mask is not None:
            if hasattr(uspd, 'mask'):
                uspd.mask += self.mask[self.view_unpad]
            else:
                uspd = ma.array(uspd, mask=self.mask[self.view_unpad])
        return uspd

    def set_interp_coeffs(self, sla, uspd):
        """
        Won't work for rotated grid
        """
        self.sla_coeffs = interpolate.RectBivariateSpline(
            self.lat[:, 0], self.lon[0], sla, kx=1, ky=1)
        self.uspd_coeffs = interpolate.RectBivariateSpline(
            self.lat[:, 0], self.lon[0], uspd, kx=1, ky=1)

    @staticmethod
    def create_index_inverse(slice_to_inverse, size):
        """Return an array of index
        """
        index = concatenate((arange(slice_to_inverse.stop, size),
                             arange(slice_to_inverse.start)))
        return index

    def gaussian_resolution(self, zwl, mwl):
        """
        Get parameters for ndimage.gaussian_filter
        See http://stackoverflow.com/questions/14531072/
        how-to-count-bugs-in-an-image
        Input: res : grid resolution in degrees
               zwl : zonal distance in degrees
               mwl : meridional distance in degrees
        """
        zres = zwl * 0.125 / self.resolution
        mres = mwl * 0.125 / self.resolution
        return zres, mres
