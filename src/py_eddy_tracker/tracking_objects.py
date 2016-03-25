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

Copyright (c) 2014-2015 by Evan Mason
Email: emason@imedea.uib-csic.es
===========================================================================


make_eddy_tracker_list_obj.py

Version 2.0.3


===========================================================================


"""
from netCDF4 import Dataset
from matplotlib.path import Path
from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline
from . import VAR_DESCR
import numpy as np
import logging
from .observations import EddiesObservations


def nearest(lon_pt, lat_pt, lon2d, lat2d):
    """
    Return the nearest i, j point to a given lon, lat point
    in a lat/lon grid
    """
    try:
        i_x = np.int_(np.interp(lon_pt,
                                lon2d,
                                np.arange(len(lon2d)),
                                left=0, right=-1))
        i_y = np.int_(np.interp(lat_pt,
                                lat2d,
                                np.arange(len(lat2d)),
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
        self.track_extra_variables = kwargs.get('TRACK_EXTRA_VARIABLES', False)
        array_properties = kwargs.get('TRACK_ARRAY_VARIABLES', dict())
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
        self.radmin = np.float64(kwargs.get('RADMIN', 0.4))
        self.radmax = np.float64(kwargs.get('RADMAX', 4.461))
        self.ampmin = np.float64(kwargs.get('AMPMIN', 1.))
        self.ampmax = np.float64(kwargs.get('AMPMAX', 150.))
        self.evolve_amp_min = np.float64(kwargs.get('EVOLVE_AMP_MIN', .0005))
        self.evolve_amp_max = np.float64(kwargs.get('EVOLVE_AMP_MAX', 500))
        self.evolve_area_min = np.float64(kwargs.get('EVOLVE_AREA_MIN', .0005))
        self.evolve_area_max = np.float64(kwargs.get('EVOLVE_AREA_MAX', 500))

        self.points = np.array([grd.lon.ravel(), grd.lat.ravel()]).T

        self.sla = None

        self.observations = EddiesObservations(
            track_array_variables=self.track_array_variables_sampling,
            array_variables=self.track_array_variables
            )

        self.index = 0  # counter
        self.pad = 2
        self.pixel_threshold = None
        # Check for a correct configuration
        #~ assert self.product in (
            #~ 'AVISO'), 'Unknown string in *product* parameter'

    #~ @property
    #~ def product(self):
        #~ return self._grd.product

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

    def create_variable(self, handler_nc, kwargs_variable,
                        attr_variable, data, scale_factor=None, add_offset=None):
        var = handler_nc.createVariable(
            zlib=True,
            complevel=1,
            **kwargs_variable)
        for attr, attr_value in attr_variable.iteritems():
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
            logging.warn('Data is empty')

    def write_netcdf(self, path='./'):
        """Write a netcdf with eddy obs
        """
        eddy_size = len(self.observations)
        filename = '%s/%s_%s.nc' % (
            path,self.sign_type, self.date.strftime('%Y%m%d'))
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
        self.points = np.array([grd.lon[self.slice_j,
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
        return nb_valid_pixel >= self.pixel_threshold[0] and \
            nb_valid_pixel <= self.pixel_threshold[1]

class RossbyWaveSpeed_(object):

    def __init__(self, the_domain, grd, rw_path=None):
        """
        Instantiate the RossbyWaveSpeed object
        """
        self.the_domain = the_domain
        self.m_val = grd.m_val
        self.earth_radius = grd.earth_radius
        self.zero_crossing = grd.zero_crossing
        self.rw_path = rw_path
        self._tree = None
        if self.the_domain in ('Global', 'Regional', 'ROMS'):
            assert self.rw_path is not None, \
                'Must supply a path for the Rossby deformation radius data'
            datas = np.genfromtxt(
                rw_path,
                dtype=[('lat', 'f2'), ('lon', 'f2'), ('defrad', 'f4')],
                usecols=(0, 1, 3))
            lon_min, lon_max = datas['lon'].min(), datas['lon'].max()
            lat_min, lat_max = datas['lat'].min(), datas['lat'].max()
            lon_step = np.diff(np.unique(datas['lon'])[:2])[0]
            lat_step = np.diff(np.unique(datas['lat'])[:2])[0]
            lon = np.arange(lon_min, lon_max + lon_step / 2, lon_step)
            lat = np.arange(lat_min, lat_max + lat_step / 2, lat_step)
            value = np.zeros((len(lon), len(lat)), dtype='f4')
            mask = np.ones((len(lon), len(lat)), dtype='bool')
            i_lon = np.int_((datas['lon'] - lon_min) / lon_step)
            i_lat = np.int_((datas['lat'] - lat_min) / lat_step)
            value[i_lon, i_lat] = datas['defrad']
            mask[i_lon, i_lat] = False
            opts_interpolation = {'kx': 1, 'ky': 1, 's': 0}
            self.interpolate_val = RectBivariateSpline(lon, lat, value,
                                                       **opts_interpolation)
            self.interpolate_mask = RectBivariateSpline(lon, lat, mask,
                                                        **opts_interpolation)
            data = np.loadtxt(rw_path)
            self._lon = data[:, 1]
            self._lat = data[:, 0]
            self._defrad = data[:, 3]
            self.limits = [grd.lonmin, grd.lonmax, grd.latmin, grd.latmax]
            if grd.lonmin < 0:
                self._lon -= 360.
            self._make_subset()._make_kdtree()
            self.vartype = 'variable'
        else:
            self.vartype = 'constant'
        self.start = True

    def interpolate(self, *args, **kwargs):
        return np.ma.array(self.interpolate_val(*args, **kwargs),
                           mask=self.interpolate_mask(*args, **kwargs) != 0)

    def get_rwdistance(self, xpt, ypt, days_btwn_records):
        """
        Return the distance required by SearchEllipse
        to construct a search ellipse for eddy tracking.

        distance (km)
        """
        if self.the_domain in ('Global', 'Regional', 'ROMS'):
            distance = self._get_rlongwave_spd(xpt, ypt)
            distance *= 86400.

        elif 'BlackSea' in self.the_domain:
            # e.g., Blokhina & Afanasyev, 2003
            distance = 15000

        elif 'MedSea' in self.the_domain:
            distance = 20000

        else:
            raise Exception('Unknown domain : %s' % self.the_domain)

        return np.abs(distance) * days_btwn_records

    def _make_subset(self):
        """
        Make a subset of _defrad data over the domain.
        If 'Global' is defined then widen the domain.
        """
        pad = 1.5  # degrees
        lonmin, lonmax, latmin, latmax = self.limits

        if self.zero_crossing:
            ieast, iwest = (((self._lon + 360.) <= lonmax + pad),
                            (self._lon > lonmin + pad))
            self._lon[ieast] += 360.
            lloi = iwest + ieast
        else:
            lloi = np.logical_and(self._lon >= lonmin - pad,
                                  self._lon <= lonmax + pad)
        lloi *= np.logical_and(self._lat >= latmin - pad,
                               self._lat <= latmax + pad)
        self._lon = self._lon[lloi]
        self._lat = self._lat[lloi]
        self._defrad = self._defrad[lloi]

        if 'Global' in self.the_domain:
            lloi = self._lon > 260.
            self._lon = np.append(self._lon, self._lon[lloi] - 360.)
            self._lat = np.append(self._lat, self._lat[lloi])
            self._defrad = np.append(self._defrad, self._defrad[lloi])

        self.x_val, self.y_val = self.m_val(self._lon, self._lat)
        return self

    def _make_kdtree(self):
        """
        Compute KDE tree for nearest indices.
        """
        points = np.vstack([self.x_val, self.y_val]).T
        self._tree = cKDTree(points)
        return self

    def _get_defrad(self, xpt, ypt):
        """
        Get a point average of the deformation radius
        at xpt, ypt
        """
        weights, i = self._tree.query(np.array([xpt, ypt]), k=4, p=2)
        weights /= weights.sum()
        self._weights = weights
        self.i = i
        return np.average(self._defrad[i], weights=weights)

    def _get_rlongwave_spd(self, xpt, ypt):
        """
        Get the longwave phase speed, see Chelton etal (1998) pg 446:
          c = -beta * defrad**2 (this only for extratropical waves...)
        """
        # km to m
        r_spd_long = (self._get_defrad(xpt, ypt) * 1000) ** 2
        # lat
        beta = np.average(self._lat[self.i], weights=self._weights)
        # 1458e-7 ~ (2 * 7.29*10**-5)
        beta = np.cos(np.radians(beta)) * 1458e-7 / self.earth_radius
        r_spd_long *= -beta
        return r_spd_long

class RossbyWaveSpeed(object):
    def __init__(self, rw_path=None, domain=None):
        """
        Instantiate the RossbyWaveSpeed object
        """        
        self.the_domain = 'Global' if domain is None else domain
        self.rw_path = rw_path
        if self.the_domain in ('Global', 'Regional', 'ROMS'):
            assert self.rw_path is not None, \
                'Must supply a path for the Rossby deformation radius data'
            datas = np.genfromtxt(
                self.rw_path,
                dtype=[('lat', 'f2'), ('lon', 'f2'), ('defrad', 'f4')],
                usecols=(0, 1, 3))
        #~ else:
        #~ self.earth_radius = 
        #~ self.zero_crossing = 
            self._tree = None
            
            lon_min, lon_max = datas['lon'].min(), datas['lon'].max()
            lat_min, lat_max = datas['lat'].min(), datas['lat'].max()
            lon_step = np.diff(np.unique(datas['lon'])[:2])[0]
            lat_step = np.diff(np.unique(datas['lat'])[:2])[0]
            lon = np.arange(lon_min, lon_max + lon_step / 2, lon_step)
            lat = np.arange(lat_min, lat_max + lat_step / 2, lat_step)
            value = np.zeros((len(lon), len(lat)), dtype='f4')
            mask = np.ones((len(lon), len(lat)), dtype='bool')
            i_lon = np.int_((datas['lon'] - lon_min) / lon_step)
            i_lat = np.int_((datas['lat'] - lat_min) / lat_step)
            value[i_lon, i_lat] = datas['defrad']
            mask[i_lon, i_lat] = False
            opts_interpolation = {'kx': 1, 'ky': 1, 's': 0}
            self.interpolate_val = RectBivariateSpline(lon, lat, value,
                                                       **opts_interpolation)
            self.interpolate_mask = RectBivariateSpline(lon, lat, mask,
                                                        **opts_interpolation)
            data = np.loadtxt(rw_path)
            self._lon = data[:, 1]
            self._lat = data[:, 0]
            self._defrad = data[:, 3]
            self.limits = [grd.lonmin, grd.lonmax, grd.latmin, grd.latmax]
            if grd.lonmin < 0:
                self._lon -= 360.
            self._make_subset()._make_kdtree()
            self.vartype = 'variable'
        else:
            self.vartype = 'constant'
        self.start = True

    def interpolate(self, *args, **kwargs):
        return np.ma.array(self.interpolate_val(*args, **kwargs),
                           mask=self.interpolate_mask(*args, **kwargs) != 0)

    def get_rwdistance(self, xpt, ypt, days_btwn_records):
        """
        Return the distance required by SearchEllipse
        to construct a search ellipse for eddy tracking.

        distance (km)
        """
        if self.the_domain in ('Global', 'Regional', 'ROMS'):
            distance = self._get_rlongwave_spd(xpt, ypt)
            distance *= 86400.

        elif 'BlackSea' in self.the_domain:
            # e.g., Blokhina & Afanasyev, 2003
            distance = 15000

        elif 'MedSea' in self.the_domain:
            distance = 20000

        else:
            raise Exception('Unknown domain : %s' % self.the_domain)

        return np.abs(distance) * days_btwn_records

    def _make_subset(self):
        """
        Make a subset of _defrad data over the domain.
        If 'Global' is defined then widen the domain.
        """
        pad = 1.5  # degrees
        lonmin, lonmax, latmin, latmax = self.limits

        if self.zero_crossing:
            ieast, iwest = (((self._lon + 360.) <= lonmax + pad),
                            (self._lon > lonmin + pad))
            self._lon[ieast] += 360.
            lloi = iwest + ieast
        else:
            lloi = np.logical_and(self._lon >= lonmin - pad,
                                  self._lon <= lonmax + pad)
        lloi *= np.logical_and(self._lat >= latmin - pad,
                               self._lat <= latmax + pad)
        self._lon = self._lon[lloi]
        self._lat = self._lat[lloi]
        self._defrad = self._defrad[lloi]

        if 'Global' in self.the_domain:
            lloi = self._lon > 260.
            self._lon = np.append(self._lon, self._lon[lloi] - 360.)
            self._lat = np.append(self._lat, self._lat[lloi])
            self._defrad = np.append(self._defrad, self._defrad[lloi])

        self.x_val, self.y_val = self.m_val(self._lon, self._lat)
        return self

    def _make_kdtree(self):
        """
        Compute KDE tree for nearest indices.
        """
        points = np.vstack([self.x_val, self.y_val]).T
        self._tree = cKDTree(points)
        return self

    def _get_defrad(self, xpt, ypt):
        """
        Get a point average of the deformation radius
        at xpt, ypt
        """
        weights, i = self._tree.query(np.array([xpt, ypt]), k=4, p=2)
        weights /= weights.sum()
        self._weights = weights
        self.i = i
        return np.average(self._defrad[i], weights=weights)

    def _get_rlongwave_spd(self, xpt, ypt):
        """
        Get the longwave phase speed, see Chelton etal (1998) pg 446:
          c = -beta * defrad**2 (this only for extratropical waves...)
        """
        # km to m
        r_spd_long = (self._get_defrad(xpt, ypt) * 1000) ** 2
        # lat
        beta = np.average(self._lat[self.i], weights=self._weights)
        # 1458e-7 ~ (2 * 7.29*10**-5)
        beta = np.cos(np.radians(beta)) * 1458e-7 / self.earth_radius
        r_spd_long *= -beta
        return r_spd_long


def west_ellips_contains(ellips_center, obs, minor, major):
    in_ellips = (((obs - ellips_center) ** 2) /
                 np.array([major, minor]) ** 2).sum(axis=1) < 1
    in_circle = (((obs - ellips_center) / minor) ** 2).sum(axis=1) < 1
    east = obs[0] > ellips_center[0]
    return in_ellips * (- east) + in_circle


def east_ellips_contains(ellips_center, obs, minor, major):
    in_ellips = (((obs - ellips_center) ** 2) /
                 np.array([major, minor]) ** 2).sum(axis=1) < 1
    in_circle = (((obs - ellips_center) / minor) ** 2).sum(axis=1) < 1
    east = obs[0] > ellips_center[0]
    return in_ellips * east + in_circle


class SearchEllipse (object):
    """
    Class to construct a search ellipse/circle around a specified point.
    See CSS11 Appendix B.4. "Automated eddy tracking" for details.
    """
    def __init__(self, the_domain, grd, days_btwn_records, rw_path=None):
        """
        Set the constant dimensions of the search ellipse.
        Instantiate a RossbyWaveSpeed object

        Arguments:

          *the_domain*: string
            Refers to the_domain specified in yaml configuration file

          *grd*: An AvisoGrid or RomsGrid object.

          *days_btwn_records*: integer
            Constant defined in yaml configuration file.

          *rw_path*: string
            Path to rossrad.dat file, specified in yaml configuration file.
        """
        self.the_domain = the_domain
        self.days_btwn_records = days_btwn_records
        self.e_w_major = self.days_btwn_records * 3e5 / 7.
        self.n_s_minor = self.days_btwn_records * 15e4 / 7.
        self.semi_n_s_minor = 0.5 * self.n_s_minor
        self.rwv = RossbyWaveSpeed(the_domain, grd, rw_path=rw_path)
        self.rw_c_mod = None

    def _set_east_ellipse(self):
        """
        The *east_ellipse* is a full ellipse, but only its eastern
        part is used to build the search ellipse.
        """
        self.east_ellipse = Ellipse((self.xpt, self.ypt),
                                    self.e_w_major, self.n_s_minor)
        return self

    def _set_west_ellipse(self):
        """
        The *west_ellipse* is a full ellipse, but only its western
        part is used to build the search ellipse.
        """
        self.west_ellipse = Ellipse((self.xpt, self.ypt),
                                    self.rw_c_mod, self.n_s_minor)
        return self

    def _set_global_ellipse(self):
        """
        Set a Path object *ellipse_path* built from the eastern vertices of
        *east_ellipse* and the western vertices of *west_ellipse*.
        """
        self._set_east_ellipse()._set_west_ellipse()
        e_verts = self.east_ellipse.get_verts()
        e_size = e_verts[:, 0].size
        e_size *= 0.5
        w_verts = self.west_ellipse.get_verts()
        w_size = w_verts[:, 0].size
        w_size *= 0.5
        ew_x = np.hstack((e_verts[e_size:, 0], w_verts[:w_size, 0]))
        ew_y = np.hstack((e_verts[e_size:, 1], w_verts[:w_size, 1]))
        self.ellipse_path = Path(np.array([ew_x, ew_y]).T)

    def _set_black_sea_ellipse(self):
        """
        Set *ellipse_path* for the *black_sea_ellipse*.
        """
        self.black_sea_ellipse = Ellipse(
            (self.xpt, self.ypt),
            2. * self.rw_c_mod,
            2. * self.rw_c_mod)
        verts = self.black_sea_ellipse.get_verts()
        self.ellipse_path = Path(np.array([verts[:, 0],
                                           verts[:, 1]]).T)
        return self

    def set_search_ellipse(self, xpt, ypt):
        """
        Set the search ellipse around a point.

        args:

            *xpt*: lon coordinate (Basemap projection)

            *ypt*: lat coordinate (Basemap projection)

        """
        self.xpt = xpt
        self.ypt = ypt
        self.rw_c_mod = 1.75

        if self.the_domain in ('Global', 'Regional', 'ROMS'):
            rw_c = self.rwv.get_rwdistance(xpt, ypt,
                                           self.days_btwn_records)
            self.rw_c_mod *= rw_c
            self.rw_c_mod = np.array([self.rw_c_mod,
                                      self.semi_n_s_minor]).max() * 2
            self._set_global_ellipse()

        elif self.the_domain in ('BlackSea', 'MedSea'):
            rw_c = self.rwv.get_rwdistance(xpt, ypt,
                                           self.days_btwn_records)
            self.rw_c_mod *= rw_c
            self._set_black_sea_ellipse()
        else:
            raise Exception()

        return self
