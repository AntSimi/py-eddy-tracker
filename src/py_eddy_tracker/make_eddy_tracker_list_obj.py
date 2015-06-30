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
from matplotlib.dates import num2julian
from scipy.spatial import cKDTree
from scipy.interpolate import Akima1DInterpolator, interp1d
from .tools import distance_vector
from . import VAR_DESCR
from datetime import datetime
import cPickle as pickle
import numpy as np
import logging


def timeit(method):
    """
    Decorator to time a function
    """
    def timed(*args, **kw):
        t_0 = datetime.now()
        result = method(*args, **kw)
        t_1 = datetime.now()
        logging.info('%s : %s sec', method.__name__, t_1 - t_0)
        return result
    return timed


# def newPosition(lonin, latin, angle, distance):
#     """
#     Given the inputs (base lon, base lat, angle, distance) return
#     the lon, lat of the new position...
#     """
#     lonin = np.asfortranarray(lonin.copy())
#     latin = np.asfortranarray(latin.copy())
#     angle = np.asfortranarray(angle.copy())
#     distance = np.asfortranarray(distance.copy())
#     lonout = np.asfortranarray(np.empty(lonin.shape))
#     latout = np.asfortranarray(np.empty(lonin.shape))
#     haversine.waypoint_vector(lonin, latin, angle, distance, lonout, latout)
#     return np.ascontiguousarray(lonout), np.ascontiguousarray(latout)


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
        print lat2d, lat_pt
        raise ValueError()
    return i_x, i_y


def uniform_resample(x, y,
#                      method='interp1d', kind='linear',
                     **kwargs
                     ):
    """
    Resample contours to have (nearly) equal spacing
       x, y    : input contour coordinates
       num_fac : factor to increase lengths of output coordinates
       method  : currently only 'interp1d' or 'Akima'
                 (Akima is slightly slower, but may be more accurate)
       kind    : type of interpolation (interp1d only)
       extrapolate : IS NOT RELIABLE (sometimes nans occur)
    """
    method = kwargs.get('method', 'interp1d')
    extrapolate = kwargs.get('extrapolate', None)
    num_fac = kwargs.get('num_fac', 2)

    # Get distances
    dist = np.zeros_like(x)
    distance_vector(
        x[:-1], y[:-1], x[1:], y[1:], dist[1:])
    dist.cumsum(out=dist)
    # Get uniform distances
    d_uniform = np.linspace(0,
                            dist.max(),
                            num=dist.size * num_fac,
                            endpoint=True)

    # Do 1d interpolations
    if strcompare('interp1d', method):
        kind = kwargs.get('kind', 'linear')
        xfunc = interp1d(dist, x, kind=kind)
        yfunc = interp1d(dist, y, kind=kind)
        xnew = xfunc(d_uniform)
        ynew = yfunc(d_uniform)

    elif strcompare('akima', method):
        xfunc = Akima1DInterpolator(dist, x)
        yfunc = Akima1DInterpolator(dist, y)
        xnew = xfunc(d_uniform, extrapolate=extrapolate)
        ynew = yfunc(d_uniform, extrapolate=extrapolate)

    else:
        Exception

    return xnew, ynew


def strcompare(str1, str2):
    return str1 in str2 and str2 in str1


class Track (object):
    """
    Class that holds eddy tracks and related info
        index  - index to each 'track', or track_number
        lon    - longitude
        lat    - latitude
        ocean_time - roms time in seconds
        uavg   - average velocity within eddy contour
        radius_s - eddy radius (as Chelton etal 2011)
        radius_e - eddy radius (as Chelton etal 2011)
        amplitude - max(abs(vorticity/f)) within eddy (as Kurian etal 2011)
        temp
        salt
        bounds - array(imin,imax,jmin,jmax) defining location of eddy
                 qparam contour
        alive - True if eddy active, set to False when eddy becomes active
        saved2nc - Becomes True once saved to netcdf file
        dayzero - True at first appearance of eddy
    """

    def __init__(self, PRODUCT, lon, lat, time, uavg, teke,
                 radius_s, radius_e, amplitude, temp=None, salt=None,
                 save_extras=False, contour_e=None, contour_s=None,
                 uavg_profile=None, shape_error=None):

        # self.eddy_index = eddy_index
        self.product = PRODUCT
        self.lon = [lon]
        self.lat = [lat]
        self.ocean_time = [time]
        self.uavg = [uavg]
        self.teke = [teke]
        self.radius_s = [radius_s]  # speed-based eddy radius
        self.radius_e = [radius_e]  # effective eddy radius
        self.amplitude = [amplitude]
        if 'ROMS' in self.product:
            # self.temp = [temp]
            # self.salt = [salt]
            pass
        self.alive = True
        self.dayzero = True
        self.saved2nc = False
        self.save_extras = save_extras
        if self.save_extras:
            self.contour_e = [contour_e]
            self.contour_s = [contour_s]
            self.uavg_profile = [uavg_profile]
            self.shape_error = [shape_error]

    def append_pos(self, lon, lat, time, uavg, teke, radius_s, radius_e,
                   amplitude, temp=None, salt=None, contour_e=None,
                   contour_s=None, uavg_profile=None, shape_error=None):
        """
        Append track updates
        """
        self.lon.append(lon)
        self.lat.append(lat)
        self.ocean_time.append(time)
        self.uavg.append(uavg)
        self.teke.append(teke)
        self.radius_s.append(radius_s)
        self.radius_e.append(radius_e)
        self.amplitude.append(amplitude)
        if 'ROMS' in self.product:
#             self.temp = np.r_[self.temp, temp]
#             self.salt = np.r_[self.salt, salt]
            pass
        if self.save_extras:
            self.contour_e.append(contour_e)
            self.contour_s.append(contour_s)
            self.uavg_profile.append(uavg_profile)
            self.shape_error = np.r_[self.shape_error, shape_error]
        return self

    def _is_alive(self, rtime):
        """
        Query if eddy is still active
          rtime is current 'ocean_time'
        If not active, kill it
        """
        # The eddy...
        # print not self.alive, self.dayzero, self.ocean_time[-1] == rtime
        if not self.alive:  # is already dead
            return self.alive
        elif self.dayzero:  # has just been initiated
            self.dayzero = False
            return self.alive
        elif self.ocean_time[-1] == rtime:  # is still alive
            return self.alive
        else:
            self.alive = False  # is now dead
            return self.alive


class TrackList (object):
    """
    Class that holds list of eddy tracks:
        tracklist - the list of 'track' objects
        qparameter: Q parameter range used for contours
        new_lon, new_lat: new lon/lat centroids
        old_lon, old_lat: old lon/lat centroids
        index:   index of eddy in track_list
    """
    def __init__(self, SIGN_TYPE, SAVE_DIR, grd, SEARCH_ELLIPSE,
                 **kwargs):
        """
        Initialise the list 'tracklist'
        """
        self.tracklist = []
        self.product = grd.product
        self.SIGN_TYPE = SIGN_TYPE
        self.SAVE_DIR = SAVE_DIR

        self.DIAGNOSTIC_TYPE = kwargs.get('DIAGNOSTIC_TYPE', 'SLA')

        self.the_domain = kwargs.get('the_domain', 'Regional')
        self.lonmin = np.float64(kwargs.get('lonmin', -40))
        self.lonmax = np.float64(kwargs.get('lonmax', -30))
        self.latmin = np.float64(kwargs.get('latmin', 20))
        self.latmax = np.float64(kwargs.get('latmax', 30))
        self.DATE_STR = np.float64(kwargs.get('DATE_STR', 20020101))
        self.DATE_END = np.float64(kwargs.get('DATE_END', 20020630))

        self.TRACK_DURATION_MIN = kwargs.get('TRACK_DURATION_MIN', 28)
        self.TRACK_EXTRA_VARIABLES = kwargs.get('TRACK_EXTRA_VARIABLES', False)
        self.INTERANNUAL = kwargs.get('INTERANNUAL', True)
        self.SEPARATION_METHOD = kwargs.get('SEPARATION_METHOD', 'ellipse')
        self.SMOOTHING = kwargs.get('SMOOTHING', True)
        self.MAX_LOCAL_EXTREMA = kwargs.get('MAX_LOCAL_EXTREMA', 1)

        self.INTERP_METHOD = kwargs.get('INTERP_METHOD', 'RectBivariate')
        self.JDAY_REFERENCE = kwargs.get('JDAY_REFERENCE', 2448623.0)

        # NOTE: '.copy()' suffix is essential here
        self.CONTOUR_PARAMETER = kwargs.get('CONTOUR_PARAMETER',
                                            np.arange(-100., 101, 1)).copy()
        self.INTERVAL = np.diff(self.CONTOUR_PARAMETER)[0]
        if 'Cyclonic' in SIGN_TYPE:
            self.CONTOUR_PARAMETER *= -1

        self.SHAPE_ERROR = kwargs.get(
            'SHAPE_ERROR',
            np.full(self.CONTOUR_PARAMETER.size, 55.))

        self.DAYS_BTWN_RECORDS = kwargs.get('DAYS_BTWN_RECORDS', 7.)

        self.RADMIN = np.float64(kwargs.get('RADMIN', 0.4))
        self.RADMAX = np.float64(kwargs.get('RADMAX', 4.461))
        self.AMPMIN = np.float64(kwargs.get('AMPMIN', 1.))
        self.AMPMAX = np.float64(kwargs.get('AMPMAX', 150.))

        self.EVOLVE_AMP_MIN = np.float64(kwargs.get('EVOLVE_AMP_MIN', 0.0005))
        self.EVOLVE_AMP_MAX = np.float64(kwargs.get('EVOLVE_AMP_MAX', 500))
        self.EVOLVE_AREA_MIN = np.float64(kwargs.get('EVOLVE_AREA_MIN',
                                                     0.0005))
        self.EVOLVE_AREA_MAX = np.float64(kwargs.get('EVOLVE_AREA_MAX', 500))

        self.AREA0 = np.pi * np.float64(kwargs.get('RAD0', 60000.))**2
        self.AMP0 = np.float64(kwargs.get('AMP0', 2.))
        self.DIST0 = np.float64(kwargs.get('DIST0', 25000.))

        self.SAVE_FIGURES = kwargs.get('SAVE_FIGURES', False)

        self.VERBOSE = kwargs.get('VERBOSE', False)

        self.m_val = grd.m_val
        self.points = np.array([grd.lon.ravel(),
                                grd.lat.ravel()]).T
        self.i_0, self.i_1 = grd.i_0, grd.i_1
        self.j_0, self.j_1 = grd.j_0, grd.j_1
        self.fillval = grd.fillval
        self.product = grd.product

        self.sla = None
        self.slacopy = None

        self.new_lon = []
        self.new_lat = []
        self.new_radii_s = []
        self.new_radii_e = []
        self.new_amp = []
        self.new_uavg = []
        self.new_teke = []
        self.new_lon_tmp = []
        self.new_lat_tmp = []
        self.new_radii_s_tmp = []
        self.new_radii_e_tmp = []
        self.new_time_tmp = []
        self.new_temp_tmp = []
        self.new_salt_tmp = []
        self.new_amp_tmp = []
        self.new_uavg_tmp = []
        self.new_teke_tmp = []
        if 'ROMS' in self.product:
            self.new_temp = []
            self.new_salt = []
        # NOTE check if new_time and old_time are necessary...
        self.new_time = []
        self.old_lon = []
        self.old_lat = []
        self.old_radii_s = []
        self.old_radii_e = []
        self.old_amp = []
        self.old_uavg = []
        self.old_teke = []
        if 'ROMS' in self.product:
            # self.old_temp = []
            pass
            # self.old_salt = []
        self.old_time = []
        if self.TRACK_EXTRA_VARIABLES:
            self.new_contour_e = []
            self.new_contour_s = []
            self.new_uavg_profile = []
            self.new_shape_error = []
            self.old_contour_e = []
            self.old_contour_s = []
            self.old_uavg_profile = []
            self.old_shape_error = []
        self.new_list = True  # flag indicating new list
        self.index = 0  # counter
        # index to write to h_nc files, will increase and increase
        self.ncind = 0
        self.ch_index = 0  # index for Chelton style h_nc files
        self.PAD = 2
        self.search_ellipse = None
        self.PIXEL_THRESHOLD = None
        # Check for a correct configuration
        assert self.product in (
            'ROMS', 'AVISO'), 'Unknown string in *product* parameter'

    def __getstate__(self):
        """
        Needed for Pickle
        """
        # print '--- removing unwanted attributes'
        """pops = ('uspd', 'uspd_coeffs', 'sla_coeffs', 'points',
                'circlon', 'circlat', 'sla', 'slacopy', 'swirl',
                'mask_eff', 'mask_eff_sum', 'mask_eff_1d')"""
        pops = ('uspd', 'uspd_coeffs', 'sla_coeffs', 'points',
                'sla', 'slacopy', 'swirl',
                'mask_eff', 'mask_eff_sum', 'mask_eff_1d')
        result = self.__dict__.copy()
        for pop in pops:
            result.pop(pop)
        return result

    def add_new_track(self, lon, lat, time, uavg, teke,
                      radius_s, radius_e, amplitude, temp=None, salt=None,
                      contour_e=None, contour_s=None, uavg_profile=None,
                      shape_error=None):
        """
        Append a new 'track' object to the list
        """
        self.tracklist.append(Track(self.product,
                                    lon, lat, time, uavg, teke,
                                    radius_s, radius_e, amplitude,
                                    temp, salt, self.TRACK_EXTRA_VARIABLES,
                                    contour_e, contour_s, uavg_profile,
                                    shape_error))

    def update_track(self, index, lon, lat, time, uavg, teke,
                     radius_s, radius_e, amplitude, temp=None, salt=None,
                     contour_e=None, contour_s=None, uavg_profile=None,
                     shape_error=None):
        """
        Update a track at index
        """
        self.tracklist[index].append_pos(
            lon, lat, time, uavg, teke,
            radius_s, radius_e, amplitude, temp=temp,
            salt=salt, contour_e=contour_e,
            contour_s=contour_s, uavg_profile=uavg_profile,
            shape_error=shape_error)

    def update_eddy_properties(self, properties):
        """
        Append new variable values to track arrays
        """
        self.new_lon_tmp.append(properties.centlon)
        self.new_lat_tmp.append(properties.centlat)
        self.new_radii_s_tmp.append(properties.eddy_radius_s)
        self.new_radii_e_tmp.append(properties.eddy_radius_e)
        self.new_amp_tmp.append(properties.amplitude)
        self.new_uavg_tmp.append(properties.uavg)
        self.new_teke_tmp.append(properties.teke)
        self.new_time_tmp.append(properties.rtime)

        if 'ROMS' in self.product:
            # self.new_temp_tmp = np.r_[self.new_temp_tmp, properties.cent_temp]
            # self.new_salt_tmp = np.r_[self.new_salt_tmp, properties.cent_salt]
            pass

        if self.TRACK_EXTRA_VARIABLES:
            self.new_contour_e_tmp.append(properties.contour_e)
            self.new_contour_s_tmp.append(properties.contour_s)
            self.new_uavg_profile_tmp.append(properties.uavg_profile)
            self.new_shape_error_tmp = np.r_[self.new_shape_error_tmp,
                                             properties.shape_error]
        return self

    def reset_holding_variables(self):
        """
        Reset temporary holding variables to empty arrays
        """
        del self.new_lon_tmp[:]
        del self.new_lat_tmp[:]
        del self.new_radii_s_tmp[:]
        del self.new_radii_e_tmp[:]
        del self.new_amp_tmp[:]
        del self.new_uavg_tmp[:]
        del self.new_teke_tmp[:]
        del self.new_time_tmp[:]
        del self.new_temp_tmp[:]
        del self.new_salt_tmp[:]
        # self.new_bounds_tmp = np.atleast_2d(np.empty(4, dtype=np.int16))
        if self.TRACK_EXTRA_VARIABLES:
            del self.new_contour_e_tmp[:]
            del self.new_contour_s_tmp[:]
            del self.new_uavg_profile_tmp[:]
            self.new_shape_error_tmp = np.atleast_1d([])
        return

    def set_old_variables(self):
        """
        Pass all values at time k+1 to k
        """
        self.old_lon = list(self.new_lon_tmp)
        self.old_lat = list(self.new_lat_tmp)
        self.old_radii_s = list(self.new_radii_s_tmp)
        self.old_radii_e = list(self.new_radii_e_tmp)
        self.old_amp = list(self.new_amp_tmp)
        self.old_uavg = list(self.new_uavg_tmp)
        self.old_teke = list(self.new_teke_tmp)
        self.old_temp = list(self.new_temp_tmp)
        self.old_salt = list(self.new_salt_tmp)
        if self.TRACK_EXTRA_VARIABLES:
            self.old_contour_e = list(self.new_contour_e_tmp)
            self.old_contour_s = list(self.new_contour_s_tmp)
            self.old_uavg_profile = list(self.new_uavg_profile_tmp)
            self.old_shape_error = np.atleast_1d([])

    def get_active_tracks(self, rtime):
        """
        Return list of indices to active tracks.
        A track is defined as active if the last record
        corresponds to current rtime (ocean_time).
        This call also identifies and removes
        inactive tracks.
        """
        active_tracks = []
        for i, track in enumerate(self.tracklist):
            if track._is_alive(rtime):
                active_tracks.append(i)
        return active_tracks

    def get_inactive_tracks(self, rtime, stopper=0):
        """
        Return list of indices to inactive tracks.
        This call also identifies and removes
        inactive tracks
        """
        inactive_tracks = []
        for i, track in enumerate(self.tracklist):
            if not track._is_alive(rtime):
                inactive_tracks.append(i)
        return inactive_tracks

    def kill_all_tracks(self):
        """
        Mark all tracks as not alive
        """
        for track in self.tracklist:
            track.alive = False
        logging.info('all %s tracks killed for final saving',
                     self.SIGN_TYPE.replace('one', 'onic').lower())

    def set_global_attr_netcdf(
            self, h_nc, directory, grd=None, YMIN=None, YMAX=None, MMIN=None,
            MMAX=None, MODEL=None, SIGMA_LEV=None, rho_ntr=None):
        h_nc.title = self.SIGN_TYPE + ' eddy tracks'
        h_nc.directory = directory
        h_nc.product = self.product
        h_nc.DAYS_BTWN_RECORDS = np.float64(self.DAYS_BTWN_RECORDS)
        h_nc.TRACK_DURATION_MIN = np.float64(self.TRACK_DURATION_MIN)

        if 'Q' in self.DIAGNOSTIC_TYPE:
            h_nc.Q_parameter_contours = self.qparameter
        elif 'SLA' in self.DIAGNOSTIC_TYPE:
            h_nc.CONTOUR_PARAMETER = self.CONTOUR_PARAMETER
            h_nc.SHAPE_ERROR = self.SHAPE_ERROR[0]
            h_nc.PIXEL_THRESHOLD = self.PIXEL_THRESHOLD

        if self.SMOOTHING in locals():
            h_nc.SMOOTHING = np.str(self.SMOOTHING)
            h_nc.SMOOTH_FAC = np.float64(self.SMOOTH_FAC)
        else:
            h_nc.SMOOTHING = 'None'

        h_nc.i_0 = np.int32(self.i_0)
        h_nc.i_1 = np.int32(self.i_1)
        h_nc.j_0 = np.int32(self.j_0)
        h_nc.j_1 = np.int32(self.j_1)

        # h_nc.lonmin = grd.lonmin
        # h_nc.lonmax = grd.lonmax
        # h_nc.latmin = grd.latmin
        # h_nc.latmax = grd.latmax

        if 'ROMS' in self.product:
            h_nc.ROMS_GRID = grd.GRDFILE
            h_nc.MODEL = MODEL
            h_nc.YMIN = np.int32(YMIN)
            h_nc.YMAX = np.int32(YMAX)
            h_nc.MMIN = np.int32(MMIN)
            h_nc.MMAX = np.int32(MMAX)
            h_nc.SIGMA_LEV_index = np.int32(SIGMA_LEV)

            if 'ip_roms' in MODEL:
                h_nc.rho_ntr = rho_ntr

        h_nc.EVOLVE_AMP_MIN = self.EVOLVE_AMP_MIN
        h_nc.EVOLVE_AMP_MAX = self.EVOLVE_AMP_MAX
        h_nc.EVOLVE_AREA_MIN = self.EVOLVE_AREA_MIN
        h_nc.EVOLVE_AREA_MAX = self.EVOLVE_AREA_MAX

    def create_variable(self, handler_nc, kwargs_variable,
                        attr_variable):
        var = handler_nc.createVariable(
            zlib=True,
            complevel=1,
            **kwargs_variable)
        for attr, attr_value in attr_variable.iteritems():
            var.setncattr(attr, attr_value)
#         var.setncattr('min', var[:].min())
#         var.setncattr('max', var[:].max())

    def create_netcdf(self, directory, savedir,
                      grd=None, YMIN=None, YMAX=None,
                      MMIN=None, MMAX=None, MODEL=None,
                      SIGMA_LEV=None, rho_ntr=None):
        """
        Create netcdf file same style as Chelton etal (2011)
        """
        if not self.TRACK_EXTRA_VARIABLES:
            self.savedir = savedir
        else:
            self.savedir = savedir.replace('.h_nc', '_ARGO_enabled.h_nc')
        h_nc = Dataset(self.savedir, 'w', format='NETCDF4')
        self.set_global_attr_netcdf(h_nc, directory, grd, YMIN, YMAX, MMIN,
                                    MMAX, MODEL, SIGMA_LEV, rho_ntr)

        # Create dimensions
        h_nc.createDimension('Nobs', None)  # len(Eddy.tracklist))

        # Create variables
        list_variable_to_create = [
            'track',
            'n',
            'lon',
            'lat',
            'type_cyc',
            'amplitude',
            'radius',
            'speed_radius',
            'eke',
            'radius_e',
            ]

        # AVISO or INTERANNUAL ROMS solution
        list_variable_to_create.append('time' if self.INTERANNUAL else
                                       'ocean_time')

        for key_name in list_variable_to_create:
            self.create_variable(
                h_nc,
                dict(varname=VAR_DESCR[key_name]['nc_name'],
                     datatype=VAR_DESCR[key_name]['nc_type'],
                     dimensions=VAR_DESCR[key_name]['nc_dims']),
                VAR_DESCR[key_name]['nc_attr']
                )

        if 'Q' in self.DIAGNOSTIC_TYPE:
            h_nc.createVariable('qparameter', 'f4', ('Nobs'),
                                fill_value=self.fillval)

        if 'Q' in self.DIAGNOSTIC_TYPE:
            h_nc.variables['A'].units = \
                'None, normalised vorticity (abs(xi)/f)'

        if 'Q' in self.DIAGNOSTIC_TYPE:
            h_nc.variables['qparameter'].units = 's^{-2}'
        if self.TRACK_EXTRA_VARIABLES:
            h_nc.createDimension('contour_points', None)
            h_nc.createDimension('uavg_contour_count',
                                 np.int(self.CONTOUR_PARAMETER.size * 0.333))
            h_nc.createVariable(
                'contour_e', 'f4',
                ('contour_points', 'Nobs'), fill_value=self.fillval)
            h_nc.createVariable(
                'contour_s', 'f4',
                ('contour_points', 'Nobs'), fill_value=self.fillval)
            h_nc.createVariable(
                'uavg_profile', 'f4',
                ('uavg_contour_count', 'Nobs'), fill_value=self.fillval)
            h_nc.createVariable(
                'shape_error', 'f4',
                ('Nobs'), fill_value=self.fillval)

            h_nc.variables['contour_e'].long_name = (
                'positions of effective contour points')
            h_nc.variables['contour_e'].description = (
                'lons/lats of effective contour points; lons (lats) in '
                'first (last) half of vector')
            h_nc.variables['contour_s'].long_name = (
                'positions of speed-based contour points')
            h_nc.variables['contour_s'].description = (
                'lons/lats of speed-based contour points; lons (lats) in first'
                ' (last) half of vector')
            h_nc.variables['uavg_profile'].long_name = 'radial profile of uavg'
            h_nc.variables['uavg_profile'].description = (
                'all uavg values from effective contour inwards to '
                'smallest inner contour (pixel == 1)')
            h_nc.variables['shape_error'].units = '%'

        h_nc.close()

    def _reduce_inactive_tracks(self):
        """
        Remove dead tracks
        """
        # start_time = time.time()
        for track in self.tracklist:
            if not track.alive:
                track.lon = []
                track.lat = []
                track.amplitude = []
                track.uavg = []
                track.teke = []
                track.radius_s = []
                track.radius_e = []
                track.ocean_time = []
                if self.TRACK_EXTRA_VARIABLES:
                    track.contour_e = []
                    track.contour_s = []
                    track.uavg_profile = []
                    track.shape_error = []
        return

    def _remove_inactive_tracks(self):
        """
        Remove dead tracks from self.tracklist and
        return indices to active tracks.
        """
        new_tracklist = []
        for track in self.tracklist:
            new_tracklist.append(track.alive)
        # print new_tracklist
        alive_inds = np.nonzero(new_tracklist)[0]
        # print alive_inds.shape
        tracklist = np.array(self.tracklist)[alive_inds]
        self.tracklist = tracklist.tolist()
        return alive_inds

    def write2netcdf(self, rtime, stopper=0):
        """
        Write inactive tracks to netcdf file.
        'ncind' is important because prevents writing of
        already written tracks.
        Each inactive track is 'emptied' after saving

        rtime - current timestamp
        stopper - dummy value (either 0 or 1)
        """
        rtime += stopper
        tracks2save = np.array([self.get_inactive_tracks(rtime)])
        DBR = self.DAYS_BTWN_RECORDS

        if np.any(tracks2save):
            # Note, this could break if all eddies become inactive at same time

            with Dataset(self.savedir, 'a') as nc:

                for i in np.nditer(tracks2save):

                    # saved2nc is a flag indicating if track[i] has been saved
                    if (not self.tracklist[i].saved2nc) and \
                       (np.all(self.tracklist[i].ocean_time)):

                        tsize = len(self.tracklist[i].lon)

                        if (tsize >= self.TRACK_DURATION_MIN / DBR) \
                                and tsize >= 1.:
                            lon = np.array([self.tracklist[i].lon])
                            lat = np.array([self.tracklist[i].lat])
                            amp = np.array([self.tracklist[i].amplitude])
                            uavg = np.array(
                                [self.tracklist[i].uavg]) * 100.  # to cm/s
                            teke = np.array([self.tracklist[i].teke])
                            radius_s = np.array(
                                [self.tracklist[i].radius_s]) * 1e-3  # to km
                            radius_e = np.array(
                                [self.tracklist[i].radius_e]) * 1e-3  # to km
                            n = np.arange(tsize, dtype=np.int32)
                            track = np.full(tsize, self.ch_index)
                            cyc = 1 if 'Anticyclonic' in self.SIGN_TYPE else -1
#                             track_max_val = np.array(
#                                 [nc.variables['track'].max_val,
#                                  np.int32(self.ch_index)]).max()
                            # eddy_duration = np.array([self.tracklist[i].ocean_time]).ptp()

                            sl_copy = slice(self.ncind, self.ncind + tsize)
                            nc.variables['cyc'][sl_copy] = cyc
                            nc.variables['lon'][sl_copy] = lon
                            nc.variables['lat'][sl_copy] = lat
                            nc.variables['A'][sl_copy] = amp
                            nc.variables['U'][sl_copy] = uavg
                            nc.variables['Teke'][sl_copy] = teke
                            nc.variables['L'][sl_copy] = radius_s
                            nc.variables['radius_e'
                                         ][sl_copy] = radius_e
                            nc.variables['n'][sl_copy] = n
                            nc.variables['track'][sl_copy] = track
#                             nc.variables['track'].max_val = track_max_val
                            # h_nc.variables['eddy_duration'][sl_copy] = eddy_duration

                            if 'ROMS' in self.product:
                                # temp = np.array([self.tracklist[i].temp])
                                # salt = np.array([self.tracklist[i].salt])
                                pass
                                # h_nc.variables['temp'][sl_copy] = temp
                                # h_nc.variables['salt'][sl_copy] = salt

                            if self.INTERANNUAL:
                                # We add 1 because 'j_1' is an integer in
                                # ncsavefile; julian day midnight has .5
                                # i.e., dt.julian2num(2448909.5) -> 727485.0
                                j_1 = num2julian(np.array([
                                    self.tracklist[i].ocean_time])) + 1
                                nc.variables['j1'][sl_copy] = j_1

                            else:
                                ocean_time = np.array(
                                    [self.tracklist[i].ocean_time])
                                nc.variables['ocean_time'
                                             ][sl_copy] = ocean_time

                            if self.TRACK_EXTRA_VARIABLES:
                                shape_error = np.array(
                                    [self.tracklist[i].shape_error])
                                nc.variables['shape_error'
                                             ][sl_copy] = shape_error

                                for j in np.arange(tend - self.ncind):
                                    jj = j + self.ncind
                                    contour_e_arr = np.array(
                                        [self.tracklist[i].contour_e[j]]).ravel()
                                    contour_s_arr = np.array(
                                        [self.tracklist[i].contour_s[j]]).ravel()
                                    uavg_profile_arr = np.array(
                                        [self.tracklist[i].uavg_profile[j]]).ravel()
                                    nc.variables['contour_e'][:contour_e_arr.size, jj] = contour_e_arr
                                    nc.variables['contour_s'][:contour_s_arr.size, jj] = contour_s_arr
                                    nc.variables['uavg_profile'][:uavg_profile_arr.size, jj] = uavg_profile_arr

                            # Flag indicating track[i] is now saved
                            self.tracklist[i].saved2nc = True
                            self.ncind += tsize
                            self.ch_index += 1
                            nc.sync()

        # Print final message and return
        if stopper:
            logging.info('All %ss saved',
                         self.SIGN_TYPE.replace('one', 'onic').lower())
            return

        # Get index to first currently active track
        # try:
            # lasti = self.get_active_tracks(rtime)[0]
        # except Exception:
            # lasti = None

        # Remove inactive tracks
        # NOTE: line below used to be below clipping lines below
        # self._reduce_inactive_tracks()
        alive_i = self._remove_inactive_tracks()
        # Clip the tracklist,
        # removes all dead tracks preceding first currently active track
        # self.tracklist = self.tracklist[alive_i:]
        self.index = len(self.tracklist)  # adjust index accordingly

        # Update old_lon and old_lat...
        self.old_lon = np.array(self.new_lon)[alive_i].tolist()
        self.old_lat = np.array(self.new_lat)[alive_i].tolist()
        self.old_radii_s = np.array(self.new_radii_s)[alive_i].tolist()
        self.old_radii_e = np.array(self.new_radii_e)[alive_i].tolist()
        self.old_amp = np.array(self.new_amp)[alive_i].tolist()
        self.old_uavg = np.array(self.new_uavg)[alive_i].tolist()
        self.old_teke = np.array(self.new_teke)[alive_i].tolist()

        self.new_lon = []
        self.new_lat = []
        self.new_radii_s = []
        self.new_radii_e = []
        self.new_amp = []
        self.new_uavg = []
        self.new_teke = []
        self.new_time = []

        if 'ROMS' in self.product:
            # self.old_temp = self.new_temp[alive_i:]
            # self.old_salt = self.new_salt[alive_i:]
            pass
            # self.new_temp = []
            # self.new_salt = []

        if self.TRACK_EXTRA_VARIABLES:
            self.old_contour_e = list(self.new_contour_e[alive_i])
            self.old_contour_s = list(self.new_contour_s[alive_i])
            self.old_uavg_profile = list(self.new_uavg_profile[alive_i])
            self.old_shape_error = self.new_shape_error[alive_i]

            self.new_contour_e = []
            self.new_contour_s = []
            self.new_uavg_profile = []
            self.new_shape_error = []

        return self

    def insert_at_index(self, xarr, ind, x):
        """
        This the same as Matlab's native functionality:
            x(3)=4 gives [0  0  4] and then
            x(5)=7 gives [0  0  4  0  7]
        """
        try:
            x = x[0]
        except Exception:
            pass

        val = getattr(self, xarr)
        try:
            val[ind] = x
        except:
            val.extend([0] * (ind - len(val) + 1))
            val[ind] = x
        setattr(self, xarr, val)

    def set_bounds(self, contlon, contlat, grd):
        """
        Get indices to a bounding box around the eddy
        WARNING won't work for a rotated grid
        """
        lonmin, lonmax = contlon.min(), contlon.max()
        latmin, latmax = contlat.min(), contlat.max()

        self.imin, self.jmin = nearest(lonmin, latmin, grd.lon[0], grd.lat[:, 0])
        self.imax, self.jmax = nearest(lonmax, latmax, grd.lon[0], grd.lat[:, 0])

        # For indexing the mins must not be less than zero
        self.imin = max(self.imin - self.PAD, 0)
        self.jmin = max(self.jmin - self.PAD, 0)
        self.imax += self.PAD + 1
        self.jmax += self.PAD + 1
        return self

    def set_mask_eff(self, contour, grd):
        """
        Set points within bounding box around eddy and calculate
        mask for effective contour
        """
        self.points = np.array([grd.lon[self.jmin:self.jmax,
                                        self.imin:self.imax].ravel(),
                                grd.lat[self.jmin:self.jmax,
                                        self.imin:self.imax].ravel()]).T
        # NOTE: Path.contains_points requires matplotlib 1.2 or higher
        self.mask_eff_1d = contour.contains_points(self.points)
        self.mask_eff_sum = self.mask_eff_1d.sum()
        return self

    def reshape_mask_eff(self, grd):
        """
        """
        shape = grd.lon[self.jmin:self.jmax, self.imin:self.imax].shape
        self.mask_eff = self.mask_eff_1d.reshape(shape)


class RossbyWaveSpeed (object):

    def __init__(self, THE_DOMAIN, grd, RW_PATH=None):
        """
        Instantiate the RossbyWaveSpeed object
        """
        self.the_domain = THE_DOMAIN
        self.m_val = grd.m_val
        self.EARTH_RADIUS = grd.EARTH_RADIUS
        self.zero_crossing = grd.zero_crossing
        self.RW_PATH = RW_PATH
        self._tree = None
        if self.the_domain in ('Global', 'Regional', 'ROMS'):
            assert self.RW_PATH is not None, \
                'Must supply a path for the Rossby deformation radius data'
            data = np.loadtxt(RW_PATH)
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
        self.distance = np.empty(1)
        self.beta = np.empty(1)
        self.r_spd_long = np.empty(1)
        self.start = True
        self.pio180 = np.pi / 180.

    def __getstate__(self):
        """
        Needed for Pickle
        """
        result = self.__dict__.copy()
        result.pop('_tree')
        return result

    def __setstate__(self, thedict):
        """
        Needed for Pickle
        """
        self.__dict__ = thedict
        self._make_kdtree()

    def get_rwdistance(self, xpt, ypt, DAYS_BTWN_RECORDS):
        """
        Return the distance required by SearchEllipse
        to construct a search ellipse for eddy tracking.
        """
        def get_lon_lat(xpt, ypt):
            """
            """
            lon, lat = self.m_val(xpt, ypt, inverse=True)
            lon, lat = np.round(lon, 2), np.round(lat, 2)
            if lon < 0.:
                lon = "".join((str(lon), 'W'))
            elif lon >= 0:
                lon = "".join((str(lon), 'E'))
            if lat < 0:
                lat = "".join((str(lat), 'S'))
            elif lat >= 0:
                lat = "".join((str(lat), 'N'))
            return lon, lat

        if self.the_domain in ('Global', 'Regional', 'ROMS'):
            # print 'xpt, ypt', xpt, ypt
            self.distance[:] = self._get_rlongwave_spd(xpt, ypt)
            self.distance *= 86400.
#             if self.the_domain in 'ROMS':
#                 self.distance *= 1.5

        elif 'BlackSea' in self.the_domain:
            self.distance[:] = 15000.  # e.g., Blokhina & Afanasyev, 2003

        elif 'MedSea' in self.the_domain:
            self.distance[:] = 20000.

        else:
            Exception  # Unknown the_domain

        if self.start:
            lon, lat = get_lon_lat(xpt, ypt)
            if 'Global' in self.the_domain:
                logging.info('--------- setting ellipse for first tracked '
                             'eddy at %s, %s in the %s domain',
                             lon, lat, self.the_domain)
                c = np.abs(self._get_rlongwave_spd(xpt, ypt))[0]
                logging.info('with extratropical long baroclinic '
                             'Rossby wave phase speed of %s m/s',
                             c)
            elif self.the_domain in ('BlackSea', 'MedSea'):
                logging.info('setting search radius of %s m for '
                             'first tracked eddy at %s, %s in the %s domain',
                             self.distance[0], lon, lat, self.the_domain)
            else:
                Exception
            self.start = False

        self.distance = np.abs(self.distance)
        return self.distance * DAYS_BTWN_RECORDS

    def _make_subset(self):
        """
        Make a subset of _defrad data over the domain.
        If 'Global' is defined then widen the domain.
        """
        pad = 1.5  # degrees
        LONMIN, LONMAX, LATMIN, LATMAX = self.limits

        if self.zero_crossing:
            ieast, iwest = (((self._lon + 360.) <= LONMAX + pad),
                            (self._lon > LONMIN + pad))
            self._lon[ieast] += 360.
            lloi = iwest + ieast
        else:
            lloi = np.logical_and(self._lon >= LONMIN - pad,
                                  self._lon <= LONMAX + pad)
        lloi *= np.logical_and(self._lat >= LATMIN - pad,
                               self._lat <= LATMAX + pad)
        self._lon = self._lon[lloi]
        self._lat = self._lat[lloi]
        self._defrad = self._defrad[lloi]

        if 'Global' in self.the_domain:
            lloi = self._lon > 260.
            self._lon = np.append(self._lon, self._lon[lloi] - 360.)
            self._lat = np.append(self._lat, self._lat[lloi])
            self._defrad = np.append(self._defrad, self._defrad[lloi])

        self.x, self.y = self.m_val(self._lon, self._lat)
        return self

    def _make_kdtree(self):
        """
        Compute KDE tree for nearest indices.
        """
        points = np.vstack([self.x, self.y]).T
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
        self.r_spd_long[:] = self._get_defrad(xpt, ypt)
        self.r_spd_long *= 1000.  # km to m
        self.r_spd_long **= 2
        self.beta[:] = np.average(self._lat[self.i],
                                  weights=self._weights)  # lat
        self.beta[:] = np.cos(self.pio180 * self.beta)
        self.beta *= 1458e-7  # 1458e-7 ~ (2 * 7.29*10**-5)
        self.beta /= self.EARTH_RADIUS
        self.r_spd_long *= -self.beta
        return self.r_spd_long


class SearchEllipse (object):
    """
    Class to construct a search ellipse/circle around a specified point.
    See CSS11 Appendix B.4. "Automated eddy tracking" for details.
    """
    def __init__(self, THE_DOMAIN, grd, DAYS_BTWN_RECORDS, RW_PATH=None):
        """
        Set the constant dimensions of the search ellipse.
        Instantiate a RossbyWaveSpeed object

        Arguments:

          *the_domain*: string
            Refers to the_domain specified in yaml configuration file

          *grd*: An AvisoGrid or RomsGrid object.

          *DAYS_BTWN_RECORDS*: integer
            Constant defined in yaml configuration file.

          *RW_PATH*: string
            Path to rossrad.dat file, specified in yaml configuration file.
        """
        self.the_domain = THE_DOMAIN
        self.DAYS_BTWN_RECORDS = DAYS_BTWN_RECORDS
        self.e_w_major = self.DAYS_BTWN_RECORDS * 3e5 / 7.
        self.n_s_minor = self.DAYS_BTWN_RECORDS * 15e4 / 7.
        self.semi_n_s_minor = 0.5 * self.n_s_minor
        self.rwv = RossbyWaveSpeed(THE_DOMAIN, grd, RW_PATH=RW_PATH)
        # debug; not relevant for MedSea / BlackSea
        # self.rwv.view_grid_subset()
        self.rw_c = np.empty(1)
        self.rw_c_mod = np.empty(1)
        self.rw_c_fac = 1.75

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
        self.rw_c_mod[:] = 1.75

        if self.the_domain in ('Global', 'Regional', 'ROMS'):
            self.rw_c[:] = self.rwv.get_rwdistance(xpt, ypt,
                                                   self.DAYS_BTWN_RECORDS)
            self.rw_c_mod *= self.rw_c
            self.rw_c_mod[:] = np.array([self.rw_c_mod,
                                         self.semi_n_s_minor]).max()
            self.rw_c_mod *= 2.
            self._set_global_ellipse()

        elif self.the_domain in ('BlackSea', 'MedSea'):
            self.rw_c[:] = self.rwv.get_rwdistance(xpt, ypt,
                                                   self.DAYS_BTWN_RECORDS)
            self.rw_c_mod *= self.rw_c
            self._set_black_sea_ellipse()

        else:
            Exception

        return self


if __name__ == '__main__':

    lon_ini = -10.
    lat_ini = 25.
    time_ini = 247893
    index = 0

    trackA = track_list(index, lon_ini, lat_ini, time_ini, 0, 0)
    print 'trackA lon:', trackA.tracklist[0].lon

    # update track 0
    trackA.update_track(0, -22, 34, 333344, 0, 0)
    trackA.update_track(0, -87, 37, 443344, 0, 0)
    trackA.update_track(0, -57, 57, 543344, 0, 0)
    print 'trackA lon:', trackA.tracklist[0].lon

    # start a new track
    trackA.append_list(-33, 45, 57435, 0, 0)
    print '\ntrackA lat:', trackA.tracklist[1].lat

    trackA.update_track(1, -32, 32, 4453344, 0, 0)
    print 'trackA lat:', trackA.tracklist[1].lat

    # Pickle
    output = open('data.pkl', 'wb')
    pickle.dump(trackA, output)
    output.close()

    # Unpickle
    pkl_file = open('data.pkl', 'rb')
    data1 = pickle.load(pkl_file)
    pkl_file.close()
