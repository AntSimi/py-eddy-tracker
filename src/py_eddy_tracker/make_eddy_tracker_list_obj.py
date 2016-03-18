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
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import RectBivariateSpline
from scipy.sparse import coo_matrix
from .tools import distance_vector
from . import VAR_DESCR
import numpy as np
import logging
from .py_eddy_tracker_property_classes import EddiesObservations


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


def uniform_resample(x_val, y_val, **kwargs):
    """
    Resample contours to have (nearly) equal spacing
       x_val, y_val    : input contour coordinates
       num_fac : factor to increase lengths of output coordinates
       method  : currently only 'interp1d' or 'Akima'
                 (Akima is slightly slower, but may be more accurate)
       extrapolate : IS NOT RELIABLE (sometimes nans occur)
    """
    method = kwargs.get('method', 'interp1d')
    extrapolate = kwargs.get('extrapolate', None)
    num_fac = kwargs.get('num_fac', 2)

    # Get distances
    dist = np.empty_like(x_val)
    dist[0] = 0
    distance_vector(
        x_val[:-1], y_val[:-1], x_val[1:], y_val[1:], dist[1:])
    dist.cumsum(out=dist)
    # Get uniform distances
    d_uniform = np.linspace(0,
                            dist[-1],
                            num=dist.size * num_fac,
                            endpoint=True)

    # Do 1d interpolations
    if 'interp1d' == method:
        x_new = np.interp(d_uniform, dist, x_val)
        y_new = np.interp(d_uniform, dist, y_val)
    elif 'akima' == method:
        xfunc = Akima1DInterpolator(dist, x_val)
        yfunc = Akima1DInterpolator(dist, y_val)
        x_new = xfunc(d_uniform, extrapolate=extrapolate)
        y_new = yfunc(d_uniform, extrapolate=extrapolate)

    else:
        raise Exception()

    return x_new, y_new


class EddiesTrack(list):

    @property
    def last_date(self):
        return np.array([item.obs['time'][-1] for item in self])

    def set_active(self, flags):
        if type(flags) is bool:
            for item in self:
                item.active = flags
        else:
            for i, item in enumerate(self):
                item.active = flags[i]

    @property
    def active(self):
        return np.array([item.active for item in self], dtype=bool)


class TrackList (object):
    """
    Class that holds list of eddy tracks:
        tracklist - the list of 'track' objects
        qparameter: Q parameter range used for contours
        new_lon, new_lat: new lon/lat centroids
        old_lon, old_lat: old lon/lat centroids
        index:   index of eddy in track_list
    """
    def __init__(self, sign_type, grd, search_ellipse, **kwargs):
        """
        Initialise the list 'tracklist'
        """
        self._grd = grd
        self.sign_type = sign_type
        self.save_dir = None

        self.diagnostic_type = kwargs.get('DIAGNOSTIC_TYPE', 'SLA')
        self.the_domain = kwargs.get('THE_DOMAIN', 'Regional')
        self.track_duration_min = kwargs.get('TRACK_DURATION_MIN', 28)
        self.track_extra_variables = kwargs.get('TRACK_EXTRA_VARIABLES', False)
        self.interannual = kwargs.get('INTERANNUAL', True)
        self.separation_method = kwargs.get('SEPARATION_METHOD', 'ellipse')
        self.smoothing = kwargs.get('SMOOTHING', True)
        self.max_local_extrema = kwargs.get('MAX_LOCAL_EXTREMA', 1)
        self.interp_method = kwargs.get('INTERP_METHOD', 'RectBivariate')
        # NOTE: '.copy()' suffix is essential here
        self.contour_parameter = kwargs.get('CONTOUR_PARAMETER').copy()
        self.interval = self.contour_parameter[1] - self.contour_parameter[0]
        if 'Cyclonic' in sign_type:
            self.contour_parameter *= -1
        self.shape_error = kwargs.get('SHAPE_ERROR', 55.)
        self.days_btwn_records = kwargs.get('DAYS_BTWN_RECORDS')
        self.radmin = np.float64(kwargs.get('RADMIN', 0.4))
        self.radmax = np.float64(kwargs.get('RADMAX', 4.461))
        self.ampmin = np.float64(kwargs.get('AMPMIN', 1.))
        self.ampmax = np.float64(kwargs.get('AMPMAX', 150.))
        self.evolve_amp_min = np.float64(kwargs.get('EVOLVE_AMP_MIN', .0005))
        self.evolve_amp_max = np.float64(kwargs.get('EVOLVE_AMP_MAX', 500))
        self.evolve_area_min = np.float64(kwargs.get('EVOLVE_AREA_MIN', .0005))
        self.evolve_area_max = np.float64(kwargs.get('EVOLVE_AREA_MAX', 500))
        self.rad0 = np.pi * np.float64(kwargs.get('RAD0', 60000.)) ** 2
        self.amp0 = np.float64(kwargs.get('AMP0', 2.))
        self.dist0 = np.float64(kwargs.get('DIST0', 25000.))

        self.points = np.array([grd.lon.ravel(), grd.lat.ravel()]).T

        self.sla = None

        self.final_tracklist = EddiesTrack()
        self.new_observations = EddiesObservations(self.track_extra_variables)
        self.tmp_observations = EddiesObservations(self.track_extra_variables)
        self.old_observations = EddiesObservations(self.track_extra_variables)
        self.index = 0  # counter
        # index to write to h_nc files, will increase and increase
        self.ncind = 0
        self.ch_index = 0  # index for Chelton style h_nc files
        self.pad = 2
        self.search_ellipse = search_ellipse
        self.pixel_threshold = None
        # Check for a correct configuration
        assert self.product in (
            'ROMS', 'AVISO'), 'Unknown string in *product* parameter'

    def accounting(self, obs, index, new_eddy):
        """
        Accounting for new or old eddy (either cyclonic or anticyclonic)
          eddy  : eddy_tracker.final_tracklist object
          index    : index to current old
          new_eddy   : flag indicating a new eddy
        """
        if new_eddy:
            # it's a new eddy
            # We extend the range of array to old_ind
            # self.new_observations.insert_observations(obs, self.index)
            self.new_observations.append(obs)
            self.final_tracklist.append(obs)
            self.index += 1
        else:
            # it's an old (i.e., active) eddy
            # self.new_observations.insert_observations(obs, index)
            self.new_observations.append(obs)
            self.final_tracklist[index] += obs

    def track_eddies(self, first_record):
        """
        Track the eddies. First a distance matrix is calculated between
        all new and old eddies. Then loop through each old eddy, sorting the
        distances, and selecting that/those within range.
        """
        far_away = 1e9
#         self.search_ellipse.xpt = -10
        self.search_ellipse.set_search_ellipse(-10, -10)
        self.search_ellipse.view_search_ellipse()
        exit()

        # We will need these in m_val for ellipse method below
        old_x, old_y = self.m_val(self.old_observations.obs['lon'],
                                  self.old_observations.obs['lat'])
        new_x, new_y = self.m_val(self.tmp_observations.obs['lon'],
                                  self.tmp_observations.obs['lat'])

        # Use haversine distance for distance matrix between every old and new
        # eddy
        dist_mat = self.old_observations.distance_matrix(self.tmp_observations)
        dist_mat_copy = dist_mat.copy()

        # Prepare distance matrix for sparse operation
        # NOTE: need to find optimal (+dynamic) choice of filter here
        dist_mat[dist_mat > 35000.] = 0

        # Use a coordinate format (COO) sparse matrix
        # https://scipy-lectures.github.io/advanced/scipy_sparse/coo_matrix.html
        sparse_mat = coo_matrix((dist_mat))
        old_sparse_inds = sparse_mat.row
        new_sparse_inds = sparse_mat.col

        # *new_eddy_inds* contains indices to every newly identified eddy
        # that are initially set to True on the assumption that it is a new
        # eddy, i.e, it has just been born.
        # Later some will be set to False, indicating the continuation
        # of an existing eddy.
        new_eddy_inds = np.ones(len(self.tmp_observations), dtype=bool)
        new_eddy = False

        # Loop over the old eddies looking for active eddies
        # for old_ind in np.arange(dist_mat.shape[0]):
        for old_ind in old_sparse_inds:
            dist_arr = np.array([])
            new_obs = EddiesObservations()
            backup_ind = np.array([], dtype=np.int16)
            '''
            # See http://stackoverflow.com/questions/11528078/determining-
            duplicate-values-in-an-array
            non_unique_inds = np.setdiff1d(np.arange(len(dist_mat[old_ind])),
                                               np.unique(dist_mat[old_ind],
                                                        return_index=True)[1])
            # Move non_unique_inds far away
            dist_mat[old_ind][non_unique_inds] = far_away  # km
            '''
            # Make an ellipse at current old_eddy location
            # (See CSS11 sec. B4, pg. 208)
            if 'ellipse' in self.separation_method:
                self.search_ellipse.set_search_ellipse(old_x[old_ind],
                                                       old_y[old_ind])

            # Loop over separation distances between old and new
            for new_ind in new_sparse_inds:
                new_dist = dist_mat[old_ind, new_ind]
                within_range = False
                if new_dist < self.search_ellipse.rw_c_mod:  # far_away:
                    if 'ellipse' in self.separation_method:
                        if self.search_ellipse.ellipse_path.contains_point(
                                (new_x[new_ind], new_y[new_ind])):
                            within_range = True
                    elif 'sum_radii' in self.separation_method:
                        sep_dist = (
                            self.tmp_observations.obs['radius_e'][new_ind] +
                            self.old_observations.obs['radius_e'][old_ind]
                            ) * self.sep_dist_fac
                        within_range = new_dist <= sep_dist
                    else:
                        raise Exception()
                # Pass only the eddies within ellipse or sep_dist
                if within_range:
                    dist_arr = np.r_[dist_arr, new_dist]
                    new_obs += self.tmp_observations.index(new_ind)
                    backup_ind = np.r_[backup_ind, new_ind]
                    # An old (active) eddy has been detected, so
                    # corresponding new_eddy_inds set to False
                    new_eddy_inds[
                        np.nonzero(
                            self.tmp_observations.obs['lon'] ==
                            self.tmp_observations.obs['lon'][new_ind])] = False
                    dist_mat[:, new_ind] = far_away  # km
            # Only one eddy within range
            if dist_arr.size == 1:  # then update the eddy track only
                self.accounting(
                    obs=new_obs,
                    index=old_ind,
                    new_eddy=first_record or new_eddy,
                    )
            # More than one eddy within range
            elif dist_arr.size > 1:
                # Loop to find the right eddy
                delta_area = np.array([])
                delta_amp = np.array([])
                for i in xrange(dist_arr.size):
                    # Choice of using effective or speed-based...
                    delta_area_tmp = np.array(
                        [np.pi *
                         self.old_observations.obs['radius_e'][old_ind] ** 2,
                         np.pi * new_rd_e[i] ** 2]
                        ).ptp()
                    delta_amp_tmp = np.array(
                        [self.old_observations.obs['amplitude'][old_ind],
                         new_am[i]]
                        ).ptp()
                    delta_area = np.r_[delta_area, delta_area_tmp]
                    delta_amp = np.r_[delta_amp, delta_amp_tmp]

                # This from Penven etal (2005)
                delta_x = ((delta_area / self.area_0) ** 2 +
                           (delta_amp / self.amp_0) ** 2 +
                           (dist_arr / self.dist_0) ** 2) ** .5

                d_x = delta_x.argsort()
                dx0 = d_x[0]  # index to the nearest eddy
                dx_unused = d_x[1:]  # index/indices to the unused eddy/eddies

                logging.debug('delta_area %s', delta_area)
                logging.debug('delta_amp %s', delta_amp)
                logging.debug('dist_arr %s', dist_arr)
                logging.debug('d_x %s', d_x)
                logging.debug('dx0 %s', dx0)
                logging.debug('dx_unused %s', dx_unused)

                # Update eddy track dx0
                new_obs += self.tmp_observations.index(dx0)
                self.accounting(
                    obs=new_obs,
                    index=old_ind,
                    new_eddy=first_record or new_eddy,
                    )

                # Use backup_ind to reinsert distances into dist_mat for the
                # unused eddy/eddies
                for bind in backup_ind[dx_unused]:
                    dist_mat[:, bind] = dist_mat_copy[:, bind]
                    new_eddy_inds[bind] = True

        # Finished looping over old eddy tracks

        # Now we need to add new eddies defined by new_eddy_inds
        if np.any(new_eddy_inds):
            logging.debug('adding %s new eddies', new_eddy_inds.sum())
            for neind, a_new_eddy in enumerate(new_eddy_inds):
                if a_new_eddy:  # Update the eddy tracks
                    self.accounting(
                        obs=self.tmp_observations.index(neind),
                        index=None,
                        new_eddy=True,
                        )

    @property
    def m_val(self):
        return self._grd.m_val

    @property
    def product(self):
        return self._grd.product

    @property
    def fillval(self):
        return self._grd.fillval

    @property
    def i_0(self):
        return self._grd.slice_i.start

    @property
    def i_1(self):
        return self._grd.slice_i.stop

    @property
    def j_0(self):
        return self._grd.slice_j.start

    @property
    def j_1(self):
        return self._grd.slice_j.stop

    def update_eddy_properties(self, properties):
        """
        Append new variable values to track arrays
        """
        self.tmp_observations += properties

    def reset_holding_variables(self):
        """
        Reset temporary holding variables to empty arrays
        """
        self.tmp_observations.reset()

    def set_old_variables(self):
        """
        Pass all values at time k+1 to k
        """
        self.old_observations = self.tmp_observations

    def get_inactive_index_tracks(self, rtime):
        """
        Return list of indices to inactive tracks.
        This call also identifies and removes
        inactive tracks
        """
        active = self.final_tracklist.last_date == int(rtime)
        self.final_tracklist.set_active(active)
        return np.where(-active)[0]

    def kill_all_tracks(self):
        """
        Mark all tracks as not alive
        """
        self.final_tracklist.set_active(False)
        logging.info('all %s tracks killed for final saving',
                     self.sign_type.replace('one', 'onic').lower())

    def set_global_attr_netcdf(
            self, h_nc, directory, grd=None, ymin=None, ymax=None, mmin=None,
            mmax=None, model=None, sigma_lev=None, rho_ntr=None):
        h_nc.title = self.sign_type + ' eddy tracks'
        h_nc.directory = directory
        h_nc.product = self.product
        h_nc.days_btwn_records = self.days_btwn_records
        h_nc.track_duration_min = self.track_duration_min

        if 'Q' in self.diagnostic_type:
            h_nc.Q_parameter_contours = self.qparameter
        elif 'SLA' in self.diagnostic_type:
            h_nc.contour_parameter = self.contour_parameter
            h_nc.shape_error = self.shape_error
            h_nc.pixel_threshold = self.pixel_threshold

        if self.smoothing in locals():
            h_nc.smoothing = self.smoothing
            h_nc.SMOOTH_FAC = self.SMOOTH_FAC
        else:
            h_nc.smoothing = 'None'

        h_nc.i_0 = self.i_0
        h_nc.i_1 = self.i_1
        h_nc.j_0 = self.j_0
        h_nc.j_1 = self.j_1

        if 'ROMS' in self.product:
            h_nc.ROMS_GRID = grd.GRDFILE
            h_nc.MODEL = model
            h_nc.YMIN = ymin
            h_nc.YMAX = ymax
            h_nc.MMIN = mmin
            h_nc.MMAX = mmax
            h_nc.SIGMA_LEV_index = sigma_lev

            if 'ip_roms' in model:
                h_nc.rho_ntr = rho_ntr

        h_nc.evolve_amp_min = self.evolve_amp_min
        h_nc.evolve_amp_max = self.evolve_amp_max
        h_nc.evolve_area_min = self.evolve_area_min
        h_nc.evolve_area_max = self.evolve_area_max

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

    def create_netcdf(self, directory, savedir, grd=None, ymin=None, ymax=None,
                      mmin=None, mmax=None, model=None, sigma_lev=None,
                      rho_ntr=None):
        """
        Create netcdf file same style as Chelton etal (2011)
        """
        if not self.track_extra_variables:
            self.savedir = savedir
        else:
            self.savedir = savedir.replace('.h_nc', '_ARGO_enabled.h_nc')
        h_nc = Dataset(self.savedir, 'w', format='NETCDF4')
        self.set_global_attr_netcdf(h_nc, directory, grd, ymin, ymax, mmin,
                                    mmax, model, sigma_lev, rho_ntr)

        # Create dimensions
        h_nc.createDimension('Nobs', None)  # len(Eddy.final_tracklist))

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

        if self.track_extra_variables:
            h_nc.createDimension('contour_points', None)
            h_nc.createDimension('uavg_contour_count',
                                 np.int(self.contour_parameter.size / 3.))
            list_variable_to_create += [
                'contour_e',
                'contour_s',
                'uavg_profile',
                'shape_error',
                ]

        # AVISO or interannual ROMS solution
        list_variable_to_create.append('time' if self.interannual else
                                       'ocean_time')

        for key_name in list_variable_to_create:
            self.create_variable(
                h_nc,
                dict(varname=VAR_DESCR[key_name]['nc_name'],
                     datatype=VAR_DESCR[key_name]['nc_type'],
                     dimensions=VAR_DESCR[key_name]['nc_dims']),
                VAR_DESCR[key_name]['nc_attr']
                )
        h_nc.close()

    def _reduce_inactive_tracks(self):
        """
        Remove dead tracks
        """
        # start_time = time.time()
        for track in self.final_tracklist:
            if not track.alive:
                track.reset()

    def _remove_inactive_tracks(self):
        """
        Remove dead tracks from self.tracklist and
        return indices to active tracks.
        """
        active = self.final_tracklist.active
        inactive_index = np.where(-active)[0][::-1]
        active_index = np.where(active)[0]
        for index in inactive_index:
            self.final_tracklist.pop(index)
        return active_index

    def write2netcdf(self, rtime, stopper=0):
        """
        Write inactive tracks to netcdf file.
        'ncind' is important because prevents writing of
        already written tracks.
        Each inactive track is 'emptied' after saving

        rtime - current timestamp
        stopper - dummy value (either 0 or 1)
        """
        logging.info('saving to nc %s', self.save_dir)
        rtime += stopper
        index_tracks_2_save = self.get_inactive_index_tracks(rtime)
        if len(index_tracks_2_save):
            h_nc = Dataset(self.savedir, 'a')
            for i in index_tracks_2_save:
                track = self.final_tracklist[i]
                self.ch_index, self.ncind = track.netcdf_incremental_saving(
                    h_nc,
                    self.ncind,
                    self.ch_index,
                    self.track_duration_min / float(self.days_btwn_records),
                    self.sign_type
                    )
            h_nc.close()

        if stopper:
            logging.info('All %ss saved',
                         self.sign_type.replace('one', 'onic').lower())
            return

        # Remove inactive tracks
        alive_i = self._remove_inactive_tracks()

        self.old_observations = self.new_observations.index(alive_i)
        self.new_observations.reset()

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


class IdentificationList (object):
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
        self.rad0 = np.pi * np.float64(kwargs.get('RAD0', 60000.)) ** 2
        self.amp0 = np.float64(kwargs.get('AMP0', 2.))
        self.dist0 = np.float64(kwargs.get('DIST0', 25000.))

        self.points = np.array([grd.lon.ravel(), grd.lat.ravel()]).T

        self.sla = None

        self.observations = EddiesObservations(self.track_extra_variables)
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
            # Iter on variables to create:
            for name, _ in self.observations.dtype:
                logging.debug('Create Variable %s', VAR_DESCR[name]['nc_name'])
                self.create_variable(
                    h_nc,
                    dict(varname=VAR_DESCR[name]['nc_name'],
                         datatype=VAR_DESCR[name]['nc_type'],
                         dimensions=VAR_DESCR[name]['nc_dims']),
                    VAR_DESCR[name]['nc_attr'],
                    self.observations.obs[name],
                    scale_factor=None if 'scale_factor' not in VAR_DESCR[name] else VAR_DESCR[name]['scale_factor'])

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
