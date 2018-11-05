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

observations.py

Version 3.0.0

===========================================================================

"""
from numpy import zeros, empty, nan, arange, where, unique, \
    ma, concatenate, cos, radians, isnan, ones, ndarray, meshgrid, \
    array, interp, int_, int32, round, maximum, floor
from scipy.interpolate  import interp1d
from netCDF4 import Dataset
from py_eddy_tracker.tools import distance_matrix, distance_vector
from shapely.geometry import Polygon
from shapely.geos import TopologicalError
from . import VAR_DESCR, VAR_DESCR_inv
import logging
from datetime import datetime


class GridDataset(object):
    """
    Class to have basic tool on NetCDF Grid
    """
    __slots__ = (
        'x_var',
        'y_var',
        'xinterp',
        'yinterp',
        'x_dim',
        'y_dim',
        'filename',
        'vars',
        'interpolators',
        )
    def __init__(self, filename, x_name, y_name):
        logging.warning('We assume the position of grid is the lower left corner for %s', filename)
        self.filename = filename
        self.vars = dict()
        self.interpolators = dict()
        self.load(x_name, y_name)

    def load(self, x_name, y_name):
        with Dataset(self.filename) as h:
            self.x_var = h.variables[x_name][:]
            self.x_var = concatenate((self.x_var, (2 * self.x_var[-1] - self.x_var[-2],)))
            self.y_var = h.variables[y_name][:]
            self.y_var = concatenate((self.y_var, (2 * self.y_var[-1] - self.y_var[-2],)))
            self.x_dim = h.variables[x_name].dimensions[0]
            self.y_dim = h.variables[y_name].dimensions[0]
        self.init_pos_interpolator()

    def init_pos_interpolator(self):
        self.xinterp = interp1d(self.x_var, range(self.x_var.shape[0]), assume_sorted=True)
        self.yinterp = interp1d(self.y_var, range(self.y_var.shape[0]), assume_sorted=True)

    def grid(self, varname):
        if varname not in self.vars:
            with Dataset(self.filename) as h:
                dims = h.variables[varname].dimensions
                sl = [slice(None) if dim in [self.x_dim, self.y_dim] else 0 for dim in dims]
                self.vars[varname] = h.variables[varname][sl]
                i_x = where(array(dims) == self.x_dim)[0][0]
                i_y = where(array(dims) == self.y_dim)[0][0]
                if i_x > i_y:
                    self.vars[varname] = self.vars[varname].T
        return self.vars[varname]

    def compute_pixel_path(self, x0, y0, x1, y1):
        # First x of grid
        x_ori = self.x_var[0]
        # Float index
        f_x0 = self.xinterp((x0 - x_ori) % 360 + x_ori)
        f_x1 = self.xinterp((x1 - x_ori) % 360 + x_ori)
        f_y0 = self.yinterp(y0)
        f_y1 = self.yinterp(y1)
        # Int index
        i_x0, i_x1 = int32(round(f_x0)), int32(round(f_x1))
        i_y0, i_y1 = int32(round(f_y0)), int32(round(f_y1))

        # Delta index of x
        d_x = i_x1 - i_x0
        nb_x = self.x_var.shape[0] - 1
        d_x = (d_x + nb_x / 2) % nb_x - nb_x / 2
        i_x1 = i_x0 + d_x

        # Delta index of y
        d_y = i_y1 - i_y0

        d_max = int_(maximum(abs(d_x), abs(d_y)))

        # Compute number of pixel which we go trought
        nb_value = int((abs(d_max) + 1).sum())
        # Create an empty array to store value of pixel across the travel
        # Max Index ~65000
        i_g = empty(nb_value, dtype='u2')
        j_g = empty(nb_value, dtype='u2')

        # Index to determine the position in the global array
        ii = 0
        # Iteration on each travel
        for i, delta in enumerate(d_max):
            # If the travel don't cross multiple pixel
            if delta == 0:
                i_g[ii: ii + delta + 1] = i_x0[i]
                j_g[ii: ii + delta + 1] = i_y0[i]
            # Vertical move
            elif d_x[i] == 0:
                sup = -1 if d_y[i] < 0 else 1
                i_g[ii: ii + delta + 1] = i_x0[i]
                j_g[ii: ii + delta + 1] = arange(i_y0[i], i_y1[i] + sup, sup)
            # Horizontal move
            elif d_y[i] == 0:
                sup = -1 if d_x[i] < 0 else 1
                i_g[ii: ii + delta + 1] = arange(i_x0[i], i_x1[i] + sup, sup) % nb_x
                j_g[ii: ii + delta + 1] = i_y0[i]
            # In case of multiple direction
            else:
                a = (i_x1[i] - i_x0[i]) / float(i_y1[i] - i_y0[i])
                if abs(d_x[i]) >= abs(d_y[i]):
                    sup = -1 if d_x[i] < 0 else 1
                    value = arange(i_x0[i], i_x1[i] + sup, sup)
                    i_g[ii: ii + delta + 1] = value % nb_x
                    j_g[ii: ii + delta + 1] = (value - i_x0[i]) / a + i_y0[i]
                else:
                    sup = -1 if d_y[i] < 0 else 1
                    j_g[ii: ii + delta + 1] = arange(i_y0[i], i_y1[i] + sup, sup)
                    i_g[ii: ii + delta + 1] = (int_(j_g[ii: ii + delta + 1]) - i_y0[i]) * a + i_x0[i]
            ii += delta + 1
        i_g %= nb_x
        return i_g, j_g, d_max


class EddiesObservations(object):
    """
    Class to hold eddy properties *amplitude* and counts of
    *local maxima/minima* within a closed region of a sea level anomaly field.

    """
    
    DELTA_JJ_JE = 2448623 - (datetime(1992, 1, 1) - datetime(1950, 1, 1)).days 
    
    ELEMENTS = ['lon', 'lat', 'radius_s', 'radius_e', 'amplitude', 'speed_radius', 'time', 'eke',
                'shape_error_e', 'shape_error_s', 'nb_contour_selected',
                'height_max_speed_contour', 'height_external_contour', 'height_inner_contour']

    def __init__(self, size=0, track_extra_variables=None,
                 track_array_variables=0, array_variables=None):
        self.track_extra_variables = \
            track_extra_variables if track_extra_variables is not None else []
        self.track_array_variables = track_array_variables
        self.array_variables = \
            array_variables if array_variables is not None else []
        for elt in self.elements:
            if elt not in VAR_DESCR:
                raise Exception('Unknown element : %s' % elt)
        self.observations = zeros(size, dtype=self.dtype)
        self.active = True
        self.sign_type = None

    @property
    def sign_legend(self):
        return 'Cyclonic' if self.sign_type == -1 else 'Anticyclonic'

    @property
    def shape(self):
        return self.observations.shape

    def __repr__(self):
        return str(self.observations)

    def __getitem__(self, attr):
        if attr in self.elements:
            return self.observations[attr]
        raise KeyError('%s unknown' % attr)

    @property
    def dtype(self):
        """Return dtype to build numpy array
        """
        dtype = list()
        for elt in self.elements:
            data_type = VAR_DESCR[elt][
                'compute_type' if 'compute_type' in VAR_DESCR[elt] else
                'nc_type']
            if elt in self.array_variables:
                dtype.append((elt, data_type, (self.track_array_variables,)))
            else:
                dtype.append((elt, data_type))
        return dtype

    @property
    def elements(self):
        """Return all variable name
        """
        elements = [i for i in self.ELEMENTS]
        if self.track_array_variables > 0:
            elements += self.array_variables

        if len(self.track_extra_variables):
            elements += self.track_extra_variables
        return list(set(elements))

    def coherence(self, other):
        """Check coherence between two dataset
        """
        test = self.track_extra_variables == other.track_extra_variables
        test *= self.track_array_variables == other.track_array_variables
        test *= self.array_variables == other.array_variables
        return test

    @classmethod
    def concatenate(cls, observations):
        nb_obs = 0
        ref_obs = observations[0]
        for obs in observations:
            if not ref_obs.coherence(obs):
                raise Exception('Merge of different type of observations')
            nb_obs += len(obs)
        eddies = cls.new_like(ref_obs, nb_obs)

        i = 0
        for obs in observations:
            nb_obs = len(obs)
            eddies.obs[i:i + nb_obs] = obs.obs
            i += nb_obs
        eddies.sign_type = ref_obs.sign_type
        return eddies

    def merge(self, other):
        """Merge two dataset
        """
        nb_obs_self = len(self)
        nb_obs = nb_obs_self + len(other)
        eddies = self.new_like(self, nb_obs)
        other_keys = other.obs.dtype.fields.keys()
        for key in eddies.obs.dtype.fields.keys():
            eddies.obs[key][:nb_obs_self] = self.obs[key][:]
            if key in other_keys:
                eddies.obs[key][nb_obs_self:] = other.obs[key][:]
        eddies.sign_type = self.sign_type
        return eddies

    def reset(self):
        self.observations = zeros(0, dtype=self.dtype)

    @property
    def obs(self):
        """return an array observations
        """
        return self.observations

    def __len__(self):
        return len(self.observations)

    def __iter__(self):
        for obs in self.obs:
            yield obs

    def insert_observations(self, other, index):
        """Insert other obs in self at the index
        """
        if not self.coherence(other):
            raise Exception('Observations with no coherence')
        insert_size = len(other.obs)
        self_size = len(self.obs)
        new_size = self_size + insert_size
        if self_size == 0:
            self.observations = other.obs
            return self
        elif insert_size == 0:
            return self
        if index < 0:
            index = self_size + index + 1
        eddies = self.new_like(self, new_size)
        eddies.obs[:index] = self.obs[:index]
        eddies.obs[index: index + insert_size] = other.obs
        eddies.obs[index + insert_size:] = self.obs[index:]
        self.observations = eddies.obs
        return self

    def append(self, other):
        """Merge
        """
        return self + other

    def __add__(self, other):
        return self.insert_observations(other, -1)

    def distance(self, other):
        """ Use haversine distance for distance matrix between every self and
        other eddies"""
        dist_result = empty((len(self), len(other)), dtype='f8') + nan
        distance_matrix(
            self.obs['lon'], self.obs['lat'],
            other.obs['lon'], other.obs['lat'],
            dist_result)
        return dist_result

    @staticmethod
    def new_like(eddies, new_size):
        return eddies.__class__(new_size,
            track_extra_variables=eddies.track_extra_variables,
            track_array_variables=eddies.track_array_variables,
            array_variables=eddies.array_variables
            )

    def index(self, index):
        """Return obs from self at the index
        """
        size = 1
        if hasattr(index, '__iter__'):
            size = len(index)
        eddies = self.new_like(self, size)
        eddies.obs[:] = self.obs[index]
        return eddies

    @classmethod
    def load_from_netcdf(cls, filename):
        array_dim = 'NbSample'
        if not isinstance(filename, str):
            filename = filename.astype(str)
        with Dataset(filename) as h_nc:
            nb_obs = len(h_nc.dimensions['Nobs'])
            kwargs = dict()
            if array_dim in h_nc.dimensions:
                kwargs['track_array_variables'] = len(
                    h_nc.dimensions[array_dim])
                kwargs['array_variables'] = []
                for variable in h_nc.variables:
                    if array_dim in h_nc.variables[variable].dimensions:
                        kwargs['array_variables'].append(str(variable))
            kwargs['track_extra_variables'] = []
            for variable in h_nc.variables:
                if variable == 'cyc':
                    continue
                var_inv = VAR_DESCR_inv[variable]
                if var_inv not in cls.ELEMENTS and var_inv not in kwargs.get('array_variables', list()):
                    # Patch
                    if var_inv == 'time_jj':
                        continue
                    #
                    kwargs['track_extra_variables'].append(var_inv)

            eddies = cls(size=nb_obs, ** kwargs)
            for variable in h_nc.variables:
                if variable == 'cyc':
                    continue
                # Patch
                var_inv = VAR_DESCR_inv[variable]
                if var_inv == 'time_jj':
                    var_inv = 'time'
                    eddies.obs[var_inv] = h_nc.variables[variable][:] + cls.DELTA_JJ_JE 
                #
                else:
                    eddies.obs[var_inv] = h_nc.variables[variable][:]
            eddies.sign_type = h_nc.variables['cyc'][0]
            if eddies.sign_type == 0:
                logging.debug('File come from another algorithm of identification')
                eddies.sign_type = -1

        return eddies

    @classmethod
    def from_netcdf(cls, handler):
        nb_obs = len(handler.dimensions['Nobs'])
        kwargs = dict()
        if hasattr(handler, 'track_array_variables'):
            kwargs['track_array_variables'] = handler.track_array_variables
            kwargs['array_variables'] = handler.array_variables.split(',')
        if len(handler.track_extra_variables) > 1:
            kwargs['track_extra_variables'] = handler.track_extra_variables.split(',')
        for variable in handler.variables:
            var_inv = VAR_DESCR_inv[variable]
        eddies = cls(size=nb_obs, **kwargs)
        for variable in handler.variables:
            # Patch
            if variable == 'time':
                eddies.obs[variable] = handler.variables[variable][:] + cls.DELTA_JJ_JE
            else:
                #
                eddies.obs[VAR_DESCR_inv[variable]] = handler.variables[variable][:]
        return eddies

    @staticmethod
    def propagate(previous_obs, current_obs, obs_to_extend, dead_track, nb_next, model):
        """
        Filled virtual obs (C)
        Args:
            previous_obs: previous obs from current (A)
            current_obs: previous obs from virtual (B)
            obs_to_extend:
            dead_track:
            nb_next:
            model: 

        Returns:
            New position C = B + AB
        """
        next_obs = VirtualEddiesObservations(
            size=nb_next,
            track_extra_variables=model.track_extra_variables,
            track_array_variables=model.track_array_variables,
            array_variables=model.array_variables)
        nb_dead = len(previous_obs)
        nb_virtual_extend = nb_next - nb_dead

        for key in model.elements:
            if key in ['lon', 'lat', 'time'] or 'contour_' in key:
                continue
            next_obs[key][:nb_dead] = current_obs[key]
        next_obs['dlon'][:nb_dead] = current_obs['lon'] - previous_obs['lon']
        next_obs['dlat'][:nb_dead] = current_obs['lat'] - previous_obs['lat']
        next_obs['lon'][:nb_dead] = current_obs['lon'] + next_obs['dlon'][:nb_dead]
        next_obs['lat'][:nb_dead] = current_obs['lat'] + next_obs['dlat'][:nb_dead]
        # Id which are extended
        next_obs['track'][:nb_dead] = dead_track
        # Add previous virtual
        if nb_virtual_extend > 0:
            for key in next_obs.elements:
                if key in ['lon', 'lat', 'time', 'track', 'segment_size'] or 'contour_' in key:
                    continue
                next_obs[key][nb_dead:] = obs_to_extend[key]
            next_obs['lon'][nb_dead:] = obs_to_extend['lon'] + obs_to_extend['dlon']
            next_obs['lat'][nb_dead:] = obs_to_extend['lat'] + obs_to_extend['dlat']
            next_obs['track'][nb_dead:] = obs_to_extend['track']
            next_obs['segment_size'][nb_dead:] = obs_to_extend['segment_size']
        # Count
        next_obs['segment_size'][:] += 1
        return next_obs

    @staticmethod
    def cost_function_common_area(records_in, records_out, distance):
        nb_records = records_in.shape[0]
        costs = ma.empty(nb_records,dtype='f4')
        for i_record in range(nb_records):
            poly_in = Polygon(
                concatenate((
                    (records_in[i_record]['contour_lon_e'],),
                    (records_in[i_record]['contour_lat_e'],))
                    ).T
                )
            poly_out = Polygon(
                concatenate((
                    (records_out[i_record]['contour_lon_e'],),
                    (records_out[i_record]['contour_lat_e'],))
                    ).T
                )
            try:
                costs[i_record] = 1 - poly_in.intersection(poly_out).area / poly_in.area
            except TopologicalError:
                costs[i_record] = 1
        costs.mask = costs == 1
        return costs

    def mask_function(self, other):
        return self.circle_mask(other, radius=125)

    @staticmethod
    def cost_function(records_in, records_out, distance):
        cost = ((records_in['amplitude'] - records_out['amplitude']
                 ) / records_in['amplitude']
                ) ** 2
        cost += ((records_in['radius_s'] - records_out['radius_s']
                  ) / records_in['radius_s']
                 ) ** 2
        cost += (distance / 125) ** 2
        cost **= 0.5
        # Mask value superior at 60 % of variation
        # return ma.array(cost, mask=m)
        return cost

    def circle_mask(self, other, radius=100):
        """Return a mask of available link"""
        return (self.distance(other).T < radius).T

    def shifted_ellipsoid_degrees_mask(self, other, minor=1.5, major=1.5):
        # c = (major ** 2 - minor ** 2) ** .5 + major
        c = major
        major = minor  + .5 * (major - minor)
        
      # r=.5*(c-c0)
      # a=c0+r
        # Focal
        f_right = self.obs['lon']
        f_left = f_right - (c - minor) 
        # Ellips center
        x_c = (f_left + f_right) * .5

        o_lat, s_lat = meshgrid(other.obs['lat'], self.obs['lat'])
        o_lon, s_lon = meshgrid(other.obs['lon'], x_c)
        dy = o_lat - s_lat
        dx = (o_lon - s_lon + 180) % 360 - 180
        dist_normalize =  (dx.T ** 2) / (major ** 2) + (dy.T ** 2) / minor ** 2
        return  dist_normalize.T < 1
        
    def fixed_ellipsoid_mask(self, other, minor=50, major=100, only_east=False, shifted_ellips=False):
        dist = self.distance(other).T
        accepted = dist < minor
        rejected = dist > major 
        rejected += isnan(dist)

        # All obs we are not in rejected and accepted, there are between
        # two circle
        needs_investigation = - (rejected + accepted)
        index_other, index_self  = where(needs_investigation)
        
        nb_case = index_self.shape[0]
        if nb_case != 0:
            if isinstance(major, ndarray):
                major = major[index_self]
            if isinstance(minor, ndarray):
                minor = minor[index_self]
            # focal distance
            f_degree = ((major ** 2 - minor ** 2) ** .5) / (111.2 * cos(radians(self.obs['lat'][index_self])))
            
            lon_self = self.obs['lon'][index_self]
            if shifted_ellips:
                x_center_ellips = lon_self - (major - minor) / 2
            else:
                x_center_ellips = lon_self

            lon_left_f = x_center_ellips - f_degree
            lon_right_f = x_center_ellips + f_degree
            
            dist_left_f = empty(nb_case, dtype='f8') + nan
            distance_vector(
                lon_left_f, self.obs['lat'][index_self],
                other.obs['lon'][index_other], other.obs['lat'][index_other],
                dist_left_f)
            dist_right_f = empty(nb_case, dtype='f8') + nan
            distance_vector(
                lon_right_f, self.obs['lat'][index_self],
                other.obs['lon'][index_other], other.obs['lat'][index_other],
                dist_right_f)
            dist_2a = (dist_left_f + dist_right_f) / 1000

            accepted[index_other, index_self] = dist_2a < (2 * major)
            if only_east:
                d_lon = (other.obs['lon'][index_other] - lon_self + 180) % 360 - 180
                mask = d_lon < 0
                accepted[index_other[mask], index_self[mask]] = False
        return accepted.T

    @staticmethod
    def basic_formula_ellips_major_axis(lats, cmin=1.5, cmax=10., c0=1.5, lat1=13.5, lat2=5., degrees=False):
        """Give major axis in km with a given latitude
        """
        # Straight line between lat1 and lat2:
        # y = a * x + b
        a = (cmin - cmax) / (lat1 - lat2)
        b = a * -lat1 + cmin

        abs_lats = abs(lats)
        major_axis = ones(lats.shape, dtype='f8') * cmin
        major_axis[abs_lats < lat2] = cmax
        m = abs_lats > lat1
        m += abs_lats < lat2

        major_axis[~m] = a * abs_lats[~m] + b
        if not degrees:
            major_axis *= 111.2
        return major_axis

    @staticmethod
    def solve_conflict(cost):
        pass

    @staticmethod
    def solve_simultaneous(cost):
        mask = ~cost.mask
        # Count number of link by self obs and other obs
        self_links = mask.sum(axis=1)
        other_links = mask.sum(axis=0)
        max_links = max(self_links.max(), other_links.max())
        if max_links > 5:
            logging.warning('One observation have %d links', max_links)

        # If some obs have multiple link, we keep only one link by eddy
        eddies_separation = 1 < self_links
        eddies_merge = 1 < other_links
        test = eddies_separation.any() or eddies_merge.any()
        if test:
            # We extract matrix which contains concflict
            obs_linking_to_self = mask[eddies_separation
                                                   ].any(axis=0)
            obs_linking_to_other = mask[:, eddies_merge
                                                    ].any(axis=1)
            i_self_keep = where(obs_linking_to_other + eddies_separation)[0]
            i_other_keep = where(obs_linking_to_self + eddies_merge)[0]

            # Cost to resolve conflict
            cost_reduce = cost[i_self_keep][:, i_other_keep]
            shape = cost_reduce.shape
            nb_conflict = (~cost_reduce.mask).sum()
            logging.debug('Shape conflict matrix : %s, %d conflicts', shape, nb_conflict)

            if nb_conflict >= (shape[0] + shape[1]):
                logging.warning('High number of conflict : %d (nb_conflict)',
                                shape[0] + shape[1])

            links_resolve = 0
            # Arbitrary value
            max_iteration = max(cost_reduce.shape)
            security_increment = 0
            while False in cost_reduce.mask:
                if security_increment > max_iteration:
                    # Maybe check if the size decrease if not rise an exception
                    # x_i, y_i = where(-cost_reduce.mask)
                    raise Exception('To many iteration: %d' % security_increment)
                security_increment += 1
                i_min_value = cost_reduce.argmin()
                i, j = floor(i_min_value / shape[1]).astype(int), i_min_value % shape[1]
                # Set to False all link
                mask[i_self_keep[i]] = False
                mask[:, i_other_keep[j]] = False
                cost_reduce.mask[i] = True
                cost_reduce.mask[:, j] = True
                # we active only this link
                mask[i_self_keep[i], i_other_keep[j]] = True
                links_resolve += 1
            logging.debug('%d links resolve', links_resolve)
        return mask

    @staticmethod
    def solve_first(cost, multiple_link=False):
        mask = ~cost.mask
        # Count number of link by self obs and other obs
        self_links = mask.sum(axis=1)
        other_links = mask.sum(axis=0)
        max_links = max(self_links.max(), other_links.max())
        if max_links > 5:
            logging.warning('One observation have %d links', max_links)

        # If some obs have multiple link, we keep only one link by eddy
        eddies_separation = 1 < self_links
        eddies_merge = 1 < other_links
        test = eddies_separation.any() or eddies_merge.any()
        if test:
            # We extract matrix which contains concflict
            obs_linking_to_self = mask[eddies_separation
                                                   ].any(axis=0)
            obs_linking_to_other = mask[:, eddies_merge
                                                    ].any(axis=1)
            i_self_keep = where(obs_linking_to_other + eddies_separation)[0]
            i_other_keep = where(obs_linking_to_self + eddies_merge)[0]

            # Cost to resolve conflict
            cost_reduce = cost[i_self_keep][:, i_other_keep]
            shape = cost_reduce.shape
            nb_conflict = (~cost_reduce.mask).sum()
            logging.debug('Shape conflict matrix : %s, %d conflicts', shape, nb_conflict)

            if nb_conflict >= (shape[0] + shape[1]):
                logging.warning('High number of conflict : %d (nb_conflict)',
                                shape[0] + shape[1])

            links_resolve = 0
            for i in range(shape[0]):
                j = cost_reduce[i].argmin()
                if hasattr(cost_reduce[i,j], 'mask'):
                    continue
                links_resolve += 1
                # Set to False all link
                mask[i_self_keep[i]] = False
                cost_reduce.mask[i] = True
                if not multiple_link:
                    mask[:, i_other_keep[j]] = False
                    cost_reduce.mask[:, j] = True
                # we active only this link
                mask[i_self_keep[i], i_other_keep[j]] = True

            logging.debug('%d links resolve', links_resolve)
        return mask

    def solve_function(self, cost_matrix):
        return where(self.solve_simultaneous(cost_matrix))

    def post_process_link(self, other, i_self, i_other):
        if unique(i_other).shape[0] != i_other.shape[0]:
            raise Exception()
        return i_self, i_other

    def tracking(self, other):
        """Track obs between self and other
        """
        dist = self.distance(other)
        mask_accept_dist = self.mask_function(other)
        indexs_closest = where(mask_accept_dist)
        
        cost_values = self.cost_function(
            self.obs[indexs_closest[0]],
            other.obs[indexs_closest[1]],
            dist[mask_accept_dist])

        cost_mat = ma.empty(mask_accept_dist.shape, dtype='f4')
        cost_mat.mask = ~mask_accept_dist
        cost_mat[mask_accept_dist] = cost_values

        i_self, i_other = self.solve_function(cost_mat)

        i_self, i_other = self.post_process_link(other, i_self, i_other)

        logging.debug('%d matched with previous', i_self.shape[0])

        return i_self, i_other

    def to_netcdf(self, handler):
        eddy_size = len(self)
        logging.debug('Create Dimensions "Nobs" : %d', eddy_size)
        handler.createDimension('Nobs', eddy_size)
        handler.track_extra_variables = ','.join(self.track_extra_variables)
        if self.track_array_variables != 0:
            handler.createDimension('NbSample', self.track_array_variables)
            handler.track_array_variables = self.track_array_variables
            handler.array_variables = ','.join(self.array_variables)
        # Iter on variables to create:
        fields = [field[0] for field in self.observations.dtype.descr]
        fields.sort()
        for ori_name in fields:
            # Patch for a transition
            name = 'time_jj' if ori_name == 'time' else ori_name
            #
            logging.debug('Create Variable %s', VAR_DESCR[name]['nc_name'])
            self.create_variable(
                handler,
                dict(varname=VAR_DESCR[name]['nc_name'],
                     datatype=VAR_DESCR[name]['output_type'],
                     dimensions=VAR_DESCR[name]['nc_dims']),
                VAR_DESCR[name]['nc_attr'],
                self.observations[ori_name],
                scale_factor=VAR_DESCR[name].get('scale_factor', None),
                add_offset=VAR_DESCR[name].get('add_offset', None)
            )
        # Add cyclonic information
        if self.sign_type is not None:
            self.create_variable(
                handler,
                dict(varname=VAR_DESCR['type_cyc']['nc_name'],
                     datatype=VAR_DESCR['type_cyc']['nc_type'],
                     dimensions=VAR_DESCR['type_cyc']['nc_dims']),
                VAR_DESCR['type_cyc']['nc_attr'],
                self.sign_type)

    @staticmethod
    def create_variable(handler_nc, kwargs_variable, attr_variable,
                        data, scale_factor=None, add_offset=None):
        var = handler_nc.createVariable(
            zlib=True,
            complevel=1,
            **kwargs_variable)
        attrs = list(attr_variable.keys())
        attrs.sort()
        for attr in attrs:
            attr_value = attr_variable[attr]
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

    def write_netcdf(self, path='./', filename='%(path)s/%(sign_type)s.nc'):
        """Write a netcdf with eddy obs
        """
        eddy_size = len(self.observations)
        filename = filename % dict(path=path, sign_type=self.sign_legend, prod_time=datetime.now().strftime('%Y%m%d'))
        logging.info('Store in %s', filename)
        with Dataset(filename, 'w', format='NETCDF4') as h_nc:
            logging.info('Create file %s', filename)
            # Create dimensions
            logging.debug('Create Dimensions "Nobs" : %d', eddy_size)
            h_nc.createDimension('Nobs', eddy_size)
            if self.track_array_variables != 0:
                h_nc.createDimension('NbSample', self.track_array_variables)
            # Iter on variables to create:
            for field in self.observations.dtype.descr:
                name = field[0]
                logging.debug('Create Variable %s', VAR_DESCR[name]['nc_name'])
                self.create_variable(
                    h_nc,
                    dict(varname=VAR_DESCR[name]['nc_name'],
                         datatype=VAR_DESCR[name]['output_type'],
                         dimensions=VAR_DESCR[name]['nc_dims']),
                    VAR_DESCR[name]['nc_attr'],
                    self.observations[name],
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
                self.sign_type)
            # Global attr
            self.set_global_attr_netcdf(h_nc)

    def set_global_attr_netcdf(self, h_nc):
        pass


class VirtualEddiesObservations(EddiesObservations):
    """Class to work with virtual obs
    """

    @property
    def elements(self):
        elements = super(VirtualEddiesObservations, self).elements
        elements.extend(['track', 'segment_size', 'dlon', 'dlat'])
        return list(set(elements))


class TrackEddiesObservations(EddiesObservations):
    """Class to practice Tracking on observations
    """

    def filled_by_interpolation(self, mask):
        """Filled selected values by interpolation
        """
        nb_filled = mask.sum()
        logging.info('%d obs will be filled (unobserved)', nb_filled)

        nb_obs = len(self)
        index = arange(nb_obs)

        for field in self.obs.dtype.descr:
            var = field[0]
            if var in ['n', 'virtual', 'track'] or var in self.array_variables:
                continue
            # to normalize longitude before interpolation
            if var== 'lon':
                lon = self.obs[var]
                first = where(self.obs['n'] == 0)[0]
                nb_obs = empty(first.shape, dtype='u4')
                nb_obs[:-1] = first[1:] - first[:-1]
                nb_obs[-1] = lon.shape[0] - first[-1]
                lon0 = (lon[first] - 180).repeat(nb_obs)
                self.obs[var] = (lon - lon0) % 360 + lon0
            self.obs[var][mask] = interp(index[mask], index[~mask], self.obs[var][~mask])

    def extract_longer_eddies(self, nb_min, nb_obs, compress_id=True):
        """Select eddies which are longer than nb_min
        """
        mask = nb_obs >= nb_min
        nb_obs_select = mask.sum()
        logging.info('Selection of %d observations', nb_obs_select)
        eddies = TrackEddiesObservations(
            size=nb_obs_select,
            track_extra_variables=self.track_extra_variables,
            track_array_variables=self.track_array_variables,
            array_variables=self.array_variables
            )
        eddies.sign_type = self.sign_type
        for field in self.obs.dtype.descr:
            logging.debug('Copy of field %s ...', field)
            var = field[0]
            eddies.obs[var] = self.obs[var][mask]
        if compress_id:
            list_id = unique(eddies.obs['track'])
            list_id.sort()
            id_translate = arange(list_id.max() + 1)
            id_translate[list_id] = arange(len(list_id)) + 1
            eddies.obs['track'] = id_translate[eddies.obs['track']]
        return eddies

    @property
    def elements(self):
        elements = super(TrackEddiesObservations, self).elements
        elements.extend(['track', 'n', 'virtual'])
        return elements

    def set_global_attr_netcdf(self, h_nc):
        """Set global attr
        """
        if self.sign_type == -1:
            h_nc.title = 'Cyclonic'
        else:
            h_nc.title = 'Anticyclonic' + ' eddy tracks'
