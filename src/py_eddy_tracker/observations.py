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


py_eddy_tracker_amplitude.py

Version 2.0.3

===========================================================================

"""
from numpy import zeros, empty, nan, arange, interp, where, unique
from netCDF4 import Dataset
from py_eddy_tracker.tools import distance_matrix
from . import VAR_DESCR, VAR_DESCR_inv
import logging


class EddiesObservations(object):
    """
    Class to hold eddy properties *amplitude* and counts of
    *local maxima/minima* within a closed region of a sea level anomaly field.

    Variables:
      centlon:
        Longitude centroid coordinate

      centlat:
        Latitude centroid coordinate

      eddy_radius_s:
        Speed based radius

      eddy_radius_e:
        Effective radius

      amplitude:
        Eddy amplitude

      uavg:
        Average eddy swirl speed

      teke:
        Average eddy kinetic energy within eddy

      rtime:
        Time
    """

    def __init__(self, size=0, track_extra_variables=False):
        self.track_extra_variables = track_extra_variables
        for elt in self.elements:
            if elt not in VAR_DESCR:
                raise Exception('Unknown element : %s' % elt)
        self.observations = zeros(size, dtype=self.dtype)
        self.active = True
        self.sign_type = None

    def __repr__(self):
        return str(self.observations)

    def __getitem__(self, attr):
        if attr in self.elements:
            return self.observations[attr]
        raise KeyError('%s unknown' % attr)
    
    @property
    def dtype(self):
        dtype = list()
        for elt in self.elements:
            dtype.append((elt, VAR_DESCR[elt][
                'compute_type' if 'compute_type' in VAR_DESCR[elt] else
                'nc_type']))
        return dtype

    @property
    def elements(self):
        elements = [
            'lon',  # 'centlon'
            'lat',  # 'centlat'
            'radius_s',  # 'eddy_radius_s'
            'radius_e',  # 'eddy_radius_e'
            'amplitude',  # 'amplitude'
            'speed_radius',  # 'uavg'
            'eke',  # 'teke'
            'time']  # 'rtime'

        if self.track_extra_variables:
            elements += ['contour_e',
                         'contour_s',
                         'uavg_profile',
                         'shape_error',
                         ]
        return elements

    def coherence(self, other):
        return self.track_extra_variables == other.track_extra_variables
    
    def merge(self, other):
        nb_obs_self = len(self)
        nb_obs = nb_obs_self + len(other)
        eddies = self.__class__(size=nb_obs)
        eddies.obs[:nb_obs_self] = self.obs[:]
        eddies.obs[nb_obs_self:] = other.obs[:]
        eddies.sign_type = self.sign_type
        return eddies

    @property
    def obs(self):
        return self.observations

    def __len__(self):
        return len(self.observations)

    def __iter__(self):
        for obs in self.obs:
            yield obs

    def insert_observations(self, other, index):
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
        eddies = self.__class__(new_size, self.track_extra_variables)
        eddies.obs[:index] = self.obs[:index]
        eddies.obs[index: index + insert_size] = other.obs
        eddies.obs[index + insert_size:] = self.obs[index:]
        self.observations = eddies.obs
        return self

    def append(self, other):
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

    def index(self, index):
        size = 1
        if hasattr(index, '__iter__'):
            size = len(index)
        eddies = self.__class__(size, self.track_extra_variables)
        eddies.obs[:] = self.obs[index]
        return eddies
    
    @staticmethod
    def load_from_netcdf(filename):
        with Dataset(filename) as h_nc:
            nb_obs = len(h_nc.dimensions['Nobs'])
            eddies = EddiesObservations(size=nb_obs)
            for variable in h_nc.variables:
                if variable == 'cyc':
                    continue
                eddies.obs[VAR_DESCR_inv[variable]] = h_nc.variables[variable][:]
            eddies.sign_type = h_nc.variables['cyc'][0]
        return eddies

    def tracking(self, other):
        """Track obs between from self to other
        """
        dist = self.distance(other)
        i_self, i_other = where(dist < 20.)

        logging.debug('%d matched with previous', i_self.shape[0])
        return i_self, i_other


class VirtualEddiesObservations(EddiesObservations):
    
    @property
    def elements(self):
        elements = super(VirtualEddiesObservations, self).elements
        elements.extend(['track', 'segment_size', 'dlon', 'dlat'])
        return elements


class TrackEddiesObservations(EddiesObservations):
    
    def filled_by_interpolation(self, mask):
        """Filled selected values by interpolation
        """
        nb_filled = mask.sum()
        logging.info('%d obs will be filled (unobserved)', nb_filled)

        nb_obs = len(self)
        index = arange(nb_obs)

        for var, _ in self.obs.dtype.descr:
            if var in ['n', 'virtual', 'track']:
                continue
            self.obs[var][mask] = interp(index[mask], index[-mask],
                                         self.obs[var][-mask])

    def extract_longer_eddies(self, nb_min, nb_obs, compress_id=True):
        m = nb_obs >= nb_min
        nb_obs_select = m.sum()
        logging.info('Selection of %d observations', nb_obs_select)
        eddies = TrackEddiesObservations(size=nb_obs_select)
        eddies.sign_type = self.sign_type
        for var, _ in eddies.obs.dtype.descr:
            eddies.obs[var] = self.obs[var][m]
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
    
    def create_variable(self, handler_nc, kwargs_variable,
                        attr_variable, data, scale_factor=None):
        var = handler_nc.createVariable(
            zlib=True,
            complevel=1,
            **kwargs_variable)
        for attr, attr_value in attr_variable.iteritems():
            var.setncattr(attr, attr_value)
            
        var[:] = data
        
        #~ var.set_auto_maskandscale(False)
        if scale_factor is not None:
            var.scale_factor = scale_factor
            
        try:
            var.setncattr('min', var[:].min())
            var.setncattr('max', var[:].max())
        except ValueError:
            logging.warn('Data is empty')

    def write_netcdf(self):
        """Write a netcdf with eddy obs
        """
        eddy_size = len(self.observations)
        sign_type = 'Cyclonic' if self.sign_type == -1 else 'Anticyclonic'
        filename = '%s.nc' % sign_type
        with Dataset(filename, 'w', format='NETCDF4') as h_nc:
            logging.info('Create file %s', filename)
            # Create dimensions
            logging.debug('Create Dimensions "Nobs" : %d', eddy_size)
            h_nc.createDimension('Nobs', eddy_size)
            # Iter on variables to create:
            for name, _ in self.observations.dtype.descr:
                logging.debug('Create Variable %s', VAR_DESCR[name]['nc_name'])
                self.create_variable(
                    h_nc,
                    dict(varname=VAR_DESCR[name]['nc_name'],
                         datatype=VAR_DESCR[name]['nc_type'],
                         dimensions=VAR_DESCR[name]['nc_dims']),
                    VAR_DESCR[name]['nc_attr'],
                    self.observations[name],
                    scale_factor=None if 'scale_factor' not in VAR_DESCR[name] else VAR_DESCR[name]['scale_factor'])

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
        h_nc.title = 'Cyclonic' if self.sign_type == -1 else 'Anticyclonic' + ' eddy tracks'
        #~ h_nc.grid_filename = self.grd.grid_filename
        #~ h_nc.grid_date = str(self.grd.grid_date)
        #~ h_nc.product = self.product

        #~ h_nc.contour_parameter = self.contour_parameter
        #~ h_nc.shape_error = self.shape_error
        #~ h_nc.pixel_threshold = self.pixel_threshold

        #~ if self.smoothing in locals():
            #~ h_nc.smoothing = self.smoothing
            #~ h_nc.SMOOTH_FAC = self.SMOOTH_FAC
        #~ else:
            #~ h_nc.smoothing = 'None'

        #~ h_nc.evolve_amp_min = self.evolve_amp_min
        #~ h_nc.evolve_amp_max = self.evolve_amp_max
        #~ h_nc.evolve_area_min = self.evolve_area_min
        #~ h_nc.evolve_area_max = self.evolve_area_max
#~ 
        #~ h_nc.llcrnrlon = self.grd.lonmin
        #~ h_nc.urcrnrlon = self.grd.lonmax
        #~ h_nc.llcrnrlat = self.grd.latmin
        #~ h_nc.urcrnrlat = self.grd.latmax