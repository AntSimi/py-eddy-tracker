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

tracking.py

Version 3.0.0

===========================================================================

"""
from numpy import empty, arange, where, unique, \
    interp
from .. import VAR_DESCR_inv
import logging
from datetime import datetime, timedelta
from .observation import EddiesObservations


class TrackEddiesObservations(EddiesObservations):
    """Class to practice Tracking on observations
    """
    __slots__ = ()

    ELEMENTS = ['lon', 'lat', 'radius_s', 'radius_e', 'amplitude', 'speed_radius', 'time',
                'shape_error_e', 'shape_error_s', 'nb_contour_selected',
                'height_max_speed_contour', 'height_external_contour', 'height_inner_contour', 'cost_association']

    def filled_by_interpolation(self, mask):
        """Filled selected values by interpolation
        """
        nb_filled = mask.sum()
        logging.info('%d obs will be filled (unobserved)', nb_filled)

        nb_obs = len(self)
        index = arange(nb_obs)

        for field in self.obs.dtype.descr:
            var = field[0]
            if var in ['n', 'virtual', 'track', 'cost_association'] or var in self.array_variables:
                continue
            # to normalize longitude before interpolation
            if var == 'lon':
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
        h_nc.title = 'Cyclonic' if self.sign_type == -1 else 'Anticyclonic'
        h_nc.Metadata_Conventions = 'Unidata Dataset Discovery v1.0'
        h_nc.comment = 'Surface product; mesoscale eddies'
        h_nc.framework_used = 'https://bitbucket.org/emason/py-eddy-tracker'
        h_nc.standard_name_vocabulary = 'NetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table'
        h_nc.date_created = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        t = h_nc.variables[VAR_DESCR_inv['j1']]
        delta = t.max - t.min + 1
        h_nc.time_coverage_duration = 'P%dD' % delta
        d_start = datetime(1950, 1, 1) + timedelta(int(t.min))
        d_end = datetime(1950, 1, 1) + timedelta(int(t.max))
        h_nc.time_coverage_start = d_start.strftime('%Y-%m-%dT00:00:00Z')
        h_nc.time_coverage_end = d_end.strftime('%Y-%m-%dT00:00:00Z')

    def extract_with_period(self, period, **kwargs):
        """
        Extract with a period
        Args:
            period: two date to define period, must be specify from 1/1/1950
            **kwargs: directly give to __extract_with_mask

        Returns:
            same object with selected data
        """
        mask = (self.time > period[0]) * (self.time < period[1])
        return self.__extract_with_mask(mask, **kwargs)

    def __extract_with_mask(self, mask, full_path=False, remove_incomplete=False):
        """
        Extract a subset of observations
        Args:
            mask: mask to select observations
            full_path: extract full path if only one part is selected
            remove_incomplete: delete path which are not totatly selected

        Returns:
            same object with selected observations
        """
        if full_path and remove_incomplete:
            logging.warning('Incompatible option, remove_incomplete option will be remove')
