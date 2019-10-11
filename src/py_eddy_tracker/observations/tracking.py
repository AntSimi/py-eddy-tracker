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
from numpy import empty, arange, where, unique, interp, ones, bool_, zeros, array
from .. import VAR_DESCR_inv
import logging
from datetime import datetime, timedelta
from .observation import EddiesObservations
from numba import njit


class TrackEddiesObservations(EddiesObservations):
    """Class to practice Tracking on observations
    """

    __slots__ = ("__obs_by_track", "__first_index_of_track")

    ELEMENTS = [
        "lon",
        "lat",
        "radius_s",
        "radius_e",
        "amplitude",
        "speed_radius",
        "time",
        "shape_error_e",
        "shape_error_s",
        "nb_contour_selected",
        "height_max_speed_contour",
        "height_external_contour",
        "height_inner_contour",
        "cost_association",
    ]

    def __init__(self, *args, **kwargs):
        super(TrackEddiesObservations, self).__init__(*args, **kwargs)
        self.__first_index_of_track = None
        self.__obs_by_track = None

    def filled_by_interpolation(self, mask):
        """Filled selected values by interpolation
        """
        nb_filled = mask.sum()
        logging.info("%d obs will be filled (unobserved)", nb_filled)

        nb_obs = len(self)
        index = arange(nb_obs)

        for field in self.obs.dtype.descr:
            var = field[0]
            if (
                var in ["n", "virtual", "track", "cost_association"]
                or var in self.array_variables
            ):
                continue
            # to normalize longitude before interpolation
            if var == "lon":
                lon = self.obs[var]
                first = where(self.obs["n"] == 0)[0]
                nb_obs = empty(first.shape, dtype="u4")
                nb_obs[:-1] = first[1:] - first[:-1]
                nb_obs[-1] = lon.shape[0] - first[-1]
                lon0 = (lon[first] - 180).repeat(nb_obs)
                self.obs[var] = (lon - lon0) % 360 + lon0
            self.obs[var][mask] = interp(
                index[mask], index[~mask], self.obs[var][~mask]
            )

    def extract_longer_eddies(self, nb_min, nb_obs, compress_id=True):
        """Select eddies which are longer than nb_min
        """
        mask = nb_obs >= nb_min
        nb_obs_select = mask.sum()
        logging.info("Selection of %d observations", nb_obs_select)
        eddies = TrackEddiesObservations(
            size=nb_obs_select,
            track_extra_variables=self.track_extra_variables,
            track_array_variables=self.track_array_variables,
            array_variables=self.array_variables,
        )
        eddies.sign_type = self.sign_type
        for field in self.obs.dtype.descr:
            logging.debug("Copy of field %s ...", field)
            var = field[0]
            eddies.obs[var] = self.obs[var][mask]
        if compress_id:
            list_id = unique(eddies.obs["track"])
            list_id.sort()
            id_translate = arange(list_id.max() + 1)
            id_translate[list_id] = arange(len(list_id)) + 1
            eddies.obs["track"] = id_translate[eddies.obs["track"]]
        return eddies

    @property
    def elements(self):
        elements = super(TrackEddiesObservations, self).elements
        elements.extend(["track", "n", "virtual"])
        return list(set(elements))

    def set_global_attr_netcdf(self, h_nc):
        """Set global attr
        """
        h_nc.title = "Cyclonic" if self.sign_type == -1 else "Anticyclonic"
        h_nc.Metadata_Conventions = "Unidata Dataset Discovery v1.0"
        h_nc.comment = "Surface product; mesoscale eddies"
        h_nc.framework_used = "https://github.com/AntSimi/py-eddy-tracker"
        h_nc.standard_name_vocabulary = (
            "NetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table"
        )
        h_nc.date_created = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        t = h_nc.variables[VAR_DESCR_inv["j1"]]
        delta = t.max - t.min + 1
        h_nc.time_coverage_duration = "P%dD" % delta
        d_start = datetime(1950, 1, 1) + timedelta(int(t.min))
        d_end = datetime(1950, 1, 1) + timedelta(int(t.max))
        h_nc.time_coverage_start = d_start.strftime("%Y-%m-%dT00:00:00Z")
        h_nc.time_coverage_end = d_end.strftime("%Y-%m-%dT00:00:00Z")

    def extract_with_area(self, area, **kwargs):
        """
        Extract with a bounding box
        Args:
            area: 4 coordinates in a dictionary to specify bounding box (lower left corner and upper right corner)
            **kwargs:

        Returns:

        """
        mask = (self.latitude > area["llcrnrlat"]) * (self.latitude < area["urcrnrlat"])
        lon0 = area["llcrnrlon"]
        lon = (self.longitude - lon0) % 360 + lon0
        mask *= (lon > lon0) * (lon < area["urcrnrlon"])
        return self.__extract_with_mask(mask, **kwargs)

    def extract_with_period(self, period, **kwargs):
        """
        Extract with a period
        Args:
            period: two date to define period, must be specify from 1/1/1950
            **kwargs: directly give to __extract_with_mask

        Returns:
            same object with selected data
        """
        dataset_period = self.period
        p_min, p_max = period
        if p_min > 0:
            mask = self.time >= p_min
        elif p_min < 0:
            mask = self.time >= (dataset_period[0] - p_min)
        else:
            mask = ones(self.time.shape, dtype=bool_)
        if p_max > 0:
            mask *= self.time <= p_max
        elif p_max < 0:
            mask *= self.time <= (dataset_period[1] + p_max)
        return self.__extract_with_mask(mask, **kwargs)

    @property
    def period(self):
        """
        Give time coverage
        Returns: 2 date
        """
        return self.time.min(), self.time.max()

    def get_mask_from_id(self, tracks):
        mask = zeros(self.tracks.shape, dtype=bool_)
        compute_mask_from_id(tracks, self.index_from_track, self.nb_obs_by_track, mask)
        return mask

    def compute_index(self):
        if self.__first_index_of_track is None:
            s = self.tracks.max() + 1
            # Doesn't work => core dump with numba, maybe he wait i8 instead of u4
            # self.__first_index_of_track = -ones(s, self.tracks.dtype)
            # self.__obs_by_track = zeros(s, self.observation_number.dtype)
            self.__first_index_of_track = -ones(s, "i8")
            self.__obs_by_track = zeros(s, "i8")
            logging.debug("Start computing index ...")
            compute_index(self.tracks, self.__first_index_of_track, self.__obs_by_track)
            logging.debug("... OK")

    @property
    def index_from_track(self):
        self.compute_index()
        return self.__first_index_of_track

    @property
    def nb_obs_by_track(self):
        self.compute_index()
        return self.__obs_by_track

    def extract_ids(self, tracks):
        mask = self.get_mask_from_id(array(tracks))
        return self.__extract_with_mask(mask)

    def extract_with_length(self, bounds):
        b0, b1 = bounds
        if b0 >= 0 and b1 >=0:
            track_mask = (self.nb_obs_by_track >= b0) * (self.nb_obs_by_track <= b1)
        elif b0 < 0 and b1 >= 0:
            track_mask = self.nb_obs_by_track <= b1
        elif b0 >= 0 and b1 < 0:
            track_mask = self.nb_obs_by_track > b0
        else:
            logging.warning('No valid value for bounds')
            raise Exception('One bounds must be positiv')
        return self.__extract_with_mask(track_mask.repeat(self.nb_obs_by_track))

    def __extract_with_mask(self, mask, full_path=False, remove_incomplete=False, compress_id=False):
        """
        Extract a subset of observations
        Args:
            mask: mask to select observations
            full_path: extract full path if only one part is selected
            remove_incomplete: delete path which are not fully selected
            compress_id: resample track number to use a little range

        Returns:
            same object with selected observations
        """
        if full_path and remove_incomplete:
            logging.warning(
                "Incompatible option, remove_incomplete option will be remove"
            )
            remove_incomplete = False

        if full_path:
            tracks = unique(self.tracks[mask])
            mask = self.get_mask_from_id(tracks)
        elif remove_incomplete:
            tracks = unique(self.tracks[~mask])
            mask = ~self.get_mask_from_id(tracks)

        nb_obs = mask.sum()
        new = TrackEddiesObservations(
            size=nb_obs,
            track_extra_variables=self.track_extra_variables,
            track_array_variables=self.track_array_variables,
            array_variables=self.array_variables,
            raw_data=self.raw_data,
        )
        new.sign_type = self.sign_type
        if nb_obs == 0:
            logging.warning("Empty dataset will be created")
        else:
            for field in self.obs.dtype.descr:
                logging.debug("Copy of field %s ...", field)
                var = field[0]
                new.obs[var] = self.obs[var][mask]
            if compress_id:
                list_id = unique(new.obs["track"])
                list_id.sort()
                id_translate = arange(list_id.max() + 1)
                id_translate[list_id] = arange(len(list_id)) + 1
                new.obs["track"] = id_translate[new.obs["track"]]
        return new


@njit(cache=True)
def compute_index(tracks, index, number):
    previous_track = -1
    for i, track in enumerate(tracks):
        if track != previous_track:
            index[track] = i
        number[track] += 1
        previous_track = track


@njit(cache=True)
def compute_mask_from_id(tracks, first_index, number_of_obs, mask):
    for track in tracks:
        mask[first_index[track]:first_index[track] + number_of_obs[track]] = True
