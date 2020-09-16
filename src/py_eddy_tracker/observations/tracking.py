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
import logging
from numpy import (
    empty,
    arange,
    where,
    unique,
    interp,
    ones,
    bool_,
    zeros,
    array,
    median,
)
from datetime import datetime, timedelta
from numba import njit
from Polygon import Polygon
from .observation import EddiesObservations
from .. import VAR_DESCR_inv
from ..generic import split_line, wrap_longitude, build_index
from ..poly import polygon_overlap, create_vertice_from_2darray


logger = logging.getLogger("pet")


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
        "speed_average",
        "time",
        "shape_error_e",
        "shape_error_s",
        "nb_contour_selected",
        "num_point_e",
        "num_point_s",
        "height_max_speed_contour",
        "height_external_contour",
        "height_inner_contour",
        "cost_association",
    ]

    NOGROUP = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__first_index_of_track = None
        self.__obs_by_track = None

    def filled_by_interpolation(self, mask):
        """Filled selected values by interpolation
        """
        nb_filled = mask.sum()
        logger.info("%d obs will be filled (unobserved)", nb_filled)

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
        logger.info("Selection of %d observations", nb_obs_select)
        eddies = self.__class__.new_like(self, nb_obs_select)
        eddies.sign_type = self.sign_type
        for field in self.obs.dtype.descr:
            logger.debug("Copy of field %s ...", field)
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
        elements = super().elements
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

        :param dict area: 4 coordinates in a dictionary to specify bounding box (lower left corner and upper right corner)

        """
        mask = (self.latitude > area["llcrnrlat"]) * (self.latitude < area["urcrnrlat"])
        lon0 = area["llcrnrlon"]
        lon = (self.longitude - lon0) % 360 + lon0
        mask *= (lon > lon0) * (lon < area["urcrnrlon"])
        return self.__extract_with_mask(mask, **kwargs)

    def extract_with_period(self, period, **kwargs):
        """
        Extract with a period

        :param (datetime.datetime,datetime.datetime) period: two date to define period, must be specify from 1/1/1950

        :return: same object with selected data
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
            logger.debug("Start computing index ...")
            compute_index(self.tracks, self.__first_index_of_track, self.__obs_by_track)
            logger.debug("... OK")

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

    def extract_first_obs_in_box(self, res):
        data = empty(
            self.obs.shape, dtype=[("lon", "f4"), ("lat", "f4"), ("track", "i4")]
        )
        data["lon"] = self.longitude - self.longitude % res
        data["lat"] = self.latitude - self.latitude % res
        data["track"] = self.obs["track"]
        _, indexs = unique(data, return_index=True)
        mask = zeros(self.obs.shape, dtype="bool")
        mask[indexs] = True
        return self.__extract_with_mask(mask)

    def extract_in_direction(self, direction, value=0):
        nb_obs = self.nb_obs_by_track
        i_start = self.index_from_track
        i_stop = i_start + nb_obs - 1
        if direction in ("S", "N"):
            d_lat = self.latitude[i_stop] - self.latitude[i_start]
            mask = d_lat < 0 if "S" == direction else d_lat > 0
            mask &= abs(d_lat) > value
        else:
            lon_start, lon_end = self.longitude[i_start], self.longitude[i_stop]
            lon_end = (lon_end - (lon_start - 180)) % 360 + lon_start - 180
            d_lon = lon_end - lon_start
            mask = d_lon < 0 if "W" == direction else d_lon > 0
            mask &= abs(d_lon) > value
        mask = mask.repeat(nb_obs)
        return self.__extract_with_mask(mask)

    def extract_with_length(self, bounds):
        b0, b1 = bounds
        if b0 >= 0 and b1 >= 0:
            track_mask = (self.nb_obs_by_track >= b0) * (self.nb_obs_by_track <= b1)
        elif b0 < 0 and b1 >= 0:
            track_mask = self.nb_obs_by_track <= b1
        elif b0 >= 0 and b1 < 0:
            track_mask = self.nb_obs_by_track > b0
        else:
            logger.warning("No valid value for bounds")
            raise Exception("One bounds must be positiv")
        return self.__extract_with_mask(track_mask.repeat(self.nb_obs_by_track))

    def loess_filter(self, half_window, xfield, yfield, inplace=True):
        track = self.obs["track"]
        x = self.obs[xfield]
        y = self.obs[yfield]
        result = track_loess_filter(half_window, x, y, track)
        if inplace:
            self.obs[yfield] = result
            return self

    def median_filter(self, half_window, xfield, yfield, inplace=True):
        track = self.obs["track"]
        x = self.obs[xfield]
        y = self.obs[yfield]
        result = track_median_filter(half_window, x, y, track)
        if inplace:
            self.obs[yfield] = result
            return self

    def position_filter(self, median_half_window, loess_half_window):
        self.median_filter(median_half_window, "time", "lon").loess_filter(
            loess_half_window, "time", "lon"
        )
        self.median_filter(median_half_window, "time", "lat").loess_filter(
            loess_half_window, "time", "lat"
        )

    def __extract_with_mask(
        self,
        mask,
        full_path=False,
        remove_incomplete=False,
        compress_id=False,
        reject_virtual=False,
    ):
        """
        Extract a subset of observations
        Args:
            mask: mask to select observations
            full_path: extract full path if only one part is selected
            remove_incomplete: delete path which are not fully selected
            compress_id: resample track number to use a little range
            reject_virtual: if track are only virtual in selection we remove track

        Returns:
            same object with selected observations
        """
        if full_path and remove_incomplete:
            logger.warning(
                "Incompatible option, remove_incomplete option will be remove"
            )
            # remove_incomplete = False

        if full_path:
            if reject_virtual:
                mask *= ~self.obs["virtual"].astype("bool")
            tracks = unique(self.tracks[mask])
            mask = self.get_mask_from_id(tracks)
        elif remove_incomplete:
            tracks = unique(self.tracks[~mask])
            mask = ~self.get_mask_from_id(tracks)

        nb_obs = mask.sum()
        new = self.__class__.new_like(self, nb_obs)
        new.sign_type = self.sign_type
        if nb_obs == 0:
            logger.warning("Empty dataset will be created")
        else:
            for field in self.obs.dtype.descr:
                logger.debug("Copy of field %s ...", field)
                var = field[0]
                new.obs[var] = self.obs[var][mask]
            if compress_id:
                list_id = unique(new.obs["track"])
                list_id.sort()
                id_translate = arange(list_id.max() + 1)
                id_translate[list_id] = arange(len(list_id)) + 1
                new.obs["track"] = id_translate[new.obs["track"]]
        return new

    def plot(self, ax, ref=None, **kwargs):
        if "label" in kwargs:
            kwargs["label"] += " (%s eddies)" % (self.nb_obs_by_track != 0).sum()
        x, y = split_line(self.longitude, self.latitude, self.tracks)
        if ref is not None:
            x, y = wrap_longitude(x, y, ref, cut=True)
        return ax.plot(x, y, **kwargs)

    def split_network(self, intern=True, **kwargs):
        """Divide each group in track
        """
        track_s, track_e, track_ref = build_index(self.tracks)
        ids = empty(
            len(self),
            dtype=[
                ("group", self.tracks.dtype),
                ("time", self.time.dtype),
                ("track", "u2"),
                ("previous_cost", "f4"),
                ("next_cost", "f4"),
                ("previous_obs", "i4"),
                ("next_obs", "i4"),
            ],
        )
        ids["group"], ids["time"] = self.tracks, self.time
        # To store id track
        ids["track"], ids["previous_cost"], ids["next_cost"] = 0, 0, 0
        ids["previous_obs"], ids["next_obs"] = -1, -1

        xname, yname = self.intern(intern)
        for i_s, i_e in zip(track_s, track_e):
            if i_s == i_e or self.tracks[i_s] == self.NOGROUP:
                continue
            sl = slice(i_s, i_e)
            local_ids = ids[sl]
            self.set_tracks(self[xname][sl], self[yname][sl], local_ids, **kwargs)
            m = local_ids["previous_obs"] == -1
            local_ids["previous_obs"][m] += i_s
            m = local_ids["next_obs"] == -1
            local_ids["next_obs"][m] += i_s
        return ids
        # ids_sort = ids[new_i]
        # # To be able to follow indices sorting
        # reverse_sort = empty(new_i.shape[0], dtype="u4")
        # reverse_sort[new_i] = arange(new_i.shape[0])
        # # Redirect indices
        # m = ids_sort["next_obs"] != -1
        # ids_sort["next_obs"][m] = reverse_sort[
        #     ids_sort["next_obs"][m]
        # ]
        # m = ids_sort["previous_obs"] != -1
        # ids_sort["previous_obs"][m] = reverse_sort[
        #     ids_sort["previous_obs"][m]
        # ]
        # # print(ids_sort)
        # display_network(
        #     x[new_i],
        #     y[new_i],
        #     ids_sort["track"],
        #     ids_sort["time"],
        #     ids_sort["next_cost"],
        # )

    def set_tracks(self, x, y, ids, window):
        # Will split one group in tracks
        time_index = build_index(ids["time"])
        nb = x.shape[0]
        used = zeros(nb, dtype="bool")
        track_id = 1
        # build all polygon (need to check if wrap is needed)
        polygons = [Polygon(create_vertice_from_2darray(x, y, i)) for i in range(nb)]
        for i in range(nb):
            # If observation already in one track, we go to the next one
            if used[i]:
                continue
            self.follow_obs(i, track_id, used, ids, polygons, *time_index, window)
            track_id += 1

    @classmethod
    def follow_obs(cls, i_next, track_id, used, ids, *args):
        while i_next != -1:
            # Flag
            used[i_next] = True
            # Assign id
            ids["track"][i_next] = track_id
            # Search next
            i_next_ = cls.next_obs(i_next, ids, *args)
            if i_next_ == -1:
                break
            ids["next_obs"][i_next] = i_next_
            # Target was previously used
            if used[i_next_]:
                if ids["next_cost"][i_next] == ids["previous_cost"][i_next_]:
                    m = ids["track"][i_next_:] == ids["track"][i_next_]
                    ids["track"][i_next_:][m] = track_id
                    ids["previous_obs"][i_next_] = i_next
                i_next_ = -1
            else:
                ids["previous_obs"][i_next_] = i_next
            i_next = i_next_

    @staticmethod
    def next_obs(i_current, ids, polygons, time_s, time_e, time_ref, window):
        time_max = time_e.shape[0] - 1
        time_cur = ids["time"][i_current]
        t0, t1 = time_cur + 1 - time_ref, min(time_cur + window - time_ref, time_max)
        if t0 > time_max:
            return -1
        for t_step in range(t0, t1 + 1):
            i0, i1 = time_s[t_step], time_e[t_step]
            # No observation at the time step
            if i0 == i1:
                continue
            # Intersection / union, to be able to separte in case of multiple inside
            c = polygon_overlap(polygons[i_current], polygons[i0:i1])
            # We remove low overlap
            c[c < 0.1] = 0
            # We get index of maximal overlap
            i = c.argmax()
            c_i = c[i]
            # No overlap found
            if c_i == 0:
                continue
            target = i0 + i
            # Check if candidate is already used
            c_target = ids["previous_cost"][target]
            if (c_target != 0 and c_target < c_i) or c_target == 0:
                ids["previous_cost"][target] = c_i
            ids["next_cost"][i_current] = c_i
            return target
        return -1


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
        mask[first_index[track] : first_index[track] + number_of_obs[track]] = True


@njit(cache=True)
def track_loess_filter(half_window, x, y, track):
    """
    Apply a loess filter on y field

    :param int,float window: parameter of smoother
    :param array_like x: must be growing for each track but could be irregular
    :param array_like y: field to smooth
    :param array_like track: field which allow to separate path

    :return: Array smoothed
    :rtype: array_like

    """
    nb = y.shape[0]
    last = nb - 1
    y_new = empty(y.shape, dtype=y.dtype)
    for i in range(nb):
        cur_track = track[i]
        y_sum = y[i]
        w_sum = 1
        if i != 0:
            i_previous = i - 1
            dx = x[i] - x[i_previous]
            while (
                dx < half_window and i_previous != 0 and cur_track == track[i_previous]
            ):
                w = (1 - (dx / half_window) ** 3) ** 3
                y_sum += y[i_previous] * w
                w_sum += w
                i_previous -= 1
                dx = x[i] - x[i_previous]
        if i != last:
            i_next = i + 1
            dx = x[i_next] - x[i]
            while dx < half_window and i_next != last and cur_track == track[i_next]:
                w = (1 - (dx / half_window) ** 3) ** 3
                y_sum += y[i_next] * w
                w_sum += w
                i_next += 1
                dx = x[i_next] - x[i]
        y_new[i] = y_sum / w_sum
    return y_new


@njit(cache=True)
def track_median_filter(half_window, x, y, track):
    """
    Apply a loess filter on y field

    :param int,float half_window: parameter of smoother
    :param array_like x: must be growing for each track but could be irregular
    :param array_like y: field to smooth
    :param array_like track: field which allow to separate path

    :return: Array smoothed
    :rtype: array_like

    """
    nb = y.shape[0]
    y_new = empty(y.shape, dtype=y.dtype)
    i_previous, i_next = 0, 0
    for i in range(nb):
        cur_track = track[i]
        while x[i] - x[i_previous] > half_window or cur_track != track[i_previous]:
            i_previous += 1
        while (
            i_next < nb
            and x[i_next] - x[i] <= half_window
            and cur_track == track[i_next]
        ):
            i_next += 1
        y_new[i] = median(y[i_previous:i_next])
    return y_new
