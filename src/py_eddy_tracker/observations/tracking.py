# -*- coding: utf-8 -*-
"""
Class to manage observations gathered in trajectories
"""
import logging
from datetime import datetime, timedelta

from numba import njit
from numpy import (
    arange,
    arctan2,
    array,
    bool_,
    concatenate,
    cos,
    degrees,
    empty,
    histogram,
    interp,
    median,
    nan,
    ones,
    radians,
    sin,
    unique,
    zeros,
)

from .. import VAR_DESCR_inv, __version__
from ..generic import build_index, cumsum_by_track, distance, split_line, wrap_longitude
from ..poly import bbox_intersection, merge, vertice_overlap
from .observation import EddiesObservations

logger = logging.getLogger("pet")


class TrackEddiesObservations(EddiesObservations):
    """Class to practice Tracking on observations"""

    __slots__ = ("__obs_by_track", "__first_index_of_track", "__nb_track")

    ELEMENTS = [
        "lon",
        "lat",
        "radius_s",
        "radius_e",
        "speed_area",
        "effective_area",
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
        self.__nb_track = None

    def iter_track(self):
        """
        Yield track
        """
        for i0, nb in zip(self.index_from_track, self.nb_obs_by_track):
            if nb == 0:
                continue
            yield self.index(slice(i0, i0 + nb))

    @property
    def nb_tracks(self):
        """
        Will count and send number of track
        """
        if self.__nb_track is None:
            if len(self) == 0:
                self.__nb_track = 0
            else:
                self.__nb_track = (self.nb_obs_by_track != 0).sum()
        return self.__nb_track

    def __repr__(self):
        content = super().__repr__()
        t0, t1 = self.period
        period = t1 - t0 + 1
        nb = self.nb_obs_by_track
        nb_obs = self.observations.shape[0]
        m = self.virtual.astype("bool")
        nb_m = m.sum()
        bins_t = (1, 30, 90, 180, 270, 365, 1000, 10000)
        nb_tracks_by_t = histogram(nb, bins=bins_t)[0]
        nb_obs_by_t = histogram(nb, bins=bins_t, weights=nb)[0]
        pct_tracks_by_t = nb_tracks_by_t / nb_tracks_by_t.sum() * 100.0
        pct_obs_by_t = nb_obs_by_t / nb_obs_by_t.sum() * 100.0
        d = self.distance_to_next() / 1000.0
        cum_d = cumsum_by_track(d, self.tracks)
        m_last = ones(d.shape, dtype="bool")
        m_last[-1] = False
        m_last[self.index_from_track[1:] - 1] = False
        content += f"""
    | {self.nb_tracks} tracks ({
        nb_obs / self.nb_tracks:.2f} obs/tracks, shorter {nb[nb!=0].min()} obs, longer {nb.max()} obs)
    |   {nb_m} filled observations ({nb_m / self.nb_tracks:.2f} obs/tracks, {nb_m / nb_obs * 100:.2f} % of total)
    |   Intepolated speed area      : {self.speed_area[m].sum() / period / 1e12:.2f} Mkm²/day
    |   Intepolated effective area  : {self.effective_area[m].sum() / period / 1e12:.2f} Mkm²/day
    |   Distance by day             : Mean {d[m_last].mean():.2f} , Median {median(d[m_last]):.2f} km/day
    |   Distance by track           : Mean {cum_d[~m_last].mean():.2f} , Median {median(cum_d[~m_last]):.2f} km/track
    ----Distribution in lifetime:
    |   Lifetime (days  )      {self.box_display(bins_t)}
    |   Percent of tracks         : {self.box_display(pct_tracks_by_t)}
    |   Percent of eddies         : {self.box_display(pct_obs_by_t)}"""
        return content

    def add_distance(self):
        """Add a field of distance (m) between two consecutive observations, 0 for the last observation of each track"""
        if "distance_next" in self.observations.dtype.descr:
            return self
        new = self.add_fields(("distance_next",))
        new["distance_next"][:1] = self.distance_to_next()
        return new

    def distance_to_next(self):
        """
        :return: array of distance in m, 0 when next obs if from another track
        :rtype: array
        """
        d = distance(
            self.longitude[:-1],
            self.latitude[:-1],
            self.longitude[1:],
            self.latitude[1:],
        )
        d[self.index_from_track[1:] - 1] = 0
        d_ = empty(d.shape[0] + 1, dtype=d.dtype)
        d_[:-1] = d
        d_[-1] = 0
        return d_

    def filled_by_interpolation(self, mask):
        """Filled selected values by interpolation

        :param array(bool) mask: True if must be filled by interpolation

        .. minigallery:: py_eddy_tracker.TrackEddiesObservations.filled_by_interpolation
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
            self.obs[var][mask] = interp(
                index[mask], index[~mask], self.obs[var][~mask]
            )

    def normalize_longitude(self):
        """Normalize all longitude

        Normalize longitude field and in the same range :
        - longitude_max
        - contour_lon_e (how to do if in raw)
        - contour_lon_s (how to do if in raw)
        """
        lon0 = (self.lon[self.index_from_track] - 180).repeat(self.nb_obs_by_track)
        logger.debug("Normalize longitude")
        self.lon[:] = (self.lon - lon0) % 360 + lon0
        if "lon_max" in self.obs.dtype.names:
            logger.debug("Normalize longitude_max")
            self.lon_max[:] = (self.lon_max - self.lon + 180) % 360 + self.lon - 180
        if not self.raw_data:
            if "contour_lon_e" in self.obs.dtype.names:
                logger.debug("Normalize effective contour longitude")
                self.contour_lon_e[:] = (
                    (self.contour_lon_e.T - self.lon + 180) % 360 + self.lon - 180
                ).T
            if "contour_lon_s" in self.obs.dtype.names:
                logger.debug("Normalize speed contour longitude")
                self.contour_lon_s[:] = (
                    (self.contour_lon_s.T - self.lon + 180) % 360 + self.lon - 180
                ).T

    def extract_longer_eddies(self, nb_min, nb_obs, compress_id=True):
        """Select the trajectories longer than nb_min"""
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
            list_id = unique(eddies.obs.track)
            list_id.sort()
            id_translate = arange(list_id.max() + 1)
            id_translate[list_id] = arange(len(list_id)) + 1
            eddies.track = id_translate[eddies.track]
        return eddies

    @property
    def elements(self):
        elements = super().elements
        elements.extend(["track", "n", "virtual"])
        return list(set(elements))

    def set_global_attr_netcdf(self, h_nc):
        """Set global attr"""
        h_nc.title = "Cyclonic" if self.sign_type == -1 else "Anticyclonic"
        h_nc.Metadata_Conventions = "Unidata Dataset Discovery v1.0"
        h_nc.comment = "Surface product; mesoscale eddies"
        h_nc.framework_used = "https://github.com/AntSimi/py-eddy-tracker"
        h_nc.framework_version = __version__
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

    def extract_with_period(self, period, **kwargs):
        """
        Extract within a time period

        :param (int,int) period: two dates to define the period, must be specify from 1/1/1950
        :param dict kwargs: look at :py:meth:`extract_with_mask`
        :return: Return all eddy tracks which are in bounds
        :rtype: TrackEddiesObservations

        .. minigallery:: py_eddy_tracker.TrackEddiesObservations.extract_with_period
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
        return self.extract_with_mask(mask, **kwargs)

    def get_azimuth(self, equatorward=False):
        """
        Return azimuth for each track.

        Azimuth is computed with first and last observation

        :param bool equatorward: If True,  Poleward are positive and equatorward negative
        :rtype: array
        """
        i0, nb = self.index_from_track, self.nb_obs_by_track
        i0 = i0[nb != 0]
        i1 = i0 - 1 + nb[nb != 0]
        lat0, lon0 = self.latitude[i0], self.longitude[i0]
        lat1, lon1 = self.latitude[i1], self.longitude[i1]
        lat0, lon0 = radians(lat0), radians(lon0)
        lat1, lon1 = radians(lat1), radians(lon1)
        dlon = lon1 - lon0
        x = cos(lat0) * sin(lat1) - sin(lat0) * cos(lat1) * cos(dlon)
        y = sin(dlon) * cos(lat1)
        azimuth = degrees(arctan2(y, x)) + 90
        if equatorward:
            south = lat0 < 0
            azimuth[south] *= -1
        return azimuth

    def get_mask_from_id(self, tracks):
        mask = zeros(self.tracks.shape, dtype=bool_)
        compute_mask_from_id(tracks, self.index_from_track, self.nb_obs_by_track, mask)
        return mask

    def compute_index(self):
        """
        If obs are not sorted by track, __first_index_of_track will be unusable
        """
        if self.__first_index_of_track is None:
            s = self.tracks.max() + 1
            # Doesn't work => core dump with numba, maybe he wants i8 instead of u4
            # self.__first_index_of_track = -ones(s, self.tracks.dtype)
            # self.__obs_by_track = zeros(s, self.observation_number.dtype)
            self.__first_index_of_track = -ones(s, "i8")
            self.__obs_by_track = zeros(s, "i8")
            logger.debug("Start computing index ...")
            compute_index(self.tracks, self.__first_index_of_track, self.__obs_by_track)
            logger.debug("... OK")

    @classmethod
    def concatenate(cls, observations):
        eddies = super().concatenate(observations)
        last_track = 0
        i_start = 0
        for obs in observations:
            nb_obs = len(obs)
            sl = slice(i_start, i_start + nb_obs)
            new_track = obs.track + last_track
            eddies.track[sl] = new_track
            last_track = new_track.max() + 1
            i_start += nb_obs
        return eddies

    def count_by_track(self, mask):
        """
        Count by track

        :param array[bool] mask: Mask of boolean count +1 if true
        :return: Return count by track
        :rtype: array
        """
        s = self.tracks.max() + 1
        obs_by_track = zeros(s, "i4")
        count_by_track(self.tracks, mask, obs_by_track)
        return obs_by_track

    @property
    def index_from_track(self):
        self.compute_index()
        return self.__first_index_of_track

    @property
    def nb_obs_by_track(self):
        self.compute_index()
        return self.__obs_by_track

    @property
    def lifetime(self):
        """Return lifetime for each observation"""
        return self.nb_obs_by_track.repeat(self.nb_obs_by_track)

    @property
    def age(self):
        """Return age in % for each observation, will be [0:100]"""
        return self.n.astype("f4") / (self.lifetime - 1) * 100.0

    def extract_ids(self, tracks):
        mask = self.get_mask_from_id(array(tracks))
        return self.extract_with_mask(mask)

    def extract_toward_direction(self, west=True, delta_lon=None):
        """
        Get trajectories going in the same direction

        :param bool west: Only eastward eddies if True return westward
        :param None,float delta_lon: Only eddies with more than delta_lon span in longitude
        :return: Only eastern eddy
        :rtype: __class__

        .. minigallery:: py_eddy_tracker.TrackEddiesObservations.extract_toward_direction
        """
        lon = self.longitude
        i0, nb = self.index_from_track, self.nb_obs_by_track
        i1 = i0 - 1 + nb
        d_lon = lon[i1] - lon[i0]
        m = d_lon < 0 if west else d_lon > 0
        if delta_lon is not None:
            m *= delta_lon < d_lon
        m = m.repeat(nb)
        return self.extract_with_mask(m)

    def extract_first_obs_in_box(self, res):
        data = empty(
            self.obs.shape, dtype=[("lon", "f4"), ("lat", "f4"), ("track", "i4")]
        )
        data["lon"] = self.longitude - self.longitude % res
        data["lat"] = self.latitude - self.latitude % res
        data["track"] = self.track
        _, indexs = unique(data, return_index=True)
        mask = zeros(self.obs.shape, dtype="bool")
        mask[indexs] = True
        return self.extract_with_mask(mask)

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
        return self.extract_with_mask(mask)

    def extract_with_length(self, bounds):
        """
        Return the observations within trajectories lasting between [b0:b1]

        :param (int,int) bounds: length min and max of the desired trajectories, if -1 this bound is not used
        :return: Return all trajectories having length between bounds
        :rtype: TrackEddiesObservations

        .. minigallery:: py_eddy_tracker.TrackEddiesObservations.extract_with_length
        """
        if len(self) == 0:
            return self.empty_dataset()
        b0, b1 = bounds
        if b0 >= 0 and b1 != -1:
            track_mask = (self.nb_obs_by_track >= b0) * (self.nb_obs_by_track <= b1)
        elif b0 == -1 and b1 >= 0:
            track_mask = self.nb_obs_by_track <= b1
        elif b0 >= 0 and b1 == -1:
            track_mask = self.nb_obs_by_track >= b0
        else:
            logger.warning("No valid value for bounds")
            raise Exception("One bounds must be positiv")
        return self.extract_with_mask(track_mask.repeat(self.nb_obs_by_track))

    def empty_dataset(self):
        return self.new_like(self, 0)

    def loess_filter(self, half_window, xfield, yfield, inplace=True):
        track = self.track
        x = self.obs[xfield]
        y = self.obs[yfield]
        result = track_loess_filter(half_window, x, y, track)
        if inplace:
            self.obs[yfield] = result
            return self

    def median_filter(self, half_window, xfield, yfield, inplace=True):
        result = track_median_filter(
            half_window, self[xfield], self[yfield], self.track
        )
        if inplace:
            self[yfield][:] = result
            return self
        return result

    def position_filter(self, median_half_window, loess_half_window):
        self.median_filter(median_half_window, "time", "lon").loess_filter(
            loess_half_window, "time", "lon"
        )
        self.median_filter(median_half_window, "time", "lat").loess_filter(
            loess_half_window, "time", "lat"
        )

    def extract_with_mask(
        self,
        mask,
        full_path=False,
        remove_incomplete=False,
        compress_id=False,
        reject_virtual=False,
    ):
        """
        Extract a subset of observations

        :param array(bool) mask: mask to select observations
        :param bool full_path: extract the full trajectory if only one part is selected
        :param bool remove_incomplete: delete trajectory if not fully selected
        :param bool compress_id: resample trajectory number to use a smaller range
        :param bool reject_virtual: if only virtual are selected, the trajectory is removed
        :return: same object with the selected observations
        :rtype: self.__class__
        """
        if full_path and remove_incomplete:
            logger.warning(
                "Incompatible option, remove_incomplete option will be remove"
            )
            # remove_incomplete = False

        if full_path:
            if reject_virtual:
                mask *= ~self.virtual.astype("bool")
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
                list_id = unique(new.track)
                list_id.sort()
                id_translate = arange(list_id.max() + 1)
                id_translate[list_id] = arange(len(list_id)) + 1
                new.track = id_translate[new.track]
        return new

    @staticmethod
    def re_reference_index(index, ref):
        if isinstance(ref, slice):
            return index + ref.start
        else:
            return ref[index]

    def shape_polygon(self, intern=False):
        """
        Get the polygon enclosing each trajectory.

        The polygon merges the non-overlapping bounds of the specified contours

        :param bool intern: If True use speed contour instead of effective contour
        :rtype: list(array, array)
        """
        xname, yname = self.intern(intern)
        return [merge(track[xname], track[yname]) for track in self.iter_track()]

    def display_shape(self, ax, ref=None, intern=False, **kwargs):
        """
        This function will draw the shape of each trajectory

        :param matplotlib.axes.Axes ax: ax to draw
        :param float,int ref: if defined all coordinates will be wrapped with ref like west boundary
        :param bool intern: If True use speed contour instead of effective contour
        :param dict kwargs: keyword arguments for Axes.plot
        :return: matplotlib mappable
        """
        if "label" in kwargs:
            kwargs["label"] = self.format_label(kwargs["label"])
        if len(self) == 0:
            x, y = [], []
        else:
            polygons = self.shape_polygon(intern)
            x, y = list(), list()
            for p_ in polygons:
                x.append((nan,))
                y.append((nan,))
                x.append(p_[0])
                y.append(p_[1])
            x, y = concatenate(x), concatenate(y)
            if ref is not None:
                x, y = wrap_longitude(x, y, ref, cut=True)
        return ax.plot(x, y, **kwargs)

    def close_tracks(self, other, nb_obs_min=10, **kwargs):
        """
        Get close trajectories from another atlas.

        :param self other: Atlas to compare
        :param int nb_obs_min: Minimal number of overlap for one trajectory
        :param dict kwargs: keyword arguments for match function
        :return: return other atlas reduce to common track with self

        .. warning::
            It could be a costly operation for huge dataset
        """
        p0, p1 = self.period
        indexs = list()
        for i_self, i_other, t0, t1 in self.align_on(other, bins=range(p0, p1 + 2)):
            i, j, s = self.index(i_self).match(other.index(i_other), **kwargs)
            indexs.append(other.re_reference_index(j, i_other))
        indexs = concatenate(indexs)
        tr, nb = unique(other.track[indexs], return_counts=True)
        return other.extract_ids(tr[nb >= nb_obs_min])

    def format_label(self, label):
        t0, t1 = self.period
        return label.format(
            t0=t0,
            t1=t1,
            nb_obs=len(self),
            nb_tracks=(self.nb_obs_by_track != 0).sum(),
        )

    def plot(self, ax, ref=None, **kwargs):
        """
        This function will draw path of each trajectory

        :param matplotlib.axes.Axes ax: ax to draw
        :param float,int ref: if defined, all coordinates will be wrapped with ref like west boundary
        :param dict kwargs: keyword arguments for Axes.plot
        :return: matplotlib mappable
        """
        if "label" in kwargs:
            kwargs["label"] = self.format_label(kwargs["label"])
        if len(self) == 0:
            x, y = [], []
        else:
            x, y = split_line(self.longitude, self.latitude, self.tracks)
            if ref is not None:
                x, y = wrap_longitude(x, y, ref, cut=True)
        return ax.plot(x, y, **kwargs)

    def split_network(self, intern=True, **kwargs):
        """Return each group (network) divided in segments"""
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
        # Initialisation
        # To store the id of the segments, the backward and forward cost associations
        ids["track"], ids["previous_cost"], ids["next_cost"] = 0, 0, 0
        # To store the indexes of the backward and forward observations associated
        ids["previous_obs"], ids["next_obs"] = -1, -1
        # At the end, ids["previous_obs"] == -1 means the start of a non-split segment
        # and ids["next_obs"] == -1 means the end of a non-merged segment

        xname, yname = self.intern(intern)
        display_iteration = logger.getEffectiveLevel() == logging.INFO
        for i_s, i_e in zip(track_s, track_e):
            if i_s == i_e or self.tracks[i_s] == self.NOGROUP:
                continue
            if display_iteration:
                print(f"Network obs from {i_s} to {i_e} on {track_e[-1]}", end="\r")
            sl = slice(i_s, i_e)
            local_ids = ids[sl]
            # built segments with local indices
            self.set_tracks(self[xname][sl], self[yname][sl], local_ids, **kwargs)
            # shift the local indices to the total indexation for the used observations
            m = local_ids["previous_obs"] != -1
            local_ids["previous_obs"][m] += i_s
            m = local_ids["next_obs"] != -1
            local_ids["next_obs"][m] += i_s
        if display_iteration:
            print()
        return ids

    def set_tracks(self, x, y, ids, window, **kwargs):
        """
        Will split one group (network) in segments

        :param array x: coordinates of group
        :param array y: coordinates of group
        :param ndarray ids: several fields like time, group, ...
        :param int windows: number of days where observations could missed
        """

        time_index = build_index(ids["time"])
        nb = x.shape[0]
        used = zeros(nb, dtype="bool")
        track_id = 1
        # build all polygons (need to check if wrap is needed)
        for i in range(nb):
            # If the observation is already in one track, we go to the next one
            if used[i]:
                continue
            # Search a possible continuation (forward)
            self.follow_obs(i, track_id, used, ids, x, y, *time_index, window, **kwargs)
            track_id += 1
            # Search a possible ancestor (backward)
            self.previous_obs(i, ids, x, y, *time_index, window, **kwargs)

    @classmethod
    def follow_obs(cls, i_next, track_id, used, ids, *args, **kwargs):
        """Associate the observations to the segments"""

        while i_next != -1:
            # Flag
            used[i_next] = True
            # Assign id
            ids["track"][i_next] = track_id
            # Search next
            i_next_ = cls.next_obs(i_next, ids, *args, **kwargs)
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
    def previous_obs(i_current, ids, x, y, time_s, time_e, time_ref, window, **kwargs):
        """Backward association of observations to the segments"""

        time_cur = ids["time"][i_current]
        t0, t1 = time_cur - 1 - time_ref, max(time_cur - window - time_ref, 0)
        for t_step in range(t0, t1 - 1, -1):
            i0, i1 = time_s[t_step], time_e[t_step]
            # No observation at the time step
            if i0 == i1:
                continue
            # Search for overlaps
            xi, yi, xj, yj = x[[i_current]], y[[i_current]], x[i0:i1], y[i0:i1]
            ii, ij = bbox_intersection(xi, yi, xj, yj)
            if len(ii) == 0:
                continue
            c = zeros(len(xj))
            c[ij] = vertice_overlap(xi[ii], yi[ii], xj[ij], yj[ij], **kwargs)
            # We remove low overlap
            c[c < 0.01] = 0
            # We get index of maximal overlap
            i = c.argmax()
            c_i = c[i]
            # No overlap found
            if c_i == 0:
                continue
            ids["previous_cost"][i_current] = c_i
            ids["previous_obs"][i_current] = i0 + i
            break

    @staticmethod
    def next_obs(i_current, ids, x, y, time_s, time_e, time_ref, window, **kwargs):
        """Forward association of observations to the segments"""
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
            # Search for overlaps
            xi, yi, xj, yj = x[[i_current]], y[[i_current]], x[i0:i1], y[i0:i1]
            ii, ij = bbox_intersection(xi, yi, xj, yj)
            if len(ii) == 0:
                continue
            c = zeros(len(xj))
            c[ij] = vertice_overlap(xi[ii], yi[ii], xj[ij], yj[ij], **kwargs)
            # We remove low overlap
            c[c < 0.01] = 0
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
def count_by_track(tracks, mask, number):
    for track, test in zip(tracks, mask):
        if test:
            number[track] += 1


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
    Apply a median filter on y field

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
