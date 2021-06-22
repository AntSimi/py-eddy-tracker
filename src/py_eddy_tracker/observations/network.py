# -*- coding: utf-8 -*-
"""
Class to create network of observations
"""
import logging
import time
from glob import glob

from numba import njit
from numpy import (
    arange,
    array,
    bincount,
    bool_,
    concatenate,
    empty,
    in1d,
    ones,
    uint16,
    uint32,
    unique,
    where,
    zeros,
)

from ..dataset.grid import GridCollection
from ..generic import build_index, wrap_longitude
from ..poly import bbox_intersection, vertice_overlap
from .groups import GroupEddiesObservations, get_missing_indices, particle_candidate
from .observation import EddiesObservations
from .tracking import TrackEddiesObservations, track_loess_filter, track_median_filter

logger = logging.getLogger("pet")


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class Buffer(metaclass=Singleton):
    __slots__ = (
        "buffersize",
        "contour_name",
        "xname",
        "yname",
        "memory",
    )
    DATA = dict()
    FLIST = list()

    def __init__(self, buffersize, intern=False, memory=False):
        self.buffersize = buffersize
        self.contour_name = EddiesObservations.intern(intern, public_label=True)
        self.xname, self.yname = EddiesObservations.intern(intern)
        self.memory = memory

    def load_contour(self, filename):
        if isinstance(filename, EddiesObservations):
            return filename[self.xname], filename[self.yname]
        if filename not in self.DATA:
            if len(self.FLIST) > self.buffersize:
                self.DATA.pop(self.FLIST.pop(0))
            if self.memory:
                # Only if netcdf
                with open(filename, "rb") as h:
                    e = EddiesObservations.load_file(h, include_vars=self.contour_name)
            else:
                e = EddiesObservations.load_file(
                    filename, include_vars=self.contour_name
                )
            self.FLIST.append(filename)
            self.DATA[filename] = e[self.xname], e[self.yname]
        return self.DATA[filename]


@njit(cache=True)
def fix_next_previous_obs(next_obs, previous_obs, flag_virtual):
    """When an observation is virtual, we have to fix the previous and next obs

    :param np.array(int)  next_obs    : index of next observation from network
    :param np.array(int   previous_obs: index of previous observation from network
    :param np.array(bool) flag_virtual: if observation is virtual or not
    """

    for i_o in range(next_obs.size):
        if not flag_virtual[i_o]:
            continue

        # if there are several consecutive virtuals, some values are written multiple times.
        # but it should not be slow
        next_obs[i_o - 1] = i_o
        next_obs[i_o] = i_o + 1
        previous_obs[i_o] = i_o - 1
        previous_obs[i_o + 1] = i_o


class NetworkObservations(GroupEddiesObservations):

    __slots__ = ("_index_network",)

    NOGROUP = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index_network = None

    def find_segments_relative(self, obs, stopped=None, order=1):
        """
        Find all relative segments from obs linked with merging/splitting events at a specific order.

        :param int obs: index of observation after the event
        :param int stopped: index of observation before the event
        :param int order: order of relatives accepted
        :return: all relative segments
        :rtype: EddiesObservations
        """

        # extraction of network where the event is
        network_id = self.tracks[obs]
        nw = self.network(network_id)

        # indice of observation in new subnetwork
        i_obs = where(nw.segment == self.segment[obs])[0][0]

        if stopped is None:
            return nw.relatives(i_obs, order=order)

        else:
            i_stopped = where(nw.segment == self.segment[stopped])[0][0]
            return nw.relatives([i_obs, i_stopped], order=order)

    def get_missing_indices(self, dt):
        """Find indices where observations are missing.

        As network have all untracked observation in tracknumber `self.NOGROUP`,
            we don't compute them

        :param int,float dt: theorical delta time between 2 observations
        """
        return get_missing_indices(
            self.time, self.track, dt=dt, flag_untrack=True, indice_untrack=self.NOGROUP
        )

    def fix_next_previous_obs(self):
        """function used after 'insert_virtual', to correct next_obs and
        previous obs.
        """

        fix_next_previous_obs(self.next_obs, self.previous_obs, self.virtual)

    @property
    def index_network(self):
        if self._index_network is None:
            self._index_network = build_index(self.track)
        return self._index_network

    def network_slice(self, id_network):
        """
        Return slice for one network

        :param int id_network: id to identify network
        """
        i = id_network - self.index_network[2]
        i_start, i_stop = self.index_network[0][i], self.index_network[1][i]
        return slice(i_start, i_stop)

    @property
    def elements(self):
        elements = super().elements
        elements.extend(
            [
                "track",
                "segment",
                "next_obs",
                "previous_obs",
                "next_cost",
                "previous_cost",
            ]
        )
        return list(set(elements))

    def astype(self, cls):
        new = cls.new_like(self, self.shape)
        print()
        for k in new.obs.dtype.names:
            if k in self.obs.dtype.names:
                new[k][:] = self[k][:]
        new.sign_type = self.sign_type
        return new

    def longer_than(self, nb_day_min=-1, nb_day_max=-1):
        """
        Select network on time duration

        :param int nb_day_min: Minimal number of days covered by one network, if negative -> not used
        :param int nb_day_max: Maximal number of days covered by one network, if negative -> not used
        """
        if nb_day_max < 0:
            nb_day_max = 1000000000000
        mask = zeros(self.shape, dtype="bool")
        t = self.time
        for i, b0, b1 in self.iter_on(self.track):
            nb = i.stop - i.start
            if nb == 0:
                continue
            if nb_day_min <= ptp(t[i]) <= nb_day_max:
                mask[i] = True
        return self.extract_with_mask(mask)

    @classmethod
    def from_split_network(cls, group_dataset, indexs, **kwargs):
        """
        Build a NetworkObservations object with Group dataset and indices

        :param TrackEddiesObservations group_dataset: Group dataset
        :param indexs: result from split_network
        :return: NetworkObservations
        """
        index_order = indexs.argsort(order=("group", "track", "time"))
        network = cls.new_like(group_dataset, len(group_dataset), **kwargs)
        network.sign_type = group_dataset.sign_type
        for field in group_dataset.elements:
            if field not in network.elements:
                continue
            network[field][:] = group_dataset[field][index_order]
        network.segment[:] = indexs["track"][index_order]
        # n & p must be re-indexed
        n, p = indexs["next_obs"][index_order], indexs["previous_obs"][index_order]
        # we add 2 for -1 index return index -1
        translate = -ones(index_order.max() + 2, dtype="i4")
        translate[index_order] = arange(index_order.shape[0])
        network.next_obs[:] = translate[n]
        network.previous_obs[:] = translate[p]
        network.next_cost[:] = indexs["next_cost"][index_order]
        network.previous_cost[:] = indexs["previous_cost"][index_order]
        return network

    def infos(self, label=""):
        return f"{len(self)} obs {unique(self.segment).shape[0]} segments"

    def correct_close_events(self, nb_days_max=20):
        """
        Transform event where
        segment A splits from segment B, then x days after segment B merges with A

        to

        segment A splits from segment B then x days after segment A merges with B (B will be longer)

        These events have to last less than `nb_days_max` to be changed.

        :param float nb_days_max: maximum time to search for splitting-merging event
        """

        _time = self.time
        # segment used to correct and track changes
        segment = self.segment_track_array.copy()
        # final segment used to copy into self.segment
        segment_copy = self.segment

        segments_connexion = dict()

        previous_obs, next_obs = self.previous_obs, self.next_obs

        # record for every segment the slice, index of next obs & index of previous obs
        for i, seg, _ in self.iter_on(segment):
            if i.start == i.stop:
                continue

            i_p, i_n = previous_obs[i.start], next_obs[i.stop - 1]
            segments_connexion[seg] = [i, i_p, i_n]

        for seg in sorted(segments_connexion.keys()):
            seg_slice, i_seg_p, i_seg_n = segments_connexion[seg]

            # the segment ID has to be corrected, because we may have changed it since
            seg_corrected = segment[seg_slice.stop - 1]

            # we keep the real segment number
            seg_corrected_copy = segment_copy[seg_slice.stop - 1]

            n_seg = segment[i_seg_n]

            # if segment is split
            if i_seg_n != -1:
                seg2_slice, i2_seg_p, i2_seg_n = segments_connexion[n_seg]
                p2_seg = segment[i2_seg_p]

                # if it merges on the first in a certain time
                if (p2_seg == seg_corrected) and (
                    _time[i_seg_n] - _time[i2_seg_p] < nb_days_max
                ):
                    my_slice = slice(i_seg_n, seg2_slice.stop)
                    # correct the factice segment
                    segment[my_slice] = seg_corrected
                    # correct the good segment
                    segment_copy[my_slice] = seg_corrected_copy
                    previous_obs[i_seg_n] = seg_slice.stop - 1

                    segments_connexion[seg_corrected][0] = my_slice

        self.segment[:] = segment_copy
        self.previous_obs[:] = previous_obs

        self.sort()

    def sort(self, order=("track", "segment", "time")):
        """
        Sort observations

        :param tuple order: order or sorting. Given to :func:`numpy.argsort`
        """
        index_order = self.obs.argsort(order=order)
        for field in self.elements:
            self[field][:] = self[field][index_order]

        translate = -ones(index_order.max() + 2, dtype="i4")
        translate[index_order] = arange(index_order.shape[0])
        self.next_obs[:] = translate[self.next_obs]
        self.previous_obs[:] = translate[self.previous_obs]

    def obs_relative_order(self, i_obs):
        self.only_one_network()
        return self.segment_relative_order(self.segment[i_obs])

    def find_link(self, i_observations, forward=True, backward=False):
        """
        Find all observations where obs `i_observation` could be
        in future or past.

        If forward=True, search all observations where water
        from obs "i_observation" could go

        If backward=True, search all observation
        where water from obs `i_observation` could come from

        :param int,iterable(int) i_observation:
            indices of observation. Can be
            int, or iterable of int.
        :param bool forward, backward:
            if forward, search observations after obs.
            else mode==backward search before obs

        """

        i_obs = (
            [i_observations]
            if not hasattr(i_observations, "__iter__")
            else i_observations
        )

        segment = self.segment_track_array
        previous_obs, next_obs = self.previous_obs, self.next_obs

        segments_connexion = dict()

        for i_slice, seg, _ in self.iter_on(segment):
            if i_slice.start == i_slice.stop:
                continue

            i_p, i_n = previous_obs[i_slice.start], next_obs[i_slice.stop - 1]
            p_seg, n_seg = segment[i_p], segment[i_n]

            # dumping slice into dict
            if seg not in segments_connexion:
                segments_connexion[seg] = [i_slice, [], []]
            else:
                segments_connexion[seg][0] = i_slice

            if i_p != -1:

                if p_seg not in segments_connexion:
                    segments_connexion[p_seg] = [None, [], []]

                # backward
                segments_connexion[seg][2].append((i_slice.start, i_p, p_seg))
                # forward
                segments_connexion[p_seg][1].append((i_p, i_slice.start, seg))

            if i_n != -1:
                if n_seg not in segments_connexion:
                    segments_connexion[n_seg] = [None, [], []]

                # forward
                segments_connexion[seg][1].append((i_slice.stop - 1, i_n, n_seg))
                # backward
                segments_connexion[n_seg][2].append((i_n, i_slice.stop - 1, seg))

        mask = zeros(segment.size, dtype=bool)

        def func_forward(seg, indice):
            seg_slice, _forward, _ = segments_connexion[seg]

            mask[indice : seg_slice.stop] = True
            for i_begin, i_end, seg2 in _forward:
                if i_begin < indice:
                    continue

                if not mask[i_end]:
                    func_forward(seg2, i_end)

        def func_backward(seg, indice):
            seg_slice, _, _backward = segments_connexion[seg]

            mask[seg_slice.start : indice + 1] = True
            for i_begin, i_end, seg2 in _backward:
                if i_begin > indice:
                    continue

                if not mask[i_end]:
                    func_backward(seg2, i_end)

        for indice in i_obs:
            if forward:
                func_forward(segment[indice], indice)

            if backward:
                func_backward(segment[indice], indice)

        return self.extract_with_mask(mask)

    def connexions(self, multi_network=False):
        """
        Create dictionnary for each segment, gives the segments in interaction with
        """
        if multi_network:
            segment = self.segment_track_array
        else:
            self.only_one_network()
            segment = self.segment
        segments_connexion = dict()

        def add_seg(father, child):
            if father not in segments_connexion:
                segments_connexion[father] = set()
            segments_connexion[father].add(child)

        previous_obs, next_obs = self.previous_obs, self.next_obs
        for i, seg, _ in self.iter_on(segment):
            if i.start == i.stop:
                continue
            i_p, i_n = previous_obs[i.start], next_obs[i.stop - 1]
            # segment in interaction
            p_seg, n_seg = segment[i_p], segment[i_n]
            # Where segment are called
            if i_p != -1:
                add_seg(p_seg, seg)
                add_seg(seg, p_seg)
            if i_n != -1:
                add_seg(n_seg, seg)
                add_seg(seg, n_seg)
        return segments_connexion

    @classmethod
    def __close_segment(cls, father, shift, connexions, distance):
        i_father = father - shift
        if distance[i_father] == -1:
            distance[i_father] = 0
        d_target = distance[i_father] + 1
        for son in connexions.get(father, list()):
            i_son = son - shift
            d_son = distance[i_son]
            if d_son == -1 or d_son > d_target:
                distance[i_son] = d_target
            else:
                continue
            cls.__close_segment(son, shift, connexions, distance)

    def segment_relative_order(self, seg_origine):
        """
        Compute the relative order of each segment to the chosen segment
        """
        i_s, i_e, i_ref = build_index(self.segment)
        segment_connexions = self.connexions()
        relative_tr = -ones(i_s.shape, dtype="i4")
        self.__close_segment(seg_origine, i_ref, segment_connexions, relative_tr)
        d = -ones(self.shape)
        for i0, i1, v in zip(i_s, i_e, relative_tr):
            if i0 == i1:
                continue
            d[i0:i1] = v
        return d

    def relatives(self, obs, order=2):
        """
        Extract the segments at a certain order from multiple observations.

        :param iterable,int obs:
            indices of observation for relatives computation. Can be one observation (int)
            or collection of observations (iterable(int))
        :param int order: order of relatives wanted. 0 means only observations in obs, 1 means direct relatives, ...
        :return: all segments' relatives
        :rtype: EddiesObservations
        """
        segment = self.segment_track_array
        previous_obs, next_obs = self.previous_obs, self.next_obs

        segments_connexion = dict()

        for i_slice, seg, _ in self.iter_on(segment):
            if i_slice.start == i_slice.stop:
                continue

            i_p, i_n = previous_obs[i_slice.start], next_obs[i_slice.stop - 1]
            p_seg, n_seg = segment[i_p], segment[i_n]

            # dumping slice into dict
            if seg not in segments_connexion:
                segments_connexion[seg] = [i_slice, []]
            else:
                segments_connexion[seg][0] = i_slice

            if i_p != -1:

                if p_seg not in segments_connexion:
                    segments_connexion[p_seg] = [None, []]

                # backward
                segments_connexion[seg][1].append(p_seg)
                segments_connexion[p_seg][1].append(seg)

            if i_n != -1:
                if n_seg not in segments_connexion:
                    segments_connexion[n_seg] = [None, []]

                # forward
                segments_connexion[seg][1].append(n_seg)
                segments_connexion[n_seg][1].append(seg)

        i_obs = [obs] if not hasattr(obs, "__iter__") else obs
        distance = zeros(segment.size, dtype=uint16) - 1

        def loop(seg, dist=1):
            i_slice, links = segments_connexion[seg]
            d = distance[i_slice.start]

            if dist < d and dist <= order:
                distance[i_slice] = dist
                for _seg in links:
                    loop(_seg, dist + 1)

        for indice in i_obs:
            loop(segment[indice], 0)

        return self.extract_with_mask(distance <= order)

    # keep old names, for backward compatibility
    relative = relatives

    def close_network(self, other, nb_obs_min=10, **kwargs):
        """
        Get close network from another atlas.

        :param self other: Atlas to compare
        :param int nb_obs_min: Minimal number of overlap for one trajectory
        :param dict kwargs: keyword arguments for match function
        :return: return other atlas reduced to common tracks with self

        .. warning::
            It could be a costly operation for huge dataset
        """
        p0, p1 = self.period
        indexs = list()
        for i_self, i_other, t0, t1 in self.align_on(other, bins=range(p0, p1 + 2)):
            i, j, s = self.match(other, i_self=i_self, i_other=i_other, **kwargs)
            indexs.append(other.re_reference_index(j, i_other))
        indexs = concatenate(indexs)
        tr, nb = unique(other.track[indexs], return_counts=True)
        m = zeros(other.track.shape, dtype=bool)
        for i in tr[nb >= nb_obs_min]:
            m[other.network_slice(i)] = True
        return other.extract_with_mask(m)

    def normalize_longitude(self):
        """Normalize all longitude

        Normalize longitude field and in the same range :
        - longitude_max
        - contour_lon_e (how to do if in raw)
        - contour_lon_s (how to do if in raw)
        """
        i_start, i_stop, _ = self.index_network
        lon0 = (self.lon[i_start] - 180).repeat(i_stop - i_start)
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

    def numbering_segment(self, start=0):
        """
        New numbering of segment
        """
        for i, _, _ in self.iter_on("track"):
            new_numbering(self.segment[i], start)

    def numbering_network(self, start=1):
        """
        New numbering of network
        """
        new_numbering(self.track, start)

    def only_one_network(self):
        """
        Raise a warning or error?
        if there are more than one network
        """
        _, i_start, _ = self.index_network
        if len(i_start) > 1:
            raise Exception("Several networks")

    def position_filter(self, median_half_window, loess_half_window):
        self.median_filter(median_half_window, "time", "lon").loess_filter(
            loess_half_window, "time", "lon"
        )
        self.median_filter(median_half_window, "time", "lat").loess_filter(
            loess_half_window, "time", "lat"
        )

    def loess_filter(self, half_window, xfield, yfield, inplace=True):
        result = track_loess_filter(
            half_window, self.obs[xfield], self.obs[yfield], self.segment_track_array
        )
        if inplace:
            self.obs[yfield] = result
            return self
        return result

    def median_filter(self, half_window, xfield, yfield, inplace=True):
        result = track_median_filter(
            half_window, self[xfield], self[yfield], self.segment_track_array
        )
        if inplace:
            self[yfield][:] = result
            return self
        return result

    def display_timeline(
        self,
        ax,
        event=True,
        field=None,
        method=None,
        factor=1,
        colors_mode="roll",
        **kwargs,
    ):
        """
        Plot the timeline of a network.
        Must be called on only one network.

        :param matplotlib.axes.Axes ax: matplotlib axe used to draw
        :param bool event: if True, draw the splitting and merging events
        :param str,array field: yaxis values, if None, segments are used
        :param str method: if None, mean values are used
        :param float factor: to multiply field
        :param str colors_mode:
            color of lines. "roll" means looping through colors,
            "y" means color adapt the y values (for matching color plots)
        :return: plot mappable
        """
        self.only_one_network()
        j = 0
        line_kw = dict(ls="-", marker="+", markersize=6, zorder=1, lw=3,)
        line_kw.update(kwargs)
        mappables = dict(lines=list())

        if event:
            mappables.update(
                self.event_timeline(
                    ax,
                    field=field,
                    method=method,
                    factor=factor,
                    colors_mode=colors_mode,
                )
            )
        for i, b0, b1 in self.iter_on("segment"):
            x = self.time[i]
            if x.shape[0] == 0:
                continue
            if field is None:
                y = b0 * ones(x.shape)
            else:
                if method == "all":
                    y = self[field][i] * factor
                else:
                    y = self[field][i].mean() * ones(x.shape) * factor

            if colors_mode == "roll":
                _color = self.get_color(j)
            elif colors_mode == "y":
                _color = self.get_color(b0 - 1)
            else:
                raise NotImplementedError(f"colors_mode '{colors_mode}' not defined")

            line = ax.plot(x, y, **line_kw, color=_color)[0]
            mappables["lines"].append(line)
            j += 1

        return mappables

    def event_timeline(self, ax, field=None, method=None, factor=1, colors_mode="roll"):
        """Mark events in plot"""
        j = 0
        events = dict(spliting=[], merging=[])

        # TODO : fill mappables dict
        y_seg = dict()
        _time = self.time

        if field is not None and method != "all":
            for i, b0, _ in self.iter_on("segment"):
                y = self[field][i]
                if y.shape[0] != 0:
                    y_seg[b0] = y.mean() * factor
        mappables = dict()
        for i, b0, b1 in self.iter_on("segment"):
            x = _time[i]
            if x.shape[0] == 0:
                continue

            if colors_mode == "roll":
                _color = self.get_color(j)
            elif colors_mode == "y":
                _color = self.get_color(b0 - 1)
            else:
                raise NotImplementedError(f"colors_mode '{colors_mode}' not defined")

            event_kw = dict(color=_color, ls="-", zorder=1)

            i_n, i_p = (
                self.next_obs[i.stop - 1],
                self.previous_obs[i.start],
            )
            if field is None:
                y0 = b0
            else:
                if method == "all":
                    y0 = self[field][i.stop - 1] * factor
                else:
                    y0 = y_seg[b0]
            if i_n != -1:
                seg_next = self.segment[i_n]
                y1 = (
                    seg_next
                    if field is None
                    else (
                        self[field][i_n] * factor
                        if method == "all"
                        else y_seg[seg_next]
                    )
                )
                ax.plot((x[-1], _time[i_n]), (y0, y1), **event_kw)[0]
                events["merging"].append((x[-1], y0))

            if i_p != -1:
                seg_previous = self.segment[i_p]
                if field is not None and method == "all":
                    y0 = self[field][i.start] * factor
                y1 = (
                    seg_previous
                    if field is None
                    else (
                        self[field][i_p] * factor
                        if method == "all"
                        else y_seg[seg_previous]
                    )
                )
                ax.plot((x[0], _time[i_p]), (y0, y1), **event_kw)[0]
                events["spliting"].append((x[0], y0))

            j += 1

        kwargs = dict(color="k", zorder=-1, linestyle=" ")
        if len(events["spliting"]) > 0:
            X, Y = list(zip(*events["spliting"]))
            ref = ax.plot(
                X, Y, marker="*", markersize=12, label="spliting events", **kwargs
            )[0]
            mappables.setdefault("events", []).append(ref)

        if len(events["merging"]) > 0:
            X, Y = list(zip(*events["merging"]))
            ref = ax.plot(
                X, Y, marker="H", markersize=10, label="merging events", **kwargs
            )[0]
            mappables.setdefault("events", []).append(ref)

        return mappables

    def mean_by_segment(self, y, **kw):
        kw["dtype"] = y.dtype
        return self.map_segment(lambda x: x.mean(), y, **kw)

    def map_segment(self, method, y, same=True, **kw):
        if same:
            out = empty(y.shape, **kw)
        else:
            out = list()
        for i, b0, b1 in self.iter_on(self.segment_track_array):
            res = method(y[i])
            if same:
                out[i] = res
            else:
                if isinstance(i, slice):
                    if i.start == i.stop:
                        continue
                elif len(i) == 0:
                    continue
                out.append(res)
        if not same:
            out = array(out)
        return out

    def map_network(self, method, y, same=True, return_dict=False, **kw):
        """
        Transform data `y` with method `method` for each track.

        :param Callable method: method to apply on each track
        :param np.array y: data where to apply method
        :param bool same: if True, return an array with the same size than y. Else, return a list with the edited tracks
        :param bool return_dict: if None, mean values are used
        :param float kw: to multiply field
        :return: array or dict of result from method for each network
        """

        if same and return_dict:
            raise NotImplementedError(
                "both conditions 'same' and 'return_dict' should no be true"
            )

        if same:
            out = empty(y.shape, **kw)

        elif return_dict:
            out = dict()

        else:
            out = list()

        for i, b0, b1 in self.iter_on(self.track):
            res = method(y[i])
            if same:
                out[i] = res

            elif return_dict:
                out[b0] = res

            else:
                if isinstance(i, slice):
                    if i.start == i.stop:
                        continue
                elif len(i) == 0:
                    continue
                out.append(res)

        if not same and not return_dict:
            out = array(out)
        return out

    def scatter_timeline(
        self,
        ax,
        name,
        factor=1,
        event=True,
        yfield=None,
        yfactor=1,
        method=None,
        **kwargs,
    ):
        """
        Must be called on only one network
        """
        self.only_one_network()
        y = (self.segment if yfield is None else self.parse_varname(yfield)) * yfactor
        if method == "all":
            pass
        else:
            y = self.mean_by_segment(y)
        mappables = dict()
        if event:
            mappables.update(
                self.event_timeline(ax, field=yfield, method=method, factor=yfactor)
            )
        if "c" not in kwargs:
            v = self.parse_varname(name)
            kwargs["c"] = v * factor
        mappables["scatter"] = ax.scatter(self.time, y, **kwargs)
        return mappables

    def event_map(self, ax, **kwargs):
        """Add the merging and splitting events to a map"""
        j = 0
        mappables = dict()
        symbol_kw = dict(markersize=10, color="k",)
        symbol_kw.update(kwargs)
        symbol_kw_split = symbol_kw.copy()
        symbol_kw_split["markersize"] += 4
        for i, b0, b1 in self.iter_on("segment"):
            nb = i.stop - i.start
            if nb == 0:
                continue
            event_kw = dict(color=self.COLORS[j % self.NB_COLORS], ls="-", **kwargs)
            i_n, i_p = (
                self.next_obs[i.stop - 1],
                self.previous_obs[i.start],
            )

            if i_n != -1:
                y0, y1 = self.lat[i.stop - 1], self.lat[i_n]
                x0, x1 = self.lon[i.stop - 1], self.lon[i_n]
                ax.plot((x0, x1), (y0, y1), **event_kw)[0]
                ax.plot(x0, y0, marker="H", **symbol_kw)[0]
            if i_p != -1:
                y0, y1 = self.lat[i.start], self.lat[i_p]
                x0, x1 = self.lon[i.start], self.lon[i_p]
                ax.plot((x0, x1), (y0, y1), **event_kw)[0]
                ax.plot(x0, y0, marker="*", **symbol_kw_split)[0]

            j += 1
        return mappables

    def scatter(
        self, ax, name="time", factor=1, ref=None, edgecolor_cycle=None, **kwargs,
    ):
        """
        This function scatters the path of each network, with the merging and splitting events

        :param matplotlib.axes.Axes ax: matplotlib axe used to draw
        :param str,array,None name:
            variable used to fill the contours, if None all elements have the same color
        :param float,None ref: if defined, ref is used as western boundary
        :param float factor: multiply value by
        :param list edgecolor_cycle: list of colors
        :param dict kwargs: look at :py:meth:`matplotlib.axes.Axes.scatter`
        :return: a dict of scattered mappables
        """
        mappables = dict()
        nb_colors = len(edgecolor_cycle) if edgecolor_cycle else None
        x = self.longitude
        if ref is not None:
            x = (x - ref) % 360 + ref
        kwargs = kwargs.copy()
        if nb_colors:
            edgecolors = list()
            seg_previous = self.segment[0]
            j = 0
            for seg in self.segment:
                if seg != seg_previous:
                    j += 1
                edgecolors.append(edgecolor_cycle[j % nb_colors])
                seg_previous = seg
            mappables["edges"] = ax.scatter(
                x, self.latitude, edgecolor=edgecolors, **kwargs
            )
            kwargs.pop("linewidths", None)
            kwargs["lw"] = 0
        if name is not None and "c" not in kwargs:
            v = self.parse_varname(name)
            kwargs["c"] = v * factor
        mappables["scatter"] = ax.scatter(x, self.latitude, **kwargs)
        return mappables

    def extract_event(self, indices):
        nb = len(indices)
        new = EddiesObservations(
            nb,
            track_extra_variables=self.track_extra_variables,
            track_array_variables=self.track_array_variables,
            array_variables=self.array_variables,
            only_variables=self.only_variables,
            raw_data=self.raw_data,
        )

        for k in new.obs.dtype.names:
            new[k][:] = self[k][indices]
        new.sign_type = self.sign_type
        return new

    @property
    def segment_track_array(self):
        """Return a unique segment id when multiple networks are considered"""
        return build_unique_array(self.segment, self.track)

    def birth_event(self):
        # FIXME how to manage group 0
        indices = list()
        previous_obs = self.previous_obs
        for i, _, _ in self.iter_on(self.segment_track_array):
            nb = i.stop - i.start
            if nb == 0:
                continue
            i_p = previous_obs[i.start]
            if i_p == -1:
                indices.append(i.start)
        return self.extract_event(list(set(indices)))

    def death_event(self):
        # FIXME how to manage group 0
        indices = list()
        next_obs = self.next_obs
        for i, _, _ in self.iter_on(self.segment_track_array):
            nb = i.stop - i.start
            if nb == 0:
                continue
            i_n = next_obs[i.stop - 1]
            if i_n == -1:
                indices.append(i.stop - 1)
        return self.extract_event(list(set(indices)))

    def merging_event(self, triplet=False, only_index=False):
        """Return observation after a merging event.

        If `triplet=True` return the eddy after a merging event, the eddy before the merging event,
        and the eddy stopped due to merging.
        """
        idx_m1 = list()
        if triplet:
            idx_m0_stop = list()
            idx_m0 = list()
        next_obs, previous_obs = self.next_obs, self.previous_obs
        for i, _, _ in self.iter_on(self.segment_track_array):
            nb = i.stop - i.start
            if nb == 0:
                continue
            i_n = next_obs[i.stop - 1]
            if i_n != -1:
                if triplet:
                    idx_m0_stop.append(i.stop - 1)
                    idx_m0.append(previous_obs[i_n])
                idx_m1.append(i_n)

        if triplet:
            if only_index:
                return (idx_m1, idx_m0, idx_m0_stop)
            else:
                return (
                    self.extract_event(idx_m1),
                    self.extract_event(idx_m0),
                    self.extract_event(idx_m0_stop),
                )
        else:
            idx_m1 = list(set(idx_m1))
            if only_index:
                return idx_m1
            else:
                return self.extract_event(idx_m1)

    def spliting_event(self, triplet=False, only_index=False):
        """Return observation before a splitting event.

        If `triplet=True` return the eddy before a splitting event, the eddy after the splitting event,
        and the eddy starting due to splitting.
        """
        idx_s0 = list()
        if triplet:
            idx_s1_start = list()
            idx_s1 = list()
        next_obs, previous_obs = self.next_obs, self.previous_obs
        for i, _, _ in self.iter_on(self.segment_track_array):
            nb = i.stop - i.start
            if nb == 0:
                continue
            i_p = previous_obs[i.start]
            if i_p != -1:
                if triplet:
                    idx_s1_start.append(i.start)
                    idx_s1.append(next_obs[i_p])
                idx_s0.append(i_p)

        if triplet:
            if only_index:
                return (idx_s0, idx_s1, idx_s1_start)
            else:
                return (
                    self.extract_event(list(idx_s0)),
                    self.extract_event(list(idx_s1)),
                    self.extract_event(list(idx_s1_start)),
                )

        else:
            idx_s0 = list(set(idx_s0))
            if only_index:
                return idx_s0
            else:
                return self.extract_event(idx_s0)

    def dissociate_network(self):
        """
        Dissociate networks with no known interaction (spliting/merging)
        """

        tags = self.tag_segment(multi_network=True)
        if self.track[0] == 0:
            tags -= 1

        self.track[:] = tags[self.segment_track_array]

        i_sort = self.obs.argsort(order=("track", "segment", "time"), kind="mergesort")
        # Sort directly obs, with hope to save memory
        self.obs.sort(order=("track", "segment", "time"), kind="mergesort")
        self._index_network = None

        # n & p must be re-indexed
        n, p = self.next_obs, self.previous_obs
        # we add 2 for -1 index return index -1
        nb_obs = len(self)
        translate = -ones(nb_obs + 1, dtype="i4")
        translate[:-1][i_sort] = arange(nb_obs)
        self.next_obs[:] = translate[n]
        self.previous_obs[:] = translate[p]

    def network(self, id_network):
        return self.extract_with_mask(self.network_slice(id_network))

    def networks(self, id_networks):
        m = zeros(self.track.shape, dtype=bool)
        for tr in id_networks:
            m[self.network_slice(tr)] = True
        return self.extract_with_mask(m)

    @classmethod
    def __tag_segment(cls, seg, tag, groups, connexions):
        """
        Will set same temporary ID for each connected segment.

        :param int seg: current ID of segment
        :param ing tag: temporary ID to set for segment and its connexion
        :param array[int] groups: array where tag is stored
        :param dict connexions: gives for one ID of segment all connected segments
        """
        # If segments are already used we stop recursivity
        if groups[seg] != 0:
            return
        # We set tag for this segment
        groups[seg] = tag
        # Get all connexions of this segment
        segs = connexions.get(seg, None)
        if segs is not None:
            for seg in segs:
                # For each connexion we apply same function
                cls.__tag_segment(seg, tag, groups, connexions)

    def tag_segment(self, multi_network=False):
        if multi_network:
            nb = self.segment_track_array[-1] + 1
        else:
            nb = self.segment.max() + 1
        sub_group = zeros(nb, dtype="u4")
        c = self.connexions(multi_network=multi_network)
        j = 1
        # for each available id
        for i in range(nb):
            # Skip if already set
            if sub_group[i] != 0:
                continue
            # we tag an unset segments and explore all connexions
            self.__tag_segment(i, j, sub_group, c)
            j += 1
        return sub_group

    def fully_connected(self):
        self.only_one_network()
        return self.tag_segment().shape[0] == 1

    def remove_trash(self):
        """
        Remove the lonely eddies (only 1 obs in segment, associated segment number is 0)
        """
        return self.extract_with_mask(self.track != 0)

    def plot(self, ax, ref=None, color_cycle=None, **kwargs):
        """
        This function draws the path of each trajectory

        :param matplotlib.axes.Axes ax: ax to draw
        :param float,int ref: if defined, all coordinates are wrapped with ref as western boundary
        :param dict kwargs: keyword arguments for Axes.plot
        :return: a list of matplotlib mappables
        """
        nb_colors = 0
        if color_cycle is not None:
            kwargs = kwargs.copy()
            nb_colors = len(color_cycle)
        mappables = list()
        if "label" in kwargs:
            kwargs["label"] = self.format_label(kwargs["label"])
        j = 0
        for i, _, _ in self.iter_on(self.segment_track_array):
            nb = i.stop - i.start
            if nb == 0:
                continue
            if nb_colors:
                kwargs["color"] = color_cycle[j % nb_colors]
            x, y = self.lon[i], self.lat[i]
            if ref is not None:
                x, y = wrap_longitude(x, y, ref, cut=True)
            mappables.append(ax.plot(x, y, **kwargs)[0])
            j += 1
        return mappables

    def remove_dead_end(self, nobs=3, ndays=0, recursive=0, mask=None):
        """
        Remove short segments that don't connect several segments

        :param int nobs: Minimal number of observation to keep a segment
        :param int ndays: Minimal number of days to keep a segment
        :param int recursive: Run method N times more
        :param int mask: if one or more observation of the segment are selected by mask, the segment is kept

        .. warning::
            It will remove short segment that splits from then merges with the same segment
        """
        segments_keep = list()
        connexions = self.connexions(multi_network=True)
        t = self.time
        for i, b0, _ in self.iter_on(self.segment_track_array):
            if mask and mask[i].any():
                segments_keep.append(b0)
                continue
            nb = i.stop - i.start
            dt = t[i.stop - 1] - t[i.start]
            if (nb < nobs or dt < ndays) and len(connexions.get(b0, tuple())) < 2:
                continue
            segments_keep.append(b0)
        if recursive > 0:
            return self.extract_segment(segments_keep, absolute=True).remove_dead_end(
                nobs, ndays, recursive - 1
            )
        return self.extract_segment(segments_keep, absolute=True)

    def extract_segment(self, segments, absolute=False):
        mask = ones(self.shape, dtype="bool")
        segments = array(segments)
        values = self.segment_track_array if absolute else "segment"
        keep = ones(values.max() + 1, dtype="bool")
        v = unique(values)
        keep[v] = in1d(v, segments)
        for i, b0, b1 in self.iter_on(values):
            if not keep[b0]:
                mask[i] = False
        return self.extract_with_mask(mask)

    def get_mask_with_period(self, period):
        """
        obtain mask within a time period

        :param (int,int) period: two dates to define the period, must be specified from 1/1/1950
        :return: mask where period is defined
        :rtype: np.array(bool)

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
        return mask

    def extract_with_period(self, period):
        """
        Extract within a time period

        :param (int,int) period: two dates to define the period, must be specified from 1/1/1950
        :return: Return all eddy trajectories in period
        :rtype: NetworkObservations

        .. minigallery:: py_eddy_tracker.NetworkObservations.extract_with_period
        """

        return self.extract_with_mask(self.get_mask_with_period(period))

    def extract_light_with_mask(self, mask):
        """extract data with mask, but only with variables used for coherence, aka self.array_variables

        :param mask: mask used to extract
        :type mask: np.array(bool)
        :return: new EddiesObservation with data wanted
        :rtype: self
        """

        if isinstance(mask, slice):
            nb_obs = mask.stop - mask.start
        else:
            nb_obs = mask.sum()

        # only time & contour_lon/lat_e/s
        variables = ["time"] + self.array_variables
        new = self.__class__(
            size=nb_obs,
            track_extra_variables=[],
            track_array_variables=self.track_array_variables,
            array_variables=self.array_variables,
            only_variables=variables,
            raw_data=self.raw_data,
        )
        new.sign_type = self.sign_type
        if nb_obs == 0:
            logger.warning("Empty dataset will be created")
        else:
            logger.info(
                f"{nb_obs} observations will be extracted ({nb_obs / self.shape[0]:.3%})"
            )

        for field in variables:
            logger.debug("Copy of field %s ...", field)
            new.obs[field] = self.obs[field][mask]
        return new

    def extract_with_mask(self, mask):
        """
        Extract a subset of observations.

        :param array(bool) mask: mask to select observations
        :return: same object with selected observations
        :rtype: self
        """
        if isinstance(mask, slice):
            nb_obs = mask.stop - mask.start
        else:
            nb_obs = mask.sum()
        new = self.__class__.new_like(self, nb_obs)
        new.sign_type = self.sign_type
        if nb_obs == 0:
            logger.warning("Empty dataset will be created")
        else:
            logger.debug(
                f"{nb_obs} observations will be extracted ({nb_obs / self.shape[0]:.3%})"
            )
            for field in self.obs.dtype.descr:
                if field in ("next_obs", "previous_obs"):
                    continue
                logger.debug("Copy of field %s ...", field)
                var = field[0]
                new.obs[var] = self.obs[var][mask]
            # n & p must be re-index
            n, p = self.next_obs[mask], self.previous_obs[mask]
            # we add 2 for -1 index return index -1
            translate = -ones(len(self) + 1, dtype="i4")
            translate[:-1][mask] = arange(nb_obs)
            new.next_obs[:] = translate[n]
            new.previous_obs[:] = translate[p]
        return new

    def analysis_coherence(
        self,
        date_function,
        uv_params,
        advection_mode="both",
        dt_advect=14,
        step_mesh=1.0 / 50,
        output_name=None,
        dissociate_network=False,
        correct_close_events=0,
        remove_dead_end=0,
    ):

        """Global function to analyse segments coherence, with network preprocessing"""

        if dissociate_network:
            self.dissociate_network()

        if correct_close_events > 0:
            self.correct_close_events(nb_days_max=correct_close_events)

        if remove_dead_end > 0:
            network_clean = self.remove_dead_end(nobs=0, ndays=remove_dead_end)
        else:
            network_clean = self

        res = network_clean.segment_coherence(
            date_function=date_function,
            uv_params=uv_params,
            advection_mode=advection_mode,
            output_name=output_name,
            dt_advect=dt_advect,
            step_mesh=step_mesh,
        )

        return network_clean, res

    def segment_coherence_backward(
        self, date_function, uv_params, n_days=14, step_mesh=1.0 / 50, output_name=None,
    ):

        """
        Percentage of particules and their targets after backward advection from a specific eddy.

        :param callable date_function: python function, takes as param `int` (julian day) and return
            data filename associated to the date (see note)
        :param dict uv_params: dict of parameters used by
            :py:meth:`~py_eddy_tracker.dataset.grid.GridCollection.from_netcdf_list`
        :param int n_days: days for advection
        :param float step_mesh: step for particule mesh in degrees
        :return: observations matchs, and percents

        .. note:: the param `date_function` should be something like :

            .. code-block:: python

                def date2file(julian_day):
                    date = datetime.timedelta(days=julian_day) + datetime.datetime(
                        1950, 1, 1
                    )

                    return f"/tmp/dt_global_{date.strftime('%Y%m%d')}.nc"
        """

        itb_final = -ones((self.obs.size, 2), dtype="i4")
        ptb_final = zeros((self.obs.size, 2), dtype="i1")

        t_start, t_end = self.period

        dates = arange(t_start, t_start + n_days + 1)
        first_files = [date_function(x) for x in dates]

        c = GridCollection.from_netcdf_list(first_files, dates, **uv_params)
        first = True
        range_start = t_start + n_days
        range_end = t_end + 1

        for _t in range(t_start + n_days, t_end + 1):
            _timestamp = time.time()
            t_shift = _t

            # skip first shift, because already included
            if first:
                first = False
            else:
                # add next date to GridCollection and delete last date
                c.shift_files(t_shift, date_function(int(t_shift)), **uv_params)
            particle_candidate(
                c, self, step_mesh, _t, itb_final, ptb_final, n_days=-n_days
            )
            logger.info((
                f"coherence {_t} / {range_end-1} ({(_t - range_start) / (range_end - range_start-1):.1%})"
                f" : {time.time()-_timestamp:5.2f}s"
            ))

        return itb_final, ptb_final

    def segment_coherence_forward(
        self, date_function, uv_params, n_days=14, step_mesh=1.0 / 50,
    ):

        """
        Percentage of particules and their targets after forward advection from a specific eddy.

        :param callable date_function: python function, takes as param `int` (julian day) and return
            data filename associated to the date (see note)
        :param dict uv_params: dict of parameters used by
            :py:meth:`~py_eddy_tracker.dataset.grid.GridCollection.from_netcdf_list`
        :param int n_days: days for advection
        :param float step_mesh: step for particule mesh in degrees
        :return: observations matchs, and percents

        .. note:: the param `date_function` should be something like :

            .. code-block:: python

                def date2file(julian_day):
                    date = datetime.timedelta(days=julian_day) + datetime.datetime(
                        1950, 1, 1
                    )

                    return f"/tmp/dt_global_{date.strftime('%Y%m%d')}.nc"
        """

        itf_final = -ones((self.obs.size, 2), dtype="i4")
        ptf_final = zeros((self.obs.size, 2), dtype="i1")

        t_start, t_end = self.period
        # if begin is not None and begin > t_start:
        #     t_start = begin
        # if end is not None and end < t_end:
        #     t_end = end

        dates = arange(t_start, t_start + n_days + 1)
        first_files = [date_function(x) for x in dates]

        c = GridCollection.from_netcdf_list(first_files, dates, **uv_params)
        first = True
        range_start = t_start
        range_end = t_end - n_days + 1

        for _t in range(range_start, range_end):
            _timestamp = time.time()
            t_shift = _t + n_days

            # skip first shift, because already included
            if first:
                first = False
            else:
                # add next date to GridCollection and delete last date
                c.shift_files(t_shift, date_function(int(t_shift)), **uv_params)
            particle_candidate(
                c, self, step_mesh, _t, itf_final, ptf_final, n_days=n_days
            )
            logger.info((
                f"coherence {_t} / {range_end-1} ({(_t - range_start) / (range_end - range_start-1):.1%})"
                f" : {time.time()-_timestamp:5.2f}s"
            ))
        return itf_final, ptf_final


class Network:
    __slots__ = (
        "window",
        "filenames",
        "nb_input",
        "buffer",
        "memory",
    )

    NOGROUP = TrackEddiesObservations.NOGROUP

    def __init__(self, input_regex, window=5, intern=False, memory=False):
        """
        Class to group observations by network
        """
        self.window = window
        self.buffer = Buffer(window, intern, memory)
        self.memory = memory

        self.filenames = glob(input_regex)
        self.filenames.sort()
        self.nb_input = len(self.filenames)

    @classmethod
    def from_eddiesobservations(cls, observations, *args, **kwargs):
        new = cls("", *args, **kwargs)
        new.filenames = observations
        new.nb_input = len(new.filenames)
        return new

    def get_group_array(self, results, nb_obs):
        """With a loop on all pair of index, we will label each obs with a group
        number
        """
        nb_obs = array(nb_obs, dtype="u4")
        day_start = nb_obs.cumsum() - nb_obs
        gr = empty(nb_obs.sum(), dtype="u4")
        gr[:] = self.NOGROUP

        merge_id = list()
        id_free = 1
        for i, j, ii, ij in results:
            gr_i = gr[slice(day_start[i], day_start[i] + nb_obs[i])]
            gr_j = gr[slice(day_start[j], day_start[j] + nb_obs[j])]
            # obs with no groups
            m = (gr_i[ii] == self.NOGROUP) * (gr_j[ij] == self.NOGROUP)
            nb_new = m.sum()
            gr_i[ii[m]] = gr_j[ij[m]] = arange(id_free, id_free + nb_new)
            id_free += nb_new
            # associate obs with no group with obs with group
            m = (gr_i[ii] != self.NOGROUP) * (gr_j[ij] == self.NOGROUP)
            gr_j[ij[m]] = gr_i[ii[m]]
            m = (gr_i[ii] == self.NOGROUP) * (gr_j[ij] != self.NOGROUP)
            gr_i[ii[m]] = gr_j[ij[m]]
            # case where 2 obs have a different group
            m = gr_i[ii] != gr_j[ij]
            if m.any():
                # Merge of group, ref over etu
                for i_, j_ in zip(ii[m], ij[m]):
                    g0, g1 = gr_i[i_], gr_j[j_]
                    if g0 > g1:
                        g0, g1 = g1, g0
                    merge_id.append((g0, g1))
        gr_transfer = self.group_translator(id_free, set(merge_id))
        return gr_transfer[gr]

    @staticmethod
    def group_translator(nb, duos):
        """
        Create a translator with all duos

        :param int nb: size of translator
        :param set((int, int)) duos: set of all groups that must be joined

        :Example:

        >>> NetworkObservations.group_translator(5, ((0, 1), (0, 2), (1, 3)))
        [3, 3, 3, 3, 5]
        """
        translate = arange(nb, dtype="u4")
        for i, j in sorted(duos):
            gr_i, gr_j = translate[i], translate[j]
            if gr_i != gr_j:
                apply_replace(translate, gr_i, gr_j)
        return translate

    def group_observations(self, **kwargs):
        results, nb_obs = list(), list()
        # To display print only in INFO
        display_iteration = logger.getEffectiveLevel() == logging.INFO
        for i, filename in enumerate(self.filenames):
            if display_iteration:
                print(f"{filename} compared to {self.window} next", end="\r")
            # Load observations with function to buffer observations
            xi, yi = self.buffer.load_contour(filename)
            # Append number of observations by filename
            nb_obs.append(xi.shape[0])
            for j in range(i + 1, min(self.window + i + 1, self.nb_input)):
                xj, yj = self.buffer.load_contour(self.filenames[j])
                ii, ij = bbox_intersection(xi, yi, xj, yj)
                m = vertice_overlap(xi[ii], yi[ii], xj[ij], yj[ij], **kwargs) > 0.2
                results.append((i, j, ii[m], ij[m]))
        if display_iteration:
            print()

        gr = self.get_group_array(results, nb_obs)
        nb_alone, nb_obs, nb_gr = (gr == self.NOGROUP).sum(), len(gr), len(unique(gr))
        logger.info(
            f"{nb_alone} alone / {nb_obs} obs, {nb_gr} groups, "
            f"{nb_alone *100./nb_obs:.2f} % alone, {(nb_obs - nb_alone) / (nb_gr - 1):.1f} obs/group"
        )
        return gr

    def build_dataset(self, group, raw_data=True):
        nb_obs = group.shape[0]
        model = TrackEddiesObservations.load_file(self.filenames[-1], raw_data=raw_data)
        eddies = TrackEddiesObservations.new_like(model, nb_obs)
        eddies.sign_type = model.sign_type
        # Get new index to re-order observations by groups
        new_i = get_next_index(group)
        display_iteration = logger.getEffectiveLevel() == logging.INFO
        elements = eddies.elements

        i = 0
        for filename in self.filenames:
            if display_iteration:
                print(f"Load {filename} to copy", end="\r")
            if self.memory:
                # Only if netcdf
                with open(filename, "rb") as h:
                    e = TrackEddiesObservations.load_file(h, raw_data=raw_data)
            else:
                e = TrackEddiesObservations.load_file(filename, raw_data=raw_data)
            stop = i + len(e)
            sl = slice(i, stop)
            for element in elements:
                eddies[element][new_i[sl]] = e[element]
            i = stop
        if display_iteration:
            print()
        eddies.track[new_i] = group
        return eddies


@njit(cache=True)
def get_next_index(gr):
    """Return for each obs index the new position to join all groups"""
    nb_obs_gr = bincount(gr)
    i_gr = nb_obs_gr.cumsum() - nb_obs_gr
    new_index = empty(gr.shape, dtype=uint32)
    for i, g in enumerate(gr):
        new_index[i] = i_gr[g]
        i_gr[g] += 1
    return new_index


@njit(cache=True)
def apply_replace(x, x0, x1):
    nb = x.shape[0]
    for i in range(nb):
        if x[i] == x0:
            x[i] = x1


@njit(cache=True)
def build_unique_array(id1, id2):
    """Give a unique id for each (id1, id2) with id1 and id2 increasing monotonically"""
    k = 0
    new_id = empty(id1.shape, dtype=id1.dtype)
    id1_previous = id1[0]
    id2_previous = id2[0]
    for i in range(id1.shape[0]):
        id1_, id2_ = id1[i], id2[i]
        if id1_ != id1_previous or id2_ != id2_previous:
            k += 1
        new_id[i] = k
        id1_previous, id2_previous = id1_, id2_
    return new_id


@njit(cache=True)
def new_numbering(segs, start=0):
    nb = len(segs)
    s0 = segs[0]
    j = start
    for i in range(nb):
        if segs[i] != s0:
            s0 = segs[i]
            j += 1
        segs[i] = j


@njit(cache=True)
def ptp(values):
    return values.max() - values.min()
