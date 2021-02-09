# -*- coding: utf-8 -*-
"""
Class to create network of observations
"""
import logging
from glob import glob

from numba import njit
from numpy import arange, array, bincount, empty, ones, uint32, unique, zeros

from ..generic import build_index, wrap_longitude
from ..poly import bbox_intersection, vertice_overlap
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


class NetworkObservations(EddiesObservations):

    __slots__ = ("_index_network",)

    NOGROUP = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index_network = None

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

        :param int nb_day_min: Minimal number of day which must be covered by one network, if negative -> not used
        :param int nb_day_max: Maximal number of day which must be covered by one network, if negative -> not used
        """
        if nb_day_max < 0:
            nb_day_max = 1000000000000
        mask = zeros(self.shape, dtype="bool")
        for i, b0, b1 in self.iter_on(self.segment_track_array):
            nb = i.stop - i.start
            if nb == 0:
                continue
            t = self.time[i]
            dt = t.max() - t.min()
            if nb_day_min <= dt <= nb_day_max:
                mask[i] = True
        return self.extract_with_mask(mask)

    @classmethod
    def from_split_network(cls, group_dataset, indexs, **kwargs):
        """
        Build a NetworkObservations object with Group dataset and indexs

        :param TrackEddiesObservations group_dataset: Group dataset
        :param indexs: result from split_network
        return NetworkObservations
        """
        index_order = indexs.argsort(order=("group", "track", "time"))
        network = cls.new_like(group_dataset, len(group_dataset), **kwargs)
        network.sign_type = group_dataset.sign_type
        for field in group_dataset.elements:
            if field not in network.elements:
                continue
            network[field][:] = group_dataset[field][index_order]
        network.segment[:] = indexs["track"][index_order]
        # n & p must be re-index
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

    def obs_relative_order(self, i_obs):
        self.only_one_network()
        return self.segment_relative_order(self.segment[i_obs])

    def connexions(self):
        self.only_one_network()
        segments_connexion = dict()

        def add_seg(father, child):
            if father not in segments_connexion:
                segments_connexion[father] = list()
            segments_connexion[father].append(child)

        for i, seg, _ in self.iter_on("segment"):
            if i.start == i.stop:
                continue
            i_p, i_n = self.previous_obs[i.start], self.next_obs[i.stop - 1]
            # segment of interaction
            p_seg, n_seg = self.segment[i_p], self.segment[i_n]
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

    def relative(self, i_obs, order=2, direct=True, only_past=False, only_future=False):
        d = self.segment_relative_order(self.segment[i_obs])
        m = (d <= order) * (d != -1)
        return self.extract_with_mask(m)

    def numbering_segment(self):
        """
        New numbering of segment
        """
        for i, _, _ in self.iter_on("track"):
            new_numbering(self.segment[i])

    def only_one_network(self):
        """
        Raise a warning or error?
        if there are more than one network
        """
        _, i_start, _ = self.index_network
        if len(i_start) > 1:
            raise Exception("Several network")

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
        self, ax, event=True, field=None, method=None, factor=1, **kwargs
    ):
        """
        Must be call on only one network
        """
        self.only_one_network()
        j = 0
        line_kw = dict(
            ls="-",
            marker=".",
            markersize=6,
            zorder=1,
            lw=3,
        )
        line_kw.update(kwargs)
        mappables = dict(lines=list())
        if event:
            mappables.update(
                self.event_timeline(ax, field=field, method=method, factor=factor)
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
            line = ax.plot(x, y, **line_kw, color=self.COLORS[j % self.NB_COLORS])[0]
            mappables["lines"].append(line)
            j += 1

        return mappables

    def event_timeline(self, ax, field=None, method=None, factor=1):
        j = 0
        # TODO : fill mappables dict
        y_seg = dict()
        if field is not None and method != "all":
            for i, b0, _ in self.iter_on("segment"):
                y = self[field][i]
                if y.shape[0] != 0:
                    y_seg[b0] = y.mean() * factor
        mappables = dict()
        for i, b0, b1 in self.iter_on("segment"):
            x = self.time[i]
            if x.shape[0] == 0:
                continue
            event_kw = dict(color=self.COLORS[j % self.NB_COLORS], ls="-", zorder=1)
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
                ax.plot((x[-1], self.time[i_n]), (y0, y1), **event_kw)[0]
                ax.plot(x[-1], y0, color="k", marker="H", markersize=10, zorder=-1)[0]
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
                ax.plot((x[0], self.time[i_p]), (y0, y1), **event_kw)[0]
                ax.plot(x[0], y0, color="k", marker="*", markersize=12, zorder=-1)[0]
            j += 1
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
        Must be call on only one network
        """
        self.only_one_network()
        y = (self.segment if yfield is None else self[yfield]) * yfactor
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
        """Add the merging and splitting events """
        j = 0
        mappables = dict()
        symbol_kw = dict(
            markersize=10,
            color="k",
        )
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
        self,
        ax,
        name="time",
        factor=1,
        ref=None,
        edgecolor_cycle=None,
        **kwargs,
    ):
        """
        This function will scatter the path of each network, with the merging and splitting events

        :param matplotlib.axes.Axes ax: matplotlib axe used to draw
        :param str,array,None name:
            variable used to fill the contour, if None all elements have the same color
        :param float,None ref: if define use like west bound
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

    def insert_virtual(self):
        # TODO
        pass

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
        return build_unique_array(self.segment, self.track)

    def birth_event(self):
        # FIXME how to manage group 0
        indices = list()
        for i, _, _ in self.iter_on(self.segment_track_array):
            nb = i.stop - i.start
            if nb == 0:
                continue
            i_p = self.previous_obs[i.start]
            if i_p == -1:
                indices.append(i.start)
        return self.extract_event(list(set(indices)))

    def death_event(self):
        # FIXME how to manage group 0
        indices = list()
        for i, _, _ in self.iter_on(self.segment_track_array):
            nb = i.stop - i.start
            if nb == 0:
                continue
            i_n = self.next_obs[i.stop - 1]
            if i_n == -1:
                indices.append(i.stop - 1)
        return self.extract_event(list(set(indices)))

    def merging_event(self):
        indices = list()
        for i, _, _ in self.iter_on(self.segment_track_array):
            nb = i.stop - i.start
            if nb == 0:
                continue
            i_n = self.next_obs[i.stop - 1]
            if i_n != -1:
                indices.append(i.stop - 1)
        return self.extract_event(list(set(indices)))

    def spliting_event(self):
        indices = list()
        for i, _, _ in self.iter_on(self.segment_track_array):
            nb = i.stop - i.start
            if nb == 0:
                continue
            i_p = self.previous_obs[i.start]
            if i_p != -1:
                indices.append(i.start)
        return self.extract_event(list(set(indices)))

    def dissociate_network(self):
        """
        Dissociate network with no known interaction (spliting/merging)
        """
        self.only_one_network()
        tags = self.tag_segment()
        # FIXME : Ok if only one network
        self.track[:] = tags[self.segment - 1]

        i_sort = self.obs.argsort(order=("track", "segment", "time"), kind="mergesort")
        # Sort directly obs, with hope to save memory
        self.obs.sort(order=("track", "segment", "time"), kind="mergesort")
        self._index_network = None

        # n & p must be re-index
        n, p = self.next_obs, self.previous_obs
        # we add 2 for -1 index return index -1
        nb_obs = len(self)
        translate = -ones(nb_obs + 1, dtype="i4")
        translate[:-1][i_sort] = arange(nb_obs)
        self.next_obs[:] = translate[n]
        self.previous_obs[:] = translate[p]

    def network(self, id_network):
        return self.extract_with_mask(self.network_slice(id_network))

    @classmethod
    def __tag_segment(cls, seg, tag, groups, connexions):
        if groups[seg] != 0:
            return
        groups[seg] = tag
        segs = connexions.get(seg + 1, None)
        if segs is not None:
            for seg in segs:
                cls.__tag_segment(seg - 1, tag, groups, connexions)

    def tag_segment(self):
        self.only_one_network()
        nb = self.segment.max()
        sub_group = zeros(nb, dtype="u4")
        c = self.connexions()
        j = 1
        for i in range(nb):
            if sub_group[i] != 0:
                continue
            self.__tag_segment(i, j, sub_group, c)
            j += 1
        return sub_group

    def fully_connected(self):
        self.only_one_network()
        return self.tag_segment().shape[0] == 1

    def plot(self, ax, ref=None, color_cycle=None, **kwargs):
        """
        This function will draw path of each trajectory

        :param matplotlib.axes.Axes ax: ax to draw
        :param float,int ref: if defined, all coordinates will be wrapped with ref like west boundary
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
        for i, _, _ in self.iter_on("segment"):
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

    def remove_dead_end(self, nobs=3, recursive=0, mask=None):
        """
        .. warning::
            It will remove short segment which splits than merges with same segment
        """
        self.only_one_network()
        segments_keep = list()
        connexions = self.connexions()
        for i, b0, b1 in self.iter_on("segment"):
            nb = i.stop - i.start
            if mask and mask[i].any():
                segments_keep.append(b0)
                continue
            if nb < nobs and len(connexions.get(b0, tuple())) < 2:
                continue
            segments_keep.append(b0)
        if recursive > 0:
            return self.extract_segment(segments_keep).remove_dead_end(
                nobs, recursive - 1
            )
        return self.extract_segment(segments_keep)

    def extract_segment(self, segments):
        mask = ones(self.shape, dtype="bool")
        for i, b0, b1 in self.iter_on("segment"):
            if b0 not in segments:
                mask[i] = False
        return self.extract_with_mask(mask)

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
            logger.info(
                f"{nb_obs} observations will be extract ({nb_obs * 100. / self.shape[0]}%)"
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
        self.filenames = glob(input_regex)
        self.filenames.sort()
        self.nb_input = len(self.filenames)
        self.memory = memory

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
                    merge_id.append((gr_i[i_], gr_j[j_]))

        gr_transfer = arange(id_free, dtype="u4")
        for i, j in merge_id:
            gr_i, gr_j = gr_transfer[i], gr_transfer[j]
            if gr_i != gr_j:
                apply_replace(gr_transfer, gr_i, gr_j)
        return gr_transfer[gr]

    def group_observations(self, **kwargs):
        results, nb_obs = list(), list()
        # To display print only in INFO
        display_iteration = logger.getEffectiveLevel() == logging.INFO
        for i, filename in enumerate(self.filenames):
            if display_iteration:
                print(f"{filename} compared to {self.window} next", end="\r")
            # Load observations with function to buffered observations
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

    def build_dataset(self, group):
        nb_obs = group.shape[0]
        model = TrackEddiesObservations.load_file(self.filenames[-1], raw_data=True)
        eddies = TrackEddiesObservations.new_like(model, nb_obs)
        eddies.sign_type = model.sign_type
        # Get new index to re-order observation by group
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
                    e = TrackEddiesObservations.load_file(h, raw_data=True)
            else:
                e = TrackEddiesObservations.load_file(filename, raw_data=True)
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
    """Return for each obs index the new position to join all group"""
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
def new_numbering(segs):
    nb = len(segs)
    s0 = segs[0]
    j = 0
    for i in range(nb):
        if segs[i] != s0:
            s0 = segs[i]
            j += 1
        segs[i] = j
