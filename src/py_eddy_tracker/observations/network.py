# -*- coding: utf-8 -*-
"""
Class to create network of observations
"""
import logging
from glob import glob

from numba import njit
from numpy import arange, array, bincount, empty, ones, uint32, unique

from ..generic import build_index, wrap_longitude
from ..poly import bbox_intersection, vertice_overlap
from .observation import EddiesObservations
from .tracking import TrackEddiesObservations, track_median_filter

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

    __slots__ = tuple()

    NOGROUP = 0

    @property
    def elements(self):
        elements = super().elements
        elements.extend(["track", "segment", "next_obs", "previous_obs"])
        return list(set(elements))

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

    def only_one_network(self):
        """
        Raise a warning or error?
        if there are more than one network
        """
        # TODO
        pass

    def median_filter(self, half_window, xfield, yfield, inplace=True):
        # FIXME: segments is not enough with several network
        result = track_median_filter(
            half_window, self[xfield], self[yfield], self.segment
        )
        if inplace:
            self[yfield][:] = result
            return self
        return result

    def display_timeline(self, ax, event=True, field=None, method=None):
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
        mappables = dict(lines=list())
        if event:
            mappables.update(self.event_timeline(ax, field=field, method=method))
        for i, b0, b1 in self.iter_on("segment"):
            x = self.time[i]
            if x.shape[0] == 0:
                continue
            if field is None:
                y = b0 * ones(x.shape)
            else:
                if method == "all":
                    y = self[field][i]
                else:
                    y = self[field][i].mean() * ones(x.shape)
            line = ax.plot(x, y, **line_kw, color=self.COLORS[j % self.NB_COLORS])[0]
            mappables["lines"].append(line)
            j += 1

        return mappables

    def event_timeline(self, ax, field=None, method=None):
        j = 0
        # TODO : fill mappables dict
        y_seg = dict()
        if field is not None and method != "all":
            for i, b0, _ in self.iter_on("segment"):
                y = self[field][i]
                if y.shape[0] != 0:
                    y_seg[b0] = y.mean()
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
                    y0 = self[field][i.stop - 1]
                else:
                    y0 = y_seg[b0]
            if i_n != -1:
                seg_next = self.segment[i_n]
                y1 = (
                    seg_next
                    if field is None
                    else (self[field][i_n] if method == "all" else y_seg[seg_next])
                )
                ax.plot((x[-1], self.time[i_n]), (y0, y1), **event_kw)[0]
                ax.plot(x[-1], y0, color="k", marker=">", markersize=10, zorder=-1)[0]
            if i_p != -1:
                seg_previous = self.segment[i_p]
                if field is not None and method == "all":
                    y0 = self[field][i.start]
                y1 = (
                    seg_previous
                    if field is None
                    else (self[field][i_p] if method == "all" else y_seg[seg_previous])
                )
                ax.plot((x[0], self.time[i_p]), (y0, y1), **event_kw)[0]
                ax.plot(x[0], y0, color="k", marker="*", markersize=12, zorder=-1)[0]
            j += 1
        return mappables

    def scatter_timeline(self, ax, name, factor=1, event=True, **kwargs):
        """
        Must be call on only one network
        """
        self.only_one_network()
        mappables = dict()
        if event:
            mappables.update(self.event_timeline(ax))
        if "c" not in kwargs:
            v = self.parse_varname(name)
            kwargs["c"] = v * factor
        mappables["scatter"] = ax.scatter(self.time, self.segment, **kwargs)
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

    def segment_track_array(self):
        return build_unique_array(self.segment, self.track)

    def birth_event(self):
        # FIXME how to manage group 0
        indices = list()
        for i, b0, b1 in self.iter_on(self.segment_track_array()):
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
        for i, b0, b1 in self.iter_on(self.segment_track_array()):
            nb = i.stop - i.start
            if nb == 0:
                continue
            i_n = self.next_obs[i.stop - 1]
            if i_n == -1:
                indices.append(i.stop - 1)
        return self.extract_event(list(set(indices)))

    def merging_event(self):
        indices = list()
        for i, b0, b1 in self.iter_on(self.segment_track_array()):
            nb = i.stop - i.start
            if nb == 0:
                continue
            i_n = self.next_obs[i.stop - 1]
            if i_n != -1:
                indices.append(i.stop - 1)
        return self.extract_event(list(set(indices)))

    def spliting_event(self):
        indices = list()
        for i, b0, b1 in self.iter_on(self.segment_track_array()):
            nb = i.stop - i.start
            if nb == 0:
                continue
            i_p = self.previous_obs[i.start]
            if i_p != -1:
                indices.append(i.start)
        return self.extract_event(list(set(indices)))

    def fully_connected(self):
        self.only_one_network()
        # TODO

    def plot(self, ax, ref=None, **kwargs):
        """
        This function will draw path of each trajectory

        :param matplotlib.axes.Axes ax: ax to draw
        :param float,int ref: if defined, all coordinates will be wrapped with ref like west boundary
        :param dict kwargs: keyword arguments for Axes.plot
        :return: a list of matplotlib mappables
        """
        mappables = list()
        if "label" in kwargs:
            kwargs["label"] = self.format_label(kwargs["label"])
        for i, b0, b1 in self.iter_on("segment"):
            nb = i.stop - i.start
            if nb == 0:
                continue
            x, y = self.lon[i], self.lat[i]
            if ref is not None:
                x, y = wrap_longitude(x, y, ref, cut=True)
            mappables.append(ax.plot(x, y, **kwargs)[0])
        return mappables

    def remove_dead_branch(self, nobs=3):
        """"""
        # TODO: bug when spliting
        self.only_one_network()

        segments_keep = list()
        interaction_segments = dict()
        segments_connexion = dict()
        for i, b0, b1 in self.iter_on("segment"):
            nb = i.stop - i.start
            i_p, i_n = self.previous_obs[i.start], self.next_obs[i.stop - 1]
            seg = self.segment[i.start]
            # segment of interaction
            p_seg, n_seg = self.segment[i_p], self.segment[i_n]
            if nb >= nobs:
                segments_keep.append(seg)
            else:
                interaction_segments[seg] = (
                    p_seg if i_p != -1 else -1,
                    n_seg if i_n != -1 else -1,
                )
            # Where segment are called
            if i_p != -1:
                if p_seg not in segments_connexion:
                    segments_connexion[p_seg] = list()
                segments_connexion[p_seg].append(seg)
            if i_n != -1:
                if n_seg not in segments_connexion:
                    segments_connexion[n_seg] = list()
                segments_connexion[n_seg].append(seg)
        print(interaction_segments)
        print(segments_connexion)
        print(segments_keep)
        return self.extract_segment(tuple(segments_keep))

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
        nb_obs = mask.sum()
        new = self.__class__.new_like(self, nb_obs)
        new.sign_type = self.sign_type
        if nb_obs == 0:
            logger.warning("Empty dataset will be created")
        else:
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
