# -*- coding: utf-8 -*-
"""
Class to create network of observations
"""
import logging
from glob import glob

from numba import njit
from numpy import arange, array, bincount, empty, ones, uint32, unique

from ..generic import build_index
from ..poly import bbox_intersection, vertice_overlap
from .observation import EddiesObservations
from .tracking import TrackEddiesObservations

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

    def display_timeline(self, ax, event=True):
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
            mappables.update(self.event_timeline(ax))
        for i, b0, b1 in self.iter_on("segment"):
            x = self.time[i]
            if x.shape[0] == 0:
                continue
            y = b0 * ones(x.shape)
            line = ax.plot(x, y, **line_kw, color=self.COLORS[j % self.NB_COLORS])[0]
            mappables["lines"].append(line)
            j += 1

        return mappables

    def event_timeline(self, ax):
        j = 0
        # TODO : fill mappables dict
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
            if i_n != -1:
                ax.plot((x[-1], self.time[i_n]), (b0, self.segment[i_n]), **event_kw)
                ax.plot(x[-1], b0, color="k", marker=">", markersize=10, zorder=-1)
            if i_p != -1:
                ax.plot((x[0], self.time[i_p]), (b0, self.segment[i_p]), **event_kw)
                ax.plot(x[0], b0, color="k", marker="*", markersize=12, zorder=-1)
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
        pass

    def merging_event(self):
        pass

    def spliting_event(self):
        pass

    def fully_connected(self):
        self.only_one_network()

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
        logger.info(
            f"{(gr == self.NOGROUP).sum()} alone / {len(gr)} obs, {len(unique(gr))} groups"
        )
        return gr

    def build_dataset(self, group):
        nb_obs = group.shape[0]
        model = EddiesObservations.load_file(self.filenames[-1], raw_data=True)
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
                    e = EddiesObservations.load_file(h, raw_data=True)
            else:
                e = EddiesObservations.load_file(filename, raw_data=True)
            stop = i + len(e)
            sl = slice(i, stop)
            for element in elements:
                eddies[element][new_i[sl]] = e[element]
            i = stop
        if display_iteration:
            print()
        eddies = eddies.add_fields(("track",))
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
