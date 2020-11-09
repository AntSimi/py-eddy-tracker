# -*- coding: utf-8 -*-
"""
Class to create network of observations
"""
import logging
from glob import glob

from numba import njit
from numpy import arange, array, bincount, empty, uint32, unique

from ..poly import bbox_intersection, vertice_overlap
from .observation import EddiesObservations
from .tracking import TrackEddiesObservations

logger = logging.getLogger("pet")


class Network:
    __slots__ = ("window", "filenames", "contour_name", "nb_input", "xname", "yname")
    # To be used like a buffer
    DATA = dict()
    FLIST = list()
    NOGROUP = TrackEddiesObservations.NOGROUP

    def __init__(self, input_regex, window=5, intern=False):
        self.window = window
        self.contour_name = EddiesObservations.intern(intern, public_label=True)
        self.xname, self.yname = EddiesObservations.intern(intern)
        self.filenames = glob(input_regex)
        self.filenames.sort()
        self.nb_input = len(self.filenames)

    def load_contour(self, filename):
        if filename not in self.DATA:
            if len(self.FLIST) > self.window:
                self.DATA.pop(self.FLIST.pop(0))
            e = EddiesObservations.load_file(filename, include_vars=self.contour_name)
            self.DATA[filename] = e[self.xname], e[self.yname]
        return self.DATA[filename]

    def get_group_array(self, results, nb_obs):
        """With a loop on all pair of index, we will label each obs with a group
        number
        """
        nb_obs = array(nb_obs)
        day_start = nb_obs.cumsum() - nb_obs
        gr = empty(nb_obs.sum(), dtype="u4")
        gr[:] = self.NOGROUP

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
                    gr_i_, gr_j_ = gr_i[i_], gr_j[j_]
                    gr[gr == gr_i_] = gr_j_
        return gr

    def group_observations(self, **kwargs):
        results, nb_obs = list(), list()
        # To display print only in INFO
        display_iteration = logger.getEffectiveLevel() == logging.INFO
        for i, filename in enumerate(self.filenames):
            if display_iteration:
                print(f"{filename} compared to {self.window} next", end="\r")
            # Load observations with function to buffered observations
            xi, yi = self.load_contour(filename)
            # Append number of observations by filename
            nb_obs.append(xi.shape[0])
            for j in range(i + 1, min(self.window + i + 1, self.nb_input)):
                xj, yj = self.load_contour(self.filenames[j])
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
