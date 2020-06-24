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

Copyright (c) 2014-2020 by Evan Mason
Email: evanmason@gmail.com
===========================================================================
"""

import logging
from glob import glob
from netCDF4 import Dataset
from Polygon import Polygon
from numba import njit, uint32
from numpy import empty, array, arange, unique, bincount
from py_eddy_tracker.poly import bbox_intersection, create_vertice
from py_eddy_tracker import EddyParser


logger = logging.getLogger("pet")

NOGROUP = 0
# To be used like a buffer
DATA = dict()
FLIST = list()


def load_contour(filename, xname, yname):
    """
    To avoid multiple load of same file we store last 20 result
    """
    if filename not in DATA:
        with Dataset(filename) as h:
            if len(FLIST) >= 20:
                DATA.pop(FLIST.pop(0))
            FLIST.append(filename)
            DATA[filename] = h.variables[xname][:], h.variables[yname][:]
    return DATA[filename]


@njit(cache=True)
def get_wrap_vertice(x0, y0, x1, y1, i):
    x0_, x1_ = x0[i], x1[i]
    if abs(x0_[0] - x1_[0]) > 180:
        ref = x0_[0] - x0.dtype.type(180)
        x1_ = x1[i]
        x1_ = (x1[i] - ref) % 360 + ref
    return create_vertice(x0_, y0[i]), create_vertice(x1_, y1[i])


def common_area(x0, y0, x1, y1, minimal_area=False):
    nb, _ = x0.shape
    cost = empty((nb))
    for i in range(nb):
        # Get wrapped vertice for index i
        v0, v1 = get_wrap_vertice(x0, y0, x1, y1, i)
        p0 = Polygon(v0)
        p1 = Polygon(v1)
        # Area of intersection
        intersection = (p0 & p1).area()
        # we divide intersection with the little one result from 0 to 1
        if minimal_area:
            p0_area = p0.area()
            p1_area = p1.area()
            cost[i] = intersection / min(p0_area, p1_area)
        # we divide intersection with polygon merging result from 0 to 1
        else:
            union = (p0 + p1).area()
            cost[i] = intersection / union
    return cost


def get_group_array(results, nb_obs):
    """With a loop on all pair of index, we will label each obs with a group
    number
    """
    nb_obs = array(nb_obs)
    day_start = nb_obs.cumsum() - nb_obs
    gr = empty(nb_obs.sum(), dtype="u4")
    gr[:] = NOGROUP

    next_id_group = 1
    for i, j, i_ref, i_etu in results:
        sl_ref = slice(day_start[i], day_start[i] + nb_obs[i])
        sl_etu = slice(day_start[j], day_start[j] + nb_obs[j])
        # obs with no groups
        m = (gr[sl_ref][i_ref] == NOGROUP) * (gr[sl_etu][i_etu] == NOGROUP)
        nb_no_groups = m.sum()
        gr[sl_ref][i_ref[m]] = gr[sl_etu][i_etu[m]] = arange(
            next_id_group, next_id_group + nb_no_groups
        )
        next_id_group += nb_no_groups
        # associate obs with no group with obs with group
        m = (gr[sl_ref][i_ref] != NOGROUP) * (gr[sl_etu][i_etu] == NOGROUP)
        gr[sl_etu][i_etu[m]] = gr[sl_ref][i_ref[m]]
        m = (gr[sl_ref][i_ref] == NOGROUP) * (gr[sl_etu][i_etu] != NOGROUP)
        gr[sl_ref][i_ref[m]] = gr[sl_etu][i_etu[m]]
        # case where 2 obs have a different group
        m = gr[sl_ref][i_ref] != gr[sl_etu][i_etu]
        if m.any():
            # Merge of group, ref over etu
            for i_, j_ in zip(i_ref[m], i_etu[m]):
                g_ref, g_etu = gr[sl_ref][i_], gr[sl_etu][j_]
                gr[gr == g_ref] = g_etu
    return gr


def save(filenames, gr, out):
    """Function to saved group output
    """
    new_i = get_next_index(gr)
    nb = gr.shape[0]
    dtype = list()
    with Dataset(out, "w") as h_out:
        with Dataset(filenames[0]) as h_model:
            for name, dim in h_model.dimensions.items():
                h_out.createDimension(name, len(dim) if name != "obs" else nb)
            v = h_out.createVariable(
                "track", "u4", ("obs",), True, chunksizes=(min(250000, nb),)
            )
            d = v[:].copy()
            d[new_i] = gr
            v[:] = d
            for k in h_model.ncattrs():
                h_out.setncattr(k, h_model.getncattr(k))
            for name, v in h_model.variables.items():
                dtype.append(
                    (
                        name,
                        (v.datatype, 50) if "NbSample" in v.dimensions else v.datatype,
                    )
                )
                new_v = h_out.createVariable(
                    name,
                    v.datatype,
                    v.dimensions,
                    True,
                    chunksizes=(min(25000, nb), 50)
                    if "NbSample" in v.dimensions
                    else (min(250000, nb),),
                )
                for k in v.ncattrs():
                    if k in ("min", "max",):
                        continue
                    new_v.setncattr(k, v.getncattr(k))
        data = empty(nb, dtype)
        i = 0
        debug_active = logger.getEffectiveLevel() == logging.DEBUG
        for filename in filenames:
            if debug_active:
                print(f'Load {filename} to copy', end="\r")
            with Dataset(filename) as h_in:
                stop = i + len(h_in.dimensions["obs"])
                sl = slice(i, stop)
                for name, _ in dtype:
                    v = h_in.variables[name]
                    v.set_auto_maskandscale(False)
                    data[name][new_i[sl]] = v[:]
                i = stop
        if debug_active:
            print()
        for name, _ in dtype:
            v = h_out.variables[name]
            v.set_auto_maskandscale(False)
            v[:] = data[name]


@njit(cache=True)
def get_next_index(gr):
    """Return for each obs index the new position to join all group
    """
    nb_obs_gr = bincount(gr)
    i_gr = nb_obs_gr.cumsum() - nb_obs_gr
    new_index = empty(gr.shape, dtype=uint32)
    for i, g in enumerate(gr):
        new_index[i] = i_gr[g]
        i_gr[g] += 1
    return new_index


def build_network():
    parser = EddyParser("Merge eddies")
    parser.add_argument(
        "identification_regex",
        help="Give an expression which will use with glob, currently only netcdf file",
    )
    parser.add_argument("out", help="output file, currently only netcdf file")
    parser.add_argument(
        "--window", "-w", type=int, help="Half time window to search eddy", default=1
    )
    parser.add_argument(
        "--intern",
        action="store_true",
        help="Use intern contour instead of outter contour",
    )
    args = parser.parse_args()
    network(args.identification_regex, args.out, window=args.window, intern=args.intern)


def network(regex, filename_out, window=1, intern=False):
    filenames = glob(regex)
    filenames.sort()
    nb_in = len(filenames)
    if intern:
        coord = "speed_contour_longitude", "speed_contour_latitude"
    else:
        coord = "effective_contour_longitude", "effective_contour_latitude"
    results, nb_obs = list(), list()
    debug_active = logger.getEffectiveLevel() == logging.DEBUG
    for i, filename in enumerate(filenames):
        if debug_active:
            print(f'{filename} compared to {window} next', end="\r")
        xi, yi = load_contour(filename, *coord)
        nb_obs.append(xi.shape[0])
        for j in range(i + 1, min(window + i + 1, nb_in)):
            xj, yj = load_contour(filenames[j], *coord)
            ii, ij = bbox_intersection(xi, yi, xj, yj)
            m = common_area(xi[ii], yi[ii], xj[ij], yj[ij], minimal_area=True) > 0.2
            results.append((i, j, ii[m], ij[m]))
    if debug_active:
        print()

    gr = get_group_array(results, nb_obs)
    logger.info(
        f"{(gr == NOGROUP).sum()} alone / {len(gr)} obs, {len(unique(gr))} groups"
    )
    save(filenames, gr, filename_out)
