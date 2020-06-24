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

from numpy import empty, where, array
from numba import njit, prange, types as numba_types
from Polygon import Polygon


@njit(cache=True)
def is_left(x_line_0, y_line_0, x_line_1, y_line_1, x_test, y_test):
    """
    http://geomalgorithms.com/a03-_inclusion.html
    isLeft(): tests if a point is Left|On|Right of an infinite line.
    Input:  three points P0, P1, and P2
    Return: >0 for P2 left of the line through P0 and P1
            =0 for P2  on the line
            <0 for P2  right of the line
    See: Algorithm 1 "Area of Triangles and Polygons"
    """
    # Vector product
    product = (x_line_1 - x_line_0) * (y_test - y_line_0) - (x_test - x_line_0) * (
        y_line_1 - y_line_0
    )
    return product > 0


@njit(cache=True)
def poly_contain_poly(xy_poly_out, xy_poly_in):
    nb_elt = xy_poly_in.shape[0]
    x = xy_poly_in[:, 0]
    x_ref = xy_poly_out[0, 0]
    d_x = abs(x[0] - x_ref)
    if d_x > 180:
        x = (x - x_ref + 180) % 360 + x_ref - 180
    for i_elt in prange(nb_elt):
        wn = winding_number_poly(x[i_elt], xy_poly_in[i_elt, 1], xy_poly_out)
        if wn == 0:
            return False
    return True


@njit(cache=True)
def winding_number_poly(x, y, xy_poly):
    nb_elt = xy_poly.shape[0]
    wn = 0
    # loop through all edges of the polygon
    for i_elt in range(nb_elt):
        if i_elt + 1 == nb_elt:
            x_next = xy_poly[0, 0]
            y_next = xy_poly[0, 1]
        else:
            x_next = xy_poly[i_elt + 1, 0]
            y_next = xy_poly[i_elt + 1, 1]
        if xy_poly[i_elt, 1] <= y:
            if y_next > y:
                if is_left(xy_poly[i_elt, 0], xy_poly[i_elt, 1], x_next, y_next, x, y):
                    wn += 1
        else:
            if y_next <= y:
                if not is_left(
                    xy_poly[i_elt, 0], xy_poly[i_elt, 1], x_next, y_next, x, y
                ):
                    wn -= 1
    return wn


@njit(cache=True)
def winding_number_grid_in_poly(x_1d, y_1d, i_x0, i_x1, x_size, i_y0, xy_poly):
    """
    http://geomalgorithms.com/a03-_inclusion.html
    wn_PnPoly(): winding number test for a point in a polygon
          Input:   P = a point,
                   V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
          Return:  wn = the winding number (=0 only when P is outside)
    """
    # the  winding number counter
    nb_x, nb_y = len(x_1d), len(y_1d)
    wn = empty((nb_x, nb_y), dtype=numba_types.bool_)
    for i in prange(nb_x):
        x_pt = x_1d[i]
        for j in range(nb_y):
            y_pt = y_1d[j]
            wn[i, j] = winding_number_poly(x_pt, y_pt, xy_poly)
    i_x, i_y = where(wn)
    i_x += i_x0
    i_y += i_y0
    if i_x1 < i_x0:
        i_x %= x_size
    return i_x, i_y


@njit(cache=True, fastmath=True)
def bbox_intersection(x0, y0, x1, y1):
    """compute bbox to check if there are a bbox intersection
    """
    nb0 = x0.shape[0]
    nb1 = x1.shape[0]
    x1_min, y1_min = empty(nb1), empty(nb1)
    x1_max, y1_max = empty(nb1), empty(nb1)
    for i1 in range(nb1):
        x1_min[i1], y1_min[i1] = x1[i1].min(), y1[i1].min()
        x1_max[i1], y1_max[i1] = x1[i1].max(), y1[i1].max()

    i, j = list(), list()
    for i0 in range(nb0):
        x_in_min, y_in_min = x0[i0].min(), y0[i0].min()
        x_in_max, y_in_max = x0[i0].max(), y0[i0].max()
        for i1 in range(nb1):
            if y_in_max < y1_min[i1] or y_in_min > y1_max[i1]:
                continue
            x1_min_ = x1_min[i1]
            x1_max_ = x1_max[i1]
            if abs(x_in_min - x1_min_) > 180:
                ref = x_in_min - 180
                x1_min_ = (x1_min_ - ref) % 360 + ref
                x1_max_ = (x1_max_ - ref) % 360 + ref
            if x_in_max < x1_min_ or x_in_min > x1_max_:
                continue
            i.append(i0)
            j.append(i1)
    return array(i), array(j)


@njit(cache=True)
def create_vertice(x, y):
    nb = x.shape[0]
    v = empty((nb, 2))
    for i in range(nb):
        v[i, 0] = x[i]
        v[i, 1] = y[i]
    return v


def common_area(x0, y0, x1, y1):
    nb, _ = x0.shape
    cost = empty((nb))
    for i in range(nb):
        x0_, x1_ = x0[i], x1[i]
        if abs(x0_[0] - x1_[0]) > 180:
            x1_ = (x1[i] - (x0_[0] - 180)) % 360 + x0_[0] - 180
        p0 = Polygon(create_vertice(x0_, y0[i]))
        p1 = Polygon(create_vertice(x1_, y1[i]))
        cost[i] = (p0 & p1).area() / (p0 + p1).area()
    return cost
