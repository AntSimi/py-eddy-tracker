# -*- coding: utf-8 -*-
"""
"""
from numpy import empty, where
from numba import njit, prange, types as numba_types


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
    product = (x_line_1 - x_line_0) * (y_test - y_line_0) - (x_test - x_line_0) * (y_line_1 - y_line_0)
    return product > 0


@njit(cache=True)
def poly_contain_poly(xy_poly_out, xy_poly_in):
    nb_elt = xy_poly_in.shape[0]
    x = xy_poly_in[:, 0]
    x_ref = xy_poly_out[0,0]
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
                if is_left(xy_poly[i_elt, 0],
                           xy_poly[i_elt, 1],
                           x_next,
                           y_next,
                           x, y
                           ):
                    wn += 1
        else:
            if y_next <= y:
                if not is_left(xy_poly[i_elt, 0],
                               xy_poly[i_elt, 1],
                               x_next,
                               y_next,
                               x, y
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
    for i in range(nb_x):
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

