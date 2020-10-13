# -*- coding: utf-8 -*-
"""
Method for polygon
"""

from numpy import empty, where, array, ones, pi, concatenate
from numpy.linalg import lstsq
from numba import njit, prange, types as numba_types
from Polygon import Polygon


@njit(cache=True)
def is_left(x_line_0, y_line_0, x_line_1, y_line_1, x_test, y_test):
    """
    Test if point is left of an infinit line.

    http://geomalgorithms.com/a03-_inclusion.html
    See: Algorithm 1 "Area of Triangles and Polygons"

    :param float x_line_0:
    :param float y_line_0:
    :param float x_line_1:
    :param float y_line_1:
    :param float x_test:
    :param float y_test:
    :return: > 0 for P2 left of the line through P0 and P1
            = 0 for P2  on the line
            < 0 for P2  right of the line
    :rtype: bool

    """
    # Vector product
    product = (x_line_1 - x_line_0) * (y_test - y_line_0) - (x_test - x_line_0) * (
        y_line_1 - y_line_0
    )
    return product > 0


@njit(cache=True)
def poly_contain_poly(xy_poly_out, xy_poly_in):
    """
    Check if poly_in is include in poly_out.

    :param vertice xy_poly_out:
    :param vertice xy_poly_in:
    :return: True if poly_in is in poly_out
    :rtype: bool
    """
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
def poly_area_vertice(v):
    """
    Compute area from vertice.

    :param vertice v: polygon vertice
    :return: area of polygon in coordinates unit
    :rtype: float
    """
    return poly_area(v[:, 0], v[:, 1])


@njit(cache=True)
def poly_area(x, y):
    """
    Must be call with local coordinates (in m, to get an area in m²).

    :param array x:
    :param array y:
    :return: area of polygon in coordinates unit
    :rtype: float
    """
    p_area = x[0] * (y[1] - y[-2])
    nb = x.shape[0]
    for i in range(1, nb - 1):
        p_area += x[i] * (y[1 + i] - y[i - 1])
    return abs(p_area) * 0.5


@njit(cache=True)
def winding_number_poly(x, y, xy_poly):
    """
    Check if x,y is in poly.

    :param float x: x to test
    :param float y: y to test
    :param vertice xy_poly: vertice of polygon
    :return: wn == 0  if x,y is not in poly
    :retype: int
    """
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
    Return index for each grid coordinates within contour.

    http://geomalgorithms.com/a03-_inclusion.html

    :param array x_1d: x of local grid
    :param array y_1d: y of local grid
    :param int i_x0: int to add at x index to have index in global grid
    :param int i_x1: last index in global grid
    :param int x_size: number of x in global grid
    :param int i_y0: int to add at y index to have index in global grid
    :param vertice xy_poly: vertices of polygon which must contain pixel
    :return: Return index in xy_poly
    :rtype: (int,int)
    """
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
def close_center(x0, y0, x1, y1, delta=.1):
    """
    Compute an overlap with circle parameter and return a percentage

    :param array x0: x centers of dataset 0
    :param array y0: y centers of dataset 0
    :param array x1: x centers of dataset 1
    :param array y1: y centers of dataset 1
    :return: Result of cost function
    :rtype: array
    """
    nb0, nb1 = x0.shape[0], x1.shape[0]
    i, j, c = list(), list(), list()
    for i0 in range(nb0):
        xi0, yi0 = x0[i0], y0[i0]
        for i1 in range(nb1):
            if abs(x1[i1] - xi0) > delta:
                continue
            if abs(y1[i1] - yi0) > delta:
                continue
            i.append(i0), j.append(i1), c.append(1)
    return array(i), array(j), array(c)


@njit(cache=True, fastmath=True)
def bbox_intersection(x0, y0, x1, y1):
    """
    Compute bbox to check if there are a bbox intersection.

    :param array x0: x for polygon list 0
    :param array y0: y for polygon list 0
    :param array x1: x for polygon list 1
    :param array y1: y for polygon list 1
    :return: index of each polygon bbox which have an intersection
    :rtype: (int, int)
    """
    nb0 = x0.shape[0]
    nb1 = x1.shape[0]
    x1_min, y1_min = empty(nb1, dtype=x1.dtype), empty(nb1, dtype=x1.dtype)
    x1_max, y1_max = empty(nb1, dtype=x1.dtype), empty(nb1, dtype=x1.dtype)
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
    """
    Return polygon vertice.

    :param array x:
    :param array y:
    :return: Return polygon vertice
    :rtype: vertice
    """
    nb = x.shape[0]
    v = empty((nb, 2), dtype=x.dtype)
    for i in range(nb):
        v[i, 0] = x[i]
        v[i, 1] = y[i]
    return v


@njit(cache=True)
def create_vertice_from_2darray(x, y, index):
    """
    Choose a polygon in x,y list and return vertice.

    :param array x:
    :param array y:
    :param int index:
    :return: Return the vertice of polygon
    :rtype: vertice
    """
    _, nb = x.shape
    v = empty((nb, 2), dtype=x.dtype)
    for i in range(nb):
        v[i, 0] = x[index, i]
        v[i, 1] = y[index, i]
    return v


@njit(cache=True)
def get_wrap_vertice(x0, y0, x1, y1, i):
    """
    Return a vertice for each polygon and check that use same reference coordinates.

    :param array x0: x for polygon list 0
    :param array y0: y for polygon list 0
    :param array x1: x for polygon list 1
    :param array y1: y for polygon list 1
    :param int i: index to use fot the 2 list
    :return: return two compatible vertice
    :rtype: (vertice, vertice)
    """
    x0_, x1_ = x0[i], x1[i]
    if abs(x0_[0] - x1_[0]) > 180:
        ref = x0_[0] - x0.dtype.type(180)
        x1_ = (x1_ - ref) % 360 + ref
    return create_vertice(x0_, y0[i]), create_vertice(x1_, y1[i])


def vertice_overlap(x0, y0, x1, y1, minimal_area=False):
    r"""
    Return percent of overlap for each item.

    :param array x0: x for polygon list 0
    :param array y0: y for polygon list 0
    :param array x1: x for polygon list 1
    :param array y1: y for polygon list 1
    :param bool minimal_area: If True, function will compute intersection/little polygon, else intersection/union
    :return: Result of cost function
    :rtype: array

    By default

        .. math:: Score = \frac{Intersection(P_0,P_1)_{area}}{Union(P_0,P_1)_{area}}

    If minimal area:

        .. math:: Score = \frac{Intersection(P_0,P_1)_{area}}{min(P_{0 area},P_{1 area})}
    """
    nb = x0.shape[0]
    cost = empty(nb)
    for i in range(nb):
        # Get wrapped vertice for index i
        v0, v1 = get_wrap_vertice(x0, y0, x1, y1, i)
        p0 = Polygon(v0)
        p1 = Polygon(v1)
        # Area of intersection
        intersection = (p0 & p1).area()
        # we divide intersection with the little one result from 0 to 1
        if minimal_area:
            cost[i] = intersection / min(p0.area(), p1.area())
        # we divide intersection with polygon merging result from 0 to 1
        else:
            cost[i] = intersection / (p0 + p1).area()
    return cost


def polygon_overlap(p0, p1, minimal_area=False):
    """
    Return percent of overlap for each item.

    :param list(Polygon) p0: List of polygon to compare with p1 list
    :param list(Polygon) p1: List of polygon to compare with p0 list
    :param bool minimal_area: If True, function will compute intersection/little polygon, else intersection/union
    :return: Result of cost function
    :rtype: array
    """
    nb = len(p1)
    cost = empty(nb)
    for i in range(nb):
        p_ = p1[i]
        # Area of intersection
        intersection = (p0 & p_).area()
        # we divide intersection with the little one result from 0 to 1
        if minimal_area:
            cost[i] = intersection / min(p0.area(), p_.area())
        # we divide intersection with polygon merging result from 0 to 1
        else:
            cost[i] = intersection / (p0 + p_).area()
    return cost


@njit(cache=True)
def fit_circle(x, y):
    """
    From a polygon, function will fit a circle.

    Must be call with local coordinates (in m, to get a radius in m).

    :param array x: x of polygon
    :param array y: y of polygon
    :return: x0, y0, radius, shape_error
    :rtype: (float,float,float,float)
    """
    nb_elt = x.shape[0]

    # last coordinates == first
    x_mean = x[1:].mean()
    y_mean = y[1:].mean()

    norme = (x[1:] - x_mean) ** 2 + (y[1:] - y_mean) ** 2
    norme_max = norme.max()
    scale = norme_max ** 0.5

    # Form matrix equation and solve it
    # Maybe put f4
    datas = ones((nb_elt - 1, 3))
    datas[:, 0] = 2.0 * (x[1:] - x_mean) / scale
    datas[:, 1] = 2.0 * (y[1:] - y_mean) / scale

    (x0, y0, radius), _, _, _ = lstsq(datas, norme / norme_max)

    # Unscale data and get circle variables
    radius += x0 ** 2 + y0 ** 2
    radius **= 0.5
    x0 *= scale
    y0 *= scale
    # radius of fitted circle
    radius *= scale
    # center X-position of fitted circle
    x0 += x_mean
    # center Y-position of fitted circle
    y0 += y_mean

    err = shape_error(x, y, x0, y0, radius)
    return x0, y0, radius, err


@njit(cache=True)
def fit_circle_(x, y):
    r"""
    From a polygon, function will fit a circle.

    Must be call with local coordinates (in m, to get a radius in m).

    .. math:: (x_i - x_0)^2 + (y_i - y_0)^2 = r^2
    .. math:: x_i^2 - 2 x_i x_0 + x_0^2 + y_i^2 - 2 y_i y_0 + y_0^2 = r^2
    .. math:: 2 x_0 x_i + 2 y_0 y_i + r^2 - x_0^2 - y_0^2 = x_i^2 + y_i^2

    we get this linear equation

    .. math:: a X + b Y + c = Z

    where :

    .. math:: a = 2 x_0 , b = 2 y_0 , c = r^2 - x_0^2 - y_0^2
    .. math:: X = x_i , Y = y_i , Z = x_i^2 + y_i^2

    Solutions:

    .. math:: x_0 = a / 2 , y_0 = b / 2 , r = \sqrt{c + x_0^2 + y_0^2}


    :param array x: x of polygon
    :param array y: y of polygon
    :return: x0, y0, radius, shape_error
    :rtype: (float,float,float,float)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from py_eddy_tracker.poly import fit_circle_
        from py_eddy_tracker.generic import build_circle
        V = np.array(((2, 2, 3, 3, 2), (-10, -9, -9, -10, -10)), dtype='f4')
        x0, y0, radius, err = fit_circle_(V[0], V[1])
        ax = plt.subplot(111)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.plot(*build_circle(x0,y0, radius), 'r')
        ax.plot(x0, y0, 'r+')
        ax.plot(*V, 'b.')
        plt.show()
    """
    datas = ones((x.shape[0] - 1, 3), dtype=x.dtype)
    # we skip first position which are the same than the last
    datas[:, 0] = x[1:]
    datas[:, 1] = y[1:]
    # Linear regression
    (a, b, c), _, _, _ = lstsq(datas, x[1:] ** 2 + y[1:] ** 2)
    x0, y0 = a / 2.0, b / 2.0
    radius = (c + x0 ** 2 + y0 ** 2) ** 0.5
    err = shape_error(x, y, x0, y0, radius)
    return x0, y0, radius, err


@njit(cache=True, fastmath=True)
def shape_error(x, y, x0, y0, r):
    r"""
    With a polygon(x,y) in local coordinates.

    and circle properties(x0, y0, r), function compute a shape error:

    .. math:: ShapeError = \frac{Polygon_{area} + Circle_{area} - 2 * Intersection_{area}}{Circle_{area}} * 100

    When error > 100, area of difference is bigger than circle area

    :param array x: x of polygon
    :param array y: y of polygon
    :param float x0: x center of circle
    :param float y0: y center of circle
    :param float r: radius of circle
    :return: shape error
    :rtype: float
    """
    # circle area
    c_area = (r ** 2) * pi
    p_area = poly_area(x, y)
    nb = x.shape[0]
    x, y = x.copy(), y.copy()
    # Find distance between circle center and polygon
    for i in range(nb):
        dx, dy = x[i] - x0, y[i] - y0
        rd = r / (dx ** 2 + dy ** 2) ** 0.5
        if rd < 1:
            x[i] = x0 + dx * rd
            y[i] = y0 + dy * rd
    return 100 + (p_area - 2 * poly_area(x, y)) / c_area * 100


@njit(cache=True, fastmath=True)
def get_pixel_in_regular(vertices, x_c, y_c, x_start, x_stop, y_start, y_stop):
    """
    Get a pixel list of a regular grid contain in a contour.

    :param array_like vertices: contour vertice (N,2)
    :param array_like x_c: longitude coordinate of grid
    :param array_like y_c: latitude coordinate of grid
    :param int x_start: west index of contour
    :param int y_start: east index of contour
    :param int x_stop: south index of contour
    :param int y_stop: north index of contour
    """
    if x_stop < x_start:
        x_ref = vertices[0, 0]
        x_array = (
            (concatenate((x_c[x_start:], x_c[:x_stop])) - x_ref + 180) % 360
            + x_ref
            - 180
        )
        return winding_number_grid_in_poly(
            x_array,
            y_c[y_start:y_stop],
            x_start,
            x_stop,
            x_c.shape[0],
            y_start,
            vertices,
        )
    else:
        return winding_number_grid_in_poly(
            x_c[x_start:x_stop],
            y_c[y_start:y_stop],
            x_start,
            x_stop,
            x_c.shape[0],
            y_start,
            vertices,
        )
