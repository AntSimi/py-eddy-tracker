# -*- coding: utf-8 -*-
"""
Method for polygon
"""

import heapq

from Polygon import Polygon
from numba import njit, prange, types as numba_types
from numpy import arctan, array, concatenate, empty, nan, ones, pi, where, zeros
from numpy.linalg import lstsq

from .generic import build_index


@njit(cache=True)
def is_left(
    x_line_0: float,
    y_line_0: float,
    x_line_1: float,
    y_line_1: float,
    x_test: float,
    y_test: float,
) -> bool:
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
    Must be called with local coordinates (in m, to get an area in m²).

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
def convexs(x, y):
    """
    Check if polygons are convex

    :param array[float] x:
    :param array[float] y:
    :return: True if convex
    :rtype: array[bool]
    """
    nb_poly = x.shape[0]
    flag = empty(nb_poly, dtype=numba_types.bool_)
    for i in range(nb_poly):
        flag[i] = convex(x[i], y[i])
    return flag


@njit(cache=True)
def convex(x, y):
    """
    Check if polygon is convex

    :param array[float] x:
    :param array[float] y:
    :return: True if convex
    :rtype: bool
    """
    nb = x.shape[0]
    x0, y0, x1, y1, x2, y2 = x[-2], y[-2], x[-1], y[-1], x[1], y[1]
    # if first is left it must be always left if it's right it must be always right
    ref = is_left(x0, y0, x1, y1, x2, y2)
    # We skip 0 because it's same than -1
    # We skip 1 because we tested previously
    for i in range(2, nb):
        # shift position
        x0, y0, x1, y1 = x1, y1, x2, y2
        x2, y2 = x[i], y[i]
        # test
        if ref != is_left(x0, y0, x1, y1, x2, y2):
            return False
    return True


@njit(cache=True)
def get_convex_hull(x, y):
    """
    Get convex polygon which enclosed current polygon

    Work only if contour is describe anti-clockwise

    :param array[float] x:
    :param array[float] y:
    :return: a convex polygon
    :rtype: array,array
    """
    nb = x.shape[0] - 1
    indices = list()
    # leftmost point
    i_first = x[:-1].argmin()
    indices.append(i_first)
    i_next = (i_first + 1) % nb
    # Will define bounds line
    x0, y0, x1, y1 = x[i_first], y[i_first], x[i_next], y[i_next]
    xf, yf = x0, y0
    # we will check if no point are right
    while True:
        i_test = (i_next + 1) % nb
        # value to test
        xt, yt = x[i_test], y[i_test]
        # We will test all the position until we touch first one,
        # If all next position are on the left we keep x1, y1
        # if not we will replace by xt,yt which are more outter
        while is_left(x0, y0, x1, y1, xt, yt):
            i_test += 1
            i_test %= nb
            if i_test == i_first:
                x0, y0 = x1, y1
                indices.append(i_next)
                i_next += 1
                i_next %= nb
                x1, y1 = x[i_next], y[i_next]
                break
            xt, yt = x[i_test], y[i_test]
        if i_test != i_first:
            i_next = i_test
            x1, y1 = x[i_next], y[i_next]
        if i_next == (i_first - 1) % nb:
            if is_left(x0, y0, x1, y1, xf, yf):
                indices.append(i_next)
            break
    indices.append(i_first)
    indices = array(indices)
    return x[indices], y[indices]


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
            # We close polygon with first value (no need to duplicate first value)
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
def close_center(x0, y0, x1, y1, delta=0.1):
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


@njit(cache=True)
def create_meshed_particles(lons, lats, step):
    x_out, y_out, i_out = list(), list(), list()
    nb = lons.shape[0]
    for i in range(nb):
        lon, lat = lons[i], lats[i]
        vertice = create_vertice(*reduce_size(lon, lat))
        lon_min, lon_max = lon.min(), lon.max()
        lat_min, lat_max = lat.min(), lat.max()
        y0 = lat_min - lat_min % step
        x = lon_min - lon_min % step
        while x <= lon_max:
            y = y0
            while y <= lat_max:
                if winding_number_poly(x, y, vertice):
                    x_out.append(x), y_out.append(y), i_out.append(i)
                y += step
            x += step
    return array(x_out), array(y_out), array(i_out)


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
    return array(i, dtype=numba_types.int32), array(j, dtype=numba_types.int32)


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


def merge(x, y):
    """
    Merge all polygon of the list

    :param array x: 2D array for a list of polygon
    :param array y: 2D array for a list of polygon
    :return: Polygons which enclosed all
    :rtype: array, array
    """
    nb = x.shape[0]
    p = None
    for i in range(nb):
        p_ = Polygon(create_vertice(x[i], y[i]))
        if p is None:
            p = p_
        else:
            p += p_
    x, y = list(), list()
    for p_ in p:
        p_ = array(p_).T
        x.append((nan,))
        y.append((nan,))
        x.append(p_[0])
        y.append(p_[1])
    return concatenate(x), concatenate(y)


def vertice_overlap(x0, y0, x1, y1, minimal_area=False, p1_area=False, hybrid_area=False, min_overlap=0):
    r"""
    Return percent of overlap for each item.

    :param array x0: x for polygon list 0
    :param array y0: y for polygon list 0
    :param array x1: x for polygon list 1
    :param array y1: y for polygon list 1
    :param bool minimal_area: If True, function will compute intersection/little polygon, else intersection/union
    :param bool p1_area: If True, function will compute intersection/p1 polygon, else intersection/union
    :param bool hybrid_area: If True, function will compute like union,
                             but if cost is under min_overlap, obs is kept in case of fully included
    :param float min_overlap: under this value cost is set to zero
    :return: Result of cost function
    :rtype: array

    By default

        .. math:: Score = \frac{Intersection(P_0,P_1)_{area}}{Union(P_0,P_1)_{area}}

    If minimal area:

        .. math:: Score = \frac{Intersection(P_0,P_1)_{area}}{min(P_{0 area},P_{1 area})}

    If P1 area:

        .. math:: Score = \frac{Intersection(P_0,P_1)_{area}}{P_{1 area}}
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
        if intersection == 0:
            cost[i] = 0
            continue
        p0_area_, p1_area_ = p0.area(), p1.area()
        if minimal_area:
            cost_ = intersection / min(p0_area_, p1_area_)
        # we divide intersection with p1
        elif p1_area:
            cost_ = intersection / p1_area_
        # we divide intersection with polygon merging result from 0 to 1
        else:
            cost_ = intersection / (p0_area_ + p1_area_ - intersection)
        if cost_ >= min_overlap:
            cost[i] = cost_
        else:
            if hybrid_area and cost_ != 0 and (intersection / min(p0_area_, p1_area_)) > .99:
                cost[i] = cost_
            else:
                cost[i] = 0
    return cost


def polygon_overlap(p0, p1, minimal_area=False):
    """
    Return percent of overlap for each item.

    :param list(Polygon) p0: List of polygon to compare with p1 list
    :param list(Polygon) p1: List of polygon to compare with p0 list
    :param bool minimal_area: If True, function will compute intersection/smaller polygon, else intersection/union
    :return: Result of cost function
    :rtype: array
    """
    nb = len(p1)
    cost = empty(nb)
    for i in range(nb):
        p_ = p1[i]
        # Area of intersection
        intersection = (p0 & p_).area()
        # we divide the intersection by the smaller area, result from 0 to 1
        if minimal_area:
            cost[i] = intersection / min(p0.area(), p_.area())
        # we divide the intersection by the merged polygons area, result from 0 to 1
        else:
            cost[i] = intersection / (p0 + p_).area()
    return cost


# FIXME: only one function is needed
@njit(cache=True)
def fit_circle(x, y):
    """
    From a polygon, function will fit a circle.

    Must be called with local coordinates (in m, to get a radius in m).

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
    scale = norme_max**0.5

    # Form matrix equation and solve it
    # Maybe put f4
    datas = ones((nb_elt - 1, 3))
    datas[:, 0] = 2.0 * (x[1:] - x_mean) / scale
    datas[:, 1] = 2.0 * (y[1:] - y_mean) / scale

    (x0, y0, radius), _, _, _ = lstsq(datas, norme / norme_max)

    # Unscale data and get circle variables
    radius += x0**2 + y0**2
    radius **= 0.5
    x0 *= scale
    y0 *= scale
    # radius of fit circle
    radius *= scale
    # center X-position of fit circle
    x0 += x_mean
    # center Y-position of fit circle
    y0 += y_mean

    err = shape_error(x, y, x0, y0, radius)
    return x0, y0, radius, err


@njit(cache=True)
def fit_ellipse(x, y):
    r"""
    From a polygon, function will fit an ellipse.

    Must be call with local coordinates (in m, to get a radius in m).

    .. math:: (\frac{x - x_0}{a})^2 + (\frac{y - y_0}{b})^2 = 1

    .. math:: (\frac{x^2 - 2 * x * x_0 + x_0 ^2}{a^2}) + \frac{y^2 - 2 * y * y_0 + y_0 ^2}{b^2}) = 1

    In case of angle
    https://en.wikipedia.org/wiki/Ellipse

    """
    nb = x.shape[0]
    datas = ones((nb, 5), dtype=x.dtype)
    datas[:, 0] = x**2
    datas[:, 1] = x * y
    datas[:, 2] = y**2
    datas[:, 3] = x
    datas[:, 4] = y
    (a, b, c, d, e), _, _, _ = lstsq(datas, ones(nb, dtype=x.dtype))
    det = b**2 - 4 * a * c
    if det > 0:
        print(det)
    x0 = (2 * c * d - b * e) / det
    y0 = (2 * a * e - b * d) / det

    AB1 = 2 * (a * e**2 + c * d**2 - b * d * e - det)
    AB2 = a + c
    AB3 = ((a - c) ** 2 + b**2) ** 0.5
    A = -((AB1 * (AB2 + AB3)) ** 0.5) / det
    B = -((AB1 * (AB2 - AB3)) ** 0.5) / det
    theta = arctan((c - a - AB3) / b)
    return x0, y0, A, B, theta


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

        V = np.array(((2, 2, 3, 3, 2), (-10, -9, -9, -10, -10)), dtype="f4")
        x0, y0, radius, err = fit_circle_(V[0], V[1])
        ax = plt.subplot(111)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.plot(*build_circle(x0, y0, radius), "r")
        ax.plot(x0, y0, "r+")
        ax.plot(*V, "b.")
        plt.show()
    """
    datas = ones((x.shape[0] - 1, 3), dtype=x.dtype)
    # we skip first position which are the same than the last
    datas[:, 0] = x[1:]
    datas[:, 1] = y[1:]
    # Linear regression
    (a, b, c), _, _, _ = lstsq(datas, x[1:] ** 2 + y[1:] ** 2)
    x0, y0 = a / 2.0, b / 2.0
    radius = (c + x0**2 + y0**2) ** 0.5
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
    c_area = (r**2) * pi
    p_area = poly_area(x, y)
    nb = x.shape[0]
    x, y = x.copy(), y.copy()
    # Find distance between circle center and polygon
    for i in range(nb):
        dx, dy = x[i] - x0, y[i] - y0
        rd = r / (dx**2 + dy**2) ** 0.5
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


@njit(cache=True)
def tri_area2(x, y, i0, i1, i2):
    """Double area of triangle

    :param array x:
    :param array y:
    :param int i0: indice of first point
    :param int i1: indice of second point
    :param int i2: indice of third point
    :return: area
    :rtype: float
    """
    x0, y0 = x[i0], y[i0]
    x1, y1 = x[i1], y[i1]
    x2, y2 = x[i2], y[i2]
    p_area2 = (x0 - x2) * (y1 - y0) - (x0 - x1) * (y2 - y0)
    return abs(p_area2)


@njit(cache=True)
def visvalingam(x, y, fixed_size=18):
    """Polygon simplification with visvalingam algorithm

    X, Y are considered like a polygon, the next point after the last one is the first one

    :param array x:
    :param array y:
    :param int fixed_size: array size of out
    :return:
        New (x, y) array, last position will be equal to first one, if array size is 6,
        there is only 5 point.
    :rtype: array,array

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from py_eddy_tracker.poly import visvalingam

        x = np.array([1, 2, 3, 4, 5, 6.75, 6, 1])
        y = np.array([-0.5, -1.5, -1, -1.75, -1, -1, -0.5, -0.5])
        ax = plt.subplot(111)
        ax.set_aspect("equal")
        ax.grid(True), ax.set_ylim(-2, -.2)
        ax.plot(x, y, "r",  lw=5)
        ax.plot(*visvalingam(x,y,6), "b", lw=2)
        plt.show()
    """
    # TODO :  in case of original size lesser than fixed size, jump at the end
    nb = x.shape[0]
    nb_ori = nb
    # Get indice of first triangle
    i0, i1 = nb - 2, nb - 1
    # Init heap with first area and tiangle
    h = [(tri_area2(x, y, i0, i1, 0), (i0, i1, 0))]
    # Roll index for next one
    i0 = i1
    i1 = 0
    # Index of previous valid point
    i_previous = empty(nb, dtype=numba_types.int64)
    # Index of next valid point
    i_next = empty(nb, dtype=numba_types.int64)
    # Mask of removed
    removed = zeros(nb, dtype=numba_types.bool_)
    i_previous[0] = -1
    i_next[0] = -1
    for i in range(1, nb):
        i_previous[i] = -1
        i_next[i] = -1
        # We add triangle area for all triangle
        heapq.heappush(h, (tri_area2(x, y, i0, i1, i), (i0, i1, i)))
        i0 = i1
        i1 = i
    # we continue until we are equal to nb_pt
    while nb >= fixed_size:
        # We pop lower area
        _, (i0, i1, i2) = heapq.heappop(h)
        # We check if triangle is valid(i0 or i2 not removed)
        if removed[i0] or removed[i2]:
            # In this cas nothing to do
            continue
        # Flag obs like removed
        removed[i1] = True
        # We count point still valid
        nb -= 1
        # Modify index for the next and previous, we jump over i1
        i_previous[i2] = i0
        i_next[i0] = i2
        # We insert 2 triangles which are modified by the deleted point
        # Previous triangle
        i_1 = i_previous[i0]
        if i_1 == -1:
            i_1 = (i0 - 1) % nb_ori
        heapq.heappush(h, (tri_area2(x, y, i_1, i0, i2), (i_1, i0, i2)))
        # Previous triangle
        i3 = i_next[i2]
        if i3 == -1:
            i3 = (i2 + 1) % nb_ori
        heapq.heappush(h, (tri_area2(x, y, i0, i2, i3), (i0, i2, i3)))
    x_new, y_new = empty(fixed_size, dtype=x.dtype), empty(fixed_size, dtype=y.dtype)
    j = 0
    for i, flag in enumerate(removed):
        if not flag:
            x_new[j] = x[i]
            y_new[j] = y[i]
            j += 1
    # we copy first value to fill array end
    x_new[j:] = x_new[0]
    y_new[j:] = y_new[0]
    return x_new, y_new


@njit(cache=True)
def reduce_size(x, y):
    """
    Reduce array size if last position is repeated, in order to save compute time

    :param array x: longitude
    :param array y: latitude

    :return: reduce arrays x,y
    :rtype: ndarray,ndarray
    """
    i = x.shape[0]
    x0, y0 = x[0], y[0]
    while True:
        i -= 1
        if x[i] != x0 or y[i] != y0:
            i += 1
            # In case of virtual obs all value could be fill with same value, to avoid empty array
            i = max(3, i)
            return x[:i], y[:i]


@njit(cache=True)
def group_obs(x, y, step, nb_x):
    """Get index k_box for each box, and indexes to sort"""
    nb = x.size
    i = empty(nb, dtype=numba_types.uint32)
    for k in range(nb):
        i[k] = box_index(x[k], y[k], step, nb_x)
    return i, i.argsort(kind="mergesort")


@njit(cache=True)
def box_index(x, y, step, nb_x):
    """Return k_box index for each value"""
    return numba_types.uint32((x % 360) // step + nb_x * ((y + 90) // step))


@njit(cache=True)
def box_indexes(x, y, step):
    """Return i_box,j_box index for each value"""
    return numba_types.uint32((x % 360) // step), numba_types.uint32((y + 90) // step)


@njit(cache=True)
def poly_indexs(x_p, y_p, x_c, y_c):
    """
    Index of contour for each postion inside a contour, -1 in case of no contour

    :param array x_p: longitude to test (must be defined, no nan)
    :param array y_p: latitude to test (must be defined, no nan)
    :param array x_c: longitude of contours
    :param array y_c: latitude of contours
    """
    nb_x = 360
    step = 1.0
    i, i_order = group_obs(x_p, y_p, step, nb_x)
    nb_p = x_p.shape[0]
    nb_c = x_c.shape[0]
    indexs = -ones(nb_p, dtype=numba_types.int32)
    # Adress table to get test bloc
    start_index, end_index, i_first = build_index(i[i_order])
    nb_bloc = end_index.size
    for i_contour in range(nb_c):
        # Build vertice and box included contour
        x_, y_ = reduce_size(x_c[i_contour], y_c[i_contour])
        x_c_min, y_c_min = x_.min(), y_.min()
        x_c_max, y_c_max = x_.max(), y_.max()
        v = create_vertice(x_, y_)
        i0, j0 = box_indexes(x_c_min, y_c_min, step)
        i1, j1 = box_indexes(x_c_max, y_c_max, step)
        # i0 could be greater than i1, (x_c is always continious) so you could have a contour over bound
        if i0 > i1:
            i1 += nb_x
        for i_x in range(i0, i1 + 1):
            # we force i_x in 0 360 range
            i_x %= nb_x
            for i_y in range(j0, j1 + 1):
                # Get box indices
                i_box = i_x + nb_x * i_y - i_first
                # Indice must be in table range
                if i_box < 0 or i_box >= nb_bloc:
                    continue
                for i_p_ordered in range(start_index[i_box], end_index[i_box]):
                    i_p = i_order[i_p_ordered]
                    if indexs[i_p] != -1:
                        continue
                    y = y_p[i_p]
                    if y > y_c_max:
                        continue
                    if y < y_c_min:
                        continue
                    # Normalize longitude at +-180° around x_c_min
                    x = (x_p[i_p] - x_c_min + 180) % 360 + x_c_min - 180
                    if x > x_c_max:
                        continue
                    if x < x_c_min:
                        continue
                    if winding_number_poly(x, y, v) != 0:
                        indexs[i_p] = i_contour
    return indexs


@njit(cache=True)
def insidepoly(x_p, y_p, x_c, y_c):
    """
    True for each postion inside a contour

    :param array x_p: longitude to test
    :param array y_p: latitude to test
    :param array x_c: longitude of contours
    :param array y_c: latitude of contours
    """
    return poly_indexs(x_p, y_p, x_c, y_c) != -1
