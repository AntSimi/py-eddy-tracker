# -*- coding: utf-8 -*-
"""
"""
from numpy import sin, pi, cos, arctan2, empty, nan, absolute, floor, ones, linspace, interp
from numba import njit, prange
from numpy.linalg import lstsq


@njit(cache=True, fastmath=True, parallel=False)
def distance_grid(lon0, lat0, lon1, lat1):
    """
    Args:
        lon0:
        lat0:
        lon1:
        lat1:

    Returns:
        nan value for far away point, and km for other
    """
    nb_0 = lon0.shape[0]
    nb_1 = lon1.shape[0]
    dist = empty((nb_0, nb_1))
    D2R = pi / 180.
    for i in prange(nb_0):
        for j in prange(nb_1):
            dlat = absolute(lat1[j] - lat0[i])
            if dlat > 15:
                dist[i, j] = nan
                continue
            dlon = absolute(lon1[j] - lon0[i])
            if dlon > 180:
                dlon = absolute((dlon + 180) % 360 - 180)
            if dlon > 20:
                dist[i, j] = nan
                continue
            sin_dlat = sin((dlat) * 0.5 * D2R)
            sin_dlon = sin((dlon) * 0.5 * D2R)
            cos_lat1 = cos(lat0[i] * D2R)
            cos_lat2 = cos(lat1[j] * D2R)
            a_val = sin_dlon ** 2 * cos_lat1 * cos_lat2 + sin_dlat ** 2
            dist[i, j] = 6370.997 * 2 * arctan2(a_val ** 0.5, (1 - a_val) ** 0.5)
    return dist


@njit(cache=True, fastmath=True)
def distance(lon0, lat0, lon1, lat1):
    D2R = pi / 180.
    sin_dlat = sin((lat1 - lat0) * 0.5 * D2R)
    sin_dlon = sin((lon1 - lon0) * 0.5 * D2R)
    cos_lat1 = cos(lat0 * D2R)
    cos_lat2 = cos(lat1 * D2R)
    a_val = sin_dlon ** 2 * cos_lat1 * cos_lat2 + sin_dlat ** 2
    return 6370997.0 * 2 * arctan2(a_val ** 0.5, (1 - a_val) ** 0.5)


@njit(cache=True)
def distance_vincenty(lon0, lat0, lon1, lat1):
    """ better than haversine but buggy ??"""
    D2R = pi / 180.
    dlon = (lon1 - lon0) * D2R
    cos_dlon = cos(dlon)
    cos_lat1 = cos(lat0 * D2R)
    cos_lat2 = cos(lat1 * D2R)
    sin_lat1 = sin(lat0 * D2R)
    sin_lat2 = sin(lat1 * D2R)
    return 6370997.0 * arctan2(
        ((cos_lat2 * sin(dlon) ** 2) + (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_dlon) ** 2) ** .5,
        sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_dlon)


@njit(cache=True, fastmath=True)
def interp2d_geo(x_g, y_g, z_g, m_g, x, y):
    """For geographic grid, test of cicularity
    Maybe test if we are out of bounds
    """
    x_ref = x_g[0]
    y_ref = y_g[0]
    x_step = x_g[1] - x_ref
    y_step = y_g[1] - y_ref
    nb_x = x_g.shape[0]
    is_circular = (x_g[-1] + x_step) % 360 == x_g[0] % 360
    z = empty(x.shape)
    for i in prange(x.size):
        x_ = (x[i] - x_ref) / x_step
        y_ = (y[i] - y_ref) / y_step
        i0 = int(floor(x_))
        i1 = i0 + 1
        if is_circular:
            xd = (x_ - i0)
            i0 %= nb_x
            i1 %= nb_x
        j0 = int(floor(y_))
        j1 = j0 + 1
        yd = (y_ - j0)
        z00 = z_g[i0, j0]
        z01 = z_g[i0, j1]
        z10 = z_g[i1, j0]
        z11 = z_g[i1, j1]
        if m_g[i0, j0] or m_g[i0, j1] or m_g[i1, j0] or m_g[i1, j1]:
            z[i] = nan
        else:
            z[i] = (z00 * (1 - xd) + (z10 * xd)) * (1 - yd) + (z01 * (1 - xd) + z11 * xd) * yd
    return z

@njit(cache=True, fastmath=True, parallel=True)
def custom_convolution(data, mask, kernel):
    """do sortin at high lattitude big part of value are masked"""
    nb_x = kernel.shape[0]
    demi_x = int((nb_x - 1) / 2)
    demi_y = int((kernel.shape[1] - 1) / 2)
    out = empty(data.shape[0] - nb_x + 1)
    for i in prange(out.shape[0]):
        if mask[i + demi_x, demi_y] == 1:
            w = (mask[i:i + nb_x] * kernel).sum()
            if w != 0:
                out[i] = (data[i:i + nb_x] * kernel).sum() / w
            else:
                out[i] = nan
        else:
            out[i] = nan
    return out


@njit(cache=True)
def fit_circle(x_vec, y_vec):
    nb_elt = x_vec.shape[0]
    p_inon_x = empty(nb_elt)
    p_inon_y = empty(nb_elt)

    # last coordinates == first
    x_mean = x_vec[1:].mean()
    y_mean = y_vec[1:].mean()

    norme = (x_vec[1:] - x_mean) ** 2 + (y_vec[1:] - y_mean) ** 2
    norme_max = norme.max()
    scale = norme_max ** .5

    # Form matrix equation and solve it
    # Maybe put f4
    datas = ones((nb_elt - 1, 3))
    datas[:, 0] = 2. * (x_vec[1:] - x_mean) / scale
    datas[:, 1] = 2. * (y_vec[1:] - y_mean) / scale

    (center_x, center_y, radius), residuals, rank, s = lstsq(datas, norme / norme_max)

    # Unscale data and get circle variables
    radius += center_x ** 2 + center_y ** 2
    radius **= .5
    center_x *= scale
    center_y *= scale
    # radius of fitted circle
    radius *= scale
    # center X-position of fitted circle
    center_x += x_mean
    # center Y-position of fitted circle
    center_y += y_mean

    # area of fitted circle
    c_area = (radius ** 2) * pi
    # Find distance between circle center and contour points_inside_poly
    for i_elt in range(nb_elt):
        # Find distance between circle center and contour points_inside_poly
        dist_poly = ((x_vec[i_elt] - center_x) ** 2 + (y_vec[i_elt] - center_y) ** 2) ** .5
        # Indices of polygon points outside circle
        # p_inon_? : polygon x or y points inside & on the circle
        if dist_poly > radius:
            p_inon_y[i_elt] = center_y + radius * (y_vec[i_elt] - center_y) / dist_poly
            p_inon_x[i_elt] = center_x - (center_x - x_vec[i_elt]) * (center_y - p_inon_y[i_elt]) / (
                        center_y - y_vec[i_elt])
        else:
            p_inon_x[i_elt] = x_vec[i_elt]
            p_inon_y[i_elt] = y_vec[i_elt]

    # Area of closed contour/polygon enclosed by the circle
    p_area_incirc = 0
    p_area = 0
    for i_elt in range(nb_elt - 1):
        # Indices of polygon points outside circle
        # p_inon_? : polygon x or y points inside & on the circle
        p_area_incirc += p_inon_x[i_elt] * p_inon_y[1 + i_elt] - p_inon_x[i_elt + 1] * p_inon_y[i_elt]
        # Shape test
        # Area and centroid of closed contour/polygon
        p_area += x_vec[i_elt] * y_vec[1 + i_elt] - x_vec[1 + i_elt] * y_vec[i_elt]
    p_area = abs(p_area) * .5
    p_area_incirc = abs(p_area_incirc) * .5

    a_err = (c_area - 2 * p_area_incirc + p_area) * 100. / c_area
    return center_x, center_y, radius, a_err


@njit(cache=True, fastmath=True)
def uniform_resample(x_val, y_val, num_fac=2, fixed_size=None):
    """
    Resample contours to have (nearly) equal spacing
       x_val, y_val    : input contour coordinates
       num_fac : factor to increase lengths of output coordinates
    """
    # Get distances
    dist = empty(x_val.shape)
    dist[0] = 0
    dist[1:] = distance(x_val[:-1], y_val[:-1], x_val[1:], y_val[1:])
    # To be still monotonous (dist is store in m)
    dist[1:][dist[1:]<1e-3] = 1e-3
    dist = dist.cumsum()
    # Get uniform distances
    if fixed_size is None:
        fixed_size = dist.size * num_fac
    d_uniform = linspace(0, dist[-1], fixed_size)
    x_new = interp(d_uniform, dist, x_val)
    y_new = interp(d_uniform, dist, y_val)
    return x_new, y_new
