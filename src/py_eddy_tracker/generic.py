# -*- coding: utf-8 -*-
"""
Tool method which use mostly numba
"""

from numba import njit, prange
from numba import types as numba_types
from numpy import (
    absolute,
    arcsin,
    arctan2,
    bool_,
    cos,
    empty,
    floor,
    histogram,
    interp,
    isnan,
    linspace,
    nan,
    ones,
    pi,
    radians,
    sin,
    where,
    zeros,
)


@njit(cache=True)
def count_consecutive(mask):
    """
    Count consecutive event every False flag count restart

    :param array[bool] mask: event to count
    :return: count when consecutive event
    :rtype: array
    """
    count = 0
    output = zeros(mask.shape, dtype=numba_types.int_)
    for i in range(mask.shape[0]):
        if not mask[i]:
            count = 0
            continue
        count += 1
        output[i] = count
    return output


@njit(cache=True)
def reverse_index(index, nb):
    """
    Compute a list of index, which are not in index.

    :param array index: index of group which will be set to False
    :param array nb: Count for each group
    :return: mask of value selected
    :rtype: array
    """
    m = ones(nb, dtype=numba_types.bool_)
    for i in index:
        m[i] = False
    return where(m)[0]


@njit(cache=True)
def build_index(groups):
    """We expected that variable is monotonous, and return index for each step change.

    :param array groups: array which contain group to be separated
    :return: (first_index of each group, last_index of each group, value to shift group)
    :rtype: (array, array, int)
    """
    i0, i1 = groups.min(), groups.max()
    amplitude = i1 - i0 + 1
    # Index of first observation for each group
    first_index = zeros(amplitude, dtype=numba_types.int_)
    for i, group in enumerate(groups[:-1]):
        # Get next value to compare
        next_group = groups[i + 1]
        # if different we need to set index
        if group != next_group:
            first_index[group - i0 + 1 : next_group - i0 + 1] = i + 1
    last_index = zeros(amplitude, dtype=numba_types.int_)
    last_index[:-1] = first_index[1:]
    # + 2 because we iterate only until -2 and we want upper bound ( 1 + 1)
    last_index[-1] = i + 2
    return first_index, last_index, i0


@njit(cache=True)
def hist_numba(x, bins):
    """Call numba histogram  to speed up."""
    return histogram(x, bins)


@njit(cache=True, fastmath=True, parallel=False)
def distance_grid(lon0, lat0, lon1, lat1):
    """
    Get distance for every couple of point.

    :param array lon0:
    :param array lat0:
    :param array lon1:
    :param array lat1:

    :return: nan value for far away point, and km for other
    :rtype: array
    """
    nb_0 = lon0.shape[0]
    nb_1 = lon1.shape[0]
    dist = empty((nb_0, nb_1))
    D2R = pi / 180.0
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
    """
    Compute distance between points from each line.

    :param float lon0:
    :param float lat0:
    :param float lon1:
    :param float lat1:
    :return: distance (in m)
    :rtype: array
    """
    D2R = pi / 180.0
    sin_dlat = sin((lat1 - lat0) * 0.5 * D2R)
    sin_dlon = sin((lon1 - lon0) * 0.5 * D2R)
    cos_lat1 = cos(lat0 * D2R)
    cos_lat2 = cos(lat1 * D2R)
    a_val = sin_dlon ** 2 * cos_lat1 * cos_lat2 + sin_dlat ** 2
    return 6370997.0 * 2 * arctan2(a_val ** 0.5, (1 - a_val) ** 0.5)


@njit(cache=True)
def cumsum_by_track(field, track):
    """
    Cumsum by track.

    :param array field: data to sum
    :pram array(int) track: id of track to separate data
    :return: cumsum with a reset at each start of track
    :rtype: array
    """
    tr_previous = 0
    d_cum = 0
    cumsum_array = empty(track.shape, dtype=field.dtype)
    for i in range(field.shape[0]):
        tr = track[i]
        if tr != tr_previous:
            d_cum = 0
        d_cum += field[i]
        cumsum_array[i] = d_cum
        tr_previous = tr
    return cumsum_array


@njit(cache=True, fastmath=True)
def interp2d_geo(x_g, y_g, z_g, m_g, x, y, nearest=False):
    """
    For geographic grid, test of cicularity.

    :param array x_g: coordinates of grid
    :param array y_g: coordinates of grid
    :param array z_g: Grid value
    :param array m_g: Boolean grid, True if value is masked
    :param array x: coordinate where interpolate z
    :param array y: coordinate where interpolate z
    :param bool nearest: if true we will take nearest pixel
    :return: z interpolated
    :rtype: array
    """
    # TODO : Maybe test if we are out of bounds
    x_ref = x_g[0]
    y_ref = y_g[0]
    x_step = x_g[1] - x_ref
    y_step = y_g[1] - y_ref
    nb_x = x_g.shape[0]
    nb_y = y_g.shape[0]
    is_circular = abs(x_g[-1] % 360 - (x_g[0] - x_step) % 360) < 1e-5
    z = empty(x.shape, dtype=z_g.dtype)
    for i in prange(x.size):
        x_ = (x[i] - x_ref) / x_step
        y_ = (y[i] - y_ref) / y_step
        i0 = int(floor(x_))
        i1 = i0 + 1
        xd = x_ - i0
        j0 = int(floor(y_))
        j1 = j0 + 1
        if is_circular:
            i0 %= nb_x
            i1 %= nb_x
        else:
            if i1 >= nb_x or i0 < 0 or j0 < 0 or j1 >= nb_y:
                z[i] = nan
                continue

        yd = y_ - j0
        z00 = z_g[i0, j0]
        z01 = z_g[i0, j1]
        z10 = z_g[i1, j0]
        z11 = z_g[i1, j1]
        if m_g[i0, j0] or m_g[i0, j1] or m_g[i1, j0] or m_g[i1, j1]:
            z[i] = nan
        else:
            if nearest:
                if xd <= 0.5:
                    if yd <= 0.5:
                        z[i] = z00
                    else:
                        z[i] = z01
                else:
                    if yd <= 0.5:
                        z[i] = z10
                    else:
                        z[i] = z11
            else:
                z[i] = (z00 * (1 - xd) + (z10 * xd)) * (1 - yd) + (
                    z01 * (1 - xd) + z11 * xd
                ) * yd
    return z


@njit(cache=True, fastmath=True)
def uniform_resample(x_val, y_val, num_fac=2, fixed_size=None):
    """
    Resample contours to have (nearly) equal spacing.

    :param array_like x_val: input x contour coordinates
    :param array_like y_val: input y contour coordinates
    :param int num_fac: factor to increase lengths of output coordinates
    :param int,None fixed_size: if define, it will used to set sampling
    """
    nb = x_val.shape[0]
    # Get distances
    dist = empty(nb)
    dist[0] = 0
    dist[1:] = distance(x_val[:-1], y_val[:-1], x_val[1:], y_val[1:])
    # To be still monotonous (dist is store in m)
    dist[1:][dist[1:] < 1e-3] = 1e-3
    dist = dist.cumsum()
    # Get uniform distances
    if fixed_size is None:
        fixed_size = dist.size * num_fac
    d_uniform = linspace(0, dist[-1], fixed_size)
    x_new = interp(d_uniform, dist, x_val)
    y_new = interp(d_uniform, dist, y_val)
    return x_new, y_new


@njit(cache=True)
def flatten_line_matrix(l_matrix):
    """
    Flat matrix and add on between each line.

    :param l_matrix: matrix of position
    :return: array with nan between line
    """
    nb_line, sampling = l_matrix.shape
    final_size = (nb_line - 1) + nb_line * sampling
    empty_dataset = False
    if final_size < 1:
        empty_dataset = True
        final_size = 1
    out = empty(final_size, dtype=l_matrix.dtype)
    if empty_dataset:
        out[:] = nan
        return out
    inc = 0
    for i in range(nb_line):
        for j in range(sampling):
            out[inc] = l_matrix[i, j]
            inc += 1
        out[inc] = nan
        inc += 1
    return out


@njit(cache=True)
def simplify(x, y, precision=0.1):
    """
    Will remove all middle/end point which are closer than precision.

    :param array x:
    :param array y:
    :param float precision: if two points have distance inferior to precision with remove next point
    :return: (x,y)
    :rtype: (array,array)
    """
    precision2 = precision ** 2
    nb = x.shape[0]
    # will be True for value keep
    mask = ones(nb, dtype=bool_)
    for j in range(0, nb):
        x_previous, y_previous = x[j], y[j]
        if isnan(x_previous) or isnan(y_previous):
            mask[j] = False
            continue
        break
    # Only nan
    if j == (nb - 1):
        return zeros(0, dtype=x.dtype), zeros(0, dtype=x.dtype)

    last_nan = False
    for i in range(j + 1, nb):
        x_, y_ = x[i], y[i]
        if isnan(x_) or isnan(y_):
            if last_nan:
                mask[i] = False
            else:
                last_nan = True
            continue
        last_nan = False
        d_x = x_ - x_previous
        if d_x > precision:
            x_previous, y_previous = x_, y_
            continue
        d_y = y_ - y_previous
        if d_y > precision:
            x_previous, y_previous = x_, y_
            continue
        d2 = d_x ** 2 + d_y ** 2
        if d2 > precision2:
            x_previous, y_previous = x_, y_
            continue
        mask[i] = False
    new_nb = mask.sum()
    new_x, new_y = empty(new_nb, dtype=x.dtype), empty(new_nb, dtype=y.dtype)
    j = 0
    for i in range(nb):
        if mask[i]:
            new_x[j], new_y[j] = x[i], y[i]
            j += 1
    return new_x, new_y


@njit(cache=True)
def split_line(x, y, i):
    """
    Split x and y at each i change.

    :param x: array
    :param y: array
    :param i: array of int at each i change, we cut x, y

    :return: x and y separate by nan at each i jump
    """
    nb_jump = len(where(i[1:] - i[:-1] != 0)[0])
    nb_value = x.shape[0]
    final_size = (nb_jump - 1) + nb_value
    new_x = empty(final_size, dtype=x.dtype)
    new_y = empty(final_size, dtype=y.dtype)
    new_j = 0
    for j in range(nb_value):
        new_x[new_j] = x[j]
        new_y[new_j] = y[j]
        new_j += 1
        if j < (nb_value - 1) and i[j] != i[j + 1]:
            new_x[new_j] = nan
            new_y[new_j] = nan
            new_j += 1
    return new_x, new_y


@njit(cache=True)
def wrap_longitude(x, y, ref, cut=False):
    """
    Will wrap contiguous longitude with reference as west bound.

    :param array x:
    :param array y:
    :param float ref: longitude of reference, all the new value will be between ref and ref + 360
    :param bool cut: if True line will be cut at the bounds
    :return: lon,lat
    :rtype: (array,array)
    """
    if cut:
        indexs = list()
        nb = x.shape[0]
        new_previous = (x[0] - ref) % 360
        x_previous = x[0]
        for i in range(1, nb):
            x_ = x[i]
            new_x = (x_ - ref) % 360
            if not isnan(x_) and not isnan(x_previous):
                d_new = new_x - new_previous
                d = x_ - x_previous
                if abs(d - d_new) > 1e-5:
                    indexs.append(i)
            x_previous, new_previous = x_, new_x

        nb_indexs = len(indexs)
        new_size = nb + nb_indexs * 3
        out_x = empty(new_size, dtype=x.dtype)
        out_y = empty(new_size, dtype=y.dtype)
        i_ = 0
        j = 0
        for i in range(nb):
            if j < nb_indexs and i == indexs[j]:
                j += 1
                cor = 360 if x[i - 1] > x[i] else -360
                out_x[i + i_] = (x[i] - ref) % 360 + ref - cor
                out_y[i + i_] = y[i]
                out_x[i + i_ + 1] = nan
                out_y[i + i_ + 1] = nan
                out_x[i + i_ + 2] = (x[i - 1] - ref) % 360 + ref + cor
                out_y[i + i_ + 2] = y[i - 1]
                i_ += 3
            out_x[i + i_] = (x[i] - ref) % 360 + ref
            out_y[i + i_] = y[i]
        return out_x, out_y

    else:
        nb = x.shape[0]
        out = empty(nb, dtype=x.dtype)
        for i in range(nb):
            out[i] = (x[i] - ref) % 360 + ref
        return out, y


@njit(cache=True, fastmath=True)
def coordinates_to_local(lon, lat, lon0, lat0):
    """
    Take latlong coordinates to transform in local coordinates (in m).

    :param array x: coordinates to transform
    :param array y: coordinates to transform
    :param float lon0: longitude of local reference
    :param float lat0: latitude of local reference
    :return: x,y
    :retype: (array, array)
    """
    D2R = pi / 180.0
    R = 6370997
    dlon = (lon - lon0) * D2R
    sin_dlat = sin((lat - lat0) * 0.5 * D2R)
    sin_dlon = sin(dlon * 0.5)
    cos_lat0 = cos(lat0 * D2R)
    cos_lat = cos(lat * D2R)
    a_val = sin_dlon ** 2 * cos_lat0 * cos_lat + sin_dlat ** 2
    module = R * 2 * arctan2(a_val ** 0.5, (1 - a_val) ** 0.5)

    azimuth = pi / 2 - arctan2(
        cos_lat * sin(dlon),
        cos_lat0 * sin(lat * D2R) - sin(lat0 * D2R) * cos_lat * cos(dlon),
    )
    return module * cos(azimuth), module * sin(azimuth)


@njit(cache=True, fastmath=True)
def local_to_coordinates(x, y, lon0, lat0):
    """
    Take local coordinates (in m) to transform to latlong.

    :param array x: coordinates to transform
    :param array y: coordinates to transform
    :param float lon0: longitude of local reference
    :param float lat0: latitude of local reference
    :return: lon,lat
    :retype: (array, array)
    """
    D2R = pi / 180.0
    R = 6370997
    d = (x ** 2 + y ** 2) ** 0.5 / R
    a = -(arctan2(y, x) - pi / 2)
    lat = arcsin(sin(lat0 * D2R) * cos(d) + cos(lat0 * D2R) * sin(d) * cos(a))
    lon = (
        lon0
        + arctan2(
            sin(a) * sin(d) * cos(lat0 * D2R), cos(d) - sin(lat0 * D2R) * sin(lat)
        )
        / D2R
    )
    return lon, lat / D2R


@njit(cache=True, fastmath=True)
def nearest_grd_indice(x, y, x0, y0, xstep, ystep):
    """
    Get nearest grid indice from a position.

    :param x: longitude
    :param y: latitude
    :param float x0: first grid longitude
    :param float y0: first grid latitude
    :param float xstep: step between two longitude
    :param float ystep: step between two latitude
    """
    return (
        numba_types.int32(round(((x - x0[0]) % 360.0) / xstep)),
        numba_types.int32(round((y - y0[0]) / ystep)),
    )


@njit(cache=True)
def bbox_indice_regular(vertices, x0, y0, xstep, ystep, N, circular, x_size):
    """
    Get bbox indice of a contour in a regular grid.

    :param vertices: vertice of contour
    :param float x0: first grid longitude
    :param float y0: first grid latitude
    :param float xstep: step between two longitude
    :param float ystep: step between two latitude
    :param int N: shift of index to enlarge window
    :param bool circular: To know if grid is wrappable
    :param int x_size: Number of longitude
    """
    lon, lat = vertices[:, 0], vertices[:, 1]
    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()
    i_x0, i_y0 = nearest_grd_indice(lon_min, lat_min, x0, y0, xstep, ystep)
    i_x1, i_y1 = nearest_grd_indice(lon_max, lat_max, x0, y0, xstep, ystep)
    if circular:
        slice_x = (i_x0 - N) % x_size, (i_x1 + N + 1) % x_size
    else:
        slice_x = max(i_x0 - N, 0), i_x1 + N + 1
    slice_y = i_y0 - N, i_y1 + N + 1
    return slice_x, slice_y


def build_circle(x0, y0, r):
    """
    Build circle from center coordinates.

    :param float x0: center coordinate
    :param float y0: center coordinate
    :param float r: radius i meter
    :return: x,y
    :rtype: (array,array)
    """
    angle = radians(linspace(0, 360, 50))
    x_norm, y_norm = cos(angle), sin(angle)
    return x_norm * r + x0, y_norm * r + y0
