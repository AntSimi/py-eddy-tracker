# -*- coding: utf-8 -*-
from cython import boundscheck, wraparound
from numpy cimport ndarray
from numpy import empty
from numpy import float64 as float_coord
from libc.math cimport sin, cos, atan2

ctypedef unsigned int DTYPE_ui
ctypedef double DTYPE_coord


cdef DTYPE_coord D2R = 0.017453292519943295


@wraparound(False)
@boundscheck(False)
def distance(
        DTYPE_coord lon0,
        DTYPE_coord lat0,
        DTYPE_coord lon1,
        DTYPE_coord lat1,
        ):

    cdef DTYPE_coord sin_dlat, sin_dlon, cos_lat1, cos_lat2, a_val
    sin_dlat = sin((lat1 - lat0) * 0.5 * D2R)
    sin_dlon = sin((lon1 - lon0) * 0.5 * D2R)
    cos_lat1 = cos(lat0 * D2R)
    cos_lat2 = cos(lat1 * D2R)
    a_val = sin_dlon ** 2 * cos_lat1 * cos_lat2 + sin_dlat ** 2
    return 6371315.0 * 2 * atan2(a_val ** 0.5, (1 - a_val) ** 0.5)


@wraparound(False)
@boundscheck(False)
def distance_vector(
        ndarray[DTYPE_coord] lon0,
        ndarray[DTYPE_coord] lat0,
        ndarray[DTYPE_coord] lon1,
        ndarray[DTYPE_coord] lat1,
        ndarray[DTYPE_coord] dist,
        ):

    cdef DTYPE_coord sin_dlat, sin_dlon, cos_lat1, cos_lat2, a_val
    cdef DTYPE_ui i_elt, nb_elt
    nb_elt = lon0.shape[0]
    for i_elt from 0 <= i_elt < nb_elt:
        sin_dlat = sin((lat1[i_elt] - lat0[i_elt]) * 0.5 * D2R)
        sin_dlon = sin((lon1[i_elt] - lon0[i_elt]) * 0.5 * D2R)
        cos_lat1 = cos(lat0[i_elt] * D2R)
        cos_lat2 = cos(lat1[i_elt] * D2R)
        a_val = sin_dlon ** 2 * cos_lat1 * cos_lat2 + sin_dlat ** 2
        dist[i_elt] = 6371315.0 * 2 * atan2(a_val ** 0.5, (1 - a_val) ** 0.5)


@wraparound(False)
@boundscheck(False)
def distance_matrix(
        ndarray[DTYPE_coord] lon0,
        ndarray[DTYPE_coord] lat0,
        ndarray[DTYPE_coord] lon1,
        ndarray[DTYPE_coord] lat1,
        ndarray[DTYPE_coord, ndim=2] dist,
        ):

    cdef DTYPE_coord sin_dlat, sin_dlon, cos_lat1, cos_lat2, a_val
    cdef DTYPE_ui i_elt0, i_elt1, nb_elt0, nb_elt1
    nb_elt0 = lon0.shape[0]
    nb_elt1 = lon1.shape[0]
    for i_elt0 from 0 <= i_elt0 < nb_elt0:
        for i_elt1 from 0 <= i_elt1 < nb_elt1:
            sin_dlat = sin((lat1[i_elt1] - lat0[i_elt0]) * 0.5 * D2R)
            sin_dlon = sin((lon1[i_elt1] - lon0[i_elt0]) * 0.5 * D2R)
            cos_lat1 = cos(lat0[i_elt0] * D2R)
            cos_lat2 = cos(lat1[i_elt1] * D2R)
            a_val = sin_dlon ** 2 * cos_lat1 * cos_lat2 + sin_dlat ** 2
            dist[i_elt0, i_elt1] = 6371315.0 * 2 * atan2(a_val ** 0.5, (1 - a_val) ** 0.5)


@wraparound(False)
@boundscheck(False)
cdef dist_array_size(
        DTYPE_ui l_i,
        DTYPE_ui nb_c_per_l,
        DTYPE_ui * nb_pt_per_c,
        DTYPE_ui * c_i
        ):
    cdef DTYPE_ui i_elt, i_start, i_end, nb_pts
    i_start = l_i
    i_end = i_start + nb_c_per_l
    nb_pts = 0

    for i_elt from i_start <= i_elt < i_end:
        nb_pts += nb_pt_per_c[i_elt]
    i_contour = c_i[i_start]
    return i_start, i_end, i_contour, i_contour + nb_pts


@wraparound(False)
@boundscheck(False)
def index_from_nearest_path(
        DTYPE_ui level_index,
        ndarray[DTYPE_ui] l_i,
        ndarray[DTYPE_ui] nb_c_per_l,
        ndarray[DTYPE_ui] nb_pt_per_c,
        ndarray[DTYPE_ui] indices_of_first_pts,
        ndarray[DTYPE_coord] x_value,
        ndarray[DTYPE_coord] y_value,
        DTYPE_coord xpt,
        DTYPE_coord ypt,
        ):
    cdef DTYPE_ui i_elt
    cdef DTYPE_ui nearesti, nb_contour

    cdef DTYPE_ui main_start, main_stop, start, end

    nb_contour = nb_c_per_l[level_index]
    if nb_contour == 0:
        return None

    main_start, main_stop, start, end = dist_array_size(
        l_i[level_index],
        nb_contour,
        & nb_pt_per_c[0],
        & indices_of_first_pts[0],
        )

    nearesti = nearest_contour_index(
        & y_value[0],
        & y_value[0],
        xpt,
        ypt,
        start,
        end,
        )

    for i_elt from main_start <= i_elt < main_stop:
        if (indices_of_first_pts[i_elt] -
                indices_of_first_pts[main_start]) > nearesti:
            return i_elt - 1 - main_start
    return i_elt - 1 - main_start


@wraparound(False)
@boundscheck(False)
cdef nearest_contour_index(
        DTYPE_coord * x_value,
        DTYPE_coord * y_value,
        DTYPE_coord xpt,
        DTYPE_coord ypt,
        DTYPE_ui start,
        DTYPE_ui end,
        ):
    cdef DTYPE_ui i_elt, i_ref
    cdef DTYPE_coord dist, dist_ref
    i_ref = start
    dist_ref = dist = (x_value[start] - xpt) ** 2 + (y_value[start] - ypt) ** 2
    for i_elt from start <= i_elt < end:
        dist = (x_value[i_elt] - xpt) ** 2 + (y_value[i_elt] - ypt) ** 2
        if dist < dist_ref:
            dist_ref = dist
            i_ref = i_elt

    return i_ref - start
