# -*- coding: utf-8 -*-
from cython import boundscheck, wraparound
from numpy cimport ndarray
from numpy import empty, ones
from numpy.linalg import lstsq
from numpy import float64 as float_coord
from libc.math cimport sin, cos, atan2
from libc.stdlib cimport malloc, free

ctypedef unsigned int DTYPE_ui
ctypedef double DTYPE_coord


cdef DTYPE_coord D2R = 0.017453292519943295
cdef DTYPE_coord EARTH_DIAMETER = 6371.3150 * 2


@wraparound(False)
@boundscheck(False)
cdef is_left(
        DTYPE_coord x_line_0,
        DTYPE_coord y_line_0,
        DTYPE_coord x_line_1,
        DTYPE_coord y_line_1,
        DTYPE_coord x_test,
        DTYPE_coord y_test,
        ):
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
    cdef DTYPE_coord product
    product = (x_line_1 - x_line_0) * (y_test - y_line_0
        ) - (x_test - x_line_0) * (y_line_1 - y_line_0)
    return product > 0


@wraparound(False)
@boundscheck(False)
def poly_contain_poly(
    ndarray[DTYPE_coord, ndim=2] xy_poly_out,
    ndarray[DTYPE_coord, ndim=2] xy_poly_in
    ):
    cdef DTYPE_ui i_elt, nb_elt
    cdef int wn = 0
    nb_elt = xy_poly_in.shape[0]
    for i_elt from 0 <= i_elt < (nb_elt):
        wn = winding_number_poly(
            xy_poly_in[i_elt, 0],
            xy_poly_in[i_elt, 1],
            xy_poly_out
            )
        if wn == 0:
            return False
    return True


@wraparound(False)
@boundscheck(False)
def winding_number_poly(
    DTYPE_coord x_test,
    DTYPE_coord y_test,
    ndarray[DTYPE_coord, ndim=2] xy_poly
    ):
    """
    http://geomalgorithms.com/a03-_inclusion.html
    wn_PnPoly(): winding number test for a point in a polygon
          Input:   P = a point,
                   V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
          Return:  wn = the winding number (=0 only when P is outside)
    """
    # the  winding number counter
    cdef int wn = 0

    cdef DTYPE_ui i_elt, nb_elt
    nb_elt = xy_poly.shape[0]

    # loop through all edges of the polygon
    for i_elt from 0 <= i_elt < (nb_elt):
        if i_elt + 1 == nb_elt:
            x_next = xy_poly[0, 0]
            y_next = xy_poly[0, 1]
        else:
            x_next = xy_poly[i_elt + 1, 0]
            y_next = xy_poly[i_elt + 1, 1]
        if xy_poly[i_elt, 1] <= y_test:
            if y_next > y_test:
                if is_left(xy_poly[i_elt, 0],
                           xy_poly[i_elt, 1],
                           x_next,
                           y_next,
                           x_test, y_test
                           ):
                    wn += 1
        else:
            if y_next <= y_test:
                if not is_left(xy_poly[i_elt, 0],
                               xy_poly[i_elt, 1],
                               x_next,
                               y_next,
                               x_test, y_test
                               ):
                    wn -= 1
    return wn


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
def distance_point_vector(
        DTYPE_coord lon0,
        DTYPE_coord lat0,
        ndarray[DTYPE_coord] lon1,
        ndarray[DTYPE_coord] lat1,
        ndarray[DTYPE_coord] dist,
        ):

    cdef DTYPE_coord sin_dlat, sin_dlon, cos_lat1, cos_lat2, a_val
    cdef DTYPE_ui i_elt, nb_elt
    nb_elt = lon1.shape[0]
    for i_elt from 0 <= i_elt < nb_elt:
        sin_dlat = sin((lat1[i_elt] - lat0) * 0.5 * D2R)
        sin_dlon = sin((lon1[i_elt] - lon0) * 0.5 * D2R)
        cos_lat1 = cos(lat0 * D2R)
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

    cdef DTYPE_coord sin_dlat, sin_dlon, cos_lat1, cos_lat2, a_val, dlon, dlat
    cdef DTYPE_ui i_elt0, i_elt1, nb_elt0, nb_elt1
    nb_elt0 = lon0.shape[0]
    nb_elt1 = lon1.shape[0]
    for i_elt0 from 0 <= i_elt0 < nb_elt0:
        for i_elt1 from 0 <= i_elt1 < nb_elt1:
            dlon = (lon1[i_elt1] - lon0[i_elt0] + 180) % 360 - 180
            if dlon > 20 or dlon < -20:
                continue
            dlat = lat1[i_elt1] - lat0[i_elt0]
            if dlat > 15 or dlat < -15:
                continue
            sin_dlat = sin(dlat * 0.5 * D2R)
            sin_dlon = sin(dlon * 0.5 * D2R)
            cos_lat1 = cos(lat0[i_elt0] * D2R)
            cos_lat2 = cos(lat1[i_elt1] * D2R)
            a_val = sin_dlon ** 2 * cos_lat1 * cos_lat2 + sin_dlat ** 2
            dist[i_elt0, i_elt1] = EARTH_DIAMETER * atan2(a_val ** 0.5, (1 - a_val) ** 0.5)


@wraparound(False)
@boundscheck(False)
def distance_matrix_vincenty(
        ndarray[DTYPE_coord] lon0,
        ndarray[DTYPE_coord] lat0,
        ndarray[DTYPE_coord] lon1,
        ndarray[DTYPE_coord] lat1,
        ndarray[DTYPE_coord, ndim=2] dist,
        ):

    cdef DTYPE_coord cos_dlon, sin_dlon, sin_lat1, sin_lat2, cos_lat1, cos_lat2, a_val, dlon, dlat, top, bottom
    cdef DTYPE_ui i_elt0, i_elt1, nb_elt0, nb_elt1
    nb_elt0 = lon0.shape[0]
    nb_elt1 = lon1.shape[0]
    for i_elt0 from 0 <= i_elt0 < nb_elt0:
        for i_elt1 from 0 <= i_elt1 < nb_elt1:
            dlon = (lon1[i_elt1] - lon0[i_elt0] + 180) % 360 - 180
            if dlon > 20 or dlon < -20:
                continue
            dlat = lat1[i_elt1] - lat0[i_elt0]
            if dlat > 15 or dlat < -15:
                continue
            sin_dlon = sin(dlon * 0.5 * D2R)
            cos_dlon = cos(dlon * 0.5 * D2R)
            sin_lat1 = cos(lat0[i_elt0] * D2R)
            sin_lat2 = cos(lat1[i_elt1] * D2R)
            cos_lat1 = cos(lat0[i_elt0] * D2R)
            cos_lat2 = cos(lat1[i_elt1] * D2R)
            top = (cos_lat2 * sin_lat1 - sin_lat2 * cos_lat1 * cos_dlon) ** 2 + cos_lat1 ** 2 * sin_dlon ** 2
            top = top ** .5
            bottom = sin_lat2 * sin_lat1 + cos_lat2 * cos_lat1 * cos_dlon
            dist[i_elt0, i_elt1] = EARTH_DIAMETER * atan2(top, bottom)


@wraparound(False)
@boundscheck(False)
cdef dist_array_size(
        DTYPE_ui i_start,
        DTYPE_ui nb_c_per_l,
        DTYPE_ui * nb_pt_per_c,
        DTYPE_ui * c_i,
        # return param
        DTYPE_ui * i_end,
        DTYPE_ui * c_start,
        DTYPE_ui * c_end
        ):
    """Give slice to select data
    """
    cdef DTYPE_ui i_elt
    i_end[0] = i_start + nb_c_per_l

    c_start[0] = c_i[i_start]
    c_end[0] = c_start[0]
    for i_elt from i_start <= i_elt < i_end[0]:
        c_end[0] += nb_pt_per_c[i_elt]


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
    """Get index from nearest path
    """
    cdef DTYPE_ui i_elt
    cdef DTYPE_ui nearesti, nb_contour

    cdef DTYPE_ui main_start, main_stop, start, end

    nb_contour = nb_c_per_l[level_index]
    if nb_contour == 0:
        return None

    main_start = l_i[level_index]
    dist_array_size(
        main_start,
        nb_contour,
        & nb_pt_per_c[0],
        & indices_of_first_pts[0],
        & main_stop,
        & start,
        & end
        )

    nearest_contour_index(
        & x_value[0],
        & y_value[0],
        xpt,
        ypt,
        start,
        end,
        &nearesti
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
        # return param
        DTYPE_ui * nearesti
        ):
    """Give index fron the nearest pts
    """
    cdef DTYPE_ui i_elt, i_ref
    cdef DTYPE_coord dist, dist_ref
    i_ref = start
    dist_ref = dist = (x_value[start] - xpt) ** 2 + (y_value[start] - ypt) ** 2
    for i_elt from start <= i_elt < end:
        dist = (x_value[i_elt] - xpt) ** 2 + (y_value[i_elt] - ypt) ** 2
        if dist < dist_ref:
            dist_ref = dist
            i_ref = i_elt
    nearesti[0] = i_ref - start


@wraparound(False)
@boundscheck(False)
def index_from_nearest_path_with_pt_in_bbox(
        DTYPE_ui level_index,
        ndarray[DTYPE_ui] l_i,
        ndarray[DTYPE_ui] nb_c_per_l,
        ndarray[DTYPE_ui] nb_pt_per_c,
        ndarray[DTYPE_ui] indices_of_first_pts,
        ndarray[DTYPE_coord] x_value,
        ndarray[DTYPE_coord] y_value,
        ndarray[DTYPE_coord] x_min_per_c,
        ndarray[DTYPE_coord] y_min_per_c,
        ndarray[DTYPE_coord] x_max_per_c,
        ndarray[DTYPE_coord] y_max_per_c,
        DTYPE_coord xpt,
        DTYPE_coord ypt,
        ):
    """Get index from nearest path in edge bbox contain pt
    """
    cdef DTYPE_ui i_start_c, i_end_c, i_elt_c, i_start_pt, i_end_pt, i_elt_pt
    cdef DTYPE_ui i_ref, nb_contour, find_contour
    cdef DTYPE_coord dist_ref, dist
    # Nb contour in level
    nb_contour = nb_c_per_l[level_index]
    if nb_contour == 0:
        return None
    # First contour in level
    i_start_c = l_i[level_index]
    # First contour of the next level
    i_end_c = i_start_c + nb_c_per_l[level_index]

    # Flag to check if we iterate
    find_contour = 0
    # We select the first pt of the first contour in the level
    # to initialize dist
    i_ref = i_start_c    
    i_start_pt = indices_of_first_pts[i_start_c]
    dist_ref = (x_value[i_start_pt] - xpt) ** 2 + (y_value[i_start_pt] - ypt) ** 2

    # We iterate over contour in the same level
    for i_elt_c from i_start_c <= i_elt_c < i_end_c:
        # if bbox of contour doesn't contain pt, we skip this contour
        if y_min_per_c[i_elt_c] > ypt:
            continue
        if y_max_per_c[i_elt_c] < ypt:
            continue
        if x_min_per_c[i_elt_c] > xpt:
            continue
        if x_max_per_c[i_elt_c] < xpt:
            continue
        # Indice of first pt of contour
        i_start_pt = indices_of_first_pts[i_elt_c]
        # Indice of first pt of the next contour
        i_end_pt = i_start_pt + nb_pt_per_c[i_elt_c]
        # We set flag to true, because we check contour
        find_contour = 1
        
        # We do iteration on pt to check dist, if it's inferior we store
        # index of contour
        for i_elt_pt from i_start_pt <= i_elt_pt < i_end_pt:
            dist = (x_value[i_elt_pt] - xpt) ** 2 + (y_value[i_elt_pt] - ypt) ** 2
            if dist < dist_ref:
                dist_ref = dist
                i_ref = i_elt_c            
    # No iteration on contour, we return no index of contour
    if find_contour == 0:
        return None
    # We return index of contour, for the specific level
    return i_ref - i_start_c
