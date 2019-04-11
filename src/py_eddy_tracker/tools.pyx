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
cdef DTYPE_coord PI = 3.141592653589793
cdef DTYPE_coord EARTH_DIAMETER = 6371.3150 * 2


@wraparound(False)
@boundscheck(False)
def fit_circle_c(
    ndarray[DTYPE_coord] x_vec,
    ndarray[DTYPE_coord] y_vec
    ):
    """
    Fit the circle
    Adapted from ETRACK (KCCMC11)
    """
    cdef DTYPE_ui i_elt, i_start, i_end, nb_elt
    cdef DTYPE_coord x_mean, y_mean, scale, norme_max, center_x, center_y, radius
    cdef DTYPE_coord p_area, c_area, a_err, p_area_incirc, dist_poly
    nb_elt = x_vec.shape[0]

    cdef DTYPE_coord * p_inon_x = <DTYPE_coord * >malloc(nb_elt * sizeof(DTYPE_coord))
    if not p_inon_x:
        raise MemoryError()
    cdef DTYPE_coord * p_inon_y = <DTYPE_coord * >malloc(nb_elt * sizeof(DTYPE_coord))
    if not p_inon_y:
        raise MemoryError()

    x_mean = 0
    y_mean = 0
    
    for i_elt from 0 <= i_elt < nb_elt:
        x_mean += x_vec[i_elt]
        y_mean += y_vec[i_elt]
    y_mean /= nb_elt
    x_mean /= nb_elt
    
    norme = (x_vec - x_mean) ** 2 + (y_vec - y_mean) ** 2
    norme_max = norme.max()
    scale = norme_max ** .5

    # Form matrix equation and solve it
    # Maybe put f4
    datas = ones((nb_elt, 3), dtype='f8')
    for i_elt from 0 <= i_elt < nb_elt:
        datas[i_elt, 0] = 2. * (x_vec[i_elt] - x_mean) / scale
        datas[i_elt, 1] = 2. * (y_vec[i_elt] - y_mean) / scale
        
    (center_x, center_y, radius), _, _, _ = lstsq(datas, norme / norme_max, rcond=None)

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
    c_area = (radius ** 2) * PI
    
    # Find distance between circle center and contour points_inside_poly
    for i_elt from 0 <= i_elt < nb_elt:
        # Find distance between circle center and contour points_inside_poly
        dist_poly = ((x_vec[i_elt] - center_x) ** 2 + (y_vec[i_elt] - center_y) ** 2) ** .5
        # Indices of polygon points outside circle
        # p_inon_? : polygon x or y points inside & on the circle
        if dist_poly > radius:
            p_inon_y[i_elt] = center_y + radius * (y_vec[i_elt] - center_y) / dist_poly
            p_inon_x[i_elt] = center_x - (center_x - x_vec[i_elt]) * (center_y - p_inon_y[i_elt]) / (center_y - y_vec[i_elt])
        else:
            p_inon_x[i_elt] = x_vec[i_elt]
            p_inon_y[i_elt] = y_vec[i_elt]

    # Area of closed contour/polygon enclosed by the circle
    p_area_incirc = 0
    p_area = 0
    for i_elt from 0 <= i_elt < (nb_elt - 1):
        # Indices of polygon points outside circle
        # p_inon_? : polygon x or y points inside & on the circle
        p_area_incirc += p_inon_x[i_elt] * p_inon_y[1 + i_elt] - p_inon_x[i_elt + 1] * p_inon_y[i_elt]
        # Shape test
        # Area and centroid of closed contour/polygon
        p_area += x_vec[i_elt] * y_vec[1 + i_elt] - x_vec[1 + i_elt] * y_vec[i_elt]
    p_area = abs(p_area) * .5
    free(p_inon_x)
    free(p_inon_y)
    p_area_incirc = abs(p_area_incirc) * .5
    
    a_err = (c_area - 2 * p_area_incirc + p_area) * 100. / c_area
    return center_x, center_y, radius, a_err


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
