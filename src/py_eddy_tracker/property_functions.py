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

Copyright (c) 2014-2017 by Evan Mason and Antoine Delepoulle
Email: emason@imedea.uib-csic.es
===========================================================================

property_functions.py

Version 3.0.0
===========================================================================


"""
# External modules
import logging
from numpy import empty, linspace, interp, isscalar, floor, int16, array, \
    isfinite, unique, nan, ma, int_, arange
from datetime import datetime
from pyproj import Proj
from .tracking_objects import nearest
from .observations import EddiesObservations
from .property_objects import Amplitude
from .tools import winding_number_poly, fit_circle_c, poly_contain_poly
from matplotlib.path import Path as BasePath
from scipy.interpolate import griddata
from .generic import distance
from scipy.interpolate import Akima1DInterpolator


def uniform_resample(x_val, y_val, method='interp1d', extrapolate=None,
                     num_fac=2, fixed_size=None):
    """
    Resample contours to have (nearly) equal spacing
       x_val, y_val    : input contour coordinates
       num_fac : factor to increase lengths of output coordinates
       method  : currently only 'interp1d' or 'Akima'
                 (Akima is slightly slower, but may be more accurate)
       extrapolate : IS NOT RELIABLE (sometimes nans occur)
    """
    # Get distances
    dist = empty(x_val.shape)
    dist[0] = 0
    dist[1:] = distance(x_val[:-1], y_val[:-1], x_val[1:], y_val[1:])
    dist.cumsum(out=dist)
    # Get uniform distances
    if fixed_size is None:
        fixed_size = dist.size * num_fac
    d_uniform = linspace(0,
                         dist[-1],
                         num=fixed_size,
                         endpoint=True)

    # Do 1d interpolations
    if 'interp1d' == method:
        x_new = interp(d_uniform, dist, x_val)
        y_new = interp(d_uniform, dist, y_val)
    elif 'akima' == method:
        xfunc = Akima1DInterpolator(dist, x_val)
        yfunc = Akima1DInterpolator(dist, y_val)
        x_new = xfunc(d_uniform, extrapolate=extrapolate)
        y_new = yfunc(d_uniform, extrapolate=extrapolate)

    else:
        raise Exception()

    return x_new, y_new


def datestr2datetime(datestr):
    """
    Take strings with format YYYYMMDD and convert to datetime instance
    """
    return datetime.strptime(datestr, '%Y%m%d')


def oceantime2ymd(ocean_time, integer=False):
    """
    Return strings *year*, *month*, *day* given ocean_time (seconds)
    If kwarg *integer*==*True* return integers rather than strings.
    """
    if isscalar(ocean_time):
        ocean_time = array([ocean_time])
    ocean_time /= 86400.
    year = floor(ocean_time / 360.)
    month = floor((ocean_time - year * 360.) / 30.)
    day = floor(ocean_time - year * 360. - month * 30.)
    year = (year.astype(int16) + 1)[0]
    month = (month.astype(int16) + 1)[0]
    day = (day.astype(int16) + 1)[0]
    if not integer:
        year = str(year)
        if len(year) < 2:
            year = year.zfill(2)
        month = str(month)
        if len(month) < 2:
            month = month.zfill(2)
        day = str(day)
        if len(day) < 2:
            day = day.zfill(2)
    return year, month, day


def half_interp(h_1, h_2):
    """
    Speed up for frequent operations
    """
    return (h_1 + h_2) * .5


def quart_interp(h_1, h_2, h_3, h_4):
    """
    Speed up for frequent operations
    """
    return 0.25 * (h_1 + h_2 + h_3 + h_4)


def get_uavg(eddy, contours, centlon_e, centlat_e, poly_eff, grd,
             anticyclonic_search, save_all_uavg=False):
    """
    Calculate geostrophic speed around successive contours
    Returns the average

    If save_all_uavg == True we want uavg for every contour
    """
    points = array([grd.lon[eddy.slice_j, eddy.slice_i].ravel(),
                    grd.lat[eddy.slice_j, eddy.slice_i].ravel()]).T

    # First contour is the outer one (effective)
    # Copy ?
    theseglon, theseglat = (poly_eff.vertices[:, 0].copy(),
                            poly_eff.vertices[:, 1].copy())

    theseglon, theseglat = uniform_resample(
        theseglon, theseglat, method='interp1d')

    if 'RectBivariate' in eddy.interp_method:
        uavg = eddy.uspd_coeffs.ev(theseglat[1:], theseglon[1:]).mean()
    elif 'griddata' in eddy.interp_method:
        uspd1d = eddy.uspd[eddy.slice_j, eddy.slice_i].ravel()
        uavg = griddata(points, uspd1d,
                        (theseglon[1:], theseglat[1:]), 'linear')
        uavg = uavg[isfinite(uavg)].mean()
    else:
        raise Exception('Unknown interpolation method : %s'
                        % eddy.interp_method)

    if save_all_uavg:
        all_uavg = [uavg]
        pixel_min = 1  # iterate until 1 pixel
    else:
        # Iterate until pixel_threshold[0] number of pixels
        pixel_min = eddy.pixel_threshold[0]

    # Flag as True if contours found within effective contour
    any_inner_contours = False

    iterator = 1 if anticyclonic_search else -1
    nb_coll = len(contours.collections)
    # Iteration on each levels
    for coll_index, coll in enumerate(contours.collections[::iterator]):
        corrected_coll_index = coll_index
        if not anticyclonic_search:
            corrected_coll_index = nb_coll - coll_index - 1

        # Leave loop if no contours at level citer.index
        theindex = eddy.swirl.get_index_nearest_path_bbox_contain_pt(
            corrected_coll_index, centlon_e, centlat_e)
        if theindex is None:
            continue

        poly_i = coll.get_paths()[theindex]

        # 1. Ensure polygon_i contains point centlon_e, centlat_e
        if winding_number_poly(centlon_e, centlat_e, poly_i.vertices) == 0:
            continue

        # 2. Ensure polygon_i is within polygon_e
        if not poly_contain_poly(poly_eff.vertices, poly_i.vertices):
            continue

        # 3. Respect size range
        mask_i_sum = poly_i.contains_points(points).sum()
        if not (pixel_min <= mask_i_sum <= eddy.pixel_threshold[1]):
            continue
        any_inner_contours = True

        seglon, seglat = (poly_i.vertices[:, 0].copy(), poly_i.vertices[:, 1].copy())
        seglon, seglat = uniform_resample(seglon, seglat, method='interp1d')

        # Interpolate uspd to seglon, seglat, then get mean
        if 'RectBivariate' in eddy.interp_method:
            uavgseg = eddy.uspd_coeffs.ev(seglat[1:], seglon[1:]).mean()

        elif 'griddata' in eddy.interp_method:
            uavgseg = griddata(points, uspd1d, (seglon[1:], seglat[1:]),
                               'linear')
            uavgseg = uavgseg[isfinite(uavgseg)].mean()
        else:
            raise Exception()

        if save_all_uavg:
            all_uavg.append(uavgseg)

        if uavgseg >= uavg:
            uavg = uavgseg.copy()
            theseglon, theseglat = seglon.copy(), seglat.copy()

        inner_seglon, inner_seglat = (
            seglon.copy(), seglat.copy())

    if not any_inner_contours:
        # set speed based contour parameters
        inner_seglon, inner_seglat = theseglon, theseglat

    if not save_all_uavg:
        return (uavg, theseglon, theseglat,
                inner_seglon, inner_seglat, any_inner_contours)
    else:
        return (uavg, theseglon, theseglat,
                inner_seglon, inner_seglat, any_inner_contours, all_uavg)


@property
def isvalid(self):
    return False not in (self.vertices[0] == self.vertices[-1]
                         ) and len(self.vertices) > 2


BasePath.isvalid = isvalid


def mean_coordinates(self):
    return self.vertices.mean(axis=0)


BasePath.mean_coordinates = mean_coordinates


@property
def lon(self):
    return self.vertices[:, 0]


BasePath.lon = lon


@property
def lat(self):
    return self.vertices[:, 1]


BasePath.lat = lat


def nearest_grd_indice(self, lon_value, lat_value, grid):
    if not hasattr(self, '_grid_indices'):
        self._grid_indices = nearest(lon_value, lat_value,
                                     grid.lon[0], grid.lat[:, 0])
    return self._grid_indices


BasePath.nearest_grd_indice = nearest_grd_indice


def fit_circle_path(self):
    if not hasattr(self, '_circle_params'):
        self._fit_circle_path()
    return self._circle_params


def _fit_circle_path(self):
    lon_mean, lat_mean = self.mean_coordinates()
    # Prepare for shape test and get eddy_radius_e
    # http://www.geo.hunter.cuny.edu/~jochen/gtech201/lectures/
    # lec6concepts/map%20coordinate%20systems/
    # how%20to%20choose%20a%20projection.htm
    proj = Proj('+proj=aeqd +ellps=WGS84 +lat_0=%s +lon_0=%s'
                % (lat_mean, lon_mean))

    c_x, c_y = proj(self.lon, self.lat)
    try:
        centlon_e, centlat_e, eddy_radius_e, aerr = fit_circle_c(c_x, c_y)
        centlon_e, centlat_e = proj(centlon_e, centlat_e, inverse=True)
        centlon_e = (centlon_e - lon_mean + 180) % 360 + lon_mean - 180
        self._circle_params = centlon_e, centlat_e, eddy_radius_e, aerr
    except ZeroDivisionError:
        # Some time, edge is only a dot of few coordinates
        if len(unique(self.lon)) == 1 and len(unique(self.lat)) == 1:
            logging.warning('An edge is only define in one position')
            logging.debug('%d coordinates %s,%s', len(self.lon), self.lon,
                          self.lat)
            self._circle_params = 0, -90, nan, nan


BasePath.fit_circle = fit_circle_path
BasePath._fit_circle_path = _fit_circle_path


def collection_loop(contours, grd, eddy, x_i=None, c_s_xi=None):
    """
    Loop over each collection of contours
    """
    if eddy.diagnostic_type not in ['Q', 'SLA']:
        raise Exception('Unknown Diagnostic : %s' % eddy.diagnostic_type)

    anticyclonic_search = 'Anticyclonic' in eddy.sign_type
    iterator = 1 if anticyclonic_search else -1

    # Set contour coordinates and indices for calculation of
    # speed-based radius
    # swirl = SwirlSpeed(eddy, c_s)

    # Loop over each collection
    for coll_ind, coll in enumerate(contours.collections[::iterator]):
        corrected_coll_index = coll_ind
        if iterator == -1:
            corrected_coll_index = - coll_ind - 1

        contour_paths = coll.get_paths()
        nb_paths = len(contour_paths)
        if nb_paths > 0:
            logging.debug('doing collection %s, contour value %s, %d paths',
                          corrected_coll_index,
                          contours.cvalues[corrected_coll_index],
                          nb_paths)
        else:
            continue
        # Loop over individual c_s contours (i.e., every eddy in field)
        for cont in contour_paths:
            # Filter for closed contours

            # I don't understand the cost of this addition
            # ~ eddy.swirl.is_valid(eddy.swirl.level_index[corrected_coll_index] + i_cont)
            if not cont.isvalid:
                continue
            centlon_e, centlat_e, eddy_radius_e, aerr = cont.fit_circle()
            # Filter for shape: >35% (>55%) is not an eddy for Q (SLA)
            # A. : Shape error must be scalar
            if aerr < 0 or aerr > eddy.shape_error:
                continue

            # Get eddy_radius_e (NOTE: if Q, we overwrite
            # eddy_radius_e defined ~8 lines above)
            if 'Q' in eddy.diagnostic_type:
                xilon, xilat = c_s_xi.find_nearest_contour(
                    centlon_e, centlat_e, pixel=False)[3:5]
                eddy_radius_e = distance(centlon_e, centlat_e, xilon, xilat)
                if not (eddy.radmin <= eddy_radius_e <= eddy.radmax):
                    continue

            # Get indices of centroid
            # Give only 1D array of lon and lat not 2D data
            centi, centj = cont.nearest_grd_indice(centlon_e, centlat_e, grd)

            if 'Q' in eddy.diagnostic_type:
                if x_i[centj, centi] == eddy.fillval:
                    continue
            elif 'SLA' in eddy.diagnostic_type:
                if eddy.sla[centj, centi] != eddy.fillval:
                    # Test to know cyclone or anticyclone
                    acyc_not_cyc = (eddy.sla[centj, centi] >=
                                    contours.cvalues[corrected_coll_index])
                    if anticyclonic_search != acyc_not_cyc:
                        continue
                else:
                    continue

            # Instantiate new EddyObservation object
            properties = EddiesObservations(
                size=1,
                track_extra_variables=eddy.track_extra_variables,
                track_array_variables=eddy.track_array_variables_sampling,
                array_variables=eddy.track_array_variables
            )

            # Set indices to bounding box around eddy
            eddy.set_bounds(cont.lon, cont.lat, grd)

            # Set masked points within bounding box around eddy
            eddy.set_mask_eff(cont, grd)

            # sum(mask) between 8 and 1000, CSS11 criterion 2
            if eddy.check_pixel_count(eddy.mask_eff_sum):
                eddy.reshape_mask_eff(grd)
                # Resample the contour points for a more even
                # circumferential distribution
                contlon_e, contlat_e = uniform_resample(cont.lon, cont.lat)

                if 'Q' in eddy.diagnostic_type:
                    # KCCMC11
                    # Note, eddy amplitude == max(abs(vort/f)) within eddy,
                    amplitude = abs(x_i[eddy.slice_j, eddy.slice_i
                                    ][eddy.mask_eff]).max()

                elif 'SLA' in eddy.diagnostic_type:

                    # Instantiate Amplitude object
                    amp = Amplitude(contlon_e, contlat_e, eddy, grd)

                    if anticyclonic_search:
                        reset_centroid = amp.all_pixels_above_h0(
                            contours.levels[corrected_coll_index])

                    else:
                        reset_centroid = amp.all_pixels_below_h0(
                            contours.levels[corrected_coll_index])

                    if reset_centroid:
                        centi = reset_centroid[0]
                        centj = reset_centroid[1]
                        centlon_e = grd.lon[centj, centi]
                        centlat_e = grd.lat[centj, centi]

                    if amp.within_amplitude_limits():
                        properties.obs['amplitude'] = amp.amplitude / 100

            if properties.obs['amplitude'][0]:
                # Get sum of eke within Ceff
                teke = grd.eke[eddy.slice_j, eddy.slice_i][eddy.mask_eff].sum()
                if 'SLA' in eddy.diagnostic_type:
                    args = (eddy, contours, centlon_e, centlat_e, cont, grd,
                            anticyclonic_search)

                    # ~ if eddy.track_array_variables > 0:

                    # ~ if not eddy.track_extra_variables:
                    if True:
                        (uavg, contlon_s, contlat_s,
                         inner_contlon, inner_contlat,
                         any_inner_contours
                         ) = get_uavg(*args)
                        # ~ else:
                        # ~ (uavg, contlon_s, contlat_s,
                        # ~ inner_contlon, inner_contlat,
                        # ~ any_inner_contours, uavg_profile
                        # ~ ) = get_uavg(
                        # ~ *args, save_all_uavg=True)

                    # Use azimuth equal projection for radius
                    proj = Proj('+proj=aeqd +ellps=WGS84 +lat_0=%s +lon_0=%s'
                                % (inner_contlat.mean(),
                                   inner_contlon.mean()))

                    # First, get position based on innermost
                    # contour
                    c_x, c_y = proj(inner_contlon, inner_contlat)
                    centx_s, centy_s, _, _ = fit_circle_c(c_x, c_y)
                    centlon_s, centlat_s = proj(centx_s, centy_s,
                                                inverse=True)
                    # Second, get speed-based radius based on
                    # contour of max uavg
                    # (perhaps we should make a new proj here
                    # based on contlon_s, contlat_s but I'm not
                    # sure it's that important ... Antoine?)
                    # A. : I dont think, the difference is tiny
                    c_x, c_y = proj(contlon_s, contlat_s)
                    _, _, eddy_radius_s, aerr_s = fit_circle_c(c_x, c_y)

                properties.obs['radius_s'] = eddy_radius_s
                properties.obs['speed_radius'] = uavg
                properties.obs['radius_e'] = eddy_radius_e
                # properties.obs['eke'] = teke
                if 'shape_error_e' in eddy.track_extra_variables:
                    properties.obs['shape_error_e'] = aerr
                if 'shape_error_s' in eddy.track_extra_variables:
                    properties.obs['shape_error_s'] = aerr_s

                if aerr > 99.9 or aerr_s > 99.9:
                    logging.warning(
                        'Strange shape at this step! shape_error : %f, %f',
                        aerr,
                        aerr_s)
                    continue

                # Update SLA eddy properties
                if 'SLA' in eddy.diagnostic_type:

                    # See CSS11 section B4
                    properties.obs['lon'] = centlon_s
                    properties.obs['lat'] = centlat_s
                    if 'contour_lon_e' in eddy.track_array_variables:
                        (properties.obs['contour_lon_e'],
                         properties.obs['contour_lat_e']) = uniform_resample(
                            cont.lon, cont.lat,
                            fixed_size=eddy.track_array_variables_sampling)
                    if 'contour_lon_s' in eddy.track_array_variables:
                        (properties.obs['contour_lon_s'],
                         properties.obs['contour_lat_s']) = uniform_resample(
                            contlon_s, contlat_s,
                            fixed_size=eddy.track_array_variables_sampling)

                    # for AVISO
                    eddy.update_eddy_properties(properties)

                    # Mask out already found eddies
                    eddy.sla[eddy.slice_j, eddy.slice_i][
                        eddy.mask_eff] = eddy.fillval


def func_hann2d_fast(var, numpasses):
    """
    The treament of edges and corners in func_HANN2D can be mimicked by
    duplicating sides on to the W,E,S,N of the respective edges and
    replacing corner points with NaN's. Then smoothing can be done in
    a single step.
    Adapted from equivalent function in ETRACK.
    """
    jsz, isz = var.shape
    # Interior points  (N,S,W,E)
    n_j = jsz + 2
    n_i = isz + 2

    def hann2d_fast(var, n_i, n_j):
        var_ext = ma.zeros((n_j, n_i))  # add 1-more line parallell to
        var_ext[1:-1, 1:-1] = var  # each of 4-sides
        var_ext[1:-1, 0] = var[:, 0]  # duplicate W-side
        var_ext[1:-1, -1] = var[:, -1]  # duplicate E-side
        var_ext[0, 1:-1] = var[0]  # duplicate N-side
        var_ext[-1, 1:-1] = var[-1]  # duplicate S-side
        var_ext.mask[0, 0] = True  # NW-corner
        var_ext.mask[0, -1] = True  # NE-corner
        var_ext.mask[-1, 0] = True  # SW-corner
        var_ext.mask[-1, -1] = True  # SE-corner

        # npts is used to count number of valid neighbors
        npts = int_(-var_ext.mask)

        # Replace nans with 0 to find a no-nan sum
        var_ext[var_ext.mask] = 0.

        # Initialize count and sum variables (maybe use some convolution)
        c_c = npts[1:-1, 1:-1] * (npts[0:-2, 1:-1] + npts[2:, 1:-1] +
                                  npts[1:-1, 0:-2] + npts[1:-1, 2:])

        var_s = (var_ext[0:-2, 1:-1] + var_ext[2:, 1:-1] +
                 var_ext[1:-1, 0:-2] + var_ext[1:-1, 2:])

        c_c[c_c == 0] = nan  # bring back nans in original data.
        weight = 8. - c_c  # This is the weight for values on each grid point,
        #                                         based on number of valid neighbours
        # Final smoothed version of var
        hsm = 0.125 * (var_s + weight * var_ext[1:jsz + 1, 1:isz + 1])
        return hsm

    for _ in arange(numpasses):
        var[:] = hann2d_fast(var, n_i, n_j)

    return var
