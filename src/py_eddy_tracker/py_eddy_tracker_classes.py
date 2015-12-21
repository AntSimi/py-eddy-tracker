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

Copyright (c) 2014-2015 by Evan Mason
Email: emason@imedea.uib-csic.es
===========================================================================

py_eddy_tracker_classes.py

Version 2.0.3
===========================================================================


"""
# External modules
import logging
import numpy as np
from datetime import datetime
from pyproj import Proj
from .make_eddy_tracker_list_obj import uniform_resample, nearest
from .py_eddy_tracker_property_classes import Amplitude, EddiesObservations
from .tools import distance, winding_number_poly, fit_circle_c
from matplotlib.path import Path as BasePath
from scipy.interpolate import griddata


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
    if np.isscalar(ocean_time):
        ocean_time = np.array([ocean_time])
    ocean_time /= 86400.
    year = np.floor(ocean_time / 360.)
    month = np.floor((ocean_time - year * 360.) / 30.)
    day = np.floor(ocean_time - year * 360. - month * 30.)
    year = (year.astype(np.int16) + 1)[0]
    month = (month.astype(np.int16) + 1)[0]
    day = (day.astype(np.int16) + 1)[0]
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


def okubo_weiss(grd):
    """
    Calculate the Okubo-Weiss parameter
    See e.g. http://ifisc.uib.es/oceantech/showOutput.php?idFile=61
    Returns: lambda2 - Okubo-Weiss parameter [s^-2]
             xi      - rel. vorticity [s^-1]
    Adapted from Roms_tools
    """
    def vorticity(u_val, v_val, d_x, d_y):
        """
        Returns vorticity calculated using np.gradient
        """
        d_x = 1 / d_x
        d_y = 1 / d_y
        u_y, _ = np.gradient(grd.u2rho_2d(u_val), d_x, d_y)
        _, v_x = np.gradient(grd.v2rho_2d(v_val), d_x, d_y)
        x_i = v_x - u_y
        return x_i

    u_val = grd.rho2u_2d(grd.u_val)
    v_val = grd.rho2v_2d(grd.v_val)
    p_m = grd.p_m()[grd.slice_j_unpad, grd.slice_i_unpad]
    p_n = grd.p_n()[grd.slice_j_unpad, grd.slice_i_unpad]

    uom = 2. * u_val / (p_m[:, :-1] + p_m[:, 1:])
    uon = 2. * u_val / (p_n[:, :-1] + p_n[:, 1:])
    von = 2. * v_val / (p_n[:-1] + p_n[1:])
    vom = 2. * v_val / (p_m[:-1] + p_m[1:])
    mn_product = p_m * p_n
    # Sigma_T
    s_t = mn_product * psi2rho(von[:, 1:] - von[:, :-1] + uom[1:] - uom[:-1])
    # Sigma_N
    s_n = np.zeros(p_m.shape)
    s_n[1:-1, 1:-1] = mn_product[1:-1, 1:-1] * (uon[1:-1, 1:] -
                                                uon[1:-1, :-1] -
                                                vom[1:, 1:-1] +
                                                vom[:-1, 1:-1])
    # Relative vorticity
    x_i = vorticity(u_val, v_val, p_m, p_n)
    # Okubo
    lambda2 = s_n**2 + s_t**2 - x_i**2
    return lambda2, x_i


def psi2rho(var_psi):
    # Convert a psi field to rho points
    var_rho = np.zeros((var_psi.shape[0] + 1, var_psi.shape[1] + 1))
    var_rho[1:-1, 1:-1] = quart_interp(var_psi[:-1, :-1], var_psi[:-1, 1:],
                                       var_psi[1:, :-1], var_psi[1:, 1:])
    var_rho[0] = var_rho[1]
    var_rho[-1] = var_rho[-2]
    var_rho[:, 0] = var_rho[:, 1]
    var_rho[:, -1] = var_rho[:, -2]
    return var_rho


def get_uavg(eddy, contours, centlon_e, centlat_e, poly_eff, grd,
             anticyclonic_search, save_all_uavg=False):
    """
    Calculate geostrophic speed around successive contours
    Returns the average

    If save_all_uavg == True we want uavg for every contour
    """
    points = np.array([grd.lon[eddy.slice_j, eddy.slice_i].ravel(),
                       grd.lat[eddy.slice_j, eddy.slice_i].ravel()]).T

    # First contour is the outer one (effective)
    # Copy ?
    theseglon, theseglat = (poly_eff.vertices[:, 0].copy(),
                            poly_eff.vertices[:, 1].copy())

    theseglon, theseglat = uniform_resample(
        theseglon, theseglat, method='akima')

    if 'RectBivariate' in eddy.interp_method:
        uavg = eddy.uspd_coeffs.ev(theseglat[1:], theseglon[1:]).mean()

    elif 'griddata' in eddy.interp_method:
        uspd1d = eddy.uspd[eddy.slice_j, eddy.slice_i].ravel()
        uavg = griddata(points, uspd1d,
                        (theseglon[1:], theseglat[1:]), 'linear')
        uavg = uavg[np.isfinite(uavg)].mean()
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

        # Get contour around centlon_e, centlat_e at level [collind:][iuavg]
        # theindex, poly_i = eddy_tracker.find_nearest_contour(
                        # c_s.collections[citer.index], centlon_e, centlat_e)

        # Leave loop if no contours at level citer.index
        theindex = eddy.swirl.get_index_nearest_path(
            corrected_coll_index, centlon_e, centlat_e)

        if theindex is None:
            continue

        poly_i = coll.get_paths()[theindex]

        # 1. Ensure polygon_i contains point centlon_e, centlat_e
        if winding_number_poly(centlon_e, centlat_e, poly_i.vertices) == 0:
            continue

        # 2. Ensure polygon_i is within polygon_e
        if not poly_eff.contains_path(poly_i):
            continue

        # 3. Respect size range
        mask_i_sum = poly_i.contains_points(points).sum()
        if not (mask_i_sum >= pixel_min and
                mask_i_sum <= eddy.pixel_threshold[1]):
            continue

        any_inner_contours = True

        seglon, seglat = (poly_i.vertices[:, 0], poly_i.vertices[:, 1])
        seglon, seglat = uniform_resample(seglon, seglat, method='interp1d')

        # Interpolate uspd to seglon, seglat, then get mean
        if 'RectBivariate' in eddy.interp_method:
            uavgseg = eddy.uspd_coeffs.ev(seglat[1:], seglon[1:]).mean()

        elif 'griddata' in eddy.interp_method:
            uavgseg = griddata(points, uspd1d, (seglon[1:], seglat[1:]),
                               'linear')
            uavgseg = uavgseg[np.isfinite(uavgseg)].mean()
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
        # proj = Proj('+proj=aeqd +lat_0=%s +lon_0=%s'
                    # % (theseglat.mean(), theseglon.mean()))
        # cx, cy = proj(theseglon, theseglat)
        # # cx, cy = eddy.m_val(theseglon, theseglat)
        # centx_s, centy_s, eddy_radius_s, junk = fit_circle(cx, cy)
        # centlon_s, centlat_s = proj(centx_s, centy_s, inverse=True)
        # # centlon_s, centlat_s = eddy.m_val(centx_s, centy_s, inverse=True)

    # else:  # use the effective contour
#         centlon_s, centlat_s = centlon_e, centlat_e
        # eddy_radius_s = eddy_radius_e
        inner_seglon, inner_seglat = theseglon, theseglat

    if not save_all_uavg:
        # return (uavg, centlon_s, centlat_s, eddy_radius_s,
                # theseglon, theseglat, inner_seglon, inner_seglat)
        return (uavg, theseglon, theseglat, inner_seglon, inner_seglat)

    else:
        return (uavg, theseglon, theseglat, inner_seglon, inner_seglat,
                all_uavg)


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
        self._circle_params = self._fit_circle_path()
    return self._circle_params


def _fit_circle_path(self):
    lon_mean, lat_mean = self.mean_coordinates()
    # Prepare for shape test and get eddy_radius_e
    # http://www.geo.hunter.cuny.edu/~jochen/gtech201/lectures/
    # lec6concepts/map%20coordinate%20systems/
    # how%20to%20choose%20a%20projection.htm
    proj = Proj('+proj=aeqd +lat_0=%s +lon_0=%s'
                % (lat_mean, lon_mean))

    c_x, c_y = proj(self.lon, self.lat)
    try:
        centlon_e, centlat_e, eddy_radius_e, aerr = fit_circle_c(c_x, c_y)
    except ZeroDivisionError:
        # Some time, edge is only a dot of few coordinates
        if len(np.unique(self.lon)) == 1 and len(np.unique(self.lat)) == 1:
            logging.warning('An edge is only define in one position')
            logging.debug('%d coordinates %s,%s', len(self.lon), self.lon,
                          self.lat)
            return 0, -90, np.nan, np.nan

    centlon_e, centlat_e = proj(centlon_e, centlat_e, inverse=True)
    centlon_e = (centlon_e - lon_mean + 180) % 360 + lon_mean - 180
    return centlon_e, centlat_e, eddy_radius_e, aerr

BasePath.fit_circle = fit_circle_path
BasePath._fit_circle_path = _fit_circle_path


def collection_loop(contours, grd, rtime, a_list_obj, c_list_obj,
                    x_i=None, c_s_xi=None):
    """
    Loop over each collection of contours
    """
    if a_list_obj is not None and c_list_obj is not None:
        raise Exception('Only one of this two parameters '
                        '(a_list_obj, c_list_obj) must be defined')
    if a_list_obj is not None:
        eddy = a_list_obj
    if c_list_obj is not None:
        eddy = c_list_obj

    if eddy.diagnostic_type not in ['Q', 'SLA']:
        raise Exception('Unknown Diagnostic : %s' % eddy.diagnostic_type)

    has_ts = False

    sign_type = eddy.sign_type
    anticyclonic_search = 'Anticyclonic' in sign_type
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
                if not (eddy_radius_e >= eddy.radmin and
                        eddy_radius_e <= eddy.radmax):
                    continue

            # Get indices of centroid
            # Give only 1D array of lon and lat not 2D data
            centi, centj = cont.nearest_grd_indice(centlon_e, centlat_e, grd)

            if 'Q' in eddy.diagnostic_type:
                if x_i[centj, centi] == eddy.fillval:
                    continue
            elif 'SLA' in eddy.diagnostic_type:
                if eddy.sla[centj, centi] != eddy.fillval:
                    acyc_not_cyc = (eddy.sla[centj, centi] >=
                                    contours.cvalues[corrected_coll_index])
                    if anticyclonic_search != acyc_not_cyc:
                        continue
                else:
                    continue

            # Instantiate new EddyObservation object
            properties = EddiesObservations(size=1)

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
                    # amplitude = np.abs(x_i[jmin:jmax,imin:imax
                    #                         ].flat[mask_eff]).max()
                    amplitude = np.abs(x_i[eddy.slice_j, eddy.slice_i
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
                        properties.obs['amplitude'] = amp.amplitude

            if properties.obs['amplitude'][0]:
                # Get sum of eke within Ceff
                teke = grd.eke[eddy.slice_j, eddy.slice_i][eddy.mask_eff].sum()
                if 'SLA' in eddy.diagnostic_type:
                    args = (eddy, contours, centlon_e, centlat_e, cont, grd,
                            anticyclonic_search)

                    if not eddy.track_extra_variables:
                        # (uavg, centlon_s, centlat_s,
                        # eddy_radius_s, contlon_s, contlat_s,
                        # inner_contlon, inner_contlat) = get_uavg(*args)
                        (uavg, contlon_s, contlat_s,
                         inner_contlon, inner_contlat) = get_uavg(*args)
                    else:
                        # (uavg, centlon_s, centlat_s,
                        # eddy_radius_s, contlon_s, contlat_s,
                        # inner_contlon, inner_contlat,
                        # uavg_profile) = get_uavg(*args, save_all_uavg=True)
                        (uavg, contlon_s, contlat_s,
                         inner_contlon, inner_contlat,
                         uavg_profile) = get_uavg(
                            *args, save_all_uavg=True)

                    # Use azimuth equal projection for radius
                    proj = Proj('+proj=aeqd +lat_0=%s +lon_0=%s'
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
                    _, _, eddy_radius_s, _ = fit_circle_c(c_x, c_y)

                properties.obs['radius_s'] = eddy_radius_s
                properties.obs['speed_radius'] = uavg
                properties.obs['radius_e'] = eddy_radius_e
                properties.obs['time'] = rtime
                properties.obs['eke'] = teke

                # Update SLA eddy properties
                if 'SLA' in eddy.diagnostic_type:

                    # See CSS11 section B4
                    properties.obs['lon'] = centlon_s
                    properties.obs['lat'] = centlat_s

                    if not has_ts:  # for AVISO
                        eddy.update_eddy_properties(properties)

                    # Mask out already found eddies
                    eddy.sla[eddy.slice_j, eddy.slice_i][
                        eddy.mask_eff] = eddy.fillval

    # Leave collection_loop
    return eddy


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
        var_ext = np.ma.zeros((n_j, n_i))  # add 1-more line parallell to
        var_ext[1:-1, 1:-1] = var        # each of 4-sides
        var_ext[1:-1, 0] = var[:, 0]   # duplicate W-side
        var_ext[1:-1, -1] = var[:, -1]  # duplicate E-side
        var_ext[0, 1:-1] = var[0]   # duplicate N-side
        var_ext[-1, 1:-1] = var[-1]  # duplicate S-side
        var_ext.mask[0, 0] = True     # NW-corner
        var_ext.mask[0, -1] = True     # NE-corner
        var_ext.mask[-1, 0] = True     # SW-corner
        var_ext.mask[-1, -1] = True     # SE-corner

        # npts is used to count number of valid neighbors
        npts = np.int(-var_ext.mask)

        # Replace nans with 0 to find a no-nan sum
        var_ext[var_ext.mask] = 0.

        # Initialize count and sum variables (maybe use some convolution)
        c_c = npts[1:-1, 1:-1] * (npts[0:-2, 1:-1] + npts[2:, 1:-1] +
                                  npts[1:-1, 0:-2] + npts[1:-1, 2:])

        var_s = (var_ext[0:-2, 1:-1] + var_ext[2:, 1:-1] +
                 var_ext[1:-1, 0:-2] + var_ext[1:-1, 2:])

        c_c[c_c == 0] = np.nan  # bring back nans in original data.
        weight = 8. - c_c  # This is the weight for values on each grid point,
#                                         based on number of valid neighbours
        # Final smoothed version of var
        hsm = 0.125 * (var_s + weight * var_ext[1:jsz+1, 1:isz+1])
        return hsm

    for _ in np.arange(numpasses):
        var[:] = hann2d_fast(var, n_i, n_j)

    return var