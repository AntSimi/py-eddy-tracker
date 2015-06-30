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
from netCDF4 import Dataset
from datetime import datetime
from pyproj import Proj
from .make_eddy_tracker_list_obj import uniform_resample, nearest
from .py_eddy_tracker_property_classes import Amplitude, EddyProperty
from .tools import distance, distance_matrix
from scipy.sparse import coo_matrix
from scipy.interpolate import griddata


def datestr2datetime(datestr):
    """
    Take strings with format YYYYMMDD and convert to datetime instance
    """
    return datetime.strptime(datestr, '%Y%m%d')


def gaussian_resolution(res, zwl, mwl):
    """
    Get parameters for ndimage.gaussian_filter
    See http://stackoverflow.com/questions/14531072/
    how-to-count-bugs-in-an-image
    Input: res : grid resolution in degrees
           zwl : zonal distance in degrees
           mwl : meridional distance in degrees
    """
    zres = zwl.copy()
    mres = mwl.copy()
    zres *= 0.125
    mres *= 0.125
    zres /= res
    mres /= res
    return zres, mres


def oceantime2ymd(ocean_time, integer=False):
    """
    Return strings *yea*r, *month*, *day* given ocean_time (seconds)
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
    h_1 += h_2
    h_1 *= 0.5
    return h_1


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
    p_m = grd.p_m()[grd.jup0:grd.jup1, grd.iup0:grd.iup1]
    p_n = grd.p_n()[grd.jup0:grd.jup1, grd.iup0:grd.iup1]

    p_m_x, p_m_y = p_m.shape
    p_m_y_reduce = p_m_y - 1
    p_m_x_reduce = p_m_x - 1
    uom = 2. * u_val / (p_m[:, :p_m_y_reduce] + p_m[:, 1:p_m_y])
    uon = 2. * u_val / (p_n[:, :p_m_y_reduce] + p_n[:, 1:p_m_y])
    von = 2. * v_val / (p_n[:p_m_x_reduce] + p_n[1:p_m_x])
    vom = 2. * v_val / (p_m[:p_m_x_reduce] + p_m[1:p_m_x])
    mn_product = p_m * p_n
    # Sigma_T
    s_t = mn_product * psi2rho(von[:, 1:p_m_y] - von[:, :p_m_y_reduce]
                               + uom[1:p_m_x, :] - uom[:p_m_x_reduce, :])
    # Sigma_N
    s_n = np.zeros((p_m_x, p_m_y))
    s_n[1:-1, 1:-1] = mn_product[1:-1, 1:-1] * (uon[1:-1, 1:]
                                                - uon[1:-1, :-1]
                                                - vom[1:, 1:-1]
                                                + vom[:-1, 1:-1])
    # Relative vorticity
    x_i = vorticity(u_val, v_val, p_m, p_n)
    # Okubo
    lambda2 = s_n**2
    lambda2 += s_t**2
    lambda2 -= x_i**2
    return lambda2, x_i


def psi2rho(var_psi):
    # Convert a psi field to rho points
    var_psi_x, var_psi_y = var_psi.shape
    var_psi_x_increase = var_psi_x + 1
    var_psi_y_increase = var_psi_y + 1
    var_psi_x_reduce = var_psi_x - 1
    var_psi_y_reduce = var_psi_y - 1
    var_rho = np.zeros((var_psi_x_increase, var_psi_y_increase))
    var_rho[1:var_psi_x, 1:var_psi_y] = quart_interp(
        var_psi[:var_psi_x_reduce, :var_psi_y_reduce],
        var_psi[:var_psi_x_reduce, 1:var_psi_y],
        var_psi[1:var_psi_x, :var_psi_y_reduce],
        var_psi[1:var_psi_x, 1:var_psi_y])
    var_rho[0] = var_rho[1]
    var_rho[var_psi_x] = var_rho[var_psi_x_reduce]
    var_rho[:, 0] = var_rho[:, 1]
    var_rho[:, var_psi_y] = var_rho[:, var_psi_y_reduce]
    return var_rho


def fit_circle(xvec, yvec):
    """
    Fit the circle
    Adapted from ETRACK (KCCMC11)
    """
    xvec, yvec = xvec.copy(), yvec.copy()

    # Moyenne
    xmean = xvec.mean()
    ymean = yvec.mean()
    # On recentre les données
    xsc = xvec - xmean
    ysc = yvec - ymean

    # On normalise
    norme = xsc ** 2 + ysc ** 2
    scale = norme.max() ** .5

    xsc /= scale
    ysc /= scale

    # Form matrix equation and solve it
    # Maybe put f4
    datas = np.ones((xvec.size, 3), dtype='f8')
    datas[:, 0] = 2. * xsc
    datas[:, 1] = 2. * ysc
    x_value, residues, rank, singular_value = np.linalg.lstsq(
        datas,
        norme / (scale ** 2))

    # Unscale data and get circle variables
    x_value[2] += x_value[0] ** 2 + x_value[1] ** 2
    x_value[2] **= .5
    x_value *= scale
    x_value[0] += xmean
    x_value[1] += ymean

    ccx = x_value[0]  # center X-position of fitted circle
    ccy = x_value[1]  # center Y-position of fitted circle
    radius = x_value[2]  # radius of fitted circle
    carea = radius**2 * np.pi  # area of fitted circle

    # Shape test
    # Area and centroid of closed contour/polygon
    polar = (xvec[:-1] * yvec[1:] - xvec[1:] * yvec[:-1]).sum()
    polar *= 0.5
    parea = np.abs(polar)

    # Find distance between circle center and contour points_inside_poly
    dist_poly = ((xvec - ccx) ** 2 + (yvec - ccy) ** 2) ** .5

    sintheta = (yvec - ccy) / dist_poly
    ptmp_y = ccy + radius * sintheta
    ptmp_x = ccx - (ccx - xvec) * (ccy - ptmp_y) / (ccy - yvec)

    # Indices of polygon points outside circle
    # p_inon_? : polygon x or y points inside & on the circle
    pout_id = (dist_poly > radius).nonzero()

    p_inon_x = xvec  # init
    p_inon_y = yvec  # init
    p_inon_x[pout_id] = ptmp_x[pout_id]
    p_inon_y[pout_id] = ptmp_y[pout_id]

    # Area of closed contour/polygon enclosed by the circle
    tmp = p_inon_x[:-1] * p_inon_y[1:] - p_inon_x[1:] * p_inon_y[:-1]

    parea_incirc = np.abs(tmp.sum()) * 0.5
    aerr = (1 - parea_incirc / carea) + (parea - parea_incirc) / carea
    aerr *= 100.

    return ccx, ccy, radius, aerr


def get_uavg(Eddy, CS, collind, centlon_e, centlat_e, poly_eff,
             grd, eddy_radius_e, save_all_uavg=False):
    """
    Calculate geostrophic speed around successive contours
    Returns the average

    If save_all_uavg == True we want uavg for every contour
    """
    # Unpack indices for convenience
    imin, imax, jmin, jmax = Eddy.imin, Eddy.imax, Eddy.jmin, Eddy.jmax

    points = np.array([grd.lon[jmin:jmax, imin:imax].ravel(),
                       grd.lat[jmin:jmax, imin:imax].ravel()]).T

    # First contour is the outer one (effective)
    theseglon, theseglat = (poly_eff.vertices[:, 0].copy(),
                            poly_eff.vertices[:, 1].copy())

    theseglon, theseglat = uniform_resample(
        theseglon, theseglat, method='akima')

    if 'RectBivariate' in Eddy.INTERP_METHOD:
        uavg = Eddy.uspd_coeffs.ev(theseglat[1:], theseglon[1:]).mean()

    elif 'griddata' in Eddy.INTERP_METHOD:
        uspd1d = Eddy.uspd[jmin:jmax, imin:imax].ravel()
        uavg = griddata(points, uspd1d,
                        (theseglon[1:], theseglat[1:]), 'linear')
        uavg = uavg[np.isfinite(uavg)].mean()
    else:
        Exception

    if save_all_uavg:
        all_uavg = [uavg]
        pixel_min = 1  # iterate until 1 pixel

    else:
        # Iterate until PIXEL_THRESHOLD[0] number of pixels
        pixel_min = Eddy.PIXEL_THRESHOLD[0]

    # Flag as True if contours found within effective contour
    any_inner_contours = False

    citer = np.nditer(CS.cvalues, flags=['c_index'])

    while not citer.finished:
        # Get contour around centlon_e, centlat_e at level [collind:][iuavg]
        # theindex, poly_i = eddy_tracker.find_nearest_contour(
                        # CS.collections[citer.index], centlon_e, centlat_e)

        # Leave loop if no contours at level citer.index
        theindex = Eddy.swirl.get_index_nearest_path(citer.index,
                                                     centlon_e, centlat_e)

        if theindex is None:
            citer.iternext()
            continue

        poly_i = CS.collections[citer.index].get_paths()[theindex]

        # 1. Ensure polygon_i contains point centlon_e, centlat_e
        if poly_i.contains_point([centlon_e, centlat_e]):

            # 2. Ensure polygon_i is within polygon_e
            if poly_eff.contains_path(poly_i):

                # 3. Respect size range
                mask_i_sum = poly_i.contains_points(points).sum()
                if (mask_i_sum >= pixel_min and
                        mask_i_sum <= Eddy.PIXEL_THRESHOLD[1]):

                    any_inner_contours = True

                    seglon, seglat = (poly_i.vertices[:, 0],
                                      poly_i.vertices[:, 1])
                    seglon, seglat = uniform_resample(
                        seglon, seglat, method='interp1d')

                    # Interpolate uspd to seglon, seglat, then get mean
                    if 'RectBivariate' in Eddy.INTERP_METHOD:
                        uavgseg = Eddy.uspd_coeffs.ev(seglat[1:],
                                                      seglon[1:]).mean()

                    elif 'griddata' in Eddy.INTERP_METHOD:
                        uavgseg = griddata(
                            points, uspd1d,
                            (seglon[1:], seglat[1:]),
                            'linear')
                        uavgseg = uavgseg[np.isfinite(uavgseg)].mean()
                    else:
                        Exception

                    if save_all_uavg:
                        all_uavg.append(uavgseg)

                    if uavgseg >= uavg:
                        uavg = uavgseg.copy()
                        theseglon, theseglat = seglon.copy(), seglat.copy()

                    inner_seglon, inner_seglat = (
                        seglon.copy(), seglat.copy())

        citer.iternext()

    if not any_inner_contours:
        # set speed based contour parameters
        # proj = Proj('+proj=aeqd +lat_0=%s +lon_0=%s'
                    # % (theseglat.mean(), theseglon.mean()))
        # cx, cy = proj(theseglon, theseglat)
        # # cx, cy = Eddy.m_val(theseglon, theseglat)
        # centx_s, centy_s, eddy_radius_s, junk = fit_circle(cx, cy)
        # centlon_s, centlat_s = proj(centx_s, centy_s, inverse=True)
        # # print 'fffffffffff', centlon_s, centlat_s
        # # centlon_s, centlat_s = Eddy.m_val(centx_s, centy_s, inverse=True)

        # '''
        # # Debug
        # plt.figure()
        # plt.contour(Eddy.grd.lon(), Eddy.grd.lat(), Eddy.slacopy, Eddy.CONTOUR_PARAMETER,
                    # colors='k', linestyles='solid', linewidths=0.25)
        # plt.scatter(centlon_s, centlat_s, c='k', edgecolors='none', s=100)
        # plt.plot(theseglon, theseglat, 'r')
        # plt.axis('image')
        # plt.show()
        # '''

    # else:  # use the effective contour
        centlon_s, centlat_s = centlon_e, centlat_e
        # eddy_radius_s = eddy_radius_e
        inner_seglon, inner_seglat = theseglon, theseglat

    if not save_all_uavg:
        # return (uavg, centlon_s, centlat_s, eddy_radius_s,
                # theseglon, theseglat, inner_seglon, inner_seglat)
        return (uavg, theseglon, theseglat, inner_seglon, inner_seglat)

    else:
        return (uavg, theseglon, theseglat, inner_seglon, inner_seglat,
                all_uavg)


def collection_loop(c_s, grd, rtime, a_list_obj, c_list_obj,
                    x_i=None, c_s_xi=None, sign_type='None'):
    """
    Loop over each collection of contours
    """
    if a_list_obj is not None:
        eddy = a_list_obj
    if c_list_obj is not None:
        eddy = c_list_obj

    if 'ROMS' in eddy.product:
        # has_ts = True
        has_ts = False
    elif 'AVISO' in eddy.product:
        has_ts = False
    else:
        Exception  # unknown product

    # Set contour coordinates and indices for calculation of
    # speed-based radius
    # swirl = SwirlSpeed(eddy, c_s)

    # Loop over each collection
    for collind, coll in enumerate(c_s.collections):

        logging.debug('doing collection %s, contour value %s',
                      collind, c_s.cvalues[collind])

        # Loop over individual c_s contours (i.e., every eddy in field)
        for cont in coll.get_paths():

            contlon_e, contlat_e = (cont.vertices[:, 0].copy(),
                                    cont.vertices[:, 1].copy())

            # Filter for closed contours
            if np.alltrue([contlon_e[0] == contlon_e[-1],
                           contlat_e[0] == contlat_e[-1],
                           contlon_e.ptp(),
                           contlat_e.ptp()]):

                # Instantiate new EddyProperty object
                properties = EddyProperty()

                # Prepare for shape test and get eddy_radius_e
                # http://www.geo.hunter.cuny.edu/~jochen/gtech201/lectures/lec6concepts/map%20coordinate%20systems/how%20to%20choose%20a%20projection.htm
                lat_mean, lon_mean = contlat_e.mean(), contlon_e.mean()
                proj = Proj('+proj=aeqd +lat_0=%s +lon_0=%s'
                            % (lat_mean, lon_mean))

                c_x, c_y = proj(contlon_e, contlat_e)

                centlon_e, centlat_e, eddy_radius_e, aerr = fit_circle(c_x,
                                                                       c_y)
                aerr = np.atleast_1d(aerr)

                # Filter for shape: >35% (>55%) is not an eddy for Q (SLA)
                if aerr >= 0. and aerr <= eddy.SHAPE_ERROR[collind]:

                    # Get centroid in lon lat
                    centlon_e, centlat_e = proj(centlon_e, centlat_e,
                                                inverse=True)
                    centlon_e = (centlon_e - lon_mean + 180
                                 ) % 360 + lon_mean - 180

                    # For some reason centlat_e is transformed
                    # by projtran to 'float'...
                    centlon_e, centlat_e = (np.float64(centlon_e),
                                            np.float64(centlat_e))

                    # Get eddy_radius_e (NOTE: if Q, we overwrite
                    # eddy_radius_e defined ~8 lines above)
                    if 'Q' in eddy.DIAGNOSTIC_TYPE:
                        xilon, xilat = c_s_xi.find_nearest_contour(
                            centlon_e, centlat_e, pixel=False)[3:5]
                        eddy_radius_e = distance(centlon_e, centlat_e,
                                                  xilon, xilat)
                        if (eddy_radius_e >= eddy.radmin and
                                eddy_radius_e <= eddy.radmax):
                            proceed0 = True
                        else:
                            proceed0 = False

                    elif 'SLA' in eddy.DIAGNOSTIC_TYPE:
                        # If 'SLA' is defined we filter below with pixel count
                        proceed0 = True

                    else:
                        raise Exception
                    if proceed0:

                        # Get indices of centroid
                        # Give only 1D array of lon and lat not 2D data
                        centi, centj = nearest(centlon_e, centlat_e,
                                               grd.lon[0], grd.lat[:, 0])

                        if 'Q' in eddy.DIAGNOSTIC_TYPE:

                            if x_i[centj, centi] != eddy.fillval:
                                proceed1 = True
                            else:
                                proceed1 = False

                        elif 'SLA' in eddy.DIAGNOSTIC_TYPE:

                            if eddy.sla[centj, centi] != eddy.fillval:
                                acyc_not_cyc = (eddy.sla[centj, centi] >=
                                                c_s.cvalues[collind])
                                if ('Anticyclonic' in sign_type and
                                        acyc_not_cyc):
                                    proceed1 = True
                                elif ('Cyclonic' in sign_type and not
                                      acyc_not_cyc):
                                    proceed1 = True
                                else:
                                    proceed1 = False  # no eddy
                            else:
                                proceed1 = False
                        else:
                            raise Exception

                        if proceed1:

                            # Set indices to bounding box around eddy
                            eddy.set_bounds(contlon_e, contlat_e, grd)

                            # Unpack indices for convenience
                            imin, imax, jmin, jmax = (eddy.imin, eddy.imax,
                                                      eddy.jmin, eddy.jmax)

                            # Set masked points within bounding box around eddy
                            eddy.set_mask_eff(cont, grd)

                            # sum(mask) between 8 and 1000, CSS11 criterion 2
                            if (eddy.mask_eff_sum >= eddy.PIXEL_THRESHOLD[0] and
                                    eddy.mask_eff_sum <= eddy.PIXEL_THRESHOLD[1]):

                                eddy.reshape_mask_eff(grd)

                                # Resample the contour points for a more even
                                # circumferential distribution
                                contlon_e, contlat_e = \
                                    uniform_resample(
                                        contlon_e, contlat_e)

                                if 'Q' in eddy.DIAGNOSTIC_TYPE:
                                    # Note, eddy amplitude == max(abs(vort/f)) within eddy, KCCMC11
                                    # amplitude = np.abs(x_i[jmin:jmax,imin:imax].flat[mask_eff]).max()
                                    amplitude = np.abs(x_i[jmin:jmax, imin:imax
                                                          ][eddy.mask_eff]
                                                       ).max()

                                elif 'SLA' in eddy.DIAGNOSTIC_TYPE:

                                    # Instantiate Amplitude object
                                    amp = Amplitude(contlon_e, contlat_e,
                                                    eddy, grd)

                                    if 'Anticyclonic' in sign_type:
                                        reset_centroid = amp.all_pixels_above_h0(
                                            c_s.levels[collind])
                                        # plt.figure(666)
                                        # plt.pcolormesh(eddy.mask_eff)
                                        # plt.show()
                                        # amp.debug_figure(grd)

                                    elif 'Cyclonic' in sign_type:
                                        reset_centroid = amp.all_pixels_below_h0(
                                            c_s.levels[collind])
                                    else:
                                        raise Exception()

                                    if reset_centroid:
                                        centi = reset_centroid[0]
                                        centj = reset_centroid[1]
                                        centlon_e = grd.lon[centj, centi]
                                        centlat_e = grd.lat[centj, centi]

                                    # amp.debug_figure(grd)

                                    if amp.within_amplitude_limits():
                                        properties.amplitude = amp.amplitude

                            if properties.amplitude:

                                # Get sum of eke within Ceff
                                teke = grd.eke[jmin:jmax, imin:imax][
                                    eddy.mask_eff].sum()

                                if 'SLA' in eddy.DIAGNOSTIC_TYPE:

                                    args = (eddy, c_s, collind,
                                            centlon_e, centlat_e,
                                            cont, grd, eddy_radius_e)

                                    if not eddy.TRACK_EXTRA_VARIABLES:
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
                                    c_x, c_y = proj(inner_contlon,
                                                    inner_contlat)
                                    centx_s, centy_s, _, _ = fit_circle(c_x,
                                                                        c_y)
                                    centlon_s, centlat_s = proj(centx_s,
                                                                centy_s,
                                                                inverse=True)
                                    # Second, get speed-based radius based on
                                    # contour of max uavg
                                    # (perhaps we should make a new proj here
                                    # based on contlon_s, contlat_s but I'm not
                                    # sure it's that important ... Antoine?)
                                    # A. : I dont think, the difference is tiny
                                    c_x, c_y = proj(contlon_s, contlat_s)
                                    _, _, eddy_radius_s, _ = fit_circle(c_x,
                                                                        c_y)

                                # Define T and S if needed
                                # if has_ts:
                                    # # Temperature at centroid
                                    # cent_temp = temp[centj, centi]
                                    # # Salinity at centroid
                                    # cent_salt = salt[centj, centi]

                                properties.eddy_radius_s = eddy_radius_s
                                properties.uavg = uavg
                                properties.eddy_radius_e = eddy_radius_e
                                properties.rtime = rtime
                                properties.teke = teke

                                # Update SLA eddy properties
                                if 'SLA' in eddy.DIAGNOSTIC_TYPE:

                                    # See CSS11 section B4
                                    properties.centlon, properties.centlat = np.copy(centlon_s), np.copy(centlat_s)

                                    if not has_ts:  # for AVISO

                                        if 'Anticyclonic' in sign_type:
                                            a_list_obj.update_eddy_properties(
                                                properties)

                                        elif 'Cyclonic' in sign_type:
                                            c_list_obj.update_eddy_properties(
                                                properties)

                                # Mask out already found eddies
                                if 'SLA' in eddy.DIAGNOSTIC_TYPE:
                                    eddy.sla[jmin:jmax, imin:imax][
                                        eddy.mask_eff] = eddy.fillval

    # Leave collection_loop
    if 'SLA' in eddy.DIAGNOSTIC_TYPE:
        if c_list_obj is None:
            return a_list_obj
        elif a_list_obj is None:
            return c_list_obj
    else:
        return a_list_obj, c_list_obj


def track_eddies(eddy, first_record):
    """
    Track the eddies. First a distance matrix is calculated between
    all new and old eddies. Then loop through each old eddy, sorting the
    distances, and selecting that/those within range.
    """
    dist_0 = eddy.DIST0
    amp_0 = eddy.AMP0
    area_0 = eddy.AREA0
    # TEMP0 = 20.
    # SALT0 = 35.

    far_away = 1e9

    # We will need these in m_val for ellipse method below
    old_lon = np.array(eddy.old_lon)
    old_lat = np.array(eddy.old_lat)
    new_lon_tmp = np.array(eddy.new_lon_tmp)
    new_lat_tmp = np.array(eddy.new_lat_tmp)
    old_x, old_y = eddy.m_val(old_lon, old_lat)
    new_x, new_y = eddy.m_val(new_lon_tmp, new_lat_tmp)

    # Use haversine distance for distance matrix between every old and new eddy
    dist_mat = np.empty((old_lon.shape[0], new_lon_tmp.shape[0]))
    distance_matrix(
        old_lon, old_lat,
        new_lon_tmp, new_lat_tmp,
        dist_mat)
    dist_mat_copy = dist_mat.copy()

    # Prepare distance matrix for sparse operation
    # NOTE: need to find optimal (+dynamic) choice of filter here
    dist_mat[dist_mat > 35000.] = 0

    # Use a coordinate format (COO) sparse matrix
    # https://scipy-lectures.github.io/advanced/scipy_sparse/coo_matrix.html
    sparse_mat = coo_matrix((dist_mat))
    old_sparse_inds = sparse_mat.row
    new_sparse_inds = sparse_mat.col

    # *new_eddy_inds* contains indices to every newly identified eddy
    # that are initially set to True on the assumption that it is a new
    # eddy, i.e, it has just been born.
    # Later some will be set to False, indicating the continuation
    # of an existing eddy.
    new_eddy_inds = np.ones_like(new_x, dtype=bool)
    new_eddy = False

    # Loop over the old eddies looking for active eddies
    # for old_ind in np.arange(dist_mat.shape[0]):
    for old_ind in old_sparse_inds:

        dist_arr = np.array([])
        (new_ln, new_lt, new_rd_s, new_rd_e, new_am,
         new_Ua, new_ek, new_tm) = ([], [], [], [], [], [], [], [])

        new_cntr_e, new_cntr_s, new_uavg_prf = [], [], []
        new_shp_err = np.array([])

        backup_ind = np.array([], dtype=np.int16)

        '''
        # See http://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
        non_unique_inds = np.setdiff1d(np.arange(len(dist_mat[old_ind])),
                                           np.unique(dist_mat[old_ind],
                                                    return_index=True)[1])
        # Move non_unique_inds far away
        dist_mat[old_ind][non_unique_inds] = far_away  # km
        '''

        # Make an ellipse at current old_eddy location
        # (See CSS11 sec. B4, pg. 208)
        if 'ellipse' in eddy.SEPARATION_METHOD:
            eddy.search_ellipse.set_search_ellipse(old_x[old_ind],
                                                   old_y[old_ind])

        # Loop over separation distances between old and new
        for new_ind in new_sparse_inds:

            new_dist = dist_mat[old_ind, new_ind]

            within_range = False
            if new_dist < eddy.search_ellipse.rw_c_mod:  # far_away:

                if 'ellipse' in eddy.SEPARATION_METHOD:
                    if eddy.search_ellipse.ellipse_path.contains_point(
                            (new_x[new_ind], new_y[new_ind])):
                        within_range = True

                elif 'sum_radii' in eddy.SEPARATION_METHOD:
                    sep_dist = eddy.new_radii_e_tmp[new_ind]
                    sep_dist += eddy.old_radii_e[old_ind]
                    sep_dist *= eddy.sep_dist_fac
                    within_range = new_dist <= sep_dist

                else:
                    Exception

            # Pass only the eddies within ellipse or sep_dist
            if within_range:
                dist_arr = np.r_[dist_arr, new_dist]
                new_ln.append(eddy.new_lon_tmp[new_ind])
                new_lt.append(eddy.new_lat_tmp[new_ind])
                new_rd_s.append(eddy.new_radii_s_tmp[new_ind])
                new_rd_e.append(eddy.new_radii_e_tmp[new_ind])
                new_am.append(eddy.new_amp_tmp[new_ind])
                new_Ua.append(eddy.new_uavg_tmp[new_ind])
                new_ek.append(eddy.new_teke_tmp[new_ind])
                new_tm.append(eddy.new_time_tmp[new_ind])

                if eddy.TRACK_EXTRA_VARIABLES:
                    new_cntr_e.append(eddy.new_contour_e_tmp[new_ind])
                    new_cntr_s.append(eddy.new_contour_s_tmp[new_ind])
                    new_uavg_prf.append(eddy.new_uavg_profile_tmp[new_ind])
                    new_shp_err = np.r_[new_shp_err,
                                        eddy.new_shape_error_tmp[new_ind]]

                backup_ind = np.r_[backup_ind, new_ind]

                # An old (active) eddy has been detected, so
                # corresponding new_eddy_inds set to False
                new_eddy_inds[
                    np.nonzero(eddy.new_lon_tmp ==
                               eddy.new_lon_tmp[new_ind])] = False

                dist_mat[:, new_ind] = far_away  # km

        if eddy.TRACK_EXTRA_VARIABLES:
            kwargs = {'contour_e': new_cntr_e, 'contour_s': new_cntr_s,
                      'uavg_profile': new_uavg_prf, 'shape_error': new_shp_err}
        else:
            kwargs = {}

        # Only one eddy within range
        if dist_arr.size == 1:  # then update the eddy track only
            # Use index 0 here because type is list and we want the scalar
            args = (eddy, old_ind, new_ln[0], new_lt[0], new_rd_s[0],
                    new_rd_e[0], new_am[0], new_Ua[0], new_ek[0], new_tm[0],
                    new_eddy, first_record)

            # NOTE accounting should be part of eddy (or some other) object...
            eddy = accounting(*args, **kwargs)

        # More than one eddy within range
        elif dist_arr.size > 1:

            # Loop to find the right eddy
            delta_area = np.array([])
            delta_amp = np.array([])

            for i in np.nditer(np.arange(dist_arr.size)):

                # Choice of using effective or speed-based...
                delta_area_tmp = np.array([np.pi *
                                           (eddy.old_radii_e[old_ind] ** 2),
                                           np.pi * (new_rd_e[i] ** 2)]).ptp()
                delta_amp_tmp = np.array([eddy.old_amp[old_ind],
                                          new_am[i]]).ptp()
                delta_area = np.r_[delta_area, delta_area_tmp]
                delta_amp = np.r_[delta_amp, delta_amp_tmp]

            # This from Penven etal (2005)
            delta_x = np.sqrt((delta_area / area_0) ** 2 +
                              (delta_amp / amp_0) ** 2 +
                              (dist_arr / dist_0) ** 2)

            d_x = delta_x.argsort()
            dx0 = d_x[0]  # index to the nearest eddy
            dx_unused = d_x[1:]  # index/indices to the unused eddy/eddies

            logging.debug('delta_area %s', delta_area)
            logging.debug('delta_amp %s', delta_amp)
            logging.debug('dist_arr %s', dist_arr)
            logging.debug('d_x %s', d_x)
            logging.debug('dx0 %s', dx0)
            logging.debug('dx_unused %s', dx_unused)

            # Update eddy track dx0
            if not eddy.TRACK_EXTRA_VARIABLES:
                new_cntr_e = new_cntr_s = new_uavg_prf = new_shp_err = None
            else:
                new_cntr_e = new_cntr_e[dx0]
                new_cntr_s = new_cntr_s[dx0]
                new_uavg_prf = new_uavg_prf[dx0]
                new_shp_err = new_shp_err[dx0]

            args = (eddy, old_ind, new_ln[dx0], new_lt[dx0], new_rd_s[dx0],
                    new_rd_e[dx0], new_am[dx0], new_Ua[dx0], new_ek[dx0],
                    new_tm[dx0], new_eddy, first_record)

            kwargs = {'contour_e': new_cntr_e, 'contour_s': new_cntr_s,
                      'uavg_profile': new_uavg_prf, 'shape_error': new_shp_err}

            if 'ROMS' in eddy.product:
                pass
                # print 'fix me for temp and salt'

            eddy = accounting(*args, **kwargs)

            # Use backup_ind to reinsert distances into dist_mat for the unused
            # eddy/eddies
            for bind in backup_ind[dx_unused]:
                dist_mat[:, bind] = dist_mat_copy[:, bind]
                new_eddy_inds[bind] = True

    # Finished looping over old eddy tracks

    # Now we need to add new eddies defined by new_eddy_inds
    if np.any(new_eddy_inds):

        if False:
            logging.info('adding %s new eddies', new_eddy_inds.sum())

        for neind, a_new_eddy in enumerate(new_eddy_inds):

            if a_new_eddy:  # Update the eddy tracks

                if not eddy.TRACK_EXTRA_VARIABLES:
                    new_contour_e_tmp = new_contour_s_tmp = None
                    new_uavg_profile_tmp = new_shape_error_tmp = None

                else:
                    new_contour_e_tmp = eddy.new_contour_e_tmp[neind]
                    new_contour_s_tmp = eddy.new_contour_s_tmp[neind]
                    new_uavg_profile_tmp = eddy.new_uavg_profile_tmp[neind]
                    new_shape_error_tmp = eddy.new_shape_error_tmp[neind]

                args = (eddy, False, eddy.new_lon_tmp[neind],
                        eddy.new_lat_tmp[neind], eddy.new_radii_s_tmp[neind],
                        eddy.new_radii_e_tmp[neind], eddy.new_amp_tmp[neind],
                        eddy.new_uavg_tmp[neind], eddy.new_teke_tmp[neind],
                        eddy.new_time_tmp[neind], True, False)

                kwargs = {'contour_e': new_contour_e_tmp,
                          'contour_s': new_contour_s_tmp,
                          'uavg_profile': new_uavg_profile_tmp,
                          'shape_error': new_shape_error_tmp}

                if 'ROMS' in eddy.product:
                    # kwargs['cent_temp'] = eddy.new_temp_tmp[neind]
                    pass
                    # kwargs['cent_salt'] = eddy.new_salt_tmp[neind]

                eddy = accounting(*args, **kwargs)

    return eddy


def accounting(Eddy, old_ind, centlon, centlat,
               eddy_radius_s, eddy_radius_e, amplitude, uavg, teke, rtime,
               new_eddy, first_record, contour_e=None, contour_s=None,
               uavg_profile=None, shape_error=None, cent_temp=None,
               cent_salt=None):
    """
    Accounting for new or old eddy (either cyclonic or anticyclonic)
      Eddy  : eddy_tracker.tracklist object
      old_ind    : index to current old
      centlon, centlat: arrays of lon/lat centroids
      eddy_radius_s:  speed-based eddy radius
      eddy_radius_e:  effective eddy radius from fit_circle
      amplitude  :  eddy amplitude/intensity (max abs vorticity/f in eddy)
      uavg       :  average velocity within eddy contour
      teke       : sum of EKE within Ceff
      rtime      :  ROMS time in seconds
      cent_temp  :  array of temperature at centroids
      cent_salt  :  array of salinity at centroids
      bounds     :  index array(imin,imax,jmin,jmax) defining location of eddy
      new_eddy   : flag indicating a new eddy
      first_record : flag indicating that we're on the first record
    """
    if first_record:  # is True then all eddies are new
        new_eddy = True
        logging.debug('writing first record')

    kwargs = {'temp': cent_temp, 'salt': cent_salt,
              'contour_e': contour_e, 'contour_s': contour_s,
              'uavg_profile': uavg_profile, 'shape_error': shape_error}

    if not new_eddy:  # it's an old (i.e., active) eddy

        Eddy.insert_at_index('new_lon', old_ind, centlon)
        Eddy.insert_at_index('new_lat', old_ind, centlat)
        Eddy.insert_at_index('new_radii_s', old_ind, eddy_radius_s)
        Eddy.insert_at_index('new_radii_e', old_ind, eddy_radius_e)
        Eddy.insert_at_index('new_amp', old_ind, amplitude)
        Eddy.insert_at_index('new_uavg', old_ind, uavg)
        Eddy.insert_at_index('new_teke', old_ind, teke)

        if 'ROMS' in Eddy.product:
            pass
            # Eddy.insert_at_index('new_temp', old_ind, cent_temp)
            # Eddy.insert_at_index('new_salt', old_ind, cent_salt)

        if Eddy.TRACK_EXTRA_VARIABLES:
            Eddy.insert_at_index('new_contour_e', old_ind, contour_e)
            Eddy.insert_at_index('new_contour_s', old_ind, contour_s)
            Eddy.insert_at_index('new_uavg_profile', old_ind, uavg_profile)
            Eddy.insert_at_index('new_shape_error', old_ind, shape_error)

        args = (old_ind, centlon, centlat, rtime, uavg, teke,
                eddy_radius_s, eddy_radius_e, amplitude)

        Eddy.update_track(*args, **kwargs)

    else:  # it's a new eddy

        # We extend the range of array to old_ind
        Eddy.insert_at_index('new_lon', Eddy.index, centlon)
        Eddy.insert_at_index('new_lat', Eddy.index, centlat)
        Eddy.insert_at_index('new_radii_s', Eddy.index, eddy_radius_s)
        Eddy.insert_at_index('new_radii_e', Eddy.index, eddy_radius_e)
        Eddy.insert_at_index('new_amp', Eddy.index, amplitude)
        Eddy.insert_at_index('new_uavg', Eddy.index, uavg)
        Eddy.insert_at_index('new_teke', Eddy.index, teke)

        if 'ROMS' in Eddy.product:
            pass
            # Eddy.insert_at_index('new_temp', Eddy.index, cent_temp)
            # Eddy.insert_at_index('new_salt', Eddy.index, cent_salt)

        if Eddy.TRACK_EXTRA_VARIABLES:
            Eddy.insert_at_index('new_contour_e', Eddy.index, contour_e)
            Eddy.insert_at_index('new_contour_s', Eddy.index, contour_s)
            Eddy.insert_at_index('new_uavg_profile', Eddy.index, uavg_profile)
            Eddy.insert_at_index('new_shape_error', Eddy.index, shape_error)

        if Eddy.new_list:  # initialise a new list
            logging.info('starting a new track list for %s',
                         Eddy.SIGN_TYPE.replace('nic', 'nes'))
            Eddy.new_list = False

        args = (centlon, centlat, rtime, uavg, teke,
                eddy_radius_s, eddy_radius_e, amplitude)

        Eddy.add_new_track(*args, **kwargs)

        Eddy.index += 1

    return Eddy  # Leave accounting


def get_ROMS_data(rfile, grd=None, index=None, sigma_lev=None,
                  DIAGNOSTIC_TYPE=None):
    """
    """
    with Dataset(rfile) as h_nc:
        if index is None:
            nc_time = h_nc.variables['ocean_time'][:]
            return nc_time

        nc_time = h_nc.variables['ocean_time'][index]
        istr, iend = grd.ip0, grd.ip1
        jstr, jend = grd.jp0, grd.jp1
        if 'SLA' in DIAGNOSTIC_TYPE:
            zeta = h_nc.variables['zeta'][index, jstr:jend, istr:iend]
            return zeta, nc_time
        else:
            return


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
    nj = jsz + 2
    ni = isz + 2

    def hann2d_fast(var, ni, nj):
        # print 'jsz, isz',jsz, isz
        var_ext = np.ma.zeros((nj, ni))  # add 1-more line parallell to
        var_ext[1:-1, 1:-1] = var        # each of 4-sides
        var_ext[1:-1, 0] = var[:, 0]   # duplicate W-side
        var_ext[1:-1, -1] = var[:, -1]  # duplicate E-side
        var_ext[0, 1:-1] = var[0]   # duplicate N-side
        var_ext[-1, 1:-1] = var[-1]  # duplicate S-side
        var_ext[0, 0] = np.nan     # NW-corner
        var_ext[0, -1] = np.nan     # NE-corner
        var_ext[-1, 0] = np.nan     # SW-corner
        var_ext[-1, -1] = np.nan     # SE-corner

        # npts is used to count number of valid neighbors
        npts = var_ext * 0. + 1.
        npts[np.isnan(npts)] = 0.

        # Replace nans with 0 to find a no-nan sum
        var_ext[np.isnan(var_ext)] = 0.

        # Initialize count and sum variables
        cc = npts[1:nj-1, 1:ni-1] * (
            npts[0:nj-2, 1:ni-1] + npts[2:nj, 1:ni-1] +
            npts[1:nj-1, 0:ni-2] + npts[1:nj-1, 2:ni])

        varS = (var_ext[0:nj-2, 1:ni-1] + var_ext[2:nj, 1:ni-1] +
                var_ext[1:nj-1, 0:ni-2] + var_ext[1:nj-1, 2:ni])

        cc[cc == 0] = np.nan  # bring back nans in original data.
        weight = 8. - cc  # This is the weight for values on each grid point,
#                                         based on number of valid neighbours
        # Final smoothed version of var
        hsm = 0.125 * (varS + weight * var_ext[1:jsz+1, 1:isz+1])
        return hsm

    for _ in np.arange(numpasses):
        var[:] = hann2d_fast(var, ni, nj)

    return var


# def get_circle_OLD(x0, y0, r, npts):
    # """
    # Return points on a circle, with specified (x0, y0) center and radius
#                     (and optional number of points too).

    # Input     : 1  - x0, scalar, center X of circle
#                 2  - y0, scalar, center Y of circle
#                 3  - r,  scalar, radius
#                 4  - npts, scalar, number of points (optional)

    # Output    : 1  - cx, circle x-points
#                 2  - cy, circle y-points

    # Example   :  [cx cy] = func_get_circle (5, 5, 3, 256)
#                 plot (cx, cy, '-')

    # Written By : UCLA ROMS Team (jaison@atmos.ucla.edu)
    # Written On : June/05/2008
    # Tool       : Eddy Tracker
    # """
    # theta = np.arange(np.float64(npts))  # NOTE npts is a constant, so
    # # *cos(theta)* and *sin(theta)* can be predefined
    # # SHOULD BE PART OF CONTOUR OBJECT
    # theta[:] = theta * 2. * (4. * np.arctan(1.)) / npts
    # cx = x0 + r * np.cos(theta)
    # cy = y0 + r * np.sin(theta)
    # return cx, cy


def get_circle(lon, lat, distance, npts):
    '''
    Given the inputs (base lon, base lat, angle [deg], distance [m]) return
    the lon, lat of the new position...
    # SHOULD BE PART OF CONTOUR OBJECT
    Could be speeded up with f2py
    '''
    pio180 = np.pi / 180.
    lon *= pio180
    lat *= pio180
    angle = np.arange(np.float64(npts))
    angle[:] = angle * 2. * (4. * np.arctan(1.)) / npts
    distance /= 6371315.0  # angular distance
    lat1 = (np.arcsin(np.sin(lat) * np.cos(distance) + np.cos(lat) *
                      np.sin(distance) * np.cos(angle)))
    lon1 = (lon + np.arctan2(np.sin(angle) * np.sin(distance) * np.cos(lat),
                             np.cos(distance) - np.sin(lat) * np.sin(lat1)))
    return lon1 / pio180, lat1 / pio180
