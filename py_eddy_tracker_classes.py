# -*- coding: utf-8 -*-
# %run py_eddy_tracker_classes.py

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

Copyright (c) 2014 by Evan Mason
Email: emason@imedea.uib-csic.es
===========================================================================

py_eddy_tracker_classes.py

Version 1.2.1
===========================================================================


"""
# External modules
import netCDF4 as netcdf
import matplotlib.pyplot as plt
import numpy as np
import numexpr as ne 
import time
from mpl_toolkits.basemap import Basemap
import matplotlib.dates as dt
from scipy import ndimage
from scipy import interpolate
from scipy import spatial
from scipy import io
import matplotlib.path as path
import matplotlib.patches as patch
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

# py-eddy-tracker modules
import make_eddy_tracker_list_obj as eddy_tracker
#import roms_grid as rg
import haversine_distmat as hav # needs compiling with f2py


def datestr2datetime(datestr):
    '''
    Take strings with format YYYYMMDD and convert to datetime instance
    '''
    assert len(datestr) == 8, "'datestr' must have length 8"
    assert datestr[0] in ('1', '2'), "first character of 'datestr' should be '1' or '2'"
    return dt.datetime.datetime(np.int(datestr[:4]),
                                np.int(datestr[4:6]),
                                np.int(datestr[6:]))


def gaussian_resolution(res, zwl, mwl):
    '''
    Get parameters for ndimage.gaussian_filter
    See http://stackoverflow.com/questions/14531072/how-to-count-bugs-in-an-image
    Input: res : grid resolution in degrees
           zwl : zonal distance in degrees
           mwl : meridional distance in degrees
    '''
    zres = np.copy(zwl)
    mres = np.copy(mwl)
    zres *= 0.125
    mres *= 0.125
    zres /= res
    mres /= res
    return zres, mres




def detect_local_minima(arr):
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    background = (arr == 0)
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    detected_minima = local_min - eroded_background
    return detected_minima



def haversine_cdist(l1, l2):
    '''
    Haversine formula to be used by scipy.spatial.cdist
    Input:
        (lon1, lat1), (lon2, lat2)
    Return:
        distance (m)
    THIS FUNCTION IS REPEATED IN 
    '''
    lon1, lat1, lon2, lat2 = l1[0], l1[1], l2[0], l2[1]
    #print 'lon1.shape',lon1
    #print 'lon2.shape',lon2
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    #print 'dlat.shape',dlat
    #print 'dlon.shape',dlon
    a = ne.evaluate('sin(0.5 * dlon) * sin(0.5 * dlon)')
    a = ne.evaluate('a * cos(lat1) * cos(lat2)')
    a = ne.evaluate('a + (sin(0.5 * dlat) * sin(0.5 * dlat))')
    c = ne.evaluate('2 * arctan2(sqrt(a), sqrt(1 - a))')
    return ne.evaluate('6371315.0 * c') # Return the distance


def do_basemap(M, ax):
    '''
    Convenience method for Basemap functions
    '''
    if np.logical_or(np.diff([M.lonmin, M.lonmax]) > 60,
                     np.diff([M.latmin, M.latmax]) > 60):
        stride = 10
    elif np.logical_or(np.diff([M.lonmin, M.lonmax]) > 40,
                     np.diff([M.latmin, M.latmax]) > 40):
        stride = 8
    elif np.logical_or(np.diff([M.lonmin, M.lonmax]) > 30,
                       np.diff([M.latmin, M.latmax]) > 30):
        stride = 5
    elif np.logical_or(np.diff([M.lonmin, M.lonmax]) > 15,
                       np.diff([M.latmin, M.latmax]) > 15):
        stride = 3
    else:
        stride = 2
    M.drawparallels(np.arange(-90, 90.+stride, stride), labels=[1,0,0,0],
                    linewidth=0.25, size=8, ax=ax)
    M.drawmeridians(np.arange(-360, 360.+stride, stride), labels=[0,0,0,1],
                    linewidth=0.25, size=8, ax=ax)
    M.fillcontinents('k', ax=ax)
    M.drawcoastlines(linewidth=0.5, ax=ax)
    return


def anim_figure(A_eddy, C_eddy, Mx, My, pMx, pMy, cmap, rtime, diag_type,
                savedir, tit, ax):
    '''
    '''
    def plot_tracks(Eddy, track_length, rtime, col, ax):
        for i in Eddy.get_active_tracks(rtime):
            if Eddy.tracklist[i].lon.size > track_length: # filter for longer tracks
                aex, aey = Eddy.M(Eddy.tracklist[i].lon, Eddy.tracklist[i].lat)
                M.plot(aex, aey, col, lw=0.5, ax=ax)
                M.scatter([aex[-1]], [aey[-1]], s=7, c=col, ax=ax)
        return
    
    #plt.figure(fignum)
    #thax = plt.subplot(111)
    track_length = 0 # for filtering below
    M = A_eddy.M
    
    if 'Q' in diag_type:
        cb = M.pcolormesh(pMx, pMy, xicopy, cmap=cmap, ax=ax)
        M.contour(Mx, My, xi, [0.], ax=ax, colors='k', linewidths=0.5)
        M.contour(Mx, My, qparam, qparameter, ax=ax, colors='g', linewidths=0.25)
        cb.set_clim(-.5, .5)
        M.contour(Mx, My, qparam, [qparameter[0]], ax=ax, colors='m', linewidths=0.25)
    
    elif 'sla' in diag_type:
        cb = M.pcolormesh(pMx, pMy, A_eddy.slacopy, cmap=cmap, ax=ax)
        M.contour(Mx, My, A_eddy.slacopy, [0.], ax=ax, colors='k', linewidths=0.5)
        M.contour(Mx, My, A_eddy.slacopy, A_eddy.slaparameter, ax=ax, colors='g',
                  linestyles='solid', linewidths=0.15)
        cb.set_clim(-20., 20.)
    plot_tracks(A_eddy, track_length, rtime, 'r', ax)
    plot_tracks(C_eddy, track_length, rtime, 'b', ax)
    
    do_basemap(M, ax)
    
    ax.set_title(tit)
    #cax = get_cax(sp)
    plt.colorbar(cb, use_gridspec=True, orientation='horizontal', aspect=30)
    plt.savefig(savedir + 'eddy_track_%s.png' %tit.replace(' ','_'), dpi=150, bbox_inches='tight')
    #plt.close(fignum)
    ax.cla()
    return



def get_cax(subplot, dx=0, width=.015, position='r'):
    '''
    Return axes for colorbar same size as subplot
      subplot - subplot object
      dx - distance from subplot to colorbar
      width - width of colorbar
      position - 'r' - right (default)
               - 'b' - bottom
    '''
    cax = subplot.get_position().get_points()
    if position=='r':
        cax = plt.axes([cax[1][0]+dx,cax[0][1],width,cax[1][1]-cax[0][1]])
    if position=='b':
        cax = plt.axes([cax[0][0],cax[0][1]-2*dx,cax[1][0]-cax[0][0],width])
    return cax


def oceantime2ymd(ocean_time, integer=False):
    '''
    Return strings y, m, d given ocean_time (seconds)
    If integer == True return integer rather than string
    '''
    if np.isscalar(ocean_time):
        ocean_time = np.array([ocean_time])
    ocean_time = ocean_time / 86400.
    year = np.floor(ocean_time / 360.)
    month = np.floor((ocean_time - year * 360.) / 30.)
    day = np.floor(ocean_time - year * 360. - month * 30.)
    year = (year.astype(np.int16) + 1)[0]
    month = (month.astype(np.int16) + 1)[0]
    day = (day.astype(np.int16) + 1)[0]
    if not integer:
        year  = str(year)
        month = str(month)
        day   = str(day)
        if len(year) < 2:
            year = year.zfill(2)
        if len(month) < 2:
            month = month.zfill(2)
        if len(day) < 2:
            day = day.zfill(2)
    return year, month, day
    


def half_interp(h1, h2):
    '''
    Speed up for frequent operations
    '''
    return ne.evaluate('0.5 * (h1 + h2)')


def quart_interp(h1, h2, h3, h4):
    '''
    Speed up for frequent operations
    '''
    return ne.evaluate('0.25 * (h1 + h2 + h3 + h4)')


def okubo_weiss(u, v, pm, pn):
    '''
    Calculate the Okubo-Weiss parameter
    See e.g. http://ifisc.uib.es/oceantech/showOutput.php?idFile=61
    Returns: lambda2 - Okubo-Weiss parameter [s^-2]
             xi      - rel. vorticity [s^-1]
    Adapted from Roms_tools
    '''
    def vorticity(u, v, pm, pn):
        '''
        Returns vorticity calculated using np.gradient
        '''
        def vort(u, v, dx, dy):
            dx = ne.evaluate('1 / dx')
            dy = ne.evaluate('1 / dy')
            uy, ux = np.gradient(u2rho_2d(u), dx, dy)
            vy, vx = np.gradient(v2rho_2d(v), dx, dy)
            xi = ne.evaluate('vx - uy')
            return xi
        return vort(u, v, pm, pn)

    Mp, Lp = pm.shape
    L = ne.evaluate('Lp - 1')
    M = ne.evaluate('Mp - 1')
    Lm = ne.evaluate('L - 1')
    Mm = ne.evaluate('M - 1')
    mn_p = np.zeros((M, L))
    uom = np.zeros((M, Lp))
    von = np.zeros((Mp, L))
    uom = 2. * u / (pm[:,:L] + pm[:,1:Lp])
    uon = 2. * u / (pn[:,:L] + pn[:,1:Lp])
    von = 2. * v / (pn[:M,:] + pn[1:Mp,:])
    vom = 2. * v / (pm[:M,:] + pm[1:Mp,:])
    mn = pm * pn
    mn_p = quart_interp(mn[:M, :L],    mn[:M,  1:Lp],
                        mn[1:Mp,1:Lp], mn[1:Mp, :L])
    # Sigma_T
    ST = mn * psi2rho(von[:,1:Lp] - von[:,:L] + uom[1:Mp,:] - uom[:M,:])
    # Sigma_N
    SN = np.zeros((Mp,Lp))
    SN[1:-1,1:-1] = mn[1:-1,1:-1] * (uon[1:-1,1:]    \
                                   - uon[1:-1,:-1]   \
                                   - vom[1:,1:-1]    \
                                   + vom[:-1,1:-1])
    # Relative vorticity
    xi = vorticity(u, v, pm, pn)
    # Okubo
    lambda2 = np.power(SN, 2)
    lambda2 += np.power(ST, 2)
    lambda2 -= np.power(xi, 2)
    return lambda2, xi


def psi2rho(var_psi):
    # Convert a psi field to rho points
    M, L = var_psi.shape
    Mp = ne.evaluate('M + 1')
    Lp = ne.evaluate('L + 1')
    Mm = ne.evaluate('M - 1')
    Lm = ne.evaluate('L - 1')
    var_rho = np.zeros((Mp, Lp))
    var_rho[1:M, 1:L] = quart_interp(var_psi[0:Mm, 0:Lm], var_psi[0:Mm, 1:L],
                                     var_psi[1:M,  0:Lm], var_psi[1:M,  1:L])
    var_rho[0] = var_rho[1]
    var_rho[M] = var_rho[Mm]
    var_rho[:,0] = var_rho[:,1]
    var_rho[:,L] = var_rho[:,Lm]
    return var_rho



def pcol_2dxy(x, y):
    '''
    Function to shift x, y for subsequent use with pcolor
    by Jeroen Molemaker UCLA 2008
    '''
    Mp, Lp = x.shape
    M = Mp - 1
    L = Lp - 1
    x_pcol = np.zeros((Mp, Lp))
    y_pcol = np.zeros((Mp, Lp))
    x_tmp = half_interp(x[:,:L], x[:,1:Lp])
    x_pcol[1:Mp,1:Lp] = half_interp(x_tmp[0:M,:], x_tmp[1:Mp,:])
    x_pcol[0,:] = 2. * x_pcol[1,:] - x_pcol[2,:]
    x_pcol[:,0] = 2. * x_pcol[:,1] - x_pcol[:,2]
    y_tmp = half_interp(y[:,0:L], y[:,1:Lp]    )
    y_pcol[1:Mp,1:Lp] = half_interp(y_tmp[0:M,:], y_tmp[1:Mp,:])
    y_pcol[0,:] = 2. * y_pcol[1,:] - y_pcol[2,:]
    y_pcol[:,0] = 2. * y_pcol[:,1] - y_pcol[:,2]
    return x_pcol, y_pcol







def fit_circle(xvec, yvec):
    '''
    Fit the circle
    Adapted from ETRACK (KCCMC11)
    '''
    xvec, yvec = xvec.copy(), yvec.copy()
    if xvec.ndim == 1:
        xvec = xvec[np.newaxis]
    if yvec.ndim == 1:
        yvec = yvec[np.newaxis]
    if xvec.shape[1] != 1:
        xvec = xvec.T
    if yvec.shape[1] != 1:
        yvec = yvec.T
    
    npts = xvec.size
    xmean = np.mean(xvec)
    ymean = np.mean(yvec)
    xsc = ne.evaluate('xvec - xmean')
    ysc = ne.evaluate('yvec - ymean')
    scale = np.max((np.hypot(xsc, ysc).max(), np.finfo(float).eps))
    xsc /= scale
    ysc /= scale

    # Form matrix equation to be solved and solve it
    xyz  = np.linalg.lstsq(
             np.concatenate((2. * xsc, 2. * ysc, np.ones((npts, 1))), axis=1),
             np.hypot(xsc, ysc)**2)
    # Unscale data and get circle variables
    p = np.array([xmean, ymean, 0.]) +  \
        np.concatenate((xyz[0][0], xyz[0][1],
                        np.sqrt(xyz[0][2] + np.hypot(xyz[0][0], xyz[0][1])**2))) * scale

    ccx = p[0] # center X-position of fitted circle
    ccy = p[1] # center Y-position of fitted circle
    r = p[2] # radius of fitted circle
    carea = np.pi       
    carea = ne.evaluate('carea * r**2') # area of fitted circle
    
    # Shape test
    # Area and centroid of closed contour/polygon
    tmp = (xvec[:npts-1] * yvec[1:npts]) - (xvec[1:npts] * yvec[:npts-1])
    polar = ne.evaluate('sum(tmp, axis=None)')
    polar = ne.evaluate('0.5 * polar')
    parea = ne.evaluate('abs(polar)')
    
    # Find distance between circle center and contour points_inside_poly
    dist_poly = np.hypot(xvec - ccx, yvec - ccy)

    sintheta = ne.evaluate('(yvec - ccy) / dist_poly')
    ptmp_y = ne.evaluate('ccy + ( r * sintheta )')
    ptmp_x = ne.evaluate('ccx - (ccx - xvec) * ((ccy - ptmp_y) / (ccy - yvec))')

    pout_id = np.nonzero(dist_poly > r) # indices of polygon points outside circle
                                        # p_inon_? : polygon x or y points inside & on the circle
    p_inon_x = xvec # init 
    p_inon_y = yvec # init 
    p_inon_x[pout_id] = ptmp_x[pout_id]
    p_inon_y[pout_id] = ptmp_y[pout_id]

    # Area of closed contour/polygon enclosed by the circle
    tmp = (p_inon_x[0:npts-1] * p_inon_y[1:npts])  -    \
          (p_inon_x[1:npts]   * p_inon_y[0:npts-1])
    
    parea_incirc = ne.evaluate('sum(tmp, axis=None)')
    parea_incirc = ne.evaluate('0.5 * abs(parea_incirc)')
    aerr = ne.evaluate('100.0 * ((1 - parea_incirc / carea) + (parea - parea_incirc) / carea)')
    return ccx, ccy, r, aerr, npts
    





def get_Uavg(Eddy, CS, collind, centlon_e, centlat_e, poly_e, grd, eddy_radius_e):
    '''
    Calculate geostrophic speed around successive contours
    Returns the average
    '''
    def calc_uavg(points, uspd, lon, lat):
        '''
        TO DO: IT SHOULD BE QUICKER, AND POSSIBLY BETTER, TO DO THIS BY CALCULATING
        WEIGHTS AND THEN USING np.average()...
        '''
        uavg = interpolate.griddata(points, uspd.ravel(), (lon, lat), 'linear')
        return np.mean(uavg[np.isfinite(uavg)])
    
    # True for debug figures
    debug_U = False
    if debug_U:
        schem_dic = {}
        conts_x = np.array([])
        conts_y = np.array([])
    
    # Unpack indices for convenience
    istr, iend, jstr, jend = Eddy.i0, Eddy.i1, Eddy.j0, Eddy.j1
    imin, imax, jmin, jmax = Eddy.imin, Eddy.imax, Eddy.jmin, Eddy.jmax
    
    # TO DO: these would be passed in instead of grd if using Cython
    #brypath = grd.brypath(istr, iend, jstr, jend)
    
    points = np.array([grd.lon()[jmin:jmax,imin:imax].ravel(),
                       grd.lat()[jmin:jmax,imin:imax].ravel()]).T
    
    # First contour is the outer one (effective)
    theseglon, theseglat = poly_e.vertices[:,0].copy(), poly_e.vertices[:,1].copy()
    theseglon, theseglat = eddy_tracker.uniform_resample(theseglon, theseglat)
    Uavg = calc_uavg(points, Eddy.Uspd[jmin:jmax,imin:imax], theseglon[:-1], theseglat[:-1])

    start = True
    citer = np.nditer(CS.cvalues, flags=['f_index'])
    while not citer.finished:
        
        # Get contour around centlon_e, centlat_e at level [collind:][iuavg]
        try:
            conti, segi, junk, junk, junk, junk = CS.find_nearest_contour(
                    centlon_e, centlat_e,  indices=np.array([citer.index]),
                                                         pixel=False)
            poly_i = CS.collections[conti].get_paths()[segi]                   
        
        except Exception:
            poly_i = False

        if poly_i:
            # NOTE: contains_points requires matplotlib 1.3 +
            mask_i = poly_i.contains_points(points)
            
            # 1. Ensure polygon_i is within polygon_e
            # 2. Ensure polygon_i contains point centlon_e, centlat_e
            # 3. Respect size range
            if np.all([poly_e.contains_path(poly_i),
                       poly_i.contains_point([centlon_e, centlat_e]),
                       np.logical_and(np.sum(mask_i) >= Eddy.pixel_threshold[0],
                                      np.sum(mask_i) <= Eddy.pixel_threshold[1])]):

                seglon, seglat = poly_i.vertices[:,0], poly_i.vertices[:,1]
                seglon, seglat = eddy_tracker.uniform_resample(seglon, seglat)
                
                if debug_U:
                    px, py = pcol_2dxy(grd.lon()[jmin:jmax,imin:imax],
                                       grd.lat()[jmin:jmax,imin:imax])
                    plt.figure(55)
                    ax1 = plt.subplot(121)
                    if start:
                        pcm = ax1.pcolormesh(px, py, Eddy.Uspd[jmin:jmax,imin:imax], cmap=plt.cm.gist_earth_r)
                        pcm.set_clim(0, .5)
                        plt.colorbar(pcm, orientation='horizontal')
                        plt.scatter(grd.lon()[jmin:jmax,imin:imax],
                                    grd.lat()[jmin:jmax,imin:imax], s=5, c='k')
                        start = False
                    plt.plot(seglon, seglat, '.-r')
                    plt.plot(Eddy.circlon, Eddy.circlat, 'b')
                    plt.scatter(centlon_e, centlat_e, s=125, c='r')
                    plt.axis('image')
                    ax2 = plt.subplot(122)
                    ax2.set_title('sum mask: %s' %np.sum(mask_i))
                    plt.pcolormesh(px, py, mask_i.reshape(px.shape), cmap=plt.cm.Accent, edgecolors='orange')
                    plt.scatter(grd.lon()[jmin:jmax,imin:imax],
                                grd.lat()[jmin:jmax,imin:imax], s=5, c='k')
                    plt.plot(seglon, seglat, '.-r')
                    conts_x = np.append(conts_x, seglon)
                    conts_y = np.append(conts_y, seglat)
                    conts_x = np.append(conts_x, 9999)
                    conts_y = np.append(conts_y, 9999)
                    plt.plot(Eddy.circlon, Eddy.circlat, 'b')
                    plt.scatter(centlon_e, centlat_e, s=125, c='r')
                
                # Interpolate Uspd to seglon, seglat, then get mean
                Uavgseg = calc_uavg(points, Eddy.Uspd[jmin:jmax,imin:imax], seglon[:-1], seglat[:-1])
                
                if Uavgseg >= Uavg:
                    Uavg = Uavgseg.copy()
                    theseglon, theseglat = seglon.copy(), seglat.copy()
                
                inner_seglon, inner_seglat = seglon.copy(), seglat.copy()
                
        citer.iternext()
    
    try: # Assuming interior contours have been found
        cx, cy = Eddy.M(theseglon, theseglat)
        # Speed based eddy radius (eddy_radius_s)
        centx_s, centy_s, eddy_radius_s, junk, junk = fit_circle(cx, cy)
        centlon_s, centlat_s = Eddy.M.projtran(centx_s, centy_s, inverse=True)
        if debug_U:
            ax1.set_title('Speed-based radius: %s km' %np.int(eddy_radius_s/1e3))
            ax1.plot(poly_e.vertices[:,0], poly_e.vertices[:,1], 'om')
            ax1.plot(theseglon, theseglat, 'g', lw=3)
            ax1.plot(inner_seglon, inner_seglat, 'k')
            ax1.scatter(centlon_s, centlat_s, s=50, c='g')
            ax1.axis('image')
            ax2.plot(theseglon, theseglat, 'g', lw=3)
            ax1.plot(inner_seglon, inner_seglat, 'k')
            ax2.axis('image')
            plt.show()
            schem_dic['lon'] = grd.lon()[jmin:jmax,imin:imax]
            schem_dic['lat'] = grd.lat()[jmin:jmax,imin:imax]
            schem_dic['spd'] = Eddy.Uspd[jmin:jmax,imin:imax]
            schem_dic['inner_seg'] = np.array([inner_seglon, inner_seglat]).T
            schem_dic['effective_seg'] = np.array([poly_e.vertices[:,0], poly_e.vertices[:,1]]).T
            schem_dic['effective_circ'] = np.array([Eddy.circlon, Eddy.circlat]).T
            schem_dic['speed_seg'] = np.array([theseglon, theseglat]).T
            schem_dic['all_segs'] = np.array([conts_x, conts_y]).T
            schem_dic['Ls_centroid'] = np.array([centlon_s, centlat_s])
            schem_dic['Leff_centroid'] = np.array([centlon_e, centlat_e])
            io.savemat('schematic_fig_%s' %np.round(eddy_radius_s/1e3, 3), schem_dic)
        return Uavg, centlon_s, centlat_s, eddy_radius_s, theseglon, theseglat, inner_seglon, inner_seglat
    
    except Exception: # If no interior contours found, use eddy_radius_e
        if debug_U and 0:
            plt.title('Effective radius: %s km' %np.int(eddy_radius_e/1e3))
            plt.plot(poly_e.vertices[:,0], poly_e.vertices[:,1], 'om')
            plt.plot(theseglon, theseglat, 'g')
            plt.scatter(centlon_e, centlat_e, s=125, c='r')
            plt.axis('image')
            plt.show()
        return Uavg, centlon_e, centlat_e, eddy_radius_e, theseglon, theseglat, theseglon, theseglat





def collection_loop(CS, grd, rtime, A_list_obj, C_list_obj,
                    xi=None, CSxi=None, sign_type='None', verbose=False):
    '''
    Loop over each collection of contours
    '''
    if A_list_obj is not None:
        Eddy = A_list_obj
    if C_list_obj is not None:
        Eddy = C_list_obj
    
    if 'ROMS' in Eddy.datatype:
        has_ts = True
    elif 'AVISO' in Eddy.datatype:
        has_ts = False
    
    # Unpack indices for convenience
    istr, iend, jstr, jend = Eddy.i0, Eddy.i1, Eddy.j0, Eddy.j1
    
    # Loop over each collection
    for collind, coll in enumerate(CS.collections):

        if verbose:
            print '------ doing collection', collind, ', contour value', CS.cvalues[collind]
                    
        # Loop over individual CS contours (i.e., every eddy in field)
        for cont in coll.get_paths():
                        
            contlon_e, contlat_e = cont.vertices[:,0].copy(), \
                                   cont.vertices[:,1].copy()
                        
            # Filter for closed contours
            if np.alltrue([contlon_e[0] == contlon_e[-1],
                           contlat_e[0] == contlat_e[-1],
                           contlon_e.ptp(),
                           contlat_e.ptp()]):

                # Prepare for shape test and get eddy_radius_e
                cx, cy = Eddy.M(contlon_e, contlat_e)
                centlon_e, centlat_e, eddy_radius_e, aerr, junk = fit_circle(cx, cy)
                
                # Filter for shape: >35% (>55%) is not an eddy for Q (SLA)
                if np.logical_and(aerr >= 0., aerr <= Eddy.shape_err[collind]):
                                
                    # Get centroid in lon lat
                    centlon_e, centlat_e = Eddy.M.projtran(centlon_e, centlat_e, inverse=True)
                                
                    # Get eddy_radius_e (NOTE: if Q, we overwrite eddy_radius_e defined ~8 lines above)
                    if 'Q' in Eddy.diag_type:
                        junk, junk, junk, xilon, xilat, junk = CSxi.find_nearest_contour(
                                                               centlon_e, centlat_e, pixel=False)
                        eddy_radius_e = haversine_distance_vector(centlon_e, centlat_e, xilon, xilat)
                        
                        if np.logical_and(eddy_radius_e >= Eddy.radmin,
                                          eddy_radius_e <= Eddy.radmax):
                            proceed0 = True
                        else:
                            proceed0 = False
                    elif 'sla' in Eddy.diag_type:
                        # If 'sla' is defined we filter below with pixel count
                        proceed0 = True
                    else:
                        raise Exception
                    
                    if proceed0:
                        # Get indices of centroid
                        centi, centj = eddy_tracker.nearest(centlon_e, centlat_e,
                                                            grd.lon(),
                                                            grd.lat())
                        # If condition not True this eddy has already been detected
                        if 'Q' in Eddy.diag_type:
                            if xi[centj, centi] != fillval:
                                proceed1 = True
                            else:
                                proceed1 = False
                        elif 'sla' in Eddy.diag_type:
                            if Eddy.sla[centj, centi] != Eddy.fillval:
                                acyc_not_cyc = Eddy.sla[centj, centi] >= CS.cvalues[collind]
                                if 'Anticyclonic' in sign_type and acyc_not_cyc:
                                    proceed1 = True
                                    #print 'AC acyc_not_cyc',acyc_not_cyc
                                elif 'Cyclonic' in sign_type and not acyc_not_cyc:
                                    proceed1 = True
                                    #print 'CC acyc_not_cyc',acyc_not_cyc
                                else:
                                    proceed1 = False
                                    #print 'no eddy'
                            else:
                                proceed1 = False

                        
                        if proceed1:
                            # Get lon,lat of bounding box around eddy
                            Eddy.get_bounds(centlon_e, centlat_e,
                                            eddy_radius_e, centi, centj, grd)
                            # Unpack indices for convenience
                            imin = Eddy.imin
                            imax = Eddy.imax
                            jmin = Eddy.jmin
                            jmax = Eddy.jmax
                            
                            # Get eddy circumference using eddy_radius_e
                            centx, centy = Eddy.M(centlon_e, centlat_e)
                            circlon, circlat = get_circle(centx, centy, eddy_radius_e, 180)
                            circlon, circlat = Eddy.M.projtran(circlon, circlat, inverse=True)
                            Eddy.circlon = circlon
                            Eddy.circlat = circlat
                            
                            # Get a rectangle around the eddy
                            rectlon = grd.lon()[jmin:jmax,imin:imax].ravel()
                            rectlat = grd.lat()[jmin:jmax,imin:imax].ravel()

                            # Get mask inside the eddy
                            poly_e = path.Path(np.array([contlon_e, contlat_e]).T)
                            # NOTE: Path.contains_points requires matplotlib 1.2 or higher
                            mask_e = poly_e.contains_points(np.array([rectlon, rectlat]).T)
                            
                            # sum(mask) between 8 and 1000, CSS11 criterion 2 
                            if np.logical_and(np.sum(mask_e) >= Eddy.pixel_threshold[0],
                                              np.sum(mask_e) <= Eddy.pixel_threshold[1]):
                                
                                points = np.array([grd.lon()[jmin:jmax,imin:imax].ravel(),
                                                   grd.lat()[jmin:jmax,imin:imax].ravel()]).T
                                
                                if 'Q' in Eddy.diag_type:
                                    # Note, eddy amplitude == max(abs(vort/f)) within eddy, KCCMC11
                                    amplitude = np.abs(xi[jmin:jmax,imin:imax].flat[mask_e]).max()
                                
                                elif 'sla' in Eddy.diag_type:
                                    # Define as difference between extremum value for SLA anomaly and
                                    # the identifying contour value
                                    # CSS11, section B3 (here we assume h0 == CS.cvalues[collind])
                                    #h0 = CS.cvalues[collind]
                                    
                                    # Resample the contour points for a more even spatial distribution
                                    contlon_e, contlat_e = eddy_tracker.uniform_resample(contlon_e, contlat_e)

                                    h0 = interpolate.griddata(points, Eddy.sla[jmin:jmax,imin:imax].ravel(),
                                                                 (contlon_e, contlat_e), 'linear')
                                    h0 = np.mean(h0[np.isfinite(h0)]) # Use 'isfinite' in case of NaNs from griddata
                                    
                                    if 'Anticyclonic' in sign_type:
                                        # Check CSS11 criterion 1: The SSH values of all of the pixels
                                        # are above a given SSH threshold for anticyclonic eddies.
                                        if np.any(Eddy.sla[jmin:jmax,imin:imax].flat[mask_e] < h0):
                                            amplitude = 0.
                                        else:
                                            local_extrema = np.ma.copy(Eddy.sla[jmin:jmax,imin:imax])
                                            local_extrema = np.ma.masked_where(mask_e == False, local_extrema.ravel())
                                            local_extrema = np.reshape(local_extrema, (Eddy.sla[jmin:jmax,imin:imax].shape))
                                            local_extrema = detect_local_minima(-local_extrema)
                                            if np.ma.sum(local_extrema) == 1: 
                                                amplitude = Eddy.sla[jmin:jmax,imin:imax].flat[mask_e].max() - h0
                                                #centlon_lmi, centlat_lmi = get_position_local_extrema(Eddy, local_extrema, CS, grd)

                                                if 0: # Debug
                                                    plt.figure(511)
                                                    plt.title('Anticyclones')
                                                    x511, y511 = Eddy.M(grd.lon()[jstr:jend,istr:iend],
                                                                        grd.lat()[jstr:jend,istr:iend])
                                                    Eddy.M.pcolormesh(x511, y511, Eddy.slacopy-Eddy.slacopy.mean())
                                                    plt.clim(-10,10)
                                                    print 'Eddy.slacopy-Eddy.slacopy.mean()',(Eddy.slacopy-Eddy.slacopy.mean()).max()
                                                    x_lmi, y_lmi = Eddy.M(centlon_lmi, centlat_lmi)
                                                    Eddy.M.scatter(x_lmi, y_lmi, c='k')
                                                    x_e, y_e = Eddy.M(centlon_e, centlat_e)
                                                    Eddy.M.scatter(x_e, y_e, c='w')
                                                    lmi_j, lmi_i = np.where(local_extrema)
                                                    lmi_i = lmi_i[0] + imin
                                                    lmi_j = lmi_j[0] + jmin
                                                    x_i, y_i = Eddy.M(grd.lon()[jstr:jend,istr:iend][lmi_j, lmi_i],
                                                                      grd.lat()[jstr:jend,istr:iend][lmi_j, lmi_i])
                                                    Eddy.M.scatter(x_i, y_i, c='gray')
                                                    Eddy.M.fillcontinents()
                                                    plt.show()
                                                
                                            else:
                                                amplitude = 0.
                                    
                                    elif 'Cyclonic' in sign_type:
                                        # Check CSS11 criterion 1: The SSH values of all of the pixels
                                        # are below a given SSH threshold for cyclonic eddies.
                                        if np.any(Eddy.sla[jmin:jmax,imin:imax].flat[mask_e] > h0):
                                            amplitude = 0.
                                        else:
                                            local_extrema = np.ma.copy(Eddy.sla[jmin:jmax,imin:imax])
                                            local_extrema = np.ma.masked_where(mask_e == False, local_extrema.ravel())
                                            local_extrema = np.reshape(local_extrema, (Eddy.sla[jmin:jmax,imin:imax].shape))
                                            local_extrema = detect_local_minima(local_extrema)
                                            if np.ma.sum(local_extrema) == 1:
                                                amplitude = h0 - Eddy.sla[jmin:jmax,imin:imax].flat[mask_e].min()
                                                #centlon_lmi, centlat_lmi = get_position_local_extrema(Eddy, local_extrema, CS, grd)

                                                if 0: # Debug
                                                    plt.figure(511)
                                                    plt.title('Cyclones')
                                                    x511, y511 = Eddy.M(grd.lon()[jstr:jend,istr:iend],
                                                                        grd.lat()[jstr:jend,istr:iend])
                                                    Eddy.M.pcolormesh(x511, y511, Eddy.slacopy-Eddy.slacopy.mean())
                                                    plt.clim(-10,10)
                                                    print 'Eddy.slacopy-Eddy.slacopy.mean()',(Eddy.slacopy-Eddy.slacopy.mean()).max()
                                                    x_lmi, y_lmi = Eddy.M(centlon_lmi, centlat_lmi)
                                                    Eddy.M.scatter(x_lmi, y_lmi, c='k')
                                                    x_e, y_e = Eddy.M(centlon_e, centlat_e)
                                                    Eddy.M.scatter(x_e, y_e, c='w')
                                                    lmi_j, lmi_i = np.where(local_extrema)
                                                    lmi_i = lmi_i[0] + imin
                                                    lmi_j = lmi_j[0] + jmin
                                                    x_i, y_i = Eddy.M(grd.lon()[jstr:jend,istr:iend][lmi_j, lmi_i],
                                                                      grd.lat()[jstr:jend,istr:iend][lmi_j, lmi_i])
                                                    Eddy.M.scatter(x_i, y_i, c='gray')
                                                    Eddy.M.fillcontinents()
                                                    plt.show()
                                            else:
                                                amplitude = 0.
                            
                            else:
                                amplitude = 0.
                            
                            #print 'amplitude',amplitude
                            if np.logical_and(amplitude >= Eddy.ampmin, amplitude <= Eddy.ampmax):
                                proceed2 = True
                            else:
                                proceed2 = False

                            if proceed2:      
                                if 'Q' in Eddy.diag_type:
                                    if 0:
                                        plt.figure()
                                        tmpx, tmpy = pcol_2dxy(grd.lon()[jmin:jmax,imin:imax],
                                                               grd.lat()[jmin:jmax,imin:imax])
                                        print 'tmpx, tmpy, Uavg',tmpx, tmpy, Uavg
                                        plt.pcolormesh(tmpx, tmpy, Uavg)
                                        plt.axis('image')
                                        plt.plot(circlon, circlat, 'w')
                                        plt.plot(contlon_e, contlat_e, 'k')
                                        plt.contour(grd.lon()[jmin:jmax,imin:imax],
                                                    grd.lat()[jmin:jmax,imin:imax],
                                                    xicopy[jmin:jmax,imin:imax], [0.], colors='g')
                                        #print 'aerr', aerr
                                        #print interpolate.griddata(points, Uavg.ravel(),(contlon_e, contlat_e), 'linear')        
                                        plt.show()

                                    Uavg = interpolate.griddata(points, U_avg[jmin:jmax,imin:imax].ravel(),
                                                                (contlon_e, contlat_e), 'cubic')
                                    Uavg = np.nan_to_num(Uavg).max()
                                
                                elif 'sla' in Eddy.diag_type:
				    #tt = time.time()
                                    Uavg, centlon_s, centlat_s, eddy_radius_s,\
                                         contlon_s, contlat_s, inner_contlon, inner_contlat = get_Uavg(Eddy, CS, collind,
                                                                       centlon_e, centlat_e, poly_e, grd, eddy_radius_e)
                                    #print '------- got Uavg in %s seconds' %(time.time() - tt)
				    #tt = time.time()
                                    centlon_lmi, centlat_lmi, junk, junk, junk = fit_circle(inner_contlon,
                                                                                            inner_contlat)
                                    #print '------- got fit_circle in %s seconds' %(time.time() - tt)
                                    
                                    # Get indices of speed-based centroid
                                    centi, centj = eddy_tracker.nearest(centlon_s, centlat_s,
                                                                        grd.lon(),
                                                                        grd.lat())
                                        
                                # Set the bounds
                                bounds = np.atleast_2d(np.int16([imin, imax, jmin, jmax]))
                                
                                # Define T and S if needed
                                if has_ts:
                                    cent_temp = temp[centj, centi] # Temperature at centroid
                                    cent_salt = salt[centj, centi] # Salinity at centroid
                                
                                # Update Q eddy properties
                                if 'Q' in Eddy.diag_type:
                                    # Anticyclone
                                    if xi[centj, centi] <= 0.:
                                        A_list_obj.update_eddy_properties(centlon, centlat,
                                                                          eddy_radius_s, eddy_radius_e,
                                                                          amplitude, Uavg, rtime, bounds,
                                                                          cent_temp=cent_temp,
                                                                          cent_salt=cent_salt)
                                    # Cyclone
                                    elif xi[centj, centi] >= 0.:
                                        C_list_obj.update_eddy_properties(centlon, centlat,
                                                                          eddy_radius_s, eddy_radius_e,
                                                                          amplitude, Uavg, rtime, bounds,
                                                                          cent_temp=cent_temp,
                                                                          cent_salt=cent_salt)
                                # Update sla eddy properties
                                elif 'sla' in Eddy.diag_type:
                                    
                                    # See CSS11 section B4
                                    #centlon, centlat = centlon_e, centlat_e
                                    centlon_s, centlat_s = np.atleast_1d(np.copy(centlon_lmi)), \
                                                           np.atleast_1d(np.copy(centlat_lmi))
                                    
                                    centlon, centlat = np.atleast_1d(np.copy(centlon_s)), \
                                                       np.atleast_1d(np.copy(centlat_s))
                                    
                                    if has_ts: # for ocean model
                                        if 'Anticyclonic' in sign_type:
                                            A_list_obj.update_eddy_properties(centlon, centlat,
                                                                              eddy_radius_s, eddy_radius_e,
                                                                              amplitude, Uavg, rtime, bounds,
                                                                              cent_temp=cent_temp,
                                                                              cent_salt=cent_salt)
                                        elif 'Cyclonic' in sign_type:
                                            C_list_obj.update_eddy_properties(centlon, centlat,
                                                                              eddy_radius_s, eddy_radius_e,
                                                                              amplitude, Uavg, rtime, bounds,
                                                                              cent_temp=cent_temp,
                                                                              cent_salt=cent_salt)
                                    else: # AVISO
				        #tt = time.time()
                                        if 'Anticyclonic' in sign_type:
                                            A_list_obj.update_eddy_properties(centlon, centlat,
                                                                              eddy_radius_s, eddy_radius_e,
                                                                              amplitude, Uavg, rtime, bounds)
                                        elif 'Cyclonic' in sign_type:
                                            C_list_obj.update_eddy_properties(centlon, centlat,
                                                                              eddy_radius_s, eddy_radius_e,
                                                                              amplitude, Uavg, rtime, bounds)
                                        #print '------- updated properties i %s seconds' %(time.time() - tt)

                                # Mask out already found eddies
                                if 'Q' in Eddy.diag_type:
                                    mask_c = nx.points_inside_poly(np.array([rectlon, rectlat]).T,
                                                                  np.array([hcirclon, hcirclat]).T)
                                    xi[jmin:jmax, imin:imax].flat[mask_c] = Eddy.fillval
                                    
                                elif 'sla' in Eddy.diag_type:
                                    Eddy.sla[jmin:jmax, imin:imax].flat[mask_e] = Eddy.fillval
                                
                                
                                # Debug. Figure showing individual contours with centroid
                                # Note: do not use with ipython
                                if 0 and np.any(Eddy.sla[jmin:jmax, imin:imax].flat[mask_c] == Eddy.fillval):
                                    plt.figure(100)
                                    jjj, iii = grd.lon()[jstr:jend,istr:iend][jmin:jmax,imin:imax].shape
                                    if not(iii == 1 or jjj == 1):
                                        plt.plot(contlon_e, contlat_e, 'm.-')
                                        plt.plot(contlon_s, contlat_s, 'g.-')
                                        plt.plot(circlon, circlat, c='m')
                                        xpcol, ypcol = pcol_2dxy(
                                            grd.lon()[jstr:jend,istr:iend][jmin:jmax+1,imin:imax+1],
                                            grd.lat()[jstr:jend,istr:iend][jmin:jmax+1,imin:imax+1])
                                        if 'Q' in Eddy.diag_type:
                                            plt.title('Shp.err. %s, Q %s' %(aerr, CS.levels[collind]))
                                            plt.pcolormesh(xpcol, ypcol, xi[jmin:jmax+1,imin:imax+1], cmap=cmap, shading='faceted')
                                            plt.clim(-np.max(np.abs(xicopy[jmin:jmax+1,imin:imax+1])),
                                                      np.max(np.abs(xicopy[jmin:jmax+1,imin:imax+1])))
                                        elif 'sla' in Eddy.diag_type:     
                                            plt.title('Shp.err. %s, SLA %s' %(aerr, CS.levels[collind]))
                                            pcm = plt.pcolormesh(xpcol, ypcol, np.ma.masked_equal(
                                                      Eddy.sla[jmin:jmax+1,imin:imax+1], Eddy.fillval), cmap=plt.cm.bwr, shading='faceted')
                                            plt.scatter(grd.lon()[jstr:jend,istr:iend][jmin:jmax,imin:imax].ravel(),
                                                        grd.lat()[jstr:jend,istr:iend][jmin:jmax,imin:imax].ravel(), s=5, c='k')
                                            plt.clim(np.min(Eddy.slacopy[jmin:jmax+1,imin:imax+1]),
                                                     np.max(Eddy.slacopy[jmin:jmax+1,imin:imax+1]))
                                        plt.axis('image')
                                        plt.colorbar(pcm, orientation='horizontal')
                                        plt.scatter(centlon_e, centlat_e, s=150, c='m')
                                        plt.scatter(centlon_s, centlat_s, s=50, c='g')
                                        plt.scatter(centlon, centlat, s=50, c='r')
                                        print centlon_e, centlat_e
                                        plt.show()
                                        
                                            
                                # Debug. Figure showing full domain and identified eddies after processing
                                # of one entire record
                                # Note: do not use with ipython
                                do_fig_250 = False
                                if do_fig_250:
                                    fig250 = plt.figure(250)
                                    jjj, iii = grd.lon()[jstr:jend,istr:iend][jmin:jmax,imin:imax].shape
                                                
                                    if np.logical_and(iii != 1, jjj != 1):
                                        circx, circy = Eddy.M(circlon, circlat)
                                        Eddy.M.plot(cx, cy, 'g')
                                        Eddy.M.plot(circx, circy, 'g')
                                        xpcol, ypcol = pcol_2dxy(grd.lon()[jstr:jend,istr:iend][jmin:jmax+1,imin:imax+1],
                                                              grd.lat()[jstr:jend,istr:iend][jmin:jmax+1,imin:imax+1])
                                        xpcol, ypcol = Eddy.M(xpcol, ypcol)
                                        try:
                                            pcol200 = Eddy.M.pcolormesh(xpcol, ypcol, xi[jmin:jmax+1,imin:imax+1],
                                                                        cmap=plt.cm.bwr, shading='faceted')
                                            plt.clim(-np.max(np.abs(xicopy[jmin:jmax+1,imin:imax+1])),
                                                      np.max(np.abs(xicopy[jmin:jmax+1,imin:imax+1])))
                                        except:
                                            #pcol200 = np.ma.masked_equal()
                                            pcol200 = Eddy.M.pcolormesh(xpcol, ypcol, Eddy.slacopy[jmin:jmax+1,imin:imax+1],
                                                                        cmap=plt.cm.RdBu_r, shading='faceted')
                                            pcol200.set_clim(-20, 20)
                                        centx, centy = Eddy.M(centlon_e, centlat_e)
                                        Eddy.M.scatter(centx, centy, s=50, c='g', zorder=9)
                                        centx, centy = Eddy.M(centlon_s, centlat_s)
                                        Eddy.M.scatter(centx, centy, s=50, c='w', zorder=10)
                                        contx, conty = Eddy.M(contlon_s, contlat_s)
                                        Eddy.M.plot(contx, conty, c='k')
                                        
    if 'do_fig_250' in locals():
        if do_fig_250:
            print Eddy.new_time_tmp
            title = dt.num2date(Eddy.new_time_tmp[-1])
            plt.title(sign_type + 's ' + str(title.day) + '/' + str(title.month) + '/' + str(title.year))
            Eddy.M.drawcoastlines()
            Eddy.M.fillcontinents('k')
            plt.show()
        
    # Leave collection_loop
    if 'sla' in Eddy.diag_type:
        if C_list_obj is None:
            return A_list_obj
        elif A_list_obj is None:
            return C_list_obj
    else:
        return A_list_obj, C_list_obj




def track_eddies(Eddy, first_record):
    '''
    Track the eddies. First a distance matrix is calculated between all new and old eddies.
    Then loop through each old eddy, sorting the distances and selecting that/those within
    range.
    This is not so slow, surprisingly, it's the main loops over the contours which are slow
    '''
    
    #print 'Entering track_eddies'
    
    dist0 = Eddy.dist0
    amp0  = Eddy.amp0
    area0 = Eddy.area0
    #temp0 = 20.
    #salt0 = 35.
    
    # True to debug
    debug_dist = False
    
    # We will need these in m for ellipse below
    old_x, old_y = Eddy.M(Eddy.old_lon, Eddy.old_lat)
    new_x, new_y = Eddy.M(Eddy.new_lon_tmp, Eddy.new_lat_tmp)
    
    # Uncomment 3 lines below and comment Haversine lines
    # further below to use scipy.spatial Euclidean distance
    #X_old = np.array([old_x, old_y]).T
    #X_new = np.array([new_x, new_y]).T
    #dist_mat = spatial.distance.cdist(X_old, X_new, 'euclidean')
    
    
    X_old = np.array([Eddy.old_lon, Eddy.old_lat]).T
    X_new = np.array([Eddy.new_lon_tmp, Eddy.new_lat_tmp]).T
    # Use haversine distance for distance matrix between every old and new eddy
    #dist_mat = spatial.distance.cdist(X_old, X_new, lambda u, v: haversine_cdist(u, v))
    # We use f2py functions from file haversine_distmat.f90
    #  to get the distance matrix, dist_mat
    dist_mat = np.asfortranarray(np.empty((X_old.shape[0], X_new.shape[0])))
    #print 'before'
    hav.haversine_distmat(np.asfortranarray(X_old),
                          np.asfortranarray(X_new),
                          np.asfortranarray(dist_mat))
    dist_mat_copy = np.copy(dist_mat)
    
    new_eddy_inds = np.ones(Eddy.new_lon_tmp.size, dtype=bool)
    new_eddy = False
    
    # Loop over the old eddies looking for active eddies
    #for old_ind in np.nditer(np.arange(dist_mat.shape[0])):
    for old_ind in np.arange(dist_mat.shape[0]):
        dist_arr = np.array([])
        new_ln = np.array([])
        new_lt = np.array([])
        new_rd_s = np.array([])
        new_rd_e = np.array([])
        new_am = np.array([])
        new_Ua = np.array([])
        new_tm = np.array([])
        new_tp = np.array([])
        new_st = np.array([])
        new_bnds = np.array([], dtype=np.int16)
        new_inds = np.array([], dtype=np.int16)
        backup_ind = np.array([], dtype=np.int16)
        backup_dist = np.array([])
        
        # See http://stackoverflow.com/questions/11528078/determining-duplicate-values-in-an-array
        non_unique_inds = np.setdiff1d(np.arange(len(dist_mat[old_ind])),
                                           np.unique(dist_mat[old_ind],
                                                    return_index=True)[1])
        # Move non_unique_inds far away
        dist_mat[old_ind][non_unique_inds] = 1e9 # km
        
        if debug_dist:
            plt.clf()
            plt.subplot(131)
            plt.imshow(dist_mat_copy, interpolation='none',origin='lower', cmap=plt.cm.Accent)
            plt.clim(0, 5e6)
            plt.colorbar()
            plt.xlabel('new')
            plt.ylabel('old')
            plt.title('original')
        

        # Make an ellipse at current old_eddy location
        # (See CSS11 sec. B4, pg. 208)
        if 'ellipse' in Eddy.separation_method:
            
            ellipse_path = Eddy.search_ellipse.get_search_ellipse(
                                             old_x[old_ind], old_y[old_ind])
            
            
        # Loop over separation distances between old and new
        for new_ind, new_dist in enumerate(dist_mat[old_ind]):
            
            if 'ellipse' in Eddy.separation_method:
                if Eddy.search_ellipse.ellipse_path.contains_point(
                                          (new_x[new_ind], new_y[new_ind])):
                    sep_proceed = True
                else:
                    sep_proceed = False
            
            elif 'sum_radii' in Eddy.separation_method: # Use separation distance method
                sep_dist = Eddy.new_radii_tmp_e[new_ind]
                sep_dist += Eddy.old_radii_e[old_ind]
                sep_dist *= Eddy.sep_dist_fac
                sep_proceed = new_dist <= sep_dist
            
            else:
                Exception
            
            # Pass only the eddies within ellipse or sep_dist
            if sep_proceed:
                oamp = Eddy.old_amp[old_ind]
                namp = Eddy.new_amp_tmp[new_ind]
                
                # Following CSS11, we use effective radius rather than speed based...
                oarea = np.power(Eddy.old_radii_s[old_ind], 2)
                oarea *= np.pi
                narea = np.power(Eddy.new_radii_tmp_s[new_ind], 2)
                narea *= np.pi
                                
                # Pass only the eddies within min and max times old amplitude and area
                # KCCMC11 and CSS11 use 0.25 and 2.5, respectively
                if np.logical_and(np.logical_and( namp >= Eddy.evolve_ammin * oamp,
                                                  namp <= Eddy.evolve_ammax * oamp),
                                  np.logical_and(narea >= Eddy.evolve_armin * oarea,
                                                 narea <= Eddy.evolve_armax * oarea)):

                    dist_arr = np.r_[dist_arr, new_dist]
                    new_ln = np.r_[new_ln, Eddy.new_lon_tmp[new_ind]]
                    new_lt = np.r_[new_lt, Eddy.new_lat_tmp[new_ind]]
                    new_rd_s = np.r_[new_rd_s, Eddy.new_radii_tmp_s[new_ind]]
                    new_rd_e = np.r_[new_rd_e, Eddy.new_radii_tmp_e[new_ind]]
                    new_am = np.r_[new_am, Eddy.new_amp_tmp[new_ind]]
                    new_Ua = np.r_[new_Ua, Eddy.new_Uavg_tmp[new_ind]]
                    new_tm = np.r_[new_tm, Eddy.new_time_tmp[new_ind]]
                    if 'ROMS' in Eddy.datatype:
                        new_tp = np.r_[new_tp, Eddy.new_temp_tmp[new_ind]]
                        new_st = np.r_[new_st, Eddy.new_salt_tmp[new_ind]]
                    try:
                        new_bnds = np.vstack((new_bnds, Eddy.new_bounds_tmp[new_ind]))
                    except Exception:
                        new_bnds = np.hstack((new_bnds, Eddy.new_bounds_tmp[new_ind]))
                    new_inds = np.r_[new_inds, new_ind]
                    backup_ind = np.r_[backup_ind, new_ind]
                    
                    # An old (active) eddy has been detected, so
                    #   corresponding new_eddy_inds set to False
                    new_eddy_inds[np.nonzero(Eddy.new_lon_tmp == 
                                             Eddy.new_lon_tmp[new_ind])] = False
                    dist_mat[:,new_ind] = 1e9 # km
        
        #print 'loop22'
        #print 'ifs1'
        # Only one eddy within range
        if dist_arr.size == 1:                
            # Nothing to be done except update the eddy tracks
            # NOTE accounting should be part of Eddy object...
            if 'ROMS' in Eddy.datatype:
                Eddy = accounting(Eddy, old_ind, new_ln, new_lt, new_rd_s, new_rd_e,
                                  new_am, new_Ua, new_tm, new_bnds, new_eddy, first_record,
                                  new_tp, new_st)
            elif 'AVISO' in Eddy.datatype:
                Eddy = accounting(Eddy, old_ind, new_ln, new_lt, new_rd_s, new_rd_e,
                                  new_am, new_Ua, new_tm, new_bnds, new_eddy, first_record)
        
        # More than one eddy within range
        elif dist_arr.size > 1:
            
            # Loop to find the right eddy
            delta_area = np.array([])
            delta_amp = np.array([])
            delta_temp = np.array([])
            delta_salt = np.array([])
            
            for i in np.nditer(np.arange(dist_arr.size)):
                
                # Choice of using effective or speed-based...
                delta_area = np.r_[delta_area, np.abs(np.diff([np.pi*(Eddy.old_radii_e[old_ind]**2),
                                                               np.pi*(new_rd_e[i]**2)]))]
                delta_amp = np.r_[delta_amp, np.abs(np.diff([Eddy.old_amp[old_ind], new_am[i]]))]
                if 'ROMS' in Eddy.datatype:
                    delta_temp = np.r_[delta_temp,
                                       np.abs(np.diff([Eddy.old_temp[old_ind], new_tp[i]]))]
                    delta_salt = np.r_[delta_salt,
                                       np.abs(np.diff([Eddy.old_salt[old_ind], new_st[i]]))]
        
            # This from Penven etal (2005)
            #deltaX = np.sqrt(np.power(dist_arr / dist0, 2) +
                             #np.power(delta_amp / amp0, 2) +
                             #np.power(delta_area / area0, 2) +
                             #np.power(delta_temp / temp0, 2) +
                             #np.power(delta_salt / salt0, 2))
            #deltaX = np.sqrt(np.power(delta_amp / amp0, 2) +
            #                 np.power(delta_temp / temp0, 2) +
            #                 np.power(delta_salt / salt0, 2))
            #deltaX = np.sqrt(np.power(delta_temp / temp0, 2) +
            #                 np.power(delta_salt / salt0, 2))
            #deltaX = np.sqrt(np.power(dist_arr / dist0, 2) +
            #                 np.power(delta_amp / amp0, 2))
            #deltaX = np.sqrt(np.power(delta_area / area0, 2) +
            #                 np.power(delta_amp / amp0, 2))
            #deltaX = ne.evaluate('sqrt((delta_area / area0)**2 + (delta_amp / amp0)**2)')
            deltaX = ne.evaluate('sqrt((delta_area / area0)**2 + (delta_amp / amp0)**2 + (dist_arr / dist0)**2)')

            dx = deltaX.argsort()
            dx0 = dx[0] # Index to the right eddy
            dx_unused = dx[1:] # Index/indices to the unused eddy/eddies
            
            # Update eddy track dx0
            if 'ROMS' in Eddy.datatype:
                Eddy = accounting(Eddy, old_ind,
                                  new_ln[dx0], new_lt[dx0], new_rd_s[dx0], new_rd_e[dx0],
                                  new_am[dx0], new_Ua[dx0], new_tm[dx0],
                                  new_bnds[dx0], new_eddy, first_record,
                                  new_tp[dx0], new_st[dx0])
            elif 'AVISO' in Eddy.datatype:
                
                Eddy = accounting(Eddy, old_ind,
                                  new_ln[dx0], new_lt[dx0], new_rd_s[dx0], new_rd_e[dx0],
                                  new_am[dx0], new_Ua[dx0], new_tm[dx0],
                                  new_bnds[dx0], new_eddy, first_record)
            
            if debug_dist:
                plt.subplot(132)
                plt.imshow(dist_mat.copy(), interpolation='none',
                           origin='lower', cmap=plt.cm.Accent)
                plt.clim(0, 5e6)
                plt.colorbar()
                plt.xlabel('new')
                plt.ylabel('old')
                plt.title('before')
            
            # Use backup_ind to reinsert distances into dist_mat for the unused eddy/eddies
            for i, bind in enumerate(backup_ind[dx_unused]):
                dist_mat[:,bind] = dist_mat_copy[:,bind]
            
            if debug_dist:
                print 'backup_ind[dx_unused].shape',backup_ind[dx_unused].shape
                print 'backup_ind[dx_unused]',backup_ind[dx_unused]
                plt.subplot(133)
                plt.imshow(dist_mat.copy(), interpolation='none',
                           origin='lower', cmap=plt.cm.Accent)
                plt.clim(0, 5e6)
                plt.colorbar()
                plt.xlabel('new')
                plt.ylabel('old')
                plt.title('after')
                plt.tight_layout()
                plt.show()
    
        #print 'ifs2'
    #print 'out2'        
    # Finished looping over old eddy tracks
    
    # Now we need to add new eddies defined by new_eddy_inds
    if np.any(new_eddy_inds):
        if False:#Eddy.verbose:
            print '------adding %s new eddies' %new_eddy_inds.sum()
            
        for neind, a_new_eddy in enumerate(new_eddy_inds):
                        
            if a_new_eddy: # Update the eddy tracks
            
                if 'ROMS' in Eddy.datatype:
                    Eddy = accounting(Eddy, False,
                                      Eddy.new_lon_tmp[neind], 
                                      Eddy.new_lat_tmp[neind],
                                      Eddy.new_radii_tmp_s[neind],
                                      Eddy.new_radii_tmp_e[neind],
                                      Eddy.new_amp_tmp[neind],
                                      Eddy.new_Uavg_tmp[neind],
                                      Eddy.new_time_tmp[neind],
                                      Eddy.new_bounds_tmp[neind], True, False,
                                      Eddy.new_temp_tmp[neind],
                                      Eddy.new_salt_tmp[neind])
                elif 'AVISO' in Eddy.datatype:
		    #print 'Eddy.new_lon_tmp[neind]',Eddy.new_lon_tmp[neind]
		    #print 'CCCC'
                    Eddy = accounting(Eddy, False,
                                      Eddy.new_lon_tmp[neind], 
                                      Eddy.new_lat_tmp[neind],
                                      Eddy.new_radii_tmp_s[neind],
                                      Eddy.new_radii_tmp_e[neind],
                                      Eddy.new_amp_tmp[neind],
                                      Eddy.new_Uavg_tmp[neind],
                                      Eddy.new_time_tmp[neind],
                                      Eddy.new_bounds_tmp[neind], True, False)  
    #print 'Leaving track_eddies'
    return Eddy





def accounting(Eddy, old_ind, centlon, centlat,
               eddy_radius_s, eddy_radius_e, amplitude, Uavg, rtime, 
               bounds, new_eddy, first_record, cent_temp=None, cent_salt=None):
    '''
    Accounting for new or old eddy (either cyclonic or anticyclonic)
      Eddy  : eddy_tracker.tracklist object
      old_ind    : index to current old 
      centlon, centlat: arrays of lon/lat centroids
      eddy_radius_s:  speed-based eddy radius
      eddy_radius_e:  effective eddy radius from fit_circle
      amplitude  :  eddy amplitude/intensity (max abs vorticity/f in eddy)
      Uavg       :  average velocity within eddy contour
      rtime      :  ROMS time in seconds
      cent_temp  :  array of temperature at centroids
      cent_salt  :  array of salinity at centroids
      bounds     :  array(imin,imax,jmin,jmax) defining location of eddy
      new_eddy   : flag indicating a new eddy
      first_record : flag indicating that we're on the first record
    '''
    #print 'Entering accounting'
    if first_record: # is True then all eddies are new...
        if Eddy.verbose:
            print '------ writing first record'
        new_eddy = True
    
    
    if not new_eddy: # It's an old (ie, active) eddy
        
        # SHOULD BE NO NEED FOR TRY->EXCEPT HERE,
        # IF IT'S AN OLD EDDY IT SHOULD HAVE AN INDEX...
        try: # Works if old_ind is within range of array

            Eddy.new_lon[old_ind] = centlon
            Eddy.new_lat[old_ind] = centlat
            Eddy.new_radii_s[old_ind] = eddy_radius_s
            Eddy.new_radii_e[old_ind] = eddy_radius_e
            Eddy.new_amp[old_ind] = amplitude
            if 'ROMS' in Eddy.datatype:
                Eddy.new_temp[old_ind] = cent_temp
                Eddy.new_salt[old_ind] = cent_salt
            #print 'aa'
        
        except Exception: # ... otherwise we extend the range of array to old_ind
            Eddy.insert_at_index('new_lon', old_ind, centlon)
            Eddy.insert_at_index('new_lat', old_ind, centlat)
            Eddy.insert_at_index('new_radii_s', old_ind, eddy_radius_s)
            Eddy.insert_at_index('new_radii_e', old_ind, eddy_radius_e)
            Eddy.insert_at_index('new_amp', old_ind, amplitude)
            if 'ROMS' in Eddy.datatype:
                Eddy.insert_at_index('new_temp', old_ind, cent_temp)
                Eddy.insert_at_index('new_salt', old_ind, cent_salt)
            #print 'bb',old_ind

        if 'ROMS' in Eddy.datatype:
            Eddy.update_track(old_ind, centlon, centlat, rtime, Uavg,
                              eddy_radius_s, eddy_radius_e, amplitude,
                              bounds, cent_temp, cent_salt)
        
        elif 'AVISO' in Eddy.datatype:
            
            Eddy.update_track(old_ind, centlon, centlat, rtime, Uavg,
                              eddy_radius_s, eddy_radius_e, amplitude,
                              bounds)

    else: # It's a new eddy
        
        # We extend the range of array to old_ind
        Eddy.insert_at_index('new_lon', Eddy.index, centlon)
        Eddy.insert_at_index('new_lat', Eddy.index, centlat)
        Eddy.insert_at_index('new_radii_s', Eddy.index, eddy_radius_s)
        Eddy.insert_at_index('new_radii_e', Eddy.index, eddy_radius_e)
        Eddy.insert_at_index('new_amp', Eddy.index, amplitude)
        if 'ROMS' in Eddy.datatype:
            Eddy.insert_at_index('new_temp', Eddy.index, cent_temp)
            Eddy.insert_at_index('new_salt', Eddy.index, cent_salt)
        
        if Eddy.new_list is True: # Initialise a new list
            print '------ starting a new track list for %ss' %Eddy.sign_type
            Eddy.new_list = False
        if 'ROMS' in Eddy.datatype:
            Eddy.append_list(centlon, centlat, rtime, Uavg,
                eddy_radius_s, eddy_radius_e, amplitude, bounds, cent_temp, cent_salt)
        elif 'AVISO' in Eddy.datatype:
            Eddy.append_list(centlon, centlat, rtime, Uavg,
                eddy_radius_s, eddy_radius_e, amplitude, bounds)
        
        Eddy.index += 1

    return Eddy # Leave accounting



def get_ROMS_data(rfile, pad=None, index=None, sigma_lev=None,
                  istr=None, iend=None, jstr=None, jend=None, diag_type=None):
    nc = netcdf.Dataset(rfile, 'r')
    if pad is None:
        time = nc.variables['ocean_time'][:]
        nc.close()
        return time
    t = nc.variables['temp'][index, sigma_lev, jstr:jend, istr:iend]
    s = nc.variables['salt'][index, sigma_lev, jstr:jend, istr:iend]
    time = nc.variables['ocean_time'][index]
    if 'Q' in diag_type:
        u = nc.variables['u'][index, sigma_lev, jstr:jend, istr:iend-1]
        v = nc.variables['v'][index, sigma_lev, jstr:jend-1, istr:iend]
        nc.close()
        return u, v, t, s, time
    elif 'sla' in diag_type:
        zeta = nc.variables['zeta'][index, jstr:jend, istr:iend]
        nc.close()
        return zeta, t, s, time




def func_hann2d_fast(var, numpasses):
    '''
    The treament of edges and corners in func_HANN2D can be mimicked by
    duplicating sides on to the W,E,S,N of the respective edges and
    replacing corner points with NaN's. Then smoothing can be done in 
    a single step.
    Adapted from equivalent function in ETRACK.
    '''
    jsz, isz = var.shape
    # Interior points  (N,S,W,E)
    nj = jsz + 2
    ni = isz + 2

    def hann2d_fast(var, ni, nj):
        #print 'jsz, isz',jsz, isz
        var_ext = np.ma.zeros((nj, ni)) # add 1-more line parallell to
        var_ext[1:-1, 1:-1] = var        # each of 4-sides
        var_ext[1:-1, 0]    = var[:,0]   # duplicate W-side
        var_ext[1:-1,-1]    = var[:,-1]  # duplicate E-side
        var_ext[0,1:-1]     = var[0,:]   # duplicate N-side
        var_ext[-1,1:-1]    = var[-1,:]  # duplicate S-side
        var_ext[0,0]        = np.nan     # NW-corner
        var_ext[0,-1]       = np.nan     # NE-corner
        var_ext[-1,0]       = np.nan     # SW-corner
        var_ext[-1,-1]      = np.nan     # SE-corner

        # npts is used to count number of valid neighbors    
        npts = ne.evaluate('var_ext * 0. + 1.')
        npts[np.isnan(npts)] = 0.

        # Replace nans with 0 to find a no-nan sum
        var_ext[np.isnan(var_ext)] = 0.

        # Initialize count and sum variables
        cc = np.ma.zeros((var.shape))
        varS = np.ma.zeros((var.shape))
    
        cc = npts[1:nj-1, 1:ni-1] * (npts[0:nj-2, 1:ni-1] + npts[2:nj,   1:ni-1] +
                                     npts[1:nj-1, 0:ni-2] + npts[1:nj-1, 2:ni])
                                 
        varS =  (var_ext[0:nj-2, 1:ni-1] + var_ext[2:nj,   1:ni-1] + 
                 var_ext[1:nj-1, 0:ni-2] + var_ext[1:nj-1, 2:ni])

        cc[cc == 0] = np.nan # Bring back NaN points in original data.
        weight = ne.evaluate('8. - cc') # This is the weight for values on each grid point,
                                        # based on number of valid neighbours 
        # Final smoothed version of var
        hsm = 0.125 * (varS + weight * var_ext[1:jsz+1, 1:isz+1])
        return hsm
        
    for i in np.arange(numpasses):
        var = hann2d_fast(var, ni, nj)
    
    return var


def get_circle(x0, y0, r, npts):
    '''
    Return points on a circle, with specified (x0,y0) center and radius
                    (and optional number of points too!).
  
    Input     : 1  - x0, scalar, center X of circle
                2  - y0, scalar, center Y of circle
                3  - r,  scalar, radius
                4  - npts, scalar, number of points (optional)

    Output    : 1  - cx, circle x-points
                2  - cy, circle y-points

    Example   :  [cx cy] = func_get_circle (5, 5, 3, 256)
                plot (cx, cy, '-')

    Written By : UCLA ROMS Team (jaison@atmos.ucla.edu) 
    Written On : June/05/2008
    Tool       : Eddy Tracker
    '''
    theta = np.arange(npts)
    theta = ne.evaluate('theta * 2. * (4. * arctan(1.)) / npts')
    cx  = ne.evaluate('x0 + r * cos(theta)')
    cy  = ne.evaluate('y0 + r * sin(theta)')
    return cx, cy


