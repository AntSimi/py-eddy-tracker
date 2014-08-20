# -*- coding: utf-8 -*-
# %run make_eddy_track.py

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

make_eddy_track_ROMS.py

Version 1.3.0



===========================================================================
"""

from py_eddy_tracker_classes import *
from make_eddy_tracker_list_obj import *
from make_eddy_track_AVISO import *
from dateutil import parser
import roms_grid as rg



#################################################################################
if __name__ == '__main__':
    
    plt.close('all')
    
    #----------------------------------------------------------------------------
    # Some user-defined input...

    # Specify the AVISO domain
    the_domain = 'Canary_Islands'
    #the_domain = 'BlackSea'
    #the_domain = 'MedSea' # not yet implemented
    
    
    
    directory = '/marula/emason/runs2009/na_2009_7pt5km/'
    #directory = '/marula/emason/runs2012/na_7pt5km/'
    #directory = '/marula/emason/runs2013/canwake4km/'


    #savedir = directory
    savedir = '/marula/emason/runs2009/na_2009_7pt5km/eddy_tracking_exps/sla_1/'

    # True to save outputs in same style as CSS11
    chelton_style_nc = True
    
    print 'savedir', savedir
    
    
    # Choose type of diagnostic 
    # Presently one of q-parameter (Q) or sea level anomaly (sla)
    #diag_type = 'Q'
    diag_type = 'sla'
    
    # Calculate Q using geostrophic velocity from zeta
    geos = True

    if 'sla' in diag_type:
        #sshfile = '/marula/emason/runs2012/na_7pt5km/seasmean.nc'
        sshfile = '/marula/emason/runs2009/na_2009_7pt5km/seasmean.nc'
        #sshfile = '/marula/emason/runs2013/canwake4km/seasmean.nc'

    Ymin         = 11
    Ymax         = 29
    Mmin         = 1
    Mmax         = 12
    #model        = 'ip_roms'
    model        = 'roms'
    filetype     = 'avg'
    sigma_lev    = -1  # -1 for surface
    
    days_btwn_recs = 3.
    
    # Set grid
    grdname = 'roms_grd_NA2009_7pt5km.nc'
    #grdname = 'cb_2009_3km_grd_smooth.nc'
    #grdname = 'grd_canwake4km.nc'
    
    pad = 2
    
    
    # Min and max permitted eddy radii [m]
    if 'Q' in diag_type:
        radmin = 15000.
        radmax = 250000.
        ampmin = 0.02
        ampmax = 100.
        
    elif 'sla' in diag_type:
        # radmin, radmax in degrees
        # NOTE: values 0.4 and 4.461 on 0.25 grid will give
        # 8 and 1000 pixels, as used by CSS11
        radmin = 0.4 #0.135 #0.4
        radmax = 4.461
        ampmin = 1. # cm
        ampmax = 150.

    # Obtain this file from:
    #  http://www-po.coas.oregonstate.edu/research/po/research/rossby_radius/
    rw_path = '/home/emason/data_tmp/chelton_eddies/rossrad.dat'
    
    # Option to track using speed-based radius or effective radius
    #track_with_speed = True
    
    # Make figures for animations
    anim_figs = True


    # Define contours
    if 'Q' in diag_type: # Set Q contours
        qparameter = np.linspace(0, 5*10**-11, 25)
        
    elif 'sla' in diag_type: # Units, cm; for anticyclones use -50 to 50
        slaparameter = np.arange(-100., 101., 1.0)
    
    
    # Apply a filter to the Q parameter
    #smoothing = 'Hanning'
    smoothing = 'Gaussian'
    if 'smoothing' not in locals():
        smooth_fac = False
    elif 'Hanning' in smoothing: # apply Hanning filter
        smooth_fac = 5 # number of passes
    elif 'Gaussian' in smoothing: # apply Gaussian filter
        smooth_fac = 'Deprecated'
        zwl = 20. # degrees, zonal wavelength (see CSS11)
        mwl = 10. # degrees, meridional wavelength
    else:
        error
    
    # Save only tracks longer than...     
    track_duration_min = 5 # days



    subdomain = True
    if the_domain in 'Canary_Islands':
    
        lonmin = -38.     # Canary
        lonmax = -5.5
        latmin = 20.5
        latmax = 38.5
    
    # for canbas4
    #lonmin = -36.
    #lonmax = -8.5
    #latmin = 18.
    #latmax = 33.2

    #lonmin = -25
    #lonmax = -14
    #latmin = 26
    #latmax = 29


    # Typical parameters
    dist0 = 25000.
    if 'Q' in diag_type:
        amp0 = 0.02 # vort/f
    elif 'sla' in diag_type:
        amp0 = 2 # 4 cm
    area0 = np.pi*(60000.**2)
    temp0 = 20.
    salt0 = 35.
    
    # Parameters used by CSS11 and KCCMC11 (Sec. 3.2) to ensure the slow evolution
    # of the eddies over time; they use min and max values of 0.25 and 2.5
    evolve_ammin = 0.05 #0.25 #0.15 # min change in amplitude
    evolve_ammax = 5 #2.5 #5.  # max change in amplitude
    evolve_armin = 0.05 #0.25 #0.15 # min change in area
    evolve_armax = 5 #2.5 #5.  # max change in area
    
    
    separation_method = 'ellipse' # see CSS11
    #separation_method = 'sum_radii' # see KCCMC11
    
    if 'sum_radii' in separation_method:
        # Separation distance factor. Adjust according to number of days between records
        # For 7 days, CSS11 use 150 km search ellipse
        # So, given typical eddy radius of r=50 km, 1.5 * (r1 + r2) = 150 km.
        #sep_dist_fac = 1.0
        sep_dist_fac = 1.15 # Seems ok for AVISO 7-day
        #sep_dist_fac = 1.5 # Causes tracks to jump for AVISO 7-days
    
    #cmap = plt.cm.Spectral_r
    #cmap = plt.cm.jet
    cmap = plt.cm.RdBu

    
    verbose = False

    
    # Note this is a global value
    fillval = -9999
    
    #----------------------------------------------------------------------------
    
    assert Mmin >= 1 and Mmin <= 12, 'Mmin must be within range 1 to 12'
    assert Mmax >= 1 and Mmax <= 12, 'Mmax must be within range 1 to 12'
    
    assert fillval < -5555., 'fillval must be << 0'
    assert diag_type in ('sla', 'Q'), 'diag_type not properly defined'
    
    # Indices to seasons (from months)
    ssns = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    
    if 'Q' in diag_type:
        
        # Search from 5e-11 onwards with fine spacing near the lower end
        qparameter = np.power(np.linspace(0., np.sqrt(qparameter.max()),
                              qparameter.size), 2)[::-1]
        
        # The shape error can maybe be played with...
        #shape_err = np.power(np.linspace(200., 40,  qparameter.size), 2) / 100.
        shape_err = np.power(np.linspace(85., 40,  qparameter.size), 2) / 100.
        shape_err[shape_err < 35.] = 35.
        #shape_err = 35. * np.ones(qparameter.size)
        
    elif 'sla' in diag_type:
        shape_err = 55. * np.ones(slaparameter.size)
        #shape_err = np.power(np.linspace(85., 40,  slaparameter.size), 2) / 100.
        #shape_err[shape_err < 50.] = 50.
    
    
    
    # Get grid
    grd = rg.RomsGrid(grdname, lonmin, lonmax, latmin, latmax, pad)
    
    
    # Get mean sea level from seasmean.nc
    if 'sla' in diag_type:
        with netcdf.Dataset(sshfile) as nc:
            sshmean = nc.variables['zeta'][-1, grd.jp0:grd.jp1, grd.ip0:grd.ip1]
    
                                                   
    # Create Basemap instance for Mercator projection.
    M = Basemap(projection='merc', llcrnrlon  = lonmin,  \
                                    urcrnrlon  = lonmax,  \
                                    llcrnrlat  = latmin,  \
                                    urcrnrlat  = latmax,  \
                                    lat_ts     = 28.,     \
                                    resolution = 'h') # 'h'-high, 'l'-low

    Mx, My = M(grd.lon(), grd.lat())
    pMx, pMy = pcol_2dxy(Mx, My)

                    
    # Instantiate search ellipse object
    search_ellipse = eddy_tracker.SearchEllipse('ROMS', days_btwn_recs,
                                    rw_path, [lonmin, lonmax, latmin, latmax])
    
    

    if 'Gaussian' in smoothing and 'sla' in diag_type:
        # Get parameters for ndimage.gaussian_filter
        zres, mres = gaussian_resolution(grd.get_resolution(), zwl, mwl)
        
        # Rotate to account for grid rotation
        zres, mres = grd.rotate_vec(zres, mres)
        zres, mres = zres.mean(), mres.mean()
    

    fig  = plt.figure(1)
    #axis = plt.axes()

    ## Initialise
    # Objects to hold data (qparameter)
    A_eddy = eddy_tracker.track_list('ROMS', track_duration_min)
    C_eddy = eddy_tracker.track_list('ROMS', track_duration_min)
    
    if 'Q' in diag_type:
        A_savefile = "".join([savedir, 'eddy_tracks_anticyclonic.nc'])
        A_eddy.qparameter = qparameter
        A_eddy.shape_err = shape_err
        C_savefile = "".join([savedir, 'eddy_tracks_cyclonic.nc'])
        C_eddy.qparameter = qparameter
        C_eddy.shape_err = shape_err  
        A_eddy.radmin = np.float(radmin)
        A_eddy.radmax = np.float(radmax)
        C_eddy.radmin = np.float(radmin)
        C_eddy.radmax = np.float(radmax)
        
    elif 'sla' in diag_type:
        A_savefile = "".join([savedir, 'eddy_tracks_SLA_anticyclonic.nc'])
        A_eddy.slaparameter = slaparameter
        A_eddy.shape_err = shape_err
        C_savefile = "".join([savedir, 'eddy_tracks_SLA_cyclonic.nc'])
        C_eddy.slaparameter = slaparameter[::-1]
        C_eddy.shape_err = shape_err[::-1]   
        # See CSS11 section B2 (0.4 degree radius)
        pixmin = np.round((np.pi * radmin**2) / grd.get_resolution()**2)
        pixmax = np.round((np.pi * radmax**2) / grd.get_resolution()**2)
        print '--- Pixel range = %s-%s' %(np.int(pixmin), np.int(pixmax))
        A_eddy.pixel_threshold = [pixmin, pixmax]
        C_eddy.pixel_threshold = [pixmin, pixmax]
        A_eddy.radmin = np.float(grd.get_resolution(radmin))
        A_eddy.radmax = np.float(grd.get_resolution(radmax))
        C_eddy.radmin = np.float(grd.get_resolution(radmin))
        C_eddy.radmax = np.float(grd.get_resolution(radmax))
    
    
    A_eddy.ampmin = np.float(ampmin)
    A_eddy.ampmax = np.float(ampmax)
    C_eddy.ampmin = np.float(ampmin)
    C_eddy.ampmax = np.float(ampmax)
    
    A_eddy.interannual = False
    C_eddy.interannual = False
    
    A_eddy.chelton_style_nc = chelton_style_nc
    C_eddy.chelton_style_nc = chelton_style_nc
    
    A_eddy.diag_type = diag_type
    C_eddy.diag_type = diag_type
    
    A_eddy.smoothing = smoothing
    C_eddy.smoothing = smoothing
    A_eddy.smooth_fac = smooth_fac
    C_eddy.smooth_fac = smooth_fac
    
    A_eddy.M = M
    C_eddy.M = M
    
    #A_eddy.rwv = rwv
    #C_eddy.rwv = rwv
    A_eddy.search_ellipse = search_ellipse
    C_eddy.search_ellipse = search_ellipse
    
    A_eddy.separation_method = separation_method
    C_eddy.separation_method = separation_method
    
    if 'sum_radii' in separation_method:
        A_eddy.sep_dist_fac = sep_dist_fac
        C_eddy.sep_dist_fac = sep_dist_fac
    
    A_eddy.evolve_ammin = np.float(evolve_ammin)
    A_eddy.evolve_ammax = np.float(evolve_ammax)
    A_eddy.evolve_armin = np.float(evolve_armin)
    A_eddy.evolve_armax = np.float(evolve_armax)
    
    C_eddy.evolve_ammin = np.float(evolve_ammin)
    C_eddy.evolve_ammax = np.float(evolve_ammax)
    C_eddy.evolve_armin = np.float(evolve_armin)
    C_eddy.evolve_armax = np.float(evolve_armax)

    A_eddy.i0, A_eddy.i1 = grd.i0, grd.i1
    A_eddy.j0, A_eddy.j1 = grd.j0, grd.j1
    C_eddy.i0, C_eddy.i1 = grd.i0, grd.i1
    C_eddy.j0, C_eddy.j1 = grd.j0, grd.j1
    
    A_eddy.lonmin, A_eddy.lonmax = np.float(lonmin), np.float(lonmax)
    A_eddy.latmin, A_eddy.latmax = np.float(latmin), np.float(latmax)
    C_eddy.lonmin, C_eddy.lonmax = np.float(lonmin), np.float(lonmax)
    C_eddy.latmin, C_eddy.latmax = np.float(latmin), np.float(latmax)
    
    A_eddy.fillval = fillval
    C_eddy.fillval = fillval
    A_eddy.verbose = verbose
    C_eddy.verbose = verbose

    
    A_eddy.area0 = np.float(area0)
    C_eddy.area0 = np.float(area0)
    A_eddy.amp0 = np.float(amp0)
    C_eddy.amp0 = np.float(amp0)
    A_eddy.dist0 = np.float(dist0)
    C_eddy.dist0 = np.float(dist0)

    start = 0
    start_time = time.time()
    
    # Loop thru the years...
    for Yr in np.arange(Ymin, Ymax+1):
        if Yr == Ymin: mo_min = Mmin
        else:       mo_min = 1
        if Yr == Ymax: mo_max = Mmax
        else:       mo_max = 12
        # Loop thru the months...
        for Mo in np.arange(mo_min, mo_max+1):
            file_time = time.time()
            filename = directory + model + '_' + filetype + \
                       '_Y' + str(Yr) + 'M' + str(Mo) + '.nc'
            print 'Opening file:', filename
            if start == 0:
                start = 1
                record_range = get_ROMS_data(filename)
                A_eddy.days_btwn_recs = np.squeeze(np.diff(record_range[0:2]) / 86400.)
                C_eddy.days_btwn_recs = np.squeeze(np.diff(record_range[0:2]) / 86400.)
                
                if 'ip_roms' in model:
                    nc = netcdf.Dataset(filename)
                    rho_ntr = nc.variables['rho_ntr'][sigma_lev] + 1027.4
                    nc.close()
                else:
                    rho_ntr = False
                
                # Create nc files for saving
                A_eddy.create_netcdf(directory, A_savefile, 'Anticyclonic', grd=grd,
                                     Ymin=Ymin, Ymax=Ymax, Mmin=Mmin, Mmax=Mmax, model=model,
                                     sigma_lev=sigma_lev, rho_ntr=rho_ntr)
                C_eddy.create_netcdf(directory, C_savefile, 'Cyclonic', grd=grd,
                                     Ymin=Ymin, Ymax=Ymax, Mmin=Mmin, Mmax=Mmax, model=model,
                                     sigma_lev=sigma_lev, rho_ntr=rho_ntr)
                
                #A_eddy.create_netcdf(directory, A_savefile, 'Anticyclonic', grd,
                                     #Ymin, Ymax, Mmin, Mmax, model, sigma_lev, rho_ntr)
                #C_eddy.create_netcdf(directory, C_savefile, 'Cyclonic', grd,
                                     #Ymin, Ymax, Mmin, Mmax, model, sigma_lev, rho_ntr)
                
                
                record_range = record_range.size
                
            for record in np.arange(record_range):
                
                #rec_start_time = time.time()
                print '--- record', record + 1
                
                # Reset holding variables to empty arrays
                A_eddy.reset_holding_variables()
                C_eddy.reset_holding_variables()
                
                
                grdmask = grd.mask()
                
                if 'Q' in diag_type:
                    u, v, temp, salt, rtime = get_ROMS_data(filename, pad, record, sigma_lev,
                                                            ipadstr, ipadend, jpadstr, jpadend, diag_type)
                    # Sort out masking (important for subsurface fields)
                    u = np.ma.masked_outside(u, -10., 10.)
                    v = np.ma.masked_outside(v, -10., 10.)

                    u.data[u.mask] = 0.
                    v.data[v.mask] = 0.
                    u = u.data
                    v = v.data

                    okubo, xi = okubo_weiss(u, v, grd.pm()[jpadstr:jpadend,
                                                           ipadstr:ipadend],
                                                  grd.pn()[jpadstr:jpadend,
                                                           ipadstr:ipadend])
                
                    qparam = np.ma.multiply(-0.25, okubo) # see KCCMC11
                
                    # Remove padded boundary
                    qparam = qparam[junpadstr:junpadend, iunpadstr:iunpadend]
                    xi = xi[junpadstr:junpadend, iunpadstr:iunpadend]
                    u = u[junpadstr:junpadend, iunpadstr:iunpadend]
                    v = v[junpadstr:junpadend, iunpadstr:iunpadend]
                
                    u = u2rho_2d(u)
                    v = v2rho_2d(v)
                    
                    
                    if smoothing:
                        if smoothing == 'Gaussian':
                            qparam = ndimage.gaussian_filter(qparam, smooth_fac, 0)
                        if smoothing == 'Hanning':
                            # smooth_fac passes of 2d Hanning filter
                            #han_time = time.time()
                            qparam = func_hann2d_fast(qparam, smooth_fac)
                            #print 'hanning', str(time.time() - han_time), ' seconds!'
                            xi = func_hann2d_fast(xi, smooth_fac)
                        
                    # Set Q over land to zero
                    qparam *= grdmask
                    #qparam = np.ma.masked_where(grdmask == False, qparam)
                    xi *= grdmask
                    xi = np.ma.masked_where(grdmask == False,
                                            xi / grd.f()[jstr:jend,istr:iend])
                    xicopy = np.ma.copy(xi)
                
                elif 'sla' in diag_type:
                    
                    # Note that sla (zeta) is returned in m
                    sla, rtime = get_ROMS_data(filename, grd, record, sigma_lev, diag_type)
                    
                    # Get anomaly
                    sla -= sshmean
                    
                    if isinstance(smoothing, str):
                        
                        if 'Gaussian' in smoothing:
                            if 'first_record' not in locals():
                                print '--- applying Gaussian high-pass filter to SLA'
                            # Set landpoints to zero
                            sla *= grd.mask()
                            # High pass filter
                            # http://stackoverflow.com/questions/6094957/high-pass-filter-for-image-processing-in-python-by-using-scipy-numpy
                            sla -= ndimage.gaussian_filter(sla, [mres, zres])
                        elif 'Hanning' in smoothing:
                            # smooth_fac passes of 2d Hanning filter
                            #han_time = time.time()
                            sla = func_hann2d_fast(sla, smooth_fac)
                            #print 'hanning', str(time.time() - han_time), ' seconds!'
                    
                    #sla *= grd.mask()
                    sla = np.ma.masked_where(grd.mask() == 0, sla)
                    
                    grd.getSurfGeostrVel(sla)#, grd.f(), grd.pm(), grd.pn(),
                                                #grd.u_mask, grd.v_mask)
                    
                    sla *= 100. # m to cm
                    
                    # Remove padded boundary
                    sla = sla[grd.jup0:grd.jup1, grd.iup0:grd.iup1]
                    #u = grd.u[grd.jup0:grd.jup1, grd.iup0:grd.iup1]
                    #v = grd.v[grd.jup0:grd.jup1, grd.iup0:grd.iup1]
                    
                    # Calculate EKE
                    grd.getEKE()
                    
                    A_eddy.sla = np.ma.copy(sla)
                    C_eddy.sla = np.ma.copy(sla)
                    A_eddy.slacopy = np.ma.copy(sla)
                    C_eddy.slacopy = np.ma.copy(sla)
                
                # Get scalar speed
                Uspd = np.hypot(grd.u, grd.v)
                Uspd = np.ma.masked_where(grd.mask()[grd.jup0:grd.jup1,
                                                     grd.iup0:grd.iup1] == True, Uspd)
                
                A_eddy.Uspd = np.ma.copy(Uspd)
                C_eddy.Uspd = np.ma.copy(Uspd)
                
                
                # Get contours of Q/sla parameter
                if 'first_record' not in locals():
                
                    print '------ getting SLA contours'
                    contfig = plt.figure(99)
                    ax = contfig.add_subplot(111)
                
                    if anim_figs:
                        animfig = plt.figure(999)
                        animax = animfig.add_subplot(111)
                        # Colorbar axis
                        animax_cbar = get_cax(animax, dx=0.03, width=.05, position='b')
                
                if 'Q' in diag_type:
                    CS = plt.contour(grd.lon()[jstr:jend,istr:iend],
                                     grd.lat()[jstr:jend,istr:iend], qparam, qparameter)
                    # Get xi contour field at zero
                    #plt.figure(99)
                    CSxi = plt.contour(grd.lon()[jstr:jend,istr:iend],
                                       grd.lat()[jstr:jend,istr:iend], xi, [0.])
                    #plt.close(99)
                
                elif 'sla' in diag_type:
                    A_CS = ax.contour(grd.lon(), grd.lat(), A_eddy.sla, slaparameter)
                    # Note that CSc is for the cyclonics, slaparameter in reverse order
                    C_CS = ax.contour(grd.lon(), grd.lat(), C_eddy.sla, slaparameter[::-1])
                
                else: Exception
                
                if True: # clear the current axis
                    ax.cla()
                else:
                    # Debug
                    if 'Q' in diag_type:
                        ax.set_title('qparameter and xi')
                        ax.clabel(CS, np.array([qparameter.min(), qparameter.max()]))
                    elif 'sla' in diag_type:
                        ax.set_title('slaparameter')
                        ax.clabel(CS, np.array([slaparameter.min(), slaparameter.max()]))
                    plt.axis('image')
                    #plt.colorbar()
                    plt.show()
                    
                
                # Now we loop over the CS collection
                if 'Q' in diag_type:
                    A_eddy, C_eddy = collection_loop(CS, grd, Uspd, rtime,
                                       A_list_obj=A_eddy, C_list_obj=C_eddy, xi=xi, CSxi=CSxi,
                                       verbose=verbose)
                
                elif 'sla' in diag_type:
		    #print 'rtime 1', rtime
                    A_eddy.sign_type = 'Anticyclonic'
                    A_eddy = collection_loop(A_CS, grd, rtime,
                                             A_list_obj=A_eddy,  C_list_obj=None,
                                             sign_type=A_eddy.sign_type, verbose=verbose)
                    # Note that CSc is reverse order
                    C_eddy.sign_type = 'Cyclonic'
                    C_eddy = collection_loop(C_CS, grd, rtime,
                                             A_list_obj=None, C_list_obj=C_eddy,
                                             sign_type=C_eddy.sign_type, verbose=verbose)
                
                
                
                # All contours now done
                #print 'All contours', time.time() - rec_start_time, 'seconds'
                
                # Debug
                if 'fig250' in locals():
                    
                    plt.figure(250)
                    dy, dm, dd = oceantime2ymd(rtime)
                    dtime = dy + dm + dd
                    tit = 'Y' + dy + 'M' + dm + 'D' + dd
                    
                    if 'Q' in diag_type:
                        plt.title('Q ' + tit)
                        M.pcolormesh(pMx, pMy, xi, cmap=cmap)
                        M.contour(Mx, My, xi, [0.], colors='k', linewidths=0.5)
                        M.contour(Mx, My, qparam, qparameter, colors='w', linewidths=0.3)
                        M.contour(Mx, My, qparam, [qparameter[0]], colors='m', linewidths=0.25)
                    elif 'sla' in diag_type:
                        plt.title('sla ' + tit)
                        M.pcolormesh(pMx, pMy, sla, cmap=cmap)
                        M.contour(Mx, My, sla, [0.], colors='k', linewidths=0.5)
                        M.contour(Mx, My, sla, slaparameter, colors='w', linewidths=0.3)
                        M.contour(Mx, My, sla, [slaparameter[0]], colors='m', linewidths=0.25)
                    plt.colorbar(orientation='horizontal')
                    plt.clim(-.5, .5)
                    M.fillcontinents()
                    M.drawcoastlines()
                    plt.show()
                    #plt.clf()
                    plt.close(250)

                if start:
                    first_record = True
                    # Set old variables equal to new variables
                    A_eddy.set_old_variables()
                    C_eddy.set_old_variables()
                    start = False
                else:
                    first_record = False
                
                
                # Track the eddies
                #tracking_start_time = time.time()
                A_eddy = track_eddies(A_eddy, first_record)
                C_eddy = track_eddies(C_eddy, first_record)
                
                #print 'Tracking the eddies', time.time() - tracking_start_time, 'seconds'
                
                if anim_figs: # Make figures for animations
                    
                    dy, dm, dd = oceantime2ymd(rtime)
                    dtime = dy + dm + dd
                    tit = 'Y' + dy.zfill(2) + 'M' + dm.zfill(2) + 'D' + dd.zfill(2)
                    
                    #if 'anim_fig' in locals():
                        ## Wait if there is a still-active anim_fig thread
                        #anim_fig.join()
                    
                    if 'Q' in diag_type:
                        anim_fig = threading.Thread(name='anim_figure', target=anim_figure,
                                     args=(33, M, pMx, pMy, xicopy, cmap, rtime, dtime, diag_type, Mx, My, 
                                           xi.copy(), qparam.copy(), qparameter, A_eddy, C_eddy,
                                           savedir, plt, 'Q ' + tit))
                    elif 'sla' in diag_type:
                        '''anim_fig = threading.Thread(name='anim_figure', target=anim_figure,
                                     args=(33, M, pMx, pMy, slacopy, plt.cm.RdBu_r, rtime, dtime, diag_type, Mx, My, 
                                           slacopy, slacopy, slaparameter, A_eddy, C_eddy,
                                           savedir, plt, 'SLA ' + tit))'''
                        #fignum = 31
                        anim_figure(A_eddy, C_eddy, Mx, My, pMx, pMy, plt.cm.RdBu_r, rtime, diag_type, 
                                savedir, 'SLA ' + tit, animax, animax_cbar)
                    #anim_fig.start()
                    
                                            
                # Save inactive eddies to nc file
                # IMPORTANT: this must be done at every time step!!
                #saving_start_time = time.time()
                if not first_record:
                    if verbose:
                        print '--- saving to nc'
                    #print 'ddsssssss', rtime
                    A_eddy.write2chelton_nc(A_savefile, rtime)
                    C_eddy.write2chelton_nc(C_savefile, rtime)
                #print 'Saving the eddies', time.time() - saving_start_time, 'seconds'
                
            # Running time for a single monthly file
            print '--- duration', str((time.time() - file_time) / 60.), 'minutes'
  
    # Total running time    
    print 'Duration', str((time.time() - start_time) / 3600.), 'hours!'

    print '\nsavedir', savedir
