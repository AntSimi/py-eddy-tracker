# -*- coding: utf-8 -*-
# %run make_eddy_track_ROMS.py

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

make_eddy_track_ROMS.py

Version 2.0.0



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
    THE_DOMAIN = 'Canary_Islands'
    #THE_DOMAIN = 'BlackSea'
    #THE_DOMAIN = 'MedSea' # not yet implemented
    
    
    
    DIRECTORY = '/marula/emason/runs2009/na_2009_7pt5km/'
    #DIRECTORY = '/marula/emason/runs2012/na_7pt5km/'
    #DIRECTORY = '/marula/emason/runs2013/canwake4km/'


    #SAVE_DIR = DIRECTORY
    SAVE_DIR = '/marula/emason/runs2009/na_2009_7pt5km/eddy_tracking_exps/sla_1/'


    
    print 'SAVE_DIR', SAVE_DIR
    
    
    # Choose type of diagnostic 
    # Presently one of q-parameter (Q) or sea level anomaly (sla)
    #DIAGNOSTIC_TYPE = 'Q'
    DIAGNOSTIC_TYPE = 'SLA'
    
    # Calculate Q using geostrophic velocity from zeta
    geos = True

    if 'SLA' in DIAGNOSTIC_TYPE:
        #SSHFILE = '/marula/emason/runs2012/na_7pt5km/seasmean.nc'
        SSHFILE = '/marula/emason/runs2009/na_2009_7pt5km/seasmean.nc'
        #SSHFILE = '/marula/emason/runs2013/canwake4km/seasmean.nc'

    YMIN = 11
    YMAX = 49
    MMIN = 1
    MMAX = 12
    #MODEL = 'ip_roms'
    MODEL = 'ROMS'
    FILETYPE = 'AVG'
    SIGMA_LEV = -1  # -1 for surface
    
    DAYS_BTWN_RECORDS = 3. # days
    # Save only tracks longer than...     
    TRACK_DURATION_MIN = 15 # days
    
    # Set grid
    GRDNAME = 'roms_grd_NA2009_7pt5km.nc'
    #GRDNAME = 'cb_2009_3km_grd_smooth.nc'
    #GRDNAME = 'grd_canwake4km.nc'
    
    PAD = 2
    
    
    # Min and max permitted eddy radii [m]
    if 'Q' in DIAGNOSTIC_TYPE:
        RADMIN = 15000.
        RADMAX = 250000.
        AMPMIN = 0.02
        AMPMAX = 100.
        
    elif 'SLA' in DIAGNOSTIC_TYPE:
        # radmin, radmax in degrees
        # NOTE: values 0.4 and 4.461 on 0.25 grid will give
        # 8 and 1000 pixels, as used by CSS11
        RADMIN = 0.2 #0.135 #0.4
        RADMAX = 4.461
        AMPMIN = 1. # cm
        AMPMAX = 150.
    else:
        Exception
    
    # Obtain this file from:
    #  http://www-po.coas.oregonstate.edu/research/po/research/rossby_radius/
    RW_PATH = '/home/emason/data_tmp/chelton_eddies/rossrad.dat'
    
    # Option to track using speed-based radius or effective radius
    #track_with_speed = True
    
    # Make figures for animations
    ANIM_FIGS = True


    # Define contours
    if 'Q' in DIAGNOSTIC_TYPE: # Set Q contours
        CONTOUR_PARAMETER = np.linspace(0, 5*10**-11, 25)
        
    elif 'SLA' in DIAGNOSTIC_TYPE: # Units, cm; for anticyclones use -50 to 50
        CONTOUR_PARAMETER = np.arange(-100., 100.25, 0.25)
        SHAPE_ERROR = 55.
    
    # Apply a filter to the Q parameter
    #SMOOTHING = 'Hanning'
    SMOOTHING = 'Gaussian'
    if 'SMOOTHING' not in locals():
        SMOOTH_FAC = False
    elif 'HANNING' in SMOOTHING: # apply Hanning filter
        SMOOTH_FAC = 5 # number of passes
    elif 'Gaussian' in SMOOTHING: # apply Gaussian filter
        smooth_fac = 'Deprecated'
        ZWL = 20. # degrees, zonal wavelength (see CSS11)
        MWL = 10. # degrees, meridional wavelength
    else:
        Exception
    



    SUBDOMAIN = True
    
    if THE_DOMAIN in 'Canary_Islands':
    
        LONMIN = -42.     # CANARY
        LONMAX = -5.5
        LATMIN = 12.
        LATMAX = 38.5
        
        #LONMIN = -25.     # CANARY
        #LONMAX = -15
        #LATMIN = 22.
        #LATMAX = 28.
    
    # FOR CANBAS4
    #LONMIN = -36.
    #LONMAX = -8.5
    #LATMIN = 18.
    #LATMAX = 33.2

    #LONMIN = -25
    #LONMAX = -14
    #LATMIN = 26
    #LATMAX = 29


    # Typical parameters
    DIST0 = 25000.
    if 'Q' in DIAGNOSTIC_TYPE:
        AMP0 = 0.02 # vort/f
    elif 'SLA' in DIAGNOSTIC_TYPE:
        AMP0 = 2 # 4 cm
    AREA0 = np.pi*(60000.**2)
    TEMP0 = 20.
    SALT0 = 35.
    
    # Parameters used by CSS11 and KCCMC11 (Sec. 3.2) to ensure the slow evolution
    # of the eddies over time; they use min and max values of 0.25 and 2.5
    EVOLVE_AMP_MIN = 0.0005 #0.25 #0.15 # min change in amplitude
    EVOLVE_AMP_MAX = 500 #2.5 #5.  # max change in amplitude
    EVOLVE_AREA_MIN = 0.0005 #0.25 #0.15 # min change in area
    EVOLVE_AREA_MAX = 500 #2.5 #5.  # max change in area
    
    
    SEPARATION_METHOD = 'ellipse' # see CSS11
    #SEPARATION_METHOD = 'sum_radii' # see KCCMC11
    
    if 'sum_radii' in SEPARATION_METHOD:
        # Separation distance factor. Adjust according to number of days between records
        # For 7 days, CSS11 use 150 km search ellipse
        # So, given typical eddy radius of r=50 km, 1.5 * (r1 + r2) = 150 km.
        #sep_dist_fac = 1.0
        sep_dist_fac = 1.15 # Seems ok for AVISO 7-day
        #sep_dist_fac = 1.5 # Causes tracks to jump for AVISO 7-days
    
    #CMAP = plt.cm.Spectral_r
    #CMAP = plt.cm.jet
    CMAP = plt.cm.RdBu

    
    VERBOSE = False

    #Define TRACK_EXTRA_VARIABLES to track and save:
    # - effective contour points
    # - speed-based contour points
    # - shape test values
    # - profiles of swirl velocity from effective contour inwards
    # Useful for working with ARGO data
    TRACK_EXTRA_VARIABLES = True
    
    
    # Note this is a global value
    FILLVAL = -9999
    
    config = {}
    config['THE_DOMAIN'] = THE_DOMAIN
    config['LONMIN'] = LONMIN
    config['LONMAX'] = LONMAX
    config['LATMIN'] = LATMIN
    config['LATMAX'] = LATMAX
    config['YMIN'] = YMIN
    config['YMAX'] = YMAX
    config['MMIN'] = MMIN
    config['MMAX'] = MMAX
    config['DAYS_BTWN_RECORDS'] = DAYS_BTWN_RECORDS
    config['CONTOUR_PARAMETER'] = CONTOUR_PARAMETER
    config['RADMIN'] = RADMIN
    config['RADMAX'] = RADMAX
    config['AMPMIN'] = AMPMIN
    config['AMPMAX'] = AMPMAX
    config['AMP0'] = AMP0
    config['TEMP0'] = TEMP0
    config['SALT0'] = SALT0
    #config[''] = 
    #config[''] = 
    #config[''] = 
    #----------------------------------------------------------------------------
    
    assert MMIN >= 1 and MMIN <= 12, 'MMIN must be within range 1 to 12'
    assert MMAX >= 1 and MMAX <= 12, 'MMAX must be within range 1 to 12'
    
    assert FILLVAL < -5555., 'FILLVAL must be << 0'
    assert DIAGNOSTIC_TYPE in ('SLA', 'Q'), 'DIAGNOSTIC_TYPE not properly defined'
    
    # Indices to seasons (from months)
    THE_SEASONS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    
    if 'Q' in DIAGNOSTIC_TYPE:
        # Search from 5e-11 onwards with fine spacing near the lower end
        qparameter = np.power(np.linspace(0., np.sqrt(qparameter.max()),
                              qparameter.size), 2)[::-1]
        
        # The shape error can maybe be played with...
        #SHAPE_ERROR = np.power(np.linspace(200., 40,  qparameter.size), 2) / 100.
        SHAPE_ERROR = np.power(np.linspace(85., 40,  qparameter.size), 2) / 100.
        SHAPE_ERROR[SHAPE_ERROR < 35.] = 35.
        #SHAPE_ERROR = 35. * np.ones(qparameter.size)
        
    elif 'SLA' in DIAGNOSTIC_TYPE:
        #SHAPE_ERROR = 55. * np.ones(slaparameter.size)
        config['SHAPE_ERROR'] = np.full(CONTOUR_PARAMETER.size, SHAPE_ERROR)
        #SHAPE_ERROR = np.power(np.linspace(85., 40,  slaparameter.size), 2) / 100.
        #SHAPE_ERROR[SHAPE_ERROR < 50.] = 50.
    
    
    
    # Get grid
    grd = rg.RomsGrid(GRDNAME, THE_DOMAIN, 'ROMS', LONMIN, LONMAX,
                      LATMIN, LATMAX, FILLVAL, with_pad=PAD)
    
    Mx, My = (grd.Mx[grd.jup0:grd.jup1, grd.iup0:grd.iup1],
              grd.My[grd.jup0:grd.jup1, grd.iup0:grd.iup1])
    
    
    # Get mean sea level from seasmean.nc
    if 'SLA' in DIAGNOSTIC_TYPE:
        with Dataset(SSHFILE) as nc:
            sshmean = nc.variables['zeta'][-1, grd.jp0:grd.jp1, grd.ip0:grd.ip1]
    
                                                   
    # Create Basemap instance for Mercator projection.
    #M = Basemap(projection='merc', llcrnrlon=LONMIN,
                                    #urcrnrlon=LONMAX,
                                    #llcrnrlat=LATMIN,
                                    #urcrnrlat=LATMAX,
                                    #lat_ts=28.,
                                    #resolution = 'h') # 'h'-high, 'l'-low

    #Mx, My = M(grd.lon(), grd.lat())
    #pMx, pMy = pcol_2dxy(Mx, My)

                    
    # Instantiate search ellipse object
    search_ellipse = eddy_tracker.SearchEllipse('Regional', grd,
                                                DAYS_BTWN_RECORDS,
                                                RW_PATH)
    
    

    if 'Gaussian' in SMOOTHING and 'SLA' in DIAGNOSTIC_TYPE:
        # Get parameters for ndimage.gaussian_filter
        ZRES, MRES = gaussian_resolution(grd.get_resolution(),
                                         np.atleast_1d(ZWL),
                                         np.atleast_1d(MWL))
        
        # Rotate to account for grid rotation
        ZRES, MRES = grd.rotate_vec(ZRES, MRES)
        ZRES, MRES = ZRES.mean(), MRES.mean()
    
    fig  = plt.figure(1)
    #axis = plt.axes()
    
    
    if 'Q' in DIAGNOSTIC_TYPE:
        A_SAVEFILE = "".join([SAVE_DIR, 'eddy_tracks_Q_ROMS_anticyclonic.nc'])
        C_SAVEFILE = "".join([SAVE_DIR, 'eddy_tracks_Q_ROMS_cyclonic.nc'])
        
    elif 'SLA' in DIAGNOSTIC_TYPE:
        A_SAVEFILE = "".join([SAVE_DIR, 'eddy_tracks_SLA_ROMS_anticyclonic.nc'])
        C_SAVEFILE = "".join([SAVE_DIR, 'eddy_tracks_SLA_ROMS_cyclonic.nc'])
    

    ## Initialise
    # Objects to hold data (qparameter)
    A_eddy = eddy_tracker.TrackList('Anticyclonic', A_SAVEFILE,
                                     grd, search_ellipse, **config)
    C_eddy = eddy_tracker.TrackList('Cyclonic', C_SAVEFILE,
                                    grd, search_ellipse, **config)
    
    A_eddy.search_ellipse = search_ellipse
    C_eddy.search_ellipse = search_ellipse
    
    if 'Q' in DIAGNOSTIC_TYPE:
        A_savefile = "".join([savedir, 'eddy_tracks_anticyclonic.nc'])
        #A_eddy.qparameter = qparameter
        A_eddy.SHAPE_ERROR = SHAPE_ERROR
        C_savefile = "".join([savedir, 'eddy_tracks_cyclonic.nc'])
        #C_eddy.qparameter = qparameter
        C_eddy.SHAPE_ERROR = SHAPE_ERROR  
        A_eddy.radmin = np.float64(radmin)
        A_eddy.radmax = np.float64(radmax)
        C_eddy.radmin = np.float64(radmin)
        C_eddy.radmax = np.float64(radmax)
        
    elif 'SLA' in DIAGNOSTIC_TYPE:
        #A_savefile = "".join([savedir, 'eddy_tracks_SLA_anticyclonic.nc'])
        #A_eddy.slaparameter = slaparameter
        #A_eddy.SHAPE_ERROR = SHAPE_ERROR
        #C_savefile = "".join([savedir, 'eddy_tracks_SLA_cyclonic.nc'])
        #C_eddy.slaparameter = slaparameter[::-1]
        #C_eddy.SHAPE_ERROR = SHAPE_ERROR[::-1]   
        # See CSS11 section B2 (0.4 degree radius)
        PIXMIN = np.round((np.pi * RADMIN**2) / grd.get_resolution()**2)
        PIXMAX = np.round((np.pi * RADMAX**2) / grd.get_resolution()**2)
        print '--- Pixel range = %s-%s' % (np.int(PIXMIN), np.int(PIXMAX))
        A_eddy.PIXEL_THRESHOLD = [PIXMIN, PIXMAX]
        C_eddy.PIXEL_THRESHOLD = [PIXMIN, PIXMAX]
        #A_eddy.radmin = np.float64(grd.get_resolution(radmin))
        #A_eddy.radmax = np.float64(grd.get_resolution(radmax))
        #C_eddy.radmin = np.float64(grd.get_resolution(radmin))
        #C_eddy.radmax = np.float64(grd.get_resolution(radmax))
    
    
    #A_eddy.ampmin = np.float64(ampmin)
    #A_eddy.ampmax = np.float64(ampmax)
    #C_eddy.ampmin = np.float64(ampmin)
    #C_eddy.ampmax = np.float64(ampmax)
    
    #A_eddy.interannual = False
    #C_eddy.interannual = False
    
    #A_eddy.chelton_style_nc = chelton_style_nc
    #C_eddy.chelton_style_nc = chelton_style_nc
    
    #A_eddy.DIAGNOSTIC_TYPE = DIAGNOSTIC_TYPE
    #C_eddy.DIAGNOSTIC_TYPE = DIAGNOSTIC_TYPE
    
    #A_eddy.SMOOTHING = SMOOTHING
    #C_eddy.SMOOTHING = SMOOTHING
    #A_eddy.smooth_fac = smooth_fac
    #C_eddy.smooth_fac = smooth_fac
    
    #A_eddy.M = M
    #C_eddy.M = M
    
    ##A_eddy.rwv = rwv
    ##C_eddy.rwv = rwv
    #A_eddy.search_ellipse = search_ellipse
    #C_eddy.search_ellipse = search_ellipse
    
    #A_eddy.separation_method = separation_method
    #C_eddy.separation_method = separation_method
    
    #if 'sum_radii' in separation_method:
        #A_eddy.sep_dist_fac = sep_dist_fac
        #C_eddy.sep_dist_fac = sep_dist_fac
    
    #A_eddy.evolve_ammin = np.float64(evolve_ammin)
    #A_eddy.evolve_ammax = np.float64(evolve_ammax)
    #A_eddy.evolve_armin = np.float64(evolve_armin)
    #A_eddy.evolve_armax = np.float64(evolve_armax)
    
    #C_eddy.evolve_ammin = np.float64(evolve_ammin)
    #C_eddy.evolve_ammax = np.float64(evolve_ammax)
    #C_eddy.evolve_armin = np.float64(evolve_armin)
    #C_eddy.evolve_armax = np.float64(evolve_armax)

    #A_eddy.i0, A_eddy.i1 = grd.i0, grd.i1
    #A_eddy.j0, A_eddy.j1 = grd.j0, grd.j1
    #C_eddy.i0, C_eddy.i1 = grd.i0, grd.i1
    #C_eddy.j0, C_eddy.j1 = grd.j0, grd.j1
    
    #A_eddy.lonmin, A_eddy.lonmax = np.float64(lonmin), np.float64(lonmax)
    #A_eddy.latmin, A_eddy.latmax = np.float64(latmin), np.float64(latmax)
    #C_eddy.lonmin, C_eddy.lonmax = np.float64(lonmin), np.float64(lonmax)
    #C_eddy.latmin, C_eddy.latmax = np.float64(latmin), np.float64(latmax)
    
    #A_eddy.FILLVAL = FILLVAL
    #C_eddy.FILLVAL = FILLVAL
    #A_eddy.verbose = verbose
    #C_eddy.verbose = verbose

    
    #A_eddy.area0 = np.float64(area0)
    #C_eddy.area0 = np.float64(area0)
    #A_eddy.amp0 = np.float64(amp0)
    #C_eddy.amp0 = np.float64(amp0)
    #A_eddy.dist0 = np.float64(dist0)
    #C_eddy.dist0 = np.float64(dist0)

    start = True
    start_time = time.time()
    
    # Loop thru the years...
    for Yr in np.arange(YMIN, YMAX+1):
        if Yr == YMIN: mo_min = MMIN
        else:       mo_min = 1
        if Yr == YMAX: mo_max = MMAX
        else:       mo_max = 12
        # Loop thru the months...
        for Mo in np.arange(mo_min, mo_max+1):
            file_time = time.time()
            filename = (DIRECTORY + MODEL.lower() + '_' + FILETYPE.lower() +
                       '_Y' + str(Yr) + 'M' + str(Mo) + '.nc')
            print 'Opening file:', filename
            if start:
                start = False
                record_range = get_ROMS_data(filename)
                
                CHECK_DAYS_BTWN_RECORDS = np.squeeze(np.diff(record_range[0:2]) / 86400.)
                assert DAYS_BTWN_RECORDS == CHECK_DAYS_BTWN_RECORDS, '"DAYS_BTWN_RECORDS" is incorrectly set'
                A_eddy.DAYS_BTWN_RECORDS = CHECK_DAYS_BTWN_RECORDS
                C_eddy.DAYS_BTWN_RECORDS = CHECK_DAYS_BTWN_RECORDS
                
                if 'ip_roms' in MODEL:
                    nc = netcdf.Dataset(filename)
                    rho_ntr = nc.variables['rho_ntr'][SIGMA_LEV] + 1027.4
                    nc.close()
                else:
                    rho_ntr = False
                
                # Create nc files for saving
                A_eddy.create_netcdf(DIRECTORY, A_SAVEFILE, grd=grd,
                                     YMIN=YMIN, YMAX=YMAX, MMIN=MMIN, MMAX=MMAX, MODEL=MODEL,
                                     SIGMA_LEV=SIGMA_LEV, rho_ntr=rho_ntr)
                C_eddy.create_netcdf(DIRECTORY, C_SAVEFILE, grd=grd,
                                     YMIN=YMIN, YMAX=YMAX, MMIN=MMIN, MMAX=MMAX, MODEL=MODEL,
                                     SIGMA_LEV=SIGMA_LEV, rho_ntr=rho_ntr)
                
                #A_eddy.create_netcdf(DIRECTORY, A_SAVEFILE, 'Anticyclonic', grd,
                                     #YMIN, YMAX, MMIN, MMAX, MODEL, SIGMA_LEV, rho_ntr)
                #C_eddy.create_netcdf(DIRECTORY, C_SAVEFILE, 'Cyclonic', grd,
                                     #YMIN, YMAX, MMIN, MMAX, MODEL, SIGMA_LEV, rho_ntr)
                
                
                record_range = record_range.size
                
            for record in np.arange(record_range):
                
                #rec_start_time = time.time()
                print '--- record', record + 1
                
                # Reset holding variables to empty arrays
                A_eddy.reset_holding_variables()
                C_eddy.reset_holding_variables()
                
                
                grdmask = grd.mask()
                
                if 'Q' in DIAGNOSTIC_TYPE:
                    u, v, temp, salt, rtime = get_ROMS_data(filename, pad, record, SIGMA_LEV,
                                                            ipadstr, ipadend, jpadstr, jpadend, DIAGNOSTIC_TYPE)
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
                    
                    
                    if SMOOTHING:
                        if SMOOTHING == 'Gaussian':
                            qparam = ndimage.gaussian_filter(qparam, smooth_fac, 0)
                        if SMOOTHING == 'Hanning':
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
                
                elif 'SLA' in DIAGNOSTIC_TYPE:
                    
                    # Note that sla (zeta) is returned in m
                    sla, rtime = get_ROMS_data(filename, grd, record, SIGMA_LEV, DIAGNOSTIC_TYPE)
                    
                    # Get anomaly
                    sla -= sshmean
                    
                    if isinstance(SMOOTHING, str):
                        
                        if 'Gaussian' in SMOOTHING:
                            if 'first_record' not in locals():
                                print '--- applying Gaussian high-pass filter to SLA'
                            # Set landpoints to zero
                            sla *= grd.mask()
                            # High pass filter
                            # http://stackoverflow.com/questions/6094957/high-pass-filter-for-image-processing-in-python-by-using-scipy-numpy
                            sla -= ndimage.gaussian_filter(sla, [MRES, ZRES])
                        elif 'Hanning' in SMOOTHING:
                            # smooth_fac passes of 2d Hanning filter
                            #han_time = time.time()
                            sla = func_hann2d_fast(sla, smooth_fac)
                            #print 'hanning', str(time.time() - han_time), ' seconds!'
                    
                    #sla *= grd.mask()
                    sla = np.ma.masked_where(grd.mask() == 0, sla)
                    
                    grd.set_geostrophic_velocity(sla)#, grd.f(), grd.pm(), grd.pn(),
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
                uspd = np.sqrt(grd.u**2, grd.v**2)
                uspd = np.ma.masked_where(grd.mask()[grd.jup0:grd.jup1,
                                                     grd.iup0:grd.iup1] == 1,
                       uspd)
                
                A_eddy.uspd = uspd.copy()
                C_eddy.uspd = uspd.copy()
                
                # Set interpolation coefficients
                grd.set_interp_coeffs(sla, uspd)
                A_eddy.sla_coeffs = grd.sla_coeffs
                A_eddy.uspd_coeffs = grd.uspd_coeffs
                C_eddy.sla_coeffs = grd.sla_coeffs
                C_eddy.uspd_coeffs = grd.uspd_coeffs
                
                
                # Get contours of Q/sla parameter
                if 'first_record' not in locals():
                
                    print '------ getting SLA contours for eddies'
                    contfig = plt.figure(99)
                    ax = contfig.add_subplot(111)
                
                    if ANIM_FIGS:
                        animfig = plt.figure(999)
                        animax = animfig.add_subplot(111)
                        # Colorbar axis
                        animax_cbar = get_cax(animax, dx=0.03,
                                              width=.05, position='b')
                
                if 'Q' in DIAGNOSTIC_TYPE:
                    CS = plt.contour(grd.lon(),
                                     grd.lat(), qparam, qparameter)
                    # Get xi contour field at zero
                    #plt.figure(99)
                    CSxi = plt.contour(grd.lon(),
                                       grd.lat(), xi, [0.])
                    #plt.close(99)
                
                elif 'SLA' in DIAGNOSTIC_TYPE:
                    A_CS = ax.contour(grd.lon(), grd.lat(),
                                      A_eddy.sla, A_eddy.CONTOUR_PARAMETER)
                    # Note that C_CS is in reverse order
                    C_CS = ax.contour(grd.lon(), grd.lat(),
                                      C_eddy.sla, C_eddy.CONTOUR_PARAMETER)
                
                else:
                    Exception
                
                if True: # clear the current axis
                    ax.cla()
                else:
                    # Debug
                    if 'Q' in DIAGNOSTIC_TYPE:
                        ax.set_title('qparameter and xi')
                        ax.clabel(CS, np.array([CONTOUR_PARAMETER.min(),
                                                CONTOUR_PARAMETER.max()]))
                    elif 'SLA' in DIAGNOSTIC_TYPE:
                        ax.set_title('CONTOUR_PARAMETER')
                        ax.clabel(CS, np.array([CONTOUR_PARAMETER.min(),
                                                CONTOUR_PARAMETER.max()]))
                    plt.axis('image')
                    #plt.colorbar()
                    plt.show()
                    
                # Set contour coordinates and indices for calculation of
                # speed-based radius
                A_eddy.swirl = SwirlSpeed(A_CS)
                C_eddy.swirl = SwirlSpeed(C_CS)
                    
                
                # Now we loop over the CS collection
                if 'Q' in DIAGNOSTIC_TYPE:
                    A_eddy, C_eddy = collection_loop(CS, grd, uspd, rtime,
                                       A_list_obj=A_eddy, C_list_obj=C_eddy, xi=xi, CSxi=CSxi,
                                       VERBOSE=VERBOSE)
                
                elif 'SLA' in DIAGNOSTIC_TYPE:
                    A_eddy.sign_type = 'Anticyclonic'
                    A_eddy = collection_loop(A_CS, grd, rtime,
                                             A_list_obj=A_eddy,  C_list_obj=None,
                                             sign_type=A_eddy.sign_type, VERBOSE=VERBOSE)
                    # Note that CSc is in reverse order
                    C_eddy.sign_type = 'Cyclonic'
                    C_eddy = collection_loop(C_CS, grd, rtime,
                                             A_list_obj=None, C_list_obj=C_eddy,
                                             sign_type=C_eddy.sign_type, VERBOSE=VERBOSE)
                
                yr, mo, da = oceantime2ymd(rtime)
                ymd_str = ''.join((str(yr), str(mo).zfill(2), str(da).zfill(2)))
                
                # All contours now done
                #print 'All contours', time.time() - rec_start_time, 'seconds'
                
                # Debug
                if 'fig250' in locals():
                    
                    plt.figure(250)
                    
                    dtime = dy + dm + dd
                    tit = 'Y' + dy + 'M' + dm + 'D' + dd
                    
                    if 'Q' in DIAGNOSTIC_TYPE:
                        plt.title('Q ' + tit)
                        M.pcolormesh(pMx, pMy, xi, cmap=cmap)
                        M.contour(Mx, My, xi, [0.], colors='k', linewidths=0.5)
                        M.contour(Mx, My, qparam, qparameter, colors='w', linewidths=0.3)
                        M.contour(Mx, My, qparam, [qparameter[0]], colors='m', linewidths=0.25)
                    elif 'SLA' in DIAGNOSTIC_TYPE:
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
                    # Set old variables equal to new variables
                    A_eddy.set_old_variables()
                    C_eddy.set_old_variables()
                    first_record = True
                    #start = False
                else:
                    first_record = False
                
                
                # Track the eddies
                #tracking_start_time = time.time()
                A_eddy = track_eddies(A_eddy, first_record)
                C_eddy = track_eddies(C_eddy, first_record)
                
                #print 'Tracking the eddies', time.time() - tracking_start_time, 'seconds'
                
                if ANIM_FIGS: # Make figures for animations
                    
                    dy, dm, dd = oceantime2ymd(rtime)
                    dtime = dy + dm + dd
                    tit = 'Y' + dy.zfill(2) + 'M' + dm.zfill(2) + 'D' + dd.zfill(2)
                    
                    #if 'anim_fig' in locals():
                        ## Wait if there is a still-active anim_fig thread
                        #anim_fig.join()
                    
                    if 'Q' in DIAGNOSTIC_TYPE:
                        anim_fig = threading.Thread(name='anim_figure', target=anim_figure,
                                     args=(33, M, pMx, pMy, xicopy, cmap, rtime, dtime, DIAGNOSTIC_TYPE, Mx, My, 
                                           xi.copy(), qparam.copy(), qparameter, A_eddy, C_eddy,
                                           savedir, plt, 'Q ' + tit))
                    elif 'SLA' in DIAGNOSTIC_TYPE:
                        '''anim_fig = threading.Thread(name='anim_figure', target=anim_figure,
                                     args=(33, M, pMx, pMy, slacopy, plt.cm.RdBu_r, rtime, dtime, DIAGNOSTIC_TYPE, Mx, My, 
                                           slacopy, slacopy, slaparameter, A_eddy, C_eddy,
                                           savedir, plt, 'SLA ' + tit))'''
                        #fignum = 31
                        anim_figure(A_eddy, C_eddy, Mx, My, plt.cm.RdBu_r, rtime,
                                    DIAGNOSTIC_TYPE, SAVE_DIR, 'SLA ' + ymd_str,
                                    animax, animax_cbar)
                    #anim_fig.start()
                    
                                            
                # Save inactive eddies to nc file
                # IMPORTANT: this must be done at every time step!!
                #saving_start_time = time.time()
                if not first_record:
                    if VERBOSE:
                        print '--- saving to nc'
                    #print 'ddsssssss', rtime
                    A_eddy.write2netcdf(rtime)
                    C_eddy.write2netcdf(rtime)
                #print 'Saving the eddies', time.time() - saving_start_time, 'seconds'
                
            # Running time for a single monthly file
            print '--- duration', str((time.time() - file_time) / 60.), 'minutes'
  
    # Total running time    
    print 'Duration', str((time.time() - start_time) / 3600.), 'hours!'

    print '\nsavedir', savedir
