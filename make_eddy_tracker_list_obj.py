# -*- coding: utf-8 -*-
# %run make_eddy_tracker_list_obj.py

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


make_eddy_tracker_list_obj.py

Version 1.4.2


===========================================================================


"""

import haversine_distmat as hav # needs compiling with f2py
#from make_eddy_track import *
from py_eddy_tracker_classes import *

#import find_closest_point_on_leg as fcpl

def haversine_distance_vector(lon1, lat1, lon2, lat2):
    """
    Haversine formula to calculate distance between two points
    Uses mean earth radius in metres (from scalars.h) = 6371315.0
    """
    lon1 = np.asfortranarray(lon1.copy())
    lat1 = np.asfortranarray(lat1.copy())
    lon2 = np.asfortranarray(lon2.copy())
    lat2 = np.asfortranarray(lat2.copy())
    dist = np.asfortranarray(np.empty_like(lon1))
    hav.haversine_distvec(lon1, lat1, lon2, lat2, dist)
    return dist


def newPosition(lonin, latin, angle, distance):
    """
    Given the inputs (base lon, base lat, angle, distance) return
     the lon, lat of the new position...
    """
    lon = np.asfortranarray(lonin.copy())
    lat = np.asfortranarray(latin.copy())
    angle = np.asfortranarray(angle.copy())
    distance = np.asfortranarray(distance.copy())
    lonout = np.asfortranarray(np.empty(lonin.shape))
    latout = np.asfortranarray(np.empty(lonin.shape))
    hav.waypoint_vec(lon, lat, angle, distance, lonout, latout)
    return lon[0], lat[0]


#def nearest(lon_pt, lat_pt, lon2d, lat2d):
    #"""
    #Return the nearest i, j point to a given lon, lat point
    #in a lat/lon grid
    #"""
    #lon2d, lat2d = lon2d.copy(), lat2d.copy()
    #lon2d -= lon_pt
    #lat2d -= lat_pt
    #d = np.hypot(lon2d, lat2d)
    #j, i = np.unravel_index(d.argmin(), d.shape)
    #return i, j
    

def nearest(lon_pt, lat_pt, lon2d, lat2d, theshape):
    """
    Return the nearest i, j point to a given lon, lat point
    in a lat/lon grid
    """
    #print type(lon_pt), lon_pt
    lon_pt += -lon2d
    lat_pt += -lat2d
    d = np.sqrt(lon_pt**2 + lat_pt**2)
    j, i = np.unravel_index(d.argmin(), theshape)
    return i, j


def uniform_resample(x, y, **kwargs):#, method='interp1d', kind='linear'):
    """
    Resample contours to have (nearly) equal spacing
       x, y    : input contour coordinates
       num_fac : factor to increase lengths of output coordinates
       method  : currently only 'interp1d' or 'Akima'
                 (Akima is slightly slower, but may be more accurate)
       kind    : type of interpolation (interp1d only)
    """
    #plt.figure(706)
    #plt.plot(x, y, '.-g', lw=2)
    ##x = ndimage.zoom(x.copy(), 2)
    ##y = ndimage.zoom(y.copy(), 2)
    #plt.plot(x, y, '.-r', lw=2)
    if kwargs.has_key('method'):
        method = kwargs['method']
    else:
        method = 'interp1d'
    
    if kwargs.has_key('num_fac'):
        num_fac = kwargs['num_fac']
    else:
        num_fac = 2
    
    # Get distances
    d = np.r_[0, np.cumsum(haversine_distance_vector(x[:-1],y[:-1], x[1:],y[1:]))]
    # Get uniform distances
    d_uniform = np.linspace(0, d.max(), num=d.size * num_fac, endpoint=True)
    
    # Do 1d interpolations
    if strcompare('interp1d', method):
        
        if kwargs.has_key('kind'):
            kind = kwargs['kind']
        else:
            kind = 'linear'
        xfunc = interpolate.interp1d(d, x, kind=kind)
        yfunc = interpolate.interp1d(d, y, kind=kind)
    
    elif strcompare('akima', method):
        
        xfunc = interpolate.Akima1DInterpolator(d, x)
        yfunc = interpolate.Akima1DInterpolator(d, y)
    
    else:
        Exception
    
    xnew = xfunc(d_uniform)
    ynew = yfunc(d_uniform)

    #plt.plot(xnew, ynew, '.-r', lw=1.5)
    #plt.plot(xxnew, yynew, '+-b', lw=1.)
    #plt.axis('image')
    #plt.show()
    return xnew, ynew
    

def strcompare(str1, str2):
    return str1 in str2 and str2 in str1







#def _find_closest_point_on_leg(p1, p2, p0):
    #"""find closest point to p0 on line segment connecting p1 and p2"""

    ##print type(p1), type(p2), type(p0)

    ## handle degenerate case
    #if np.all(p2 == p1):
        #d = p0 - p1
        #d **= 2
        #return d.sum()

    #d21 = p2 - p1
    #d01 = p0 - p1

    ## project on to line segment to find closest point
    #proj = np.dot(d01, d21) 
    #proj /= np.dot(d21, d21)
    #if proj < 0:
        #proj = 0
    #elif proj > 1:
        #proj = 1
    #pc = p1 + proj * d21

    ## find squared distance
    #d = pc - p0
    #d **= 2

    #return d.sum()#, pc


#def _find_closest_point_on_leg(p1, p2, p0):
    #"""find closest point to p0 on line segment connecting p1 and p2"""

    ## handle degenerate case
    #if np.all(p2 == p1):
        #d = np.sum((p0 - p1)**2)
        #return d, p1

    #d21 = p2 - p1
    #d01 = p0 - p1

    ## project on to line segment to find closest point
    #proj = np.dot(d01, d21) / np.dot(d21, d21)
    #if proj < 0:
        #proj = 0
    #if proj > 1:
        #proj = 1
    #pc = p1 + proj * d21

    ## find squared distance
    #d = np.sum((pc-p0)**2)

    #return d, pc


#def _find_closest_point_on_path(lc, point):
    #"""
    #lc: coordinates of vertices
    #point: coordinates of test point
    #"""

    ## find index of closest vertex for this segment
    #ds = np.sum((lc - point[None, :])**2, 1)
    #imin = np.argmin(ds)

    #dmin = np.inf
    #xcmin = None
    #legmin = (None, None)

    ##closed = mlab.is_closed_polygon(lc)
    #closed = np.alltrue([lc[0,0] == lc[-1,0], lc[0,1] == lc[-1,1]])
    
    ## build list of legs before and after this vertex
    #legs = []
    #if imin > 0 or closed:
        #legs.append(((imin-1) % len(lc), imin))
    #if imin < len(lc) - 1 or closed:
        #legs.append((imin, (imin+1) % len(lc)))

    #for leg in legs:
        ##d, xc = _find_closest_point_on_leg(lc[leg[0]], lc[leg[1]], point)
        #d = _find_closest_point_on_leg(lc[leg[0]], lc[leg[1]], point)
        #if d < dmin:
            #dmin = d
            ##xcmin = xc
            #legmin = leg

    #return dmin#, xcmin, legmin)

#def _find_closest_point_on_path(lc, point):
    #"""
    #lc: coordinates of vertices
    #point: coordinates of test point
    #"""
    ##import matplotlib.mlab as mlab; print 'delete me'
    ## Find index of closest vertex for this segment
    #ds = np.sum((lc - point[None, :])**2, 1)
    #imin = ds.argmin()

    #dmin = np.inf
    #closed = np.alltrue([lc[0,0] == lc[-1,0], lc[0,1] == lc[-1,1]])
    ##closedd = mlab.is_closed_polygon(lc)
    
    ##assert closed == closedd, 'should be same'
    ##print '***************************'
    
    ## Build list of legs before and after this vertex
    #legs = []
    #if imin > 0 or closed:
        #legs.append(((imin-1) % len(lc), imin))
    #if imin < len(lc) - 1 or closed:
        #legs.append((imin, (imin+1) % len(lc)))

    #for leg in legs:
        ##d, xc = _find_closest_point_on_leg(lc[leg[0]], lc[leg[1]], point)
        ##dd = _find_closest_point_on_leg(lc[leg[0]], lc[leg[1]], point)
        ##print type(point), type(lc[leg[0]]), type(lc[leg[1]])
        #d = fcpl.find_closest_point_on_leg(lc[leg[0]], lc[leg[1]], point)
        
        ##assert dd == d, 'not the same'
        
        #if d < dmin:
            #dmin = d

    #return dmin


#def find_nearest_contour(contour_obj, x, y, indices):
    #"""
    #Finds contour that is closest to a point.

    #Returns a tuple containing the contour, segment of minimum point.

    #Call signature::

      #conmin, segmin = find_nearest_contour(contour_obj, x, y, indices)
    #"""
    #dmin = np.inf
    ##conmin = None
    #segmin = None
    
    #point = np.array([x, y])

    #for icon in indices:
        #con = contour_obj.collections[icon]
        #paths = con.get_paths()

        #for segNum, linepath in enumerate(paths):
            #lc = linepath.vertices
            ##d = _find_closest_point_on_path(lc, point)
            #d = fcpl.find_closest_point_on_path(lc, point)
            
            #if d < dmin:
                #dmin = d
                ##conmin = icon
                #segmin = segNum

    #return segmin



def find_nearest_contour(contcoll, x, y):
    """
    Finds contour that is closest to a point.

    Returns a tuple containing the contour & segment.

    Call signature::

      segmin = find_nearest_contour(contcoll, x, y)

    """
    dmin = 1e10
    segmin = None
    linepathmin = None

    paths = contcoll.get_paths()
    for segNum, linepath in enumerate(paths):
        lc = linepath.vertices
        ds = lc[:,0] - x
        ds **= 2
        dss = lc[:,1] - y
        dss **= 2
        ds += dss
        d = ds.min()
        if d < dmin:
            dmin = d
            segmin = segNum
            linepathmin = linepath
    return (segmin, linepathmin)






class Track (object):
    """
    Class that holds eddy tracks and related info
        index  - index to each 'track', or track_number
        lon    - longitude
        lat    - latitude
        ocean_time - roms time in seconds
        uavg   - average velocity within eddy contour
        radius_s - eddy radius (as Chelton etal 2011)
        radius_e - eddy radius (as Chelton etal 2011)
        amplitude - max(abs(vorticity/f)) within eddy (as Kurian etal 2011)
        temp
        salt
        bounds - array(imin,imax,jmin,jmax) defining location of eddy
                 qparam contour
        alive - True if eddy active, set to False when eddy becomes active
        saved2nc - Becomes True once saved to netcdf file
        dayzero - True at first appearance of eddy
    """
    def __init__(self, eddy_index, DATATYPE, lon, lat, time, uavg, teke,
                       radius_s, radius_e, amplitude, temp=None, salt=None, 
                       save_extras=False, contour_e=None, contour_s=None,
                       uavg_profile=None, shape_error=None):

        self.eddy_index = eddy_index
        self.DATATYPE   = DATATYPE
        #self.lon        = np.atleast_1d(lon)
        self.lon        = [lon]
        self.lat        = [lat]
        self.ocean_time = [time]
        self.uavg       = [uavg]
        self.teke       = [teke]
        self.radius_s   = [radius_s] # speed-based eddy radius
        self.radius_e   = [radius_e] # effective eddy radius
        self.amplitude  = [amplitude]
        if 'ROMS' in self.DATATYPE:
            self.temp   = [temp]
            self.salt   = [salt]
        #self.bounds     = np.atleast_2d(bounds)
        self.alive      = True
        self.dayzero    = True
        self.saved2nc   = False
        self.save_extras = save_extras
        if self.save_extras:
            self.contour_e = [contour_e]
            self.contour_s = [contour_s]
            self.uavg_profile = [uavg_profile]
            self.shape_error = [shape_error]
        
    
    
    def append_pos(self, lon, lat, time, uavg, teke, radius_s, radius_e,
                   amplitude, temp=None, salt=None, contour_e=None, contour_s=None,
                   uavg_profile=None, shape_error=None):
        """
        Append track updates
        """
        #self.lon = np.r_[self.lon, lon]
        #print 'ccccccccccccccccccccccc', type(lon)
        self.lon.append(lon)
        self.lat.append(lat)
        self.ocean_time.append(time)
        self.uavg.append(uavg)
        self.teke.append(teke)
        self.radius_s.append(radius_s)
        self.radius_e.append(radius_e)
        #print 'self.amplitude_e222', self.amplitude, amplitude
        self.amplitude.append(amplitude)
        if 'ROMS' in self.DATATYPE:
            self.temp = np.r_[self.temp, temp]
            self.salt = np.r_[self.salt, salt]
        #try:
            #self.bounds = np.hstack((self.bounds, bounds))
        #except Exception:
            #self.bounds = np.vstack((self.bounds, bounds))
        if self.save_extras:
            self.contour_e.append(contour_e)
            self.contour_s.append(contour_s)
            self.uavg_profile.append(uavg_profile)
            self.shape_error = np.r_[self.shape_error, shape_error]
        return self



    def _is_alive(self, rtime):
        """
        Query if eddy is still active
          rtime is current 'ocean_time'
        If not active, kill it
        """
        # The eddy...
        if self.alive is False: # is already dead
            return self.alive
        elif self.dayzero: # has just been initiated
            self.dayzero = False
            return self.alive
        elif self.ocean_time[-1] == rtime: # is still alive
            return self.alive
        else:
            self.alive = False # is now dead
            return self.alive
        
        

class TrackList (object):
    """
    Class that holds list of eddy tracks:
        tracklist - the list of 'track' objects
        qparameter: Q parameter range used for contours
        new_lon, new_lat: new lon/lat centroids
        old_lon, old_lat: old lon/lat centroids
        eddy_index:   index of eddy in track_list
    """
    def __init__(self, DATATYPE, TRACK_DURATION_MIN, TRACK_EXTRA_VARIABLES):
        """
        Initialise the list 'tracklist'
        """
        self.tracklist = []
        self.DATATYPE = DATATYPE
        self.TRACK_DURATION_MIN = TRACK_DURATION_MIN
        self.TRACK_EXTRA_VARIABLES = TRACK_EXTRA_VARIABLES
        self.new_lon    = [] #np.array([])
        self.new_lat    = []
        self.new_radii_s = []
        self.new_radii_e = []
        self.new_amp    = []
        self.new_uavg   = []
        self.new_teke   = []
        if 'ROMS' in self.DATATYPE:
            self.new_temp = []
            self.new_salt = []
        self.new_time   = []
        #self.new_bounds = np.atleast_2d(np.empty(4, dtype=np.int16))
        self.old_lon    = [] #np.array([])
        self.old_lat    = []
        self.old_radii_s  = []
        self.old_radii_e  = []
        self.old_amp    = []
        self.old_uavg   = []
        self.old_teke   = []
        if 'ROMS' in self.DATATYPE:
            self.old_temp = []
            self.old_salt = []
        self.old_time   = []
        #self.old_bounds = np.atleast_2d([])
        if self.TRACK_EXTRA_VARIABLES:
            self.new_contour_e = []
            self.new_contour_s = []
            self.new_uavg_profile = []
            self.new_shape_error = []
            self.old_contour_e = []
            self.old_contour_s = []
            self.old_uavg_profile = []
            self.old_shape_error = []
        self.new_list   = True # flag indicating new list
        self.index = 0 # counter
        self.ncind = 0 # index to write to nc files, will increase and increase...
        self.ch_index = 0 # index for Chelton style nc files
        self.PAD = 2
        # Check for a correct configuration
        assert DATATYPE in ('ROMS', 'AVISO'), "Unknown string in 'DATATYPE' parameter"


    def add_new_track(self, lon, lat, time, uavg, teke,
            radius_s, radius_e, amplitude, temp=None, salt=None,
            contour_e=None, contour_s=None, uavg_profile=None, shape_error=None):
        """
        Append a new 'track' object to the list
        """
        self.tracklist.append(Track(self.index, self.DATATYPE,
                                    lon, lat, time, uavg, teke,
                                    radius_s, radius_e, amplitude,
                                    temp, salt, self.TRACK_EXTRA_VARIABLES,
                                    contour_e, contour_s, uavg_profile, shape_error))


    def update_track(self, index, lon, lat, time, uavg, teke,
             radius_s, radius_e, amplitude, temp=None, salt=None,
             contour_e=None, contour_s=None, uavg_profile=None, shape_error=None):
        """
        Update a track at index
        """
        self.tracklist[index].append_pos(lon, lat, time, uavg, teke,
                                         radius_s, radius_e,
                                         amplitude,
                                         temp=temp, salt=salt,
                                         contour_e=contour_e, contour_s=contour_s,
                                         uavg_profile=uavg_profile, shape_error=shape_error)


    #def update_eddy_properties(self, centlon, centlat, eddy_radius_s, eddy_radius_e,
                               #amplitude, uavg, teke, rtime,
                               #contour_e=None, contour_s=None,
                               #uavg_profile=None, shape_error=None,
                               #cent_temp=None, cent_salt=None):
    def update_eddy_properties(self, properties):
        """
        Append new variable values to track arrays
        """
        self.new_lon_tmp.append(properties.centlon)
        self.new_lat_tmp.append(properties.centlat)
        self.new_radii_tmp_s.append(properties.eddy_radius_s)
        self.new_radii_tmp_e.append(properties.eddy_radius_e)
        self.new_amp_tmp.append(properties.amplitude)
        self.new_uavg_tmp.append(properties.uavg)
        self.new_teke_tmp.append(properties.teke)
        self.new_time_tmp.append(properties.rtime)
        #print self.new_time_tmp
        #print self.new_lon_tmp
        if 'ROMS' in self.DATATYPE:
            #self.new_temp_tmp = np.r_[self.new_temp_tmp, cent_temp]
            #self.new_salt_tmp = np.r_[self.new_salt_tmp, cent_salt]
            pass
        #try:
            #self.new_bounds_tmp = np.vstack((self.new_bounds_tmp, bounds))
        #except Exception:    
            #self.new_bounds_tmp = np.hstack((self.new_bounds_tmp, bounds))
        if self.TRACK_EXTRA_VARIABLES:
            self.new_contour_e_tmp.append(properties.contour_e)
            self.new_contour_s_tmp.append(properties.contour_s)
            self.new_uavg_profile_tmp.append(properties.uavg_profile)
            self.new_shape_error_tmp = np.r_[self.new_shape_error_tmp, properties.shape_error]
        return self



    def reset_holding_variables(self):
        """
        Reset temporary holding variables to empty arrays
        """
        self.new_lon_tmp = [] #np.array([])
        self.new_lat_tmp = []
        self.new_radii_tmp_s = []
        self.new_radii_tmp_e = []
        self.new_amp_tmp = []
        self.new_uavg_tmp = []
        self.new_teke_tmp = []
        self.new_time_tmp = []
        self.new_temp_tmp = []
        self.new_salt_tmp = []
        #self.new_bounds_tmp = np.atleast_2d(np.empty(4, dtype=np.int16))
        if self.TRACK_EXTRA_VARIABLES:
            self.new_contour_e_tmp = []
            self.new_contour_s_tmp = []
            self.new_uavg_profile_tmp = []
            self.new_shape_error_tmp = np.atleast_1d([])
        return


    def set_old_variables(self):
        """
        Pass all values at time k+1 to k
        """
        self.old_lon = list(self.new_lon_tmp)
        self.old_lat = list(self.new_lat_tmp)
        self.old_radii_s = list(self.new_radii_tmp_s)
        self.old_radii_e = list(self.new_radii_tmp_e)
        self.old_amp = list(self.new_amp_tmp)
        self.old_uavg = list(self.new_uavg_tmp)
        self.old_teke = list(self.new_teke_tmp)
        self.old_temp = list(self.new_temp_tmp)
        self.old_salt = list(self.new_salt_tmp)
        if self.TRACK_EXTRA_VARIABLES:
            self.old_contour_e = list(self.new_contour_e_tmp)
            self.old_contour_s = list(self.new_contour_s_tmp)
            self.old_uavg_profile = list(self.new_uavg_profile_tmp)
            self.old_shape_error = np.atleast_1d([])


    def get_active_tracks(self, rtime):
        """
        Return list of indices to active tracks.
        A track is defined as active if the last record
        corresponds to current rtime (ocean_time).
        This call also identifies and removes
        inactive tracks.
        """
        active_tracks = []
        for i, item in enumerate(self.tracklist):
            if item._is_alive(rtime):
                active_tracks.append(i)
        return active_tracks


    def get_inactive_tracks(self, rtime):
        """
        Return list of indices to inactive tracks.
        This call also identifies and removes
        inactive tracks
        """
        inactive_tracks = []
        for i, item in enumerate(self.tracklist):
            if not item._is_alive(rtime):
                inactive_tracks.append(i)
        return inactive_tracks


        

    def create_netcdf(self, directory, savedir, title,
                      grd=None, Ymin=None, Ymax=None,
                      Mmin=None, Mmax=None, model=None,
                      sigma_lev=None, rho_ntr=None):
        """
        Create netcdf file same style as Chelton etal (2011)
        """
        if not self.TRACK_EXTRA_VARIABLES:
            self.savedir = savedir
        else:
           self.savedir = savedir.replace('.nc', '_ARGO_enabled.nc')
        nc = Dataset(self.savedir, 'w', format='NETCDF4')
        nc.title = ''.join((title, ' eddy tracks'))
        nc.directory = directory
        nc.days_between_records = np.float64(self.DAYS_BTWN_RECORDS)
        nc.TRACK_DURATION_MIN = np.float64(self.TRACK_DURATION_MIN)
        
        if 'Q' in self.DIAGNOSTIC_TYPE:
            nc.Q_parameter_contours = self.qparameter
        elif 'SLA' in self.DIAGNOSTIC_TYPE:
            nc.sla_parameter_contours = self.CONTOUR_PARAMETER
            nc.shape_error = self.SHAPE_ERROR
            nc.pixmin = self.PIXEL_THRESHOLD[0]
            nc.pixmax = self.PIXEL_THRESHOLD[1]
        
        if self.SMOOTHING in locals():
            nc.SMOOTHING = np.str(self.SMOOTHING)
            nc.SMOOTH_FAC = np.float64(self.SMOOTH_FAC)
        else:
            nc.SMOOTHING = 'None'
        
        nc.i0 = np.int32(self.i0)
        nc.i1 = np.int32(self.i1)
        nc.j0 = np.int32(self.j0)
        nc.j1 = np.int32(self.j1)
        
        if 'ROMS' in self.DATATYPE:
            nc.ROMS_grid = grd.grdfile
            nc.model = model
            nc.Ymin = np.int32(Ymin)
            nc.Ymax = np.int32(Ymax)
            nc.Mmin = np.int32(Mmin)
            nc.Mmax = np.int32(Mmax)
            nc.sigma_lev_index = np.int32(sigma_lev)
            
            if 'ip_roms' in model:
                nc.rho_ntr = rho_ntr
        
        nc.evolve_amplitude_min = self.EVOLVE_AMP_MIN
        nc.evolve_amplitude_max = self.EVOLVE_AMP_MAX
        nc.evolve_area_min = self.EVOLVE_AREA_MIN
        nc.evolve_area_max = self.EVOLVE_AREA_MAX
        
        # Create dimensions     
        nc.createDimension('Nobs', None)#len(Eddy.tracklist))
        #nc.createDimension('time', None) #len(maxlen(ocean_time)))
        #nc.createDimension('four', 4)
            
            
            
        # Create variables     
        nc.createVariable('track', np.int32, ('Nobs'), fill_value=self.fillval)   
        nc.createVariable('n', np.int32, ('Nobs'), fill_value=self.fillval)  
        # Use of jday should depend on clim vs interann
        if self.INTERANNUAL: # AVISO or INTERANNUAL ROMS solution
            nc.createVariable('j1', np.int32, ('Nobs'), fill_value=self.fillval)
        else: # climatological ROMS solution
            nc.createVariable('ocean_time', 'f8', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('lon', 'f4', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('lat', 'f4', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('A', 'f4', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('L', 'f4', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('U', 'f4', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('Teke', 'f4', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('radius_e', 'f4', ('Nobs'), fill_value=self.fillval)
        if 'Q' in self.DIAGNOSTIC_TYPE:
            nc.createVariable('qparameter', 'f4', ('Nobs'), fill_value=self.fillval)
        if 'ROMS' in self.DATATYPE:
            nc.createVariable('temp', 'f4', ('Nobs'), fill_value=self.fillval)
            nc.createVariable('salt', 'f4', ('Nobs'), fill_value=self.fillval)
            
        #nc.createVariable('NLP', 'f8', ('Nobs'), fill_value=self.fillval)
        #nc.createVariable('bounds', np.int16, ('Nobs','four'), fill_value=self.fillval)
        nc.createVariable('eddy_duration', np.int16, ('Nobs'), fill_value=self.fillval)
        
        # Meta data for variables
        nc.variables['track'].units = 'ordinal'
        nc.variables['track'].min_val = np.int32(0)
        nc.variables['track'].long_name = 'track number'
        nc.variables['track'].description = 'eddy identification number'
        
        nc.variables['n'].units = 'ordinal'
        #nc.variables['n'].min_val = 4
        #nc.variables['n'].max_val = 293
        nc.variables['n'].long_name = 'observation number'
        # Add option here to set length of intervals.....
        nc.variables['n'].description = 'observation sequence number (XX day intervals)'
        
        ## Use of jday should depend on clim vs interann
        if self.INTERANNUAL: # AVISO or INTERANNUAL ROMS solution
            nc.variables['j1'].units = 'days'
            #nc.variables['j1'].min_val = 2448910
            #nc.variables['j1'].max_val = 2455560
            nc.variables['j1'].long_name = 'Julian date'
            nc.variables['j1'].description = 'date of this observation'
            nc.variables['j1'].reference = self.JDAY_REFERENCE
            nc.variables['j1'].reference_description = 'Julian date on Jan 1, 1992'
        else: # climatological ROMS solution
            nc.variables['ocean_time'].units = 'ROMS ocean_time (seconds)'
        
        nc.variables['eddy_duration'].units = 'days'
        nc.variables['lon'].units = 'deg. longitude'
        nc.variables['lon'].min_val = self.LONMIN
        nc.variables['lon'].max_val = self.LONMAX
        nc.variables['lat'].units = 'deg. latitude'
        nc.variables['lat'].min_val = self.LATMIN
        nc.variables['lat'].max_val = self.LATMAX

        if 'Q' in self.DIAGNOSTIC_TYPE:
            nc.variables['A'].units = 'None, normalised vorticity (abs(xi)/f)'
        elif 'SLA' in self.DIAGNOSTIC_TYPE:
            nc.variables['A'].units = 'cm'
        nc.variables['A'].min_val = self.AMPMIN
        nc.variables['A'].max_val = self.AMPMAX
        nc.variables['A'].long_name = 'amplitude'
        nc.variables['A'].description = 'magnitude of the height difference ' + \
                                        'between the extremum of SSH within ' + \
                                        'the eddy and the SSH around the contour ' + \
                                        'defining the eddy perimeter'
        
        nc.variables['L'].units = 'km'
        nc.variables['L'].min_val = self.RADMIN / 1000.
        nc.variables['L'].max_val = self.RADMAX / 1000.
        nc.variables['L'].long_name = 'speed radius scale'
        nc.variables['L'].description = 'radius of a circle whose area is equal ' + \
                                        'to that enclosed by the contour of ' + \
                                        'maximum circum-average speed'
        
        nc.variables['U'].units = 'cm/sec'
        #nc.variables['U'].min = 0.
        #nc.variables['U'].max = 376.6
        nc.variables['U'].long_name = 'maximum circum-averaged speed'
        nc.variables['U'].description = 'average speed of the contour defining ' + \
                                        'the radius scale L'
        nc.variables['Teke'].units = 'm^2/sec^2'
        #nc.variables['Teke'].min = 0.
        #nc.variables['Teke'].max = 376.6
        nc.variables['Teke'].long_name = 'sum EKE within contour Ceff'
        nc.variables['Teke'].description = 'sum of eddy kinetic energy within contour defining ' + \
                                        'the effective radius'
        
        nc.variables['radius_e'].units = 'km'
        nc.variables['radius_e'].min_val = self.RADMIN / 1000.
        nc.variables['radius_e'].max_val = self.RADMAX / 1000.
        nc.variables['radius_e'].long_name = 'effective radius scale'
        nc.variables['radius_e'].description = 'effective eddy radius'
        
        
        if 'Q' in self.DIAGNOSTIC_TYPE:
            nc.variables['qparameter'].units = 's^{-2}'
        #nc.variables['NLP'].units = 'None, swirl vel. / propagation vel.'
        #nc.variables['NLP'].long_name = 'Non-linear parameter'

        if 'ROMS' in self.DATATYPE:
            nc.variables['temp'].units = 'deg. C'
            nc.variables['salt'].units = 'psu'
        #nc.variables['bounds'].units = 'indices to eddy location (imin,imax,jmin,jmax);' +  \
            ' use like this: var[jstr:jend,istr:iend][jmin:jmax,imin:imax]'
        
        if self.TRACK_EXTRA_VARIABLES:
            
            nc.createDimension('contour_points', None)
            nc.createDimension('uavg_contour_count', np.int(self.slaparameter.size * 0.333))
            nc.createVariable('contour_e', 'f4', ('contour_points','Nobs'), fill_value=self.fillval)
            nc.createVariable('contour_s', 'f4', ('contour_points','Nobs'), fill_value=self.fillval)
            nc.createVariable('uavg_profile', 'f4', ('uavg_contour_count','Nobs'), fill_value=self.fillval)
            nc.createVariable('shape_error', 'f4', ('Nobs'), fill_value=self.fillval)
            
            nc.variables['contour_e'].long_name = 'positions of effective contour points'
            nc.variables['contour_e'].description = 'lons/lats of effective contour points; ' + \
                                                    'lons (lats) in first (last) half of vector'
            nc.variables['contour_s'].long_name = 'positions of speed-based contour points'
            nc.variables['contour_s'].description = 'lons/lats of speed-based contour points; ' + \
                                                    'lons (lats) in first (last) half of vector'
            nc.variables['uavg_profile'].long_name = 'radial profile of uavg'
            nc.variables['uavg_profile'].description = 'all uavg values from effective contour inwards to ' + \
                                                       'smallest inner contour (pixel == 1)'
            nc.variables['shape_error'].units = '%'
        
        nc.close()
        return self
        


    def _reduce_inactive_tracks(self):
        """
        Remove dead tracks
        """
        #start_time = time.time()
        for track in self.tracklist:
            if not track.alive:
                track.lon = []
                track.lat = []
                track.qparameter = []
                track.amplitude = []
                track.uavg = []
                track.teke = []
                track.radius_s = []
                track.radius_e = []
                #track.bounds = []
                track.ocean_time = []
                if self.TRACK_EXTRA_VARIABLES:
                    track.contour_e = []
                    track.contour_s = []
                    track.uavg_profile = []
                    track.shape_error = []
        #print '_reduce_inactive_tracks', str(time.time() - start_time), ' seconds!'
        return
    
    
    def write2netcdf(self, rtime):
        """
        Write inactive tracks to netcdf file.
        'ncind' is important because prevents writing of 
        already written tracks.
        Each inactive track is 'emptied' after saving
        """
        tracks2save = np.asarray(self.get_inactive_tracks(rtime))
        DBR = self.DAYS_BTWN_RECORDS
        
        if np.any(tracks2save): # Note, this could break if all eddies become inactive at same time
            
            with Dataset(self.savedir, 'a') as nc:
                
                for i in np.nditer(tracks2save):

                    # saved2nc is a flag indicating if track[i] has been saved
                    if (not self.tracklist[i].saved2nc) and \
                       (np.all(self.tracklist[i].ocean_time)):
     
                        tsize = len(self.tracklist[i].lon)
                        if (tsize >= self.TRACK_DURATION_MIN / DBR) and tsize > 1.:

                            tend = self.ncind + tsize
                            nc.variables['lon'][self.ncind:tend] = np.asarray(self.tracklist[i].lon)
                            nc.variables['lat'][self.ncind:tend] = np.asarray([self.tracklist[i].lat])
                            nc.variables['A'][self.ncind:tend] = np.array([self.tracklist[i].amplitude])
                            self.tracklist[i].uavg *= np.array(100.) # to cm/s
                            nc.variables['U'][self.ncind:tend] = self.tracklist[i].uavg
                            nc.variables['Teke'][self.ncind:tend] = np.array([self.tracklist[i].teke])
                            self.tracklist[i].radius_s *= np.array(1e-3) # to km
                            nc.variables['L'][self.ncind:tend] = self.tracklist[i].radius_s
                            self.tracklist[i].radius_e *= np.array(1e-3) # to km
                            nc.variables['radius_e'][self.ncind:tend] = self.tracklist[i].radius_e
                            if 'ROMS' in self.DATATYPE:
                                nc.variables['temp'][self.ncind:tend] = np.array([self.tracklist[i].temp])
                                nc.variables['salt'][self.ncind:tend] = np.array([self.tracklist[i].salt])
                            #nc.variables['bounds'][self.ncind:tend] = np.array([self.tracklist[i].bounds])
                            if self.INTERANNUAL:
                                # We add 1 because 'j1' is an integer in ncsavefile; julian day midnight has .5
                                # i.e., dt.julian2num(2448909.5) -> 727485.0
                                nc.variables['j1'][self.ncind:tend] = dt.num2julian(np.array([self.tracklist[i].ocean_time])) + 1
                            else:
                                nc.variables['ocean_time'][self.ncind:tend] = np.array([self.tracklist[i].ocean_time])
                            nc.variables['n'][self.ncind:tend] = np.arange(tsize, dtype=np.int32)
                            nc.variables['track'][self.ncind:tend] = np.full(tsize, self.ch_index)
                            nc.variables['track'].max_val = np.int32(self.ch_index)
                            nc.variables['eddy_duration'][self.ncind:tend] = np.array([self.tracklist[i].ocean_time]).size \
                                                                                     * self.DAYS_BTWN_RECORDS
                            if self.TRACK_EXTRA_VARIABLES:
                                nc.variables['shape_error'][self.ncind:tend] = np.array([self.tracklist[i].shape_error])
                                for j in np.arange(tend - self.ncind):
                                    jj = j + self.ncind
                                    contour_e_arr = np.asarray(self.tracklist[i].contour_e[j]).ravel()
                                    nc.variables['contour_e'][:contour_e_arr.size,jj] = contour_e_arr
                                    contour_s_arr = np.asarray(self.tracklist[i].contour_s[j]).ravel()
                                    nc.variables['contour_s'][:contour_s_arr.size,jj] = contour_s_arr
                                    uavg_profile_arr = np.asarray(self.tracklist[i].uavg_profile[j]).ravel()
                                    nc.variables['uavg_profile'][:uavg_profile_arr.size,jj] = uavg_profile_arr #np.asarray(self.tracklist[i].uavg_profile[j]).ravel()
                        
                            # Flag indicating track[i] is now saved
                            self.tracklist[i].saved2nc = True
                            self.ncind += tsize
                            self.ch_index += 1
                            nc.sync()
        
        # Get index to first currently active track
        try:
            lasti = self.get_active_tracks(rtime)[0]
        except Exception:
            lasti = None
        
        # Clip the tracklist,
        # removes all dead tracks preceding first currently active track
        self.tracklist = self.tracklist[lasti:]
        self.index = len(self.tracklist) # adjust index accordingly
        
        # Remove inactive tracks
        self._reduce_inactive_tracks()
        
        # Update old_lon and old_lat...
        self.old_lon = self.new_lon[lasti:]
        self.old_lat = self.new_lat[lasti:]
        self.old_radii_s = self.new_radii_s[lasti:]
        self.old_radii_e = self.new_radii_e[lasti:]
        self.old_amp = self.new_amp[lasti:]
        self.old_uavg = self.new_uavg[lasti:]
        self.old_teke = self.new_teke[lasti:]
        
        self.new_lon = []
        self.new_lat = []
        self.new_radii_s = []
        self.new_radii_e = []
        self.new_amp = []
        self.new_uavg = []
        self.new_teke = []
        self.new_time = []
        
        if 'ROMS' in self.DATATYPE:
            self.old_temp = self.new_temp[lasti:]
            self.old_salt = self.new_salt[lasti:]
            
            self.new_temp = []
            self.new_salt = []
        
        if self.TRACK_EXTRA_VARIABLES:
            self.old_contour_e = list(self.new_contour_e[lasti:])
            self.old_contour_s = list(self.new_contour_s[lasti:])
            self.old_uavg_profile = list(self.new_uavg_profile[lasti:])
            self.old_shape_error = self.new_shape_error[lasti:]
            
            self.new_contour_e = []
            self.new_contour_s = []
            self.new_uavg_profile = []
            self.new_shape_error = []
            
        return self
        
        
    def get_distances(self, centlon, centlat):
        """
        Return all distances between the current centroid *centlon*,
        *centlat* and arrays *self.old_lon*, *self.old_lat*
        """
        clon = self.old_lon.copy()
        clat = self.old_lat.copy()
        clon.fill(centlon)
        clat.fill(centlat)
        return haversine_distance_vector(clon, clat,
                                         self.old_lon, self.old_lat)


    def insert_at_index(self, xarr, ind, x):
        """
        This the same as Matlab's native functionality:
            x(3)=4 gives [0  0  4] and then
            x(5)=7 gives [0  0  4  0  7]
        """
        try:
            x = x[0]
        except Exception:
            pass
        
        tmp = eval('self.' + xarr)
        tmp = tmp[:]
        if ind < len(tmp):
            newsize = len(tmp)
        else:
            newsize = ind + 1

        if strcompare('new_lon', xarr):
            try:
                self.new_lon[ind] = x
            except:
                self.new_lon.extend([0] *
                    (ind - len(self.new_lon) + 1))
                self.new_lon[ind] = x
        elif strcompare('new_lat', xarr):
            try:
                self.new_lat[ind] = x
            except:
                self.new_lat.extend([0] *
                    (ind - len(self.new_lat) + 1))
                self.new_lat[ind] = x
        elif strcompare('new_radii_s', xarr):
            try:
                self.new_radii_s[ind] = x
            except:
                self.new_radii_s.extend([0] *
                    (ind - len(self.new_radii_s) + 1))
                self.new_radii_s[ind] = x
        elif strcompare('new_radii_e', xarr):
            try:
                self.new_radii_e[ind] = x
            except:
                self.new_radii_e.extend([0] *
                    (ind - len(self.new_radii_e) + 1))
                self.new_radii_e[ind] = x
        elif strcompare('new_amp', xarr):
            try:
                self.new_amp[ind] = x
            except:
                self.new_amp.extend([0] *
                    (ind - len(self.new_amp) + 1))
                self.new_amp[ind] = x
        elif strcompare('new_uavg', xarr):
            try:
                self.new_uavg[ind] = x
            except:
                self.new_uavg.extend([0] *
                    (ind - len(self.new_uavg) + 1))
                self.new_uavg[ind] = x
        elif strcompare('new_teke', xarr):
            try:
                self.new_teke[ind] = x
            except:
                self.new_teke.extend([0] *
                    (ind - len(self.new_teke) + 1))
                self.new_teke[ind] = x
        elif strcompare('new_temp', xarr):
            try:
                self.new_temp[ind] = x
            except:
                self.new_temp.extend([0] *
                    (ind - len(self.new_temp) + 1))
                self.new_temp[ind] = x
        elif strcompare('new_salt', xarr):
            try:
                self.new_salt[ind] = x
            except:
                self.new_salt.extend([0] *
                    (ind - len(self.new_salt) + 1))
                self.new_salt[ind] = x
        elif strcompare('new_shape_error', xarr):
            try:
                self.new_shape_error[ind] = x
            except:
                self.new_shape_error.extend([0] *
                    (ind - len(self.new_shape_error) + 1))
                self.new_shape_error[ind] = x

        elif strcompare('new_contour_e', xarr):
            try:
                self.new_contour_e[ind] = x
            except:
                self.new_contour_e.append([0] *
                    (ind - len(self.new_contour_e) + 1))
                self.new_contour_e[ind] = x
        elif strcompare('new_contour_s', xarr):
            try:
                self.new_contour_s[ind] = x
            except:
                self.new_contour_s.append([0] *
                    (ind - len(self.new_contour_s) + 1))
                self.new_contour_s[ind] = x
        elif strcompare('new_uavg_profile', xarr):
            try:
                self.new_uavg_profile[ind] = x
            except:
                self.new_uavg_profile.append([0] *
                    (ind - len(self.new_uavg_profile) + 1))
                self.new_uavg_profile[ind] = x
        else:
            raise Exception
        
        return self
    
    
        
    
    #def set_bounds(self, centlon, centlat, radius, i, j, grd):
        #"""
        #Get indices to a bounding box around the eddy
        #"""
        #def get_angle(deg, ang):
            #return deg - np.rad2deg(ang)
        
        #grdangle = grd.angle()[j,i]
        ##print type(centlon), type(centlat)
        #a_lon, a_lat = newPosition(centlon, centlat, get_angle(0, grdangle), radius)
        #b_lon, b_lat = newPosition(centlon, centlat, get_angle(90, grdangle), radius)
        #c_lon, c_lat = newPosition(centlon, centlat, get_angle(180, grdangle), radius)
        #d_lon, d_lat = newPosition(centlon, centlat, get_angle(270, grdangle), radius)
                            
        ## Get i,j of bounding box around eddy
        ##print grd.lon().shape, grd.lat().shape, grd.shape
        #a_i, a_j = nearest(a_lon, a_lat, grd.lon(), grd.lat(), grd.shape)
        #b_i, b_j = nearest(b_lon, b_lat, grd.lon(), grd.lat(), grd.shape)
        #c_i, c_j = nearest(c_lon, c_lat, grd.lon(), grd.lat(), grd.shape)
        #d_i, d_j = nearest(d_lon, d_lat, grd.lon(), grd.lat(), grd.shape)
                                        
        #self.imin = np.maximum(np.min([a_i, b_i, c_i, d_i]) - 5, 0) # Must not go
        #self.jmin = np.maximum(np.min([a_j, b_j, c_j, d_j]) - 5, 0) # below zero
        #self.imax = np.max([a_i, b_i, c_i, d_i]) + 5
        #self.jmax = np.max([a_j, b_j, c_j, d_j]) + 5
        #return self
    
    def set_bounds(self, contlon, contlat, grd):
        """
        Get indices to a bounding box around the eddy
        WARNING won't work for a rotated grid
        """
        lonmin, lonmax = contlon.min(), contlon.max()
        latmin, latmax = contlat.min(), contlat.max()
        bl_i, bl_j = nearest(lonmin, latmin, grd.lon(), grd.lat(), grd.shape)
        tl_i, tl_j = nearest(lonmin, latmax, grd.lon(), grd.lat(), grd.shape)
        br_i, br_j = nearest(lonmax, latmin, grd.lon(), grd.lat(), grd.shape)
        tr_i, tr_j = nearest(lonmax, latmax, grd.lon(), grd.lat(), grd.shape)
        
        iarr = np.array([bl_i, tl_i, br_i, tr_i])
        jarr = np.array([bl_j, tl_j, br_j, tr_j])
        self.imin, self.imax = iarr.min(), iarr.max()
        self.jmin, self.jmax = jarr.min(), jarr.max()
        
        # For indexing the mins must not be less than zero
        self.imin = np.maximum(self.imin - self.PAD, 0)
        self.jmin = np.maximum(self.jmin - self.PAD, 0)
        self.imax += self.PAD + 1
        self.jmax += self.PAD + 1
        return self
    
    
    def set_mask_eff(self, contour, grd):
        """
        Set points within bounding box around eddy and calculate
        mask for effective contour
        """
        self.points = np.array([grd.lon()[self.jmin:self.jmax,
                                          self.imin:self.imax].ravel(),
                                grd.lat()[self.jmin:self.jmax,
                                          self.imin:self.imax].ravel()]).T
        # NOTE: Path.contains_points requires matplotlib 1.2 or higher
        self.mask_eff_1d = contour.contains_points(self.points)
        self.mask_eff_sum = self.mask_eff_1d.sum()
        return self
    
    
    def reshape_mask_eff(self, grd):
        """
        """
        shape = grd.lon()[self.jmin:self.jmax, self.imin:self.imax].shape
        self.mask_eff = self.mask_eff_1d.reshape(shape)

        
        
class RossbyWaveSpeed (object):
  
    def __init__(self, domain, limits, rw_path=None):
        """
        Initialise the RossbyWaveSpeedsobject
        """
        self.domain = domain
        self.rw_path = rw_path
        if self.domain in ('Global', 'ROMS'):
            assert self.rw_path is not None, \
                'Must supply a path for the Rossby deformation radius data'
            data = np.loadtxt(rw_path)
            self._lon = data[:, 1] - 360.
            self._lat = data[:, 0]
            self._defrad = data[:, 3]
            self.limits = limits
            self._make_subset()
            self.vartype = 'variable'
        else:
            self.vartype = 'constant'
        self.distance = np.empty(1)
        self.start = True
    
        
    def get_rwdistance(self, x, y, days_between_records):
        """
        """
        if self.domain in ('Global', 'ROMS'):
	    #print 'x,y', x,y
            self.distance[:] = self._get_rlongwave_spd(x, y)
            self.distance *= 86400.
            #if self.domain in 'ROMS':
                #self.distance *= 1.5
        elif 'BlackSea' in self.domain:
            self.distance[:] = 15000. # e.g., Blokhina & Afanasyev, 2003
        elif 'MedSea' in self.domain:
            self.distance[:] = 20000.
        else: Exception # Unknown domain
        
        if self.start:
            print '--------- setting ellipse for %s domain' %self.domain
            print '--------- using %s Rossby deformation radius of %s m' \
                                %(self.vartype, np.abs(self.distance[0]))
            self.start = False
        self.distance *= days_between_records
        return np.abs(self.distance)
    
    
    def _make_subset(self):
        """
        """
        pad = 1.5 # degrees
        LONMIN, LONMAX, LATMIN, LATMAX = self.limits
        lloi = np.logical_and(self._lon >= LONMIN - pad,
                              self._lon <= LONMAX + pad)
        lloi *= np.logical_and(self._lat >= LATMIN - pad,
                               self._lat <= LATMAX + pad)
        self._lon = self._lon[lloi]
        self._lat = self._lat[lloi]
        self._defrad = self._defrad[lloi]
        self._make_kdtree()
    
    
    def _make_kdtree(self):
        points = np.vstack([self._lon, self._lat]).T
        self._tree = spatial.cKDTree(points)
    
    
    def _get_defrad(self, x, y):
        """
        Get a point average of the deformation radius
        at x, y
        """
        weights, i = self._tree.query(np.array([x, y]), k=4, p=2)
        weights /= weights.sum()
        self._weights = weights
        self.i = i
        return np.average(self._defrad[i], weights=weights)
    
    
    def _get_rlongwave_spd(self, x, y):
        """
        Get the longwave phase speed, see Chelton etal (2008) pg 446:
          c = -beta * defrad**2 (this only for extratropical waves...)
        """
        r_spd_long = self._get_defrad(x, y)
        r_spd_long *= 1000. # km to m
        r_spd_long **= 2
        lat = np.average(self._lat[self.i], weights=self._weights)
        beta = np.cos(np.deg2rad(lat))
        beta *= 1458e-7 # 1458e-7 ~ (2 * 7.29*10**-5)
        beta /= 6371315.0
        r_spd_long *= -beta
        return r_spd_long
    

    
    
class SearchEllipse (object):
    
    def __init__(self, domain, DAYS_BTWN_RECORDS, rw_path, limits):
        """
        Class to construct a search ellipse/circle around a specified point.
        
        """
        self.domain = domain
        self.DAYS_BTWN_RECORDS = DAYS_BTWN_RECORDS
        self.rw_path = rw_path
        self.limits = limits
        self.rw_d_fac = 1.75
        self.e_w_major = self.DAYS_BTWN_RECORDS * 3e5 / 7.
        self.n_s_minor = self.DAYS_BTWN_RECORDS * 15e4 / 7.
        self.rwv = RossbyWaveSpeed(self.domain,
                               self.limits, rw_path=self.rw_path)
        self.rw_d = np.empty(1)
        self.rw_d_mod = np.empty(1)
    
    
    def _set_east_ellipse(self):
        """
        """
        self.east_ellipse = patch.Ellipse((self.x, self.y),
                                           self.e_w_major, self.n_s_minor)
        return self
    
    
    def _set_west_ellipse(self):
        """
        """
        self.west_ellipse = patch.Ellipse((self.x, self.y),
                                           self.rw_d_mod, self.n_s_minor)
        return self
    
    
    def _set_global_ellipse(self):
        """
        """
        self._set_east_ellipse()._set_west_ellipse()
        e_verts = self.east_ellipse.get_verts()
        e_size = e_verts[:, 0].size
        e_size *= 0.5
        w_verts = self.west_ellipse.get_verts()
        w_size = w_verts[:, 0].size
        w_size *= 0.5
        ew_x = np.hstack((e_verts[:e_size, 0], w_verts[w_size:, 0]))
        ew_y = np.hstack((e_verts[:e_size, 1], w_verts[w_size:, 1]))
        self.ellipse_path = path.Path(np.array([ew_x, ew_y]).T)
        return self#.ellipse_path
    
    
    def _set_black_sea_ellipse(self):
        """
        """
        self.black_sea_ellipse = patch.Ellipse((self.x, self.y),
                               2. * self.rw_d_mod, 2. * self.rw_d_mod)
        verts = self.black_sea_ellipse.get_verts()
        self.ellipse_path = path.Path(np.array([verts[:, 0],
                                                verts[:, 1]]).T)
        return self
    
    
    def set_search_ellipse(self, x, y):
        """
        """
        self.x = x
        self.y = y
        
        if self.domain in ('Global', 'ROMS'):
            self.rw_d[:] = self.rwv.get_rwdistance(x, y,
                                                   self.DAYS_BTWN_RECORDS)
            self.rw_d_mod[:] = 1.75
            self.rw_d_mod *= self.rw_d
            self.rw_d_mod[:] = np.maximum(self.rw_d_mod, self.n_s_minor)
            self.rw_d_mod *= 2.
            self._set_global_ellipse()
        elif 'BlackSea'  in self.domain:
            self.rw_d_mod[:] = 1.75
            self.rw_d[:] = self.rwv.get_rwdistance(x, y,
                                                   self.DAYS_BTWN_RECORDS)
            self.rw_d_mod *= self.rw_d
            self._set_black_sea_ellipse()
        else: Exception
        
        return self
    
    
    def view_search_ellipse(self, Eddy):
        """
        Input A_eddy or C_eddy
        """
        plt.figure()
        ax = plt.subplot(111)
        ax.set_title('Rossby def. rad %s m' %self.rw_d[0])
        Eddy.M.scatter(self.x, self.y, c='b')
        Eddy.M.plot(self.ellipse_path.vertices[:, 0],
                    self.ellipse_path.vertices[:, 1], 'r')
        Eddy.M.drawcoastlines()
        plt.show()
    
    
    
"""
def pickle_track(track_list, file):
    file = open(file, 'wb')
    pickle.dump(track_list, file)
    file.close()
    return

def unpickle_track(file):
    file = open(file, 'rb')
    data = pickle.load(file)
    file.close()
    return data
"""
##############################################################


if __name__ == '__main__':

    lon_ini = -10.
    lat_ini = 25.
    time_ini = 247893
    index = 0
    
    trackA = track_list(index, lon_ini, lat_ini, time_ini, 0, 0)
    print 'trackA lon:', trackA.tracklist[0].lon
    
    # update track 0
    trackA.update_track(0, -22, 34, 333344, 0, 0)
    trackA.update_track(0, -87, 37, 443344, 0, 0)
    trackA.update_track(0, -57, 57, 543344, 0, 0)
    print 'trackA lon:', trackA.tracklist[0].lon
    
    # start a new track
    trackA.append_list(-33, 45, 57435, 0, 0)
    print '\ntrackA lat:', trackA.tracklist[1].lat
    
    trackA.update_track(1, -32, 32, 4453344, 0, 0)
    print 'trackA lat:', trackA.tracklist[1].lat
    
    
    # Pickle
    output = open('data.pkl', 'wb')
    pickle.dump(trackA, output)
    output.close()
    
    # Unpickle
    pkl_file = open('data.pkl', 'rb')
    data1 = pickle.load(pkl_file)
    pkl_file.close()










