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

Version 1.0.1


===========================================================================


"""

import haversine_distmat as hav # needs compiling with f2py
from make_eddy_track import *


def haversine_distance_vector(lon1, lat1, lon2, lat2):
    '''
    Haversine formula to calculate distance between two points
    Uses mean earth radius in metres (from scalars.h) = 6371315.0
    '''
    lon1 = np.asfortranarray(lon1.copy())
    lat1 = np.asfortranarray(lat1.copy())
    lon2 = np.asfortranarray(lon2.copy())
    lat2 = np.asfortranarray(lat2.copy())
    dist = np.asfortranarray(np.empty_like(lon1))
    hav.haversine_distvec(lon1, lat1, lon2, lat2, dist)
    return dist


def newPosition(lonin, latin, angle, distance):
    '''
    Given the inputs (base lon, base lat, angle, distance) return
     the lon, lat of the new position...
    '''
    lonin = np.asfortranarray(np.copy(lonin))
    latin = np.asfortranarray(np.copy(latin))
    angle = np.asfortranarray(np.copy(angle))
    distance = np.asfortranarray(np.copy(distance))
    lon = np.asfortranarray(np.empty_like(lonin))
    lat = np.asfortranarray(np.empty_like(lonin))
    hav.waypoint_vec(lonin, latin, angle, distance, lon, lat)
    return lon, lat


def nearest(lon_pt, lat_pt, lon2d, lat2d):
    """
    Return the nearest i, j point to a given lon, lat point
    in a lat/lon grid
    """
    lon2d, lat2d = lon2d.copy(), lat2d.copy()
    lon2d -= lon_pt
    lat2d -= lat_pt
    d = np.hypot(lon2d, lat2d)
    j, i = np.unravel_index(d.argmin(), d.shape)
    return i, j
    

def uniform_resample(x, y, num_fac=4, kind='linear'):
    '''
    Resample contours to have (nearly) equal spacing
    '''
    #plt.figure(706)
    #plt.plot(x, y, '.-g', lw=2)
    ##x = ndimage.zoom(x.copy(), 2)
    ##y = ndimage.zoom(y.copy(), 2)
    #plt.plot(x, y, '.-r', lw=2)
    # Get distances
    d = np.r_[0, np.cumsum(haversine_distance_vector(x[:-1],y[:-1], x[1:],y[1:]))]
    # Get uniform distances
    d_uniform = np.linspace(0, d.max(), num=d.size * num_fac, endpoint=True)
    # Do 1d interpolations
    xfunc = interpolate.interp1d(d, x, kind=kind)
    yfunc = interpolate.interp1d(d, y, kind=kind)
    xnew = xfunc(d_uniform)
    ynew = yfunc(d_uniform)
    #plt.plot(xnew, ynew, '.-r', lw=2)
    #plt.axis('image')
    #plt.show()
    return xnew, ynew
    
    

class track (object):
    '''
    Class that holds eddy tracks and related info
        index  - index to each 'track', or track_number
        lon    - longitude
        lat    - latitude
        ocean_time - roms time in seconds
        Uavg   - average velocity within eddy contour
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
    '''
    def __init__(self, eddy_index, datatype, lon, lat, time, Uavg,
                       radius_s, radius_e, amplitude, bounds,
                       temp=None,
                       salt=None):

        self.eddy_index = eddy_index
        self.datatype   = datatype
        self.lon        = np.atleast_1d(lon)
        self.lat        = np.atleast_1d(lat)
        self.ocean_time = np.atleast_1d(time)
        self.Uavg       = np.atleast_1d(Uavg)
        self.radius_s   = np.atleast_1d(radius_s) # speed-based eddy radius
        self.radius_e   = np.atleast_1d(radius_e) # effective eddy radius
        self.amplitude  = np.atleast_1d(amplitude)
        if 'ROMS' in self.datatype:
            self.temp   = np.atleast_1d(temp)
            self.salt   = np.atleast_1d(salt)
        self.bounds     = np.atleast_2d(bounds)
        self.alive      = True
        self.dayzero    = True
        self.saved2nc   = False


    def append_pos(self, lon, lat, time, Uavg,
                   radius_s, radius_e, amplitude, bounds, temp, salt):
        '''
        Append track updates
        '''
        self.lon = np.r_[self.lon, lon]
        self.lat = np.r_[self.lat, lat]
        self.ocean_time = np.r_[self.ocean_time, time]
        self.Uavg = np.r_[self.Uavg, Uavg]
        self.radius_s = np.r_[self.radius_s, radius_s]
        self.radius_e = np.r_[self.radius_e, radius_e]
        self.amplitude = np.r_[self.amplitude, amplitude]
        if 'ROMS' in self.datatype:
            self.temp = np.r_[self.temp, temp]
            self.salt = np.r_[self.salt, salt]
        try:
            self.bounds = np.hstack((self.bounds, bounds))
        except Exception:
            self.bounds = np.vstack((self.bounds, bounds))



    def _is_alive(self, rtime):
        '''
        Query if eddy is still active
          rtime is current 'ocean_time'
        If not active, kill it
        '''
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
        
        

class track_list (object):
    '''
    Class that holds list of eddy tracks:
        tracklist - the list of 'track' objects
        qparameter: Q parameter range used for contours
        new_lon, new_lat: new lon/lat centroids
        old_lon, old_lat: old lon/lat centroids
        eddy_index:   index of eddy in track_list
    '''
    def __init__(self, datatype, track_duration_min):
        '''
        Initialise the list 'tracklist'
        '''
        self.tracklist = []
        self.datatype = datatype
        self.track_duration_min = track_duration_min
        self.new_lon    = np.array([])
        self.new_lat    = np.array([])
        self.new_radii_s = np.array([])
        self.new_radii_e = np.array([])
        self.new_amp    = np.array([])
        self.new_Uavg   = np.array([])
        if 'ROMS' in self.datatype:
            self.new_temp = np.array([])
            self.new_salt = np.array([])
        self.new_time   = np.array([])
        self.new_bounds = np.atleast_2d([])
        self.old_lon    = np.array([])
        self.old_lat    = np.array([])
        self.old_radii_s  = np.array([])
        self.old_radii_e  = np.array([])
        self.old_amp    = np.array([])
        self.old_Uavg   = np.array([])
        if 'ROMS' in self.datatype:
            self.old_temp = np.array([])
            self.old_salt = np.array([])
        self.old_time   = np.array([])
        self.old_bounds = np.atleast_2d([])
        self.new_list   = True # flag indicating new list
        self.index = 0 # counter
        self.ncind = 0 # index to write to nc files, will increase and increase...
        self.ch_index = 0 # index for Chelton style nc files
        
        # Check for a correct configuration
        assert datatype in ('ROMS', 'AVISO'), "Unknown string in 'datatype' parameter"


    def append_list(self, lon, lat, time, Uavg,
            radius_s, radius_e, amplitude, bounds, temp=None, salt=None):
        '''
        Append a new 'track' object to the list
        '''
        self.tracklist.append(track(self.index, self.datatype,
                                    lon, lat, time, Uavg,
                                    radius_s, radius_e,
                                    amplitude, bounds,
                                    temp, salt))


    def update_track(self, index, lon, lat, time, Uavg,
             radius_s, radius_e, amplitude, bounds, temp=None, salt=None):
        '''
        Update a track at index
        '''
        self.tracklist[index].append_pos(lon, lat, time, Uavg,
                                         radius_s, radius_e,
                                         amplitude, bounds,
                                         temp, salt)


    def reset_holding_variables(self):
        '''
        Reset temporary holding variables to empty arrays
        '''
        self.new_lon_tmp = np.array([])
        self.new_lat_tmp = np.array([])
        self.new_radii_tmp_s = np.array([])
        self.new_radii_tmp_e = np.array([])
        self.new_amp_tmp = np.array([])
        self.new_Uavg_tmp = np.array([])
        self.new_time_tmp = np.array([])
        self.new_temp_tmp = np.array([])
        self.new_salt_tmp = np.array([])
        self.new_bounds_tmp = np.atleast_2d([])
        return


    def set_old_variables(self):
        '''
        Pass all values at time k+1 to k
        '''
        self.old_lon = np.copy(self.new_lon_tmp)
        self.old_lat = np.copy(self.new_lat_tmp)
        self.old_radii_s = np.copy(self.new_radii_tmp_s)
        self.old_radii_e = np.copy(self.new_radii_tmp_e)
        self.old_amp = np.copy(self.new_amp_tmp)
        self.old_Uavg = np.copy(self.new_Uavg_tmp)
        self.old_temp = np.copy(self.new_temp_tmp)
        self.old_salt = np.copy(self.new_salt_tmp)


    def get_active_tracks(self, rtime):
        '''
        Return list of indices to active tracks.
        A track is defined as active if the last record
        corresponds to current rtime (ocean_time).
        This call also identifies and removes
        inactive tracks.
        '''
        self.active_tracks = []
        for i, item in enumerate(self.tracklist):
            if item._is_alive(rtime):
                self.active_tracks.append(i)
        return self.active_tracks


    def get_inactive_tracks(self, rtime):
        '''
        Return list of indices to inactive tracks.
        This call also identifies and removes
        inactive tracks
        '''
        self.inactive_tracks = []
        for i, item in enumerate(self.tracklist):
            if not item._is_alive(rtime):
                self.inactive_tracks.append(i)
        return self.inactive_tracks


        

    def create_netcdf(self, directory, savedir, title,
                                    grd=None,
                                    Ymin=None,
                                    Ymax=None,
                                    Mmin=None,
                                    Mmax=None,
                                    model=None,
                                    sigma_lev=None,
                                    rho_ntr=None):
        '''
        Create netcdf file same style as Chelton etal (2011)
        '''
        nc = netcdf.Dataset(savedir, 'w', format='NETCDF4')
        nc.title = title
        nc.directory = directory
        nc.days_between_records = np.float64(self.days_btwn_recs)
        nc.min_track_duration = np.float64(self.track_duration_min)
        
        if 'Q' in self.diag_type:
            nc.Q_parameter_contours = self.qparameter
        elif 'sla' in self.diag_type:
            nc.sla_parameter_contours = self.slaparameter
            nc.shape_error = self.shape_err
            nc.pixmin = self.pixel_threshold[0]
            nc.pixmax = self.pixel_threshold[1]
        
        if self.smoothing in locals():
            nc.smoothing = np.str(self.smoothing)
            nc.smoothing_fac = np.float(self.smooth_fac)
        else:
            nc.smoothing = 'None'
        
        nc.i0 = np.int32(self.i0)
        nc.i1 = np.int32(self.i1)
        nc.j0 = np.int32(self.j0)
        nc.j1 = np.int32(self.j1)
        
        if 'ROMS' in self.datatype:
            nc.ROMS_grid = grd.grdfile
            nc.model = model
            nc.Ymin = np.int32(Ymin)
            nc.Ymax = np.int32(Ymax)
            nc.Mmin = np.int32(Mmin)
            nc.Mmax = np.int32(Mmax)
            nc.sigma_lev_index = np.int32(sigma_lev)
            
            if model=='ip_roms':
                nc.rho_ntr = rho_ntr
        
        nc.evolve_amp_min = self.evolve_ammin
        nc.evolve_amp_max = self.evolve_ammax
        nc.evolve_area_min = self.evolve_armin
        nc.evolve_area_max = self.evolve_armax
        
        # Create dimensions     
        nc.createDimension('Nobs', None)#len(Eddy.tracklist))
        #nc.createDimension('time', None) #len(maxlen(ocean_time)))
        nc.createDimension('four', 4)

        # Create variables     
        nc.createVariable('track', np.int32, ('Nobs'), fill_value=self.fillval)   
        nc.createVariable('n', np.int32, ('Nobs'), fill_value=self.fillval)  
        # Use of jday should depend on clim vs interann
        if self.interannual: # AVISO or interannual ROMS solution
            nc.createVariable('j1', np.int32, ('Nobs'), fill_value=self.fillval)
        else: # climatological ROMS solution
            nc.createVariable('ocean_time', 'f8', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('lon', 'f8', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('lat', 'f8', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('A', 'f8', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('L', 'f8', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('U', 'f8', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('radius_e', 'f8', ('Nobs'), fill_value=self.fillval)
        if 'Q' in self.diag_type:
            nc.createVariable('qparameter', 'f8', ('Nobs'), fill_value=self.fillval)
        if 'ROMS' in self.datatype:
            nc.createVariable('temp', 'f8', ('Nobs'), fill_value=self.fillval)
            nc.createVariable('salt', 'f8', ('Nobs'), fill_value=self.fillval)
            
        nc.createVariable('NLP', 'f8', ('Nobs'), fill_value=self.fillval)
        nc.createVariable('bounds', np.int16, ('Nobs','four'), fill_value=self.fillval)
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
        if self.interannual: # AVISO or interannual ROMS solution
            nc.variables['j1'].units = 'days'
            #nc.variables['j1'].min_val = 2448910
            #nc.variables['j1'].max_val = 2455560
            nc.variables['j1'].long_name = 'Julian date'
            nc.variables['j1'].description = 'date of this observation'
            nc.variables['j1'].reference = self.jday_ref
            nc.variables['j1'].reference_description = 'Julian date at Jan 1, 1992'
        else: # climatological ROMS solution
            nc.variables['ocean_time'].units = 'ROMS ocean_time (seconds)'
        
        nc.variables['eddy_duration'].units = 'days'
        nc.variables['lon'].units = 'deg. longitude'
        nc.variables['lon'].min_val = self.lonmin
        nc.variables['lon'].max_val = self.lonmax
        nc.variables['lat'].units = 'deg. latitude'
        nc.variables['lat'].min_val = self.latmin
        nc.variables['lat'].max_val = self.latmax

        if 'Q' in self.diag_type:
            nc.variables['A'].units = 'None, normalised vorticity (abs(xi)/f)'
        elif 'sla' in self.diag_type:
            nc.variables['A'].units = 'cm'
        nc.variables['A'].min_val = self.ampmin
        nc.variables['A'].max_val = self.ampmax
        nc.variables['A'].long_name = 'amplitude'
        nc.variables['A'].description = 'magnitude of the height difference ' + \
                                        'between the extremum of SSH within ' + \
                                        'the eddy and the SSH around the contour ' + \
                                        'defining the eddy perimeter'
        nc.variables['L'].units = 'km'
        nc.variables['L'].min_val = self.radmin / 1000.
        nc.variables['L'].max_val = self.radmax / 1000.
        nc.variables['L'].long_name = 'radius scale'
        nc.variables['L'].description = 'radius of a circle whose area is equal ' + \
                                        'to that enclosed by the contour of ' + \
                                        'maximum circum-average speed'
        
        nc.variables['U'].units = 'cm/sec'
        #nc.variables['U'].min = 0.
        #nc.variables['U'].max = 376.6
        nc.variables['U'].long_name = 'maximum circum-averaged speed'
        nc.variables['U'].description = 'average speed of the contour defining ' + \
                                        'the radius scale L'
        nc.variables['radius_e'].units = 'km'
        nc.variables['radius_e'].min_val = self.radmin / 1000.
        nc.variables['radius_e'].max_val = self.radmax / 1000.
        nc.variables['radius_e'].long_name = 'effective radius scale'
        nc.variables['radius_e'].description = 'effective eddy radius'
        
        
        if 'Q' in self.diag_type:
            nc.variables['qparameter'].units = 's^{-2}'
        nc.variables['NLP'].units = 'None, swirl vel. / propagation vel.'
        nc.variables['NLP'].long_name = 'Non-linear parameter'

        if 'ROMS' in self.datatype:
            nc.variables['temp'].units = 'deg. C'
            nc.variables['salt'].units = 'psu'
        nc.variables['bounds'].units = 'indices to eddy location (imin,imax,jmin,jmax);' +  \
            ' use like this: var[jstr:jend,istr:iend][jmin:jmax,imin:imax]'
        
        nc.close()
        return
        


    def _reduce_inactive_tracks(self):
        '''
        Remove dead tracks
        '''
        #start_time = time.time()
        for track in self.tracklist:
            if not track.alive:
                track.lon = False
                track.lat = False
                track.qparameter = False
                track.amplitude = False
                track.Uavg = False
                track.radius_s = False
                track.radius_e = False
                track.bounds = False
                track.ocean_time = False
        #print '_reduce_inactive_tracks', str(time.time() - start_time), ' seconds!'
        return
    
    
    def get_non_lin_param(self, i):
        '''
        Return the non-linearity parameter along a track
        '''
        #start_time = time.time()
        distances = haversine_distance_vector(self.tracklist[i].lon[:-1],
                               self.tracklist[i].lat[:-1],
                               self.tracklist[i].lon[1:],
                               self.tracklist[i].lat[1:])
        distances = np.concatenate((distances[0][np.newaxis],
                                    distances,
                                    distances[-1][np.newaxis]))
        distances = distances[:-1]
        distances += distances[1:]
        distances *= 0.5
        distances /= (self.days_btwn_recs * 86400.)
        return np.array([self.tracklist[i].Uavg]) / distances
    
    
    def write2chelton_nc(self, savedir, rtime):
        '''
        Write inactive tracks to netcdf file.
        'ncind' is important because prevents writing of 
        already written tracks.
        Each inactive track is 'emptied' after saving
        '''
        nc = netcdf.Dataset(savedir, 'a')
        
        tracks2save = np.array([self.get_inactive_tracks(rtime)])
        if np.any(tracks2save):
            
            for i in np.nditer(tracks2save):

                # saved2nc is a flag indicating if track[i] has been saved
                if np.logical_and(not self.tracklist[i].saved2nc,
                                  np.all(self.tracklist[i].ocean_time)):
     
                    tsize = self.tracklist[i].ocean_time.size
                    
                    if tsize >= self.track_duration_min / self.days_btwn_recs:

                        tend = self.ncind + tsize
                        nc.variables['lon'][self.ncind:tend] = np.array([self.tracklist[i].lon])
                        nc.variables['lat'][self.ncind:tend] = np.array([self.tracklist[i].lat])
                        nc.variables['A'][self.ncind:tend] = np.array([self.tracklist[i].amplitude])
                        nc.variables['U'][self.ncind:tend] = np.array([self.tracklist[i].Uavg]) * 100. # to cm/s
                        nc.variables['L'][self.ncind:tend] = np.array([self.tracklist[i].radius_s]) * np.array(1e-3)
                        nc.variables['radius_e'][self.ncind:tend] = np.array([self.tracklist[i].radius_e]) * np.array(1e-3)
                        if 'ROMS' in self.datatype:
                            nc.variables['temp'][self.ncind:tend] = np.array([self.tracklist[i].temp])
                            nc.variables['salt'][self.ncind:tend] = np.array([self.tracklist[i].salt])
                        nc.variables['bounds'][self.ncind:tend] = np.array([self.tracklist[i].bounds])
                        if self.interannual:
                            # We add 1 because 'j1' is an integer in ncsavefile; julian day midnight has .5
                            # i.e., dt.julian2num(2448909.5) -> 727485.0
                            nc.variables['j1'][self.ncind:tend] = dt.num2julian(np.array([self.tracklist[i].ocean_time])) + 1
                        else:
                            nc.variables['ocean_time'][self.ncind:tend] = np.array([self.tracklist[i].ocean_time])
                        nc.variables['n'][self.ncind:tend] = np.arange(tsize, dtype=np.int32)
                        nc.variables['track'][self.ncind:tend] = np.ones(tsize) * self.ch_index
                        nc.variables['track'].max_val = np.int32(self.ch_index)
                        nc.variables['eddy_duration'][self.ncind:tend] = np.array([self.tracklist[i].ocean_time]).size \
                                                                                 * self.days_btwn_recs
                    
                        # Flag indicating track[i] is now saved
                        self.tracklist[i].saved2nc = True
                        self.ncind += len(self.tracklist[i].lon)
                        self.ch_index += 1
                        nc.sync()
        
        nc.close()

        # Get index to first currently active track
        try:
            lasti = self.get_active_tracks(rtime)[0]
        except Exception:
            lasti = None
        
        # Clip tracklist, removing all dead tracks preceding first currently active track
        self.tracklist = self.tracklist[lasti:]
        self.index = len(self.tracklist) # adjust index accordingly
        # Remove inactive tracks
        self._reduce_inactive_tracks()
        
        # Update old_lon and old_lat...
        self.old_lon     = self.new_lon[lasti:]
        self.old_lat     = self.new_lat[lasti:]
        self.old_radii_s = self.new_radii_s[lasti:]
        self.old_radii_e = self.new_radii_e[lasti:]
        self.old_amp     = self.new_amp[lasti:]
        self.old_Uavg    = self.new_Uavg[lasti:]
        if 'ROMS' in self.datatype:
            self.old_temp = self.new_temp[lasti:]
            self.old_salt = self.new_salt[lasti:]
        self.old_bounds  = self.new_bounds[lasti:]
        self.new_lon     = np.array([])
        self.new_lat     = np.array([])
        self.new_radii_s = np.array([])
        self.new_radii_e = np.array([])
        self.new_amp     = np.array([])
        self.new_Uavg    = np.array([])
        self.new_time    = np.array([])
        if 'ROMS' in self.datatype:
            self.new_temp = np.array([])
            self.new_salt = np.array([])
        self.new_bounds  = np.atleast_2d([])
        
        return self
        
        
    def get_distances(self, centlon, centlat):
        '''
        Return distances for the current centroid (centlon,
        centlat) and arrays (self.old_lon, self.old_lat)
        '''
        clon = self.old_lon.copy()
        clat = self.old_lat.copy()
        clon.fill(centlon)
        clat.fill(centlat)
        return haversine_distance_vector(clon, clat, self.old_lon, self.old_lat)


    def insert_at_index(self, xarr, ind, x):
        '''
        For some reason Python doesn't do matlab's
        x(3)=4 gives [0  0  4] and then
        x(5)=7 gives [0  0  4  0  7]
        ------------------------------------------------------------------
        THIS IS SLOW BECAUSE REMAKES THE ARRAY EVERY TIME
        TRY UPDATING WITH VERY LONG ZERO ARRAYS...
        
        This is the way to do it, all tracks as lists rather np arrays
        try:
            self.new_lon[ind] = x
        except:
            self.new_lon.append([0] * (ind - len(self.new_lon) + 1)
            self.new_lon[ind] = x
        THIS IMPLIES CHANGING TO LISTS ALL THROUGH THE CODE... NOT URGENT FOR NOW
        ------------------------------------------------------------------
        
        '''
        tmp  = eval('np.copy(self.' + xarr +')')
        if ind < tmp.size:
            newsize = tmp.size
        else:
            newsize = ne.evaluate('ind + 1')
        if 'new_lon' in xarr:
            self.new_lon = np.zeros((newsize))
            self.new_lon[:tmp.size] = tmp
            self.new_lon[ind] = x
        elif 'new_lat' in xarr:
            self.new_lat = np.zeros((newsize))
            self.new_lat[:tmp.size] = tmp
            self.new_lat[ind] = x
        elif 'new_radii_s' in xarr:
            self.new_radii_s = np.zeros((newsize))
            self.new_radii_s[:tmp.size] = tmp
            self.new_radii_s[ind] = x
        elif 'new_radii_e' in xarr:
            self.new_radii_e = np.zeros((newsize))
            self.new_radii_e[:tmp.size] = tmp
            self.new_radii_e[ind] = x
        elif 'new_amp' in xarr:
            self.new_amp = np.zeros((newsize))
            self.new_amp[:tmp.size] = tmp
            self.new_amp[ind] = x
        elif 'new_temp' in xarr:
            self.new_temp = np.zeros((newsize))
            self.new_temp[:tmp.size] = tmp
            self.new_temp[ind] = x
        elif 'new_salt' in xarr:
            self.new_salt = np.zeros((newsize))
            self.new_salt[:tmp.size] = tmp
            self.new_salt[ind] = x
        else:
            raise Exception
        return self


    def update_eddy_properties(self, centlon, centlat, eddy_radius_s, eddy_radius_e,
                               amplitude, Uavg, rtime, bounds, cent_temp=None, cent_salt=None):
        '''
        Append new variable values to track arrays
        '''
        self.new_lon_tmp = np.r_[self.new_lon_tmp, centlon]
        self.new_lat_tmp = np.r_[self.new_lat_tmp, centlat]
        self.new_radii_tmp_s = np.r_[self.new_radii_tmp_s, eddy_radius_s]
        self.new_radii_tmp_e = np.r_[self.new_radii_tmp_e, eddy_radius_e]
        self.new_amp_tmp = np.r_[self.new_amp_tmp, amplitude]
        self.new_Uavg_tmp = np.r_[self.new_Uavg_tmp, Uavg]
        self.new_time_tmp = np.r_[self.new_time_tmp, rtime]
        if 'ROMS' in self.datatype:
            self.new_temp_tmp = np.r_[self.new_temp_tmp, cent_temp]
            self.new_salt_tmp = np.r_[self.new_salt_tmp, cent_salt]
        try:
            self.new_bounds_tmp = np.vstack((self.new_bounds_tmp, bounds))
        except Exception:    
            self.new_bounds_tmp = np.hstack((self.new_bounds_tmp, bounds))
        return self
    
    
        
    
    def get_bounds(self, centlon, centlat, radius, i, j, grd):
        '''
        Get indices to a bounding box around the eddy
        '''
        def get_angle(deg, ang):
            return deg - np.rad2deg(ang)
        
        grdangle = grd.angle()[j,i]
        
        a_lon, a_lat = newPosition(centlon, centlat, get_angle(0, grdangle), radius)
        b_lon, b_lat = newPosition(centlon, centlat, get_angle(90, grdangle), radius)
        c_lon, c_lat = newPosition(centlon, centlat, get_angle(180, grdangle), radius)
        d_lon, d_lat = newPosition(centlon, centlat, get_angle(270, grdangle), radius)
                            
        # Get i,j of bounding box around eddy
        a_i, a_j = nearest(a_lon, a_lat, grd.lon(), grd.lat())
        b_i, b_j = nearest(b_lon, b_lat, grd.lon(), grd.lat())
        c_i, c_j = nearest(c_lon, c_lat, grd.lon(), grd.lat())
        d_i, d_j = nearest(d_lon, d_lat, grd.lon(), grd.lat())
                                        
        self.imin = np.maximum(np.min([a_i, b_i, c_i, d_i]) - 5, 0) # Must not go
        self.jmin = np.maximum(np.min([a_j, b_j, c_j, d_j]) - 5, 0) # below zero
        self.imax = np.max([a_i, b_i, c_i, d_i]) + 5
        self.jmax = np.max([a_j, b_j, c_j, d_j]) + 5
        return self
    


class rossby_wave_speeds (object):
  
    def __init__(self, path):
        '''
        Initialise the rossby_wave_speeds object
        '''
        data = np.loadtxt(path)
        self._lon = data[:,1] - 360.
        self._lat = data[:,0]
        self._defrad = data[:,3]
        self._distance = np.array([])
    
        
    def get_rwdistance(self, x, y, days_between_records):
        '''
        '''
        distance = self._get_rlongwave_spd(x, y)
        distance *= 86400.
        distance *= days_between_records
        return np.abs(distance)
    
    
    def make_subset(self, lonmin, lonmax, latmin, latmax):
        '''
        '''
        lloi = np.logical_and(self._lon >= lonmin, self._lon <= lonmax)
        lloi *= np.logical_and(self._lat >= latmin, self._lat <= latmax)
        self._lon = self._lon[lloi]
        self._lat = self._lat[lloi]
        self._defrad = self._defrad[lloi]
        self._make_kdtree()
    
    
    def _make_kdtree(self):
        points = np.vstack([self._lon, self._lat]).T
        self._tree = spatial.cKDTree(points)
    
    
    def _get_defrad(self, x, y):
        '''
        Get a point average of the deformation radius
        at x, y
        '''
        weights, i = self._tree.query(np.array([x, y]), k=4, p=2)
        weights /= weights.sum()
        self._weights = weights
        self.i = i
        return np.average(self._defrad[i], weights=weights)
    
    
    def _get_rlongwave_spd(self, x, y):
        '''
        Get the longwave phase speed, see Chelton etal (2008) pg 446:
          c = -beta * defrad**2 (this only for extratropical waves...)
        '''
        r_spd_long = self._get_defrad(x, y)
        r_spd_long *= 1000. # km to m
        r_spd_long **= 2
        lat = np.average(self._lat[self.i], weights=self._weights)
        beta = np.cos(np.deg2rad(lat))
        beta *= 1458e-7 # 1458e-7 ~ (2 * 7.29*10**-5)
        beta /= 6371315.0
        r_spd_long *= -beta
        return r_spd_long
    

    
    
    
    
    
    
    
'''
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
'''
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










