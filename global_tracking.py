# %run global_tracking.py

from netCDF4 import Dataset
import numpy as np


class GlobalTracking(object):
    """
    
    """
    def __init__(self, eddy, ymd_str):
        """
        """
        self.tracklist = eddy.tracklist
        self.SIGN_TYPE = eddy.SIGN_TYPE
        self.ymd_str = ymd_str
        self.num_tracks = np.sum([i.alive for i in self.tracklist])
    
    
    def create_netcdf(self):
        """
        """
        with Dataset('%s_%s.nc' % (self.SIGN_TYPE, self.ymd_str),
                     'w', format='NETCDF4') as nc:
            
            # Create dimensions
            nc.createDimension('tracks', self.num_tracks)
            nc.createDimension('properties', None)
            
            # Create variables
            t = 0
            for track in self.tracklist:
                if track.alive:
                    varname = 'track_%s' % np.str(t).zfill(4)
                    nc.createVariable(varname, np.float64, ('properties'))
                    t += 1
            nc.createVariable('track_lengths', np.int, ('tracks'))
    
    
    def write_tracks(self):
        """
        """
        t = 0
        for track in self.tracklist:
            tracklen = len(track.lon)
            if track.alive:
                properties = np.hstack((track.dayzero, track.saved2nc,
                                        track.save_extras, track.lon, track.lat,
                                        track.amplitude, track.radius_s,
                                        track.radius_e, track.uavg,
                                        track.teke, track.ocean_time))

                with Dataset('%s_%s.nc' % (self.SIGN_TYPE, self.ymd_str), 'a') as nc:
                    varname = 'track_%s' % np.str(t).zfill(4)
                    nc.variables[varname][:] = properties
                    nc.variables['track_lengths'][t] = tracklen
                t += 1
    
    
    def read_tracks(self):
        """
        Read and sort the property data for returning to the
        Eddy object
        """
        with Dataset('Anticyclonic_20140312.nc') as nc:
            tracklens = nc.variables['track_lengths'][:]
            for t, tracklen in enumerate(tracklengths):
                
                varname = 'track_%s' % np.str(t).zfill(4)
                track = nc.variables[varname][:]
                dayzero = track[0].astype(bool)
                saved2nc = track[1].astype(bool)
                save_extras = track[2].astype(bool)
                
                inds = np.arange(3, 3 + (tracklen * 8), tracklen)
                lon = track[inds[0]:inds[1]]
                lat = track[inds[1]:inds[2]]
                amplitude = track[inds[2]:inds[3]]
                radius_s = track[inds[3]:inds[4]]
                radius_e = track[inds[4]:inds[5]]
                uavg = track[inds[5]:inds[6]]
                teke = track[inds[6]:inds[7]]
                ocean_time = track[inds[7]:]
                
            
            properties = nc.variables['track_0000'][:]
        
        
        
        