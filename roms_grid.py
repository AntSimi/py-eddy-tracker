# -*- coding: utf-8 -*-
# %run roms_grid.py
'''
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

roms_grid.py

Version 2.0.3

===========================================================================

Class to access ROMS gridfile information
Useful when the output files don't contain the grid
information

'''
import netCDF4 as netcdf
#from matplotlib.mlab import load
import numpy as np
import matplotlib.path as Path
from make_eddy_track_AVISO import PyEddyTracker
from mpl_toolkits.basemap import Basemap


def root(): ## Define just one root path here
    #root_dir = '/acacia/emason/runs/'
    #root_dir = '/media/hd/roms/'
    root_dir = '/home/emason/runs/'
    #root_dir = '/Users/evan/runs/'
    return root_dir
  

def getcoast(coastfile):
    if coastfile is None:
        #coastfile = '/msasa/emason/data/bathymetry/ne_atlantic_coast.dat'
        #coastfile = '/baobab/data/coastline/ne_atlantic_coast.dat'
        coastfile = '/hun/emason/data/coastline/ne_atlantic_coast.dat'
    data = np.load(coastfile)
    data = np.ma.masked_where(np.isnan(data), data)
    return data[:,0],data[:,1]


def read_nc(GRDFILE, var):
    nc = netcdf.Dataset(GRDFILE,'r')
    var = nc.variables[var][:]
    nc.close()
    return var


#def grid_info():
    



class RomsGrid (PyEddyTracker):
    '''
    The 'RomsGrid' class

    Initialises with:
      obcs : [0,0,0,1] where 1 is open [S,E,N,W]
    '''
    def __init__(self, GRDFILE, THE_DOMAIN, PRODUCT,
                 LONMIN, LONMAX, LATMIN, LATMAX, FILLVAL,
                 with_pad=True):
        """
        # Initialise the grid object
        """
        super(RomsGrid, self).__init__()
        print '\nInitialising the *RomsGrid*'
        self.THE_DOMAIN = THE_DOMAIN
        self.PRODUCT = PRODUCT
        self.LONMIN = LONMIN
        self.LONMAX = LONMAX
        self.LATMIN = LATMIN
        self.LATMAX = LATMAX
        self.FILLVAL = FILLVAL
        self.GRDFILE = GRDFILE
        
        try:
            with netcdf.Dataset(self.GRDFILE) as nc:
                pass
        except Exception:
            try:
                with netcdf.Dataset(root() + self.GRDFILE) as nc:
                    self.GRDFILE = root() + self.GRDFILE
            except Exception:
                print 'No file at: ', self.GRDFILE
                print 'or at ', root() + self.GRDFILE
                raise Exception # no grid file found

        with netcdf.Dataset(self.GRDFILE) as nc:
            self._lon = nc.variables['lon_rho'][:]
            self._lat = nc.variables['lat_rho'][:]
            self._pm = nc.variables['pm'][:]
            self._pn = nc.variables['pn'][:]
            self._f = nc.variables['f'][:]
            self._angle = nc.variables['angle'][:]
            self._mask = nc.variables['mask_rho'][:]
            self._gof = self.GRAVITY / self._f

        self.set_initial_indices()
        self.set_index_padding()
        self.set_basemap(with_pad=with_pad)
        self.uvpmask()
        self.set_u_v_eke()
        self.shape = self.lon().shape
        #pad2 = 2 * self.pad
        #self.shape = (self.f().shape[0] - pad2, self.f().shape[1] - pad2)
        
        # Parameters for different grid files; modify accordingly
        if self.GRDFILE.split('/')[-1] in 'roms_grd_NA2009_7pt5km.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            6.0, 0.0, 120.0, 32.0, 2, [[1,'S'],[1,'E'],[1,'N'],[1,'W']]
        ####
        elif self.GRDFILE.split('/')[-1] in 'gc_2009_1km_grd_smooth.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            6.0,          2.0,          120.0,   60.0,   2,           [[1,'S'],[0,'E'],[1,'N'],[1,'W']]
        ####
        elif self.GRDFILE.split('/')[-1] in 'cb_2009_3km_grd_smooth.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            6.0,          2.0,          120.0,   42.0,   2,           [[1,'S'],[0,'E'],[1,'N'],[1,'W']]
        ####
        elif self.GRDFILE.split('/')[-1] in 'roms_grd_CanBas_smooth_bnd.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            6.0,          0.0,          120.0,   32.0,   1,           [[1,'S'],[0,'E'],[1,'N'],[1,'W']]
        ####
        elif self.GRDFILE.split('/')[-1] in 'grd_Med14km_2010.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            6.0,          2.0,          120.0,   50.0,   2,           [[0,'S'],[0,'E'],[0,'N'],[1,'W']]
        ####
        elif self.GRDFILE.split('/')[-1] in 'grd_MedWest4pt75km_2010_smooth.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            6.0,          2.0,          120.0,   50.0,   2,           [[0,'S'],[0,'E'],[0,'N'],[1,'W']]
        ####
        elif self.GRDFILE.split('/')[-1] in 'grd_ATL_15km.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            10.0,          2.0,          400.0,   40.0,   2,           [[1,'S'],[1,'E'],[1,'N'],[0,'W']]
        ####
        elif self.GRDFILE.split('/')[-1] in 'grd_NA2011_7pt5km.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            6.0,          2.0,          120.0,   42.0,   2,           [[1,'S'],[1,'E'],[1,'N'],[1,'W']]
        ####
        elif self.GRDFILE.split('/')[-1] in 'grd_MedSea15.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            6.0,          2.0,          120.0,   42.0,   2,           [[0,'S'],[0,'E'],[1,'N'],[1,'W']]
        elif self.GRDFILE.split('/')[-1] in 'grd_MedSea5.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            6.0,          2.0,          120.0,   42.0,   2,           [[0,'S'],[0,'E'],[1,'N'],[1,'W']]
        elif self.GRDFILE.split('/')[-1] in 'grd_canbas2.5.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            6.0,          2.0,          120.0,   42.0,   2,           [[1,'S'],[0,'E'],[1,'N'],[1,'W']]
        elif self.GRDFILE.split('/')[-1] in 'grd_canwake4km.nc':
            self.theta_s, self.theta_b, self.hc, self.N, self.scoord, self.obcs = \
            6.0,          0.0,          200.0,   42.0,   2,           [[1,'S'],[0,'E'],[1,'N'],[1,'W']]
        else: 
            Exception # grid not specified, add a new 'elif' on line above with gridfile name and parameters 
        

            


    def lon(self):
        return self._lon[self.j0:self.j1, self.i0:self.i1]

    def lat(self):
        return self._lat[self.j0:self.j1, self.i0:self.i1]
    
    def lonpad(self):
        return self._lon[self.jp0:self.jp1, self.ip0:self.ip1]

    def latpad(self):
        return self._lat[self.jp0:self.jp1, self.ip0:self.ip1]
    
    def pm(self):
        return self._pm[self.jp0:self.jp1, self.ip0:self.ip1]

    def pn(self):
        return self._pn[self.jp0:self.jp1, self.ip0:self.ip1]

    def mask(self):
        return self._mask[self.jp0:self.jp1, self.ip0:self.ip1]

    def angle(self):
        return self._angle[self.jp0:self.jp1, self.ip0:self.ip1]

    def h(self):
        return self._h[self.jp0:self.jp1, self.ip0:self.ip1]

    def hraw(self):
        return self._hraw[self.jp0:self.jp1, self.ip0:self.ip1]

    def f(self):
        return self._f[self.jp0:self.jp1, self.ip0:self.ip1]
      
    def gof(self):
        return self._gof[self.jp0:self.jp1, self.ip0:self.ip1]
      
    def umask(self):
        return self._umask[self.jp0:self.jp1, self.ip0:self.ip1-1]
      
    def vmask(self):
        return self._vmask[self.jp0:self.jp1-1, self.ip0:self.ip1]


    def uvpmask(self):
        '''
        Get mask at u, v, psi points
        '''
        Mp, Lp = self._mask.shape
        M = Mp - 1
        L = Lp - 1
        self._umask = self._mask[:,:L] * self._mask[:,1:Lp]
        self._vmask = self._mask[:M] * self._mask[1:Mp]
        self._psimask = self._umask[:M] * self._umask[1:Mp]
        return self


    def boundary(self, imin=0, imax=-1, jmin=0, jmax=-1):
        '''
        Return lon, lat of perimeter around a ROMS grid
        Indices to get boundary of specified subgrid
        '''
        lon = np.r_[(self.lon()[jmin:jmax, imin],   self.lon()[jmax, imin:imax],
                     self.lon()[jmax:jmin:-1,imax], self.lon()[jmin, imax:imin:-1])]
        lat = np.r_[(self.lat()[jmin:jmax, imin],   self.lat()[jmax, imin:imax],
                     self.lat()[jmax:jmin:-1,imax], self.lat()[jmin, imax:imin:-1])]
        return lon, lat


    def brypath(self, imin=0, imax=-1, jmin=0, jmax=-1):
        '''
        Return Path object of perimeter around a ROMS grid
        Indices to get boundary of specified subgrid
        '''
        lon, lat = self.boundary(imin, imax, jmin, jmax)
        brypath = np.array([lon, lat]).T
        return Path.Path(brypath)


    def coastline(self):
        return getcoast(self.coastfile)



    def VertCoordType(self):
        nc = netcdf.Dataset(self.GRDFILE,'r')
        var = nc.VertCoordType
        nc.close()
        return var
    
    def get_resolution(self, meters=False):
        '''
        Get mean grid resolution in degrees or meters
        If meters defined, return degrees
        '''
        mean_earth_radius = 6371315.
        if meters: # Degrees to meters
            res = np.copy(meters)
            res *= np.pi * mean_earth_radius / 180.0
        else: # Meters to degrees
            res = np.mean(np.sqrt((1. / self.pm()) * (1. / self.pn())))
            res /= np.pi * mean_earth_radius / 180.
        return res


    def title(self):
        nc = netcdf.Dataset(self.GRDFILE,'r')
        var = nc.title
        nc.close()
        return var



    def scoord2z_r(self, zeta=0., alpha=0., beta=0., verbose=False):
        zr = scoord2z(self.h(), self.theta_s, self.theta_b, self.hc, self.N, 'r', 
                      self.scoord, zeta=zeta, alpha=alpha, beta=beta, verbose=verbose)
        return zr



    def scoord2z_w(self, zeta=0., alpha=0., beta=0., verbose=False):
        zw = scoord2z(self.h(), self.theta_s, self.theta_b, self.hc, self.N, 'w', 
                      self.scoord, zeta=zeta, alpha=alpha, beta=beta, verbose=verbose)
        return zw
    
    
    def dz(self):
        '''
        Returns cell heights
        '''
        zw = self.scoord2z_w()
        dz = np.diff(zw, axis=0)
        return dz
    
    
    def rotate_vec(self, z, m):
        '''
        Rotate to account for grid rotation
        '''
        #angle = self.angle()[jstr:jend,istr:iend].mean()
        cosa = np.cos(self.angle())
        sina = np.sin(self.angle())
        zres = (z * cosa) + (m * sina)
        mres = (m * cosa) - (z * sina)
        return zres, mres
    

    def transect(self, ln1, lt1, ln2, lt2, dx):
        '''
        Return lon/lat arrays for a transect between
        two points with resolution dx
        TO DO: check points are within domain
        Input:   1. lon/lat points  (ln1,lt1,ln2,lt2)
                 2. dx [km]
        Returns: 1. the two arrays
                 2. the distance [m]
                 2. the angle [degrees]
        '''
        dist    = et.distLonLat(ln1,lt1,ln2,lt2)
        num_stn = np.round(dist[0] / (dx * 1000.))
        tr_ln   = np.linspace(ln1,ln2,num=num_stn)
        tr_lt   = np.linspace(lt1,lt2,num=num_stn)
        return tr_ln,tr_lt,dist[0],dist[1]






