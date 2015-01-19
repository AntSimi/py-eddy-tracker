# -*- coding: utf-8 -*-
# %run make_eddy_track_AVISO.py

"""
===============================================================================
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
===============================================================================

make_eddy_track_AVISO.py

Version 1.4.2


Scroll down to line ~640 to get started
===============================================================================
"""
import sys
import glob as glob
from py_eddy_tracker_classes import plt, np, dt, Dataset, ndimage, time, \
                                    datestr2datetime, gaussian_resolution, \
                                    get_cax, collection_loop, track_eddies, \
                                    anim_figure
import make_eddy_tracker_list_obj as eddy_tracker
from dateutil import parser
from mpl_toolkits.basemap import Basemap
import yaml


class PyEddyTracker (object):
    """
    Base object
    
    Methods defined here are grouped into categories based on input data
    source, i.e., AVISO, ROMS, etc.  To introduce a new data source new methods
    can be introduced here.
    
    METHODS:
      Common: read_nc
              read_nc_att
              set_initial_indices
              set_index_padding
              haversine_dist
              half_interp
      AVISO:  get_AVISO_f_pm_pn
      ROMS:   get_ROMS_f_pm_pn
    
    """
    def __init__(self):
        '''
        Set some constants
        '''
        self.GRAVITY = 9.81
        self.EARTH_RADIUS = 6371315.0
        self.ZERO_CROSSING = False
    
    def read_nc(self, varfile, varname, indices="[:]"):
        '''
        Read data from nectdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          indices : string of index ranges, eg. '[0,:,0]'
        '''
        with Dataset(varfile) as nc:
            try:
                var = eval(''.join(("nc.variables[varname]", indices)))
            except Exception:
                return None
            else:
                return var


    def read_nc_att(self, varfile, varname, att):
        '''
        Read data attribute from nectdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          att : string of attribute, eg. 'valid_range'
        '''
        with Dataset(varfile) as nc:
            return eval(''.join(("nc.variables[varname].", att)))


    def set_initial_indices(self, LONMIN, LONMAX, LATMIN, LATMAX):
        '''
        Get indices for desired domain
        '''
        print '--- Setting initial indices to LONMIN, LONMAX, LATMIN, LATMAX'
        self.i0, junk = self.nearest_point(LONMIN, 
                                           LATMIN + 0.5 * (LATMAX - LATMIN))
        self.i1, junk = self.nearest_point(LONMAX,
                                           LATMIN + 0.5 * (LATMAX - LATMIN))
        junk, self.j0 = self.nearest_point(LONMIN + 0.5 * (LONMAX - LONMIN),
                                                                     LATMIN)
        junk, self.j1 = self.nearest_point(LONMIN + 0.5 * (LONMAX - LONMIN),
                                                                     LATMAX)

        def kdt(lon, lat, limits, k=4):
            ppoints = np.array([lon.ravel(), lat.ravel()]).T
            ptree = spatial.cKDTree(ppoints)
            pindices = ptree.query(limits, k=k)[1]
            iind, jind = np.array([], dtype=int), np.array([], dtype=int)
            for pind in pindices.ravel():
                j, i = np.unravel_index(pind, lon.shape)
                iind = np.r_[iind, i]
                jind = np.r_[jind, j]
            return iind, jind

        if 'AvisoGrid' in self.__class__.__name__:
            
            if self.ZERO_CROSSING is True:
                '''
                Used for a zero crossing, e.g., across Agulhas region
                '''
                def half_limits(lon, lat):
                    return np.array([np.array([lon.min(), lon.max(),
                                               lon.max(), lon.min()]),
                                     np.array([lat.min(), lat.min(),
                                               lat.max(), lat.max()])]).T
                # Get bounds for right part of grid
                lat = self._lat[self._lon >= 360 + LONMIN - 0.5]
                lon = self._lon[self._lon >= 360 + LONMIN - 0.5]
                limits = half_limits(lon, lat)
                iind, jind = kdt(self._lon, self._lat, limits)
                self.i1 = iind.min()
                # Get bounds for left part of grid
                lat = self._lat[self._lon <= LONMAX + 0.5]
                lon = self._lon[self._lon <= LONMAX + 0.5]
                limits = half_limits(lon, lat)
                iind, jind = kdt(self._lon, self._lat, limits)
                self.i0 = iind.max()
        
        return self


    def set_index_padding(self, pad=2):
        '''
        Set start and end indices for temporary padding and later unpadding
        around 2d variables.
        Padded matrices are needed only for geostrophic velocity computation.
        '''
        print '--- Setting padding indices with pad=%s' %pad
        
        self.pad = pad
        
        def get_str(thestr, pad):
            '''
            Get start indices for pad
            Returns:
              pad_str   - index to add pad
              unpad_str - index for later unpadding
            '''
            pad_str = np.max([0, thestr - pad])
            if pad > 0:
                unpad_str = np.max([0, np.diff([pad_str, thestr])])
                return pad_str, unpad_str
            else:
                unpad_str = np.min([0, np.diff([pad_str, thestr])])
                return pad_str, -1 * unpad_str
    
        def get_end(theend, shape, pad):
            '''
            Get end indices for pad
            Returns:
              pad_end   - index to add pad
              unpad_end - index for later unpadding
            '''
            if theend is None:
                pad_end = None
                unpad_end = None
            else:
                pad_end = np.minimum(shape, theend + pad)
                if shape == theend + pad:
                    unpad_end = -pad
                elif shape == theend + pad - 1:
                    unpad_end = -1
                elif shape == pad_end:
                    unpad_end = None
                else:
                    unpad_end = -pad
            if pad > 0:
                return pad_end, unpad_end
            else:
                return pad_end, -1 * unpad_end
        
        self.jp0, self.jup0 = get_str(self.j0, pad)
        self.jp1, self.jup1 = get_end(self.j1, self._lon.shape[0], pad)
        if self.ZERO_CROSSING:
            pad = -pad
        self.ip0, self.iup0 = get_str(self.i0, pad)
        self.ip1, self.iup1 = get_end(self.i1, self._lon.shape[1], pad)
        return self



    def haversine_dist(self, lon1, lat1, lon2, lat2):
        '''
        TO DO: change to use f2py version
        Haversine formula to calculate distance between two lon/lat points
        Uses mean earth radius in metres (from ROMS scalars.h) = 6371315.0
        Input:
          lon1, lat1, lon2, lat2
        Return:
          distance (m)
        '''
        lon1, lat1, lon2, lat2 = (lon1.copy(), lat1.copy(),
                                  lon2.copy(), lat2.copy())
        dlat = np.deg2rad(lat2 - lat1)
        dlon = np.deg2rad(lon2 - lon1)
        np.deg2rad(lat1, out=lat1)
        np.deg2rad(lat2, out=lat2)
        a = (np.sin(0.5 * dlon))**2
        a *= np.cos(lat1) * np.cos(lat2)
        a += (np.sin(0.5 * dlat))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return 6371315.0 * c # Return the distance



    def nearest_point(self, lon, lat):
        '''
        Get indices to a point (lon, lat) in the grid
        '''
        i, j = eddy_tracker.nearest(lon, lat, self._lon, self._lat,
                                                    self._lon.shape)
        return i, j
    
    
    def half_interp(self, h_one, h_two):
        '''
        Speed up frequent operations of type 0.5 * (arr[:-1] + arr[1:])
        '''
        h_one += h_two
        h_one *= 0.5
        return h_one


    def get_AVISO_f_pm_pn(self):
        '''
        Padded matrices are used here because Coriolis (f), pm and pn
        are needed for the geostrophic velocity computation in 
        method getSurfGeostrVel()
        NOTE: this should serve for ROMS too
        '''
        print '--- Computing Coriolis (f), dx (pm), dy (pn) for padded grid'
        # Get GRAVITY / Coriolis
        self._gof = np.sin(np.deg2rad(self.latpad()))
        self._gof *= 4.
        self._gof *= np.pi
        self._gof /= 86400.
        self._f = self._gof.copy()
        self._gof = self.GRAVITY / self._gof
        
        lonu = self.half_interp(self.lonpad()[:,:-1], self.lonpad()[:,1:])
        latu = self.half_interp(self.latpad()[:,:-1], self.latpad()[:,1:])
        lonv = self.half_interp(self.lonpad()[:-1], self.lonpad()[1:])
        latv = self.half_interp(self.latpad()[:-1], self.latpad()[1:])

        # Get pm and pn
        pm = np.zeros_like(self.lonpad())
        pm[:,1:-1] = self.haversine_dist(lonu[:,:-1], latu[:,:-1],
                                         lonu[:,1:],  latu[:,1:])
        pm[:,0] = pm[:,1]
        pm[:,-1] = pm[:,-2]
        self._dx = pm
        self._pm = np.reciprocal(pm)
                
        pn = np.zeros_like(self.lonpad())
        pn[1:-1] = self.haversine_dist(lonv[:-1], latv[:-1],
                                       lonv[1:],  latv[1:])
        pn[0] = pn[1]
        pn[-1] = pn[-2]
        self._dy = pn
        self._pn = np.reciprocal(pn)
        return self


    def u2rho_2d(self, uu_in):
        '''
        Convert a 2D field at u points to a field at rho points
        '''
        def uu2ur(uu_in, Mp, Lp):
            L = Lp - 1
            Lm = L  - 1
            u_out = np.zeros((Mp, Lp))
            u_out[:, 1:L] = self.half_interp(uu_in[:, 0:Lm], uu_in[:, 1:L])
            u_out[:, 0] = u_out[:, 1]
            u_out[:, L] = u_out[:, Lm]
            return (u_out.squeeze())
        Mshp, Lshp = uu_in.shape
        return uu2ur(uu_in, Mshp, Lshp + 1)


    def v2rho_2d(self, vv_in):
        # Convert a 2D field at v points to a field at rho points
        def vv2vr(vv_in, Mp, Lp):
            M = Mp - 1
            Mm = M  - 1
            v_out = np.zeros((Mp, Lp))
            v_out[1:M] = self.half_interp(vv_in[:Mm], vv_in[1:M])
            v_out[0] = v_out[1]
            v_out[M] = v_out[Mm]
            return (v_out.squeeze())
        Mshp, Lshp = vv_in.shape
        return vv2vr(vv_in, Mshp + 1, Lshp)


    def rho2u_2d(self, rho_in):
        '''
        Convert a 2D field at rho points to a field at u points
        '''
        def _r2u(rho_in, Lp):
            u_out = rho_in[:, :Lp - 1]
            u_out += rho_in[:, 1:Lp]
            u_out *= 0.5
            return u_out.squeeze()
        assert rho_in.ndim == 2, 'rho_in must be 2d'
        Mshp, Lshp = rho_in.shape
        return _r2u(rho_in, Lshp)


    def rho2v_2d(self, rho_in):
        '''
        Convert a 2D field at rho points to a field at v points
        '''
        def _r2v(rho_in, Mp):
            v_out = rho_in[:Mp - 1]
            v_out += rho_in[1:Mp]
            v_out *= 0.5
            return v_out.squeeze()
        assert rho_in.ndim == 2, 'rho_in must be 2d'
        Mshp, Lshp = rho_in.shape
        return _r2v(rho_in, Mshp)


    def uvmask(self):
        '''
        Get mask at U and V points
        '''
        print '--- Computing umask and vmask for padded grid'
        Mp, Lp = self.mask.shape
        M = Mp - 1
        L = Lp - 1
        self._umask = self.mask[:,:L] * self.mask[:,1:Lp]
        self._vmask = self.mask[:M] * self.mask[1:Mp]
        return self


    def make_gridmask(self, with_pad=True, use_maskoceans=False):
        '''
        Use Basemap to make a landmask
        '''
        print '--- Computing Basemap'
        # Create Basemap instance for Mercator projection.
        self.M = Basemap(projection='merc', llcrnrlon = self.LONMIN - 1,
                                            urcrnrlon = self.LONMAX + 1,
                                            llcrnrlat = self.LATMIN - 1,
                                            urcrnrlat = self.LATMAX + 1,
                                            lat_ts = 0.5 * (LATMIN + LATMAX),
                                            resolution = 'h')
        if with_pad:
            x, y = self.M(self.lonpad(), self.latpad())
        else:
            x, y = self.M(self.lon(), self.lat())
        print '--- Computing Basemap mask'
        self.mask = np.ones_like(x, dtype=bool)
        if use_maskoceans:
            print "------ using Basemap *maskoceans*: this is fast but may be"
            print "------ marginally less accurate than Basemap's *is_land* method..."
            from mpl_toolkits.basemap import maskoceans
            if with_pad:
                self.mask = maskoceans(self.lonpad(), self.latpad(), self.mask,
                                      inlands=False, resolution='f', grid=1.25)
            else:
                self.mask = maskoceans(self.lon(), self.lat())
            self.mask = self.mask.mask.astype(int)
        else:
            print "------ using Basemap *is_land*: this is slow for larger domains"
            print "------ but can be speeded up once Basemap's *maskoceans* method is introduced"
            print "------ (currently longitude wrapping behaviour is unclear...)"
            it = np.nditer([x, y], flags=['multi_index'])
            while not it.finished:
                self.mask[it.multi_index] = self.M.is_land(x[it.multi_index],
                                                           y[it.multi_index])
                it.iternext()
            self.mask = np.atleast_2d(-self.mask).astype(int)
        self.Mx, self.My = x, y
        return self


    def set_geostrophic_velocity(self, zeta):
        '''
        Set u and v geostrophic velocity at
        surface from variables f, zeta, pm, pn...
        Note: output at rho points
        '''
        gof = self.gof().view()
        
        vmask = self.vmask().view()
        zeta1, zeta2 = zeta.data[1:].view(), zeta.data[:-1].view()
        pn1, pn2 = self.pn()[1:].view(), self.pn()[:-1].view()
        self.upad[:] = self.v2rho_2d(vmask * (zeta1 - zeta2) *
                                                0.5 * (pn1 + pn2))
        self.upad *= -gof
        
        umask = self.umask().view()
        zeta1, zeta2 = zeta.data[:,1:].view(), zeta.data[:,:-1].view()
        pm1, pm2 = self.pm()[:,1:].view(), self.pm()[:,:-1].view()
        self.vpad[:] = self.u2rho_2d(umask * (zeta1 - zeta2) *
                                                0.5 * (pm1 + pm2))
        self.vpad *= gof
        return self


    def set_u_v_eke(self, pad=2):
        '''
        '''
        #double_pad = pad * 2
        if self.ZERO_CROSSING:
            u1 = np.empty((self.jp1 - self.jp0, self.ip0))
            u0 = np.empty((self.jp1 - self.jp0, self._lon.shape[1] - self.ip1))
            self.upad = np.ma.concatenate((u0, u1), axis=1)
        else:
            self.upad = np.empty((self.jp1 - self.jp0, self.ip1 - self.ip0))
        self.vpad = np.empty_like(self.upad)
        self.eke = np.empty_like(self.upad[self.jup0:self.jup1,
                                           self.iup0:self.iup1])
        self.u = np.empty_like(self.eke)
        self.v = np.empty_like(self.eke)
        return self
    
    
    def getEKE(self):
        '''
        '''
        self.u[:] = self.upad[self.jup0:self.jup1, self.iup0:self.iup1]
        self.v[:] = self.vpad[self.jup0:self.jup1, self.iup0:self.iup1]
        u, v = self.u.view(), self.v.view()
        self.eke[:] = u**2 + v**2
        self.eke *= 0.5
        return self



class AvisoGrid (PyEddyTracker):
    '''
    Class to satisfy the need of the eddy tracker
    to have a grid class
    '''
    def __init__(self, AVISO_FILE, LONMIN, LONMAX, LATMIN, LATMAX,
                 with_pad=True, use_maskoceans=False):
        '''
        Initialise the grid object
        '''
        super(AvisoGrid, self).__init__()
        print '\nInitialising the AVISO_grid'
        self.i0, self.j0 = 0, 0
        self.i1, self.j1 = None, None
        self.LONMIN = LONMIN
        self.LONMAX = LONMAX
        self.LATMIN = LATMIN
        self.LATMAX = LATMAX
        
        try: # new AVISO (2014)
            self._lon = self.read_nc(AVISO_FILE, 'lon')
            self._lat = self.read_nc(AVISO_FILE, 'lat')
            self.fillval = self.read_nc_att(AVISO_FILE, 'SLA', '_FillValue')
            base_date = self.read_nc_att(AVISO_FILE, 'time', 'units')
            self.base_date = dt.date2num(
                                parser.parse(base_date.split(' ')[2:4][0]))
        
        except Exception: # old AVISO
            self._lon = self.read_nc(AVISO_FILE, 'NbLongitudes')
            self._lat = self.read_nc(AVISO_FILE, 'NbLatitudes')
            self.fillval = self.read_nc_att(AVISO_FILE,
                                           'Grid_0001', '_FillValue')
        
        if LONMIN < 0 and LONMAX <=0:
            self._lon -= 360.
        self._lon, self._lat = np.meshgrid(self._lon, self._lat)
        self._angle = np.zeros_like(self._lon)
        # To be used for handling a longitude range that
        # crosses 0 degree meridian
        if LONMIN < 0 and LONMAX >= 0:
            self.ZERO_CROSSING = True
        
        self.set_initial_indices(LONMIN, LONMAX, LATMIN, LATMAX)
        self.set_index_padding()
        self.make_gridmask(with_pad, use_maskoceans).uvmask()
        self.get_AVISO_f_pm_pn()
        self.set_u_v_eke()
        pad2 = 2 * self.pad
        self.shape = (self.f().shape[0] - pad2, self.f().shape[1] - pad2)
        

    def get_AVISO_data(self, AVISO_FILE):
        '''
        Read nc data from AVISO file
        '''
        if self.ZERO_CROSSING:
            
            try: # new AVISO (2014)
                ssh1 = self.read_nc(AVISO_FILE, 'SLA',
                       indices='[:, self.jp0:self.jp1, :self.ip0]')
                ssh0 = self.read_nc(AVISO_FILE, 'SLA',
                       indices='[:,self.jp0:self.jp1, self.ip1:]')
                ssh0, ssh1 = ssh0.squeeze(), ssh1.squeeze()
                ssh0 *= 100. # m to cm
                ssh1 *= 100. # m to cm
            
            except Exception: # old AVISO
                ssh1 = self.read_nc(AVISO_FILE, 'Grid_0001',
                       indices='[:self.ip0, self.jp0:self.jp1]').T
                ssh0 = self.read_nc(AVISO_FILE, 'Grid_0001',
                       indices='[self.ip1:,self.jp0:self.jp1]').T
            
            zeta = np.ma.concatenate((ssh0, ssh1), axis=1)
        
        else:
            
            try: # new AVISO (2014)
                zeta = self.read_nc(AVISO_FILE, 'SLA',
                       indices='[:, self.jp0:self.jp1, self.ip0:self.ip1]')
                zeta = zeta.squeeze()
                zeta *= 100. # m to cm
            
            except Exception: # old AVISO
                zeta = self.read_nc(AVISO_FILE, 'Grid_0001',
                       indices='[self.ip0:self.ip1, self.jp0:self.jp1]').T
                #date = self.read_nc_att(AVISO_FILE, 'Grid_0001', 'date') # cm

        try: # Extrapolate over land points with fillmask
            zeta = fillmask(zeta, self.mask == 1)
            #zeta = fillmask(zeta, 1 + (-1 * zeta.mask))
        except Exception: # In case no landpoints
            zeta = np.ma.masked_array(zeta)
        return zeta.astype(np.float64)



    
    
    def fillmask(self, x, mask):
        '''
        Fill missing values in an array with an average of nearest  
        neighbours
        From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
        '''
        assert x.ndim == 2, 'x must be a 2D array.'
        fill_value = 9999.99
        x[mask == 0] = fill_value
    
        # Create (i, j) point arrays for good and bad data.
        # Bad data are marked by the fill_value, good data elsewhere.
        igood = np.vstack(np.where(x != fill_value)).T
        ibad  = np.vstack(np.where(x == fill_value)).T

        # Create a tree for the bad points, the points to be filled
        tree = spatial.cKDTree(igood)

        # Get the four closest points to the bad points
        # here, distance is squared
        dist, iquery = tree.query(ibad, k=4, p=2)

        # Create a normalised weight, the nearest points are weighted as 1.
        #   Points greater than one are then set to zero
        weight = dist / (dist.min(axis=1)[:,np.newaxis])
        weight *= np.ones_like(dist)
        np.place(weight, weight > 1., 0.)

        # Multiply the queried good points by the weight, selecting only the
        # nearest points. Divide by the number of nearest points to get average
        xfill = weight * x[igood[:,0][iquery], igood[:,1][iquery]]
        xfill = (xfill / weight.sum(axis=1)[:,np.newaxis]).sum(axis=1)

        # Place average of nearest good points, xfill, into bad point locations
        x[ibad[:,0], ibad[:,1]] = xfill
        return x


    def lon(self):
        if self.ZERO_CROSSING:
            # TO DO: These concatenations are possibly expensive, they
            # shouldn't need to happen with every call to self.lon()
            lon0 = self._lon[self.j0:self.j1,  self.i1:]
            lon1 = self._lon[self.j0:self.j1, :self.i0]
            return np.concatenate((lon0 - 360., lon1), axis=1)
        else:
            return self._lon[self.j0:self.j1, self.i0:self.i1]
    
    def lat(self):
        if self.ZERO_CROSSING:
            lat0 = self._lat[self.j0:self.j1,  self.i1:]
            lat1 = self._lat[self.j0:self.j1, :self.i0]
            return np.concatenate((lat0, lat1), axis=1)
        else:            
            return self._lat[self.j0:self.j1, self.i0:self.i1]

    def lonpad(self):
        if self.ZERO_CROSSING:
            lon0 = self._lon[self.jp0:self.jp1,  self.ip1:]
            lon1 = self._lon[self.jp0:self.jp1, :self.ip0]
            return np.concatenate((lon0 - 360., lon1), axis=1)
        else:
            return self._lon[self.jp0:self.jp1, self.ip0:self.ip1]
    
    def latpad(self):
        if self.ZERO_CROSSING:
            lat0 = self._lat[self.jp0:self.jp1,  self.ip1:]
            lat1 = self._lat[self.jp0:self.jp1, :self.ip0]
            return np.concatenate((lat0, lat1), axis=1)
        else:            
            return self._lat[self.jp0:self.jp1, self.ip0:self.ip1]

    def angle(self):
        return self._angle[self.j0:self.j1, self.i0:self.i1]
    
    
    def umask(self): # Mask at U points
        return self._umask
    
    def vmask(self): # Mask at V points
        return self._vmask
    
    def f(self): #  Coriolis
        return self._f
      
    def gof(self): # Gravity / Coriolis
        return self._gof

    def dx(self): # Grid spacing along X direction
        return self._dx
    
    def dy(self): # Grid spacing along Y direction
        return self._dy
        
    def pm(self): # Reciprocal of dx
        return self._pm
    
    def pn(self): # Reciprocal of dy
        return self._pn




    def get_resolution(self):
        return np.sqrt(np.diff(self.lon()[1:], axis=1) *
                       np.diff(self.lat()[:,1:], axis=0)).mean()




    def boundary(self):
        '''
        Return lon, lat of perimeter around a ROMS grid
        Input:
          indices to get boundary of specified subgrid
        Returns:
          lon/lat boundary points
        '''
        lon = np.r_[(self.lon()[:,0],     self.lon()[-1],
                     self.lon()[::-1,-1], self.lon()[0,::-1])]
        lat = np.r_[(self.lat()[:,0],     self.lat()[-1],
                     self.lat()[::-1,-1], self.lat()[0,::-1])]
        return lon, lat


    def brypath(self, imin=0, imax=-1, jmin=0, jmax=-1):
        '''
        Return Path object of perimeter around a ROMS grid
        Indices to get boundary of specified subgrid
        '''
        lon, lat = self.boundary()
        brypath = np.array([lon, lat]).T
        return path.Path(brypath)


    def pcol_2dxy(self, x, y):
        '''
        Function to shift x, y for subsequent use with pcolor
        by Jeroen Molemaker UCLA 2008
        '''
        Mp, Lp = x.shape
        M = Mp - 1
        L = Lp - 1
        x_pcol = np.zeros((Mp, Lp))
        y_pcol = np.zeros((Mp, Lp))
        x_tmp = self.half_interp(x[:,:L], x[:,1:Lp])
        x_pcol[1:Mp,1:Lp] = self.half_interp(x_tmp[0:M,:], x_tmp[1:Mp,:])
        x_pcol[0,:] = 2. * x_pcol[1,:] - x_pcol[2,:]
        x_pcol[:,0] = 2. * x_pcol[:,1] - x_pcol[:,2]
        y_tmp = self.half_interp(y[:,0:L], y[:,1:Lp]    )
        y_pcol[1:Mp,1:Lp] = self.half_interp(y_tmp[0:M,:], y_tmp[1:Mp,:])
        y_pcol[0,:] = 2. * y_pcol[1,:] - y_pcol[2,:]
        y_pcol[:,0] = 2. * y_pcol[:,1] - y_pcol[:,2]
        return x_pcol, y_pcol









#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    
    
    
    YAML_FILE = sys.argv[1]
    print "\nLaunching with yaml file: %s" % YAML_FILE
    # Choose a yaml configuration file
    #YAML_FILE = 'eddy_tracker_configuration.yaml'
    #YAML_FILE = 'BlackSea.yaml'
   
   #--------------------------------------------------------------------------
    
    plt.close('all')
    
    # Read yaml configuration file
    with open(YAML_FILE, 'r') as stream:
        config = yaml.load(stream)
    
    # Setup configuration
    DATA_DIR = config['PATHS']['DATA_DIR']
    SAVE_DIR = config['PATHS']['SAVE_DIR']
    print '\nOutputs saved to', SAVE_DIR
    RW_PATH = config['PATHS']['RW_PATH']
    
    DIAGNOSTIC_TYPE = config['DIAGNOSTIC_TYPE']
    
    THE_DOMAIN = config['DOMAIN']['THE_DOMAIN']
    LONMIN = config['DOMAIN']['LONMIN']
    LONMAX = config['DOMAIN']['LONMAX']
    LATMIN = config['DOMAIN']['LATMIN']
    LATMAX = config['DOMAIN']['LATMAX']
    DATE_STR = config['DOMAIN']['DATE_STR']
    DATE_END = config['DOMAIN']['DATE_END']
    
    AVISO_DT14 = config['AVISO']['AVISO_DT14']
    AVISO_FILES = config['AVISO']['AVISO_FILES']
    if AVISO_DT14:
        AVISO_DT14_SUBSAMP = config['AVISO']['AVISO_DT14_SUBSAMP']
        if AVISO_DT14_SUBSAMP:
            DAYS_BTWN_RECORDS = config['AVISO']['DAYS_BTWN_RECORDS']
        else:
            DAYS_BTWN_RECORDS = 1.
    else:
        DAYS_BTWN_RECORDS = 7. # old seven day AVISO
        
    TRACK_DURATION_MIN = config['TRACK_DURATION_MIN']
   
    if 'SLA' in DIAGNOSTIC_TYPE:
        MAX_SLA = config['CONTOUR_PARAMETER']['CONTOUR_PARAMETER_SLA']['MAX_SLA']
        INTERVAL = config['CONTOUR_PARAMETER']['CONTOUR_PARAMETER_SLA']['INTERVAL']
        CONTOUR_PARAMETER = np.arange(-MAX_SLA, MAX_SLA + INTERVAL, INTERVAL)
        SHAPE_ERROR = config['SHAPE_ERROR'] * np.ones(CONTOUR_PARAMETER.size)
    elif 'Q' in DIAGNOSTIC_TYPE:
        MAX_Q = config['CONTOUR_PARAMETER']['CONTOUR_PARAMETER_Q']['MAX_Q']
        NUM_LEVS = config['CONTOUR_PARAMETER']['CONTOUR_PARAMETER_Q']['NUM_LEVS']
        CONTOUR_PARAMETER = np.linspace(0, MAX_Q, NUM_LEVS)[::-1]
    else: Exception
    
    JDAY_REFERENCE = config['JDAY_REFERENCE']

    RADMIN = config['RADMIN']
    RADMAX = config['RADMAX']
    
    if 'SLA' in DIAGNOSTIC_TYPE:
        AMPMIN = config['AMPMIN']
        AMPMAX = config['AMPMAX']
    elif 'Q' in DIAGNOSTIC_TYPE:
        AMPMIN = 0.02 # max(abs(xi/f)) within the eddy
        AMPMAX = 100.
    else: Exception

    SAVE_FIGURES = config['SAVE_FIGURES']

    SMOOTHING = config['SMOOTHING']
    if SMOOTHING:
        if 'SLA' in DIAGNOSTIC_TYPE:
            ZWL = config['SMOOTHING_SLA']['ZWL']
            MWL = config['SMOOTHING_SLA']['MWL']
        elif 'Q' in DIAGNOSTIC_TYPE:
            SMOOTH_FAC = config['SMOOTHING_Q']['SMOOTH_FAC']
        else: Exception
        
    DIST0 = config['DIST0']
    AREA0 = np.pi * config['RAD0']**2
    if 'Q' in DIAGNOSTIC_TYPE:
        AMP0 = 0.02 # vort/f
    elif 'SLA' in DIAGNOSTIC_TYPE:
        AMP0 = config['AMP0']
    TEMP0 = config['TEMP0']
    SALT0 = config['SALT0']
    
    EVOLVE_AMP_MIN = config['EVOLVE_AMP_MIN']
    EVOLVE_AMP_MAX = config['EVOLVE_AMP_MAX']
    EVOLVE_AREA_MIN = config['EVOLVE_AREA_MIN']
    EVOLVE_AREA_MAX = config['EVOLVE_AREA_MAX']
    
    SEPARATION_METHOD = config['SEPARATION_METHOD']
    
    MAX_LOCAL_EXTREMA = config['MAX_LOCAL_EXTREMA']
    
    TRACK_EXTRA_VARIABLES = config['TRACK_EXTRA_VARIABLES']

    VERBOSE = config['VERBOSE']

    CMAP = plt.cm.RdBu


    # End user configuration setup options
    #--------------------------------------------------------------------------
    
    assert DATE_STR < DATE_END, 'DATE_END must be larger than DATE_STR'
    assert DIAGNOSTIC_TYPE in ('Q', 'SLA'), 'DIAGNOSTIC_TYPE not properly defined'
    
    thestartdate = dt.date2num(datestr2datetime(str(DATE_STR)))
    theenddate = dt.date2num(datestr2datetime(str(DATE_END)))
    
    # Get complete AVISO file list
    AVISO_FILES = sorted(glob.glob(DATA_DIR + AVISO_FILES))
    
    # Use this for subsampling to get identical list as old_AVISO
    #AVISO_FILES = AVISO_FILES[5:-5:7]
    if AVISO_DT14 and AVISO_DT14_SUBSAMP:
        AVISO_FILES = AVISO_FILES[5:-5:np.int(DAYS_BTWN_RECORDS)]
    
    # Set up a grid object using first AVISO file in the list
    sla_grd = AvisoGrid(AVISO_FILES[0], LONMIN, LONMAX, LATMIN, LATMAX)
    
    
    


    Mx, My = (sla_grd.Mx[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1],
              sla_grd.My[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1])
    pMx, pMy = sla_grd.pcol_2dxy(Mx, My)
    
    

    
    
    # Instantiate search ellipse object
    search_ellipse = eddy_tracker.SearchEllipse(THE_DOMAIN, DAYS_BTWN_RECORDS,
                                    RW_PATH, [LONMIN, LONMAX, LATMIN, LATMAX])
    
    
    if isinstance(SMOOTHING, str):
        if 'Gaussian' in SMOOTHING:
            # Get parameters for ndimage.gaussian_filter
            zres, mres = gaussian_resolution(sla_grd.get_resolution(),
                         zwl, mwl)

    fig  = plt.figure(1)

    # Initialise two eddy objects to hold data
    A_eddy = eddy_tracker.TrackList('AVISO', TRACK_DURATION_MIN,
                                             TRACK_EXTRA_VARIABLES)
    C_eddy = eddy_tracker.TrackList('AVISO', TRACK_DURATION_MIN,
                                             TRACK_EXTRA_VARIABLES)

    if 'Q' in DIAGNOSTIC_TYPE:
        A_savefile = "".join([SAVE_DIR, 'eddy_tracks_Q_AVISO_anticyclonic.nc'])
        A_eddy.qparameter = qparameter
        C_eddy.qparameter = qparameter
        C_savefile = "".join([SAVE_DIR, 'eddy_tracks_Q_AVISO_cyclonic.nc'])
        A_eddy.SHAPE_ERROR = SHAPE_ERROR
        C_eddy.SHAPE_ERROR = SHAPE_ERROR
    
    elif 'SLA' in DIAGNOSTIC_TYPE:
        A_savefile = "".join([SAVE_DIR, 'eddy_tracks_SLA_AVISO_anticyclonic.nc'])
        A_eddy.CONTOUR_PARAMETER = CONTOUR_PARAMETER
        A_eddy.SHAPE_ERROR = SHAPE_ERROR
        C_savefile = "".join([SAVE_DIR, 'eddy_tracks_SLA_AVISO_cyclonic.nc'])
        C_eddy.CONTOUR_PARAMETER = CONTOUR_PARAMETER[::-1]
        C_eddy.SHAPE_ERROR = SHAPE_ERROR[::-1]
    
    A_eddy.JDAY_REFERENCE = JDAY_REFERENCE
    C_eddy.JDAY_REFERENCE = JDAY_REFERENCE
    
    A_eddy.INTERANNUAL = True
    C_eddy.INTERANNUAL = True
    
    A_eddy.DIAGNOSTIC_TYPE = DIAGNOSTIC_TYPE
    C_eddy.DIAGNOSTIC_TYPE = DIAGNOSTIC_TYPE
    
    A_eddy.SMOOTHING = SMOOTHING
    C_eddy.SMOOTHING = SMOOTHING
    
    A_eddy.MAX_LOCAL_EXTREMA = MAX_LOCAL_EXTREMA
    C_eddy.MAX_LOCAL_EXTREMA = MAX_LOCAL_EXTREMA
    
    A_eddy.M = sla_grd.M
    C_eddy.M = sla_grd.M
    
    #A_eddy.rwv = rwv
    #C_eddy.rwv = rwv
    
    A_eddy.search_ellipse = search_ellipse
    C_eddy.search_ellipse = search_ellipse
    
    
    A_eddy.SEPARATION_METHOD = SEPARATION_METHOD
    C_eddy.SEPARATION_METHOD = SEPARATION_METHOD
    
    if 'sum_radii' in SEPARATION_METHOD:
        A_eddy.sep_dist_fac = sep_dist_fac
        C_eddy.sep_dist_fac = sep_dist_fac
    
    
    A_eddy.points = np.array([sla_grd.lon().ravel(),
                              sla_grd.lat().ravel()]).T
    C_eddy.points = np.array([sla_grd.lon().ravel(),
                              sla_grd.lat().ravel()]).T
    
    A_eddy.EVOLVE_AMP_MIN = np.float(EVOLVE_AMP_MIN)
    A_eddy.EVOLVE_AMP_MAX = np.float(EVOLVE_AMP_MAX)
    A_eddy.EVOLVE_AREA_MIN = np.float(EVOLVE_AREA_MIN)
    A_eddy.EVOLVE_AREA_MAX = np.float(EVOLVE_AREA_MAX)
    
    C_eddy.EVOLVE_AMP_MIN = np.float(EVOLVE_AMP_MIN)
    C_eddy.EVOLVE_AMP_MAX = np.float(EVOLVE_AMP_MAX)
    C_eddy.EVOLVE_AREA_MIN = np.float(EVOLVE_AREA_MIN)
    C_eddy.EVOLVE_AREA_MAX = np.float(EVOLVE_AREA_MAX)

    A_eddy.i0, A_eddy.i1 = sla_grd.i0, sla_grd.i1
    A_eddy.j0, A_eddy.j1 = sla_grd.j0, sla_grd.j1
    C_eddy.i0, C_eddy.i1 = sla_grd.i0, sla_grd.i1
    C_eddy.j0, C_eddy.j1 = sla_grd.j0, sla_grd.j1
    
    A_eddy.LONMIN, A_eddy.LONMAX = np.float(LONMIN), np.float(LONMAX)
    A_eddy.LATMIN, A_eddy.LATMAX = np.float(LATMIN), np.float(LATMAX)
    C_eddy.LONMIN, C_eddy.LONMAX = np.float(LONMIN), np.float(LONMAX)
    C_eddy.LATMIN, C_eddy.LATMAX = np.float(LATMIN), np.float(LATMAX)
    
    A_eddy.RADMIN = np.float(RADMIN)
    A_eddy.RADMAX = np.float(RADMAX)
    A_eddy.AMPMIN = np.float(AMPMIN)
    A_eddy.AMPMAX = np.float(AMPMAX)
    C_eddy.RADMIN = np.float(RADMIN)
    C_eddy.RADMAX = np.float(RADMAX)
    C_eddy.AMPMIN = np.float(AMPMIN)
    C_eddy.AMPMAX = np.float(AMPMAX)
    
    A_eddy.fillval = sla_grd.fillval
    C_eddy.fillval = sla_grd.fillval
    A_eddy.VERBOSE = VERBOSE
    C_eddy.VERBOSE = VERBOSE
    
    
    # See Chelton section B2 (0.4 degree radius)
    # These should give 8 and 1000 for 0.25 deg resolution
    PIXMIN = np.round((np.pi * RADMIN**2) / sla_grd.get_resolution()**2)
    PIXMAX = np.round((np.pi * RADMAX**2) / sla_grd.get_resolution()**2)
    print '--- Pixel range = %s-%s' %(np.int(PIXMIN), np.int(PIXMAX))
    
    A_eddy.PIXEL_THRESHOLD = [PIXMIN, PIXMAX]
    C_eddy.PIXEL_THRESHOLD = [PIXMIN, PIXMAX]
    
    A_eddy.AREA0 = np.float(AREA0)
    C_eddy.AREA0 = np.float(AREA0)
    A_eddy.AMP0 = np.float(AMP0)
    C_eddy.AMP0 = np.float(AMP0)
    A_eddy.DIST0 = np.float(DIST0)
    C_eddy.DIST0 = np.float(DIST0)
    
    A_eddy.DAYS_BTWN_RECORDS = DAYS_BTWN_RECORDS
    C_eddy.DAYS_BTWN_RECORDS = DAYS_BTWN_RECORDS
    
    # Create nc files for saving of eddy tracks
    A_eddy.create_netcdf(DATA_DIR, A_savefile, 'Anticyclonic')
    C_eddy.create_netcdf(DATA_DIR, C_savefile, 'Cyclonic')

    
    # Loop through the AVISO files...
    start = True
    
    start_time = time.time()
    
    print '\nStart tracking'
    
    for AVISO_FILE in AVISO_FILES:
        
        with Dataset(AVISO_FILE) as nc:
    
            try:
                thedate = nc.OriginalName
                if 'qd_' in thedate:
                    thedate = thedate.partition('qd_')[2].partition('_')[0]
                else:
                    thedate = thedate.partition('h_')[2].partition('_')[0]
                thedate = datestr2datetime(thedate)
                thedate = dt.date2num(thedate)
            except Exception:
                thedate = nc.variables['time'][:]
                thedate += sla_grd.base_date
        
        rtime = thedate
        
        if thedate >= thestartdate and thedate <= theenddate:
            active = True
        else:
            active = False
        
        
        if active:
            
            #rec_start_time = time.time()
            print '--- AVISO_FILE:', AVISO_FILE
            
            # Holding variables
            A_eddy.reset_holding_variables()
            C_eddy.reset_holding_variables()
            
            #grdmask = grd.mask()[j0:j1,i0:i1]
            
            sla = sla_grd.get_AVISO_data(AVISO_FILE)
                
            if isinstance(SMOOTHING, str):
                    
                if 'Gaussian' in SMOOTHING:
                    
                    if 'first_record' not in locals():
                        print '------ applying Gaussian high-pass filter'
                    # Set landpoints to zero
                    np.place(sla, sla_grd.mask == False, 0.)
                    np.place(sla, sla.data == sla_grd.fillval, 0.)
                    # High pass filter, see
                    # http://stackoverflow.com/questions/6094957/high-pass-filter-for-image-processing-in-python-by-using-scipy-numpy
                    sla -= ndimage.gaussian_filter(sla, [mres, zres])
                
                elif 'Hanning' in SMOOTHING:
                    
                    print '------ applying %s passes of Hanning filter' \
                                                               % smooth_fac
                    # Do smooth_fac passes of 2d Hanning filter
                    sla = func_hann2d_fast(sla, smooth_fac)
                
                
            # Expand the landmask
            sla = np.ma.masked_where(sla_grd.mask == False, sla)
                
            # Get timing
            try:
                thedate = dt.num2date(rtime)[0]
            except:
                thedate = dt.num2date(rtime)
            yr = thedate.year
            mo = thedate.month
            da = thedate.day
                
            # Multiply by 0.01 for m
            sla_grd.set_geostrophic_velocity(sla * 0.01)
                
            # Remove padded boundary
            sla = sla[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1]
                
            # Calculate EKE
            sla_grd.getEKE()
            
            
            if 'Q' in DIAGNOSTIC_TYPE:

                okubo, xi = okubo_weiss(sla_grd)
            
                qparam = np.ma.multiply(-0.25, okubo) # see Kurian etal 2011
                
                qparam = func_hann2d_fast(qparam, hanning_passes)
                    
                # Set Q over land to zero
                qparam *= sla_grd.mask[sla_grd.jup0:sla_grd.jup1,
                                       sla_grd.iup0:sla_grd.iup1]
                #qparam = np.ma.masked_where(grdmask == False, qparam)
                xi *= sla_grd.mask[sla_grd.jup0:sla_grd.jup1,
                                   sla_grd.iup0:sla_grd.iup1]
                xi = np.ma.masked_where(sla_grd.mask[
                                        sla_grd.jup0:sla_grd.jup1,
                                        sla_grd.iup0:sla_grd.iup1] == False,
                                   xi / sla_grd.f()[sla_grd.jup0:sla_grd.jup1,
                                                    sla_grd.iup0:sla_grd.iup1])
                
                xicopy = np.ma.copy(xi)
            
            elif 'SLA' in DIAGNOSTIC_TYPE:
            
                A_eddy.sla = np.ma.copy(sla)
                C_eddy.sla = np.ma.copy(sla)
                A_eddy.slacopy = np.ma.copy(sla)
                C_eddy.slacopy = np.ma.copy(sla)
            
            # Get scalar speed
            Uspd = np.hypot(sla_grd.u, sla_grd.v)
            Uspd = np.ma.masked_where(
                        sla_grd.mask[sla_grd.jup0:sla_grd.jup1,
                        sla_grd.iup0:sla_grd.iup1] == False, Uspd)
            A_eddy.Uspd = np.ma.copy(Uspd)
            C_eddy.Uspd = np.ma.copy(Uspd)
            
            
            # Get contours of Q/sla parameter
            if 'first_record' not in locals():
                
                print '------ processing SLA contours for eddies'
                contfig = plt.figure(99)
                ax = contfig.add_subplot(111)
                
                if SAVE_FIGURES:
                    animfig = plt.figure(999)
                    animax = animfig.add_subplot(111)
                    # Colorbar axis
                    animax_cbar = get_cax(animax, dx=0.03,
                                          width=.05, position='b')
                   
            if 'Q' in DIAGNOSTIC_TYPE:
                CS = ax.contour(sla_grd.lon(),
                                sla_grd.lat(), qparam, CONTOUR_PARAMETER)
                # Get xi contour field at zero
                CSxi = ax.contour(sla_grd.lon(),
                                  sla_grd.lat(), xi, [0.])
                
            elif 'SLA' in DIAGNOSTIC_TYPE:
                A_CS = ax.contour(sla_grd.lon(),
                                  sla_grd.lat(), A_eddy.sla, CONTOUR_PARAMETER)
                # Note that CSc is for the cyclonics,
                #   CONTOUR_PARAMETER in reverse order
                C_CS = ax.contour(sla_grd.lon(),
                                  sla_grd.lat(),
                                  C_eddy.sla, CONTOUR_PARAMETER[::-1])
            
            else: Exception
            
            if True: # clear the current axis
                ax.cla()
            else: # draw debug figure
                if 'Q' in DIAGNOSTIC_TYPE:
                    ax.set_title('qparameter and xi')
                    ax.clabel(CS, np.array([CONTOUR_PARAMETER.min(),
                                            CONTOUR_PARAMETER.max()]))
                elif 'SLA' in DIAGNOSTIC_TYPE:
                    ax.set_title('CONTOUR_PARAMETER')
                    ax.clabel(A_CS, np.array([CONTOUR_PARAMETER.min(),
                                              CONTOUR_PARAMETER.max()]))
                plt.axis('image')
                plt.show()
            


            # Now we loop over the CS collection
            A_eddy.sign_type = 'Anticyclonic'
            C_eddy.sign_type = 'Cyclonic'
            if 'Q' in DIAGNOSTIC_TYPE:
                A_eddy, C_eddy = collection_loop(CS, sla_grd, rtime,
                                   A_list_obj=A_eddy, C_list_obj=C_eddy,
                                   xi=xi, CSxi=CSxi, VERBOSE=VERBOSE)
            
            elif 'SLA' in DIAGNOSTIC_TYPE:
                A_eddy = collection_loop(A_CS, sla_grd, rtime,
                                   A_list_obj=A_eddy, C_list_obj=None,
                                   sign_type=A_eddy.sign_type, VERBOSE=VERBOSE)
                # Note that C_CS is reverse order
                C_eddy = collection_loop(C_CS, sla_grd, rtime,
                                   A_list_obj=None, C_list_obj=C_eddy,
                                   sign_type=C_eddy.sign_type, VERBOSE=VERBOSE)
            
            # Debug
            if 'fig250' in locals():
                
                plt.figure(250)
                tit = 'Y' + str(yr) + 'M' + str(mo).zfill(2) + \
                                      'D' + str(da).zfill(2)
                
                if 'Q' in DIAGNOSTIC_TYPE:
                    plt.title('Q ' + tit)
                    sla_grd.M.pcolormesh(pMx, pMy, xi, CMAP=CMAP)
                    sla_grd.M.contour(Mx, My, xi, [0.],
                                      colors='k',linewidths=0.5)
                    sla_grd.M.contour(Mx, My, qparam, qparameter,
                                      colors='w', linewidths=0.3)
                    sla_grd.M.contour(Mx, My, qparam, [qparameter[0]],
                                      colors='m', linewidths=0.25)
                elif 'SLA' in DIAGNOSTIC_TYPE:
                    plt.title('sla ' + tit)
                    sla_grd.M.pcolormesh(pMx, pMy, sla, CMAP=CMAP)
                    sla_grd.M.contour(Mx, My, sla, [0.],
                                      colors='k', linewidths=0.5)
                    sla_grd.M.contour(Mx, My, sla, CONTOUR_PARAMETER,
                                      colors='w', linewidths=0.3)
                    sla_grd.M.contour(Mx, My, sla, [CONTOUR_PARAMETER[0]],
                                      colors='m', linewidths=0.25)
                plt.colorbar(orientation='horizontal')
                plt.clim(-.5, .5)
                sla_grd.M.fillcontinents()
                sla_grd.M.drawcoastlines()
                plt.show()
                #plt.clf()
                plt.close(250)


            if start:
                first_record = True
                # Set old variables equal to new variables
                A_eddy.set_old_variables()
                C_eddy.set_old_variables()
                start = False
                print '------ tracking eddies'
            else:
                first_record = False
            
            
            # Track the eddies
            #print 'start A tracking'
            #tt = time.time()
            A_eddy = track_eddies(A_eddy, first_record)
            #print 'end A tracking in %s seconds\n' %(time.time() - tt)
            
            #print 'start C tracking'
            #tt = time.time()
            C_eddy = track_eddies(C_eddy, first_record)
            #print 'end C tracking in %s seconds\n' %(time.time() - tt)
            #print 'dddddddd'
            
            if SAVE_FIGURES: # Make figures for animations
                
                #tit = 'Y' + str(yr) + 'M' + str(mo).zfill(2) + 'D' + str(da).zfill(2)
                tit = ''.join((str(yr), str(mo).zfill(2), str(da).zfill(2)))
                #tit = str(yr) + str(mo).zfill(2) + str(da).zfill(2)
                
                #if 'anim_fig' in locals():
                    ## Wait if there is a still-active anim_fig thread
                    #anim_fig.join()
                
                if 'Q' in DIAGNOSTIC_TYPE:
                    #anim_fig = threading.Thread(name='anim_figure', target=anim_figure,
                             #args=(33, M, pMx, pMy, xicopy, CMAP, rtime, DIAGNOSTIC_TYPE, Mx, My, 
                                   #xi.copy(), qparam.copy(), qparameter, A_eddy, C_eddy,
                                   #SAVE_DIR, plt, 'Q ' + tit))
                    anim_figure(A_eddy, C_eddy, Mx, My, pMx, pMy, plt.cm.RdBu_r, rtime, DIAGNOSTIC_TYPE, 
                                SAVE_DIR, 'Q-parameter ' + tit, animax, animax_cbar,
                                qparam=qparam, qparameter=qparameter, xi=xi, xicopy=xicopy)
                
                elif 'SLA' in DIAGNOSTIC_TYPE:
                    """anim_fig = threading.Thread(name='anim_figure', target=anim_figure,
                             args=(33, M, pMx, pMy, slacopy, plt.cm.RdBu_r, rtime, DIAGNOSTIC_TYPE, Mx, My, 
                                   slacopy, slacopy, CONTOUR_PARAMETER, A_eddy, C_eddy,
                                   SAVE_DIR, plt, 'SLA ' + tit))"""

                    #print 'figure saving'
                    #tt = time.time()
                    
                    anim_figure(A_eddy, C_eddy, Mx, My, pMx, pMy, plt.cm.RdBu_r, rtime, DIAGNOSTIC_TYPE, 
                                SAVE_DIR, 'SLA ' + tit, animax, animax_cbar)
                    #print 'figure saving done in %s seconds\n' %(time.time() - tt)
                #anim_fig.start()
                
            # Save inactive eddies to nc file
            # IMPORTANT: this must be done at every time step!!
            #saving_start_time = time.time()
            if not first_record:
                if VERBOSE:
                    print '--- saving to nc', A_eddy.SAVE_DIR
                    print '--- saving to nc', C_eddy.SAVE_DIR
                    print '+++'
                A_eddy.write2netcdf(rtime)
                C_eddy.write2netcdf(rtime)
                
            #print 'Saving the eddies', time.time() - saving_start_time, 'seconds'
            # Running time for a single monthly file
            #print '--- duration', str((time.time() - file_time) / 60.), 'minutes'
        
        if str(DATE_END) in AVISO_FILE:
            active = False
    
    # Total running time    
    print 'Duration', str((time.time() - start_time) / 3600.), 'hours!'

    print '\nOutputs saved to', SAVE_DIR
