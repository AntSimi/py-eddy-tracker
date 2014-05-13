# -*- coding: utf-8 -*-
# %run make_eddy_track_AVISO.py

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

make_eddy_track_AVISO.py

Version 1.2.0


Scroll down to line ~640 to get started
===========================================================================
"""


import glob as glob
#import matplotlib.pyplot as plt
from py_eddy_tracker_classes import *
from make_eddy_tracker_list_obj import *
from dateutil import parser

class py_eddy_tracker (object):
    '''
    Base object
    
    Methods defined here are grouped into categories based on input data source,
    i.e., AVISO, ROMS, etc.  To introduce a new data source new methods can be
    introduced here.
    
    METHODS:
      Common: read_nc
              read_nc_att
              set_initial_indices
              set_index_padding
              haversine_dist
              half_interp
      AVISO:  get_AVISO_f_pm_pn
      ROMS:   get_ROMS_f_pm_pn
    
    '''
    def __init__(self):
        '''
        Set some constants
        '''
        gravity = 9.81
        earth_radius = 6371315.0
    
    
    def read_nc(self, varfile, varname, indices="[:]"):
        '''
        Read data from nectdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          indices : string of index ranges, eg. '[0,:,0]'
        '''
        with netcdf.Dataset(varfile) as nc:
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
        with netcdf.Dataset(varfile) as nc:
            return eval(''.join(("nc.variables[varname].", att)))


    def set_initial_indices(self, lonmin, lonmax, latmin, latmax):
        '''
        Get indices for desired domain
        '''
        print '--- Setting initial indices to lonmin, lonmax, latmin, latmax'
        self.i0, junk = self.nearest(lonmin, latmin + 0.5 * (latmax - latmin))
        self.i1, junk = self.nearest(lonmax, latmin + 0.5 * (latmax - latmin))
        junk, self.j0 = self.nearest(lonmin + 0.5 * (lonmax - lonmin), latmin)
        junk, self.j1 = self.nearest(lonmin + 0.5 * (lonmax - lonmin), latmax)

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

        if self.zero_crossing is True:
            '''
            Used for a zero crossing, e.g. across Agulhas region
            '''
            def half_limits(lon, lat):
                return np.array([np.array([lon.min(), lon.max(),
                                           lon.max(), lon.min()]),
                                 np.array([lat.min(), lat.min(),
                                           lat.max(), lat.max()])]).T
            # Get bounds for right part of grid
            lat = self._lat[self._lon >= 360 + lonmin - 0.5]
            lon = self._lon[self._lon >= 360 + lonmin - 0.5]
            limits = half_limits(lon, lat)
            iind, jind = kdt(self._lon, self._lat, limits)
            self.i1 = iind.min()
            # Get bounds for left part of grid
            lat = self._lat[self._lon <= lonmax + 0.5]
            lon = self._lon[self._lon <= lonmax + 0.5]
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
        if self.zero_crossing:
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
        lon1, lat1, lon2, lat2 = lon1.copy(), lat1.copy(), lon2.copy(), lat2.copy()
        dlat = np.deg2rad(lat2 - lat1)
        dlon = np.deg2rad(lon2 - lon1)
        np.deg2rad(lat1, out=lat1)
        np.deg2rad(lat2, out=lat2)
        a = ne.evaluate('sin(0.5 * dlon) * sin(0.5 * dlon)')
        a = ne.evaluate('a * cos(lat1) * cos(lat2)')
        a = ne.evaluate('a + (sin(0.5 * dlat) * sin(0.5 * dlat))')
        c = ne.evaluate('2 * arctan2(sqrt(a), sqrt(1 - a))')
        return ne.evaluate('6371315.0 * c') # Return the distance



    
    
    def half_interp(self, h_one, h_two):
        '''
        Speed up frequent operations of type 0.5 * (arr[:-1] + arr[1:])
        '''
        h_one += h_two
        h_one *= 0.5
        return h_one
        #return ne.evaluate('0.5 * (h_one + h_two)')


    def get_AVISO_f_pm_pn(self):
        '''
        Padded matrices are used here because Coriolis (f), pm and pn
        are needed for the geostrophic velocity computation in 
        method getSurfGeostrVel()
        NOTE: this should serve for ROMS too
        '''
        print '--- Computing g/f (gravity/Coriolis), pm (dx) and pn (dy) for padded grid'
        # Get gravity / Coriolis
        self._gof = np.sin(np.deg2rad(self.latpad()))
        self._gof *= 4.
        self._gof *= np.pi
        self._gof /= 86400.
        self._gof = 9.81 / self._gof
        
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
        self.M = Basemap(projection='merc', llcrnrlon = self.lonmin - 1, \
                                            urcrnrlon = self.lonmax + 1, \
                                            llcrnrlat = self.latmin - 1, \
                                            urcrnrlat = self.latmax + 1, \
                                            lat_ts = 0.5 * (latmin + latmax), \
                                            resolution = 'h') # 'h'-high, 'l'-low
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








class AVISO_grid (py_eddy_tracker):
    '''
    Class to satisfy the need of the ROMS eddy tracker
    to have a grid class
    '''
    def __init__(self, AVISO_file, lonmin, lonmax, latmin, latmax, with_pad=True, use_maskoceans=False):
        '''
        Initialise the grid object
        '''
        super(py_eddy_tracker, self).__init__()
        print '\nInitialising the AVISO_grid'
        self.i0, self.j0 = 0, 0
        self.i1, self.j1 = None, None
        self.lonmin = lonmin
        self.lonmax = lonmax
        self.latmin = latmin
        self.latmax = latmax
        
        try: # new AVISO (2014)
            self._lon = self.read_nc(AVISO_file, 'lon')
            self._lat = self.read_nc(AVISO_file, 'lat')
            self.fillval = self.read_nc_att(AVISO_file, 'sla', '_FillValue')
            base_date = self.read_nc_att(AVISO_file, 'time', 'units')
            self.base_date = dt.date2num(parser.parse(base_date.split(' ')[2:4][0]))
        except Exception: # old AVISO
            self._lon = self.read_nc(AVISO_file, 'NbLongitudes')
            self._lat = self.read_nc(AVISO_file, 'NbLatitudes')
            self.fillval = self.read_nc_att(AVISO_file, 'Grid_0001', '_FillValue')
        if np.logical_and(lonmin < 0, lonmax <=0):
            self._lon -= 360.
        self._lon, self._lat = np.meshgrid(self._lon, self._lat)
        self._angle = np.zeros_like(self._lon)
        # To be used for handling a longitude range that crosses 0 degree meridian
        if np.logical_and(lonmin < 0, lonmax >= 0):
            self.zero_crossing = True
        else:
            self.zero_crossing = False
        
        self.set_initial_indices(lonmin, lonmax, latmin, latmax)
        self.set_index_padding()
        self.make_gridmask(with_pad, use_maskoceans).uvmask()
        self.get_AVISO_f_pm_pn()
        


    def get_AVISO_data(self, AVISO_file):
        '''
        Read nc data from AVISO file
        '''
        if self.zero_crossing:
            try: # new AVISO (2014)
                ssh1 = self.read_nc(AVISO_file, 'sla',
                       indices='[:, self.jp0:self.jp1, :self.ip0]')
                ssh0 = self.read_nc(AVISO_file, 'sla',
                       indices='[:,self.jp0:self.jp1, self.ip1:]')
                ssh0, ssh1 = ssh0.squeeze(), ssh1.squeeze()
                ssh0 *= 100. # m to cm
                ssh1 *= 100. # m to cm
            except Exception: # old AVISO
                ssh1 = self.read_nc(AVISO_file, 'Grid_0001',
                       indices='[:self.ip0, self.jp0:self.jp1]').T
                ssh0 = self.read_nc(AVISO_file, 'Grid_0001',
                       indices='[self.ip1:,self.jp0:self.jp1]').T
            zeta = np.ma.concatenate((ssh0, ssh1), axis=1)
        else:
            try: # new AVISO (2014)
                zeta = self.read_nc(AVISO_file, 'sla',
                       indices='[:, self.jp0:self.jp1, self.ip0:self.ip1]')
                zeta = zeta.squeeze()
                zeta *= 100. # m to cm
            except Exception: # old AVISO
                zeta = self.read_nc(AVISO_file, 'Grid_0001',
                       indices='[self.ip0:self.ip1, self.jp0:self.jp1]').T
                #date = self.read_nc_att(AVISO_file, 'Grid_0001', 'date') # cm

        try: # Extrapolate over land points with fillmask
            zeta = fillmask(zeta, self.mask == 1)
            #zeta = fillmask(zeta, 1 + (-1 * zeta.mask))
        except Exception: # In case no landpoints
            zeta = np.ma.masked_array(zeta)
        return zeta



    
    
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
        np.place(weight, weight>1., 0.)

        # Multiply the queried good points by the weight, selecting only the
        # nearest points.  Divide by the number of nearest points to get average
        xfill = weight * x[igood[:,0][iquery], igood[:,1][iquery]]
        xfill = (xfill / weight.sum(axis=1)[:,np.newaxis]).sum(axis=1)

        # Place average of nearest good points, xfill, into bad point locations
        x[ibad[:,0], ibad[:,1]] = xfill
        return x


    def getSurfGeostrVel(self, zeta):
        '''
        Returns u and v geostrophic velocity at
        surface from variables f, zeta, pm, pn...
        Note: output at rho points
        Adapted from IRD surf_geostr_vel.m function
        by Evan Mason
        Changed to gv2, 14 May 07...
        '''
        def gv2(zeta, gof, pm, pn, umask, vmask): # Pierrick's version
            ugv = -gof * self.v2rho_2d(vmask * (zeta[1:] - zeta[:-1]) \
                                        * 0.5 * (pn[1:] + pn[:-1]))
            vgv =  gof * self.u2rho_2d(umask * (zeta[:, 1:] - zeta[:, :-1]) \
                                        * 0.5 * (pm[:, 1:] + pm[:, :-1]))
            return ugv, vgv
        return gv2(zeta, self.gof(), self.pm(), self.pn(), self.umask(), self.vmask())


    def lon(self):
        if self.zero_crossing:
            # TO DO: These concatenations are possibly expensive, they shouldn't need
            # to happen with every call to self.lon()
            lon0 = self._lon[self.j0:self.j1,  self.i1:]
            lon1 = self._lon[self.j0:self.j1, :self.i0]
            return np.concatenate((lon0 - 360., lon1), axis=1)
        else:
            return self._lon[self.j0:self.j1, self.i0:self.i1]
    
    def lat(self):
        if self.zero_crossing:
            lat0 = self._lat[self.j0:self.j1,  self.i1:]
            lat1 = self._lat[self.j0:self.j1, :self.i0]
            return np.concatenate((lat0, lat1), axis=1)
        else:            
            return self._lat[self.j0:self.j1, self.i0:self.i1]

    def lonpad(self):
        if self.zero_crossing:
            lon0 = self._lon[self.jp0:self.jp1,  self.ip1:]
            lon1 = self._lon[self.jp0:self.jp1, :self.ip0]
            return np.concatenate((lon0 - 360., lon1), axis=1)
        else:
            return self._lon[self.jp0:self.jp1, self.ip0:self.ip1]
    
    def latpad(self):
        if self.zero_crossing:
            lat0 = self._lat[self.jp0:self.jp1,  self.ip1:]
            lat1 = self._lat[self.jp0:self.jp1, :self.ip0]
            return np.concatenate((lat0, lat1), axis=1)
        else:            
            return self._lat[self.jp0:self.jp1, self.ip0:self.ip1]

    def angle(self):
        return self._angle[self.j0:self.j1, self.i0:self.i1]

    #def mask(self, mask):
        #return -mask.mask
    
    
    

    
    
    def umask(self): # Mask at U points
        return self._umask
    
    def vmask(self): # Mask at V points
        return self._vmask
    
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
        return np.mean(np.sqrt(np.diff(self.lon()[1:], axis=1) *
                               np.diff(self.lat()[:,1:], axis=0)))

    def nearest(self, lon, lat):
        '''
        Get indices to point (lon, lat) in sla_grd
        '''
        i, j = eddy_tracker.nearest(lon, lat, self._lon, self._lat)
        return i, j



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









#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':
    
    plt.close('all')
    
    #----------------------------------------------------------------------------
    # Some user-defined input...


    # Specify the AVISO domain
    the_domain = 'Global'
    #the_domain = 'BlackSea'
    #the_domain = 'MedSea' # not yet implemented
    
    # Specify use of new AVISO 2014 data
    new_AVISO = True
    
    # Set path(s) to directory where SSH data are stored...
    if 'Global' in the_domain:
        if new_AVISO:
            directory = '/path/to/your/aviso_data/'
            #directory = '/shared/Altimetry/global/delayed-time/grids/msla/all-sat-merged/h/'
            AVISO_files = 'dt_global_allsat_msla_h_????????_20140106.nc'
            days_btwn_recs = 1.
        else:
            directory = '/path/to/your/aviso_data/'
            #directory = '/marula/emason/data/altimetry/MSLA/GLOBAL/DT/REF/'
            AVISO_files = 'dt_ref_global_merged_msla_h_qd_????????_*.nc'
            days_btwn_recs = 7.
    
    elif 'MedSea' in the_domain:
        if new_AVISO:
            directory = '/path/to/your/aviso_data/'
            #directory = '/shared/Altimetry/regional-mediterranean/delayed-time/grids/msla/all-sat-merged/h/'
            AVISO_files = 'dt_blacksea_allsat_msla_h_????????_*.nc'
        else:
            pass
    
    elif 'BlackSea' in the_domain:
        if new_AVISO:
            directory = '/path/to/your/aviso_data/'
            #irectory = '/shared/Altimetry/regional-blacksea/delayed-time/grids/msla/all-sat-merged/h/'
            AVISO_files = 'dt_blacksea_allsat_msla_h_????????_*.nc'
            days_btwn_recs = 1.
    else:
        Exception #no_domain_specified
    
    
    # Set date range (YYYYMMDD)
    date_str, date_end = 19980107, 19991110 # 
    #date_str, date_end = 20081107, 20100110 # 
    #date_str, date_end = 19930101, 20121231 # 
    
    # Choose type of diagnostic: either q-parameter ('Q') or sea level anomaly ('sla')
    #diag_type = 'Q' <<< not implemented in 1.2.0
    diag_type = 'sla'
    
    
    # Path to directory where outputs are to be saved...
    #savedir = directory
    #savedir = '/marula/emason/aviso_eddy_tracking/pablo_exp/'
    #savedir = '/marula/emason/aviso_eddy_tracking/new_AVISO_test/'
    #savedir = '/marula/emason/aviso_eddy_tracking/new_AVISO_test_0.35/'
    #savedir = '/marula/emason/aviso_eddy_tracking/new_AVISO_test_0.3/'
    #savedir = '/shared/emason/eddy_tracks/Barbara/2009/'
    #savedir = '/marula/emason/aviso_eddy_tracking/new_AVISO_test/BlackSea/'
    savedir = '/path/to/save/your/outputs/'
    
    
    # True to save outputs in same style as Chelton
    chelton_style_nc = True
    
    print '\nOutputs saved to', savedir
    
    
    
    # Reference Julian day (Julian date at Jan 1, 1992)
    jday_ref = 2448623

    # Min and max permitted eddy radii [m]
    if 'Q' in diag_type:
        Exception # Not implemented
        radmin = 15000.
        radmax = 250000.
        ampmin = 0.02
        ampmax = 100.
        
    elif 'sla' in diag_type:
        radmin = 0.3 # degrees (Chelton recommends ~50 km minimum)
        radmax = 4.461 # degrees
        ampmin = 1. # cm
        ampmax = 150.

    
    # Obtain this file from:
    # http://www-po.coas.oregonstate.edu/research/po/research/rossby_radius/
    rw_path = '/home/emason/data_tmp/chelton_eddies/rossrad.dat'

    
    # Make figures for animations
    anim_figs = True


    # Define contours
    if 'Q' in diag_type:
        # Set Q contour spacing 
        qparameter = np.linspace(0, 5*10**-11, 25)
    elif 'sla' in diag_type:
        # Set SLA contour spacing
        slaparameter = np.arange(-100., 101., 1.0) # cm

    
    
    # Apply a filter to the Q parameter
    #smoothing = 'Hanning'
    smoothing = 'Gaussian'
    if 'smoothing' not in locals():
        smoothing = False
        smooth_fac = False
    elif 'Hanning' in smoothing: # apply Hanning filter
        smooth_fac = 5 # number of passes
    elif 'Gaussian' in smoothing: # apply Gaussian filter
        smooth_fac = 'Deprecated'
        zwl = 20. # degrees, zonal wavelength (see Chelton etal 2011)
        mwl = 10. # degrees, meridional wavelength
    else:
        Exception
    
    # Save only tracks longer than...     
    track_duration_min = 28. # days

    subdomain = True
    if the_domain in 'Global':
        lonmin = -36.     # Canary
        lonmax = -7.5
        latmin = 18.
        latmax = 33.2

        #lonmin = -65.     # Corridor
        #lonmax = -5.5
        #latmin = 11.5
        #latmax = 38.5

        #lonmin = -179.     # SEP
        #lonmax = -65
        #latmin = -40.
        #latmax = -5.
    
        #lonmin = -70.     # Souza
        #lonmax = 30.
        #latmin = -50.
        #latmax = -15.

        #lonmin = -1.5     # Charlotte
        #lonmax = 51.
        #latmin = -47.
        #latmax = -24.
    
    elif the_domain in 'MedSea':
        lonmin = 354.     # SEP
        lonmax = 396
        latmin = 30.
        latmax = 46.
    
    elif the_domain in 'BlackSea':
        lonmin = 27.     # SEP
        lonmax = 42.
        latmin = 40.
        latmax = 47.
    
    # Typical parameters
    dist0 = 25000. # m separation distance after ~7 days (see CSS11 fig 22)
    if 'Q' in diag_type:
        amp0 = 0.02 # vort/f
    elif 'sla' in diag_type:
        amp0 = 2. # cm
    area0 = np.pi * 60000.**2
    temp0 = 15.
    salt0 = 35.
    
    # Parameters used by Chelton etal and Kurian etal (Sec. 3.2) to ensure the slow evolution
    # of the eddies over time; they use min and max values of 0.25 and 2.5
    evolve_ammin = 0.05# 0.25 # min change in amplitude
    evolve_ammax = 5#2.5  # max change in amplitude
    evolve_armin = 0.05# 0.25 # min change in area
    evolve_armax = 5  # max change in area
    
    
    separation_method = 'ellipse' # see CSS11
    #separation_method = 'sum_radii' # see Kurian etal (2011)
    
    #if 'sum_radii' in separation_method:
        ## Separation distance factor. Adjust according to number of days between records
        ## For 7 days, Chelton uses 150 km search ellipse
        ## So, given typical eddy radius of r=50 km, 1.5 * (r1 + r2) = 150 km.
        ##sep_dist_fac = 1.0
        #sep_dist_fac = 1.15 # Seems ok for AVISO 7-day
        ##sep_dist_fac = 1.5 # Causes tracks to jump for AVISO 7-day
    
    

    cmap = plt.cm.RdBu
    
    verbose = False


    # End user defined options (edit below at own risk)
    #----------------------------------------------------------------------------
    
    assert date_str < date_end, 'date_end must be larger than date_str'
    assert diag_type in ('Q','sla'), 'diag_type not properly defined'
    
    thestartdate = dt.date2num(datestr2datetime(str(date_str)))
    theenddate = dt.date2num(datestr2datetime(str(date_end)))
    
    # Get complete AVISO file list
    AVISO_files = sorted(glob.glob(directory + AVISO_files))
    
    # Set up a grid object using first AVISO file in the list
    sla_grd = AVISO_grid(AVISO_files[0], lonmin, lonmax, latmin, latmax)
    
    if 'Q' in diag_type:
        
        # Search from 5e-11 onwards with fine spacing near the lower end
        qparameter = np.power(np.linspace(0., np.sqrt(qparameter.max()),
                              qparameter.size), 2)[::-1]
        
        # The shape error can maybe be played with...
        shape_err = np.power(np.linspace(85., 40,  qparameter.size), 2) / 100.
        shape_err[shape_err < 35.] = 35.
        
    elif 'sla' in diag_type:
        shape_err = 55. * np.ones(slaparameter.size)
        #shape_err = 1000. * np.ones(slaparameter.size)
        #shape_err = np.power(np.linspace(85., 40,  slaparameter.size), 2) / 100.
        #shape_err[shape_err < 50.] = 50.
    


    Mx, My = sla_grd.Mx[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1], \
             sla_grd.My[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1]
    pMx, pMy = sla_grd.pcol_2dxy(Mx, My)
    
    
    ## Instantiate Rossby deformation radius object
    #rwv = eddy_tracker.RossbyWaveSpeed(the_domain,
            #[lonmin, lonmax, latmin, latmax], rw_path=rw_path)
    
    
    # Instantiate search ellipse object
    search_ellipse = eddy_tracker.SearchEllipse(the_domain, days_btwn_recs,
                                    rw_path, [lonmin, lonmax, latmin, latmax])
    
    
    if isinstance(smoothing, str):
        if 'Gaussian' in smoothing:
            # Get parameters for ndimage.gaussian_filter
            zres, mres = gaussian_resolution(sla_grd.get_resolution(), zwl, mwl)
    
    fig  = plt.figure(1)
    #axis = plt.axes()

    # Initialise two eddy objects to hold data
    A_eddy = eddy_tracker.track_list('AVISO', track_duration_min)
    C_eddy = eddy_tracker.track_list('AVISO', track_duration_min)
    
    if 'Q' in diag_type:
        A_savefile = "".join([savedir, 'eddy_tracks_Q_AVISO_anticyclonic.nc'])
        A_eddy.qparameter = qparameter
        C_eddy.qparameter = qparameter
        C_savefile = "".join([savedir, 'eddy_tracks_Q_AVISO_cyclonic.nc'])
        A_eddy.shape_err = shape_err
        C_eddy.shape_err = shape_err
    
    elif 'sla' in diag_type:
        A_savefile = "".join([savedir, 'eddy_tracks_SLA_AVISO_anticyclonic.nc'])
        A_eddy.slaparameter = slaparameter
        A_eddy.shape_err = shape_err
        C_savefile = "".join([savedir, 'eddy_tracks_SLA_AVISO_cyclonic.nc'])
        C_eddy.slaparameter = slaparameter[::-1]
        C_eddy.shape_err = shape_err[::-1]
    
    A_eddy.chelton_style_nc = chelton_style_nc
    C_eddy.chelton_style_nc = chelton_style_nc
    
    A_eddy.jday_ref = jday_ref
    C_eddy.jday_ref = jday_ref
    
    A_eddy.interannual = True
    C_eddy.interannual = True
    
    A_eddy.diag_type = diag_type
    C_eddy.diag_type = diag_type
    
    A_eddy.smoothing = smoothing
    C_eddy.smoothing = smoothing
    A_eddy.smooth_fac = smooth_fac
    C_eddy.smooth_fac = smooth_fac
    
    A_eddy.M = sla_grd.M
    C_eddy.M = sla_grd.M
    
    #A_eddy.rwv = rwv
    #C_eddy.rwv = rwv
    
    A_eddy.search_ellipse = search_ellipse
    C_eddy.search_ellipse = search_ellipse
    
    
    A_eddy.separation_method = separation_method
    C_eddy.separation_method = separation_method
    
    if 'sum_radii' in separation_method:
        A_eddy.sep_dist_fac = sep_dist_fac
        C_eddy.sep_dist_fac = sep_dist_fac
    
    
    A_eddy.points = np.array([sla_grd.lon().ravel(),
                              sla_grd.lat().ravel()]).T
    C_eddy.points = np.array([sla_grd.lon().ravel(),
                              sla_grd.lat().ravel()]).T
    
    A_eddy.evolve_ammin = np.float(evolve_ammin)
    A_eddy.evolve_ammax = np.float(evolve_ammax)
    A_eddy.evolve_armin = np.float(evolve_armin)
    A_eddy.evolve_armax = np.float(evolve_armax)
    
    C_eddy.evolve_ammin = np.float(evolve_ammin)
    C_eddy.evolve_ammax = np.float(evolve_ammax)
    C_eddy.evolve_armin = np.float(evolve_armin)
    C_eddy.evolve_armax = np.float(evolve_armax)

    A_eddy.i0, A_eddy.i1 = sla_grd.i0, sla_grd.i1
    A_eddy.j0, A_eddy.j1 = sla_grd.j0, sla_grd.j1
    C_eddy.i0, C_eddy.i1 = sla_grd.i0, sla_grd.i1
    C_eddy.j0, C_eddy.j1 = sla_grd.j0, sla_grd.j1
    
    A_eddy.lonmin, A_eddy.lonmax = np.float(lonmin), np.float(lonmax)
    A_eddy.latmin, A_eddy.latmax = np.float(latmin), np.float(latmax)
    C_eddy.lonmin, C_eddy.lonmax = np.float(lonmin), np.float(lonmax)
    C_eddy.latmin, C_eddy.latmax = np.float(latmin), np.float(latmax)
    
    A_eddy.radmin = np.float(radmin)
    A_eddy.radmax = np.float(radmax)
    A_eddy.ampmin = np.float(ampmin)
    A_eddy.ampmax = np.float(ampmax)
    C_eddy.radmin = np.float(radmin)
    C_eddy.radmax = np.float(radmax)
    C_eddy.ampmin = np.float(ampmin)
    C_eddy.ampmax = np.float(ampmax)
    
    A_eddy.fillval = sla_grd.fillval
    C_eddy.fillval = sla_grd.fillval
    A_eddy.verbose = verbose
    C_eddy.verbose = verbose
    
    # See Chelton section B2 (0.4 degree radius)
    # These should give 8 and 1000 for 0.25 deg resolution
    pixmin = np.round((np.pi * radmin**2) / sla_grd.get_resolution()**2)
    pixmax = np.round((np.pi * radmax**2) / sla_grd.get_resolution()**2)
    
    print '--- Pixel range = %s-%s' %(np.int(pixmin), np.int(pixmax))
    
    A_eddy.pixel_threshold = [pixmin, pixmax]
    C_eddy.pixel_threshold = [pixmin, pixmax]
    
    A_eddy.area0 = np.float(area0)
    C_eddy.area0 = np.float(area0)
    A_eddy.amp0 = np.float(amp0)
    C_eddy.amp0 = np.float(amp0)
    A_eddy.dist0 = np.float(dist0)
    C_eddy.dist0 = np.float(dist0)
    
    A_eddy.days_btwn_recs = days_btwn_recs
    C_eddy.days_btwn_recs = days_btwn_recs
    
    # Create nc files for saving of eddy tracks
    A_eddy.create_netcdf(directory, A_savefile, 'Anticyclonic')
    C_eddy.create_netcdf(directory, C_savefile, 'Cyclonic')

    
    # Loop through the AVISO files...
    start = True
    start_time = time.time()
    print '\nStart tracking'
    for AVISO_file in AVISO_files:
        
        
        with netcdf.Dataset(AVISO_file) as nc:
	    
            try:
               thedate = nc.OriginalName
               thedate = thedate.partition('qd_')[2].partition('_')[0]
               thedate = datestr2datetime(thedate)
               thedate = dt.date2num(thedate)
            except Exception:
               thedate = nc.variables['time'][:]
               thedate += sla_grd.base_date
        rtime = thedate
        
        if np.logical_and(thedate >= thestartdate,
                          thedate <= theenddate):
            active = True
        else:
            active = False
        
        
        if active:
            
            #rec_start_time = time.time()
            print '--- AVISO_file:', AVISO_file
                
            # Holding variables
            A_eddy.reset_holding_variables()
            C_eddy.reset_holding_variables()
            
            #grdmask = grd.mask()[j0:j1,i0:i1]
            
            if 'Q' in diag_type:
                u, v, temp, salt, rtime = get_ROMS_data(filename, pad, record, sigma_lev,
                                                ip0, ip1, jp0, jp1, diag_type)
                # Sort out masking (important for subsurface fields)
                u = np.ma.masked_outside(u, -10., 10.)
                v = np.ma.masked_outside(v, -10., 10.)

                u.data[u.mask] = 0.
                v.data[v.mask] = 0.
                u = u.data
                v = v.data

                okubo, xi = okubo_weiss(u, v, grd.pm()[jp0:jp1,
                                                       ip0:ip1],
                                              grd.pn()[jp0:jp1,
                                                       ip0:ip1])
            
                qparam = np.ma.multiply(-0.25, okubo) # see Kurian etal 2011
            
                # Remove padded boundary
                qparam = qparam[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1]
                xi = xi[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1]
                u = u[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1]
                v = v[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1]
            
                u = u2rho_2d(u)
                v = v2rho_2d(v)
                
                if 'smoothing' in locals():
                    
                    if 'Gaussian' in smoothing:
                        qparam = ndimage.gaussian_filter(qparam, smooth_fac, 0)
                    
                    elif 'Hanning' in smoothing:
                        # smooth_fac passes of 2d Hanning filter
                        #han_time = time.time()
                        qparam = func_hann2d_fast(qparam, smooth_fac)
                        #print 'hanning', str(time.time() - han_time), ' seconds!'
                        xi = func_hann2d_fast(xi, smooth_fac)
                    
                # Set Q over land to zero
                qparam *= sla_grd.mask
                #qparam = np.ma.masked_where(grdmask == False, qparam)
                xi *= sla_grd.mask
                xi = np.ma.masked_where(sla_grd.mask == False,
                                    xi / grd.f()[j0:j1,i0:i1])
                xicopy = np.ma.copy(xi)
            
            
            elif 'sla' in diag_type:
                
                sla = sla_grd.get_AVISO_data(AVISO_file)
                
                if isinstance(smoothing, str):
                    
                    if 'Gaussian' in smoothing:
                        
                        if 'first_record' not in locals():
                            print '------ applying Gaussian high-pass filter'
                            
                        # Set landpoints to zero
                        np.place(sla, sla_grd.mask == False, 0.)
                        np.place(sla, sla.data == sla_grd.fillval, 0.)
                        # High pass filter, see
                        # http://stackoverflow.com/questions/6094957/high-pass-filter-for-image-processing-in-python-by-using-scipy-numpy
                        #print 'start smoothing'
                        sla -= ndimage.gaussian_filter(sla, [mres, zres])
                        #print 'end smoothing'
                    elif 'Hanning' in smoothing:
                        
                        print '------ applying %s passes of Hanning filter' %smooth_fac
                        # Do smooth_fac passes of 2d Hanning filter
                        sla = func_hann2d_fast(sla, smooth_fac)
                
                
                # Expand the landmask
                sla = np.ma.masked_where(sla_grd.mask == False, sla)
                
                # Get timing
                #ymd = rtime.split(' ')[0].split('-')
                #yr, mo, da = np.int(ymd[0]), np.int(ymd[1]), np.int(ymd[2])
                #rtime = dt.date2num(dt.datetime.datetime(yr, mo, da))
                try:
                    thedate = dt.num2date(rtime)[0]
                except:
                    thedate = dt.num2date(rtime)
                yr = thedate.year
                mo = thedate.month
                da = thedate.day
                
                #if sla.mask.size > 1:
                    #umask, vmask, junk = sla_grd.uvpmask(-sla.mask)
                #else:
                    #umask, vmask, junk = sla_grd.uvpmask(np.zeros_like(sla))
                
                # Multiply by 0.01 for m/s
                u, v = sla_grd.getSurfGeostrVel(sla * 0.01)
                
                # Remove padded boundary
                sla = sla[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1]
                u = u[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1]
                v = v[sla_grd.jup0:sla_grd.jup1, sla_grd.iup0:sla_grd.iup1]

                A_eddy.sla = np.ma.copy(sla)
                C_eddy.sla = np.ma.copy(sla)
                A_eddy.slacopy = np.ma.copy(sla)
                C_eddy.slacopy = np.ma.copy(sla)
                

            # Get scalar speed
            Uspd = np.hypot(u, v)
            Uspd = np.ma.masked_where(sla_grd.mask[sla_grd.jup0:sla_grd.jup1,
                                                   sla_grd.iup0:sla_grd.iup1] == False, Uspd)
            
            A_eddy.Uspd = np.ma.copy(Uspd)
            C_eddy.Uspd = np.ma.copy(Uspd)
            
            
            # Get contours of Q/sla parameter
            if 'first_record' not in locals():
                print '------ getting SLA contours'
            plt.figure(99)
            if 'Q' in diag_type:
                CS = plt.contour(sla_grd.lon(),
                                 sla_grd.lat(), qparam, qparameter)
                # Get xi contour field at zero
                CSxi = plt.contour(sla_grd.lon(),
                                   sla_grd.lat(), xi, [0.])
                
            elif 'sla' in diag_type:
                A_CS = plt.contour(sla_grd.lon(),
                                   sla_grd.lat(), A_eddy.sla, slaparameter)
                # Note that CSc is for the cyclonics, slaparameter in reverse order
                C_CS = plt.contour(sla_grd.lon(),
                                   sla_grd.lat(), C_eddy.sla, slaparameter[::-1])
            else: Exception
            
            if True:
                plt.close(99)
            else:
                # Debug
                if 'Q' in diag_type:
                    plt.title('qparameter and xi')
                    plt.clabel(CS, np.array([qparameter.min(), qparameter.max()]))
                elif 'sla' in diag_type:
                    plt.title('slaparameter')
                    plt.clabel(A_CS, np.array([slaparameter.min(), slaparameter.max()]))
                plt.axis('image')
                plt.show()
            


            # Now we loop over the CS collection
            if 'Q' in diag_type:
                A_eddy, C_eddy = collection_loop(CS, sla_grd, rtime,
                                                 A_list_obj=A_eddy, C_list_obj=C_eddy,
                                                 xi=xi, CSxi=CSxi, verbose=verbose)
            
            elif 'sla' in diag_type:
                A_eddy.sign_type = 'Anticyclonic'
                A_eddy = collection_loop(A_CS, sla_grd, rtime,
                                         A_list_obj=A_eddy, C_list_obj=None,
                                         sign_type=A_eddy.sign_type, verbose=verbose)
                # Note that C_CS is reverse order
                C_eddy.sign_type='Cyclonic'
                C_eddy = collection_loop(C_CS, sla_grd, rtime,
                                         A_list_obj=None, C_list_obj=C_eddy,
                                         sign_type=C_eddy.sign_type, verbose=verbose)
            
            # Debug
            if 'fig250' in locals():
                
                plt.figure(250)
                tit = 'Y' + str(yr) + 'M' + str(mo) + 'D' + str(da)
                
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
            #print 'start A tracking'
            #tt = time.time()
            A_eddy = track_eddies(A_eddy, first_record)
            #print 'end A tracking in %s seconds\n' %(time.time() - tt)
            
            #print 'start C tracking'
            #tt = time.time()
            C_eddy = track_eddies(C_eddy, first_record)
            #print 'end C tracking in %s seconds\n' %(time.time() - tt)

            
            if anim_figs: # Make figures for animations
                
                tit = 'Y' + str(yr) + 'M' + str(mo) + 'D' + str(da)
                
                #if 'anim_fig' in locals():
                    ## Wait if there is a still-active anim_fig thread
                    #anim_fig.join()
                
                if 'Q' in diag_type:
                    anim_fig = threading.Thread(name='anim_figure', target=anim_figure,
                             args=(33, M, pMx, pMy, xicopy, cmap, rtime, diag_type, Mx, My, 
                                   xi.copy(), qparam.copy(), qparameter, A_eddy, C_eddy,
                                   savedir, plt, 'Q ' + tit))
                elif 'sla' in diag_type:
                    '''anim_fig = threading.Thread(name='anim_figure', target=anim_figure,
                             args=(33, M, pMx, pMy, slacopy, plt.cm.RdBu_r, rtime, diag_type, Mx, My, 
                                   slacopy, slacopy, slaparameter, A_eddy, C_eddy,
                                   savedir, plt, 'SLA ' + tit))'''
                    fignum = 31
                    #print 'figure saving'
                    #tt = time.time()
                    anim_figure(A_eddy, C_eddy, Mx, My, pMx, pMy, plt.cm.RdBu_r, rtime, diag_type, 
                                savedir, 'SLA ' + tit, fignum)
                    #print 'figure saving done in %s seconds\n' %(time.time() - tt)
                #anim_fig.start()
                
                                    
            # Save inactive eddies to nc file
            # IMPORTANT: this must be done at every time step!!
            #saving_start_time = time.time()
            if not first_record:
                if verbose:
                    print '--- saving to nc', A_savefile
                    print '--- saving to nc', C_savefile
                    print '+++'
                if chelton_style_nc: # Recommended
                    A_eddy.write2chelton_nc(A_savefile, rtime)
                    C_eddy.write2chelton_nc(C_savefile, rtime)
                else:
                    A_eddy.write2nc(A_savefile, rtime)
                    C_eddy.write2nc(C_savefile, rtime)
                    
            #print 'Saving the eddies', time.time() - saving_start_time, 'seconds'
                
            # Running time for a single monthly file
            #print '--- duration', str((time.time() - file_time) / 60.), 'minutes'
        
        if str(date_end) in AVISO_file:
            active = False
    
    # Total running time    
    print 'Duration', str((time.time() - start_time) / 3600.), 'hours!'

    print '\nOutputs saved to', savedir
