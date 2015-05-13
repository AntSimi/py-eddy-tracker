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

Copyright (c) 2014-2015 by Evan Mason
Email: emason@imedea.uib-csic.es
===============================================================================

make_eddy_track_AVISO.py

Version 2.0.1


Scroll down to line ~640 to get started
===============================================================================
"""
from matplotlib import use as mpl_use
mpl_use('Agg')
import sys
import glob as glob
from py_eddy_tracker_classes import plt, np, dt, Dataset, time, \
                                    datestr2datetime, gaussian_resolution, \
                                    get_cax, collection_loop, track_eddies, \
                                    anim_figure, pcol_2dxy
from py_eddy_tracker_property_classes import SwirlSpeed
import make_eddy_tracker_list_obj as eddy_tracker
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
import scipy.spatial as spatial
from dateutil import parser
from mpl_toolkits.basemap import Basemap
import yaml
from datetime import datetime
import cPickle as pickle

import global_tracking as gbtk


def timeit(method):
    """
    Decorator to time a function
    """
    def timed(*args, **kw):
        ts = datetime.now()
        result = method(*args, **kw)
        te = datetime.now()
        print '-----> %s : %s sec' % (method.__name__, te - ts)
        return result
    return timed


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
        """
        Set some constants

          *ZERO_CROSSING*:
            Boolean, *True* if THE_DOMAIN crosses 0 degree meridian
        """
        self.GRAVITY = 9.81
        self.EARTH_RADIUS = 6371315.0
        self.ZERO_CROSSING = False
        self.i0, self.i1 = 0, None
        self.j0, self.j1 = 0, None
        self.ip0, self.ip1 = 0, None
        self.jp0, self.jp1 = 0, None
        self.iup0, self.iup1 = 0, None
        self.jup0, self.jup1 = 0, None
        self.eke = None
        self.u, self.upad = None, None
        self.v, self.vpad = None, None
        self.M = None
        self.Mx = None
        self.My = None
        self._lon = None
        self._lat = None
        self._f = None
        self._gof = None
        self._pm = None
        self._pn = None
        self._dx = None
        self._dy = None
        self._umask = None
        self._vmask = None

    def read_nc(self, varfile, varname, indices="[:]"):
        """
        Read data from nectdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          indices : string of index ranges, eg. '[0,:,0]'
        """
        with Dataset(varfile) as nc:
            try:
                var = eval(''.join(("nc.variables[varname]", indices)))
            except Exception:
                return None
            else:
                return var

    def read_nc_att(self, varfile, varname, att):
        """
        Read data attribute from nectdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          att : string of attribute, eg. 'valid_range'
        """
        with Dataset(varfile) as nc:
            return eval(''.join(("nc.variables[varname].", att)))

    def set_initial_indices(self):
        """
        Set indices for desired domain
        """
        LONMIN, LONMAX = self.LONMIN, self.LONMAX
        LATMIN, LATMAX = self.LATMIN, self.LATMAX
        
        print '--- Setting initial indices to *%s* domain' % self.THE_DOMAIN
        print '------ LONMIN = %s, LONMAX = %s, LATMIN = %s, LATMAX = %s' % (
                                           LONMIN, LONMAX, LATMIN, LATMAX)
        LATMIN_OFFSET = LATMIN + (0.5 * (LATMAX - LATMIN))
        self.i0, _ = self.nearest_point(LONMIN, LATMIN_OFFSET)
        self.i1, _ = self.nearest_point(LONMAX, LATMIN_OFFSET)
        LONMIN_OFFSET = LONMIN + (0.5 * (LONMAX - LONMIN))
        _, self.j0 = self.nearest_point(LONMIN_OFFSET, LATMIN)
        _, self.j1 = self.nearest_point(LONMIN_OFFSET, LATMAX)

        def kdt(lon, lat, limits, k=4):
            """
            Make kde tree for indices if domain crosses zero meridian
            """
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
                """
                Used for a zero crossing, e.g., across Agulhas region
                """
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
        """
        Set start and end indices for temporary padding and later unpadding
        around 2d variables.
        Padded matrices are needed only for geostrophic velocity computation.
        """
        print '--- Setting padding indices with PAD = %s' % pad

        self.pad = pad

        def get_str(thestr, pad):
            """
            Get start indices for pad
            Returns:
              pad_str   - index to add pad
              unpad_str - index for later unpadding
            """
            pad_str = np.array([0, thestr - pad]).max()
            if pad > 0:
                unpad_str = np.array([0, np.diff([pad_str, thestr])]).max()
                return pad_str, unpad_str
            else:
                unpad_str = np.array([0, np.diff([pad_str, thestr])]).min()
                return pad_str, -1 * unpad_str

        def get_end(theend, shape, pad):
            """
            Get end indices for pad
            Returns:
              pad_end   - index to add pad
              unpad_end - index for later unpadding
            """
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
        """
        TO DO: change to use f2py version
        Haversine formula to calculate distance between two lon/lat points
        Uses mean earth radius in metres (from ROMS scalars.h) = 6371315.0
        Input:
          lon1, lat1, lon2, lat2
        Return:
          distance (m)
        """
        #print 'ssssssssssssssss'
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
        return 6371315.0 * c  # Return the distance

    def nearest_point(self, lon, lat):
        """
        Get indices to a point (lon, lat) in the grid
        """
        i, j = eddy_tracker.nearest(lon, lat, self._lon, self._lat,
                                    self._lon.shape)
        return i, j

    def half_interp(self, h_one, h_two):
        """
        Speed up frequent operations of type 0.5 * (arr[:-1] + arr[1:])
        """
        h_one += h_two
        h_one *= 0.5
        return h_one

    def get_AVISO_f_pm_pn(self):
        """
        Padded matrices are used here because Coriolis (f), pm and pn
        are needed for the geostrophic velocity computation in
        method getSurfGeostrVel()
        NOTE: this should serve for ROMS too
        """
        print '--- Computing Coriolis (f), dx (pm), dy (pn) for padded grid'
        # Get GRAVITY / Coriolis
        self._gof = np.sin(np.deg2rad(self.latpad()))
        self._gof *= 4.
        self._gof *= np.pi
        self._gof /= 86400.
        self._f = self._gof.copy()
        self._gof = self.GRAVITY / self._gof

        lonu = self.half_interp(self.lonpad()[:, :-1], self.lonpad()[:, 1:])
        latu = self.half_interp(self.latpad()[:, :-1], self.latpad()[:, 1:])
        lonv = self.half_interp(self.lonpad()[:-1], self.lonpad()[1:])
        latv = self.half_interp(self.latpad()[:-1], self.latpad()[1:])

        # Get pm and pn
        pm = np.zeros_like(self.lonpad())
        pm[:, 1:-1] = self.haversine_dist(lonu[:, :-1], latu[:, :-1],
                                          lonu[:, 1:],  latu[:, 1:])
        pm[:, 0] = pm[:, 1]
        pm[:, -1] = pm[:, -2]
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
        """
        Convert a 2D field at u points to a field at rho points
        """
        def uu2ur(uu_in, mp, lp):
            l = lp - 1
            lm = l - 1
            u_out = np.zeros((mp, lp))
            u_out[:, 1:l] = self.half_interp(uu_in[:, 0:lm], uu_in[:, 1:l])
            u_out[:, 0] = u_out[:, 1]
            u_out[:, l] = u_out[:, lm]
            return (u_out.squeeze())
        mshp, lshp = uu_in.shape
        return uu2ur(uu_in, mshp, lshp + 1)

    def v2rho_2d(self, vv_in):
        # Convert a 2D field at v points to a field at rho points
        def vv2vr(vv_in, mp, lp):
            m = mp - 1
            mm = m - 1
            v_out = np.zeros((mp, lp))
            v_out[1:m] = self.half_interp(vv_in[:mm], vv_in[1:m])
            v_out[0] = v_out[1]
            v_out[m] = v_out[mm]
            return (v_out.squeeze())
        mshp, lshp = vv_in.shape
        return vv2vr(vv_in, mshp + 1, lshp)

    def rho2u_2d(self, rho_in):
        """
        Convert a 2D field at rho points to a field at u points
        """
        def _r2u(rho_in, lp):
            u_out = rho_in[:, :lp - 1]
            u_out += rho_in[:, 1:lp]
            u_out *= 0.5
            return u_out.squeeze()
        assert rho_in.ndim == 2, 'rho_in must be 2d'
        mshp, lshp = rho_in.shape
        return _r2u(rho_in, rho_in.shape[1])

    def rho2v_2d(self, rho_in):
        """
        Convert a 2D field at rho points to a field at v points
        """
        def _r2v(rho_in, mp):
            v_out = rho_in[:mp - 1]
            v_out += rho_in[1:mp]
            v_out *= 0.5
            return v_out.squeeze()
        assert rho_in.ndim == 2, 'rho_in must be 2d'
        return _r2v(rho_in, rho_in.shape[0])

    def uvmask(self):
        """
        Get mask at U and V points
        """
        #print '--- Computing umask and vmask for padded grid'
        mp, lp = self.mask.shape
        m = mp - 1
        l = lp - 1
        self._umask = self.mask[:, :l] * self.mask[:, 1:lp]
        self._vmask = self.mask[:m] * self.mask[1:mp]
        return self

    def set_basemap(self, with_pad=True):
        """
        Use Basemap to make a landmask
        Format is 1 == ocean, 0 == land
        """
        print '--- Computing Basemap'
        # Create Basemap instance for Mercator projection.
        self.M = Basemap(
            projection='merc',
            llcrnrlon=self.LONMIN - 1, urcrnrlon=self.LONMAX + 1,
            llcrnrlat=self.LATMIN - 1, urcrnrlat=self.LATMAX + 1,
            lat_ts=0.5 * (self.LATMIN + self.LATMAX), resolution='h')

        if with_pad:
            x, y = self.M(self.lonpad(), self.latpad())
        else:
            x, y = self.M(self.lon(), self.lat())
        self.Mx, self.My = x, y
        return self

    #@timeit
    def set_geostrophic_velocity(self, zeta):
        """
        Set u and v geostrophic velocity at
        surface from variables f, zeta, pm, pn...
        Note: output at rho points
        """
        gof = self.gof().view()

        vmask = self.vmask().view()
        zeta1, zeta2 = zeta.data[1:].view(), zeta.data[:-1].view()
        pn1, pn2 = self.pn()[1:].view(), self.pn()[:-1].view()
        self.upad[:] = self.v2rho_2d(vmask * (zeta1 - zeta2) *
                                     0.5 * (pn1 + pn2))
        self.upad *= -gof

        umask = self.umask().view()
        zeta1, zeta2 = zeta.data[:, 1:].view(), zeta.data[:, :-1].view()
        pm1, pm2 = self.pm()[:, 1:].view(), self.pm()[:, :-1].view()
        self.vpad[:] = self.u2rho_2d(umask * (zeta1 - zeta2) *
                                     0.5 * (pm1 + pm2))
        self.vpad *= gof
        return self

    #@timeit
    def set_u_v_eke(self, pad=2):
        """
        Set empty arrays for u, v, upad, vpad
        """
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
        """
        Set EKE; also sets u/v from upad/vpad
        """
        self.u[:] = self.upad[self.jup0:self.jup1, self.iup0:self.iup1]
        self.v[:] = self.vpad[self.jup0:self.jup1, self.iup0:self.iup1]
        self.eke[:] = self.u**2 + self.v**2
        self.eke *= 0.5
        return self
    
    def set_interp_coeffs(self, sla, uspd):
        """
        Won't work for rotated grid
        """
        if 'AVISO' in self.PRODUCT:
            self.sla_coeffs = interpolate.RectBivariateSpline(
                self.lat()[:, 0], self.lon()[0], sla, kx=1, ky=1)
            self.uspd_coeffs = interpolate.RectBivariateSpline(
                self.lat()[:, 0], self.lon()[0], uspd, kx=1, ky=1)
        elif 'ROMS' in self.PRODUCT:
            points = np.array([self.lon().ravel(), self.lat().ravel()]).T
            self.sla_coeffs = interpolate.CloughTocher2DInterpolator(
                points, sla.ravel())
            self.uspd_coeffs = interpolate.CloughTocher2DInterpolator(
                points, uspd.ravel())
        return self


class AvisoGrid (PyEddyTracker):
    """
    Class to satisfy the need of the eddy tracker
    to have a grid class
    """
    def __init__(self, AVISO_FILE, THE_DOMAIN, PRODUCT,
                 LONMIN, LONMAX, LATMIN, LATMAX, with_pad=True):
        """
        Initialise the grid object
        """
        super(AvisoGrid, self).__init__()
        print '\nInitialising the *AVISO_grid*'
        self.THE_DOMAIN = THE_DOMAIN
        if 'AVISO' in PRODUCT:
            self.PRODUCT = 'AVISO'
        else:
            self.PRODUCT = PRODUCT
        self.LONMIN = LONMIN
        self.LONMAX = LONMAX
        self.LATMIN = LATMIN
        self.LATMAX = LATMAX

        try:  # new AVISO (2014)
            self._lon = self.read_nc(AVISO_FILE, 'lon')
            self._lat = self.read_nc(AVISO_FILE, 'lat')
            self.FILLVAL = self.read_nc_att(AVISO_FILE, 'sla', '_FillValue')
            base_date = self.read_nc_att(AVISO_FILE, 'time', 'units')
            self.base_date = dt.date2num(
                parser.parse(base_date.split(' ')[2:4][0]))

        except Exception:  # old AVISO
            self._lon = self.read_nc(AVISO_FILE, 'NbLongitudes')
            self._lat = self.read_nc(AVISO_FILE, 'NbLatitudes')
            self.FILLVAL = self.read_nc_att(AVISO_FILE,
                                            'Grid_0001', '_FillValue')

        if LONMIN < 0 and LONMAX <= 0:
            self._lon -= 360.
        self._lon, self._lat = np.meshgrid(self._lon, self._lat)
        self._angle = np.zeros_like(self._lon)
        # ZERO_CROSSING, used for handling a longitude range that
        # crosses zero degree meridian
        if LONMIN < 0 and LONMAX >= 0:
            self.ZERO_CROSSING = True

        self.sla_coeffs = None
        self.uspd_coeffs = None

        self.set_initial_indices()
        self.set_index_padding()
        self.set_basemap(with_pad=with_pad)
        self.get_AVISO_f_pm_pn()
        self.set_u_v_eke()
        self.shape = self.lon().shape
        #pad2 = 2 * self.pad
        #self.shape = (self.f().shape[0] - pad2, self.f().shape[1] - pad2)

    def __getstate__(self):
        """
        Remove references to unwanted attributes in self.
        This reduces the size of saved cPickle objects.
        """
        #print '--- removing unwanted attributes'
        pops = ('Mx', 'My', '_f', '_angle', '_dx', '_dy', '_gof', '_lon',
                '_lat', '_pm', '_pn', '_umask', '_vmask', 'eke', 'u', 'v',
                'mask', 'pad', 'vpad', 'upad')
        result = self.__dict__.copy()
        for pop in pops:
            result.pop(pop)
        return result

    #@timeit
    def get_AVISO_data(self, AVISO_FILE):
        """
        Read nc data from AVISO file
        """
        if self.ZERO_CROSSING:

            try:  # new AVISO (2014)
                ssh1 = self.read_nc(
                    AVISO_FILE, 'sla',
                    indices='[:, self.jp0:self.jp1, :self.ip0]')
                ssh0 = self.read_nc(
                    AVISO_FILE, 'sla',
                    indices='[:, self.jp0:self.jp1, self.ip1:]')
                ssh0, ssh1 = ssh0.squeeze(), ssh1.squeeze()
                ssh0 *= 100.  # m to cm
                ssh1 *= 100.  # m to cm

            except Exception:  # old AVISO
                ssh1 = self.read_nc(AVISO_FILE, 'Grid_0001',
                                    indices='[:self.ip0, self.jp0:self.jp1]').T
                ssh0 = self.read_nc(AVISO_FILE, 'Grid_0001',
                                    indices='[self.ip1:, self.jp0:self.jp1]').T

            zeta = np.ma.concatenate((ssh0, ssh1), axis=1)

        else:

            try:  # new AVISO (2014)
                zeta = self.read_nc(
                    AVISO_FILE, 'sla',
                    indices='[:, self.jp0:self.jp1, self.ip0:self.ip1]')
                zeta = zeta.squeeze()
                zeta *= 100.  # m to cm

            except Exception:  # old AVISO
                zeta = self.read_nc(
                    AVISO_FILE, 'Grid_0001',
                    indices='[self.ip0:self.ip1, self.jp0:self.jp1]').T

        try:  # Extrapolate over land points with fillmask
            zeta = fillmask(zeta, self.mask == 1)
            #zeta = fillmask(zeta, 1 + (-1 * zeta.mask))
        except Exception:  # In case no landpoints
            zeta = np.ma.masked_array(zeta)
        return zeta.astype(np.float64)

    def set_mask(self, sla):
        """
        """
        if sla.mask.size == 1:  # all sea points
            self.mask = np.ones_like(sla.data)

        else:
            self.mask = np.logical_not(sla.mask).astype(np.int)

            if 'Global' in self.THE_DOMAIN:

                # Close Drake Passage
                minus70 = np.argmin(np.abs(self.lonpad()[0] + 70))
                self.mask[:125, minus70] = 0

                # DT10 mask is open around Panama, so close it...
                if 'AVISO_DT10' in self.PRODUCT:

                    mask = 0
                    self.mask[348, 92:110] = mask
                    self.mask[348:356, 92] = mask
                    self.mask[355, 71:92] = mask
                    self.mask[355:363, 71] = mask
                    self.mask[362, 66:71] = mask
                    self.mask[362:380, 66] = mask
                    self.mask[380, 47:67] = mask
                    self.mask[380:389, 47] = mask
                    self.mask[388, 13:47] = mask
                    self.mask[388:393, 13] = mask
                    self.mask[392, :13] = mask
                    ind = 4 * 360
                    self.mask[348, 92 + ind:110 + ind] = mask
                    self.mask[348:356, 92 + ind] = mask
                    self.mask[355, 71 + ind:92 + ind] = mask
                    self.mask[355:363, 71 + ind] = mask
                    self.mask[362, 66 + ind:71 + ind] = mask
                    self.mask[362:380, 66 + ind] = mask
                    self.mask[380, 47 + ind:67 + ind] = mask
                    self.mask[380:389, 47 + ind] = mask
                    self.mask[388, 13 + ind:47 + ind] = mask
                    self.mask[388:393, 13 + ind] = mask
                    self.mask[392,  ind:13 + ind] = mask

                # Mask all unwanted regions (Caspian Sea, etc)
                labels = ndimage.label(self.mask)[0]

                self.labels = labels
                # Set to known sea point
                plus200 = np.argmin(np.abs(self.lonpad()[0] - 200))
                plus9 = np.argmin(np.abs(self.latpad()[:, 0] - 9))
                sea_label = labels[plus9, plus200]
                np.place(self.mask, labels != sea_label, 0)
        return self

    def fillmask(self, x, mask):
        """
        Fill missing values in an array with an average of nearest
        neighbours
        From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
        """
        assert x.ndim == 2, 'x must be a 2D array.'
        fill_value = 9999.99
        x[mask == 0] = fill_value

        # Create (i, j) point arrays for good and bad data.
        # Bad data are marked by the fill_value, good data elsewhere.
        igood = np.vstack(np.where(x != fill_value)).T
        ibad = np.vstack(np.where(x == fill_value)).T

        # Create a tree for the bad points, the points to be filled
        tree = spatial.cKDTree(igood)

        # Get the four closest points to the bad points
        # here, distance is squared
        dist, iquery = tree.query(ibad, k=4, p=2)

        # Create a normalised weight, the nearest points are weighted as 1.
        #   Points greater than one are then set to zero
        weight = dist / (dist.min(axis=1)[:, np.newaxis])
        weight *= np.ones_like(dist)
        np.place(weight, weight > 1., 0.)

        # Multiply the queried good points by the weight, selecting only the
        # nearest points. Divide by the number of nearest points to get average
        xfill = weight * x[igood[:, 0][iquery], igood[:, 1][iquery]]
        xfill = (xfill / weight.sum(axis=1)[:, np.newaxis]).sum(axis=1)

        # Place average of nearest good points, xfill, into bad point locations
        x[ibad[:, 0], ibad[:, 1]] = xfill
        return x

    def lon(self):
        if self.ZERO_CROSSING:
            # TO DO: These concatenations are possibly expensive, they
            # shouldn't need to happen with every call to self.lon()
            lon0 = self._lon[self.j0:self.j1, self.i1:]
            lon1 = self._lon[self.j0:self.j1, :self.i0]
            return np.concatenate((lon0 - 360., lon1), axis=1)
        else:
            return self._lon[self.j0:self.j1, self.i0:self.i1]

    def lat(self):
        if self.ZERO_CROSSING:
            lat0 = self._lat[self.j0:self.j1, self.i1:]
            lat1 = self._lat[self.j0:self.j1, :self.i0]
            return np.concatenate((lat0, lat1), axis=1)
        else:
            return self._lat[self.j0:self.j1, self.i0:self.i1]

    def lonpad(self):
        if self.ZERO_CROSSING:
            lon0 = self._lon[self.jp0:self.jp1, self.ip1:]
            lon1 = self._lon[self.jp0:self.jp1, :self.ip0]
            return np.concatenate((lon0 - 360., lon1), axis=1)
        else:
            return self._lon[self.jp0:self.jp1, self.ip0:self.ip1]

    def latpad(self):
        if self.ZERO_CROSSING:
            lat0 = self._lat[self.jp0:self.jp1, self.ip1:]
            lat1 = self._lat[self.jp0:self.jp1, :self.ip0]
            return np.concatenate((lat0, lat1), axis=1)
        else:
            return self._lat[self.jp0:self.jp1, self.ip0:self.ip1]

    def angle(self):
        return self._angle[self.j0:self.j1, self.i0:self.i1]

    def umask(self):  # Mask at U points
        return self._umask

    def vmask(self):  # Mask at V points
        return self._vmask

    def f(self):  # Coriolis
        return self._f

    def gof(self):  # Gravity / Coriolis
        return self._gof

    def dx(self):  # Grid spacing along X direction
        return self._dx

    def dy(self):  # Grid spacing along Y direction
        return self._dy

    def pm(self):  # Reciprocal of dx
        return self._pm

    def pn(self):  # Reciprocal of dy
        return self._pn

    def get_resolution(self):
        return np.sqrt(np.diff(self.lon()[1:], axis=1) *
                       np.diff(self.lat()[:, 1:], axis=0)).mean()

    def boundary(self):
        """
        Return lon, lat of perimeter around a ROMS grid
        Input:
          indices to get boundary of specified subgrid
        Returns:
          lon/lat boundary points
        """
        lon = np.r_[(self.lon()[:, 0], self.lon()[-1],
                     self.lon()[::-1, -1], self.lon()[0, ::-1])]
        lat = np.r_[(self.lat()[:, 0], self.lat()[-1],
                     self.lat()[::-1, -1], self.lat()[0, ::-1])]
        return lon, lat

    def brypath(self, imin=0, imax=-1, jmin=0, jmax=-1):
        """
        Return Path object of perimeter around a ROMS grid
        Indices to get boundary of specified subgrid
        """
        lon, lat = self.boundary()
        brypath = np.array([lon, lat]).T
        return path.Path(brypath)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if __name__ == '__main__':

    # Run using:
    # python make_eddy_track_AVISO.py eddy_tracker_configuration.yaml
    try:
        YAML_FILE = sys.argv[1]
    except Exception:
        print "To run use 'python make_eddy_track_AVISO.py eddy_tracker_configuration.yaml'"

    print "\nLaunching with yaml file: %s" % YAML_FILE
    # Choose a yaml configuration file
    #YAML_FILE = 'eddy_tracker_configuration.yaml'
    #YAML_FILE = 'BlackSea.yaml'

    # --------------------------------------------------------------------------

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

    config['THE_DOMAIN'] = config['DOMAIN']['THE_DOMAIN']

    # It is not recommended to change values given below
    # for 'Global', 'BlackSea' or 'MedSea'...
    if 'Global' in config['THE_DOMAIN']:
        config['LONMIN'] = -100.
        config['LONMAX'] = 290.
        config['LATMIN'] = -80.
        config['LATMAX'] = 80.

    elif 'Regional' in config['THE_DOMAIN']:
        config['LONMIN'] = config['DOMAIN']['LONMIN']
        config['LONMAX'] = config['DOMAIN']['LONMAX']
        config['LATMIN'] = config['DOMAIN']['LATMIN']
        config['LATMAX'] = config['DOMAIN']['LATMAX']

    DATE_STR = config['DATE_STR'] = config['DOMAIN']['DATE_STR']
    DATE_END = config['DATE_END'] = config['DOMAIN']['DATE_END']

    AVISO_DT14 = config['AVISO']['AVISO_DT14']
    if AVISO_DT14:
        PRODUCT = 'AVISO_DT14'
        AVISO_DT14_SUBSAMP = config['AVISO']['AVISO_DT14_SUBSAMP']
        if AVISO_DT14_SUBSAMP:
            DAYS_BTWN_RECORDS = config['AVISO']['DAYS_BTWN_RECORDS']
        else:
            DAYS_BTWN_RECORDS = 1.
    else:
        PRODUCT = 'AVISO_DT10'
        DAYS_BTWN_RECORDS = 7.  # old seven day AVISO

    AVISO_FILES = config['AVISO']['AVISO_FILES']

    #TRACK_DURATION_MIN = config['TRACK_DURATION_MIN']

    if 'SLA' in DIAGNOSTIC_TYPE:
        MAX_SLA = config['CONTOUR_PARAMETER']['CONTOUR_PARAMETER_SLA']['MAX_SLA']
        INTERVAL = config['CONTOUR_PARAMETER']['CONTOUR_PARAMETER_SLA']['INTERVAL']
        config['CONTOUR_PARAMETER'] = np.arange(-MAX_SLA, MAX_SLA + INTERVAL,
                                                INTERVAL)
        config['SHAPE_ERROR'] = np.full(config['CONTOUR_PARAMETER'].size,
                                        config['SHAPE_ERROR'])

    elif 'Q' in DIAGNOSTIC_TYPE:
        MAX_Q = config['CONTOUR_PARAMETER']['CONTOUR_PARAMETER_Q']['MAX_Q']
        NUM_LEVS = config['CONTOUR_PARAMETER']['CONTOUR_PARAMETER_Q']['NUM_LEVS']
        config['CONTOUR_PARAMETER'] = np.linspace(0, MAX_Q, NUM_LEVS)[::-1]
    else:
        Exception

    #JDAY_REFERENCE = config['JDAY_REFERENCE']

    #RADMIN = config['RADMIN']
    #RADMAX = config['RADMAX']

    if 'SLA' in DIAGNOSTIC_TYPE:
        #AMPMIN = config['AMPMIN']
        #AMPMAX = config['AMPMAX']
        pass
    elif 'Q' in DIAGNOSTIC_TYPE:
        AMPMIN = 0.02  # max(abs(xi/f)) within the eddy
        AMPMAX = 100.
    else:
        Exception

    SAVE_FIGURES = config['SAVE_FIGURES']

    SMOOTHING = config['SMOOTHING']
    if SMOOTHING:
        if 'SLA' in DIAGNOSTIC_TYPE:
            ZWL = np.atleast_1d(config['SMOOTHING_SLA']['ZWL'])
            MWL = np.atleast_1d(config['SMOOTHING_SLA']['MWL'])
        elif 'Q' in DIAGNOSTIC_TYPE:
            SMOOTH_FAC = config['SMOOTHING_Q']['SMOOTH_FAC']
        else:
            Exception
        SMOOTHING_TYPE = config['SMOOTHING_SLA']['TYPE']
    
    if 'Q' in DIAGNOSTIC_TYPE:
        AMP0 = 0.02  # vort/f
    elif 'SLA' in DIAGNOSTIC_TYPE:
        AMP0 = config['AMP0']
    TEMP0 = config['TEMP0']
    SALT0 = config['SALT0']

    #EVOLVE_AMP_MIN = config['EVOLVE_AMP_MIN']
    #EVOLVE_AMP_MAX = config['EVOLVE_AMP_MAX']
    #EVOLVE_AREA_MIN = config['EVOLVE_AREA_MIN']
    #EVOLVE_AREA_MAX = config['EVOLVE_AREA_MAX']

    CMAP = plt.cm.RdBu

    # End user configuration setup options
    # --------------------------------------------------------------------------

    assert DATE_STR < DATE_END, 'DATE_END must be larger than DATE_STR'
    assert DIAGNOSTIC_TYPE in ('Q', 'SLA'), 'Undefined DIAGNOSTIC_TYPE'

    thestartdate = dt.date2num(datestr2datetime(str(DATE_STR)))
    theenddate = dt.date2num(datestr2datetime(str(DATE_END)))

    # Get complete AVISO file list
    AVISO_FILES = sorted(glob.glob(DATA_DIR + AVISO_FILES))

    # For subsampling to get identical list as old_AVISO use:
    # AVISO_FILES = AVISO_FILES[5:-5:7]
    if AVISO_DT14 and AVISO_DT14_SUBSAMP:
        AVISO_FILES = AVISO_FILES[5:-5:np.int(DAYS_BTWN_RECORDS)]

    # Set up a grid object using first AVISO file in the list
    sla_grd = AvisoGrid(AVISO_FILES[0], config['THE_DOMAIN'], PRODUCT,
                        config['LONMIN'], config['LONMAX'],
                        config['LATMIN'], config['LATMAX'])
    
    # Set coordinates for figures
    Mx, My = sla_grd.M(sla_grd.lon(), sla_grd.lat())
    MMx, MMy = pcol_2dxy(Mx, My)

    # Instantiate search ellipse object
    search_ellipse = eddy_tracker.SearchEllipse(config['THE_DOMAIN'],
                                                sla_grd, DAYS_BTWN_RECORDS,
                                                RW_PATH)

    if 'Gaussian' in SMOOTHING_TYPE:
        # Get parameters for ndimage.gaussian_filter
        ZRES, MRES = gaussian_resolution(sla_grd.get_resolution(),
                                         ZWL, MWL)

    fig = plt.figure(1)

    if 'Q' in DIAGNOSTIC_TYPE:
        A_SAVEFILE = "".join([SAVE_DIR, 'eddy_tracks_Q_AVISO_anticyclonic.nc'])
        C_SAVEFILE = "".join([SAVE_DIR, 'eddy_tracks_Q_AVISO_cyclonic.nc'])

    elif 'SLA' in DIAGNOSTIC_TYPE:
        A_SAVEFILE = "".join([SAVE_DIR, 'eddy_tracks_SLA_AVISO_anticyclonic.nc'])
        C_SAVEFILE = "".join([SAVE_DIR, 'eddy_tracks_SLA_AVISO_cyclonic.nc'])

    # Initialise two eddy objects to hold data
    #kwargs = config
    A_eddy = eddy_tracker.TrackList('Anticyclonic', A_SAVEFILE,
                                    sla_grd, search_ellipse, **config)

    C_eddy = eddy_tracker.TrackList('Cyclonic', C_SAVEFILE,
                                    sla_grd, search_ellipse, **config)

    A_eddy.search_ellipse = search_ellipse
    C_eddy.search_ellipse = search_ellipse

    if 'sum_radii' in config['SEPARATION_METHOD']:
        A_eddy.SEP_DIST_FAC = SEP_DIST_FACTOR
        C_eddy.SEP_DIST_FACTOR = SEP_DIST_FACTOR

    # See Chelton section B2 (0.4 degree radius)
    # These should give 8 and 1000 for 0.25 deg resolution
    PIXMIN = np.round((np.pi * config['RADMIN']**2) /
                      sla_grd.get_resolution()**2)
    PIXMAX = np.round((np.pi * config['RADMAX']**2) /
                      sla_grd.get_resolution()**2)
    print '--- Pixel range = %s-%s' % (np.int(PIXMIN),
                                       np.int(PIXMAX))

    A_eddy.PIXEL_THRESHOLD = [PIXMIN, PIXMAX]
    C_eddy.PIXEL_THRESHOLD = [PIXMIN, PIXMAX]

    # Create nc files for saving of eddy tracks
    A_eddy.create_netcdf(DATA_DIR, A_SAVEFILE)
    C_eddy.create_netcdf(DATA_DIR, C_SAVEFILE)

    # Loop through the AVISO files...
    START = True

    START_TIME = time.time()

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

            #rec_START_TIME = time.time()
            print '--- AVISO_FILE:', AVISO_FILE

            # Holding variables
            A_eddy.reset_holding_variables()
            C_eddy.reset_holding_variables()

            sla = sla_grd.get_AVISO_data(AVISO_FILE)
            sla_grd.set_mask(sla).uvmask()

            if SMOOTHING:

                if 'Gaussian' in SMOOTHING_TYPE:

                    if 'first_record' not in locals():
                        print '------ applying Gaussian high-pass filter'
                    # Set landpoints to zero
                    np.place(sla, sla_grd.mask == 0, 0.)
                    np.place(sla, sla.data == sla_grd.FILLVAL, 0.)
                    # High pass filter, see
                    # http://stackoverflow.com/questions/6094957/high-pass-filter-for-image-processing-in-python-by-using-scipy-numpy
                    sla -= ndimage.gaussian_filter(sla, [MRES, ZRES])

                elif 'Hanning' in SMOOTHING_TYPE:

                    if 'first_record' not in locals():
                        print '------ applying %s passes of Hanning filter' \
                                                               % SMOOTH_FAC
                    # Do SMOOTH_FAC passes of 2d Hanning filter
                    sla = func_hann2d_fast(sla, SMOOTH_FAC)

                else:
                    Exception

            # Apply the landmask
            sla = np.ma.masked_where(sla_grd.mask == 0, sla)

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

                qparam = np.ma.multiply(-0.25, okubo)  # see Kurian etal 2011

                qparam = func_hann2d_fast(qparam, hanning_passes)

                # Set Q over land to zero
                qparam *= sla_grd.mask[sla_grd.jup0:sla_grd.jup1,
                                       sla_grd.iup0:sla_grd.iup1]
                #qparam = np.ma.masked_where(grdmask == False, qparam)
                xi *= sla_grd.mask[sla_grd.jup0:sla_grd.jup1,
                                   sla_grd.iup0:sla_grd.iup1]
                xi = np.ma.masked_where(sla_grd.mask[
                                        sla_grd.jup0:sla_grd.jup1,
                                        sla_grd.iup0:sla_grd.iup1] == 0,
                                   xi / sla_grd.f()[sla_grd.jup0:sla_grd.jup1,
                                                    sla_grd.iup0:sla_grd.iup1])

                xicopy = np.ma.copy(xi)

            elif 'SLA' in DIAGNOSTIC_TYPE:

                A_eddy.sla = sla.copy()
                C_eddy.sla = sla.copy()
                A_eddy.slacopy = sla.copy()
                C_eddy.slacopy = sla.copy()

            # Get scalar speed
            uspd = np.sqrt(sla_grd.u**2 + sla_grd.v**2)
            uspd = np.ma.masked_where(
                sla_grd.mask[sla_grd.jup0:sla_grd.jup1,
                             sla_grd.iup0:sla_grd.iup1] == 0,
                uspd)
            A_eddy.uspd = uspd.copy()
            C_eddy.uspd = uspd.copy()

            # Set interpolation coefficients
            sla_grd.set_interp_coeffs(sla, uspd)
            A_eddy.sla_coeffs = sla_grd.sla_coeffs
            A_eddy.uspd_coeffs = sla_grd.uspd_coeffs
            C_eddy.sla_coeffs = sla_grd.sla_coeffs
            C_eddy.uspd_coeffs = sla_grd.uspd_coeffs

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
                                  sla_grd.lat(),
                                  A_eddy.sla, A_eddy.CONTOUR_PARAMETER)
                # Note that C_CS is in reverse order
                C_CS = ax.contour(sla_grd.lon(),
                                  sla_grd.lat(),
                                  C_eddy.sla, C_eddy.CONTOUR_PARAMETER)

            else:
                Exception

            if True:  # clear the current axis
                ax.cla()
            else:  # draw debug figure
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

            # Set contour coordinates and indices for calculation of
            # speed-based radius
            A_eddy.swirl = SwirlSpeed(A_CS)
            C_eddy.swirl = SwirlSpeed(C_CS)

            # Now we loop over the CS collection
            if 'Q' in DIAGNOSTIC_TYPE:
                A_eddy, C_eddy = collection_loop(CS, sla_grd, rtime,
                                                 A_list_obj=A_eddy,
                                                 C_list_obj=C_eddy,
                                                 xi=xi, CSxi=CSxi,
                                                 VERBOSE=VERBOSE)

            elif 'SLA' in DIAGNOSTIC_TYPE:
                A_eddy = collection_loop(A_CS, sla_grd, rtime,
                                         A_list_obj=A_eddy, C_list_obj=None,
                                         sign_type=A_eddy.SIGN_TYPE,
                                         VERBOSE=A_eddy.VERBOSE)
                # Note that C_CS is reverse order
                C_eddy = collection_loop(C_CS, sla_grd, rtime,
                                         A_list_obj=None, C_list_obj=C_eddy,
                                         sign_type=C_eddy.SIGN_TYPE,
                                         VERBOSE=C_eddy.VERBOSE)

            ymd_str = ''.join((str(yr), str(mo).zfill(2), str(da).zfill(2)))

            # Test pickling
            with open("".join((SAVE_DIR, 'A_eddy_%s.pkl' % ymd_str)), 'wb') as save_pickle:
                pickle.dump(A_eddy, save_pickle, 2)

            with open("".join((SAVE_DIR, 'C_eddy_%s.pkl' % ymd_str)), 'wb') as save_pickle:
                pickle.dump(C_eddy, save_pickle, 2)
            
            print 'EXIT here'
            
            print '' 
            save2nc = gbtk.save_netcdf(A_eddy, ymd_str)
            save2nc.create_netcdf()
            save2nc.write_tracks()
            
            #exit()
            ## Unpickle
            #with open('C_eddy.pkl', 'rb') as load_pickle:
                #C_eddy = pickle.load(load_pickle)

            # Debug
            if 'fig250' in locals():

                plt.figure(250)
                tit = 'Y' + str(yr) + 'M' + str(mo).zfill(2) + \
                                      'D' + str(da).zfill(2)

                if 'Q' in DIAGNOSTIC_TYPE:
                    plt.title('Q ' + tit)
                    sla_grd.M.pcolormesh(Mx, My, xi, CMAP=CMAP)
                    sla_grd.M.contour(Mx, My, xi, [0.],
                                      colors='k', linewidths=0.5)
                    sla_grd.M.contour(Mx, My, qparam, qparameter,
                                      colors='w', linewidths=0.3)
                    sla_grd.M.contour(Mx, My, qparam, [qparameter[0]],
                                      colors='m', linewidths=0.25)

                elif 'SLA' in DIAGNOSTIC_TYPE:
                    plt.title('sla ' + tit)
                    sla_grd.M.pcolormesh(Mx, My, sla, CMAP=CMAP)
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

            if START:
                first_record = True
                # Set old variables equal to new variables
                A_eddy.set_old_variables()
                C_eddy.set_old_variables()
                START = False
                print '------ tracking eddies'
            else:
                first_record = False

            # Track the eddies
            A_eddy = track_eddies(A_eddy, first_record)
            C_eddy = track_eddies(C_eddy, first_record)

            if SAVE_FIGURES:  # Make figures for animations

                if 'Q' in DIAGNOSTIC_TYPE:
                    anim_figure(A_eddy, C_eddy, Mx, My, plt.cm.RdBu_r, rtime,
                                DIAGNOSTIC_TYPE, SAVE_DIR,
                                'Q-parameter ' + ymd_str, animax, animax_cbar,
                                qparam=qparam, qparameter=qparameter,
                                xi=xi, xicopy=xicopy)

                elif 'SLA' in DIAGNOSTIC_TYPE:
                    anim_figure(A_eddy, C_eddy, Mx, My, MMx, MMy, plt.cm.RdBu_r, rtime,
                                DIAGNOSTIC_TYPE, SAVE_DIR, 'SLA ' + ymd_str,
                                animax, animax_cbar)

            # Save inactive eddies to nc file
            if not first_record:

                if A_eddy.VERBOSE:
                    print '--- saving to nc', A_eddy.SAVE_DIR
                    print '--- saving to nc', C_eddy.SAVE_DIR
                    print '+++'

                A_eddy.write2netcdf(rtime)
                C_eddy.write2netcdf(rtime)

        if str(DATE_END) in AVISO_FILE:
            active = False

    # Total running time
    print 'Duration', str((time.time() - START_TIME) / 3600.), 'hours!'

    print '\nOutputs saved to', SAVE_DIR
