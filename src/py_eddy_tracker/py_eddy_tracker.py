# -*- coding: utf-8 -*-
from netCDF4 import Dataset
from scipy import interpolate
from scipy import spatial
from pyproj import Proj
import numpy as np
import logging
from .make_eddy_tracker_list_obj import nearest


class PyEddyTracker(object):
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

    GRAVITY = 9.81
    EARTH_RADIUS = 6371315.0

    def __init__(self):
        """
        Set some constants
          *zero_crossing*:
            Boolean, *True* if the_domain crosses 0 degree meridian
        """
        self.zero_crossing = False
        self.i_0, self.i_1 = 0, None
        self.j_0, self.j_1 = 0, None
        self.ip0, self.ip1 = 0, None
        self.jp0, self.jp1 = 0, None
        self.iup0, self.iup1 = 0, None
        self.jup0, self.jup1 = 0, None
        self.eke = None
        self.u_val, self.upad = None, None
        self.v_val, self.vpad = None, None
        self.m_val = None
        self.m_x = None
        self.m_y = None
        self._lon = None
        self._lat = None
        self._f_val = None
        self._gof = None
        self._pm = None
        self._pn = None
        self._dx = None
        self._dy = None
        self._umask = None
        self._vmask = None

    def read_nc(self, varfile, varname, indices=slice(None)):
        """
        Read data from nectdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          indices : string of index ranges, eg. '[0,:,0]'
        """
        with Dataset(varfile) as h_nc:
            return h_nc.variables[varname][indices]

    def read_nc_att(self, varfile, varname, att):
        """
        Read data attribute from nectdf file
          varname : variable ('temp', 'mask_rho', etc) to read
          att : string of attribute, eg. 'valid_range'
        """
        with Dataset(varfile) as h_nc:
            return getattr(h_nc.variables[varname], att)

    def set_initial_indices(self):
        """
        Set indices for desired domain
        """
        # print self.zero_crossing
        # print 'wwwwwwww', self._lon.min(), self._lon.max()
        lonmin, lonmax = self.lonmin, self.lonmax
        latmin, latmax = self.latmin, self.latmax

        logging.info('Setting initial indices to *%s* domain', self.the_domain)
        logging.info('lonmin = %s, lonmax = %s, latmin = %s, latmax = %s',
                     lonmin, lonmax, latmin, latmax)
        latmin_offset = latmin + (0.5 * (latmax - latmin))
        self.i_0, _ = self.nearest_point(lonmin, latmin_offset)
        self.i_1, _ = self.nearest_point(lonmax, latmin_offset)
        lonmin_offset = lonmin + (0.5 * (lonmax - lonmin))
        _, self.j_0 = self.nearest_point(lonmin_offset, latmin)
        _, self.j_1 = self.nearest_point(lonmin_offset, latmax)

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

            if self.zero_crossing is True:
                """
                Used for a zero crossing, e.g., across Agulhas region
                """
                def half_limits(lon, lat):
                    return np.array([[lon.min(), lon.max(),
                                      lon.max(), lon.min()],
                                     [lat.min(), lat.min(),
                                      lat.max(), lat.max()]]).T
                # Get bounds for right part of grid
                lat = self._lat[self._lon >= 360 + lonmin - 0.5]
                lon = self._lon[self._lon >= 360 + lonmin - 0.5]
                limits = half_limits(lon, lat)
                iind, _ = kdt(self._lon, self._lat, limits)
                self.i_1 = iind.min()
                # Get bounds for left part of grid
                lat = self._lat[self._lon <= lonmax + 0.5]
                lon = self._lon[self._lon <= lonmax + 0.5]
                limits = half_limits(lon, lat)
                iind, _ = kdt(self._lon, self._lat, limits)
                self.i_0 = iind.max()

        return self

    def set_index_padding(self, pad=2):
        """
        Set start and end indices for temporary padding and later unpadding
        around 2d variables.
        Padded matrices are needed only for geostrophic velocity computation.
        """
        logging.info('\tSetting padding indices with PAD = %s', pad)

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

        self.jp0, self.jup0 = get_str(self.j_0, pad)
        self.jp1, self.jup1 = get_end(self.j_1, self._lon.shape[0], pad)
        if self.zero_crossing:
            pad = -pad
        self.ip0, self.iup0 = get_str(self.i_0, pad)
        self.ip1, self.iup1 = get_end(self.i_1, self._lon.shape[1], pad)
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
        sin_dlat = np.sin(np.deg2rad(lat2 - lat1) * 0.5)
        sin_dlon = np.sin(np.deg2rad(lon2 - lon1) * 0.5)
        cos_lat1 = np.cos(np.deg2rad(lat1))
        cos_lat2 = np.cos(np.deg2rad(lat2))
        a_val = sin_dlon ** 2 * cos_lat1 * cos_lat2 + sin_dlat ** 2
        c_val = 2 * np.arctan2(np.sqrt(a_val), np.sqrt(1 - a_val))
        return 6371315.0 * c_val  # Return the distance

    def nearest_point(self, lon, lat):
        """
        Get indices to a point (lon, lat) in the grid
        """
        return nearest(lon, lat, self._lon[0], self._lat[:, 0])

    def half_interp(self, h_one, h_two):
        """
        Speed up frequent operations of type 0.5 * (arr[:-1] + arr[1:])
        """
        h_one = np.copy(h_one)
        h_one += h_two
        h_one *= 0.5
        return h_one

    def get_AVISO_f_pm_pn(self):
        """
        Padded matrices are used here because Coriolis (f), p_m and p_n
        are needed for the geostrophic velocity computation in
        method getSurfGeostrVel()
        NOTE: this should serve for ROMS too
        """
        logging.info('--- Computing Coriolis (f), d_x(p_m),'
                     'd_y (p_n) for padded grid')
        # Get GRAVITY / Coriolis
        self._gof = np.sin(np.deg2rad(self.latpad))
        self._gof *= 4.
        self._gof *= np.pi
        self._gof /= 86400.
        self._f_val = self._gof.copy()
        self._gof = self.GRAVITY / self._gof

        lonu = self.half_interp(self.lonpad[:, :-1], self.lonpad[:, 1:])
        latu = self.half_interp(self.latpad[:, :-1], self.latpad[:, 1:])
        lonv = self.half_interp(self.lonpad[:-1], self.lonpad[1:])
        latv = self.half_interp(self.latpad[:-1], self.latpad[1:])

        # Get p_m and p_n
        p_m = np.zeros_like(self.lonpad)
        p_m[:, 1:-1] = self.haversine_dist(lonu[:, :-1], latu[:, :-1],
                                           lonu[:, 1:], latu[:, 1:])
        p_m[:, 0] = p_m[:, 1]
        p_m[:, -1] = p_m[:, -2]
        self._dx = p_m
        self._pm = np.reciprocal(p_m)

        p_n = np.zeros_like(self.lonpad)
        p_n[1:-1] = self.haversine_dist(lonv[:-1], latv[:-1],
                                        lonv[1:], latv[1:])
        p_n[0] = p_n[1]
        p_n[-1] = p_n[-2]
        self._dy = p_n
        self._pn = np.reciprocal(p_n)
        return self

    def u2rho_2d(self, uu_in):
        """
        Convert a 2D field at u_val points to a field at rho points
        """
        def uu2ur(uu_in, m_p, l_p):
            l_reduce = l_p - 1
            l_m = l_reduce - 1
            u_out = np.zeros((m_p, l_p))
            u_out[:, 1:l_reduce] = self.half_interp(uu_in[:, 0:l_m],
                                                    uu_in[:, 1:l_reduce])
            u_out[:, 0] = u_out[:, 1]
            u_out[:, l_reduce] = u_out[:, l_m]
            return (u_out.squeeze())
        mshp, lshp = uu_in.shape
        return uu2ur(uu_in, mshp, lshp + 1)

    def v2rho_2d(self, vv_in):
        # Convert a 2D field at v_val points to a field at rho points
        def vv2vr(vv_in, m_p, l_p):
            m_reduce = m_p - 1
            m_m = m_reduce - 1
            v_out = np.zeros((m_p, l_p))
            v_out[1:m_reduce] = self.half_interp(vv_in[:m_m],
                                                 vv_in[1:m_reduce])
            v_out[0] = v_out[1]
            v_out[m_reduce] = v_out[m_m]
            return (v_out.squeeze())
        mshp, lshp = vv_in.shape
        return vv2vr(vv_in, mshp + 1, lshp)

    def rho2u_2d(self, rho_in):
        """
        Convert a 2D field at rho points to a field at u_val points
        """
        def _r2u(rho_in, l_p):
            u_out = rho_in[:, :l_p - 1]
            u_out += rho_in[:, 1:l_p]
            u_out *= 0.5
            return u_out.squeeze()
        assert rho_in.ndim == 2, 'rho_in must be 2d'
        return _r2u(rho_in, rho_in.shape[1])

    def rho2v_2d(self, rho_in):
        """
        Convert a 2D field at rho points to a field at v_val points
        """
        def _r2v(rho_in, m_p):
            v_out = rho_in[:m_p - 1]
            v_out += rho_in[1:m_p]
            v_out *= 0.5
            return v_out.squeeze()
        assert rho_in.ndim == 2, 'rho_in must be 2d'
        return _r2v(rho_in, rho_in.shape[0])

    def uvmask(self):
        """
        Get mask at U and V points
        """
        # print '--- Computing umask and vmask for padded grid'
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
        logging.info('Computing Basemap')
        # Create Basemap instance for Mercator projection.
        self.m_val = Proj(
            '+proj=merc '
            '+llcrnrlon=%(llcrnrlon)f '
            '+llcrnrlat=%(llcrnrlat)f '
            '+urcrnrlon=%(urcrnrlon)f '
            '+urcrnrlat=%(urcrnrlat)f '
            '+lat_ts=%(lat_ts)f' %
            dict(
                llcrnrlon=self.lonmin - 1,
                urcrnrlon=self.lonmax + 1,
                llcrnrlat=self.latmin - 1,
                urcrnrlat=self.latmax + 1,
                lat_ts=0.5 * (self.latmin + self.latmax)
                )
            )

        if with_pad:
            x_val, y_val = self.m_val(self.lonpad, self.latpad)
        else:
            x_val, y_val = self.m_val(self.lon, self.lat)
        self.m_x, self.m_y = x_val, y_val
        return self

    def set_geostrophic_velocity(self, zeta):
        """
        Set u_val and v_val geostrophic velocity at
        surface from variables f, zeta, p_m, p_n...
        Note: output at rho points
        """
        gof = self.gof.view()

        vmask = self.vmask.view()
        zeta1, zeta2 = zeta.data[1:].view(), zeta.data[:-1].view()
        pn1, pn2 = self.p_n[1:].view(), self.p_n[:-1].view()
        self.upad[:] = self.v2rho_2d(vmask * (zeta1 - zeta2) *
                                     0.5 * (pn1 + pn2))
        self.upad *= -gof

        umask = self.umask.view()
        zeta1, zeta2 = zeta.data[:, 1:].view(), zeta.data[:, :-1].view()
        pm1, pm2 = self.p_m[:, 1:].view(), self.p_m[:, :-1].view()
        self.vpad[:] = self.u2rho_2d(umask * (zeta1 - zeta2) *
                                     0.5 * (pm1 + pm2))
        self.vpad *= gof
        return self

    def set_u_v_eke(self, pad=2):
        """
        Set empty arrays for u_val, v_val, upad, vpad
        """
        if self.zero_crossing:
            u1 = np.empty((self.jp1 - self.jp0, self.ip0))
            u0 = np.empty((self.jp1 - self.jp0, self._lon.shape[1] - self.ip1))
            self.upad = np.ma.concatenate((u0, u1), axis=1)
        else:
            self.upad = np.empty((self.jp1 - self.jp0, self.ip1 - self.ip0))
        self.vpad = np.empty_like(self.upad)
        self.eke = np.empty_like(self.upad[self.jup0:self.jup1,
                                           self.iup0:self.iup1])
        self.u_val = np.empty_like(self.eke)
        self.v_val = np.empty_like(self.eke)
        return self

    def getEKE(self):
        """
        Set EKE; also sets u_val/v_val from upad/vpad
        """
        self.u_val[:] = self.upad[self.jup0:self.jup1, self.iup0:self.iup1]
        self.v_val[:] = self.vpad[self.jup0:self.jup1, self.iup0:self.iup1]
        self.eke[:] = self.u_val**2 + self.v_val**2
        self.eke *= 0.5
        return self

    def set_interp_coeffs(self, sla, uspd):
        """
        Won't work for rotated grid
        """
        if 'AVISO' in self.product:
            self.sla_coeffs = interpolate.RectBivariateSpline(
                self.lat[:, 0], self.lon[0], sla, kx=1, ky=1)
            self.uspd_coeffs = interpolate.RectBivariateSpline(
                self.lat[:, 0], self.lon[0], uspd, kx=1, ky=1)
        elif 'ROMS' in self.product:
            points = np.array([self.lon.ravel(), self.lat.ravel()]).T
            self.sla_coeffs = interpolate.CloughTocher2DInterpolator(
                points, sla.ravel())
            self.uspd_coeffs = interpolate.CloughTocher2DInterpolator(
                points, uspd.ravel())
        return self
