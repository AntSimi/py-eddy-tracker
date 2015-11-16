# -*- coding: utf-8 -*-
from matplotlib.dates import date2num
from scipy import ndimage
from scipy import spatial
from dateutil import parser
import numpy as np
import logging

from . import PyEddyTracker


class AvisoGrid(PyEddyTracker):
    """
    Class to satisfy the need of the eddy tracker
    to have a grid class
    """
    def __init__(self, aviso_file, the_domain, product,
                 lonmin, lonmax, latmin, latmax, with_pad=True):
        """
        Initialise the grid object
        """
        super(AvisoGrid, self).__init__()
        logging.info('Initialising the *AVISO_grid*')
        self.the_domain = the_domain
        if 'AVISO' in product:
            self.product = 'AVISO'
        else:
            self.product = product
        self.lonmin = lonmin
        self.lonmax = lonmax
        self.latmin = latmin
        self.latmax = latmax

        try:  # new AVISO (2014)
            self._lon = self.read_nc(aviso_file, 'lon')
            self._lat = self.read_nc(aviso_file, 'lat')
            self.fillval = self.read_nc_att(aviso_file, 'sla', '_FillValue')
            base_date = self.read_nc_att(aviso_file, 'time', 'units')
            self.base_date = date2num(
                parser.parse(base_date.split(' ')[2:4][0]))

        except Exception:  # old AVISO
            self._lon = self.read_nc(aviso_file, 'NbLongitudes')
            self._lat = self.read_nc(aviso_file, 'NbLatitudes')
            self.fillval = self.read_nc_att(aviso_file,
                                            'Grid_0001', '_FillValue')

        if lonmin < 0 and lonmax <= 0:
            self._lon -= 360.
        self._lon, self._lat = np.meshgrid(self._lon, self._lat)
        self._angle = np.zeros_like(self._lon)

        if 'MedSea' in self.the_domain:
            self._lon -= 360.

        # zero_crossing, used for handling a longitude range that
        # crosses zero degree meridian
        if lonmin < 0 and lonmax >= 0 and 'MedSea' not in self.the_domain:
            self.zero_crossing = True

        self.sla_coeffs = None
        self.uspd_coeffs = None

        self.__lon, self.__lat = None, None
        self.__lonpad, self.__latpad = None, None
        self.set_initial_indices()
        self.set_index_padding()
        self.set_basemap(with_pad=with_pad)
        self.get_aviso_f_pm_pn()
        self.set_u_v_eke()
        self.shape = self.lon.shape
#         pad2 = 2 * self.pad
#         self.shape = (self.f_coriolis.shape[0] - pad2,
#                       self.f_coriolis.shape[1] - pad2)

    def get_aviso_data(self, aviso_file):
        """
        Read nc data from AVISO file
        """
        if 'Grid_0001' in self.nc_variables(aviso_file):
            # old AVISO
            zeta = self.read_nc(aviso_file, 'Grid_0001',
                                indices=list(self.view_pad)[::-1]).T
        else:
            zeta = self.read_nc(aviso_file, 'sla',
                                indices=[0, self.view_pad[0], self.view_pad[1]]).T
            zeta *= 100.  # m to cm
        if hasattr(zeta, 'mask'):
            return zeta
        else:
            try:
                # Extrapolate over land points with fillmask
                # Something to check in this function
                zeta = np.ma.array(zeta)
            except:  # In case no landpoints
                zeta = np.ma.array(zeta)
            return zeta

    def set_mask(self, sla):
        """
        """
        if sla.mask.size == 1:  # all sea points
            self.mask = None
        else:
            self.mask = sla.mask.copy()
            if 'Global' in self.the_domain:

                # Close Drake Passage
                minus70 = np.argmin(np.abs(self.lonpad[0] + 70))
                self.mask[:125, minus70] = True

                # DT10 mask is open around Panama, so close it...
                if 'AVISO_DT10' in self.product:
                    self.mask[348, 92:110] = True
                    self.mask[348:356, 92] = True
                    self.mask[355, 71:92] = True
                    self.mask[355:363, 71] = True
                    self.mask[362, 66:71] = True
                    self.mask[362:380, 66] = True
                    self.mask[380, 47:67] = True
                    self.mask[380:389, 47] = True
                    self.mask[388, 13:47] = True
                    self.mask[388:393, 13] = True
                    self.mask[392, :13] = True
                    ind = 4 * 360
                    self.mask[348, 92 + ind:110 + ind] = True
                    self.mask[348:356, 92 + ind] = True
                    self.mask[355, 71 + ind:92 + ind] = True
                    self.mask[355:363, 71 + ind] = True
                    self.mask[362, 66 + ind:71 + ind] = True
                    self.mask[362:380, 66 + ind] = True
                    self.mask[380, 47 + ind:67 + ind] = True
                    self.mask[380:389, 47 + ind] = True
                    self.mask[388, 13 + ind:47 + ind] = True
                    self.mask[388:393, 13 + ind] = True
                    self.mask[392,  ind:13 + ind] = True

                # Mask all unwanted regions (Caspian Sea, etc)
                self.labels = ndimage.label(-self.mask)[0]

                # Set to known sea point
                plus200 = np.argmin(np.abs(self.lonpad[0] - 200))
                plus9 = np.argmin(np.abs(self.latpad[:, 0] - 9))
                sea_label = self.labels[plus9, plus200]
                self.mask += self.labels != sea_label
        return self

    def fillmask(self, data, mask):
        """
        Fill missing values in an array with an average of nearest
        neighbours
        From http://permalink.gmane.org/gmane.comp.python.scientific.user/19610
        """
        assert data.ndim == 2, 'data must be a 2D array.'
        fill_value = 9999.99
        data[mask == 0] = fill_value

        # Create (i, j) point arrays for good and bad data.
        # Bad data are marked by the fill_value, good data elsewhere.
        igood = np.vstack(np.where(data != fill_value)).T
        ibad = np.vstack(np.where(data == fill_value)).T

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
        xfill = weight * data[igood[:, 0][iquery], igood[:, 1][iquery]]
        xfill = (xfill / weight.sum(axis=1)[:, np.newaxis]).sum(axis=1)

        # Place average of nearest good points, xfill, into bad point locations
        data[ibad[:, 0], ibad[:, 1]] = xfill
        return data

    @property
    def lon(self):
        if self.__lon is None:
            # It must be an 1D array and not an 2d array ?
            self.__lon = self._lon[self.view]
            if self.zero_crossing:
                self.__lon[:, :self._lon.shape[1] - self.slice_i.stop] -= 360
        return self.__lon

    @property
    def lat(self):
        # It must be an 1D array and not an 2d array ?
        if self.__lat is None:
            self.__lat = self._lat[self.view]
        return self.__lat

    @property
    def lonpad(self):
        if self.__lonpad is None:
            self.__lonpad = self._lon[self.view_pad]
            if self.zero_crossing:
                self.__lonpad[:, :self._lon.shape[1] - self.slice_i_pad.stop
                              ] -= 360
        return self.__lonpad

    @property
    def latpad(self):
        if self.__latpad is None:
            self.__latpad = self._lat[self.view_pad]
        return self.__latpad

    @property
    def angle(self):
        return self._angle[self.view]

    @property
    def umask(self):  # Mask at U points
        return self._umask

    @property
    def vmask(self):  # Mask at V points
        return self._vmask

    @property
    def f_coriolis(self):  # Coriolis
        return self._f_val

    @property
    def gof(self):  # Gravity / Coriolis
        return self._gof

    @property
    def d_x(self):  # Grid spacing along X direction
        return self._dx

    @property
    def d_y(self):  # Grid spacing along Y direction
        return self._dy

    @property
    def p_m(self):  # Reciprocal of d_x
        return self._pm

    @property
    def p_n(self):  # Reciprocal of d_y
        return self._pn

    @property
    def resolution(self):
        return np.sqrt(np.diff(self.lon[1:], axis=1) *
                       np.diff(self.lat[:, 1:], axis=0)).mean()

    @property
    def boundary(self):
        """
        Return lon, lat of perimeter around a ROMS grid
        Input:
          indices to get boundary of specified subgrid
        Returns:
          lon/lat boundary points
        """
        lon = np.r_[(self.lon[:, 0], self.lon[-1],
                     self.lon[::-1, -1], self.lon[0, ::-1])]
        lat = np.r_[(self.lat[:, 0], self.lat[-1],
                     self.lat[::-1, -1], self.lat[0, ::-1])]
        return lon, lat
