# -*- coding: utf-8 -*-
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


py_eddy_tracker_amplitude.py

Version 2.0.3

===========================================================================

"""
from scipy.interpolate import griddata
# from scipy.ndimage import generate_binary_structure, binary_erosion
from scipy.ndimage import binary_erosion
from scipy.ndimage import minimum_filter
from numpy import array, isfinite, ma, where, ones
from .tools import index_from_nearest_path, \
    index_from_nearest_path_with_pt_in_bbox, distance_matrix
import logging


class Amplitude (object):
    """
    Class to calculate *amplitude* and counts of *local maxima/minima*
    within a closed region of a sea level anomaly field.

    Attributes:
      contlon:
        Longitude coordinates of contour

      contlat:
        Latitude coordinates of contour

      eddy:
        A tracklist object holding the SLA data

      grd:
        A grid object
    """
    def __init__(self, contlon, contlat, eddy, grd):
        """
        """
        self.contlon = contlon.copy()
        self.contlat = contlat.copy()
        eddy.grd = grd  # temporary fix
        self.eddy = eddy
        self.sla = self.eddy.sla[self.jslice,
                                 self.islice].copy()

        if 'RectBivariate' in eddy.interp_method:
            h_0 = grd.sla_coeffs.ev(self.contlat[1:], self.contlon[1:])

        elif 'griddata' in eddy.interp_method:
            points = array([grd.lon()[self.jslice, self.islice].ravel(),
                               grd.lat()[self.jslice, self.islice].ravel()]).T
            h_0 = griddata(points, self.sla.ravel(),
                           (self.contlon[1:], self.contlat[1:]),
                           'linear')
        else:
            raise Exception('Unknown method : %s' % eddy.interp_method)

        self.h_0 = h_0[isfinite(h_0)].mean()
        self.amplitude = 0  # atleast_1d(0.)
        self.local_extrema = None  # int(0)
        self.local_extrema_inds = None
        self.sla = ma.masked_where(-self.mask, self.sla)

    @property
    def islice(self):
        return self.eddy.slice_i

    @property
    def jslice(self):
        return self.eddy.slice_j

    @property
    def mask(self):
        return self.eddy.mask_eff

    @property
    def mle(self):
        return self.eddy.max_local_extrema
    
    def within_amplitude_limits(self):
        """
        """
        return (self.amplitude >= self.eddy.ampmin and
                self.amplitude <= self.eddy.ampmax)

    def _set_cyc_amplitude(self):
        """
        """
        self.amplitude = self.h_0
        self.amplitude -= self.sla.min()

    def _set_acyc_amplitude(self):
        """
        """
        self.amplitude = self.sla.max()
        self.amplitude -= self.h_0

    def all_pixels_below_h0(self, level):
        """
        Check CSS11 criterion 1: The SSH values of all of the pixels
        are below a given SSH threshold for cyclonic eddies.
        """
        if (self.sla > self.h_0).any():
            return False  # i.e., with self.amplitude == 0
        else:
            self._set_local_extrema(1)
            if (self.local_extrema > 0 and
                    self.local_extrema <= self.mle):
                self._set_cyc_amplitude()
            elif self.local_extrema > self.mle:
                lmi_j, lmi_i = where(self.local_extrema_inds)
                levnm2 = level - (2 * self.eddy.interval)
                slamin = 1e5
                for j, i in zip(lmi_j, lmi_i):
                    if slamin >= self.sla[j, i]:
                        slamin = self.sla[j, i]
                        jmin, imin = j, i
                    if self.sla[j, i] >= levnm2:
                        self._set_cyc_amplitude()
                        # Prevent further calls to_set_cyc_amplitude
                        levnm2 = 1e5
                jmin += self.eddy.jmin
                imin += self.eddy.imin
                return (imin, jmin)
        return False

    def all_pixels_above_h0(self, level):
        """
        Check CSS11 criterion 1: The SSH values of all of the pixels
        are above a given SSH threshold for anticyclonic eddies.
        """
        if (self.sla < self.h_0).any():
            # i.e.,with self.amplitude == 0
            return False
        else:
            self._set_local_extrema(-1)
            if (self.local_extrema > 0 and
                    self.local_extrema <= self.mle):
                self._set_acyc_amplitude()

            elif self.local_extrema > self.mle:
                lmi_j, lmi_i = where(self.local_extrema_inds)
                levnp2 = level + (2 * self.eddy.interval)
                slamax = -1e5
                for j, i in zip(lmi_j, lmi_i):
                    if slamax <= self.sla[j, i]:
                        slamax = self.sla[j, i]
                        jmax, imax = j, i
                    if self.sla[j, i] <= levnp2:
                        self._set_acyc_amplitude()
                        # Prevent further calls to_set_acyc_amplitude
                        levnp2 = -1e5
                jmax += self.eddy.jmin
                imax += self.eddy.imin
                return (imax, jmax)
        return False

    def _set_local_extrema(self, sign):
        """
        Set count of local SLA maxima/minima within eddy
        """
        self._detect_local_minima(self.sla * sign)

    def _detect_local_minima(self, arr):
        """
        Take an array and detect the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e., 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        # Equivalent
        neighborhood = ones((3, 3), dtype='bool')
        #~ neighborhood = generate_binary_structure(arr.ndim, 2)

        # Get local mimima
        detected_minima = minimum_filter(
            arr, footprint=neighborhood) == arr
        background = (arr == 0)
        # Aims ?
        eroded_background = binary_erosion(
            background, structure=neighborhood, border_value=1)
        detected_minima -= eroded_background
        # mask of minima
        self.local_extrema_inds = detected_minima
        # nb of minima
        self.local_extrema = detected_minima.sum()


class SwirlSpeed(object):
    """
    Class to calculate average geostrophic velocity along
    a contour, *uavg*, and return index to contour with maximum
    *uavg* within a series of closed contours.

    Attributes:
      contour:
        A matplotlib contour object of high-pass filtered SLA

      eddy:
        A tracklist object holding the SLA data

      grd:
        A grid object
    """
    def __init__(self, contour, nearest_contain_in_bbox=False):
        """
        c_i : index to contours
        l_i : index to levels
        """
        x_list, y_list, ci_list, li_list = [], [], [], []
        x_min_list, y_min_list, x_max_list, y_max_list = [], [], [], []

        for cont in contour.collections:
            for coll in cont.get_paths():
                x_val, y_val = coll.vertices[:, 0], coll.vertices[:, 1]
                x_min_list.append(x_val.min())
                x_max_list.append(x_val.max())
                y_min_list.append(y_val.min())
                y_max_list.append(y_val.max())
                x_list.append(x_val)
                y_list.append(y_val)
                ci_list.append(coll.vertices.shape[0])
            li_list.append(len(cont.get_paths()))

        self.x_value = array([val for sublist in x_list for val in sublist])
        self.y_value = array([val for sublist in y_list for val in sublist])

        self.x_min_per_c = array(x_min_list)
        self.y_min_per_c = array(y_min_list)
        self.x_max_per_c = array(x_max_list)
        self.y_max_per_c = array(y_max_list)

        self.nb_pt_per_c = array(ci_list, dtype='u4')
        self.c_i = array(self.nb_pt_per_c.cumsum() - self.nb_pt_per_c,
                         dtype='u4')
        self.nb_c_per_l = array(li_list, dtype='u4')
        self.l_i = array(self.nb_c_per_l.cumsum() - self.nb_c_per_l,
                         dtype='u4')
                         
        self.nearest_contain_in_bbox = nearest_contain_in_bbox

    def get_index_nearest_path_bbox_contain_pt(self, level, xpt, ypt):
        """
        """
        return index_from_nearest_path_with_pt_in_bbox(
            level,
            self.l_i,
            self.nb_c_per_l,
            self.nb_pt_per_c,
            self.c_i,
            self.x_value,
            self.y_value,
            self.x_min_per_c,
            self.y_min_per_c,
            self.x_max_per_c,
            self.y_max_per_c,
            xpt,
            ypt
            )

    def get_index_nearest_path(self, level, xpt, ypt):
        """
        """
        return index_from_nearest_path(
            level,
            self.l_i,
            self.nb_c_per_l,
            self.nb_pt_per_c,
            self.c_i,
            self.x_value,
            self.y_value,
            xpt,
            ypt
            )
