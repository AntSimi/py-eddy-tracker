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
from scipy.ndimage import generate_binary_structure, binary_erosion
from scipy.ndimage import minimum_filter
import numpy as np
import matplotlib.pyplot as plt
from .tools import index_from_nearest_path
import logging


class EddyProperty (object):
    """
    Class to hold eddy properties *amplitude* and counts of
    *local maxima/minima* within a closed region of a sea level anomaly field.

    Variables:
      centlon:
        Longitude centroid coordinate

      centlat:
        Latitude centroid coordinate

      eddy_radius_s:
        Speed based radius

      eddy_radius_e:
        Effective radius

      amplitude:
        Eddy amplitude

      uavg:
        Average eddy swirl speed

      teke:
        Average eddy kinetic energy within eddy

      rtime:
        Time
    """
    def __init__(self):
        """
        """
        self.centlon = 0
        self.centlat = 0
        self.eddy_radius_s = 0
        self.eddy_radius_e = 0
        self.amplitude = 0
        self.uavg = 0
        self.teke = 0
        self.rtime = 0


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
        self.mle = self.eddy.MAX_LOCAL_EXTREMA
        self.islice = slice(self.eddy.imin, self.eddy.imax)
        self.jslice = slice(self.eddy.jmin, self.eddy.jmax)
        self.sla = self.eddy.sla[self.jslice,
                                 self.islice].copy()

        if 'RectBivariate' in eddy.INTERP_METHOD:
            h_0 = grd.sla_coeffs.ev(self.contlat[1:], self.contlon[1:])

        elif 'griddata' in eddy.INTERP_METHOD:
            points = np.array([grd.lon()[self.jslice, self.islice].ravel(),
                               grd.lat()[self.jslice, self.islice].ravel()]).T
            h_0 = griddata(points, self.sla.ravel(),
                           (self.contlon[1:], self.contlat[1:]),
                           'linear')
        else:
            Exception

        self.h0_check = h_0
        self.h_0 = h_0[np.isfinite(h_0)].mean()
        self.amplitude = 0  # np.atleast_1d(0.)
        self.local_extrema = None  # np.int(0)
        self.local_extrema_inds = None
        self.mask = self.eddy.mask_eff
        self.sla = np.ma.masked_where(-self.mask, self.sla)
        self.num_features = np.atleast_1d(0)

    def within_amplitude_limits(self):
        """
        """
        return (self.amplitude >= self.eddy.AMPMIN and
                self.amplitude <= self.eddy.AMPMAX)

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
        if np.any(self.sla > self.h_0):
            return False  # i.e., with self.amplitude == 0
        else:
            self._set_local_extrema(1)
            if (self.local_extrema > 0 and
                    self.local_extrema <= self.mle):
                self._set_cyc_amplitude()

            elif self.local_extrema > self.mle:
                lmi_j, lmi_i = np.where(self.local_extrema_inds)
                levnm2 = level - (2 * self.eddy.INTERVAL)
                slamin = np.atleast_1d(1e5)
                for j, i in zip(lmi_j, lmi_i):
                    if slamin >= self.sla[j, i]:
                        slamin[:] = self.sla[j, i]
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
        if np.any(self.sla < self.h_0):
            return False  # i.e.,with self.amplitude == 0

        else:
            self._set_local_extrema(-1)

            if (self.local_extrema > 0 and
                    self.local_extrema <= self.mle):
                self._set_acyc_amplitude()

            elif self.local_extrema > self.mle:
                lmi_j, lmi_i = np.where(self.local_extrema_inds)
                levnp2 = level + (2 * self.eddy.INTERVAL)
                slamax = np.atleast_1d(-1e5)
                for j, i in zip(lmi_j, lmi_i):
                    if slamax <= self.sla[j, i]:
                        slamax[:] = self.sla[j, i]
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
        # local_extrema = np.ma.masked_where(self.mask == False, self.sla)
        local_extrema = self.sla.copy()
        local_extrema *= sign
        self._detect_local_minima(local_extrema)
        return self

    def _detect_local_minima(self, arr):
        """
        Take an array and detect the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e., 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        neighborhood = generate_binary_structure(arr.ndim, 2)
        # Get local mimima
        detected_minima = (minimum_filter(arr,
                           footprint=neighborhood) == arr)
        background = (arr == 0)
        eroded_background = binary_erosion(
            background, structure=neighborhood, border_value=1)
        detected_minima -= eroded_background
        self.local_extrema_inds = detected_minima
        self.local_extrema = detected_minima.sum()
        return self

    def debug_figure(self, grd):
        """
        Uncomment in py-eddy-tracker-classes.py
        """
        if self.local_extrema >= 1 and self.amplitude:
            plt.figure(587)
            cmin, cmax = -8, 8
#             cmin, cmax = (self.h0_check.min() - 2,
#                         self.h0_check.max() + 2)
#             cmin, cmax = (self.sla.min(), self.sla.max())

            plt.title('Local max/min count: %s, Amp: %s' % (
                self.local_extrema, self.amplitude))

            pcm = plt.pcolormesh(
                grd.lon[self.jslice, self.islice],
                grd.lat[self.jslice, self.islice],
                self.sla.data, cmap='gist_ncar')
            plt.clim(cmin, cmax)
            plt.plot(self.contlon, self.contlat)
            plt.scatter(self.contlon[1:], self.contlat[1:], s=100,
                        c=self.h0_check, cmap='gist_ncar', vmin=cmin,
                        vmax=cmax)
            # plt.scatter(centlon_lmi, centlat_lmi, c='k')
            # plt.scatter(centlon_e, centlat_e, c='w')
            lmi_j, lmi_i = np.where(self.local_extrema_inds)
            # lmi_i = lmi_i[0] + self.eddy.imin
            # lmi_j = lmi_j[0] + self.eddy.jmin
            # print lmi_i
            lmi_i = np.array(lmi_i) + self.eddy.imin
            lmi_j = np.array(lmi_j) + self.eddy.jmin
            x_i, y_i = grd.lon()[lmi_j, lmi_i], grd.lat()[lmi_j, lmi_i]
            plt.scatter(x_i, y_i, s=40, c='w')
            plt.axis('image')
            plt.colorbar(pcm)
            plt.show()
        return


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
    def __init__(self, contour):
        """
        c_i : index to contours
        l_i : index to levels
        """
        x_list, y_list, ci_list, li_list = [], [], [], []

        for cont in contour.collections:
            for coll in cont.get_paths():
                x_list.append(coll.vertices[:, 0])
                y_list.append(coll.vertices[:, 1])
                ci_list.append(len(coll.vertices[:, 0]))
            li_list.append(len(cont.get_paths()))

        self.x_value = np.array([val for sublist in x_list for val in sublist])
        self.y_value = np.array([val for sublist in y_list for val in sublist])
        self.nb_pt_per_c = np.array(ci_list, dtype='u4')
        self.c_i = np.array(self.nb_pt_per_c.cumsum() - self.nb_pt_per_c,
                            dtype='u4')
        self.nb_c_per_l = np.array(li_list, dtype='u4')
        self.l_i = np.array(self.nb_c_per_l.cumsum() - self.nb_c_per_l,
                            dtype='u4')

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
