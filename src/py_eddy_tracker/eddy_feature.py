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


class Amplitude(object):
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

    def __init__(self, contour_x, contour_y, i_contour_x, i_contour_y, contour_height, data):
        """
        """

        # self.sla = self.eddy.sla[self.jslice,
        # self.islice].copy()

        self.h_0 = contour_height
        self.data = data
        self.ix = i_contour_x
        self.iy = i_contour_y
        self.contour_y = contour_y
        self.contour_x = contour_x
        self.amplitude = 0
        self.local_extrema = None
        self.local_extrema_inds = None

    def within_amplitude_limits(self):
        """
        """
        return True
        return self.eddy.ampmin <= self.amplitude <= self.eddy.ampmax

    def _set_cyc_amplitude(self):
        """Get amplitude for cyclone
        """
        self.amplitude = self.h_0 - self.sla.min()

    def _set_acyc_amplitude(self):
        """Get amplitude for anticyclone
        """
        self.amplitude = self.sla.max() - self.h_0

    def all_pixels_below_h0(self, level):
        """
        Check CSS11 criterion 1: The SSH values of all of the pixels
        are below a given SSH threshold for cyclonic eddies.
        """
        if (self.sla > self.h_0).any():
            return False  # i.e., with self.amplitude == 0
        else:
            self._set_local_extrema(1)
            if 0 < self.local_extrema <= self.mle:
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
                return imin, jmin
        return False

    def all_pixels_above_h0(self):
        """
        Check CSS11 criterion 1: The SSH values of all of the pixels
        are above a given SSH threshold for anticyclonic eddies.
        """
        sla = self.data[self.ix, self.iy]
        if (sla < self.h_0).any():
            # i.e.,with self.amplitude == 0
            return False
        else:
            self._set_local_extrema(-1)
            # If we have a number of extrema avoid, we compute amplitude
            if 0 < self.local_extrema <= self.mle:
                self._set_acyc_amplitude()

            # More than avoid
            elif self.local_extrema > self.mle:
                # index of extrema
                lmi_j, lmi_i = where(self.local_extrema_inds)

                levnp2 = level + (2 * self.eddy.interval)
                slamax = -1e5
                # Iteration on extrema
                for j, i in zip(lmi_j, lmi_i):

                    # We iterate on max and search the first sup of slamax
                    if slamax <= self.sla[j, i]:
                        slamax = self.sla[j, i]
                        jmax, imax = j, i

                    if self.sla[j, i] <= levnp2:
                        self._set_acyc_amplitude()
                        # Prevent further calls to_set_acyc_amplitude
                        levnp2 = -1e5
                jmax += self.eddy.jmin
                imax += self.eddy.imin
                return imax, jmax
        return False

    def _set_local_extrema(self, sign):
        """
        Set count of local SLA maxima/minima within eddy
        """
        # mask of minima
        self.local_extrema_inds = self.detect_local_minima(self.sla * sign)
        # nb of minima
        self.local_extrema = self.local_extrema_inds.sum()

    @staticmethod
    def detect_local_minima(grid):
        """
        Take an array and detect the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e., 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        # Equivalent
        neighborhood = ones((3, 3), dtype='bool')
        # ~ neighborhood = generate_binary_structure(grid.ndim, 2)

        # Get local mimima
        detected_minima = minimum_filter(
            grid, footprint=neighborhood) == grid
        background = (grid == 0)
        # Aims ?
        eroded_background = binary_erosion(
            background, structure=neighborhood, border_value=1)
        detected_minima -= eroded_background
        # mask of minima
        return detected_minima
