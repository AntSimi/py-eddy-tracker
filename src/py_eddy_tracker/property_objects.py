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

Copyright (c) 2014-2017 by Evan Mason and Antoine Delepoulle
Email: emason@imedea.uib-csic.es
===========================================================================

property_objects.py

Version 3.0.0

===========================================================================

"""
from scipy.interpolate import griddata
from scipy.ndimage import binary_erosion
from scipy.ndimage import minimum_filter
from numpy import array, isfinite, ma, where, ones, empty
from .tools import index_from_nearest_path_with_pt_in_bbox


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
    __slots__ = (
        'eddy',
        'sla',
        'h_0',
        'amplitude',
        'local_extrema',
        'local_extrema_inds',
        )
    
    def __init__(self, contlon, contlat, eddy, grd):
        """
        """
        eddy.grd = grd  # temporary fix
        self.eddy = eddy
        self.sla = self.eddy.sla[self.jslice,
                                 self.islice].copy()

        if 'RectBivariate' in eddy.interp_method:
            h_0 = grd.sla_coeffs.ev(contlat[1:], contlon[1:])

        elif 'griddata' in eddy.interp_method:
            points = array([grd.lon()[self.jslice, self.islice].ravel(),
                               grd.lat()[self.jslice, self.islice].ravel()]).T
            h_0 = griddata(points, self.sla.ravel(),
                           (contlon[1:], contlat[1:]),
                           'linear')
        else:
            raise Exception('Unknown method : %s' % eddy.interp_method)

        self.h_0 = h_0[isfinite(h_0)].mean()
        self.amplitude = 0  # atleast_1d(0.)
        self.local_extrema = None  # int(0)
        self.local_extrema_inds = None
        self.sla = ma.array(self.sla, mask=~self.mask)

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
                index = self.sla[lmi_j, lmi_i].argmin()
                jmin_, imin_ = lmi_j[index], lmi_i[index]
                if self.sla[jmin_, imin_] >= levnm2:
                    self._set_cyc_amplitude()

                jmin_ += self.eddy.jmin
                imin_ += self.eddy.imin

                return imin_, jmin_
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
            if 0 < self.local_extrema <= self.mle:
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
                return imax, jmax
        return False

    def _set_local_extrema(self, sign):
        """
        Set count of local SLA maxima/minima within eddy
        """
        self._detect_local_minima(self.sla * sign)

    def _detect_local_minima(self, grid):
        """
        Take an array and detect the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e., 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        # Equivalent
        neighborhood = ones((3, 3), dtype='bool')
        #~ neighborhood = generate_binary_structure(grid.ndim, 2)

        # Get local mimima
        detected_minima = minimum_filter(
            grid, footprint=neighborhood) == grid
        background = (grid == 0)
        # Aims ?
        eroded_background = binary_erosion(
            background, structure=neighborhood, border_value=1)
        detected_minima ^= eroded_background
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
    __slots__= (
        'x_value',
        'y_value',
        'contour_index',
        'level_index',
        'x_min_per_contour',
        'y_min_per_contour',
        'x_max_per_contour',
        'y_max_per_contour',
        'nb_pt_per_contour',
        'nb_contour_per_level',
        '_is_valid',
        )

    def __init__(self, contours):
        """
        c_i : index to contours
        l_i : index to levels
        """
        nb_level = 0
        nb_contour = 0
        nb_pt = 0
        # Count level and contour
        for collection in contours.collections:
            nb_level += 1
            for contour in collection.get_paths():
                nb_contour += 1
                nb_pt += contour.vertices.shape[0]
        # Type for coordinates
        coord_dtype = contour.vertices.dtype
        
        # Array declaration
        self.x_value = empty(nb_pt, dtype=coord_dtype)
        self.y_value = empty(nb_pt, dtype=coord_dtype)
        
        self.level_index = empty(nb_level, dtype='u4')
        self.nb_contour_per_level = empty(nb_level, dtype='u4')

        self.nb_pt_per_contour = empty(nb_contour, dtype='u4')

        self.x_min_per_contour = empty(nb_contour, dtype=coord_dtype)
        self.x_max_per_contour = empty(nb_contour, dtype=coord_dtype)
        self.y_min_per_contour = empty(nb_contour, dtype=coord_dtype)
        self.y_max_per_contour = empty(nb_contour, dtype=coord_dtype)
        
        #~ self._is_valid = empty((nb_contour), dtype='bool')

        # Filled array
        i_pt = 0
        i_c = 0
        i_l = 0
        for collection in contours.collections:
            self.level_index[i_l] = i_c
            for contour in collection.get_paths():
                nb_pt = contour.vertices.shape[0]
                # Copy pt
                self.x_value[i_pt:i_pt + nb_pt] = contour.vertices[:, 0]
                self.y_value[i_pt:i_pt + nb_pt] = contour.vertices[:, 1]

                # Set bbox
                self.x_min_per_contour[i_c], self.y_min_per_contour[i_c] = \
                    contour.vertices.min(axis=0)
                self.x_max_per_contour[i_c], self.y_max_per_contour[i_c] = \
                    contour.vertices.max(axis=0)

                # Count pt
                self.nb_pt_per_contour[i_c] = nb_pt
                i_pt += nb_pt
                i_c += 1
            i_l += 1

        self.contour_index = array(
            self.nb_pt_per_contour.cumsum() - self.nb_pt_per_contour,
            dtype='u4'
            )
        self.level_index[0] = 0
        self.nb_contour_per_level[:-1] = self.level_index[1:] - self.level_index[:-1]
        self.nb_contour_per_level[-1] = nb_contour - self.level_index[-1]

    def get_index_nearest_path_bbox_contain_pt(self, level, xpt, ypt):
        """Get index from the nearest path in the level, if the bbox of the
        path contain pt
        """
        return index_from_nearest_path_with_pt_in_bbox(
            level,
            self.level_index,
            self.nb_contour_per_level,
            self.nb_pt_per_contour,
            self.contour_index,
            self.x_value,
            self.y_value,
            self.x_min_per_contour,
            self.y_min_per_contour,
            self.x_max_per_contour,
            self.y_max_per_contour,
            xpt,
            ypt
            )
