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

Version 2.0.3

===========================================================================

"""

import logging
from numpy import where, empty, array, concatenate, ma, zeros, unique, round
from scipy.ndimage import minimum_filter
from matplotlib.figure import Figure
from .tools import index_from_nearest_path, index_from_nearest_path_with_pt_in_bbox


class Amplitude(object):
    """
    Class to calculate *amplitude* and counts of *local maxima/minima*
    within a closed region of a sea level anomaly field.
    """
    EPSILON = 1e-8
    __slots__ = (
        'h_0',
        'grid_extract',
        'mask',
        'sla',
        'contour',
        'interval',
        'amplitude',
        'mle',
        )

    def __init__(self, contour, contour_height, data, interval):
        # Height of the contour
        self.h_0 = contour_height
        # Step between two level
        self.interval = interval
        # Indices of all pixels in contour
        self.contour = contour
        # Link on original grid (local view) or copy if it's on bound
        slice_x, slice_y = contour.bbox_slice
        if slice_x.start > slice_x.stop:
            self.grid_extract = ma.concatenate((data[slice_x.start:, slice_y], data[:slice_x.stop, slice_y]))
        else:
            self.grid_extract = data[slice_x, slice_y]
        # => maybe replace pixel out of contour by nan?
        self.mask = zeros(self.grid_extract.shape, dtype='bool')
        self.mask[contour.pixels_index[0] - slice_x.start, contour.pixels_index[1] - slice_y.start] = True

        # Only pixel in contour
        self.sla = data[contour.pixels_index]
        # Amplitude which will be provide
        self.amplitude = 0
        # Maximum local extrema accepted
        self.mle = 1

    def within_amplitude_limits(self):
        """Need update
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
        # In some case pixel value must be very near of contour bounds
        if ((self.sla - self.h_0) > self.EPSILON).any() or (hasattr(self.sla, 'mask') and self.sla.mask.any()):
            return False
        else:
            # All local extrema index on th box
            lmi_i, lmi_j = self._set_local_extrema(1)
            slice_x, slice_y = self.contour.bbox_slice
            if len(lmi_i) == 1:
                i, j = lmi_i[0] + slice_x.start, lmi_j[0] + slice_y.start
            else:
                # Verify if several extrema are seriously below contour
                nb_real_extrema = ((level - self.grid_extract[lmi_i, lmi_j]) >= 2 * self.interval).sum()
                if nb_real_extrema > self.mle:
                    return False
                index = self.grid_extract[lmi_i, lmi_j].argmin()
                i, j = lmi_i[index] + slice_x.start, lmi_j[index] + slice_y.start
            self._set_cyc_amplitude()
            return i, j

    def all_pixels_above_h0(self, level):
        """
        Check CSS11 criterion 1: The SSH values of all of the pixels
        are above a given SSH threshold for anticyclonic eddies.
        """
        # In some case pixel value must be very near of contour bounds
        if ((self.sla - self.h_0) < - self.EPSILON).any() or (hasattr(self.sla, 'mask') and self.sla.mask.any()):
            # i.e.,with self.amplitude == 0
            return False
        else:

            # All local extrema index on th box
            lmi_i, lmi_j = self._set_local_extrema(-1)
            slice_x, slice_y = self.contour.bbox_slice
            if len(lmi_i) == 1:
                i, j = lmi_i[0] + slice_x.start, lmi_j[0] + slice_y.start
            else:
                # Verify if several extrema are seriously above contour
                nb_real_extrema = ((self.grid_extract[lmi_i, lmi_j] - level) >= 2 * self.interval).sum()
                if nb_real_extrema > self.mle:
                    return False
                index = self.grid_extract[lmi_i, lmi_j].argmax()
                i, j = lmi_i[index] + slice_x.start, lmi_j[index] + slice_y.start
            self._set_cyc_amplitude()
            return i, j

    def _set_local_extrema(self, sign):
        """
        Set count of local SLA maxima/minima within eddy
        """
        # index of minima on whole grid extract
        i_x, i_y = self.detect_local_minima(self.grid_extract * sign)
        # Only index in contour
        m = self.mask[i_x, i_y]
        i_x, i_y = i_x[m], i_y[m]
        # Verify if some extramum is contigus
        nb_extrema = len(i_x)
        if nb_extrema > 1:
            # Group
            nb_group = 1
            gr = zeros(nb_extrema, dtype='u2')
            for i1, (i, j) in enumerate(zip(i_x[:-1], i_y[:-1])):
                for i2, (k, l) in enumerate(zip(i_x[i1 + 1:], i_y[i1 + 1:])):
                    if (abs(i - k) + abs(j - l)) == 1:
                        i2_ = i2 + i1 + 1
                        if gr[i1] == 0 and gr[i2_] == 0:
                            # Nobody was link with a know group
                            gr[i1] = nb_group
                            gr[i2_] = nb_group
                            nb_group += 1
                        elif gr[i1] == 0 and gr[i2_] != 0:
                            # i2 is link not i1
                            gr[i1] = gr[i2_]
                        elif gr[i2_] == 0 and gr[i1] != 0:
                            # i1 is link not i2
                            gr[i2_] = gr[i1]
                        else:
                            # there already linked in two different group
                            # we replace group from i1 with group from i2
                            gr[gr == gr[i1]] = gr[i2_]
            m = gr != 0
            grs = unique(gr[m])
            # all non grouped extremum
            i_x_new, i_y_new = list(i_x[~m]), list(i_y[~m])
            for gr_ in grs:
                m = gr_ == gr
                # Choose barycentre of group
                i_x_new.append(round(i_x[m].mean(axis=0)).astype('i2'))
                i_y_new.append(round(i_y[m].mean(axis=0)).astype('i2'))
            return i_x_new, i_y_new
        return i_x, i_y

    @staticmethod
    def detect_local_minima(grid):
        """
        Take an array and detect the troughs using the local maximum filter.
        Returns a boolean mask of the troughs (i.e., 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise)
        http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
        """
        # To don't perturbate filter
        if hasattr(grid, 'mask'):
            grid[grid.mask] = 2e10
        # Get local mimima
        detected_minima = minimum_filter(grid, size=3) == grid
        # index of minima
        return where(detected_minima)


class Contours(object):
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
    __slots__ = (
        'contours',
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
    )

    DELTA_PREC = 1e-10
    DELTA_SUP= 1e-2

    def get_next(self, origin, paths_left, paths_right):
        for i, path in enumerate(paths_right):
            if abs(origin.vertices[-1, 1] - path.vertices[0, 1]) < self.DELTA_PREC:
                if (path.vertices[0, 0] - origin.vertices[-1, 0]) > 1:
                    path.vertices[:, 0] -= 360
                origin.vertices = concatenate((origin.vertices, path.vertices))
                paths_right.pop(i)
                if self.check_closing(origin):
                    origin.vertices[-1] = origin.vertices[0]
                    return True
                return self.get_next(origin, paths_right, paths_left)
        return False

    def check_closing(self, path):
        return abs(path.vertices[0, 1] - path.vertices[-1, 1]) < self.DELTA_PREC

    def find_wrapcut_path_and_join(self, x0, x1):
        poly_solve = 0
        for collection in self.contours.collections:
            paths = collection.get_paths()
            paths_left = []
            paths_right = []
            paths_solve = []
            paths_out = list()
            # All path near meridian bounds
            for path in paths:
                x_start, x_end = path.vertices[0, 0], path.vertices[-1, 0]
                if abs(x_start - x0) < self.DELTA_PREC and abs(x_end - x0) < self.DELTA_PREC:
                    paths_left.append(path)
                elif abs(x_start - x1) < self.DELTA_PREC and abs(x_end - x1) < self.DELTA_PREC:
                    paths_right.append(path)
                else:
                    paths_out.append(path)
            if paths_left and paths_right:
                polys_to_pop_left = list()
                # Solve simple close (2 segment)
                for i_left, path_left in enumerate(paths_left):
                    for i_right, path_right in enumerate(paths_right):
                        if abs(path_left.vertices[0, 1] - path_right.vertices[-1, 1]) < self.DELTA_PREC and abs(
                                path_left.vertices[-1, 1] - path_right.vertices[0, 1]) < self.DELTA_PREC:
                            polys_to_pop_left.append(i_left)
                            path_right.vertices[:, 0] -= 360
                            path_left.vertices = concatenate((path_left.vertices, path_right.vertices))
                            path_left.vertices[-1] = path_left.vertices[0]
                            paths_solve.append(path_left)
                            paths_right.pop(i_right)
                            break
                for i in polys_to_pop_left[::-1]:
                    paths_left.pop(i)
                # Solve multiple segment:
                if paths_left and paths_right:
                    while len(paths_left):
                        origin = paths_left.pop(0)
                        if self.get_next(origin, paths_left, paths_right):
                            paths_solve.append(origin)

                poly_solve += len(paths_solve)

                paths_out.extend(paths_solve)
                paths_out.extend(paths_left)
                paths_out.extend(paths_right)
                collection._paths = paths_out
        logging.info('%d contours close over the bounds', poly_solve)

    def __init__(self, x, y, z, levels, bbox_surface_min_degree, wrap_x=False, keep_unclose=False):
        """
        c_i : index to contours
        l_i : index to levels
        """
        logging.info('Start computing iso lines')
        fig = Figure()
        ax = fig.add_subplot(111)
        if wrap_x:
            logging.debug('wrapping activate to compute contour')
            x = concatenate((x, x[:1] + 360))
            z = ma.concatenate((z, z[:1]))
        logging.debug('X shape : %s', x.shape)
        logging.debug('Y shape : %s', y.shape)
        logging.debug('Z shape : %s', z.shape)
        logging.info('Start computing iso lines with %d levels from %f to %f ...', len(levels), levels[0], levels[-1])
        self.contours = ax.contour(x, y, z.T, levels, cmap='Spectral_r')
        if wrap_x:
            self.find_wrapcut_path_and_join(x[0], x[-1])
        logging.info('Finish computing iso lines')

        nb_level = 0
        nb_contour = 0
        nb_pt = 0
        almost_closed_contours = 0
        closed_contours = 0
        # Count level and contour
        for i, collection in enumerate(self.contours.collections):
            collection.get_nearest_path_bbox_contain_pt = \
                lambda x, y, i=i: self.get_index_nearest_path_bbox_contain_pt(i, x, y)
            nb_level += 1

            keep_path = list()

            for contour in collection.get_paths():
                # Contour with less vertices than 4 are popped
                if contour.vertices.shape[0] < 4:
                    continue
                # Check if side of bbox is greater than ... => Avoid tiny shape
                x_min, y_min = contour.vertices.min(axis=0)
                x_max, y_max = contour.vertices.max(axis=0)
                d_x, d_y = x_max - x_min, y_max - y_min
                square_root = bbox_surface_min_degree ** .5
                if d_x <= square_root or d_y <= square_root:
                    continue
                if keep_unclose:
                    keep_path.append(contour)
                    continue
                # Remove unclosed path
                d_closed = ((contour.vertices[0, 0] - contour.vertices[-1, 0]) **2 + (contour.vertices[0, 1] - contour.vertices[-1, 1]) ** 2) ** .5
                if d_closed > self.DELTA_SUP:
                    continue
                elif d_closed != 0:
                    # Repair almost closed contour
                    if d_closed > self.DELTA_PREC:
                        almost_closed_contours += 1
                    else:
                        closed_contours += 1
                    contour.vertices[-1] = contour.vertices[0]
                keep_path.append(contour)
            collection._paths = keep_path
            for contour in collection.get_paths():
                contour.used = False
                nb_contour += 1
                nb_pt += contour.vertices.shape[0]
        logging.info('Repair %d closed contours and %d almost closed contours / %d contours', closed_contours, almost_closed_contours, nb_contour)
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

        # Filled array
        i_pt = 0
        i_c = 0
        i_l = 0
        for collection in self.contours.collections:
            self.level_index[i_l] = i_c
            for contour in collection.get_paths():
                nb_pt = contour.vertices.shape[0]
                # Copy pt
                self.x_value[i_pt:i_pt + nb_pt] = contour.vertices[:, 0]
                self.y_value[i_pt:i_pt + nb_pt] = contour.vertices[:, 1]

                # Set bbox
                self.x_min_per_contour[i_c], self.y_min_per_contour[i_c] = contour.vertices.min(axis=0)
                self.x_max_per_contour[i_c], self.y_max_per_contour[i_c] = contour.vertices.max(axis=0)

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

    def iter(self, start=None, stop=None, step=None):
        return self.contours.collections[slice(start, stop, step)]

    @property
    def cvalues(self):
        return self.contours.cvalues

    @property
    def levels(self):
        return self.contours.levels

    def get_index_nearest_path_bbox_contain_pt(self, level, xpt, ypt):
        """Get index from the nearest path in the level, if the bbox of the
        path contain pt
        """
        index = index_from_nearest_path_with_pt_in_bbox(
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
        if index is None:
            return None
        else:
            return self.contours.collections[level]._paths[index]

    def display(self, ax, **kwargs):
        from matplotlib.collections import LineCollection
        for collection in self.contours.collections:
            ax.add_collection(LineCollection(
                (i.vertices for i in collection.get_paths()),
                color=collection.get_color(),
                **kwargs
            ))
        ax.update_datalim([self.contours._mins, self.contours._maxs])
        ax.autoscale_view()