# -*- coding: utf-8 -*-
"""
Class to compute Amplitude and average speed profile
"""

import logging

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from numba import njit
from numba import types as numba_types
from numpy import (
    array,
    concatenate,
    digitize,
    empty,
    int_,
    ma,
    ones,
    round,
    unique,
    zeros,
)

from .poly import winding_number_poly

logger = logging.getLogger("pet")


class Amplitude(object):
    """
    Class to calculate *amplitude* and counts of *local maxima/minima*
    within a closed region of a sea level anomaly field.
    """

    EPSILON = 1e-8
    __slots__ = (
        "h_0",
        "grid_extract",
        "pixel_mask",
        "nb_pixel",
        "sla",
        "contour",
        "interval_min",
        "interval_min_secondary",
        "amplitude",
        "mle",
    )

    def __init__(
        self,
        contour,
        contour_height,
        data,
        interval,
        mle=1,
        nb_step_min=2,
        nb_step_to_be_mle=2,
    ):
        """
        Create amplitude object

        :param Contours contour:
        :param float contour_height:
        :param array data:
        :param float interval:
        :param int mle: maximum number of local maxima in contour
        :param int nb_step_min: number of interval to consider like an eddy
        :param int nb_step_to_be_mle: number of interval to be consider like another maxima
        """

        # Height of the contour
        self.h_0 = contour_height
        # Step minimal to consider amplitude
        self.interval_min = interval * nb_step_min
        self.interval_min_secondary = interval * nb_step_to_be_mle
        # Indices of all pixels in contour
        self.contour = contour
        # Link on original grid (local view) or copy if it's on bound
        (x_start, x_stop), (y_start, y_stop) = contour.bbox_slice
        on_bounds = x_start > x_stop
        if on_bounds:
            self.grid_extract = ma.concatenate(
                (data[x_start:, y_start:y_stop], data[:x_stop, y_start:y_stop])
            )
            if self.grid_extract.mask.size == 1:
                self.grid_extract = ma.array(
                    self.grid_extract,
                    mask=ones(self.grid_extract.shape, dtype="bool")
                    * self.grid_extract.mask,
                )
        else:
            self.grid_extract = data[x_start:x_stop, y_start:y_stop]
        # => maybe replace pixel out of contour by nan?
        self.pixel_mask = zeros(self.grid_extract.shape, dtype="bool")
        i_x = contour.pixels_index[0] - x_start
        if on_bounds:
            i_x %= data.shape[0]

        self.pixel_mask[i_x, contour.pixels_index[1] - y_start] = True
        self.nb_pixel = i_x.shape[0]

        # Only pixel in contour
        self.sla = data[contour.pixels_index]
        # Amplitude which will be provide
        self.amplitude = 0
        # Maximum local extrema accepted
        self.mle = mle

    def within_amplitude_limits(self):
        """Need update"""
        return self.interval_min <= self.amplitude

    def all_pixels_below_h0(self, level):
        """
        Check CSS11 criterion 1: The SSH values of all of the pixels
        are below a given SSH threshold for cyclonic eddies.
        """
        # In some case pixel value must be very near of contour bounds
        if self.sla.mask.any() or ((self.sla.data - self.h_0) > self.EPSILON).any():
            return False
        else:
            # All local extrema index on th box
            lmi_i, lmi_j = detect_local_minima_(
                self.grid_extract.data,
                self.grid_extract.mask,
                self.pixel_mask,
                self.mle,
                1,
            )
            # After we use grid.data because index are in contour and we check before than no pixel are hide
            nb = len(lmi_i)
            if nb == 0:
                logger.warning(
                    "No extrema found in contour of %d pixels in level %f",
                    self.nb_pixel,
                    level,
                )
                return False
            elif nb == 1:
                i, j = lmi_i[0], lmi_j[0]
            else:
                # Verify if several extrema are seriously below contour
                nb_real_extrema = (
                    (level - self.grid_extract.data[lmi_i, lmi_j])
                    >= self.interval_min_secondary
                ).sum()
                if nb_real_extrema > self.mle:
                    return False
                index = self.grid_extract.data[lmi_i, lmi_j].argmin()
                i, j = lmi_i[index], lmi_j[index]
            self.amplitude = abs(self.grid_extract.data[i, j] - self.h_0)
            (x_start, _), (y_start, _) = self.contour.bbox_slice
            i += x_start
            j += y_start
            return i, j

    def all_pixels_above_h0(self, level):
        """
        Check CSS11 criterion 1: The SSH values of all of the pixels
        are above a given SSH threshold for anticyclonic eddies.
        """
        # In some case pixel value must be very near of contour bounds
        if self.sla.mask.any() or ((self.h_0 - self.sla.data) > self.EPSILON).any():
            return False
        else:
            # All local extrema index on th box
            lmi_i, lmi_j = detect_local_minima_(
                self.grid_extract.data,
                self.grid_extract.mask,
                self.pixel_mask,
                self.mle,
                -1,
            )
            nb = len(lmi_i)
            if nb == 0:
                logger.warning(
                    "No extrema found in contour of %d pixels in level %f",
                    self.nb_pixel,
                    level,
                )
                return False
            elif nb == 1:
                i, j = lmi_i[0], lmi_j[0]
            else:
                # Verify if several extrema are seriously above contour
                nb_real_extrema = (
                    (self.grid_extract.data[lmi_i, lmi_j] - level)
                    >= self.interval_min_secondary
                ).sum()
                if nb_real_extrema > self.mle:
                    return False
                index = self.grid_extract.data[lmi_i, lmi_j].argmax()
                i, j = lmi_i[index], lmi_j[index]
            self.amplitude = abs(self.grid_extract.data[i, j] - self.h_0)
            (x_start, _), (y_start, _) = self.contour.bbox_slice
            i += x_start
            j += y_start
            return i, j


@njit(cache=True)
def detect_local_minima_(grid, general_mask, pixel_mask, maximum_local_extremum, sign):
    """
    Take an array and detect the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e., 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    nb_x, nb_y = grid.shape
    # init with fake value because numba need to type data in list
    xs, ys = [0], [0]
    xs.pop(0)
    ys.pop(0)
    g = empty((3, 3))
    for i in range(1, nb_x - 1):
        for j in range(1, nb_y - 1):
            # Copy footprint
            for i_ in range(-1, 2):
                for j_ in range(-1, 2):
                    if general_mask[i_ + i, j_ + j]:
                        g[i_ + 1, j_ + 1] = 2e10
                    else:
                        g[i_ + 1, j_ + 1] = grid[i_ + i, j_ + j] * sign
            # if center equal to min
            # TODO if center and neigboor have same value we have problem, i don't know how
            if g.min() == (grid[i, j] * sign) and pixel_mask[i, j]:
                xs.append(i)
                ys.append(j)
    nb_extrema = len(xs)

    # If several extrema we try to separate them
    if nb_extrema > maximum_local_extremum:
        # Group
        nb_group = 1
        gr = zeros(nb_extrema, dtype=numba_types.int16)
        for i0 in range(nb_extrema - 1):
            for i1 in range(i0 + 1, nb_extrema):
                if (abs(xs[i0] - xs[i1]) + abs(ys[i0] - ys[i1])) == 1:
                    if gr[i0] == 0 and gr[i1] == 0:
                        # Nobody was link with a known group
                        gr[i0] = nb_group
                        gr[i1] = nb_group
                        nb_group += 1
                    elif gr[i0] == 0 and gr[i1] != 0:
                        # i1 is link not i0
                        gr[i0] = gr[i1]
                    elif gr[i1] == 0 and gr[i0] != 0:
                        # i0 is link not i1
                        gr[i1] = gr[i0]
                    else:
                        # there already linked in two different group
                        # we replace group from i0 with group from i1
                        gr[gr == gr[i0]] = gr[i1]

        m = gr != 0
        grs = unique(gr[m])
        # all non grouped extremum
        # Numba work around
        xs_new, ys_new = [0], [0]
        xs_new.pop(0), ys_new.pop(0)
        for i in range(nb_extrema):
            if m[i]:
                continue
            xs_new.append(xs[i])
            ys_new.append(ys[i])
        for gr_ in grs:
            nb = 0
            x_mean = 0
            y_mean = 0
            # Choose barycentre of group
            for i in range(nb_extrema):
                if gr_ == gr[i]:
                    x_mean += xs[i]
                    y_mean += ys[i]
                    nb += 1
            x_mean /= nb
            y_mean /= nb

            xs_new.append(numba_types.int32(round(x_mean)))
            ys_new.append(numba_types.int32(round(y_mean)))
        return xs_new, ys_new
    return xs, ys


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
        "contours",
        "x_value",
        "y_value",
        "contour_index",
        "level_index",
        "x_min_per_contour",
        "y_min_per_contour",
        "x_max_per_contour",
        "y_max_per_contour",
        "nb_pt_per_contour",
        "nb_contour_per_level",
    )

    DELTA_PREC = 1e-10
    DELTA_SUP = 1e-2

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
                if (
                    abs(x_start - x0) < self.DELTA_PREC
                    and abs(x_end - x0) < self.DELTA_PREC
                ):
                    paths_left.append(path)
                elif (
                    abs(x_start - x1) < self.DELTA_PREC
                    and abs(x_end - x1) < self.DELTA_PREC
                ):
                    paths_right.append(path)
                else:
                    paths_out.append(path)
            if paths_left and paths_right:
                polys_to_pop_left = list()
                # Solve simple close (2 segment)
                for i_left, path_left in enumerate(paths_left):
                    for i_right, path_right in enumerate(paths_right):
                        if (
                            abs(path_left.vertices[0, 1] - path_right.vertices[-1, 1])
                            < self.DELTA_PREC
                            and abs(
                                path_left.vertices[-1, 1] - path_right.vertices[0, 1]
                            )
                            < self.DELTA_PREC
                        ):
                            polys_to_pop_left.append(i_left)
                            path_right.vertices[:, 0] -= 360
                            path_left.vertices = concatenate(
                                (path_left.vertices, path_right.vertices[1:])
                            )
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
        logger.info("%d contours close over the bounds", poly_solve)

    def __init__(self, x, y, z, levels, wrap_x=False, keep_unclose=False):
        """
        c_i : index to contours
        l_i : index to levels
        """
        logger.info("Start computing iso lines")
        fig = Figure()
        ax = fig.add_subplot(111)
        if wrap_x:
            logger.debug("wrapping activate to compute contour")
            x = concatenate((x, x[:1] + 360))
            z = ma.concatenate((z, z[:1]))
        logger.debug("X shape : %s", x.shape)
        logger.debug("Y shape : %s", y.shape)
        logger.debug("Z shape : %s", z.shape)
        logger.info(
            "Start computing iso lines with %d levels from %f to %f ...",
            len(levels),
            levels[0],
            levels[-1],
        )
        self.contours = ax.contour(
            x, y, z.T if z.shape != x.shape else z, levels, cmap="rainbow"
        )
        if wrap_x:
            self.find_wrapcut_path_and_join(x[0], x[-1])
        logger.info("Finish computing iso lines")

        nb_level = 0
        nb_contour = 0
        nb_pt = 0
        almost_closed_contours = 0
        closed_contours = 0
        # Count level and contour
        for i, collection in enumerate(self.contours.collections):
            collection.get_nearest_path_bbox_contain_pt = (
                lambda x, y, i=i: self.get_index_nearest_path_bbox_contain_pt(i, x, y)
            )
            nb_level += 1

            keep_path = list()

            for contour in collection.get_paths():
                # Contour with less vertices than 4 are popped
                if contour.vertices.shape[0] < 4:
                    continue
                # Remove unclosed path
                d_closed = (
                    (contour.vertices[0, 0] - contour.vertices[-1, 0]) ** 2
                    + (contour.vertices[0, 1] - contour.vertices[-1, 1]) ** 2
                ) ** 0.5
                if d_closed > self.DELTA_SUP and not keep_unclose:
                    continue
                elif d_closed != 0 and d_closed <= self.DELTA_SUP:
                    # Repair almost closed contour
                    if d_closed > self.DELTA_PREC:
                        almost_closed_contours += 1
                    else:
                        closed_contours += 1
                    contour.vertices[-1] = contour.vertices[0]
                x_min, y_min = contour.vertices.min(axis=0)
                x_max, y_max = contour.vertices.max(axis=0)
                ptp_min = self.DELTA_PREC * 100
                if abs(x_min - x_max) < ptp_min or abs(y_min - y_max) < ptp_min:
                    continue
                # Store to use latter
                contour.xmin = x_min
                contour.xmax = x_max
                contour.ymin = y_min
                contour.ymax = y_max
                keep_path.append(contour)
            collection._paths = keep_path
            for contour in collection.get_paths():
                contour.used = False
                contour.reject = 0
                nb_contour += 1
                nb_pt += contour.vertices.shape[0]
        logger.info(
            "Repair %d closed contours and %d almost closed contours / %d contours",
            closed_contours,
            almost_closed_contours,
            nb_contour,
        )
        # Type for coordinates
        coord_dtype = contour.vertices.dtype

        # Array declaration
        self.x_value = empty(nb_pt, dtype=coord_dtype)
        self.y_value = empty(nb_pt, dtype=coord_dtype)

        self.level_index = empty(nb_level, dtype="u4")
        self.nb_contour_per_level = empty(nb_level, dtype="u4")

        self.nb_pt_per_contour = empty(nb_contour, dtype="u4")

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
                self.x_value[i_pt : i_pt + nb_pt] = contour.vertices[:, 0]
                self.y_value[i_pt : i_pt + nb_pt] = contour.vertices[:, 1]

                # Set bbox
                self.x_min_per_contour[i_c], self.y_min_per_contour[i_c] = (
                    contour.xmin,
                    contour.ymin,
                )
                self.x_max_per_contour[i_c], self.y_max_per_contour[i_c] = (
                    contour.xmax,
                    contour.ymax,
                )

                # Count pt
                self.nb_pt_per_contour[i_c] = nb_pt
                i_pt += nb_pt
                i_c += 1
            i_l += 1

        self.contour_index = array(
            self.nb_pt_per_contour.cumsum() - self.nb_pt_per_contour, dtype="u4"
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

        overhead of python is huge with numba, cython little bit best??
        """
        index = index_from_nearest_path_with_pt_in_bbox_(
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
            ypt,
        )
        if index == -1:
            return None
        else:
            return self.contours.collections[level]._paths[index]

    def display(
        self,
        ax,
        step=1,
        only_used=False,
        only_unused=False,
        only_contain_eddies=False,
        display_criterion=False,
        field=None,
        bins=None,
        cmap="Spectral_r",
        **kwargs
    ):
        """
        Display contour

        :param matplotlib.axes.Axes ax:
        :param int step: display only contour every step
        :param bool only_used: display only contour used in an eddy
        :param bool only_unused: display only contour unused in an eddy
        :param bool only_contain_eddies: display only contour which enclosed an eddiy
        :param bool display_criterion:
            display only unused contour with criterion color

            0. - Accepted (green)
            1. - Reject for shape error (red)
            2. - Masked value in contour (blue)
            3. - Under or over pixel limit bound (black)
            4. - Amplitude criterion (yellow)
        :param str field:
            Must be 'shape_error', 'x', 'y' or 'radius'.
            If define display_criterion is not use.
            bins argument must be define
        :param array bins: bins use to colorize contour
        :param str cmap: Name of cmap to use for field display
        :param dict kwargs: look at :py:meth:`matplotlib.collections.LineCollection`

        .. minigallery:: py_eddy_tracker.Contours.display
        """
        from matplotlib.collections import LineCollection

        overide_color = display_criterion or field is not None
        if display_criterion:
            paths = {0: list(), 1: list(), 2: list(), 3: list(), 4: list()}
        elif field is not None:
            paths = dict()
            for i in range(len(bins)):
                paths[i] = list()
            paths[i + 1] = list()
        for j, collection in enumerate(self.contours.collections[::step]):
            if not overide_color:
                paths = list()
            for i in collection.get_paths():
                if only_used and not i.used:
                    continue
                elif only_unused and i.used:
                    continue
                elif only_contain_eddies and not i.contain_eddies:
                    continue
                if display_criterion:
                    paths[i.reject].append(i.vertices)
                elif field is not None:
                    x, y, radius, shape_error = i.fit_circle()
                    if field == "shape_error":
                        i_ = digitize(shape_error, bins)
                    elif field == "radius":
                        i_ = digitize(radius, bins)
                    elif field == "x":
                        i_ = digitize(x, bins)
                    elif field == "y":
                        i_ = digitize(y, bins)
                    paths[i_].append(i.vertices)
                else:
                    paths.append(i.vertices)
            local_kwargs = kwargs.copy()
            if "color" not in kwargs:
                local_kwargs["color"] = collection.get_color()
                local_kwargs.pop("label", None)
            elif j != 0:
                local_kwargs.pop("label", None)
            if not overide_color:
                ax.add_collection(LineCollection(paths, **local_kwargs))
        if display_criterion:
            colors = {0: "g", 1: "r", 2: "b", 3: "k", 4: "y"}
            for k, v in paths.items():
                local_kwargs = kwargs.copy()
                local_kwargs.pop("label", None)
                local_kwargs["colors"] = colors[k]
                ax.add_collection(LineCollection(v, **local_kwargs))
        elif field is not None:
            nb_bins = len(bins) - 1
            cmap = get_cmap(cmap, lut=nb_bins)
            for k, v in paths.items():
                local_kwargs = kwargs.copy()
                local_kwargs.pop("label", None)
                if k == 0:
                    local_kwargs["colors"] = cmap(0.0)
                elif k > nb_bins:
                    local_kwargs["colors"] = cmap(1.0)
                else:
                    local_kwargs["colors"] = cmap((k - 1.0) / nb_bins)
                mappable = LineCollection(v, **local_kwargs)
                ax.add_collection(mappable)
            mappable.cmap = cmap
            mappable.norm = Normalize(vmin=bins[0], vmax=bins[-1])
            # TODO : need to create an object with all collections
            return mappable
        else:
            if hasattr(self.contours, "_mins"):
                ax.update_datalim([self.contours._mins, self.contours._maxs])
            ax.autoscale_view()

    def label_contour_unused_which_contain_eddies(self, eddies):
        """Select contour which contain several eddies"""
        if eddies.sign_type == 1:
            # anticyclonic
            sl = slice(None, -1)
            cor = 1
        else:
            # cyclonic
            sl = slice(1, None)
            cor = -1

        # On each level
        for j, collection in enumerate(self.contours.collections[sl]):
            # get next height
            contour_height = self.contours.cvalues[j + cor]
            # On each contour
            for i in collection.get_paths():
                i.contain_eddies = False
                if i.used:
                    continue
                nb = 0
                # try with each eddy
                for eddy in eddies:
                    if abs(eddy["height_external_contour"] - contour_height) > 1e-8:
                        continue
                    # If eddy center in contour
                    wn = winding_number_poly(
                        eddy["lon_max"], eddy["lat_max"], i.vertices
                    )
                    if wn != 0:
                        # Count
                        nb += 1

                if nb > 1:
                    i.contain_eddies = True


@njit(cache=True, fastmath=True)
def index_from_nearest_path_with_pt_in_bbox_(
    level_index,
    l_i,
    nb_c_per_l,
    nb_pt_per_c,
    indices_of_first_pts,
    x_value,
    y_value,
    x_min_per_c,
    y_min_per_c,
    x_max_per_c,
    y_max_per_c,
    xpt,
    ypt,
):
    """Get index from nearest path in edge bbox contain pt"""
    # Nb contour in level
    if nb_c_per_l[level_index] == 0:
        return -1
    # First contour in level
    i_start_c = l_i[level_index]
    # First contour of the next level
    i_end_c = i_start_c + nb_c_per_l[level_index]

    # Flag to check if we iterate
    find_contour = 0
    # We select the first pt of the first contour in the level
    # to initialize dist
    i_ref = i_start_c
    i_start_pt = indices_of_first_pts[i_start_c]
    dist_ref = (x_value[i_start_pt] - xpt) ** 2 + (y_value[i_start_pt] - ypt) ** 2

    # We iterate over contour in the same level
    for i_elt_c in range(i_start_c, i_end_c):
        # if bbox of contour doesn't contain pt, we skip this contour
        if y_min_per_c[i_elt_c] > ypt:
            continue
        if y_max_per_c[i_elt_c] < ypt:
            continue
        x_min = x_min_per_c[i_elt_c]
        xpt_ = (xpt - x_min) % 360 + x_min
        if x_min > xpt_:
            continue
        if x_max_per_c[i_elt_c] < xpt_:
            continue
        # Indice of first pt of contour
        i_start_pt = indices_of_first_pts[i_elt_c]
        # Indice of first pt of the next contour
        i_end_pt = i_start_pt + nb_pt_per_c[i_elt_c]
        # We set flag to true, because we check contour
        find_contour = 1

        # We do iteration on pt to check dist, if it's inferior we store
        # index of contour
        for i_elt_pt in range(i_start_pt, i_end_pt):
            d_x = x_value[i_elt_pt] - xpt_
            if abs(d_x) > 180:
                d_x = (d_x + 180) % 360 - 180
            dist = d_x ** 2 + (y_value[i_elt_pt] - ypt) ** 2
            if dist < dist_ref:
                dist_ref = dist
                i_ref = i_elt_c
    # No iteration on contour, we return no index of contour
    if find_contour == 0:
        return int_(-1)
    # We return index of contour, for the specific level
    return int_(i_ref - i_start_c)
