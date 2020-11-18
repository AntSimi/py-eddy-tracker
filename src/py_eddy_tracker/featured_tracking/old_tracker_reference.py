from os import path

from numba import njit
from numpy import arange, bincount, bool_, ones, unique, where

from ..dataset.grid import RegularGridDataset
from ..observations.observation import EddiesObservations as Model


class CheltonTracker(Model):

    __slots__ = tuple()

    GROUND = RegularGridDataset(
        path.join(path.dirname(__file__), "../data/mask_1_60.nc"), "lon", "lat"
    )

    @staticmethod
    def cost_function(records_in, records_out, distance):
        """We minimize on distance between two obs"""
        return distance

    def mask_function(self, other, distance):
        """We mask link with ellips and ratio"""
        # Compute Parameter of ellips
        minor, major = 1.05, 1.5
        y = self.basic_formula_ellips_major_axis(
            self.lat, degrees=True, c0=minor, cmin=minor, cmax=major, lat1=23, lat2=5
        )
        # mask from ellips
        mask = self.shifted_ellipsoid_degrees_mask(
            other, minor=minor, major=y  # Minor can be bigger than major??
        )

        # We check ratio (maybe not usefull)
        check_ratio(
            mask, self.amplitude, other.amplitude, self.radius_e, other.radius_e
        )
        indexs_closest = where(mask)
        mask[indexs_closest] = self.across_ground(
            self.obs[indexs_closest[0]], other.obs[indexs_closest[1]]
        )
        return mask

    @classmethod
    def across_ground(cls, record0, record1):
        i, j, d_pix = cls.GROUND.compute_pixel_path(
            x0=record0["lon"], y0=record0["lat"], x1=record1["lon"], y1=record1["lat"]
        )

        data = cls.GROUND.grid("mask")[i, j]
        i_ground = unique(arange(len(record0)).repeat(d_pix + 1)[data == 1])
        mask = ones(record1.shape, dtype="bool")
        mask[i_ground] = False
        return mask

    def solve_function(self, cost_matrix):
        """Give the best link for each self obs"""
        return where(self.solve_first(cost_matrix, multiple_link=True))

    def post_process_link(self, other, i_self, i_other):
        """When two self obs use the same other obs, we keep the self obs
        with amplitude max
        """
        if unique(i_other).shape[0] != i_other.shape[0]:
            nb_link = bincount(i_other)
            mask = ones(i_self.shape, dtype=bool_)
            for i in where(nb_link > 1)[0]:
                m = i == i_other
                multiple_in = i_self[m]
                i_keep = self.amplitude[multiple_in].argmax()
                m[where(m)[0][i_keep]] = False
                mask[m] = False

            i_self = i_self[mask]
            i_other = i_other[mask]
        return i_self, i_other


@njit(cache=True)
def check_ratio(
    current_mask, self_amplitude, other_amplitude, self_radius, other_radius
):
    """
    Only very few case are remove with selection

    :param current_mask:
    :param self_amplitude:
    :param other_amplitude:
    :param self_radius:
    :param other_radius:
    :return:
    """
    self_shape, other_shape = current_mask.shape
    r_min = 1 / 2.5
    r_max = 2.5
    for i in range(self_shape):
        for j in range(other_shape):
            if current_mask[i, j]:
                r_amplitude = other_amplitude[j] / self_amplitude[i]
                if r_amplitude >= r_max or r_amplitude <= r_min:
                    current_mask[i, j] = False
                    continue
                r_radius = other_radius[j] / self_radius[i]
                if r_radius >= r_max or r_radius <= r_min:
                    current_mask[i, j] = False
