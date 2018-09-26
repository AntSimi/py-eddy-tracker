from ..observations import EddiesObservations as Model
from ..observations import GridDataset
from numpy import where, meshgrid, bincount, ones, unique, bool_, arange
import logging
from os import path


class CheltonTracker(Model):
    GROUND = GridDataset(path.join(path.dirname(__file__), '/data/adelepoulle/Test/Test_eddy/20180220_high_res_mask/mask_1_60.nc'), 'lon', 'lat')

    @staticmethod
    def cost_function(records_in, records_out, distance):
        """We minimize on distance between two obs
        """
        return distance

    def mask_function(self, other):
        """We mask link with ellips and ratio
        """
        # Compute Parameter of ellips
        minor, major = 1.05, 1.5
        y = self.basic_formula_ellips_major_axis(
            self.obs['lat'],
            degrees=True,
            c0=minor,
            cmin=minor,
            cmax=major,
            lat1=23,
            lat2=5,
        )
        # mask from ellips
        mask = self.shifted_ellipsoid_degrees_mask(
            other,
            minor=minor, # Minor can be bigger than major??
            major=y)

        # We check ratio
        other_amplitude, self_amplitude = meshgrid(
            other.obs['amplitude'], self.obs['amplitude'])
        other_radius, self_radius = meshgrid(
            other.obs['radius_e'], self.obs['radius_e'])
        ratio_amplitude = other_amplitude / self_amplitude
        ratio_radius = other_radius / self_radius
        mask *= \
            (ratio_amplitude < 2.5) * \
            (ratio_amplitude > 1 / 2.5) * \
            (ratio_radius < 2.5) * \
            (ratio_radius > 1 / 2.5)
        indexs_closest = where(mask)
        mask[indexs_closest] = self.across_ground(self.obs[indexs_closest[0]], other.obs[indexs_closest[1]])
        return mask

    @classmethod
    def across_ground(cls, record0, record1):
        i, j, d_pix = cls.GROUND.compute_pixel_path(
            x0=record0['lon'],
            y0=record0['lat'],
            x1=record1['lon'],
            y1=record1['lat'],
        )

        data = cls.GROUND.grid('mask')[i, j]
        i_ground = unique(arange(len(record0)).repeat(d_pix + 1)[data == 1])
        mask = ones(record1.shape, dtype='bool')
        mask[i_ground] = False
        return mask

    def solve_function(self, cost_matrix):
        """Give the best link for each self obs
        """
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
                i_keep = self.obs['amplitude'][multiple_in].argmax()
                m[where(m)[0][i_keep]] = False
                mask[m] = False

            i_self = i_self[mask]
            i_other = i_other[mask]
        return i_self, i_other
