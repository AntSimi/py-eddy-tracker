import logging

from numba import njit
from numpy import empty, ma, ones

from ..observations.observation import EddiesObservations as Model

logger = logging.getLogger("pet")


class AreaTracker(Model):

    __slots__ = ("cmin",)

    def __init__(self, *args, cmin=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmin = cmin

    def merge(self, *args, **kwargs):
        eddies = super().merge(*args, **kwargs)
        eddies.cmin = self.cmin
        return eddies

    @classmethod
    def needed_variable(cls):
        vars = ["longitude", "latitude"]
        vars.extend(cls.intern(False, public_label=True))
        return vars

    def tracking(self, other):
        shape = (self.shape[0], other.shape[0])
        i, j, c = self.match(other, intern=False)
        cost_mat = ma.array(empty(shape, dtype="f4"), mask=ones(shape, dtype="bool"))
        mask_cmin(i, j, c, self.cmin, cost_mat.data, cost_mat.mask)

        i_self, i_other = self.solve_function(cost_mat)
        i_self, i_other = self.post_process_link(other, i_self, i_other)
        logger.debug("%d matched with previous", i_self.shape[0])
        return i_self, i_other, cost_mat[i_self, i_other]

    def propagate(
        self, previous_obs, current_obs, obs_to_extend, dead_track, nb_next, model
    ):
        virtual = super().propagate(
            previous_obs, current_obs, obs_to_extend, dead_track, nb_next, model
        )
        nb_dead = len(previous_obs)
        nb_virtual_extend = nb_next - nb_dead
        for key in model.elements:
            if "contour_" not in key:
                continue
            virtual[key][:nb_dead] = current_obs[key]
            if nb_virtual_extend > 0:
                virtual[key][nb_dead:] = obs_to_extend[key]
        return virtual


@njit(cache=True)
def mask_cmin(i, j, c, cmin, cost_mat, mask):
    for k in range(c.shape[0]):
        c_ = c[k]
        if c_ > cmin:
            i_, j_ = i[k], j[k]
            cost_mat[i_, j_] = 1 - c_
            mask[i_, j_] = False
