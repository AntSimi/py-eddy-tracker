from ..observations.observation import EddiesObservations as Model
from numpy import ma
import logging

logger = logging.getLogger("pet")


class AreaTracker(Model):
    def tracking(self, other):
        shape = (self.shape[0], other.shape[0])
        i, j, c = self.match(other, intern=False)
        cost_mat = ma.empty(shape, dtype="f4")
        cost_mat.mask = ma.ones(shape, dtype="bool")
        m = c > .2
        i, j, c = i[m], j[m], c[m]
        cost_mat[i, j] = 1 - c

        i_self, i_other = self.solve_function(cost_mat)
        i_self, i_other = self.post_process_link(other, i_self, i_other)
        logger.debug("%d matched with previous", i_self.shape[0])
        return i_self, i_other, cost_mat[i_self, i_other]

    def propagate(self, previous_obs, current_obs, obs_to_extend, dead_track, nb_next, model):
        virtual = super().propagate(previous_obs, current_obs, obs_to_extend, dead_track, nb_next, model)
        nb_dead = len(previous_obs)
        nb_virtual_extend = nb_next - nb_dead
        for key in model.elements:
            if "contour_" not in key:
                continue
            virtual[key][:nb_dead] = current_obs[key]
            if nb_virtual_extend > 0:
                virtual[key][nb_dead:] = obs_to_extend[key]
        return virtual
