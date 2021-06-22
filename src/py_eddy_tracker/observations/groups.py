import logging
from abc import ABC, abstractmethod

from numba import njit
from numba import types as nb_types
from numpy import arange, int32, interp, median, where, zeros

from .observation import EddiesObservations

logger = logging.getLogger("pet")


@njit(cache=True)
def get_missing_indices(
    array_time, array_track, dt=1, flag_untrack=True, indice_untrack=0
):
    """Return indexes where values are missing

    :param np.array(int) array_time : array of strictly increasing int representing time
    :param np.array(int) array_track: N° track where observations belong
    :param int,float dt: theorical timedelta between 2 observations
    :param bool flag_untrack: if True, ignore observations where n°track equal `indice_untrack`
    :param int indice_untrack: n° representing where observations are untracked


    ex : array_time = np.array([67, 68, 70, 71, 74, 75])
        array_track= np.array([ 1,  1,  1,  1,  1,  1])
        return : np.array([2, 4, 4])
    """

    t0 = array_time[0]
    t1 = t0

    tr0 = array_track[0]
    tr1 = tr0

    nbr_step = zeros(array_time.shape, dtype=int32)

    for i in range(array_time.size - 1):
        t0 = t1
        tr0 = tr1

        t1 = array_time[i + 1]
        tr1 = array_track[i + 1]

        if flag_untrack & (tr1 == indice_untrack):
            continue

        if tr1 != tr0:
            continue

        diff = t1 - t0
        if diff > dt:
            nbr_step[i] = int(diff / dt) - 1

    indices = zeros(nbr_step.sum(), dtype=int32)

    j = 0
    for i in range(array_time.size - 1):
        nbr_missing = nbr_step[i]

        if nbr_missing != 0:
            for k in range(nbr_missing):
                indices[j] = i + 1
                j += 1
    return indices


def advect(x, y, c, t0, n_days):
    """
    Advect particle from t0 to t0 + n_days, with data cube.

    :param np.array(float) x: longitude of particles
    :param np.array(float) y: latitude  of particles
    :param `~py_eddy_tracker.dataset.grid.GridCollection` c: GridCollection with speed for particles
    :param int t0: julian day of advection start
    :param int n_days: number of days to advect
    """

    kw = dict(nb_step=6, time_step=86400 / 6)
    if n_days < 0:
        kw["backward"] = True
        n_days = -n_days
    p = c.advect(x, y, "u", "v", t_init=t0, **kw)
    for _ in range(n_days):
        t, x, y = p.__next__()
    return t, x, y


def particle_candidate(c, eddies, step_mesh, t_start, i_target, pct, **kwargs):
    """Select particles within eddies, advect them, return target observation and associated percentages

    :param `~py_eddy_tracker.dataset.grid.GridCollection` c: GridCollection with speed for particles
    :param GroupEddiesObservations eddies: GroupEddiesObservations considered
    :param int t_start: julian day of the advection
    :param np.array(int) i_target: corresponding obs where particles are advected
    :param np.array(int) pct: corresponding percentage of avected particles
    :params dict kwargs: dict of params given to `advect`

    """
    # Obs from initial time
    m_start = eddies.time == t_start
    e = eddies.extract_with_mask(m_start)

    # to be able to get global index
    translate_start = where(m_start)[0]

    x, y, i_start = e.create_particles(step_mesh)

    # Advection
    t_end, x, y = advect(x, y, c, t_start, **kwargs)

    # eddies at last date
    m_end = eddies.time == t_end / 86400
    e_end = eddies.extract_with_mask(m_end)

    # to be able to get global index
    translate_end = where(m_end)[0]

    # Id eddies for each alive particle (in core and extern)
    i_end = e_end.contains(x, y)

    # compute matrix and fill target array
    get_matrix(i_start, i_end, translate_start, translate_end, i_target, pct)


@njit(cache=True)
def get_matrix(i_start, i_end, translate_start, translate_end, i_target, pct):
    """Compute target observation and associated percentages

    :param np.array(int) i_start: indices of associated contours at starting advection day
    :param np.array(int) i_end: indices of associated contours after advection
    :param np.array(int) translate_start: corresponding global indices at starting advection day
    :param np.array(int) translate_end: corresponding global indices after advection
    :param np.array(int) i_target: corresponding obs where particles are advected
    :param np.array(int) pct: corresponding percentage of avected particles
    """

    nb_start, nb_end = translate_start.size, translate_end.size
    # Matrix which will store count for every couple
    count = zeros((nb_start, nb_end), dtype=nb_types.int32)
    # Number of particles in each origin observation
    ref = zeros(nb_start, dtype=nb_types.int32)
    # For each particle
    for i in range(i_start.size):
        i_end_ = i_end[i]
        i_start_ = i_start[i]
        if i_end_ != -1:
            count[i_start_, i_end_] += 1
        ref[i_start_] += 1
    for i in range(nb_start):
        for j in range(nb_end):
            pct_ = count[i, j]
            # If there are particles from i to j
            if pct_ != 0:
                # Get percent
                pct_ = pct_ / ref[i] * 100.0
                # Get indices in full dataset
                i_, j_ = translate_start[i], translate_end[j]
                pct_0 = pct[i_, 0]
                if pct_ > pct_0:
                    pct[i_, 1] = pct_0
                    pct[i_, 0] = pct_
                    i_target[i_, 1] = i_target[i_, 0]
                    i_target[i_, 0] = j_
                elif pct_ > pct[i_, 1]:
                    pct[i_, 1] = pct_
                    i_target[i_, 1] = j_
    return i_target, pct


class GroupEddiesObservations(EddiesObservations, ABC):
    @abstractmethod
    def fix_next_previous_obs(self):
        pass

    @abstractmethod
    def get_missing_indices(self, dt):
        "Find indexes where observations are missing"
        pass

    def filled_by_interpolation(self, mask):
        """Fill selected values by interpolation

        :param array(bool) mask: True if must be filled by interpolation

        .. minigallery:: py_eddy_tracker.TrackEddiesObservations.filled_by_interpolation
        """

        nb_filled = mask.sum()
        logger.info("%d obs will be filled (unobserved)", nb_filled)

        nb_obs = len(self)
        index = arange(nb_obs)

        for field in self.obs.dtype.descr:
            # print(f"field : {field}")
            var = field[0]
            if (
                var in ["n", "virtual", "track", "cost_association"]
                or var in self.array_variables
            ):
                continue
            self.obs[var][mask] = interp(
                index[mask], index[~mask], self.obs[var][~mask]
            )

    def insert_virtual(self):
        """insert virtual observations on segments where observations are missing"""

        dt_theorical = median(self.time[1:] - self.time[:-1])
        indices = self.get_missing_indices(dt_theorical)

        logger.info("%d virtual observation will be added", indices.size)

        # new observations size
        size_obs_corrected = self.time.size + indices.size

        # correction of indexes for new size
        indices_corrected = indices + arange(indices.size)

        # creating mask with indexes
        mask = zeros(size_obs_corrected, dtype=bool)
        mask[indices_corrected] = 1

        new_TEO = self.new_like(self, size_obs_corrected)
        new_TEO.obs[~mask] = self.obs
        new_TEO.filled_by_interpolation(mask)
        new_TEO.virtual[:] = mask
        new_TEO.fix_next_previous_obs()
        return new_TEO

    def keep_tracks_by_date(self, date, nb_days):
        """
        Find tracks that exist at date `date` and lasted at least `nb_days` after.

        :param int,float date: date where the tracks must exist
        :param int,float nb_days: number of times the tracks must exist. Can be negative

        If nb_days is negative, it searches a track that exists at the date,
        but existed at least `nb_days` before the date
        """

        time = self.time

        mask = zeros(time.shape, dtype=bool)

        for i, b0, b1 in self.iter_on(self.tracks):
            _time = time[i]

            if date in _time and (date + nb_days) in _time:
                mask[i] = True

        return self.extract_with_mask(mask)
