import logging
from abc import ABC, abstractmethod

from numba import njit
from numpy import arange, int32, interp, median, zeros

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
