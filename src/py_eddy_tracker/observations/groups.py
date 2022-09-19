from abc import ABC, abstractmethod
import logging

from numba import njit, types as nb_types
from numpy import arange, full, int32, interp, isnan, median, where, zeros

from ..generic import window_index
from ..poly import create_meshed_particles, poly_indexs
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


def advect(x, y, c, t0, n_days, u_name="u", v_name="v"):
    """
    Advect particles from t0 to t0 + n_days, with data cube.

    :param np.array(float) x: longitude of particles
    :param np.array(float) y: latitude  of particles
    :param `~py_eddy_tracker.dataset.grid.GridCollection` c: GridCollection with speed for particles
    :param int t0: julian day of advection start
    :param int n_days: number of days to advect
    :param str u_name: variable name for u component
    :param str v_name: variable name for v component
    """

    kw = dict(nb_step=6, time_step=86400 / 6)
    if n_days < 0:
        kw["backward"] = True
        n_days = -n_days
    p = c.advect(x, y, u_name, v_name, t_init=t0, **kw)
    for _ in range(n_days):
        t, x, y = p.__next__()
    return t, x, y


def particle_candidate_step(
    t_start, contours_start, contours_end, space_step, dt, c, **kwargs
):
    """Select particles within eddies, advect them, return target observation and associated percentages.
    For one time step.

    :param int t_start: julian day of the advection
    :param (np.array(float),np.array(float)) contours_start: origin contour
    :param (np.array(float),np.array(float)) contours_end: destination contour
    :param float space_step: step between 2 particles
    :param int dt: duration of advection
    :param `~py_eddy_tracker.dataset.grid.GridCollection` c: GridCollection with speed for particles
    :params dict kwargs: dict of params given to advection
    :return (np.array,np.array): return target index and percent associate
    """
    # Create particles in start contour
    x, y, i_start = create_meshed_particles(*contours_start, space_step)
    # Advect particles
    kw = dict(nb_step=6, time_step=86400 / 6)
    p = c.advect(x, y, t_init=t_start, **kwargs, **kw)
    for _ in range(dt):
        _, x, y = p.__next__()
    m = ~(isnan(x) + isnan(y))
    i_end = full(x.shape, -1, dtype="i4")
    if m.any():
        # Id eddies for each alive particle in start contour
        i_end[m] = poly_indexs(x[m], y[m], *contours_end)
    shape = (contours_start[0].shape[0], 2)
    # Get target for each contour
    i_target, pct_target = full(shape, -1, dtype="i4"), zeros(shape, dtype="f8")
    nb_end = contours_end[0].shape[0]
    get_targets(i_start, i_end, i_target, pct_target, nb_end)
    return i_target, pct_target.astype("i1")


def particle_candidate(
    c,
    eddies,
    step_mesh,
    t_start,
    i_target,
    pct,
    contour_start="speed",
    contour_end="effective",
    **kwargs
):
    """Select particles within eddies, advect them, return target observation and associated percentages

    :param `~py_eddy_tracker.dataset.grid.GridCollection` c: GridCollection with speed for particles
    :param GroupEddiesObservations eddies: GroupEddiesObservations considered
    :param int t_start: julian day of the advection
    :param np.array(int) i_target: corresponding obs where particles are advected
    :param np.array(int) pct: corresponding percentage of avected particles
    :param str contour_start: contour where particles are injected
    :param str contour_end: contour where particles are counted after advection
    :params dict kwargs: dict of params given to `advect`

    """
    # Obs from initial time
    m_start = eddies.time == t_start
    e = eddies.extract_with_mask(m_start)

    # to be able to get global index
    translate_start = where(m_start)[0]

    # Create particles in specified contour
    intern = False if contour_start == "effective" else True
    x, y, i_start = e.create_particles(step_mesh, intern=intern)
    # Advection
    t_end, x, y = advect(x, y, c, t_start, **kwargs)

    # eddies at last date
    m_end = eddies.time == t_end / 86400
    e_end = eddies.extract_with_mask(m_end)

    # to be able to get global index
    translate_end = where(m_end)[0]

    # Id eddies for each alive particle in specified contour
    intern = False if contour_end == "effective" else True
    i_end = e_end.contains(x, y, intern=intern)

    # compute matrix and fill target array
    get_matrix(i_start, i_end, translate_start, translate_end, i_target, pct)


@njit(cache=True)
def get_targets(i_start, i_end, i_target, pct, nb_end):
    """Compute target observation and associated percentages

    :param array(int) i_start: indices in time 0
    :param array(int) i_end: indices in time N
    :param array(int) i_target: corresponding obs where particles are advected
    :param array(int) pct: corresponding percentage of avected particles
    :param int nb_end: number of contour at time N
    """
    nb_start = i_target.shape[0]
    # Matrix which will store count for every couple
    counts = zeros((nb_start, nb_end), dtype=nb_types.int32)
    # Number of particles in each origin observation
    ref = zeros(nb_start, dtype=nb_types.int32)
    # For each particle
    for i in range(i_start.size):
        i_end_ = i_end[i]
        i_start_ = i_start[i]
        ref[i_start_] += 1
        if i_end_ != -1:
            counts[i_start_, i_end_] += 1
    # From i to j
    for i in range(nb_start):
        for j in range(nb_end):
            count = counts[i, j]
            if count == 0:
                continue
            pct_ = count / ref[i] * 100
            pct_0 = pct[i, 0]
            # If percent is higher than previous stored in rank 0
            if pct_ > pct_0:
                pct[i, 1] = pct_0
                pct[i, 0] = pct_
                i_target[i, 1] = i_target[i, 0]
                i_target[i, 0] = j
            # If percent is higher than previous stored in rank 1
            elif pct_ > pct[i, 1]:
                pct[i, 1] = pct_
                i_target[i, 1] = j


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
        if self.track.size == 0:
            return
        nb_filled = mask.sum()
        logger.info("%d obs will be filled (unobserved)", nb_filled)

        nb_obs = len(self)
        index = arange(nb_obs)

        for field in self.obs.dtype.descr:
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
        """Insert virtual observations on segments where observations are missing"""

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

    def particle_candidate_atlas(
        self, cube, space_step, dt, start_intern=False, end_intern=False, **kwargs
    ):
        """Select particles within eddies, advect them, return target observation and associated percentages

        :param `~py_eddy_tracker.dataset.grid.GridCollection` cube: GridCollection with speed for particles
        :param float space_step: step between 2 particles
        :param int dt: duration of advection
        :param bool start_intern: Use intern or extern contour at injection, defaults to False
        :param bool end_intern: Use intern or extern contour at end of advection, defaults to False
        :params dict kwargs: dict of params given to advection
        :return (np.array,np.array): return target index and percent associate
        """
        t_start, t_end = int(self.period[0]), int(self.period[1])
        # Pre-compute to get time index
        i_sort, i_start, i_end = window_index(
            self.time, arange(t_start, t_end + 1), 0.5
        )
        # Out shape
        shape = (len(self), 2)
        i_target, pct = full(shape, -1, dtype="i4"), zeros(shape, dtype="i1")
        # Backward or forward
        times = arange(t_start, t_end - dt) if dt > 0 else arange(t_start + dt, t_end)
        for t in times:
            # Get index for origin
            i = t - t_start
            indexs0 = i_sort[i_start[i] : i_end[i]]
            # Get index for end
            i = t + dt - t_start
            indexs1 = i_sort[i_start[i] : i_end[i]]
            # Get contour data
            contours0 = [self[label][indexs0] for label in self.intern(start_intern)]
            contours1 = [self[label][indexs1] for label in self.intern(end_intern)]
            # Get local result
            i_target_, pct_ = particle_candidate_step(
                t, contours0, contours1, space_step, dt, cube, **kwargs
            )
            # Merge result
            m = i_target_ != -1
            i_target_[m] = indexs1[i_target_[m]]
            i_target[indexs0] = i_target_
            pct[indexs0] = pct_
        return i_target, pct
