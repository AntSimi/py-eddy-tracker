"""
Normalised Eddy Lifetimes
=========================

Example from Evan Mason
"""
from matplotlib import pyplot as plt
from numba import njit
from numpy import interp, linspace, zeros
from py_eddy_tracker_sample import get_demo_path

from py_eddy_tracker.observations.tracking import TrackEddiesObservations


# %%
@njit(cache=True)
def sum_profile(x_new, y, out):
    """Will sum all interpolated given array"""
    out += interp(x_new, linspace(0, 1, y.size), y)


class MyObs(TrackEddiesObservations):
    def eddy_norm_lifetime(self, name, nb, factor=1):
        """
        :param str,array name: Array or field name
        :param int nb: size of output array
        """
        y = self.parse_varname(name)
        x = linspace(0, 1, nb)
        out = zeros(nb, dtype=y.dtype)
        nb_track = 0
        for i, b0, b1 in self.iter_on("track"):
            y_ = y[i]
            size_ = y_.size
            if size_ == 0:
                continue
            sum_profile(x, y_, out)
            nb_track += 1
        return x, out / nb_track * factor


# %%
# Load atlas
# ----------
kw = dict(include_vars=("speed_radius", "amplitude", "track"))
a = MyObs.load_file(
    get_demo_path("eddies_med_adt_allsat_dt2018/Anticyclonic.zarr"), **kw
)
c = MyObs.load_file(get_demo_path("eddies_med_adt_allsat_dt2018/Cyclonic.zarr"), **kw)

nb_max_a = a.nb_obs_by_track.max()
nb_max_c = c.nb_obs_by_track.max()

# %%
# Compute normalised lifetime
# ---------------------------

# Radius
AC_radius = a.eddy_norm_lifetime("speed_radius", nb=nb_max_a, factor=1e-3)
CC_radius = c.eddy_norm_lifetime("speed_radius", nb=nb_max_c, factor=1e-3)
# Amplitude
AC_amplitude = a.eddy_norm_lifetime("amplitude", nb=nb_max_a, factor=1e2)
CC_amplitude = c.eddy_norm_lifetime("amplitude", nb=nb_max_c, factor=1e2)

# %%
# Figure
# ------
fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(8, 6))

ax0.set_title("Normalised Mean Radius")
ax0.plot(*AC_radius), ax0.plot(*CC_radius)
ax0.set_ylabel("Radius (km)"), ax0.grid()
ax0.set_xlim(0, 1), ax0.set_ylim(0, None)

ax1.set_title("Normalised Mean Amplitude")
ax1.plot(*AC_amplitude, label="AC"), ax1.plot(*CC_amplitude, label="CC")
ax1.set_ylabel("Amplitude (cm)"), ax1.grid(), ax1.legend()
_ = ax1.set_xlim(0, 1), ax1.set_ylim(0, None)
