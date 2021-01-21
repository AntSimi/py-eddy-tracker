# -*- coding: utf-8 -*-
"""
Base class to manage eddy observation
"""
import logging
from datetime import datetime
from io import BufferedReader
from tarfile import ExFileObject
from tokenize import TokenError

import zarr
from matplotlib.cm import get_cmap
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from netCDF4 import Dataset
from numba import njit
from numba import types as numba_types
from numpy import (
    absolute,
    arange,
    array,
    array_equal,
    ceil,
    concatenate,
    cos,
    digitize,
    empty,
    errstate,
    floor,
    histogram,
    histogram2d,
    in1d,
    isnan,
    linspace,
    ma,
    nan,
    ndarray,
    ones,
    percentile,
    radians,
    sin,
    unique,
    where,
    zeros,
)
from pint import UnitRegistry
from pint.errors import UndefinedUnitError
from Polygon import Polygon

from .. import VAR_DESCR, VAR_DESCR_inv, __version__
from ..generic import (
    bbox_indice_regular,
    build_index,
    distance,
    distance_grid,
    flatten_line_matrix,
    hist_numba,
    local_to_coordinates,
    reverse_index,
    wrap_longitude,
)
from ..poly import (
    bbox_intersection,
    close_center,
    convexs,
    create_vertice,
    get_pixel_in_regular,
    vertice_overlap,
    winding_number_poly,
)

logger = logging.getLogger("pet")


@njit(cache=True, fastmath=True)
def shifted_ellipsoid_degrees_mask2(lon0, lat0, lon1, lat1, minor=1.5, major=1.5):
    """
    Work only if major is an array but faster * 6
    """
    # c = (major ** 2 - minor ** 2) ** .5 + major
    c = major
    major = minor + 0.5 * (major - minor)
    # r=.5*(c-c0)
    # a=c0+r
    # Focal
    f_right = lon0
    f_left = f_right - (c - minor)
    # Ellips center
    x_c = (f_left + f_right) * 0.5

    nb_0, nb_1 = lat0.shape[0], lat1.shape[0]
    m = empty((nb_0, nb_1), dtype=numba_types.bool_)

    for j in range(nb_0):
        for i in range(nb_1):
            dy = absolute(lat1[i] - lat0[j])
            if dy > minor:
                m[j, i] = False
                continue
            dx = absolute(lon1[i] - x_c[j])
            if dx > 180:
                dx = absolute((dx + 180) % 360 - 180)
            if dx > major[j]:
                m[j, i] = False
                continue
            d_normalize = dx ** 2 / major[j] ** 2 + dy ** 2 / minor ** 2
            m[j, i] = d_normalize < 1.0
    return m


class EddiesObservations(object):
    """
    Class to store eddy observations.
    """

    __slots__ = (
        "track_extra_variables",
        "track_array_variables",
        "array_variables",
        "only_variables",
        "observations",
        "sign_type",
        "raw_data",
        "period_",
    )

    ELEMENTS = [
        "lon",
        "lat",
        "radius_s",
        "radius_e",
        "amplitude",
        "speed_average",
        "time",
        "shape_error_e",
        "shape_error_s",
        "speed_area",
        "effective_area",
        "nb_contour_selected",
        "num_point_e",
        "num_point_s",
        "height_max_speed_contour",
        "height_external_contour",
        "height_inner_contour",
    ]

    COLORS = [
        "sienna",
        "red",
        "darkorange",
        "gold",
        "palegreen",
        "limegreen",
        "forestgreen",
        "mediumblue",
        "dodgerblue",
        "lightskyblue",
        "violet",
        "blueviolet",
        "darkmagenta",
        "darkgrey",
        "dimgrey",
        "steelblue",
    ]

    NB_COLORS = len(COLORS)

    def __init__(
        self,
        size=0,
        track_extra_variables=None,
        track_array_variables=0,
        array_variables=None,
        only_variables=None,
        raw_data=False,
    ):
        self.period_ = None
        self.only_variables = only_variables
        self.raw_data = raw_data
        self.track_extra_variables = (
            track_extra_variables if track_extra_variables is not None else []
        )
        self.track_array_variables = track_array_variables
        self.array_variables = array_variables if array_variables is not None else []
        for elt in self.elements:
            if elt not in VAR_DESCR:
                raise Exception("Unknown element : %s" % elt)
        self.observations = zeros(size, dtype=self.dtype)
        self.sign_type = None

    @property
    def tracks(self):
        return self.track

    def __eq__(self, other):
        if self.sign_type != other.sign_type:
            return False
        if self.dtype != other.dtype:
            return False
        return array_equal(self.obs, other.obs)

    @property
    def sign_legend(self):
        return "Cyclonic" if self.sign_type != 1 else "Anticyclonic"

    @property
    def shape(self):
        return self.observations.shape

    def get_infos(self):
        infos = dict(
            bins_lat=(-90, -60, -15, 15, 60, 90),
            bins_amplitude=array((0, 1, 2, 3, 4, 5, 10, 500)),
            bins_radius=array((0, 15, 30, 45, 60, 75, 100, 200, 2000)),
            nb_obs=self.observations.shape[0],
        )
        t0, t1 = self.period
        infos["t0"], infos["t1"] = t0, t1
        infos["period"] = t1 - t0 + 1
        return infos

    def _repr_html_(self):
        infos = self.get_infos()
        return f"""<b>{infos['nb_obs']} observations from {infos['t0']} to {infos['t1']} </b>"""

    def parse_varname(self, name):
        return self[name] if isinstance(name, str) else name

    def hist(self, varname, x, bins, percent=False, mean=False, nb=False):
        """Build histograms.

        :param str,array varname: variable to use to compute stat
        :param str,array x: variable to use to know in which bins
        :param array bins:
        :param bool percent: normalize by sum of all bins
        :param bool mean: compute mean by bins
        :param bool nb: only count by bins
        :return: value by bins
        :rtype: array
        """
        x = self.parse_varname(x)
        if nb:
            v = hist_numba(x, bins=bins)[0]
        else:
            v = histogram(x, bins=bins, weights=self.parse_varname(varname))[0]
        if percent:
            v = v.astype("f4") / v.sum() * 100
        elif mean:
            v /= hist_numba(x, bins=bins)[0]
        return v

    @staticmethod
    def box_display(value):
        """Return value evenly spaced with few numbers"""
        return "".join([f"{v_:10.2f}" for v_ in value])

    def __repr__(self):
        """
        Return general informations on dataset as strings.

        :return: informations on datasets
        :rtype: str
        """
        t0, t1 = self.period
        period = t1 - t0 + 1
        bins_lat = (-90, -60, -15, 15, 60, 90)
        bins_amplitude = array((0, 1, 2, 3, 4, 5, 10, 500))
        bins_radius = array((0, 15, 30, 45, 60, 75, 100, 200, 2000))
        nb_obs = self.observations.shape[0]

        return f"""    | {nb_obs} observations from {t0} to {t1} ({period} days, ~{nb_obs / period:.0f} obs/day)
    |   Speed area      : {self.speed_area.sum() / period / 1e12:.2f} Mkm²/day
    |   Effective area  : {self.effective_area.sum() / period / 1e12:.2f} Mkm²/day
    ----Distribution in Amplitude:
    |   Amplitude bounds (cm)  {self.box_display(bins_amplitude)}
    |   Percent of eddies         : {
        self.box_display(self.hist('time', 'amplitude', bins_amplitude / 100., percent=True, nb=True))}
    ----Distribution in Radius:
    |   Speed radius (km)      {self.box_display(bins_radius)}
    |   Percent of eddies         : {
        self.box_display(self.hist('time', 'radius_s', bins_radius * 1000., percent=True, nb=True))}
    |   Effective radius (km)  {self.box_display(bins_radius)}
    |   Percent of eddies         : {
        self.box_display(self.hist('time', 'radius_e', bins_radius * 1000., percent=True, nb=True))}
    ----Distribution in Latitude
        Latitude bounds        {self.box_display(bins_lat)}
        Percent of eddies         : {self.box_display(self.hist('time', 'lat', bins_lat, percent=True, nb=True))}
        Percent of speed area     : {self.box_display(self.hist('speed_area', 'lat', bins_lat, percent=True))}
        Percent of effective area : {self.box_display(self.hist('effective_area', 'lat', bins_lat, percent=True))}
        Mean speed radius (km)    : {self.box_display(self.hist('radius_s', 'lat', bins_lat, mean=True) / 1000.)}
        Mean effective radius (km): {self.box_display(self.hist('radius_e', 'lat', bins_lat, mean=True) / 1000.)}
        Mean amplitude (cm)       : {self.box_display(self.hist('amplitude', 'lat', bins_lat, mean=True) * 100.)}"""

    def __dir__(self):
        """Provide method name lookup and completion."""
        base = set(dir(type(self)))
        intern_name = set(self.elements)
        extern_name = set([VAR_DESCR[k]["nc_name"] for k in intern_name])
        # Must be check in init not here
        if base & intern_name:
            logger.warning(
                "Some variable name have a common name with class attrs: %s",
                base & intern_name,
            )
        if base & extern_name:
            logger.warning(
                "Some variable name have a common name with class attrs: %s",
                base & extern_name,
            )
        return sorted(base.union(intern_name).union(extern_name))

    def __getitem__(self, attr: str):
        if attr in self.elements:
            return self.obs[attr]
        elif attr in VAR_DESCR_inv:
            return self.obs[VAR_DESCR_inv[attr]]
        elif attr in ("lifetime", "age"):
            return getattr(self, attr)
        raise KeyError("%s unknown" % attr)

    def __getattr__(self, attr):
        if attr in self.elements:
            return self.obs[attr]
        elif attr in VAR_DESCR_inv:
            return self.obs[VAR_DESCR_inv[attr]]
        raise AttributeError(
            "{!r} object has no attribute {!r}".format(type(self).__name__, attr)
        )

    @classmethod
    def needed_variable(cls):
        return None

    @classmethod
    def obs_dimension(cls, handler):
        for candidate in ("obs", "Nobs", "observation", "i"):
            if candidate in handler.dimensions.keys():
                return candidate

    def add_fields(self, fields=list(), array_fields=list()):
        """
        Add a new field.
        """
        nb_obs = self.obs.shape[0]
        new = self.__class__(
            size=nb_obs,
            track_extra_variables=list(
                concatenate((self.track_extra_variables, fields))
            ),
            track_array_variables=self.track_array_variables,
            array_variables=list(concatenate((self.array_variables, array_fields))),
            only_variables=list(
                concatenate((self.obs.dtype.names, fields, array_fields))
            ),
            raw_data=self.raw_data,
        )
        new.sign_type = self.sign_type
        for field in self.obs.dtype.descr:
            logger.debug("Copy of field %s ...", field)
            var = field[0]
            new.obs[var] = self.obs[var]
        return new

    def add_rotation_type(self):
        new = self.add_fields(("type_cyc",))
        new.type_cyc[:] = self.sign_type
        return new

    def circle_contour(self, only_virtual=False):
        """
        Set contours as a circles from radius and center data.

        .. minigallery:: py_eddy_tracker.EddiesObservations.circle_contour
        """
        angle = radians(linspace(0, 360, self.track_array_variables))
        x_norm, y_norm = cos(angle), sin(angle)
        radius_s = "contour_lon_s" in self.obs.dtype.names
        radius_e = "contour_lon_e" in self.obs.dtype.names
        for i, obs in enumerate(self):
            if only_virtual and not obs["virtual"]:
                continue
            x, y = obs["lon"], obs["lat"]
            if radius_s:
                r_s = obs["radius_s"]
                obs["contour_lon_s"], obs["contour_lat_s"] = local_to_coordinates(
                    x_norm * r_s, y_norm * r_s, x, y
                )
            if radius_e:
                r_e = obs["radius_e"]
                obs["contour_lon_e"], obs["contour_lat_e"] = local_to_coordinates(
                    x_norm * r_e, y_norm * r_e, x, y
                )

    @property
    def dtype(self):
        """Return dtype to build numpy array."""
        dtype = list()
        for elt in self.elements:
            data_type = (
                VAR_DESCR[elt].get("compute_type", VAR_DESCR[elt].get("nc_type"))
                if not self.raw_data
                else VAR_DESCR[elt]["output_type"]
            )
            if elt in self.array_variables:
                dtype.append((elt, data_type, (self.track_array_variables,)))
            else:
                dtype.append((elt, data_type))
        return dtype

    @property
    def elements(self):
        """Return all the names of the variables."""
        elements = [i for i in self.ELEMENTS]
        if self.track_array_variables > 0:
            elements += self.array_variables

        if len(self.track_extra_variables):
            elements += self.track_extra_variables
        if self.only_variables is not None:
            elements = [i for i in elements if i in self.only_variables]
        return list(set(elements))

    def coherence(self, other):
        """Check coherence between two datasets."""
        test = self.track_extra_variables == other.track_extra_variables
        test *= self.track_array_variables == other.track_array_variables
        test *= self.array_variables == other.array_variables
        return test

    @classmethod
    def concatenate(cls, observations):
        nb_obs = 0
        ref_obs = observations[0]
        for obs in observations:
            if not ref_obs.coherence(obs):
                raise Exception("Merge of different type of observations")
            nb_obs += len(obs)
        eddies = cls.new_like(ref_obs, nb_obs)

        i = 0
        for obs in observations:
            nb_obs = len(obs)
            eddies.obs[i : i + nb_obs] = obs.obs
            i += nb_obs
        eddies.sign_type = ref_obs.sign_type
        return eddies

    def merge(self, other):
        """Merge two datasets."""
        nb_obs_self = len(self)
        nb_obs = nb_obs_self + len(other)
        eddies = self.new_like(self, nb_obs)
        other_keys = other.obs.dtype.fields.keys()
        self_keys = self.obs.dtype.fields.keys()
        for key in eddies.obs.dtype.fields.keys():
            eddies.obs[key][:nb_obs_self] = self.obs[key][:]
            if key in other_keys:
                eddies.obs[key][nb_obs_self:] = other.obs[key][:]
        if "track" in other_keys and "track" in self_keys:
            last_track = eddies.track[nb_obs_self - 1] + 1
            eddies.track[nb_obs_self:] += last_track
        eddies.sign_type = self.sign_type
        return eddies

    def reset(self):
        self.observations = zeros(0, dtype=self.dtype)

    @property
    def obs(self):
        """Return observations."""
        return self.observations

    def __len__(self):
        return len(self.observations)

    def __iter__(self):
        for obs in self.obs:
            yield obs

    def iter_on(self, xname, bins=None):
        """
        Yield observation group for each bin.

        :param str,array xname:
        :param array bins: bounds of each bin ,
        :return: Group observations
        :rtype: self.__class__
        """
        x = self[xname] if isinstance(xname, str) else xname
        d = x[1:] - x[:-1]
        if bins is None:
            bins = arange(x.min(), x.max() + 2)
        nb_bins = len(bins) - 1
        i = digitize(x, bins) - 1
        # Not monotonous
        if (d < 0).any():
            i_sort = i.argsort()
            i0, i1, _ = build_index(i[i_sort])
            m = ~(i0 == i1)
            i0, i1 = i0[m], i1[m]
            for i0_, i1_ in zip(i0, i1):
                i_bins = i[i_sort[i0_]]
                if i_bins == -1 or i_bins == nb_bins:
                    continue
                yield i_sort[i0_:i1_], bins[i_bins], bins[i_bins + 1]
        else:
            i0, i1, _ = build_index(i)
            m = ~(i0 == i1)
            i0, i1 = i0[m], i1[m]
            for i0_, i1_ in zip(i0, i1):
                i_bins = i[i0_]
                yield slice(i0_, i1_), bins[i_bins], bins[i_bins + 1]

    def align_on(self, other, var_name="time", **kwargs):
        """
        Align the time indexes of two datasets.
        """
        iter_self, iter_other = (
            self.iter_on(var_name, **kwargs),
            other.iter_on(var_name, **kwargs),
        )
        indexs_other, b0_other, b1_other = iter_other.__next__()
        for indexs_self, b0_self, b1_self in iter_self:
            if b0_self > b0_other:
                try:
                    while b0_other < b0_self:
                        indexs_other, b0_other, b1_other = iter_other.__next__()
                except StopIteration:
                    break
            if b0_self < b0_other:
                continue
            yield indexs_self, indexs_other, b0_self, b1_self

    def insert_observations(self, other, index):
        """Insert other obs in self at the index."""
        if not self.coherence(other):
            raise Exception("Observations with no coherence")
        insert_size = len(other.obs)
        self_size = len(self.obs)
        new_size = self_size + insert_size
        if self_size == 0:
            self.observations = other.obs
            return self
        elif insert_size == 0:
            return self
        if index < 0:
            index = self_size + index + 1
        eddies = self.new_like(self, new_size)
        eddies.obs[:index] = self.obs[:index]
        eddies.obs[index : index + insert_size] = other.obs
        eddies.obs[index + insert_size :] = self.obs[index:]
        self.observations = eddies.obs
        return self

    def append(self, other):
        """Merge."""
        return self + other

    def __add__(self, other):
        return self.insert_observations(other, -1)

    def distance(self, other):
        """Use haversine distance for distance matrix between every self and
        other eddies."""
        return distance_grid(self.lon, self.lat, other.lon, other.lat)

    def __copy__(self):
        eddies = self.new_like(self, len(self))
        for k in self.obs.dtype.names:
            eddies[k][:] = self[k][:]
        eddies.sign_type = self.sign_type
        return eddies

    def copy(self):
        return self.__copy__()

    @classmethod
    def new_like(cls, eddies, new_size: int):
        return cls(
            new_size,
            track_extra_variables=eddies.track_extra_variables,
            track_array_variables=eddies.track_array_variables,
            array_variables=eddies.array_variables,
            only_variables=eddies.only_variables,
            raw_data=eddies.raw_data,
        )

    def index(self, index, reverse=False):
        """Return obs from self at the index."""
        if reverse:
            index = reverse_index(index, len(self))
        size = 1
        if hasattr(index, "__iter__"):
            size = len(index)
        elif isinstance(index, slice):
            size = index.stop - index.start
        eddies = self.new_like(self, size)
        eddies.obs[:] = self.obs[index]
        eddies.sign_type = self.sign_type
        return eddies

    @staticmethod
    def zarr_dimension(filename):
        if isinstance(filename, zarr.storage.MutableMapping):
            h = filename
        else:
            h = zarr.open(filename)
        dims = list()
        for varname in h:
            dims.extend(list(getattr(h, varname).shape))
        return set(dims)

    @classmethod
    def load_file(cls, filename, **kwargs):
        """
        Load the netcdf or the zarr file.

        Load only latitude and longitude on the first 300 obs :

        .. code-block:: python

            kwargs_latlon_300 = dict(
                include_vars=[
                    "longitude",
                    "latitude",
                ],
                indexs=dict(obs=slice(0, 300)),
            )
            small_dataset = TrackEddiesObservations.load_file(
                filename, **kwargs_latlon_300
            )

        For `**kwargs` look at :py:meth:`load_from_zarr` or :py:meth:`load_from_netcdf`
        """
        filename_ = (
            filename.filename if isinstance(filename, ExFileObject) else filename
        )
        if isinstance(filename, zarr.storage.MutableMapping):
            return cls.load_from_zarr(filename, **kwargs)
        if isinstance(filename, (bytes, str)):
            end = b".zarr" if isinstance(filename_, bytes) else ".zarr"
            zarr_file = filename_.endswith(end)
        else:
            zarr_file = False
        if zarr_file:
            return cls.load_from_zarr(filename, **kwargs)
        else:
            return cls.load_from_netcdf(filename, **kwargs)

    @classmethod
    def load_from_zarr(
        cls,
        filename,
        raw_data=False,
        remove_vars=None,
        include_vars=None,
        indexs=None,
        buffer_size=5000000,
        **class_kwargs,
    ):
        """Load data from zarr.

        :param str,store filename: path or store to load data
        :param bool raw_data: If true load data without apply scale_factor and add_offset
        :param None,list(str) remove_vars: List of variable name which will be not loaded
        :param None,list(str) include_vars: If defined only this variable will be loaded
        :param None,dict indexs: Indexs to laad only a slice of data
        :param int buffer_size: Size of buffer used to load zarr data
        :param class_kwargs: argument to set up observations class
        :return: Obsevations selected
        :return type: class
        """
        # FIXME
        array_dim = -1
        if isinstance(filename, zarr.storage.MutableMapping):
            h_zarr = filename
        else:
            if not isinstance(filename, str):
                filename = filename.astype(str)
            h_zarr = zarr.open(filename)
        var_list = cls.build_var_list(list(h_zarr.keys()), remove_vars, include_vars)

        nb_obs = getattr(h_zarr, var_list[0]).shape[0]
        dims = list(cls.zarr_dimension(filename))
        if len(dims) == 2 and nb_obs in dims:
            # FIXME must be investigate, in zarr no dimensions name (or could be add in attr)
            array_dim = dims[1] if nb_obs == dims[0] else dims[0]
        if indexs is not None and "obs" in indexs:
            sl = indexs["obs"]
            sl = slice(sl.start, min(sl.stop, nb_obs))
            if sl.stop is not None:
                nb_obs = sl.stop
            if sl.start is not None:
                nb_obs -= sl.start
            if sl.step is not None:
                indexs["obs"] = slice(sl.start, sl.stop)
                logger.warning("step of slice won't be use")
        logger.debug("%d observations will be load", nb_obs)
        kwargs = dict()

        if array_dim in dims:
            kwargs["track_array_variables"] = array_dim
            kwargs["array_variables"] = list()
            for variable in var_list:
                if array_dim in h_zarr[variable].shape:
                    var_inv = VAR_DESCR_inv[variable]
                    kwargs["array_variables"].append(var_inv)
        array_variables = kwargs.get("array_variables", list())
        kwargs["track_extra_variables"] = []
        for variable in var_list:
            var_inv = VAR_DESCR_inv[variable]
            if var_inv not in cls.ELEMENTS and var_inv not in array_variables:
                kwargs["track_extra_variables"].append(var_inv)
        kwargs["raw_data"] = raw_data
        kwargs["only_variables"] = (
            None if include_vars is None else [VAR_DESCR_inv[i] for i in include_vars]
        )
        kwargs.update(class_kwargs)
        eddies = cls(size=nb_obs, **kwargs)
        for variable in var_list:
            var_inv = VAR_DESCR_inv[variable]
            logger.debug("%s will be loaded", variable)
            # find unit factor
            input_unit = h_zarr[variable].attrs.get("unit", None)
            if input_unit is None:
                input_unit = h_zarr[variable].attrs.get("units", None)
            output_unit = VAR_DESCR[var_inv]["nc_attr"].get("units", None)
            factor = cls.compare_units(input_unit, output_unit, variable)
            sl_obs = slice(None) if indexs is None else indexs.get("obs", slice(None))
            scale_factor = VAR_DESCR[var_inv].get("scale_factor", None)
            add_offset = VAR_DESCR[var_inv].get("add_offset", None)
            cls.copy_data_to_zarr(
                h_zarr[variable],
                eddies.obs[var_inv],
                sl_obs,
                buffer_size,
                factor,
                raw_data,
                scale_factor,
                add_offset,
            )

        eddies.sign_type = int(h_zarr.attrs.get("rotation_type", 0))
        if eddies.sign_type == 0:
            logger.debug("File come from another algorithm of identification")
            eddies.sign_type = -1
        return eddies

    @staticmethod
    def copy_data_to_zarr(
        handler_zarr,
        handler_eddies,
        sl_obs,
        buffer_size,
        factor,
        raw_data,
        scale_factor,
        add_offset,
    ):
        """
        Copy with buffer for zarr.

        Zarr need to get real value, and size could be huge, so we use a buffer to manage memory
        :param zarr_dataset handler_zarr:
        :param array handler_eddies:
        :param slice zarr_dataset sl_obs:
        :param int zarr_dataset buffer_size:
        :param float zarr_dataset factor:
        :param bool zarr_dataset raw_data:
        :param None,float zarr_dataset scale_factor:
        :param None,float add_offset:
        """
        i_start, i_stop = sl_obs.start, sl_obs.stop
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = handler_zarr.shape[0]
        for i in range(i_start, i_stop, buffer_size):
            sl_in = slice(i, min(i + buffer_size, i_stop))
            data = handler_zarr[sl_in]
            if factor != 1:
                data *= factor
            if raw_data:
                if add_offset is not None:
                    data -= add_offset
                if scale_factor is not None:
                    data /= scale_factor
            sl_out = slice(i - i_start, i - i_start + buffer_size)
            handler_eddies[sl_out] = data

    @classmethod
    def load_from_netcdf(
        cls,
        filename,
        raw_data=False,
        remove_vars=None,
        include_vars=None,
        indexs=None,
        **class_kwargs,
    ):
        """Load data from netcdf.

        :param str,ExFileObject filename: path or handler to load data
        :param bool raw_data: If true load data without apply scale_factor and add_offset
        :param None,list(str) remove_vars: List of variable name which will be not loaded
        :param None,list(str) include_vars: If defined only this variable will be loaded
        :param None,dict indexs: Indexs to laad only a slice of data
        :param class_kwargs: argument to set up observations class
        :return: Obsevations selected
        :return type: class
        """
        array_dim = "NbSample"
        if isinstance(filename, bytes):
            filename = filename.astype(str)
        if isinstance(filename, (ExFileObject, BufferedReader)):
            filename.seek(0)
            args, kwargs = ("in-mem-file",), dict(memory=filename.read())
        else:
            args, kwargs = (filename,), dict()
        with Dataset(*args, **kwargs) as h_nc:
            var_list = cls.build_var_list(
                list(h_nc.variables.keys()), remove_vars, include_vars
            )

            obs_dim = cls.obs_dimension(h_nc)
            nb_obs = len(h_nc.dimensions[obs_dim])
            if indexs is not None and obs_dim in indexs:
                sl = indexs[obs_dim]
                sl = slice(sl.start, min(sl.stop, nb_obs))
                if sl.stop is not None:
                    nb_obs = sl.stop
                if sl.start is not None:
                    nb_obs -= sl.start
                if sl.step is not None:
                    indexs[obs_dim] = slice(sl.start, sl.stop)
                    logger.warning("step of slice won't be use")
            logger.debug("%d observations will be load", nb_obs)
            kwargs = dict()
            if array_dim in h_nc.dimensions:
                kwargs["track_array_variables"] = len(h_nc.dimensions[array_dim])
                kwargs["array_variables"] = list()
                for variable in var_list:
                    if array_dim in h_nc.variables[variable].dimensions:
                        var_inv = VAR_DESCR_inv[variable]
                        kwargs["array_variables"].append(var_inv)
            array_variables = kwargs.get("array_variables", list())
            kwargs["track_extra_variables"] = []
            for variable in var_list:
                var_inv = VAR_DESCR_inv[variable]
                if var_inv not in cls.ELEMENTS and var_inv not in array_variables:
                    kwargs["track_extra_variables"].append(var_inv)
            kwargs["raw_data"] = raw_data
            kwargs["only_variables"] = (
                None
                if include_vars is None
                else [VAR_DESCR_inv[i] for i in include_vars]
            )
            kwargs.update(class_kwargs)
            eddies = cls(size=nb_obs, **kwargs)
            for variable in var_list:
                var_inv = VAR_DESCR_inv[variable]
                # Patch
                h_nc.variables[variable].set_auto_maskandscale(not raw_data)
                logger.debug(
                    "Up load %s variable%s",
                    variable,
                    ", with raw mode" if raw_data else "",
                )
                # find unit factor
                factor = 1
                if not raw_data:
                    input_unit = getattr(h_nc.variables[variable], "unit", None)
                    if input_unit is None:
                        input_unit = getattr(h_nc.variables[variable], "units", None)
                    output_unit = VAR_DESCR[var_inv]["nc_attr"].get("units", None)
                    factor = cls.compare_units(input_unit, output_unit, variable)
                if indexs is None:
                    indexs = dict()
                var_sl = [
                    indexs.get(dim, slice(None))
                    for dim in h_nc.variables[variable].dimensions
                ]
                if factor != 1:
                    eddies.obs[var_inv] = h_nc.variables[variable][var_sl] * factor
                else:
                    eddies.obs[var_inv] = h_nc.variables[variable][var_sl]

            for variable in var_list:
                var_inv = VAR_DESCR_inv[variable]
                if var_inv == "type_cyc":
                    eddies.sign_type = h_nc.variables[variable][0]
            if eddies.sign_type is None:
                title = getattr(h_nc, "title", None)
                if title is None:
                    eddies.sign_type = getattr(h_nc, "rotation_type", 0)
                else:
                    eddies.sign_type = -1 if title == "Cyclonic" else 1
            if eddies.sign_type == 0:
                logger.debug("File come from another algorithm of identification")
                eddies.sign_type = -1

        return eddies

    @staticmethod
    def build_var_list(var_list, remove_vars, include_vars):
        if include_vars is not None:
            var_list = [i for i in var_list if i in include_vars]
        elif remove_vars is not None:
            var_list = [i for i in var_list if i not in remove_vars]
        return var_list

    @staticmethod
    def compare_units(input_unit, output_unit, name):
        if output_unit is None or input_unit is None or output_unit == input_unit:
            return 1
        units = UnitRegistry()
        try:
            input_unit = units.parse_expression(input_unit, case_sensitive=False)
            output_unit = units.parse_expression(output_unit, case_sensitive=False)
        except UndefinedUnitError:
            input_unit = None
        except TokenError:
            input_unit = None
        if input_unit is not None:
            factor = input_unit.to(output_unit).to_tuple()[0]
            # If we are able to find a conversion
            if factor != 1:
                logger.info(
                    "%s will be multiply by %f to take care of units(%s->%s)",
                    name,
                    factor,
                    input_unit,
                    output_unit,
                )

    @classmethod
    def from_zarr(cls, handler):
        nb_obs = len(handler.dimensions[cls.obs_dimension(handler)])
        kwargs = dict()
        if hasattr(handler, "track_array_variables"):
            kwargs["track_array_variables"] = handler.track_array_variables
            kwargs["array_variables"] = handler.array_variables.split(",")
        if len(handler.track_extra_variables) > 1:
            kwargs["track_extra_variables"] = handler.track_extra_variables.split(",")
        eddies = cls(size=nb_obs, **kwargs)
        for variable in handler.variables:
            # Patch
            if variable == "time":
                eddies.obs[variable] = handler.variables[variable][:]
            else:
                eddies.obs[VAR_DESCR_inv[variable]] = handler.variables[variable][:]
        return eddies

    @classmethod
    def from_netcdf(cls, handler):
        nb_obs = len(handler.dimensions[cls.obs_dimension(handler)])
        kwargs = dict()
        if hasattr(handler, "track_array_variables"):
            kwargs["track_array_variables"] = handler.track_array_variables
            kwargs["array_variables"] = handler.array_variables.split(",")
        if len(handler.track_extra_variables) > 1:
            kwargs["track_extra_variables"] = handler.track_extra_variables.split(",")
        eddies = cls(size=nb_obs, **kwargs)
        for variable in handler.variables:
            # Patch
            if variable == "time":
                eddies.obs[variable] = handler.variables[variable][:]
            else:
                eddies.obs[VAR_DESCR_inv[variable]] = handler.variables[variable][:]
        return eddies

    def propagate(
        self, previous_obs, current_obs, obs_to_extend, dead_track, nb_next, model
    ):
        """
        Filled virtual obs (C).

        :param previous_obs: previous obs from current (A)
        :param current_obs: previous obs from virtual (B)
        :param obs_to_extend:
        :param dead_track:
        :param nb_next:
        :param model:

        :return: New position C = B + AB
        """
        next_obs = VirtualEddiesObservations(
            size=nb_next,
            track_extra_variables=model.track_extra_variables,
            track_array_variables=model.track_array_variables,
            array_variables=model.array_variables,
        )
        next_obs.sign_type = self.sign_type
        nb_dead = len(previous_obs)
        nb_virtual_extend = nb_next - nb_dead

        for key in model.elements:
            if key in ["lon", "lat", "time"] or "contour_" in key:
                continue
            next_obs[key][:nb_dead] = current_obs[key]
        next_obs["dlon"][:nb_dead] = current_obs["lon"] - previous_obs["lon"]
        next_obs["dlat"][:nb_dead] = current_obs["lat"] - previous_obs["lat"]
        next_obs["lon"][:nb_dead] = current_obs["lon"] + next_obs["dlon"][:nb_dead]
        next_obs["lat"][:nb_dead] = current_obs["lat"] + next_obs["dlat"][:nb_dead]
        # Id which are extended
        next_obs["track"][:nb_dead] = dead_track
        # Add previous virtual
        if nb_virtual_extend > 0:
            for key in next_obs.elements:
                if (
                    key in ["lon", "lat", "time", "track", "segment_size"]
                    or "contour_" in key
                ):
                    continue
                next_obs[key][nb_dead:] = obs_to_extend[key]
            next_obs["lon"][nb_dead:] = obs_to_extend["lon"] + obs_to_extend["dlon"]
            next_obs["lat"][nb_dead:] = obs_to_extend["lat"] + obs_to_extend["dlat"]
            next_obs["track"][nb_dead:] = obs_to_extend["track"]
            next_obs["segment_size"][nb_dead:] = obs_to_extend["segment_size"]
        # Count
        next_obs["segment_size"][:] += 1
        return next_obs

    @staticmethod
    def intern(flag, public_label=False):
        if flag:
            labels = "contour_lon_s", "contour_lat_s"
        else:
            labels = "contour_lon_e", "contour_lat_e"
        if public_label:
            labels = [VAR_DESCR[label]["nc_name"] for label in labels]
        return labels

    def match(self, other, method="overlap", intern=False, cmin=0, **kwargs):
        """Return index and score computed on the effective contour.

        :param EddiesObservations other: Observations to compare
        :param str method:
            - "overlap": the score is computed with contours;
            - "circle": circles are computed and used for score (TODO)
        :param bool intern: if True, speed contour is used (default = effective contour)
        :param float cmin: 0 < cmin < 1, return only couples with score >= cmin
        :param dict kwargs: look at :py:meth:`vertice_overlap`
        :return: return the indexes of the eddies in self coupled with eddies in
            other and their associated score
        :rtype: (array(int), array(int), array(float))

        .. minigallery:: py_eddy_tracker.EddiesObservations.match
        """
        # if method is "overlap" method will use contour to compute score,
        # if method is "circle" method will apply a formula of circle overlap
        x_name, y_name = self.intern(intern)
        if method == "overlap":
            i, j = bbox_intersection(
                self[x_name], self[y_name], other[x_name], other[y_name]
            )
            c = vertice_overlap(
                self[x_name][i],
                self[y_name][i],
                other[x_name][j],
                other[y_name][j],
                **kwargs,
            )
        elif method == "close_center":
            i, j, c = close_center(
                self.latitude, self.longitude, other.latitude, other.longitude, **kwargs
            )

        m = c >= cmin  # ajout >= pour garder la cmin dans la sélection
        return i[m], j[m], c[m]

    @classmethod
    def cost_function_common_area(cls, xy_in, xy_out, distance, intern=False):
        """How does it work on x bound ?

        :param xy_in:
        :param xy_out:
        :param distance:
        :param bool intern:

        """
        x_name, y_name = cls.intern(intern)
        nb_records = xy_in.shape[0]
        x_in, y_in = xy_in[x_name], xy_in[y_name]
        x_out, y_out = xy_out[x_name], xy_out[y_name]
        x_in_min, y_in_min = x_in.min(axis=1), y_in.min(axis=1)
        x_in_max, y_in_max = x_in.max(axis=1), y_in.max(axis=1)
        x_out_min, y_out_min = x_out.min(axis=1), y_out.min(axis=1)
        x_out_max, y_out_max = x_out.max(axis=1), y_out.max(axis=1)
        costs = ma.empty(nb_records, dtype="f4")

        for i in range(nb_records):
            if x_in_max[i] < x_out_min[i] or x_in_min[i] > x_out_max[i]:
                costs[i] = 1
                continue
            if y_in_max[i] < y_out_min[i] or y_in_min[i] > y_out_max[i]:
                costs[i] = 1
                continue

            x_in_, x_out_ = x_in[i], x_out[i]
            p_in = Polygon(create_vertice(x_in_, y_in[i]))
            if abs(x_in_[0] - x_out_[0]) > 180:
                x_out_ = (x_out[i] - (x_in_[0] - 180)) % 360 + x_in_[0] - 180
            p_out = Polygon(create_vertice(x_out_, y_out[i]))
            costs[i] = 1 - (p_in & p_out).area() / min(p_in.area(), p_out.area())
        costs.mask = costs == 1
        return costs

    def mask_function(self, other, distance):
        return distance < 125

    @staticmethod
    def cost_function(records_in, records_out, distance):
        r"""Return the cost function between two obs.

        .. math::

            cost = \sqrt{({Amp_{_{in}} - Amp_{_{out}} \over Amp_{_{in}}}) ^2 +
              ({Rspeed_{_{in}} - Rspeed_{_{out}} \over Rspeed_{_{in}}}) ^2 +
              ({distance \over 125}) ^2
            }

        :param records_in: starting observations
        :param records_out: observations to associate
        :param distance: computed between in and out

        """
        cost = (
            (records_in["amplitude"] - records_out["amplitude"])
            / records_in["amplitude"]
        ) ** 2
        cost += (
            (records_in["radius_s"] - records_out["radius_s"]) / records_in["radius_s"]
        ) ** 2
        cost += (distance / 125) ** 2
        cost **= 0.5
        # Mask value superior at 60 % of variation
        # return ma.array(cost, mask=m)
        return cost

    def shifted_ellipsoid_degrees_mask(self, other, minor=1.5, major=1.5):
        return shifted_ellipsoid_degrees_mask2(
            self.lon, self.lat, other.lon, other.lat, minor, major
        )

    def fixed_ellipsoid_mask(
        self, other, minor=50, major=100, only_east=False, shifted_ellips=False
    ):
        dist = self.distance(other).T
        accepted = dist < minor
        rejected = dist > major
        rejected += isnan(dist)

        # All obs we are not in rejected and accepted, there are between
        # two circle
        needs_investigation = -(rejected + accepted)
        index_other, index_self = where(needs_investigation)

        nb_case = index_self.shape[0]
        if nb_case != 0:
            if isinstance(major, ndarray):
                major = major[index_self]
            if isinstance(minor, ndarray):
                minor = minor[index_self]
            # focal distance
            f_degree = ((major ** 2 - minor ** 2) ** 0.5) / (
                111.2 * cos(radians(self.lat[index_self]))
            )

            lon_self = self.lon[index_self]
            if shifted_ellips:
                x_center_ellips = lon_self - (major - minor) / 2
            else:
                x_center_ellips = lon_self

            lon_left_f = x_center_ellips - f_degree
            lon_right_f = x_center_ellips + f_degree

            dist_left_f = distance(
                lon_left_f,
                self.lat[index_self],
                other.lon[index_other],
                other.lat[index_other],
            )
            dist_right_f = distance(
                lon_right_f,
                self.lat[index_self],
                other.lon[index_other],
                other.lat[index_other],
            )
            dist_2a = (dist_left_f + dist_right_f) / 1000

            accepted[index_other, index_self] = dist_2a < (2 * major)
            if only_east:
                d_lon = (other.lon[index_other] - lon_self + 180) % 360 - 180
                mask = d_lon < 0
                accepted[index_other[mask], index_self[mask]] = False
        return accepted.T

    @staticmethod
    def basic_formula_ellips_major_axis(
        lats, cmin=1.5, cmax=10.0, c0=1.5, lat1=13.5, lat2=5.0, degrees=False
    ):
        """Give major axis in km with a given latitude"""
        # Straight line between lat1 and lat2:
        # y = a * x + b
        a = (cmin - cmax) / (lat1 - lat2)
        b = a * -lat1 + cmin

        abs_lats = abs(lats)
        major_axis = ones(lats.shape, dtype="f8") * cmin
        major_axis[abs_lats < lat2] = cmax
        m = abs_lats > lat1
        m += abs_lats < lat2

        major_axis[~m] = a * abs_lats[~m] + b
        if not degrees:
            major_axis *= 111.2
        return major_axis

    @staticmethod
    def solve_conflict(cost):
        pass

    @staticmethod
    def solve_simultaneous(cost):
        """Write something (TODO)"""
        mask = ~cost.mask
        # Count number of link by self obs and other obs
        self_links, other_links = sum_row_column(mask)
        max_links = max(self_links.max(), other_links.max())
        if max_links > 5:
            logger.warning("One observation have %d links", max_links)

        # If some obs have multiple link, we keep only one link by eddy
        eddies_separation = 1 < self_links
        eddies_merge = 1 < other_links
        test = eddies_separation.any() or eddies_merge.any()
        if test:
            # We extract matrix which contains concflict
            obs_linking_to_self = mask[eddies_separation].any(axis=0)
            obs_linking_to_other = mask[:, eddies_merge].any(axis=1)
            i_self_keep = where(obs_linking_to_other + eddies_separation)[0]
            i_other_keep = where(obs_linking_to_self + eddies_merge)[0]

            # Cost to resolve conflict
            cost_reduce = cost[i_self_keep][:, i_other_keep]
            shape = cost_reduce.shape
            nb_conflict = (~cost_reduce.mask).sum()
            logger.debug("Shape conflict matrix : %s, %d conflicts", shape, nb_conflict)

            if nb_conflict >= (shape[0] + shape[1]):
                logger.warning(
                    "High number of conflict : %d (nb_conflict)", shape[0] + shape[1]
                )

            links_resolve = 0
            # Arbitrary value
            max_iteration = max(cost_reduce.shape)
            security_increment = 0
            while False in cost_reduce.mask:
                if security_increment > max_iteration:
                    # Maybe check if the size decrease if not rise an exception
                    # x_i, y_i = where(-cost_reduce.mask)
                    raise Exception("To many iteration: %d" % security_increment)
                security_increment += 1
                i_min_value = cost_reduce.argmin()
                i, j = floor(i_min_value / shape[1]).astype(int), i_min_value % shape[1]
                # Set to False all link
                mask[i_self_keep[i]] = False
                mask[:, i_other_keep[j]] = False
                cost_reduce.mask[i] = True
                cost_reduce.mask[:, j] = True
                # we active only this link
                mask[i_self_keep[i], i_other_keep[j]] = True
                links_resolve += 1
            logger.debug("%d links resolve", links_resolve)
        return mask

    @staticmethod
    def solve_first(cost, multiple_link=False):
        mask = ~cost.mask
        # Count number of link by self obs and other obs
        self_links = mask.sum(axis=1)
        other_links = mask.sum(axis=0)
        max_links = max(self_links.max(), other_links.max())
        if max_links > 5:
            logger.warning("One observation have %d links", max_links)

        # If some obs have multiple link, we keep only one link by eddy
        eddies_separation = 1 < self_links
        eddies_merge = 1 < other_links
        test = eddies_separation.any() or eddies_merge.any()
        if test:
            # We extract matrix which contains concflict
            obs_linking_to_self = mask[eddies_separation].any(axis=0)
            obs_linking_to_other = mask[:, eddies_merge].any(axis=1)
            i_self_keep = where(obs_linking_to_other + eddies_separation)[0]
            i_other_keep = where(obs_linking_to_self + eddies_merge)[0]

            # Cost to resolve conflict
            cost_reduce = cost[i_self_keep][:, i_other_keep]
            shape = cost_reduce.shape
            nb_conflict = (~cost_reduce.mask).sum()
            logger.debug("Shape conflict matrix : %s, %d conflicts", shape, nb_conflict)

            if nb_conflict >= (shape[0] + shape[1]):
                logger.warning(
                    "High number of conflict : %d (nb_conflict)", shape[0] + shape[1]
                )

            links_resolve = 0
            for i in range(shape[0]):
                j = cost_reduce[i].argmin()
                if hasattr(cost_reduce[i, j], "mask"):
                    continue
                links_resolve += 1
                # Set all links to False
                mask[i_self_keep[i]] = False
                cost_reduce.mask[i] = True
                if not multiple_link:
                    mask[:, i_other_keep[j]] = False
                    cost_reduce.mask[:, j] = True
                # We activate this link only
                mask[i_self_keep[i], i_other_keep[j]] = True

            logger.debug("%d links resolve", links_resolve)
        return mask

    def solve_function(self, cost_matrix):
        return numba_where(self.solve_simultaneous(cost_matrix))

    def post_process_link(self, other, i_self, i_other):
        if unique(i_other).shape[0] != i_other.shape[0]:
            raise Exception()
        return i_self, i_other

    def tracking(self, other):
        """Track obs between self and other"""
        dist = self.distance(other)
        mask_accept_dist = self.mask_function(other, dist)
        indexs_closest = where(mask_accept_dist)

        cost_values = self.cost_function(
            self.obs[indexs_closest[0]],
            other.obs[indexs_closest[1]],
            dist[mask_accept_dist],
        )

        cost_mat = ma.empty(mask_accept_dist.shape, dtype="f4")
        cost_mat.mask = ~mask_accept_dist
        cost_mat[mask_accept_dist] = cost_values

        i_self, i_other = self.solve_function(cost_mat)

        i_self, i_other = self.post_process_link(other, i_self, i_other)

        logger.debug("%d matched with previous", i_self.shape[0])

        return i_self, i_other, cost_mat[i_self, i_other]

    def to_zarr(self, handler, **kwargs):
        handler.attrs["track_extra_variables"] = ",".join(self.track_extra_variables)
        if self.track_array_variables != 0:
            handler.attrs["track_array_variables"] = self.track_array_variables
            handler.attrs["array_variables"] = ",".join(self.array_variables)
        # Iter on variables to create:
        fields = [field[0] for field in self.observations.dtype.descr]
        for ori_name in fields:
            # Patch for a transition
            name = ori_name
            #
            logger.debug("Create Variable %s", VAR_DESCR[name]["nc_name"])
            self.create_variable_zarr(
                handler,
                dict(
                    name=VAR_DESCR[name]["nc_name"],
                    store_dtype=VAR_DESCR[name]["output_type"],
                    dtype=VAR_DESCR[name]["nc_type"],
                    dimensions=VAR_DESCR[name]["nc_dims"],
                ),
                VAR_DESCR[name]["nc_attr"],
                self.obs[ori_name],
                scale_factor=VAR_DESCR[name].get("scale_factor", None),
                add_offset=VAR_DESCR[name].get("add_offset", None),
                filters=VAR_DESCR[name].get("filters", None),
                **kwargs,
            )
        self.set_global_attr_zarr(handler)

    @staticmethod
    def netcdf_create_dimensions(handler, dim, nb):
        if dim not in handler.dimensions:
            handler.createDimension(dim, nb)
        else:
            old_nb = len(handler.dimensions[dim])
            if nb != old_nb:
                raise Exception(
                    f"{dim} dimensions previously set to a different size {old_nb} (current value : {nb})"
                )

    def to_netcdf(self, handler, **kwargs):
        eddy_size = len(self)
        logger.debug('Create Dimensions "obs" : %d', eddy_size)
        self.netcdf_create_dimensions(handler, "obs", eddy_size)
        handler.track_extra_variables = ",".join(self.track_extra_variables)
        if self.track_array_variables != 0:
            self.netcdf_create_dimensions(
                handler, "NbSample", self.track_array_variables
            )
            handler.track_array_variables = self.track_array_variables
            handler.array_variables = ",".join(self.array_variables)
        # Iter on variables to create:
        fields = [field[0] for field in self.observations.dtype.descr]
        fields_ = array(
            [VAR_DESCR[field[0]]["nc_name"] for field in self.observations.dtype.descr]
        )
        i = fields_.argsort()
        for ori_name in array(fields)[i]:
            # Patch for a transition
            name = ori_name
            #
            logger.debug("Create Variable %s", VAR_DESCR[name]["nc_name"])
            self.create_variable(
                handler,
                dict(
                    varname=VAR_DESCR[name]["nc_name"],
                    datatype=VAR_DESCR[name]["output_type"],
                    dimensions=VAR_DESCR[name]["nc_dims"],
                ),
                VAR_DESCR[name]["nc_attr"],
                self.obs[ori_name],
                scale_factor=VAR_DESCR[name].get("scale_factor", None),
                add_offset=VAR_DESCR[name].get("add_offset", None),
                **kwargs,
            )
        self.set_global_attr_netcdf(handler)

    def create_variable(
        self,
        handler_nc,
        kwargs_variable,
        attr_variable,
        data,
        scale_factor=None,
        add_offset=None,
        **kwargs,
    ):
        dims = kwargs_variable.get("dimensions", None)
        # Manage chunk in 2d case
        if dims is not None and len(dims) > 1:
            chunk = [1]
            cum = 1
            for dim in dims[1:]:
                nb = len(handler_nc.dimensions[dim])
                chunk.append(nb)
                cum *= nb
            chunk[0] = min(int(400000 / cum), len(handler_nc.dimensions[dims[0]]))
            kwargs_variable["chunksizes"] = chunk
        kwargs_variable["zlib"] = True
        kwargs_variable["complevel"] = 1
        kwargs_variable.update(kwargs)
        var = handler_nc.createVariable(**kwargs_variable)
        attrs = list(attr_variable.keys())
        attrs.sort()
        for attr in attrs:
            attr_value = attr_variable[attr]
            var.setncattr(attr, attr_value)
        if self.raw_data:
            var[:] = data
        if scale_factor is not None:
            var.scale_factor = scale_factor
            if add_offset is not None:
                var.add_offset = add_offset
            else:
                var.add_offset = 0
        if not self.raw_data:
            var[:] = data
        try:
            if len(var.dimensions) == 1 or var.size < 1e7:
                var.setncattr("min", var[:].min())
                var.setncattr("max", var[:].max())
        except ValueError:
            logger.warning("Data is empty")

    def create_variable_zarr(
        self,
        handler_zarr,
        kwargs_variable,
        attr_variable,
        data,
        scale_factor=None,
        add_offset=None,
        filters=None,
        compressor=None,
        chunck_size=2500000,
    ):
        kwargs_variable["shape"] = data.shape
        kwargs_variable["compressor"] = (
            zarr.Blosc(cname="zstd", clevel=2) if compressor is None else compressor
        )
        kwargs_variable["filters"] = list()
        store_dtype = kwargs_variable.pop("store_dtype", None)
        if scale_factor is not None or add_offset is not None:
            if add_offset is None:
                add_offset = 0
            kwargs_variable["filters"].append(
                zarr.FixedScaleOffset(
                    offset=add_offset,
                    scale=1 / scale_factor,
                    dtype=kwargs_variable["dtype"],
                    astype=store_dtype,
                )
            )
        if filters is not None:
            kwargs_variable["filters"].extend(filters)
        dims = kwargs_variable.get("dimensions", None)
        # Manage chunk in 2d case
        if len(dims) == 1:
            kwargs_variable["chunks"] = (chunck_size,)
        if len(dims) == 2:
            second_dim = data.shape[1]
            kwargs_variable["chunks"] = (chunck_size // second_dim, second_dim)

        kwargs_variable.pop("dimensions")
        v = handler_zarr.create_dataset(**kwargs_variable)
        attrs = list(attr_variable.keys())
        attrs.sort()
        for attr in attrs:
            attr_value = attr_variable[attr]
            v.attrs[attr] = str(attr_value)
        if self.raw_data:
            if scale_factor is not None:
                s_bloc = kwargs_variable["chunks"][0]
                nb_bloc = int(ceil(data.shape[0] / s_bloc))
                for i in range(nb_bloc):
                    sl = slice(i * s_bloc, (i + 1) * s_bloc)
                    v[sl] = data[sl] * scale_factor + add_offset
            else:
                v[:] = data
        if not self.raw_data:
            v[:] = data
        try:
            if v.size < 1e8:
                v.attrs["min"] = str(v[:].min())
                v.attrs["max"] = str(v[:].max())
        except ValueError:
            logger.warning("Data is empty")

    def write_file(
        self, path="./", filename="%(path)s/%(sign_type)s.nc", zarr_flag=False, **kwargs
    ):
        """Write a netcdf or zarr with eddy obs.
        Zarr is usefull for large dataset > 10M observations

        :param str path: set path variable
        :param str filename: model to store file
        :param bool zarr_flag: If True, method will use zarr format instead of netcdf
        :param dict kwargs: look at :py:meth:`to_zarr` or :py:meth:`to_netcdf`
        """
        filename = filename % dict(
            path=path,
            sign_type=self.sign_legend,
            prod_time=datetime.now().strftime("%Y%m%d"),
        )
        if zarr_flag:
            filename = filename.replace(".nc", ".zarr")
        if filename.endswith(".zarr"):
            zarr_flag = True
        logger.info("Store in %s", filename)
        if zarr_flag:
            handler = zarr.open(filename, "w")
            self.to_zarr(handler, **kwargs)
        else:
            with Dataset(filename, "w", format="NETCDF4") as handler:
                self.to_netcdf(handler, **kwargs)

    @property
    def global_attr(self):
        return dict(
            Metadata_Conventions="Unidata Dataset Discovery v1.0",
            comment="Surface product; mesoscale eddies",
            framework_used="https://github.com/AntSimi/py-eddy-tracker",
            framework_version=__version__,
            standard_name_vocabulary="NetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table",
            rotation_type=self.sign_type,
        )

    def set_global_attr_zarr(self, h_zarr):
        for key, item in self.global_attr.items():
            h_zarr.attrs[key] = str(item)

    def set_global_attr_netcdf(self, h_nc):
        for key, item in self.global_attr.items():
            h_nc.setncattr(key, item)

    def extract_with_area(self, area, **kwargs):
        """
        Extract geographically with a bounding box.

        :param dict area: 4 coordinates in a dictionary to specify bounding box (lower left corner and upper right corner)
        :param dict kwargs: look at :py:meth:`extract_with_mask`
        :return: Return all eddy tracks which are in bounds
        :rtype: EddiesObservations

        .. code-block:: python

            area = dict(llcrnrlon=x0, llcrnrlat=y0, urcrnrlon=x1, urcrnrlat=y1)

        .. minigallery:: py_eddy_tracker.EddiesObservations.extract_with_area
        """
        mask = (self.latitude > area["llcrnrlat"]) * (self.latitude < area["urcrnrlat"])
        lon0 = area["llcrnrlon"]
        lon = (self.longitude - lon0) % 360 + lon0
        mask *= (lon > lon0) * (lon < area["urcrnrlon"])
        return self.extract_with_mask(mask, **kwargs)

    def extract_with_mask(self, mask):
        """
        Extract a subset of observations.

        :param array(bool) mask: mask to select observations
        :return: same object with selected observations
        :rtype: self
        """

        nb_obs = mask.sum()
        new = self.__class__.new_like(self, nb_obs)
        new.sign_type = self.sign_type
        if nb_obs == 0:
            logger.warning("Empty dataset will be created")
        else:
            for field in self.obs.dtype.descr:
                logger.debug("Copy of field %s ...", field)
                var = field[0]
                new.obs[var] = self.obs[var][mask]
        return new

    def scatter(self, ax, name=None, ref=None, factor=1, **kwargs):
        """
        Scatter data.

        :param matplotlib.axes.Axes ax: matplotlib axe used to draw
        :param str,array,None name:
            variable used to fill the contour, if None all elements have the same color
        :param float,None ref: if define use like west bound
        :param float factor: multiply value by
        :param dict kwargs: look at :py:meth:`matplotlib.axes.Axes.scatter`
        :return: scatter mappable

        .. minigallery:: py_eddy_tracker.EddiesObservations.scatter
        """
        x = self.longitude
        if ref is not None:
            x = (x - ref) % 360 + ref
        kwargs = kwargs.copy()
        if name is not None and "c" not in kwargs:
            v = self.parse_varname(name)
            kwargs["c"] = v * factor
        return ax.scatter(x, self.latitude, **kwargs)

    def filled(
        self,
        ax,
        varname=None,
        ref=None,
        intern=False,
        cmap="magma_r",
        lut=10,
        vmin=None,
        vmax=None,
        factor=1,
        **kwargs,
    ):
        """
        :param matplotlib.axes.Axes ax: matplotlib axe used to draw
        :param str,array,None varname: variable used to fill the contours, or an array of same size than obs
        :param float,None ref: if define use like west bound?
        :param bool intern: if True draw speed contours instead of effective contours
        :param str cmap: matplotlib colormap name
        :param int,None lut: Number of colors in the colormap
        :param float,None vmin: Min value of the colorbar
        :param float,None vmax: Max value of the colorbar
        :param float factor: multiply value by
        :return: Collection drawed
        :rtype: matplotlib.collections.PolyCollection

        .. minigallery:: py_eddy_tracker.EddiesObservations.filled
        """
        x_name, y_name = self.intern(intern)
        x, y = self[x_name], self[y_name]
        if ref is not None:
            # TODO : maybe buggy with global display
            shape_out = x.shape
            x, y = wrap_longitude(x.reshape(-1), y.reshape(-1), ref)
            x, y = x.reshape(shape_out), y.reshape(shape_out)
        verts = list()
        for x_, y_ in zip(x, y):
            verts.append(create_vertice(x_, y_))
        if "facecolors" not in kwargs:
            kwargs = kwargs.copy()
            cmap = get_cmap(cmap, lut)
            v = self.parse_varname(varname) * factor
            if vmin is None:
                vmin = v.min()
            if vmax is None:
                vmax = v.max()
            v = (v - vmin) / (vmax - vmin)
            colors = [cmap(v_) for v_ in v]
            kwargs["facecolors"] = colors
        c = PolyCollection(verts, **kwargs)
        ax.add_collection(c)
        c.cmap = cmap
        c.norm = Normalize(vmin=vmin, vmax=vmax)
        return c

    def __merge_filters__(self, *filters):
        """
        Compute an intersection between all filters after to evaluate each of them

        :param list(slice,array[int],array[bool]) filters:

        :return: Return applicable object to numpy.array
        :rtype: slice, index, mask
        """
        filter1 = filters[0]
        if len(filters) > 2:
            filter2 = self.__merge_filters__(*filters[1:])
        elif len(filters) == 2:
            filter2 = filters[1]
        # Merge indexs and filter
        if isinstance(filter1, slice):
            reject = ones(len(self), dtype="bool")
            reject[filter1] = False
            if isinstance(filter2, slice):
                reject[filter2] = False
                return ~reject
            # Mask case
            elif filter2.dtype == bool:
                return ~reject * filter2
            # index case
            else:
                return filter2[~reject[filter2]]
        # mask case
        elif filter1.dtype == bool:
            if isinstance(filter2, slice):
                select = zeros(len(self), dtype="bool")
                select[filter2] = True
                return select * filter1
            # Mask case
            elif filter2.dtype == bool:
                return filter2 * filter1
            # index case
            else:
                return filter2[filter1[filter2]]
        # index case
        else:
            if isinstance(filter2, slice):
                select = zeros(len(self), dtype="bool")
                select[filter2] = True
                return filter1[select[filter1]]
            # Mask case
            elif filter2.dtype == bool:
                return filter1[filter2[filter1]]
            # index case
            else:
                return filter1[in1d(filter1, filter2)]

    def merge_filters(self, *filters):
        """
        Compute an intersection between all filters after to evaluate each of them

        :param list(callable,None,slice,array[int],array[bool]) filters:

        :return: Return applicable object to numpy.array
        :rtype: slice, index, mask
        """
        if len(filters) == 1 and isinstance(filters[0], list):
            filters = filters[0]
        filters_ = list()
        # Remove all filter which select all obs
        for filter in filters:
            if callable(filter):
                filter = filter(self)
            if filter is None:
                continue
            if isinstance(filter, slice):
                if filter == slice(None):
                    continue
            elif filter.dtype == "bool":
                if filter.all():
                    continue
                if not filter.any():
                    return empty(0, dtype=int)
            filters_.append(filter)
        if len(filters_) == 1:
            return filters_[0]
        elif len(filters_) == 0:
            return slice(None)
        else:
            return self.__merge_filters__(*filters_)

    def bins_stat(self, xname, bins=None, yname=None, method=None, mask=None):
        """
        :param str,array xname: variable to compute stats on
        :param array, None bins: bins to perform statistics, if None bins = arange(variable.min(), variable.max() + 2)
        :param None,str,array yname: variable used to apply method
        :param None,str method: If None method counts the number of observations in each bin, can be "mean", "std"
        :param None,array(bool) mask: If defined use only True position
        :return: x array and y array
        :rtype: array,array

        .. minigallery:: py_eddy_tracker.EddiesObservations.bins_stat
        """
        v = self.parse_varname(xname)
        mask = self.merge_filters(mask)
        v = v[mask]
        if bins is None:
            bins = arange(v.min(), v.max() + 2)
        y, x = hist_numba(v, bins=bins)
        x = (x[1:] + x[:-1]) / 2
        if method == "mean":
            y_v = self.parse_varname(yname)
            y_v = y_v[mask]
            y_, _ = histogram(v, bins=bins, weights=y_v)
            with errstate(divide="ignore", invalid="ignore"):
                y = y_ / y
        return x, y

    def format_label(self, label):
        t0, t1 = self.period
        return label.format(
            t0=t0,
            t1=t1,
            nb_obs=len(self),
        )

    def display(self, ax, ref=None, extern_only=False, intern_only=False, **kwargs):
        """Plot the speed and effective (dashed) contour of the eddies

        :param matplotlib.axes.Axes ax: matplotlib axe used to draw
        :param float,None ref: western longitude reference used
        :param bool extern_only: if True, draw only the effective contour
        :param bool intern_only: if True, draw only the speed contour
        :param dict kwargs: look at :py:meth:`matplotlib.axes.Axes.plot`

        .. minigallery:: py_eddy_tracker.EddiesObservations.display
        """
        if not extern_only:
            lon_s = flatten_line_matrix(self.contour_lon_s)
            lat_s = flatten_line_matrix(self.contour_lat_s)
        if not intern_only:
            lon_e = flatten_line_matrix(self.contour_lon_e)
            lat_e = flatten_line_matrix(self.contour_lat_e)
        if "label" in kwargs:
            kwargs["label"] = self.format_label(kwargs["label"])
        kwargs_e = kwargs.copy()
        if "ls" not in kwargs_e and "linestyle" not in kwargs_e:
            kwargs_e["linestyle"] = "-."
        if not extern_only:
            kwargs_e.pop("label", None)

        mappables = list()
        if not extern_only:
            if ref is not None:
                lon_s, lat_s = wrap_longitude(lon_s, lat_s, ref, cut=True)
            mappables.append(ax.plot(lon_s, lat_s, **kwargs)[0])
        if not intern_only:
            if ref is not None:
                lon_e, lat_e = wrap_longitude(lon_e, lat_e, ref, cut=True)
            mappables.append(ax.plot(lon_e, lat_e, **kwargs_e)[0])
        return mappables

    def first_obs(self):
        """
        Get first obs of each trajectory.

        :rtype: __class__

        .. minigallery:: py_eddy_tracker.EddiesObservations.first_obs
        """
        return self.extract_with_mask(self.n == 0)

    def last_obs(self):
        """
        Get Last obs of each trajectory.

        :rtype: __class__

        .. minigallery:: py_eddy_tracker.EddiesObservations.last_obs
        """
        m = zeros(len(self), dtype="bool")
        m[-1] = True
        m[:-1][self.n[1:] == 0] = True
        return self.extract_with_mask(m)

    def is_convex(self, intern=False):
        """
        Get flag of the eddy's convexity

        :param bool intern: If True use speed contour instead of effective contour
        :return: True if the contour is convex
        :rtype: array[bool]
        """
        xname, yname = self.intern(intern)
        return convexs(self[xname], self[yname])

    def contains(self, x, y, intern=False):
        """
        Return index of contour which contain (x,y)

        :param array x: longitude
        :param array y: latitude
        :param bool intern: If true use speed contour instead of effective contour
        :return: indexs, -1 if no index
        :rtype: array[int32]
        """
        xname, yname = self.intern(intern)
        return poly_indexs(x, y, self[xname], self[yname])

    def inside(self, x, y, intern=False):
        """
        True for each point inside the effective contour of an eddy

        :param array x: longitude
        :param array y: latitude
        :param bool intern: If true use speed contour instead of effective contour
        :return: flag
        :rtype: array[bool]
        """
        xname, yname = self.intern(intern)
        return insidepoly(x, y, self[xname], self[yname])

    def grid_count(self, bins, intern=False, center=False, filter=slice(None)):
        """
        Count the eddies in each bin (use all pixels in each contour)

        :param (numpy.array,numpy.array) bins: bins (grid) to count
        :param bool intern: if True use speed contour only
        :param bool center: if True use of center to count
        :param array,mask,slice filter: keep the data selected with the filter
        :return: return the grid of counts
        :rtype: py_eddy_tracker.dataset.grid.RegularGridDataset

        .. minigallery:: py_eddy_tracker.EddiesObservations.grid_count
        """
        filter = self.merge_filters(filter)
        x_name, y_name = self.intern(intern)
        x_bins, y_bins = arange(*bins[0]), arange(*bins[1])
        x0 = bins[0][0]
        grid = ma.zeros((x_bins.shape[0] - 1, y_bins.shape[0] - 1), dtype="u4")
        from ..dataset.grid import RegularGridDataset

        regular_grid = RegularGridDataset.with_array(
            coordinates=("lon", "lat"),
            datas=dict(
                count=grid,
                lon=(x_bins[1:] + x_bins[:-1]) / 2,
                lat=(y_bins[1:] + y_bins[:-1]) / 2,
            ),
            variables_description=dict(
                count=dict(long_name="Number of times the pixel is within an eddy")
            ),
            centered=True,
        )
        if center:
            x, y = (self.longitude[filter] - x0) % 360 + x0, self.latitude[filter]
            grid[:] = histogram2d(x, y, (x_bins, y_bins))[0]
            grid.mask = grid.data == 0
        else:
            x_ref = ((self.longitude[filter] - x0) % 360 + x0 - 180).reshape(-1, 1)
            x_contour, y_contour = self[x_name][filter], self[y_name][filter]
            grid_count_pixel_in(
                grid,
                x_contour,
                y_contour,
                x_ref,
                regular_grid.x_bounds,
                regular_grid.y_bounds,
                regular_grid.xstep,
                regular_grid.ystep,
                regular_grid.N,
                regular_grid.is_circular(),
                regular_grid.x_size,
                regular_grid.x_c,
                regular_grid.y_c,
            )
            grid.mask = grid == 0
        return regular_grid

    def grid_box_stat(self, bins, varname, method=50, data=None, filter=slice(None)):
        """
        Get percentile of eddies in each bin

        :param (numpy.array,numpy.array) bins: bins (grid) to count
        :param str varname: variable to apply the method if data is None and will be output name
        :param str,float method: method to apply. If float, use ?
        :param array data: Array used to compute stat if defined
        :param array,mask,slice filter: keep the data selected with the filter
        :return: return grid of method
        :rtype: py_eddy_tracker.dataset.grid.RegularGridDataset

        .. minigallery:: py_eddy_tracker.EddiesObservations.grid_box_stat
        """
        x_bins, y_bins = arange(*bins[0]), arange(*bins[1])
        x0 = bins[0][0]
        x, y = (self.longitude - x0) % 360 + x0, self.latitude
        data = self[varname] if data is None else data
        if hasattr(data, "mask"):
            filter = self.merge_filters(~data.mask, self.merge_filters(filter))
        else:
            filter = self.merge_filters(filter)
        x, y, data = x[filter], y[filter], data[filter]

        from ..dataset.grid import RegularGridDataset

        shape = (x_bins.shape[0] - 1, y_bins.shape[0] - 1)
        grid = ma.empty(shape, dtype=data.dtype)
        grid.mask = ones(shape, dtype="bool")
        regular_grid = RegularGridDataset.with_array(
            coordinates=("x", "y"),
            datas={varname: grid, "x": x_bins[:-1], "y": y_bins[:-1]},
            centered=False,
        )
        grid_box_stat(
            regular_grid.x_c,
            regular_grid.y_c,
            grid.data,
            grid.mask,
            x,
            y,
            data,
            regular_grid.is_circular(),
            method,
        )

        return regular_grid

    def grid_stat(self, bins, varname, data=None):
        """
        Return the mean of the eddies' variable in each bin

        :param (numpy.array,numpy.array) bins: bins (grid) to compute the mean on
        :param str varname: name of variable to compute the mean on and output grid_name
        :param array data: Array used to compute stat if defined
        :return: return the gridde mean variable
        :rtype: py_eddy_tracker.dataset.grid.RegularGridDataset

        .. minigallery:: py_eddy_tracker.EddiesObservations.grid_stat
        """
        x_bins, y_bins = arange(*bins[0]), arange(*bins[1])
        x0 = bins[0][0]
        x, y = (self.longitude - x0) % 360 + x0, self.latitude
        data = self[varname] if data is None else data
        if hasattr(data, "mask"):
            m = ~data.mask
            sum_obs = histogram2d(x[m], y[m], (x_bins, y_bins), weights=data[m])[0]
            nb_obs = histogram2d(x[m], y[m], (x_bins, y_bins))[0]
        else:
            sum_obs = histogram2d(x, y, (x_bins, y_bins), weights=data)[0]
            nb_obs = histogram2d(x, y, (x_bins, y_bins))[0]
        from ..dataset.grid import RegularGridDataset

        with errstate(divide="ignore", invalid="ignore"):
            regular_grid = RegularGridDataset.with_array(
                coordinates=("x", "y"),
                datas={
                    varname: ma.array(sum_obs / nb_obs, mask=nb_obs == 0),
                    "x": x_bins[:-1],
                    "y": y_bins[:-1],
                },
                centered=False,
            )
        return regular_grid

    def interp_grid(
        self, grid_object, varname, method="center", dtype=None, intern=None
    ):
        """
        Interpolate a grid on a center or contour with mean, min or max method

        :param grid_object: Handler of grid to interp
        :type grid_object: py_eddy_tracker.dataset.grid.RegularGridDataset
        :param str varname: Name of variable to use
        :param str method: 'center', 'mean', 'max', 'min', 'nearest'
        :param str dtype: if None we use var dtype
        :param bool intern: Use extern or intern contour
        """
        if method in ("center", "nearest"):
            return grid_object.interp(varname, self.longitude, self.latitude, method)
        elif method in ("min", "max", "mean", "count"):
            x0 = grid_object.x_bounds[0]
            x_name, y_name = self.intern(False if intern is None else intern)
            x_ref = ((self.longitude - x0) % 360 + x0 - 180).reshape(-1, 1)
            x, y = (self[x_name] - x_ref) % 360 + x_ref, self[y_name]
            grid = grid_object.grid(varname)
            result = empty(self.shape, dtype=grid.dtype if dtype is None else dtype)
            min_method = method == "min"
            grid_stat(
                grid_object.x_c,
                grid_object.y_c,
                -grid if min_method else grid,
                x,
                y,
                result,
                grid_object.is_circular(),
                method="max" if min_method else method,
            )
            return -result if min_method else result
        else:
            raise Exception(f'method "{method}" unknown')

    @property
    def period(self):
        """
        Give the time coverage

        :return: first and last date
        :rtype: (int,int)
        """
        if self.period_ is None:
            self.period_ = self.time.min(), self.time.max()
        return self.period_

    @property
    def nb_days(self):
        """Return period days cover by dataset

        :return: Number of days
        :rtype: int
        """
        return self.period[1] - self.period[0] + 1


@njit(cache=True)
def grid_count_(grid, i, j):
    """
    Add one to each index
    """
    for i_, j_ in zip(i, j):
        grid[i_, j_] += 1


@njit(cache=True)
def grid_count_pixel_in(
    grid,
    x,
    y,
    x_ref,
    x_bounds,
    y_bounds,
    xstep,
    ystep,
    N,
    is_circular,
    x_size,
    x_c,
    y_c,
):
    """
    Count how many time a pixel is used.

    :param array grid:
    :param array x: x for all contour
    :param array y: y for all contour
    :param array x_ref: x reference for wrapping
    :param array x_bounds: grid longitude
    :param array y_bounds: grid latitude
    :param float xstep: step between two longitude
    :param float ystep: step between two latitude
    :param int N: shift of index to enlarge window
    :param bool is_circular: To know if grid is wrappable
    :param int x_size: Number of longitude
    :param array x_c: longitude coordinate of grid
    :param array y_c: latitude coordinate of grid
    """
    nb = x_ref.shape[0]
    for i_ in range(nb):
        x_, y_, x_ref_ = x[i_], y[i_], x_ref[i_]
        x_ = (x_ - x_ref_) % 360 + x_ref_
        v = create_vertice(x_, y_)
        (x_start, x_stop), (y_start, y_stop) = bbox_indice_regular(
            v,
            x_bounds,
            y_bounds,
            xstep,
            ystep,
            N,
            is_circular,
            x_size,
        )

        i, j = get_pixel_in_regular(v, x_c, y_c, x_start, x_stop, y_start, y_stop)
        grid_count_(grid, i, j)


@njit(cache=True)
def poly_indexs(x_p, y_p, x_c, y_c):
    """
    index of contour for each postion inside a contour, -1 in case of no contour

    :param array x_p: longitude to test
    :param array y_p: latitude to test
    :param array x_c: longitude of contours
    :param array y_c: latitude of contours
    """
    nb_p = x_p.shape[0]
    nb_c = x_c.shape[0]
    indexs = -ones(nb_p, dtype=numba_types.int32)
    for i in range(nb_c):
        x_c_min, y_c_min = x_c[i].min(), y_c[i].min()
        x_c_max, y_c_max = x_c[i].max(), y_c[i].max()
        v = create_vertice(x_c[i], y_c[i])
        for j in range(nb_p):
            if indexs[j] != -1:
                continue
            x, y = x_p[j], y_p[j]
            if x > x_c_min and x < x_c_max and y > y_c_min and y < y_c_max:
                if winding_number_poly(x, y, v) != 0:
                    indexs[j] = i
    return indexs


@njit(cache=True)
def insidepoly(x_p, y_p, x_c, y_c):
    """
    True for each postion inside a contour

    :param array x_p: longitude to test
    :param array y_p: latitude to test
    :param array x_c: longitude of contours
    :param array y_c: latitude of contours
    """
    nb_p = x_p.shape[0]
    nb_c = x_c.shape[0]
    flag = zeros(nb_p, dtype=numba_types.bool_)
    for i in range(nb_c):
        x_c_min, y_c_min = x_c[i].min(), y_c[i].min()
        x_c_max, y_c_max = x_c[i].max(), y_c[i].max()
        v = create_vertice(x_c[i], y_c[i])
        for j in range(nb_p):
            if flag[j]:
                continue
            x, y = x_p[j], y_p[j]
            if x > x_c_min and x < x_c_max and y > y_c_min and y < y_c_max:
                if winding_number_poly(x, y, v) != 0:
                    flag[j] = True
    return flag


@njit(cache=True)
def grid_box_stat(x_c, y_c, grid, mask, x, y, value, circular=False, method=50):
    """
    Compute method on each set (one set by box)

    :param array_like x_c: grid longitude coordinates
    :param array_like y_c: grid latitude coordinates
    :param array_like grid: grid to store the result
    :param array[bool] mask: grid to store unused boxes
    :param array_like x: longitude of observations
    :param array_like y: latitude of observations
    :param array_like value: value to group to apply method
    :param bool circular: True if grid is wrappable
    :param float method: percentile
    """
    xstep, ystep = x_c[1] - x_c[0], y_c[1] - y_c[0]
    x0, y0 = x_c[0] - xstep / 2.0, y_c[0] - ystep / 2.0
    nb_x = x_c.shape[0]
    nb_y = y_c.shape[0]
    i, j = (
        ((x - x0) // xstep).astype(numba_types.int32),
        ((y - y0) // ystep).astype(numba_types.int32),
    )
    if circular:
        i %= nb_x
    else:
        if (i < 0).any():
            raise Exception("x indices underflow")
        if (i >= nb_x).any():
            raise Exception("x indices overflow")
        if (j < 0).any():
            raise Exception("y indices underflow")
        if (j >= nb_y).any():
            raise Exception("y indices overflow")
    abs_i = j * nb_x + i
    k_sort = abs_i.argsort()
    i0, j0 = i[k_sort[0]], j[k_sort[0]]
    values = list()
    for k in k_sort:
        i_, j_ = i[k], j[k]
        # group change
        if i_ != i0 or j_ != j0:
            # apply method and store result
            grid[i_, j_] = percentile(values, method)
            mask[i_, j_] = False
            # start new group
            i0, j0 = i_, j_
            # reset list
            values.clear()
        values.append(value[k])


@njit(cache=True)
def grid_stat(x_c, y_c, grid, x, y, result, circular=False, method="mean"):
    """
    Compute the mean or the max of the grid for each contour

    :param array_like x_c: the grid longitude coordinates
    :param array_like y_c: the grid latitude coordinates
    :param array_like grid: grid value
    :param array_like x: longitude of contours
    :param array_like y: latitude of contours
    :param array_like result: return values
    :param bool circular: True if grid is wrappable
    :param str method: 'mean', 'max'
    """
    # FIXME : how does it work on grid bound
    nb = result.shape[0]
    xstep, ystep = x_c[1] - x_c[0], y_c[1] - y_c[0]
    x0, y0 = x_c - xstep / 2.0, y_c - ystep / 2.0
    nb_x = x_c.shape[0]
    max_method = "max" == method
    mean_method = "mean" == method
    count_method = "count" == method
    for elt in range(nb):
        v = create_vertice(x[elt], y[elt])
        (x_start, x_stop), (y_start, y_stop) = bbox_indice_regular(
            v, x0, y0, xstep, ystep, 1, circular, nb_x
        )
        i, j = get_pixel_in_regular(v, x_c, y_c, x_start, x_stop, y_start, y_stop)

        if count_method:
            result[elt] = i.shape[0]
        elif mean_method:
            v_sum = 0
            for i_, j_ in zip(i, j):
                v_sum += grid[i_, j_]
            nb_ = i.shape[0]
            # FIXME : how does it work on grid bound,
            if nb_ == 0:
                result[elt] = nan
            else:
                result[elt] = v_sum / nb_
        elif max_method:
            v_max = -1e40
            for i_, j_ in zip(i, j):
                v_max = max(v_max, grid[i_, j_])
            result[elt] = v_max


class VirtualEddiesObservations(EddiesObservations):
    """Class to work with virtual obs"""

    __slots__ = ()

    @property
    def elements(self):
        elements = super().elements
        elements.extend(["track", "segment_size", "dlon", "dlat"])
        return list(set(elements))


@njit(cache=True)
def numba_where(mask):
    """Usefull when mask is close to be empty"""
    return where(mask)


@njit(cache=True)
def sum_row_column(mask):
    """
    Compute sum on row and column at same time
    """
    nb_x, nb_y = mask.shape
    row_sum = zeros(nb_x, dtype=numba_types.int32)
    column_sum = zeros(nb_y, dtype=numba_types.int32)
    for i in range(nb_x):
        for j in range(nb_y):
            if mask[i, j]:
                row_sum[i] += 1
                column_sum[j] += 1
    return row_sum, column_sum
