# -*- coding: utf-8 -*-
"""
Class to load and manipulate RegularGrid and UnRegularGrid
"""
from datetime import datetime
import logging

from cv2 import filter2D
from matplotlib.path import Path as BasePath
from netCDF4 import Dataset
from numba import njit, prange, types as numba_types
from numpy import (
    arange,
    array,
    ceil,
    concatenate,
    cos,
    deg2rad,
    empty,
    errstate,
    exp,
    float_,
    floor,
    histogram2d,
    int_,
    interp,
    isnan,
    linspace,
    ma,
    mean as np_mean,
    meshgrid,
    nan,
    nanmean,
    ones,
    percentile,
    pi,
    radians,
    round_,
    sin,
    sinc,
    where,
    zeros,
)
from pint import UnitRegistry
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import welch
from scipy.spatial import cKDTree
from scipy.special import j1

from .. import VAR_DESCR
from ..eddy_feature import Amplitude, Contours
from ..generic import (
    bbox_indice_regular,
    coordinates_to_local,
    distance,
    interp2d_geo,
    local_to_coordinates,
    nearest_grd_indice,
    uniform_resample,
)
from ..observations.observation import EddiesObservations
from ..poly import (
    create_vertice,
    fit_circle,
    get_pixel_in_regular,
    poly_area,
    poly_contain_poly,
    visvalingam,
    winding_number_poly,
)

logger = logging.getLogger("pet")


def raw_resample(datas, fixed_size):
    nb_value = datas.shape[0]
    if nb_value == 1:
        raise Exception()
    return interp(
        arange(fixed_size), arange(nb_value) * (fixed_size - 1) / (nb_value - 1), datas
    )


@property
def mean_coordinates(self):
    # last coordinates == first
    return self.vertices[1:].mean(axis=0)


@property
def lon(self):
    return self.vertices[:, 0]


@property
def lat(self):
    return self.vertices[:, 1]


BasePath.mean_coordinates = mean_coordinates
BasePath.lon = lon
BasePath.lat = lat


@njit(cache=True)
def uniform_resample_stack(vertices, num_fac=2, fixed_size=None):
    x_val, y_val = vertices[:, 0], vertices[:, 1]
    x_new, y_new = uniform_resample(x_val, y_val, num_fac, fixed_size)
    data = empty((x_new.shape[0], 2), dtype=vertices.dtype)
    data[:, 0] = x_new
    data[:, 1] = y_new
    return data


@njit(cache=True)
def value_on_regular_contour(x_g, y_g, z_g, m_g, vertices, num_fac=2, fixed_size=None):
    x_val, y_val = vertices[:, 0], vertices[:, 1]
    x_new, y_new = uniform_resample(x_val, y_val, num_fac, fixed_size)
    return interp2d_geo(x_g, y_g, z_g, m_g, x_new[1:], y_new[1:])


@njit(cache=True)
def mean_on_regular_contour(
    x_g, y_g, z_g, m_g, vertices, num_fac=2, fixed_size=None, nan_remove=False
):
    x_val, y_val = vertices[:, 0], vertices[:, 1]
    x_new, y_new = uniform_resample(x_val, y_val, num_fac, fixed_size)
    values = interp2d_geo(x_g, y_g, z_g, m_g, x_new[1:], y_new[1:])
    if nan_remove:
        return nanmean(values)
    else:
        return values.mean()


def fit_circle_path(self, method="fit"):
    if not hasattr(self, "_circle_params"):
        self._circle_params = dict()
    if method not in self._circle_params.keys():
        if method == "fit":
            self._circle_params["fit"] = _fit_circle_path(self.vertices)
        if method == "equal_area":
            self._circle_params["equal_area"] = _circle_from_equal_area(self.vertices)
    return self._circle_params[method]


@njit(cache=True, fastmath=True)
def _circle_from_equal_area(vertice):
    lons, lats = vertice[:, 0], vertice[:, 1]
    # last coordinates == first
    lon0, lat0 = lons[1:].mean(), lats[1:].mean()
    c_x, c_y = coordinates_to_local(lons, lats, lon0, lat0)
    # Sometimes, edge is only a dot of few coordinates
    d_lon = lons.max() - lons.min()
    d_lat = lats.max() - lats.min()
    if d_lon < 1e-7 and d_lat < 1e-7:
        # logger.warning('An edge is only define in one position')
        # logger.debug('%d coordinates %s,%s', len(lons),lons,
        # lats)
        return 0, -90, nan, nan
    return lon0, lat0, (poly_area(c_x, c_y) / pi) ** 0.5, nan


@njit(cache=True, fastmath=True)
def _fit_circle_path(vertice):
    lons, lats = vertice[:, 0], vertice[:, 1]
    # last coordinates == first
    lon0, lat0 = lons[1:].mean(), lats[1:].mean()
    c_x, c_y = coordinates_to_local(lons, lats, lon0, lat0)
    # Some time, edge is only a dot of few coordinates
    d_lon = lons.max() - lons.min()
    d_lat = lats.max() - lats.min()
    if d_lon < 1e-7 and d_lat < 1e-7:
        # logger.warning('An edge is only define in one position')
        # logger.debug('%d coordinates %s,%s', len(lons),lons,
        # lats)
        return 0, -90, nan, nan
    centlon, centlat, eddy_radius, err = fit_circle(c_x, c_y)
    centlon, centlat = local_to_coordinates(centlon, centlat, lon0, lat0)
    centlon = (centlon - lon0 + 180) % 360 + lon0 - 180
    return centlon, centlat, eddy_radius, err


@njit(cache=True, fastmath=True)
def _get_pixel_in_unregular(vertices, x_c, y_c, x_start, x_stop, y_start, y_stop):
    nb_x, nb_y = x_stop - x_start, y_stop - y_start
    wn = empty((nb_x, nb_y), dtype=numba_types.bool_)
    for i in range(nb_x):
        for j in range(nb_y):
            x_pt = x_c[i + x_start, j + y_start]
            y_pt = y_c[i + x_start, j + y_start]
            wn[i, j] = winding_number_poly(x_pt, y_pt, vertices)
    i_x, i_y = where(wn)
    i_x += x_start
    i_y += y_start
    return i_x, i_y


BasePath.fit_circle = fit_circle_path


def pixels_in(self, grid):
    if not hasattr(self, "_slice"):
        self._slice = grid.bbox_indice(self.vertices)
    if not hasattr(self, "_pixels_in"):
        self._pixels_in = grid.get_pixels_in(self)
    return self._pixels_in


@property
def bbox_slice(self):
    if not hasattr(self, "_slice"):
        raise Exception("No pixels_in call before!")
    return self._slice


@property
def pixels_index(self):
    if not hasattr(self, "_slice"):
        raise Exception("No pixels_in call before!")
    return self._pixels_in


@property
def nb_pixel(self):
    if not hasattr(self, "_pixels_in"):
        raise Exception("No pixels_in call before!")
    return self._pixels_in[0].shape[0]


BasePath.pixels_in = pixels_in
BasePath.pixels_index = pixels_index
BasePath.bbox_slice = bbox_slice
BasePath.nb_pixel = nb_pixel


class GridDataset(object):
    """
    Class for basic tools on NetCDF Grid
    """

    __slots__ = (
        "x_c",
        "y_c",
        "x_bounds",
        "y_bounds",
        "centered",
        "x_dim",
        "y_dim",
        "coordinates",
        "filename",
        "dimensions",
        "indexs",
        "variables_description",
        "global_attrs",
        "vars",
        "contours",
        "nan_mask",
    )

    GRAVITY = 9.807
    EARTH_RADIUS = 6370997.0
    # EARTH_RADIUS = 6378136.3
    # indice margin (if put to 0, raise warning that i don't understand)
    N = 1

    def __init__(
        self,
        filename,
        x_name,
        y_name,
        centered=None,
        indexs=None,
        unset=False,
        nan_masking=False,
    ):
        """
        :param str filename: Filename to load
        :param str x_name: Name of longitude coordinates
        :param str y_name: Name of latitude coordinates
        :param bool,None centered: Allow to know how coordinates could be used with pixel
        :param dict indexs: A dictionary that sets indexes to use for non-coordinate dimensions
        :param bool unset: Set to True to create an empty grid object without file
        :param bool nan_masking: Set to True to replace data.mask with isnan method result
        """
        self.dimensions = None
        self.variables_description = None
        self.global_attrs = None
        self.x_c = None
        self.y_c = None
        self.x_bounds = None
        self.y_bounds = None
        self.x_dim = None
        self.y_dim = None
        self.nan_mask = nan_masking
        self.centered = centered
        self.contours = None
        self.filename = filename
        self.coordinates = x_name, y_name
        self.vars = dict()
        self.indexs = dict() if indexs is None else indexs
        if centered is None:
            logger.warning(
                "We assume pixel position of grid is centered for %s", filename
            )
        if not unset:
            self.populate()

    def populate(self):
        if self.dimensions is None:
            self.load_general_features()
            self.load()

    def clean(self):
        self.dimensions = None
        self.variables_description = None
        self.global_attrs = None
        self.x_c = None
        self.y_c = None
        self.x_bounds = None
        self.y_bounds = None
        self.x_dim = None
        self.y_dim = None
        self.contours = None
        self.vars = dict()

    @property
    def is_centered(self):
        """Give True if pixel is described with its center's position or
        a corner

        :return: True if centered
        :rtype: bool
        """
        if self.centered is None:
            return True
        else:
            return self.centered

    def load_general_features(self):
        """Load attrs to be stored in object"""
        logger.debug(
            "Load general feature from %(filename)s", dict(filename=self.filename)
        )
        with Dataset(self.filename) as h:
            # Load generals
            self.dimensions = {i: len(v) for i, v in h.dimensions.items()}
            self.variables_description = dict()
            for i, v in h.variables.items():
                args = (i, v.datatype)
                kwargs = dict(dimensions=v.dimensions, zlib=True)
                if hasattr(v, "_FillValue"):
                    kwargs["fill_value"] = (v._FillValue,)
                attrs = dict()
                for attr in v.ncattrs():
                    if attr in kwargs.keys():
                        continue
                    if attr == "_FillValue":
                        continue
                    attrs[attr] = getattr(v, attr)
                self.variables_description[i] = dict(
                    args=args, kwargs=kwargs, attrs=attrs, infos=dict()
                )
            self.global_attrs = {attr: getattr(h, attr) for attr in h.ncattrs()}

    def write(self, filename):
        """Write dataset output with same format as input

        :param str filename: filename used to save the grid
        """
        with Dataset(filename, "w") as h_out:
            for dimension, size in self.dimensions.items():
                test = False
                for varname, variable in self.variables_description.items():
                    if (
                        varname not in self.coordinates
                        and varname not in self.vars.keys()
                    ):
                        continue
                    if dimension in variable["kwargs"]["dimensions"]:
                        test = True
                        break
                if test:
                    h_out.createDimension(dimension, size)

            for varname, variable in self.variables_description.items():
                if varname not in self.coordinates and varname not in self.vars.keys():
                    continue
                var = h_out.createVariable(*variable["args"], **variable["kwargs"])
                for key, value in variable["attrs"].items():
                    setattr(var, key, value)

                infos = self.variables_description[varname]["infos"]
                if infos.get("transpose", False):
                    var[:] = self.vars[varname].T
                else:
                    var[:] = self.vars[varname]

            for attr, value in self.global_attrs.items():
                setattr(h_out, attr, value)

    def load(self):
        """
        Load variable (data).
        Get coordinates and setup coordinates function
        """
        x_name, y_name = self.coordinates
        with Dataset(self.filename) as h:
            self.x_dim = h.variables[x_name].dimensions
            self.y_dim = h.variables[y_name].dimensions

            sl_x = [self.indexs.get(dim, slice(None)) for dim in self.x_dim]
            sl_y = [self.indexs.get(dim, slice(None)) for dim in self.y_dim]
            self.vars[x_name] = h.variables[x_name][sl_x]
            self.vars[y_name] = h.variables[y_name][sl_y]

        self.setup_coordinates()

    @staticmethod
    def get_mask(a):
        if len(a.mask.shape):
            m = a.mask
        else:
            m = ones(a.shape, dtype="bool") if a.mask else zeros(a.shape, dtype="bool")
        return m

    @staticmethod
    def c_to_bounds(c):
        """
        Centered coordinates to bounds coordinates

        :param array c: centered coordinates to translate
        :return: bounds coordinates
        """
        bounds = concatenate((c, (2 * c[-1] - c[-2],)))
        d = bounds[1:] - bounds[:-1]
        bounds[:-1] -= d / 2
        bounds[-1] -= d[-1] / 2
        return bounds

    def setup_coordinates(self):
        x_name, y_name = self.coordinates
        if self.is_centered:
            # logger.info("Grid center")
            self.x_c = self.vars[x_name].astype("float64")
            self.y_c = self.vars[y_name].astype("float64")

            self.x_bounds = concatenate((self.x_c, (2 * self.x_c[-1] - self.x_c[-2],)))
            self.y_bounds = concatenate((self.y_c, (2 * self.y_c[-1] - self.y_c[-2],)))
            d_x = self.x_bounds[1:] - self.x_bounds[:-1]
            d_y = self.y_bounds[1:] - self.y_bounds[:-1]
            self.x_bounds[:-1] -= d_x / 2
            self.x_bounds[-1] -= d_x[-1] / 2
            self.y_bounds[:-1] -= d_y / 2
            self.y_bounds[-1] -= d_y[-1] / 2

        else:
            self.x_bounds = self.vars[x_name].astype("float64")
            self.y_bounds = self.vars[y_name].astype("float64")

            if len(self.x_dim) == 1:
                self.x_c = self.x_bounds.copy()
                dx2 = (self.x_bounds[1:] - self.x_bounds[:-1]) / 2
                self.x_c[:-1] += dx2
                self.x_c[-1] += dx2[-1]
                self.y_c = self.y_bounds.copy()
                dy2 = (self.y_bounds[1:] - self.y_bounds[:-1]) / 2
                self.y_c[:-1] += dy2
                self.y_c[-1] += dy2[-1]
            else:
                raise Exception("not write")

    def is_circular(self):
        """Check grid circularity"""
        return False

    def units(self, varname):
        """Get unit from variable"""
        stored_units = self.variables_description[varname]["attrs"].get("units", None)
        if stored_units is not None:
            return stored_units
        with Dataset(self.filename) as h:
            var = h.variables[varname]
            if hasattr(var, "units"):
                return var.units

    @property
    def variables(self):
        return self.variables_description.keys()

    def copy(self, grid_in, grid_out):
        """
        Duplicate the variable from grid_in in grid_out

        :param grid_in:
        :param grid_out:

        """
        h_dict = self.variables_description[grid_in]
        self.variables_description[grid_out] = dict(
            infos=h_dict["infos"].copy(),
            attrs=h_dict["attrs"].copy(),
            args=tuple((grid_out, *h_dict["args"][1:])),
            kwargs=h_dict["kwargs"].copy(),
        )
        self.vars[grid_out] = self.grid(grid_in).copy()

    def add_grid(self, varname, grid):
        """
        Add a grid in handler

        :param str varname: name of the future grid
        :param array grid: grid array
        """
        self.vars[varname] = grid

    def grid(self, varname, indexs=None):
        """Give the grid required

        :param str varname: Variable to get
        :param dict,None indexs: If defined dict must have dimensions name as key
        :return: array asked, reduced by the indexes
        :rtype: array

        .. minigallery:: py_eddy_tracker.GridDataset.grid
        """
        if indexs is None:
            indexs = dict()
        if varname not in self.vars:
            coordinates_dims = list(self.x_dim)
            coordinates_dims.extend(list(self.y_dim))
            logger.debug(
                "Load %(varname)s from %(filename)s",
                dict(varname=varname, filename=self.filename),
            )
            with Dataset(self.filename) as h:
                dims = h.variables[varname].dimensions
                sl = [
                    indexs.get(
                        dim,
                        self.indexs.get(
                            dim, slice(None) if dim in coordinates_dims else 0
                        ),
                    )
                    for dim in dims
                ]
                self.vars[varname] = h.variables[varname][sl]
                if len(self.x_dim) == 1:
                    i_x = where(array(dims) == self.x_dim)[0][0]
                    i_y = where(array(dims) == self.y_dim)[0][0]
                    if i_x > i_y:
                        self.variables_description[varname]["infos"]["transpose"] = True
                        self.vars[varname] = self.vars[varname].T
            if self.nan_mask:
                self.vars[varname] = ma.array(
                    self.vars[varname],
                    mask=isnan(self.vars[varname]),
                )
            if not hasattr(self.vars[varname], "mask"):
                self.vars[varname] = ma.array(
                    self.vars[varname],
                    mask=zeros(self.vars[varname].shape, dtype="bool"),
                )
        return self.vars[varname]

    def grid_tiles(self, varname, slice_x, slice_y):
        """Give the grid tiles required, without buffer system"""
        coordinates_dims = list(self.x_dim)
        coordinates_dims.extend(list(self.y_dim))
        logger.debug(
            "Extract %(varname)s from %(filename)s with slice(x:%(slice_x)s,y:%(slice_y)s)",
            dict(
                varname=varname,
                filename=self.filename,
                slice_y=slice_y,
                slice_x=slice_x,
            ),
        )
        with Dataset(self.filename) as h:
            dims = h.variables[varname].dimensions
            sl = [
                (slice_x if dim in list(self.x_dim) else slice_y)
                if dim in coordinates_dims
                else 0
                for dim in dims
            ]
            data = h.variables[varname][sl]
            if len(self.x_dim) == 1:
                i_x = where(array(dims) == self.x_dim)[0][0]
                i_y = where(array(dims) == self.y_dim)[0][0]
                if i_x > i_y:
                    data = data.T
        if not hasattr(data, "mask"):
            data = ma.array(data, mask=zeros(data.shape, dtype="bool"))
        return data

    def high_filter(self, grid_name, w_cut, **kwargs):
        """Return the high-pass filtered grid, by substracting to the initial grid the low-pass filtered grid (default: order=1)

        :param grid_name: the name of the grid
        :param int, w_cut: the half-power wavelength cutoff (km)
        """
        result = self._low_filter(grid_name, w_cut, **kwargs)
        self.vars[grid_name] -= result

    def low_filter(self, grid_name, w_cut, **kwargs):
        """Return the low-pass filtered grid (default: order=1)

        :param grid_name: the name of the grid
        :param int, w_cut: the half-power wavelength cutoff (km)
        """
        result = self._low_filter(grid_name, w_cut, **kwargs)
        self.vars[grid_name] -= self.vars[grid_name] - result

    @property
    def bounds(self):
        """Give bounds"""
        return (
            self.x_bounds.min(),
            self.x_bounds.max(),
            self.y_bounds.min(),
            self.y_bounds.max(),
        )

    def eddy_identification(
        self,
        grid_height,
        uname,
        vname,
        date,
        step=0.005,
        shape_error=55,
        presampling_multiplier=10,
        sampling=50,
        sampling_method="visvalingam",
        pixel_limit=None,
        precision=None,
        force_height_unit=None,
        force_speed_unit=None,
        **kwargs,
    ):
        """
        Compute eddy identification on the specified grid

        :param str grid_height: Grid name of Sea Surface Height
        :param str uname: Grid name of u speed component
        :param str vname: Grid name of v speed component
        :param datetime.datetime date: Date to be stored in object to date data
        :param float,int step: Height between two layers in m
        :param float,int shape_error: Maximal error allowed for outermost contour in %
        :param int presampling_multiplier:
            Evenly oversample the initial number of points in the contour by nb_pts x presampling_multiplier to fit circles
        :param int sampling: Number of points to store contours and speed profile
        :param str sampling_method: Method to resample the stored contours, 'uniform' or 'visvalingam'
        :param (int,int),None pixel_limit:
            Min and max number of pixels inside the inner and the outermost contour to be considered as an eddy
        :param float,None precision: Truncate values at the defined precision in m
        :param str force_height_unit: Unit used for height unit
        :param str force_speed_unit: Unit used for speed unit
        :param dict kwargs: Arguments given to amplitude (mle, nb_step_min, nb_step_to_be_mle).
            Look at :py:meth:`py_eddy_tracker.eddy_feature.Amplitude`
            The amplitude threshold is given by `step*nb_step_min`


        :return: Return a list of 2 elements: Anticyclones and Cyclones
        :rtype: py_eddy_tracker.observations.observation.EddiesObservations

        .. minigallery:: py_eddy_tracker.GridDataset.eddy_identification
        """
        if not isinstance(date, datetime):
            raise Exception("Date argument must be a datetime object")
        # The inf limit must be in pixel and sup limit in surface
        if pixel_limit is None:
            pixel_limit = (4, 1000)

        # Compute an interpolator for eke
        self.init_speed_coef(uname, vname)

        # Get unit of h grid
        h_units = (
            self.units(grid_height) if force_height_unit is None else force_height_unit
        )
        units = UnitRegistry()
        in_h_unit = units.parse_expression(h_units)
        if in_h_unit is not None:
            factor, _ = in_h_unit.to("m").to_tuple()
            logger.info(
                "We will apply on step a factor to be coherent with grid : %f",
                1 / factor,
            )
            step /= factor
            if precision is not None:
                precision /= factor

        # Get ssh grid
        data = self.grid(grid_height).astype("f8")
        # In case of a reduced mask
        if len(data.mask.shape) == 0 and not data.mask:
            data.mask = zeros(data.shape, dtype="bool")
        # we remove noisy data
        if precision is not None:
            data = (data / precision).round() * precision
        # Compute levels for ssh
        z_min, z_max = data.min(), data.max()
        d_z = z_max - z_min
        data_tmp = data[~data.mask]
        epsilon = 0.001  # in %
        z_min_p, z_max_p = (
            percentile(data_tmp, epsilon),
            percentile(data_tmp, 100 - epsilon),
        )
        d_zp = z_max_p - z_min_p
        if d_z / d_zp > 2:
            logger.warning(
                "Maybe some extrema are present zmin %f (m) and zmax %f (m) will be replace by %f and %f",
                z_min,
                z_max,
                z_min_p,
                z_max_p,
            )
            z_min, z_max = z_min_p, z_max_p

        logger.debug("Levels from %f to %f", z_min, z_max)
        levels = arange(z_min - z_min % step, z_max - z_max % step + 2 * step, step)

        # Get x and y values
        x, y = self.x_c, self.y_c

        # Compute ssh contour
        self.contours = Contours(x, y, data, levels, wrap_x=self.is_circular())

        out_sampling = dict(fixed_size=sampling)
        resample = visvalingam if sampling_method == "visvalingam" else uniform_resample
        track_extra_variables = [
            "height_max_speed_contour",
            "height_external_contour",
            "height_inner_contour",
            "lon_max",
            "lat_max",
        ]
        array_variables = [
            "contour_lon_e",
            "contour_lat_e",
            "contour_lon_s",
            "contour_lat_s",
            "uavg_profile",
        ]
        # Complete cyclonic and anticylonic research:
        a_and_c = list()
        for anticyclonic_search in [True, False]:
            eddies = list()
            iterator = 1 if anticyclonic_search else -1

            # Loop over each collection
            for coll_ind, coll in enumerate(self.contours.iter(step=iterator)):
                corrected_coll_index = coll_ind
                if iterator == -1:
                    corrected_coll_index = -coll_ind - 1

                contour_paths = coll.get_paths()
                nb_paths = len(contour_paths)
                if nb_paths == 0:
                    continue
                cvalues = self.contours.cvalues[corrected_coll_index]
                logger.debug(
                    "doing collection %s, contour value %.4f, %d paths",
                    corrected_coll_index,
                    cvalues,
                    nb_paths,
                )

                # Loop over individual c_s contours (i.e., every eddy in field)
                for contour in contour_paths:
                    if contour.used:
                        continue
                    # FIXME : center could be outside the contour due to the fit
                    # FIXME : warning : the fit is made on raw sampling
                    _, _, _, aerr = contour.fit_circle()

                    # Filter for shape
                    if aerr < 0 or aerr > shape_error or isnan(aerr):
                        contour.reject = 1
                        continue

                    # Find all pixels in the contour
                    i_x_in, i_y_in = contour.pixels_in(self)

                    # Check if pixels in contour are masked
                    if has_masked_value(data.mask, i_x_in, i_y_in):
                        if contour.reject == 0:
                            contour.reject = 2
                        continue

                    # Test of the rotating sense: cyclone or anticyclone
                    if has_value(
                        data, i_x_in, i_y_in, cvalues, below=anticyclonic_search
                    ):
                        continue

                    # Test the number of pixels within the outermost contour
                    # FIXME : Maybe limit max must be replaced with a maximum of surface
                    if (
                        contour.nb_pixel < pixel_limit[0]
                        or contour.nb_pixel > pixel_limit[1]
                    ):
                        contour.reject = 3
                        continue

                    # Here the considered contour passed shape_error test, masked_pixels test,
                    # values strictly above (AEs) or below (CEs) the contour, number_pixels test)

                    # Compute amplitude
                    reset_centroid, amp = self.get_amplitude(
                        contour,
                        cvalues,
                        data,
                        anticyclonic_search=anticyclonic_search,
                        level=self.contours.levels[corrected_coll_index],
                        interval=step,
                        **kwargs,
                    )
                    # If we have a valid amplitude
                    if (not amp.within_amplitude_limits()) or (amp.amplitude == 0):
                        contour.reject = 4
                        continue
                    if reset_centroid:

                        if self.is_circular():
                            centi = self.normalize_x_indice(reset_centroid[0])
                        else:
                            centi = reset_centroid[0]
                        centj = reset_centroid[1]
                        # FIXME : To move in regular and unregular grid
                        if len(x.shape) == 1:
                            centlon_e = x[centi]
                            centlat_e = y[centj]
                        else:
                            centlon_e = x[centi, centj]
                            centlat_e = y[centi, centj]

                    # centlat_e and centlon_e must be indexes of maximum, we will loose some inner contour if it's not
                    (
                        max_average_speed,
                        speed_contour,
                        inner_contour,
                        speed_array,
                        i_max_speed,
                        i_inner,
                    ) = self.get_uavg(
                        self.contours,
                        centlon_e,
                        centlat_e,
                        contour,
                        anticyclonic_search,
                        corrected_coll_index,
                        pixel_min=pixel_limit[0],
                    )

                    # FIXME : Instantiate new EddyObservation object (high cost, need to be reviewed)
                    obs = EddiesObservations(
                        size=1,
                        track_extra_variables=track_extra_variables,
                        track_array_variables=sampling,
                        array_variables=array_variables,
                    )
                    obs.height_max_speed_contour[:] = self.contours.cvalues[i_max_speed]
                    obs.height_external_contour[:] = cvalues
                    obs.height_inner_contour[:] = self.contours.cvalues[i_inner]
                    array_size = speed_array.shape[0]
                    obs.nb_contour_selected[:] = array_size
                    if speed_array.shape[0] == 1:
                        obs.uavg_profile[:] = speed_array[0]
                    else:
                        obs.uavg_profile[:] = raw_resample(speed_array, sampling)
                    obs.amplitude[:] = amp.amplitude
                    obs.speed_average[:] = max_average_speed
                    obs.num_point_e[:] = contour.lon.shape[0]
                    obs.num_point_s[:] = speed_contour.lon.shape[0]

                    # Evenly resample contours with nb_pts = nb_pts_original x presampling_multiplier
                    xy_i = uniform_resample(
                        inner_contour.lon,
                        inner_contour.lat,
                        num_fac=presampling_multiplier,
                    )
                    xy_e = uniform_resample(
                        contour.lon,
                        contour.lat,
                        num_fac=presampling_multiplier,
                    )
                    xy_s = uniform_resample(
                        speed_contour.lon,
                        speed_contour.lat,
                        num_fac=presampling_multiplier,
                    )

                    # First, get position of max SSH based on best fit circle with resampled innermost contour
                    centlon_i, centlat_i, _, _ = _fit_circle_path(create_vertice(*xy_i))
                    obs.lon_max[:] = centlon_i
                    obs.lat_max[:] = centlat_i

                    # Second, get speed-based radius, shape error, eddy center, area based on resampled contour of max uavg
                    centlon_s, centlat_s, eddy_radius_s, aerr_s = _fit_circle_path(
                        create_vertice(*xy_s)
                    )
                    obs.radius_s[:] = eddy_radius_s
                    obs.shape_error_s[:] = aerr_s
                    obs.speed_area[:] = poly_area(
                        *coordinates_to_local(*xy_s, lon0=centlon_s, lat0=centlat_s)
                    )
                    obs.lon[:] = centlon_s
                    obs.lat[:] = centlat_s

                    # Third, compute effective radius, shape error, area from resampled effective contour
                    _, _, eddy_radius_e, aerr_e = _fit_circle_path(
                        create_vertice(*xy_e)
                    )
                    obs.radius_e[:] = eddy_radius_e
                    obs.shape_error_e[:] = aerr_e
                    obs.effective_area[:] = poly_area(
                        *coordinates_to_local(*xy_e, lon0=centlon_s, lat0=centlat_s)
                    )

                    # Finally, resample contours with output parameters
                    xy_e_f = resample(*xy_e, **out_sampling)
                    xy_s_f = resample(*xy_s, **out_sampling)

                    obs.contour_lon_s[:], obs.contour_lat_s[:] = xy_s_f
                    obs.contour_lon_e[:], obs.contour_lat_e[:] = xy_e_f

                    if aerr > 99.9 or aerr_s > 99.9:
                        logger.warning(
                            "Strange shape at this step! shape_error : %f, %f",
                            aerr,
                            aerr_s,
                        )

                    eddies.append(obs)
                    # To reserve definitively the area
                    data.mask[i_x_in, i_y_in] = True
            if len(eddies) == 0:
                eddies = EddiesObservations(
                    track_extra_variables=track_extra_variables,
                    track_array_variables=sampling,
                    array_variables=array_variables,
                )
            else:
                eddies = EddiesObservations.concatenate(eddies)
            eddies.sign_type = 1 if anticyclonic_search else -1
            eddies.time[:] = (date - datetime(1950, 1, 1)).total_seconds() / 86400.0

            # normalization longitude between 0 - 360, because storage have an offset on 180
            eddies.lon_max[:] %= 360
            eddies.lon[:] %= 360
            ref = eddies.lon - 180
            eddies.contour_lon_e[:] = ((eddies.contour_lon_e.T - ref) % 360 + ref).T
            eddies.contour_lon_s[:] = ((eddies.contour_lon_s.T - ref) % 360 + ref).T
            a_and_c.append(eddies)

        if in_h_unit is not None:
            for name in [
                "amplitude",
                "height_max_speed_contour",
                "height_external_contour",
                "height_inner_contour",
            ]:
                out_unit = units.parse_expression(VAR_DESCR[name]["nc_attr"]["units"])
                factor, _ = in_h_unit.to(out_unit).to_tuple()
                a_and_c[0].obs[name] *= factor
                a_and_c[1].obs[name] *= factor
        u_units = self.units(uname) if force_speed_unit is None else force_speed_unit
        in_u_units = units.parse_expression(u_units)
        if in_u_units is not None:
            for name in ["speed_average", "uavg_profile"]:
                out_unit = units.parse_expression(VAR_DESCR[name]["nc_attr"]["units"])
                factor, _ = in_u_units.to(out_unit).to_tuple()
                a_and_c[0].obs[name] *= factor
                a_and_c[1].obs[name] *= factor
        return a_and_c

    def get_uavg(
        self,
        all_contours,
        centlon_e,
        centlat_e,
        original_contour,
        anticyclonic_search,
        level_start,
        pixel_min=3,
    ):
        """
        Compute geostrophic speed around successive contours
        Returns the average
        """
        # Init max speed to search maximum
        max_average_speed = self.speed_coef_mean(original_contour)
        speed_array = [max_average_speed]

        eddy_contours = [original_contour]
        inner_contour = selected_contour = original_contour
        # Must start only on upper or lower contour, no need to test the two part
        step = 1 if anticyclonic_search else -1
        i_inner = i_max_speed = -1

        for i, coll in enumerate(
            all_contours.iter(start=level_start + step, step=step)
        ):
            level_contour = coll.get_nearest_path_bbox_contain_pt(centlon_e, centlat_e)
            # Leave loop if no contours at level
            if level_contour is None:
                break
            # Ensure polygon_i is within polygon_e
            if not poly_contain_poly(original_contour.vertices, level_contour.vertices):
                break
            # 3. Respect size range (for max speed)
            # nb_pixel properties need to call pixels_in before with a grid of pixel
            level_contour.pixels_in(self)
            # Interpolate uspd to seglon, seglat, then get mean
            level_average_speed = self.speed_coef_mean(level_contour)
            speed_array.append(level_average_speed)
            if (
                pixel_min < level_contour.nb_pixel
                and level_average_speed >= max_average_speed
            ):
                max_average_speed = level_average_speed
                i_max_speed = i
                selected_contour = level_contour
            inner_contour = level_contour
            eddy_contours.append(level_contour)
            i_inner = i
        for contour in eddy_contours:
            contour.used = True
        i_max_speed = level_start + step + step * i_max_speed
        i_inner = level_start + step + step * i_inner
        return (
            max_average_speed,
            selected_contour,
            inner_contour,
            array(speed_array),
            i_max_speed,
            i_inner,
        )

    @staticmethod
    def _gaussian_filter(data, sigma, mode="reflect"):
        """Standard gaussian filter"""
        local_data = data.copy()
        local_data[data.mask] = 0

        v = gaussian_filter(local_data, sigma=sigma, mode=mode)
        w = gaussian_filter(float_(~data.mask), sigma=sigma, mode=mode)

        with errstate(invalid="ignore"):
            return ma.array(v / w, mask=w == 0)

    @staticmethod
    def get_amplitude(
        contour, contour_height, data, anticyclonic_search=True, level=None, **kwargs
    ):
        # Instantiate Amplitude object
        amp = Amplitude(
            # Indices of all pixels in contour
            contour=contour,
            # Height of level
            contour_height=contour_height,
            # All grid
            data=data,
            **kwargs,
        )
        if anticyclonic_search:
            reset_centroid = amp.all_pixels_above_h0(level)
        else:
            reset_centroid = amp.all_pixels_below_h0(level)
        return reset_centroid, amp


class UnRegularGridDataset(GridDataset):
    """Class managing unregular grid"""

    __slots__ = (
        "index_interp",
        "_speed_norm",
    )

    def load(self):
        """Load variable (data)"""
        x_name, y_name = self.coordinates
        with Dataset(self.filename) as h:
            self.x_dim = h.variables[x_name].dimensions
            self.y_dim = h.variables[y_name].dimensions

            sl_x = [self.indexs.get(dim, slice(None)) for dim in self.x_dim]
            sl_y = [self.indexs.get(dim, slice(None)) for dim in self.y_dim]
            self.vars[x_name] = h.variables[x_name][sl_x]
            self.vars[y_name] = h.variables[y_name][sl_y]

            self.x_c = self.vars[x_name]
            self.y_c = self.vars[y_name]

            self.init_pos_interpolator()

    @property
    def bounds(self):
        """Give bounds"""
        return self.x_c.min(), self.x_c.max(), self.y_c.min(), self.y_c.max()

    def bbox_indice(self, vertices):
        dist, idx = self.index_interp.query(vertices, k=1)
        i_y = idx % self.x_c.shape[1]
        i_x = int_((idx - i_y) / self.x_c.shape[1])
        return (
            (max(i_x.min() - self.N, 0), i_x.max() + self.N + 1),
            (max(i_y.min() - self.N, 0), i_y.max() + self.N + 1),
        )

    def get_pixels_in(self, contour):
        (x_start, x_stop), (y_start, y_stop) = contour.bbox_slice
        return _get_pixel_in_unregular(
            contour.vertices, self.x_c, self.y_c, x_start, x_stop, y_start, y_stop
        )

    def normalize_x_indice(self, indices):
        """Not do"""
        return indices

    def nearest_grd_indice(self, x, y):
        dist, idx = self.index_interp.query((x, y), k=1)
        i_y = idx % self.x_c.shape[1]
        i_x = int_((idx - i_y) / self.x_c.shape[1])
        return i_x, i_y

    def compute_pixel_path(self, x0, y0, x1, y1):
        pass

    def init_pos_interpolator(self):
        logger.debug("Create a KdTree, could be long ...")
        self.index_interp = cKDTree(
            create_vertice(self.x_c.reshape(-1), self.y_c.reshape(-1))
        )

        logger.debug("... OK")

    def _low_filter(self, grid_name, w_cut, factor=8.0):
        data = self.grid(grid_name)
        x = self.grid(self.coordinates[0])
        y = self.grid(self.coordinates[1])
        regrid_step = w_cut / 111.0 / factor
        x_min, x_max, y_min, y_max = self.bounds
        x_array = arange(x_min, x_max + regrid_step, regrid_step)
        y_array = arange(y_min, min(y_max + regrid_step, 89), regrid_step)
        bins = (x_array, y_array)

        x_flat, y_flat, z_flat = x.reshape((-1,)), y.reshape((-1,)), data.reshape((-1,))
        m = ~self.get_mask(z_flat)
        x_flat, y_flat, z_flat = x_flat[m], y_flat[m], z_flat[m]

        nb_value, _, _ = histogram2d(x_flat, y_flat, bins=bins)

        sum_value, _, _ = histogram2d(x_flat, y_flat, bins=bins, weights=z_flat)

        with errstate(invalid="ignore"):
            z_grid = ma.array(sum_value / nb_value, mask=nb_value == 0)
        regular_grid = RegularGridDataset.with_array(
            coordinates=self.coordinates,
            datas={
                grid_name: z_grid,
                self.coordinates[0]: x_array[:-1],
                self.coordinates[1]: y_array[:-1],
            },
            centered=False,
        )
        regular_grid.bessel_low_filter(grid_name, w_cut, order=1)
        z_filtered = regular_grid.grid(grid_name)
        x_center = (x_array[:-1] + x_array[1:]) / 2
        y_center = (y_array[:-1] + y_array[1:]) / 2
        opts_interpolation = dict(kx=1, ky=1, s=0)
        m_interp = RectBivariateSpline(
            x_center, y_center, z_filtered.mask, **opts_interpolation
        )
        z_filtered.data[z_filtered.mask] = 0
        z_interp = RectBivariateSpline(
            x_center, y_center, z_filtered.data, **opts_interpolation
        ).ev(x, y)
        return ma.array(z_interp, mask=m_interp.ev(x, y) > 0.00001)

    def speed_coef_mean(self, contour):
        dist, idx = self.index_interp.query(
            uniform_resample_stack(contour.vertices)[1:], k=4
        )
        i_y = idx % self.x_c.shape[1]
        i_x = int_((idx - i_y) / self.x_c.shape[1])
        # A simplified solution to be change by a weight mean
        return self._speed_norm[i_x, i_y].mean(axis=1).mean()

    def init_speed_coef(self, uname="u", vname="v"):
        self._speed_norm = (self.grid(uname) ** 2 + self.grid(vname) ** 2) ** 0.5


class RegularGridDataset(GridDataset):
    """Class only for regular grid"""

    __slots__ = (
        "_speed_ev",
        "_is_circular",
        "x_size",
        "_x_step",
        "_y_step",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_circular = None

    def setup_coordinates(self):
        super().setup_coordinates()
        self.x_size = self.x_c.shape[0]
        if len(self.x_c.shape) != 1:
            raise Exception(
                "Coordinates in RegularGridDataset must be 1D array, or think to use UnRegularGridDataset"
            )
        dx = self.x_bounds[1:] - self.x_bounds[:-1]
        dy = self.y_bounds[1:] - self.y_bounds[:-1]
        if (dx < 0).any() or (dy < 0).any():
            raise Exception(
                "Coordinates in RegularGridDataset must be strictly increasing"
            )
        self._x_step = (self.x_c[1:] - self.x_c[:-1]).mean()
        self._y_step = (self.y_c[1:] - self.y_c[:-1]).mean()

    @classmethod
    def with_array(cls, coordinates, datas, variables_description=None, **kwargs):
        """
        Geo matrix data must be ordered like this (X,Y) and masked with numpy.ma.array
        """
        vd = dict() if variables_description is None else variables_description
        x_name, y_name = coordinates[0], coordinates[1]
        obj = cls("array", x_name, y_name, unset=True, **kwargs)
        obj.x_dim = (x_name,)
        obj.y_dim = (y_name,)
        obj.variables_description = dict()
        obj.dimensions = {i: v.shape[0] for i, v in datas.items() if i in coordinates}
        for k, v in datas.items():
            obj.vars[k] = v
            obj.variables_description[k] = dict(
                attrs=vd.get(k, dict()),
                args=(k, v.dtype),
                kwargs=dict(
                    dimensions=coordinates if k not in coordinates else (k,),
                    complevel=1,
                    zlib=True,
                ),
                infos=dict(),
            )
        obj.global_attrs = dict(history="Grid setup with an array")
        obj.setup_coordinates()
        return obj

    def bbox_indice(self, vertices):
        return bbox_indice_regular(
            vertices,
            self.x_bounds,
            self.y_bounds,
            self.xstep,
            self.ystep,
            self.N,
            self.is_circular(),
            self.x_size,
        )

    def get_pixels_in(self, contour):
        """
        Get indexes of pixels in contour.

        :param vertice,Path contour: Contour that encloses some pixels
        :return: Indexes of grid in contour
        :rtype: array[int],array[int]
        """
        if isinstance(contour, BasePath):
            (x_start, x_stop), (y_start, y_stop) = contour.bbox_slice
            return get_pixel_in_regular(
                contour.vertices, self.x_c, self.y_c, x_start, x_stop, y_start, y_stop
            )
        else:
            (x_start, x_stop), (y_start, y_stop) = self.bbox_indice(contour)
            return get_pixel_in_regular(
                contour, self.x_c, self.y_c, x_start, x_stop, y_start, y_stop
            )

    def normalize_x_indice(self, indices):
        return indices % self.x_size

    def nearest_grd_indice(self, x, y):
        return nearest_grd_indice(
            x, y, self.x_bounds, self.y_bounds, self.xstep, self.ystep
        )

    @property
    def xstep(self):
        """Only for regular grid with no step variation"""
        return self._x_step

    @property
    def ystep(self):
        """Only for regular grid with no step variation"""
        return self._y_step

    def compute_pixel_path(self, x0, y0, x1, y1):
        """Give a series of indexes describing the path between two positions"""
        return compute_pixel_path(
            x0,
            y0,
            x1,
            y1,
            self.x_bounds[0],
            self.y_bounds[0],
            self.xstep,
            self.ystep,
            self.x_size,
        )

    def clean_land(self):
        """Function to remove all land pixel"""
        pass

    def is_circular(self):
        """Check if the grid is circular"""
        if self._is_circular is None:
            self._is_circular = (
                abs((self.x_bounds[0] % 360) - (self.x_bounds[-1] % 360)) < 0.0001
            )
        return self._is_circular

    @staticmethod
    def check_order(order):
        if order < 1:
            logger.warning("order must be superior to 0")
        return ceil(order).astype(int)

    def get_step_in_km(self, lat, wave_length):
        step_y_km = self.ystep * distance(0, 0, 0, 1) / 1000
        step_x_km = self.xstep * distance(0, lat, 1, lat) / 1000
        min_wave_length = max(step_x_km, step_y_km) * 2
        if wave_length < min_wave_length:
            logger.error(
                "wave_length too short for resolution, must be > %d km",
                ceil(min_wave_length),
            )
            raise Exception()
        return step_x_km, step_y_km

    def estimate_kernel_shape(self, lat, wave_length, order):
        step_x_km, step_y_km = self.get_step_in_km(lat, wave_length)
        # half size will be multiply with by order
        half_x_pt, half_y_pt = (
            ceil(wave_length / step_x_km).astype(int),
            ceil(wave_length / step_y_km).astype(int),
        )
        # x size is not good over 60 degrees
        y = arange(
            lat - self.ystep * half_y_pt * order,
            lat + self.ystep * half_y_pt * order + 0.01 * self.ystep,
            self.ystep,
        )
        # We compute half + 1 and the other part will be compute by symetry
        x = arange(0, self.xstep * half_x_pt * order + 0.01 * self.xstep, self.xstep)
        y, x = meshgrid(y, x)
        dist_norm = distance(0, lat, x, y) / 1000.0 / wave_length
        return half_x_pt, half_y_pt, dist_norm

    def finalize_kernel(self, kernel, order, half_x_pt, half_y_pt):
        # Symetry
        kernel_ = empty((half_x_pt * 2 * order + 1, half_y_pt * 2 * order + 1))
        kernel_[half_x_pt * order :] = kernel
        kernel_[: half_x_pt * order] = kernel[:0:-1]
        # remove unused row/column
        k_valid = kernel_ != 0
        x_valid = where(k_valid.sum(axis=1))[0]
        x_slice = slice(x_valid[0], x_valid[-1] + 1)
        y_valid = where(k_valid.sum(axis=0))[0]
        y_slice = slice(y_valid[0], y_valid[-1] + 1)
        return kernel_[x_slice, y_slice]

    def kernel_lanczos(self, lat, wave_length, order=1):
        """Not really operational
        wave_length in km
        order must be int
        """
        order = self.check_order(order)
        half_x_pt, half_y_pt, dist_norm = self.estimate_kernel_shape(
            lat, wave_length, order
        )
        kernel = sinc(dist_norm / order) * sinc(dist_norm)
        kernel[dist_norm > order] = 0
        return self.finalize_kernel(kernel, order, half_x_pt, half_y_pt)

    def kernel_bessel(self, lat, wave_length, order=1):
        """wave_length in km
        order must be int
        """
        order = self.check_order(order)
        half_x_pt, half_y_pt, dist_norm = self.estimate_kernel_shape(
            lat, wave_length, order
        )
        with errstate(invalid="ignore"):
            kernel = sinc(dist_norm / order) * j1(2 * pi * dist_norm) / dist_norm
        kernel[0, half_y_pt * order] = pi
        kernel[dist_norm > order] = 0
        return self.finalize_kernel(kernel, order, half_x_pt, half_y_pt)

    def _low_filter(self, grid_name, w_cut, **kwargs):
        """low filtering"""
        return self.convolve_filter_with_dynamic_kernel(
            grid_name, self.kernel_bessel, wave_length=w_cut, **kwargs
        )

    def convolve_filter_with_dynamic_kernel(
        self, grid, kernel_func, lat_max=85, extend=False, **kwargs_func
    ):
        """
        :param str grid: grid name
        :param func kernel_func: function of kernel to use
        :param float lat_max: absolute latitude above no filtering apply
        :param bool extend: if False, only non masked value will return a filtered value
        :param dict kwargs_func: look at kernel_func
        :return: filtered value
        :rtype: array
        """
        if (abs(self.y_c) > lat_max).any():
            logger.warning("No filtering above %f degrees of latitude", lat_max)
        if isinstance(grid, str):
            data = self.grid(grid).copy()
        else:
            data = grid.copy()
        # Matrix for result
        data_out = ma.empty(data.shape)
        data_out.mask = ones(data_out.shape, dtype=bool)
        nb_lines = self.y_c.shape[0]
        dt = list()

        debug_active = logger.getEffectiveLevel() == logging.DEBUG

        for i, lat in enumerate(self.y_c):
            if abs(lat) > lat_max or data[:, i].mask.all():
                data_out.mask[:, i] = True
                continue
            # Get kernel
            kernel = kernel_func(lat, **kwargs_func)
            # Kernel shape
            k_shape = kernel.shape
            t0 = datetime.now()
            if debug_active and len(dt) > 0:
                dt_mean = np_mean(dt) * (nb_lines - i)
                print(
                    "Remain ",
                    dt_mean,
                    "ETA ",
                    t0 + dt_mean,
                    "current kernel size :",
                    k_shape,
                    "Step : %d/%d    " % (i, nb_lines),
                    end="\r",
                )

            # Half size, k_shape must be always impair
            d_lat = int((k_shape[1] - 1) / 2)
            d_lon = int((k_shape[0] - 1) / 2)
            # Temporary matrix to have exact shape at outuput
            tmp_matrix = ma.zeros((2 * d_lon + data.shape[0], k_shape[1]))
            tmp_matrix.mask = ones(tmp_matrix.shape, dtype=bool)
            # Slice to apply on input data
            # +1 for upper bound, to take in acount this column
            sl_lat_data = slice(max(0, i - d_lat), min(i + d_lat + 1, data.shape[1]))
            # slice to apply on temporary matrix to store input data
            sl_lat_in = slice(
                d_lat - (i - sl_lat_data.start), d_lat + (sl_lat_data.stop - i)
            )
            # If global => manual wrapping
            if self.is_circular():
                tmp_matrix[:d_lon, sl_lat_in] = data[-d_lon:, sl_lat_data]
                tmp_matrix[-d_lon:, sl_lat_in] = data[:d_lon, sl_lat_data]
            # Copy data
            tmp_matrix[d_lon:-d_lon, sl_lat_in] = data[:, sl_lat_data]
            # Convolution
            m = ~tmp_matrix.mask
            tmp_matrix[~m] = 0

            demi_x, demi_y = k_shape[0] // 2, k_shape[1] // 2
            values_sum = filter2D(tmp_matrix.data, -1, kernel)[demi_x:-demi_x, demi_y]
            kernel_sum = filter2D(m.astype(float), -1, kernel)[demi_x:-demi_x, demi_y]
            with errstate(invalid="ignore", divide="ignore"):
                if extend:
                    data_out[:, i] = ma.array(
                        values_sum / kernel_sum,
                        mask=kernel_sum < (extend * kernel.sum()),
                    )
                else:
                    data_out[:, i] = values_sum / kernel_sum
            dt.append(datetime.now() - t0)
            if len(dt) == 100:
                dt.pop(0)
        if extend:
            out = ma.array(data_out, mask=data_out.mask)
        else:
            out = ma.array(data_out, mask=data.mask + data_out.mask)
        if debug_active:
            print()
        if out.dtype != data.dtype:
            return out.astype(data.dtype)
        return out

    def lanczos_high_filter(
        self, grid_name, wave_length, order=1, lat_max=85, **kwargs
    ):
        logger.warning("It could be not safe to use lanczos filter")
        data_out = self.convolve_filter_with_dynamic_kernel(
            grid_name,
            self.kernel_lanczos,
            lat_max=lat_max,
            wave_length=wave_length,
            order=order,
            **kwargs,
        )
        self.vars[grid_name] -= data_out

    def lanczos_low_filter(self, grid_name, wave_length, order=1, lat_max=85, **kwargs):
        logger.warning("It could be not safe to use lanczos filter")
        data_out = self.convolve_filter_with_dynamic_kernel(
            grid_name,
            self.kernel_lanczos,
            lat_max=lat_max,
            wave_length=wave_length,
            order=order,
            **kwargs,
        )
        self.vars[grid_name] = data_out

    def bessel_band_filter(self, grid_name, wave_length_inf, wave_length_sup, **kwargs):
        data_out = self.convolve_filter_with_dynamic_kernel(
            grid_name, self.kernel_bessel, wave_length=wave_length_inf, **kwargs
        )
        self.vars[grid_name] = data_out
        data_out = self.convolve_filter_with_dynamic_kernel(
            grid_name, self.kernel_bessel, wave_length=wave_length_sup, **kwargs
        )
        self.vars[grid_name] -= data_out

    def bessel_high_filter(self, grid_name, wave_length, order=1, lat_max=85, **kwargs):
        """
        :param str grid_name: grid to filter, data will replace original one
        :param float wave_length: in km
        :param int order: order to use, if > 1 negative values of the cardinal sinus are present in kernel
        :param float lat_max: absolute latitude, no filtering above
        :param dict kwargs: look at :py:meth:`RegularGridDataset.convolve_filter_with_dynamic_kernel`

        .. minigallery:: py_eddy_tracker.RegularGridDataset.bessel_high_filter
        """
        logger.debug(
            "Run filtering with wavelength of %(wave_length)s km and order of %(order)s ...",
            dict(wave_length=wave_length, order=order),
        )
        data_out = self.convolve_filter_with_dynamic_kernel(
            grid_name,
            self.kernel_bessel,
            lat_max=lat_max,
            wave_length=wave_length,
            order=order,
            **kwargs,
        )
        logger.debug("Filtering done")
        self.vars[grid_name] -= data_out

    def bessel_low_filter(self, grid_name, wave_length, order=1, lat_max=85, **kwargs):
        data_out = self.convolve_filter_with_dynamic_kernel(
            grid_name,
            self.kernel_bessel,
            lat_max=lat_max,
            wave_length=wave_length,
            order=order,
            **kwargs,
        )
        self.vars[grid_name] = data_out

    def spectrum_lonlat(self, grid_name, area=None, ref=None, **kwargs):
        if area is None:
            area = dict(llcrnrlon=190, urcrnrlon=280, llcrnrlat=-62, urcrnrlat=8)
        scaling = kwargs.pop("scaling", "density")
        ref_grid_name = kwargs.pop("ref_grid_name", None)
        x0, y0 = self.nearest_grd_indice(area["llcrnrlon"], area["llcrnrlat"])
        x1, y1 = self.nearest_grd_indice(area["urcrnrlon"], area["urcrnrlat"])

        data = self.grid(grid_name)[x0:x1, y0:y1]

        # Lat spectrum
        pws = list()
        step_y_km = self.ystep * distance(0, 0, 0, 1) / 1000
        nb_invalid = 0
        for i, _ in enumerate(self.x_c[x0:x1]):
            f, pw = welch(data[i, :], 1 / step_y_km, scaling=scaling, **kwargs)
            if isnan(pw).any():
                nb_invalid += 1
                continue
            pws.append(pw)
        if nb_invalid:
            logger.warning("%d/%d columns invalid", nb_invalid, i + 1)
        with errstate(divide="ignore"):
            lat_content = 1 / f, array(pws).mean(axis=0)

        # Lon spectrum
        fs, pws = list(), list()
        f_min, f_max = None, None
        nb_invalid = 0
        for i, lat in enumerate(self.y_c[y0:y1]):
            step_x_km = self.xstep * distance(0, lat, 1, lat) / 1000
            f, pw = welch(data[:, i], 1 / step_x_km, scaling=scaling, **kwargs)
            if isnan(pw).any():
                nb_invalid += 1
                continue
            if f_min is None:
                f_min = f.min()
                f_max = f.max()
            else:
                f_min = max(f_min, f.min())
                f_max = min(f_max, f.max())
            fs.append(f)
            pws.append(pw)
        if nb_invalid:
            logger.warning("%d/%d lines invalid", nb_invalid, i + 1)
        f_interp = linspace(f_min, f_max, f.shape[0])
        pw_m = array(
            [
                interp1d(f, pw, fill_value=0.0, bounds_error=False)(f_interp)
                for f, pw in zip(fs, pws)
            ]
        ).mean(axis=0)
        with errstate(divide="ignore"):
            lon_content = 1 / f_interp, pw_m
        if ref is None:
            return lon_content, lat_content
        else:
            if ref_grid_name is not None:
                grid_name = ref_grid_name
            ref_lon_content, ref_lat_content = ref.spectrum_lonlat(
                grid_name, area, **kwargs
            )
            return (
                (lon_content[0], lon_content[1] / ref_lon_content[1]),
                (lat_content[0], lat_content[1] / ref_lat_content[1]),
            )

    def compute_finite_difference(self, data, schema=1, mode="reflect", vertical=False):
        if not isinstance(schema, int) and schema < 1:
            raise Exception("schema must be a positive int")

        data2 = data.copy()
        data1 = data.copy()
        if vertical:
            data1[:, :-schema] = data[:, schema:]
            data2[:, schema:] = data[:, :-schema]
            # put nan
            data1[:, -schema:] = nan
            data2[:, :schema] = nan
        else:
            data1[:-schema] = data[schema:]
            data2[schema:] = data[:-schema]
            if mode == "wrap":
                data1[-schema:] = data[:schema]
                data2[:schema] = data[-schema:]
            else:
                # put nan
                data1[-schema:] = nan
                data2[:schema] = nan

        d = self.EARTH_RADIUS * 2 * pi / 360 * 2 * schema
        if vertical:
            d *= self.ystep
        else:
            d *= self.xstep * cos(deg2rad(self.y_c))
        return (data1 - data2) / d

    def compute_stencil(
        self, data, stencil_halfwidth=4, mode="reflect", vertical=False
    ):
        r"""
        Apply stencil ponderation on field.

        :param array data: array where apply stencil
        :param int stencil_halfwidth: from 1 t0 4, maximal stencil used
        :param str mode: convolution mode
        :param bool vertical: if True, method apply a vertical convolution
        :return: gradient array from stencil application
        :rtype: array

        Short story, how to get stencil coefficient for stencil (3 points, 5 points and 7 points)

        Taylor's theorem:

        .. math::
            f(x \pm h) = f(x) \pm f'(x)h
                + \frac{f''(x)h^2}{2!} \pm \frac{f^{(3)}(x)h^3}{3!}
                + \frac{f^{(4)}(x)h^4}{4!} \pm \frac{f^{(5)}(x)h^5}{5!}
                + O(h^6)

        If we stop at `O(h^2)`, we get classic differenciation (stencil 3 points):

        .. math:: f(x+h) - f(x-h) = f(x) - f(x) + 2 f'(x)h + O(h^2)

        .. math:: f'(x) = \frac{f(x+h) - f(x-h)}{2h} + O(h^2)

        If we stop at `O(h^4)`, we will get stencil 5 points:

        .. math::
            f(x+h) - f(x-h) = 2 f'(x)h + 2 \frac{f^{(3)}(x)h^3}{3!} + O(h^4)
            :label: E1

        .. math::
            f(x+2h) - f(x-2h) = 4 f'(x)h + 16 \frac{f^{(3)}(x)h^3}{3!} + O(h^4)
            :label: E2

        If we multiply equation :eq:`E1` by 8 and substract equation :eq:`E2`, we get:

        .. math:: 8(f(x+h) - f(x-h)) - (f(x+2h) - f(x-2h)) = 16 f'(x)h - 4 f'(x)h + O(h^4)

        .. math:: f'(x) = \frac{f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)}{12h} + O(h^4)

        If we stop at `O(h^6)`, we will get stencil 7 points:

        .. math::
            f(x+h) - f(x-h) = 2 f'(x)h + 2 \frac{f^{(3)}(x)h^3}{3!} + 2 \frac{f^{(5)}(x)h^5}{5!} + O(h^6)
            :label: E3

        .. math::
            f(x+2h) - f(x-2h) = 4 f'(x)h + 16 \frac{f^{(3)}(x)h^3}{3!} + 64 \frac{f^{(5)}(x)h^5}{5!} + O(h^6)
            :label: E4

        .. math::
            f(x+3h) - f(x-3h) = 6 f'(x)h + 54 \frac{f^{(3)}(x)h^3}{3!} + 486 \frac{f^{(5)}(x)h^5}{5!} + O(h^6)
            :label: E5

        If we multiply equation :eq:`E3` by 45 and substract equation :eq:`E4` multiply by 9
        and add equation :eq:`E5`, we get:

        .. math::
            45(f(x+h) - f(x-h)) - 9(f(x+2h) - f(x-2h)) + (f(x+3h) - f(x-3h)) =
            90 f'(x)h - 36 f'(x)h + 6 f'(x)h + O(h^6)

        .. math::
            f'(x) = \frac{-f(x-3h) + 9f(x-2h) - 45f(x-h) + 45f(x+h) - 9f(x+2h) +f(x+3h)}{60h} + O(h^6)

        ...

        """
        stencil_halfwidth = max(min(int(stencil_halfwidth), 4), 1)
        logger.debug("Stencil half width apply : %d", stencil_halfwidth)
        g, m = compute_stencil(
            self.x_c,
            self.y_c,
            data.data,
            self.get_mask(data),
            self.EARTH_RADIUS,
            vertical=vertical,
            stencil_halfwidth=stencil_halfwidth,
        )
        return ma.array(g, mask=m)

    def add_uv_lagerloef(self, grid_height, uname="u", vname="v", schema=15):
        self.add_uv(grid_height, uname, vname)
        latmax = 5
        _, (i_start, i_end) = self.nearest_grd_indice((0, 0), (-latmax, latmax))
        sl = slice(i_start, i_end)
        # Divide by sideral day
        lat = self.y_c[sl]
        gob = (
            cos(deg2rad(lat))
            * ones((self.x_c.shape[0], 1))
            * 4.0
            * pi
            / (23 * 3600 + 56 * 60 + 4.1)
            / self.EARTH_RADIUS
        )
        with errstate(divide="ignore"):
            gob = self.GRAVITY / (gob * ones((self.x_c.shape[0], 1)))
        mode = "wrap" if self.is_circular() else "reflect"

        # fill data to compute a finite difference on all point
        data = self.convolve_filter_with_dynamic_kernel(
            grid_height,
            self.kernel_bessel,
            lat_max=10,
            wave_length=500,
            order=1,
            extend=0.1,
        )
        data = self.convolve_filter_with_dynamic_kernel(
            data, self.kernel_bessel, lat_max=10, wave_length=500, order=1, extend=0.1
        )
        data = self.convolve_filter_with_dynamic_kernel(
            data, self.kernel_bessel, lat_max=10, wave_length=500, order=1, extend=0.1
        )
        v_lagerloef = (
            self.compute_finite_difference(
                self.compute_finite_difference(data, mode=mode, schema=schema),
                mode=mode,
                schema=schema,
            )[:, sl]
            * gob
        )
        u_lagerloef = (
            -self.compute_finite_difference(
                self.compute_finite_difference(data, vertical=True, schema=schema),
                vertical=True,
                schema=schema,
            )[:, sl]
            * gob
        )
        w = 1 - exp(-((lat / 2.2) ** 2))
        self.vars[vname][:, sl] = self.vars[vname][:, sl] * w + v_lagerloef * (1 - w)
        self.vars[uname][:, sl] = self.vars[uname][:, sl] * w + u_lagerloef * (1 - w)

    def add_uv(self, grid_height, uname="u", vname="v", stencil_halfwidth=4):
        r"""Compute a u and v grid

        :param str grid_height: grid name where the funtion will apply stencil method
        :param str uname: future name of u
        :param str vname: future name of v
        :param int stencil_halfwidth: largest stencil could be apply (max: 4)

        .. math::
            u = \frac{g}{f} \frac{dh}{dy}

            v = -\frac{g}{f} \frac{dh}{dx}

        where

        .. math::
            g = gravity

            f = 2 \Omega sin(\phi)


        .. minigallery:: py_eddy_tracker.RegularGridDataset.add_uv
        """
        logger.info("Add u/v variable with stencil method")
        data = self.grid(grid_height)
        h_dict = self.variables_description[grid_height]
        for variable in (uname, vname):
            self.variables_description[variable] = dict(
                infos=h_dict["infos"].copy(),
                attrs=h_dict["attrs"].copy(),
                args=tuple((variable, *h_dict["args"][1:])),
                kwargs=h_dict["kwargs"].copy(),
            )
            if "units" in self.variables_description[variable]["attrs"]:
                self.variables_description[variable]["attrs"]["units"] += "/s"
            if "long_name" in self.variables_description[variable]["attrs"]:
                self.variables_description[variable]["attrs"][
                    "long_name"
                ] += " gradient"
        # Divide by sideral day
        gof = (
            sin(deg2rad(self.y_c))
            * ones((self.x_c.shape[0], 1))
            * 4.0
            * pi
            / (23 * 3600 + 56 * 60 + 4.1)
        )
        with errstate(divide="ignore"):
            gof = self.GRAVITY / (gof * ones((self.x_c.shape[0], 1)))

        # Compute v
        mode = "wrap" if self.is_circular() else "reflect"
        self.vars[vname] = (
            self.compute_stencil(data, mode=mode, stencil_halfwidth=stencil_halfwidth)
            * gof
        )
        # Compute u
        self.vars[uname] = (
            -self.compute_stencil(
                data, vertical=True, stencil_halfwidth=stencil_halfwidth
            )
            * gof
        )

    def speed_coef_mean(self, contour):
        """Some nan can be computed over contour if we are near borders,
        something to explore
        """
        return mean_on_regular_contour(
            self.x_c,
            self.y_c,
            self._speed_ev,
            self._speed_ev.mask,
            contour.vertices,
            nan_remove=True,
        )

    def init_speed_coef(self, uname="u", vname="v"):
        """Draft"""
        self._speed_ev = (self.grid(uname) ** 2 + self.grid(vname) ** 2) ** 0.5

    def display(self, ax, name, factor=1, ref=None, **kwargs):
        """
        :param matplotlib.axes.Axes ax: matplotlib axes used to draw
        :param str,array name: variable to display, could be an array
        :param float factor: multiply grid by
        :param float,None ref: if defined, all coordinates are wrapped with ref as western boundary
        :param dict kwargs: look at :py:meth:`matplotlib.axes.Axes.pcolormesh`

        .. minigallery:: py_eddy_tracker.RegularGridDataset.display
        """
        if "cmap" not in kwargs:
            kwargs["cmap"] = "coolwarm"
        data = self.grid(name) if isinstance(name, str) else name
        if ref is None:
            x = self.x_bounds
        else:
            x = (self.x_c - ref) % 360 + ref
            i = x.argsort()
            x = self.c_to_bounds(x[i])
            data = data[i]
        return ax.pcolormesh(x, self.y_bounds, data.T * factor, **kwargs)

    def contour(self, ax, name, factor=1, ref=None, **kwargs):
        """
        :param matplotlib.axes.Axes ax: matplotlib axes used to draw
        :param str,array name: variable to display, could be an array
        :param float factor: multiply grid by
        :param float,None ref: if defined, all coordinates are wrapped with ref as western boundary
        :param dict kwargs: look at :py:meth:`matplotlib.axes.Axes.contour`

        .. minigallery:: py_eddy_tracker.RegularGridDataset.contour
        """
        data = self.grid(name) if isinstance(name, str) else name
        if ref is None:
            x = self.x_c
        else:
            x = (self.x_c - ref) % 360 + ref
            i = x.argsort()
            x = x[i]
            data = data[i]
        return ax.contour(x, self.y_c, data.T * factor, **kwargs)

    def regrid(self, other, grid_name, new_name=None):
        """
        Interpolate another grid at the current grid position

        :param RegularGridDataset other:
        :param str grid_name: variable name to interpolate
        :param str new_name: name used to store, if None method will use current ont

        .. minigallery:: py_eddy_tracker.RegularGridDataset.regrid
        """
        if new_name is None:
            new_name = grid_name
        x, y = meshgrid(self.x_c, self.y_c)
        # interp and reshape
        v_interp = (
            other.interp(grid_name, x.reshape(-1), y.reshape(-1)).reshape(x.shape).T
        )
        v_interp = ma.array(v_interp, mask=isnan(v_interp))
        # and add it to self
        self.add_grid(new_name, v_interp)
        self.variables_description[new_name] = other.variables_description[grid_name]
        # self.variables_description[new_name]['infos'] = False
        # self.variables_description[new_name]['kwargs']['dimensions'] = ...

    def interp(self, grid_name, lons, lats, method="bilinear"):
        """
        Compute z over lons, lats

        :param str grid_name: Grid to be interpolated
        :param lons: new x
        :param lats: new y
        :param str method: Could be 'bilinear' or 'nearest'

        :return: new z
        """
        g = self.grid(grid_name)
        m = self.get_mask(g)
        return interp2d_geo(
            self.x_c, self.y_c, g, m, lons, lats, nearest=method == "nearest"
        )

    def uv_for_advection(self, u_name=None, v_name=None, time_step=600, h_name=None, backward=False, factor=1):
        """
        Get U,V to be used in degrees with precomputed time step

        :param None,str,array u_name: U field to advect obs, if h_name is None
        :param None,str,array v_name: V field to advect obs, if h_name is None
        :param None,str,array h_name: H field to compute UV to advect obs, if u_name and v_name are None
        :param int time_step: Number of second for each advection
        """
        if h_name is not None:
            u_name, v_name = 'u', 'v'
            if u_name not in self.vars:
                self.add_uv(h_name)
                self.vars.pop(h_name, None)

        u = (self.grid(u_name) if isinstance(u_name, str) else u_name).copy()
        v = (self.grid(v_name) if isinstance(v_name, str) else v_name).copy()
        # N seconds / 1 degrees in m
        coef = time_step * 180 / pi / self.EARTH_RADIUS * factor
        u *= coef / cos(radians(self.y_c))
        v *= coef
        if backward:
            u = -u
            v = -v
        m = u.mask + v.mask
        return u, v, m

    def advect(self, x, y, u_name, v_name, nb_step=10, rk4=True, **kw):
        """
        At each call it will update position in place with u & v field

        It's a dummy advection using only one layer of current

        :param array x: Longitude of obs to move
        :param array y: Latitude of obs to move
        :param str,array u_name: U field to advect obs
        :param str,array v_name: V field to advect obs
        :param int nb_step: Number of iterations before releasing data

        .. minigallery:: py_eddy_tracker.GridDataset.advect
        """
        u, v, m = self.uv_for_advection(u_name, v_name, **kw)
        m_p = isnan(x) + isnan(y)
        advect_ = advect_rk4 if rk4 else advect
        while True:
            advect_(self.x_c, self.y_c, u, v, m, x, y, m_p, nb_step)
            yield x, y

    def filament(
        self, x, y, u_name, v_name, nb_step=10, filament_size=6, rk4=True, **kw
    ):
        """
        Produce filament with concatenation of advection

        It's a dummy advection using only one layer of current

        :param array x: Longitude of obs to move
        :param array y: Latitude of obs to move
        :param str,array u_name: U field to advect obs
        :param str,array v_name: V field to advect obs
        :param int nb_step: Number of iteration before releasing data
        :param int filament_size: Number of point by filament
        :return: x,y for a line

        .. minigallery:: py_eddy_tracker.GridDataset.filament
        """
        u, v, m = self.uv_for_advection(u_name, v_name, **kw)
        x, y = x.copy(), y.copy()
        nb = x.shape[0]
        filament_size_ = filament_size + 1
        f_x = empty(nb * filament_size_, dtype="f4")
        f_y = empty(nb * filament_size_, dtype="f4")
        f_x[:] = nan
        f_y[:] = nan
        f_x[::filament_size_] = x
        f_y[::filament_size_] = y
        mp = isnan(x) + isnan(y)
        advect_ = advect_rk4 if rk4 else advect
        while True:
            # Shift position
            f_x[1:] = f_x[:-1]
            f_y[1:] = f_y[:-1]
            # Remove last position
            f_x[filament_size::filament_size_] = nan
            f_y[filament_size::filament_size_] = nan
            advect_(self.x_c, self.y_c, u, v, m, x, y, mp, nb_step)
            f_x[::filament_size_] = x
            f_y[::filament_size_] = y
            yield f_x, f_y


@njit(cache=True)
def advect_rk4(x_g, y_g, u_g, v_g, m_g, x, y, m, nb_step):
    # Grid coordinates
    x_ref, y_ref = x_g[0], y_g[0]
    x_step, y_step = x_g[1] - x_ref, y_g[1] - y_ref
    is_circular = abs(x_g[-1] % 360 - (x_g[0] - x_step) % 360) < 1e-5
    nb_x_ = x_g.size
    nb_x = nb_x_ if is_circular else 0
    # cache
    i_cache, j_cache = -1000000, -1000000
    masked = False
    u00, u01, u10, u11 = 0.0, 0.0, 0.0, 0.0
    v00, v01, v10, v11 = 0.0, 0.0, 0.0, 0.0
    # On each particle
    for i in prange(x.size):
        # If particle is not valid => continue
        if m[i]:
            continue
        x_, y_ = x[i], y[i]
        # Iterate on whole steps
        for _ in range(nb_step):
            # k1, slope at origin
            ii_, jj_, xd, yd = get_grid_indices(
                x_ref, y_ref, x_step, y_step, x_, y_, nb_x
            )
            if ii_ != i_cache or jj_ != j_cache:
                i_cache, j_cache = ii_, jj_
                if not is_circular and (ii_ < 0 or ii_ > nb_x_):
                    masked = True
                else:
                    masked, u00, u01, u10, u11, v00, v01, v10, v11 = get_uv_quad(
                        ii_, jj_, u_g, v_g, m_g, nb_x
                    )
            # The 3 following could be in cache operation but this one must be tested in any case
            if masked:
                x_, y_ = nan, nan
                m[i] = True
                break
            u1, v1 = interp_uv(xd, yd, u00, u01, u10, u11, v00, v01, v10, v11)
            # k2, slope at middle with first guess position
            x1, y1 = x_ + u1 * 0.5, y_ + v1 * 0.5
            ii_, jj_, xd, yd = get_grid_indices(
                x_ref, y_ref, x_step, y_step, x1, y1, nb_x
            )
            if ii_ != i_cache or jj_ != j_cache:
                i_cache, j_cache = ii_, jj_
                if not is_circular and (ii_ < 0 or ii_ > nb_x_):
                    masked = True
                else:
                    masked, u00, u01, u10, u11, v00, v01, v10, v11 = get_uv_quad(
                        ii_, jj_, u_g, v_g, m_g, nb_x
                    )
                if masked:
                    x_, y_ = nan, nan
                    m[i] = True
                    break
            u2, v2 = interp_uv(xd, yd, u00, u01, u10, u11, v00, v01, v10, v11)
            # k3, slope at middle with updated guess position
            x2, y2 = x_ + u2 * 0.5, y_ + v2 * 0.5
            ii_, jj_, xd, yd = get_grid_indices(
                x_ref, y_ref, x_step, y_step, x2, y2, nb_x
            )
            if ii_ != i_cache or jj_ != j_cache:
                i_cache, j_cache = ii_, jj_
                if not is_circular and (ii_ < 0 or ii_ > nb_x_):
                    masked = True
                else:
                    masked, u00, u01, u10, u11, v00, v01, v10, v11 = get_uv_quad(
                        ii_, jj_, u_g, v_g, m_g, nb_x
                    )
                if masked:
                    x_, y_ = nan, nan
                    m[i] = True
                    break
            u3, v3 = interp_uv(xd, yd, u00, u01, u10, u11, v00, v01, v10, v11)
            # k4, slope at end with updated guess position
            x3, y3 = x_ + u3, y_ + v3
            ii_, jj_, xd, yd = get_grid_indices(
                x_ref, y_ref, x_step, y_step, x3, y3, nb_x
            )
            if ii_ != i_cache or jj_ != j_cache:
                i_cache, j_cache = ii_, jj_
                if not is_circular and (ii_ < 0 or ii_ > nb_x_):
                    masked = True
                else:
                    masked, u00, u01, u10, u11, v00, v01, v10, v11 = get_uv_quad(
                        ii_, jj_, u_g, v_g, m_g, nb_x
                    )
                if masked:
                    x_, y_ = nan, nan
                    m[i] = True
                    break
            u4, v4 = interp_uv(xd, yd, u00, u01, u10, u11, v00, v01, v10, v11)
            # RK4 compute
            dx = (u1 + 2 * u2 + 2 * u3 + u4) / 6
            dy = (v1 + 2 * v2 + 2 * v3 + v4) / 6
            # Compute new x,y
            x_ += dx
            y_ += dy
        x[i] = x_
        y[i] = y_


@njit(cache=True)
def advect(x_g, y_g, u_g, v_g, m_g, x, y, m, nb_step):
    # Grid coordinates
    x_ref, y_ref = x_g[0], y_g[0]
    x_step, y_step = x_g[1] - x_ref, y_g[1] - y_ref
    is_circular = abs(x_g[-1] % 360 - (x_g[0] - x_step) % 360) < 1e-5
    nb_x_ = x_g.size
    nb_x = nb_x_ if is_circular else 0
    # Indexes which should be never exist
    i0_old, j0_old = -100000, -100000
    masked = False
    u00, u01, u10, u11 = 0.0, 0.0, 0.0, 0.0
    v00, v01, v10, v11 = 0.0, 0.0, 0.0, 0.0
    # On each particule
    for i in prange(x.size):
        # If particule is not valid => continue
        if m[i]:
            continue
        # Iterate on whole steps
        for _ in range(nb_step):
            i0, j0, xd, yd = get_grid_indices(
                x_ref, y_ref, x_step, y_step, x[i], y[i], nb_x
            )
            # corners are the same, need only a new xd and yd
            if i0 != i0_old or j0 != j0_old:
                # Need to be stored only on change
                i0_old, j0_old = i0, j0
                if not is_circular and (i0 < 0 or i0 > nb_x_):
                    masked = True
                else:
                    masked, u00, u01, u10, u11, v00, v01, v10, v11 = get_uv_quad(
                        i0, j0, u_g, v_g, m_g, nb_x
                    )
            if masked:
                x[i], y[i] = nan, nan
                m[i] = True
                break
            u, v = interp_uv(xd, yd, u00, u01, u10, u11, v00, v01, v10, v11)
            # Compute new x,y
            x[i] += u
            y[i] += v


@njit(cache=True, fastmath=True)
def compute_pixel_path(x0, y0, x1, y1, x_ori, y_ori, x_step, y_step, nb_x):
    """Give a serie of indexes describing the path between two positions"""
    # index
    nx = x0.shape[0]
    i_x0 = empty(nx, dtype=numba_types.int_)
    i_x1 = empty(nx, dtype=numba_types.int_)
    i_y0 = empty(nx, dtype=numba_types.int_)
    i_y1 = empty(nx, dtype=numba_types.int_)
    # Because round_ is not accepted with array in numba
    for i in range(nx):
        i_x0[i] = round_(((x0[i] - x_ori) % 360) / x_step)
        i_x1[i] = round_(((x1[i] - x_ori) % 360) / x_step)
        i_y0[i] = round_((y0[i] - y_ori) / y_step)
        i_y1[i] = round_((y1[i] - y_ori) / y_step)
    # Delta index of x
    d_x = i_x1 - i_x0
    d_x = (d_x + nb_x // 2) % nb_x - (nb_x // 2)
    i_x1 = i_x0 + d_x
    # Delta index of y
    d_y = i_y1 - i_y0
    # max and abs sum do not work on array?
    d_max = empty(nx, dtype=numba_types.int32)
    nb_value = 0
    for i in range(nx):
        d_max[i] = max(abs(d_x[i]), abs(d_y[i]))
        # Compute number of pixel we go trought
        nb_value += d_max[i] + 1

    # Create an empty array to store value of pixel across the path
    i_g = empty(nb_value, dtype=numba_types.int32)
    j_g = empty(nb_value, dtype=numba_types.int32)

    # Index to determine the position in the global array
    ii = 0
    # Iteration on each path
    for i, delta in enumerate(d_max):
        # If the path doesn't cross multiple pixels
        if delta == 0:
            i_g[ii : ii + delta + 1] = i_x0[i]
            j_g[ii : ii + delta + 1] = i_y0[i]
        # Vertical move
        elif d_x[i] == 0:
            sup = -1 if d_y[i] < 0 else 1
            i_g[ii : ii + delta + 1] = i_x0[i]
            j_g[ii : ii + delta + 1] = arange(i_y0[i], i_y1[i] + sup, sup)
        # Horizontal move
        elif d_y[i] == 0:
            sup = -1 if d_x[i] < 0 else 1
            i_g[ii : ii + delta + 1] = arange(i_x0[i], i_x1[i] + sup, sup)
            j_g[ii : ii + delta + 1] = i_y0[i]
        # In case of multiple directions
        else:
            a = (i_x1[i] - i_x0[i]) / float(i_y1[i] - i_y0[i])
            if abs(d_x[i]) >= abs(d_y[i]):
                sup = -1 if d_x[i] < 0 else 1
                value = arange(i_x0[i], i_x1[i] + sup, sup)
                i_g[ii : ii + delta + 1] = value
                j_g[ii : ii + delta + 1] = (value - i_x0[i]) / a + i_y0[i]
            else:
                sup = -1 if d_y[i] < 0 else 1
                value = arange(i_y0[i], i_y1[i] + sup, sup)
                j_g[ii : ii + delta + 1] = value
                i_g[ii : ii + delta + 1] = (value - i_y0[i]) * a + i_x0[i]
        ii += delta + 1
    i_g %= nb_x
    return i_g, j_g, d_max


@njit(cache=True)
def has_masked_value(grid, i_x, i_y):
    for i, j in zip(i_x, i_y):
        if grid[i, j]:
            return True
    return False


@njit(cache=True)
def has_value(grid, i_x, i_y, value, below=False):
    for i, j in zip(i_x, i_y):
        if below:
            if grid[i, j] < value:
                return True
        else:
            if grid[i, j] > value:
                return True
    return False


class GridCollection:
    def __init__(self):
        self.datasets = list()

    @classmethod
    def from_netcdf_cube(cls, filename, x_name, y_name, t_name, heigth=None, **kwargs):
        new = cls()
        with Dataset(filename) as h:
            for i, t in enumerate(h.variables[t_name][:]):
                d = RegularGridDataset(
                    filename, x_name, y_name, indexs={t_name: i}, **kwargs
                )
                if heigth is not None:
                    d.add_uv(heigth)
                new.datasets.append((t, d))
        return new

    @classmethod
    def from_netcdf_list(
        cls, filenames, t, x_name, y_name, indexs=None, heigth=None, **kwargs
    ):
        new = cls()
        for i, _t in enumerate(t):
            filename = filenames[i]
            logger.debug(f"load file {i:02d}/{len(t)} t={_t} : {filename}")
            d = RegularGridDataset(filename, x_name, y_name, indexs=indexs, **kwargs)
            if heigth is not None:
                d.add_uv(heigth)
            new.datasets.append((_t, d))
        return new

    @property
    def are_loaded(self):
        return ~array([d.dimensions is None for _, d in self.datasets])

    def __repr__(self):
        nb_dataset = len(self.datasets)
        return f"{self.are_loaded.sum()}/{nb_dataset} datasets loaded"

    def shift_files(self, t, filename, heigth=None, **rgd_kwargs):
        """Add next file to the list and remove the oldest"""

        self.datasets = self.datasets[1:]

        d = RegularGridDataset(filename, **rgd_kwargs)
        if heigth is not None:
            d.add_uv(heigth)
        self.datasets.append((t, d))
        logger.debug(f"shift and adding i={len(self.datasets)} t={t} : {filename}")

    def interp(self, grid_name, t, lons, lats, method="bilinear"):
        """
        Compute z over lons, lats

        :param str grid_name: Grid to be interpolated
        :param float, t: time for interpolation
        :param lons: new x
        :param lats: new y
        :param str method: Could be 'bilinear' or 'nearest'

        :return: new z
        """
        # FIXME: we do assumption on time step
        t0 = int(t)
        t1 = t0 + 1
        h0, h1 = self[t0], self[t1]
        g0, g1 = h0.grid(grid_name), h1.grid(grid_name)
        m0, m1 = h0.get_mask(g0), h0.get_mask(g1)
        kw = dict(x=lons, y=lats, nearest=method == "nearest")
        v0 = interp2d_geo(h0.x_c, h0.y_c, g0, m0, **kw)
        v1 = interp2d_geo(h1.x_c, h1.y_c, g1, m1, **kw)
        w = (t - t0) / (t1 - t0)
        return v1 * w + v0 * (1 - w)

    def __iter__(self):
        for _, d in self.datasets:
            yield d

    def __getitem__(self, item):
        for t, d in self.datasets:
            if t == item:
                d.populate()
                return d
        raise KeyError(item)

    def filament(
        self,
        x,
        y,
        u_name,
        v_name,
        t_init,
        nb_step=10,
        time_step=600,
        filament_size=6,
        rk4=True,
        **kw,
    ):
        """
        Produce filament with concatenation of advection

        :param array x: Longitude of obs to move
        :param array y: Latitude of obs to move
        :param str,array u_name: U field to advect obs
        :param str,array v_name: V field to advect obs
        :param int nb_step: Number of iteration before to release data
        :param int time_step: Number of second for each advection
        :param int filament_size: Number of point by filament
        :return: x,y for a line

        .. minigallery:: py_eddy_tracker.GridCollection.filament
        """
        x, y = x.copy(), y.copy()
        nb = x.shape[0]
        filament_size_ = filament_size + 1
        f_x = empty(nb * filament_size_, dtype="f4")
        f_y = empty(nb * filament_size_, dtype="f4")
        f_x[:] = nan
        f_y[:] = nan
        f_x[::filament_size_] = x
        f_y[::filament_size_] = y

        backward = kw.get("backward", False)
        if backward:
            generator = self.get_previous_time_step(t_init)
            dt = -nb_step * time_step
            t_step = -time_step
        else:
            generator = self.get_next_time_step(t_init)
            dt = nb_step * time_step
            t_step = time_step
        t0, d0 = generator.__next__()
        u0, v0, m0 = d0.uv_for_advection(u_name, v_name, time_step, **kw)
        t1, d1 = generator.__next__()
        u1, v1, m1 = d1.uv_for_advection(u_name, v_name, time_step, **kw)
        t0 = t0 * 86400
        t1 = t1 * 86400
        t = t_init * 86400
        mp = isnan(x) + isnan(y)
        advect_ = advect_t_rk4 if rk4 else advect_t
        while True:
            # Shift position
            f_x[1:] = f_x[:-1]
            f_y[1:] = f_y[:-1]
            # Remove last position
            f_x[filament_size::filament_size_] = nan
            f_y[filament_size::filament_size_] = nan

            if (backward and t <= t1) or (not backward and t >= t1):
                t0, u0, v0, m0 = t1, u1, v1, m1
                t1, d1 = generator.__next__()
                t1 = t1 * 86400
                u1, v1, m1 = d1.uv_for_advection(u_name, v_name, time_step, **kw)
            w = 1 - (arange(t, t + dt, t_step) - t0) / (t1 - t0)
            half_w = t_step / 2.0 / (t1 - t0)
            advect_(d0.x_c, d0.y_c, u0, v0, m0, u1, v1, m1, x, y, mp, w, half_w=half_w)
            f_x[::filament_size_] = x
            f_y[::filament_size_] = y
            t += dt
            yield t, f_x, f_y

    def reset_grids(self, N=None):
        if N is not None:
            m = self.are_loaded
            if m.sum() > N:
                for i in where(m)[0]:
                    self.datasets[i][1].clean()

    def advect(
        self,
        x,
        y,
        t_init,
        mask_particule=None,
        nb_step=10,
        time_step=600,
        rk4=True,
        reset_grid=None,
        **kw,
    ):
        """
        At each call it will update position in place with u & v field

        :param array x: Longitude of obs to move
        :param array y: Latitude of obs to move
        :param float t_init: time to start advection
        :param array,None mask_particule: advect only i mask is True
        :param int nb_step: Number of iteration before to release data
        :param int time_step: Number of second for each advection
        :param bool rk4: Use rk4 algorithm instead of finite difference
        :param int reset_grid: Delete all loaded data in cube if there are more than N grid loaded

        :return: t,x,y position

        .. minigallery:: py_eddy_tracker.GridCollection.advect
        """
        self.reset_grids(reset_grid)
        backward = kw.get("backward", False)
        if backward:
            generator = self.get_previous_time_step(t_init)
            dt = -nb_step * time_step
            t_step = -time_step
        else:
            generator = self.get_next_time_step(t_init)
            dt = nb_step * time_step
            t_step = time_step
        t0, d0 = generator.__next__()
        u0, v0, m0 = d0.uv_for_advection(time_step=time_step, **kw)
        t1, d1 = generator.__next__()
        u1, v1, m1 = d1.uv_for_advection(time_step=time_step, **kw)
        t0 = t0 * 86400
        t1 = t1 * 86400
        t = t_init * 86400
        advect_ = advect_t_rk4 if rk4 else advect_t
        if mask_particule is None:
            mask_particule = isnan(x) + isnan(y)
        else:
            mask_particule += isnan(x) + isnan(y)
        while True:
            logger.debug(f"advect : t={t/86400}")
            if (backward and t <= t1) or (not backward and t >= t1):
                t0, u0, v0, m0 = t1, u1, v1, m1
                t1, d1 = generator.__next__()
                t1 = t1 * 86400
                u1, v1, m1 = d1.uv_for_advection(time_step=time_step, **kw)
            w = 1 - (arange(t, t + dt, t_step) - t0) / (t1 - t0)
            half_w = t_step / 2.0 / (t1 - t0)
            advect_(
                d0.x_c,
                d0.y_c,
                u0,
                v0,
                m0,
                u1,
                v1,
                m1,
                x,
                y,
                mask_particule,
                w,
                half_w=half_w,
            )
            t += dt
            yield t, x, y

    def get_next_time_step(self, t_init):
        for i, (t, dataset) in enumerate(self.datasets):
            if t < t_init:
                continue
            dataset.populate()
            logger.debug(f"i={i}, t={t}, dataset={dataset}")
            yield t, dataset

    def get_previous_time_step(self, t_init):
        i = len(self.datasets)
        for t, dataset in reversed(self.datasets):
            i -= 1
            if t > t_init:
                continue
            dataset.populate()
            logger.debug(f"i={i}, t={t}, dataset={dataset}")
            yield t, dataset

    def path(self, x0, y0, *args, nb_time=2, **kwargs):
        """
        At each call it will update position in place with u & v field

        :param array x0: Longitude of obs to move
        :param array y0: Latitude of obs to move
        :param int nb_time: Number of iteration for particle
        :param dict kwargs: look at :py:meth:`GridCollection.advect`

        :return: t,x,y

        .. minigallery:: py_eddy_tracker.GridCollection.path
        """
        particles = self.advect(x0.copy(), y0.copy(), *args, **kwargs)
        t = empty(nb_time + 1, dtype="f8")
        x = empty((nb_time + 1, x0.size), dtype=x0.dtype)
        y = empty(x.shape, dtype=y0.dtype)
        t[0], x[0], y[0] = kwargs.get("t_init"), x0, y0
        for i in range(nb_time):
            t[i + 1], x[i + 1], y[i + 1] = particles.__next__()
        return t, x, y


@njit(cache=True)
def advect_t(x_g, y_g, u_g0, v_g0, m_g0, u_g1, v_g1, m_g1, x, y, m, weigths, half_w=0):
    # Grid coordinates
    x_ref, y_ref = x_g[0], y_g[0]
    x_step, y_step = x_g[1] - x_ref, y_g[1] - y_ref
    is_circular = abs(x_g[-1] % 360 - (x_g[0] - x_step) % 360) < 1e-5
    nb_x_ = x_g.size
    nb_x = nb_x_ if is_circular else 0
    # Indexes that should never exist
    i0_old, j0_old = -100000, -100000
    m0, m1 = False, False
    u000, u001, u010, u011 = 0.0, 0.0, 0.0, 0.0
    v000, v001, v010, v011 = 0.0, 0.0, 0.0, 0.0
    u100, u101, u110, u111 = 0.0, 0.0, 0.0, 0.0
    v100, v101, v110, v111 = 0.0, 0.0, 0.0, 0.0
    # On each particle
    for i in prange(x.size):
        # If particle is not valid => continue
        if m[i]:
            continue
        # Iterate on whole steps
        for w in weigths:
            i0, j0, xd, yd = get_grid_indices(
                x_ref, y_ref, x_step, y_step, x[i], y[i], nb_x
            )
            if i0 != i0_old or j0 != j0_old:
                # Need to be stored only on change
                i0_old, j0_old = i0, j0
                if not is_circular and (i0 < 0 or i0 > nb_x_):
                    m0, m1 = True, True
                else:
                    (m0, u000, u001, u010, u011, v000, v001, v010, v011) = get_uv_quad(
                        i0, j0, u_g0, v_g0, m_g0, nb_x
                    )
                    (m1, u100, u101, u110, u111, v100, v101, v110, v111) = get_uv_quad(
                        i0, j0, u_g1, v_g1, m_g1, nb_x
                    )
            if m0 or m1:
                x[i], y[i] = nan, nan
                m[i] = True
                break
            # Compute distance
            xd_i, yd_i = 1 - xd, 1 - yd
            # Compute new x,y
            dx0 = (u000 * xd_i + u010 * xd) * yd_i + (u001 * xd_i + u011 * xd) * yd
            dx1 = (u100 * xd_i + u110 * xd) * yd_i + (u101 * xd_i + u111 * xd) * yd
            dy0 = (v000 * xd_i + v010 * xd) * yd_i + (v001 * xd_i + v011 * xd) * yd
            dy1 = (v100 * xd_i + v110 * xd) * yd_i + (v101 * xd_i + v111 * xd) * yd
            x[i] += dx0 * w + dx1 * (1 - w)
            y[i] += dy0 * w + dy1 * (1 - w)


@njit(cache=True, fastmath=True)
def get_uv_quad(i0, j0, u, v, m, nb_x=0):
    """
    Return u/v for (i0, j0), (i1, j0), (i0, j1), (i1, j1)

    :param int i0: indexes of longitude
    :param int j0: indexes of latitude
    :param array[float] u: current along x axis
    :param array[float] v: current along y axis
    :param array[bool] m: flag to know if position is valid
    :param int nb_x: If different of 0 we check if wrapping is needed

    :return: if cell is valid 4 u, 4 v
    :rtype: bool,float,float,float,float,float,float,float,float
    """
    i1, j1 = i0 + 1, j0 + 1
    if nb_x != 0:
        i1 %= nb_x
    i_max, j_max = m.shape

    if i1 >= i_max or j1 >= j_max:
        return True, nan, nan, nan, nan, nan, nan, nan, nan

    if m[i0, j0] or m[i0, j1] or m[i1, j0] or m[i1, j1]:
        return True, nan, nan, nan, nan, nan, nan, nan, nan
    # Extract value for u and v
    u00, u01, u10, u11 = u[i0, j0], u[i0, j1], u[i1, j0], u[i1, j1]
    v00, v01, v10, v11 = v[i0, j0], v[i0, j1], v[i1, j0], v[i1, j1]
    return False, u00, u01, u10, u11, v00, v01, v10, v11


@njit(cache=True, fastmath=True)
def get_grid_indices(x0, y0, x_step, y_step, x, y, nb_x=0):
    """
    Return grid indexes and weight

    :param float x0: first longitude
    :param float y0: first latitude
    :param float x_step: longitude grid step
    :param float y_step: latitude grid step
    :param float x: longitude to interpolate
    :param float y: latitude to interpolate
    :param int nb_x: If different of 0 we check if wrapping is needed

    :return: indexes and weight
    :rtype: int,int,float,float
    """
    i, j = (x - x0) / x_step, (y - y0) / y_step
    i0, j0 = int(floor(i)), int(floor(j))
    xd, yd = i - i0, j - j0
    if nb_x != 0:
        i0 %= nb_x
    return i0, j0, xd, yd


@njit(cache=True, fastmath=True)
def interp_uv(xd, yd, u00, u01, u10, u11, v00, v01, v10, v11):
    """
    Return u/v interpolated in cell

    :param float xd: x weight
    :param float yd: y weight
    :param float u00: u lower left
    :param float u01: u upper left
    :param float u10: u lower right
    :param float u11: u upper right
    :param float v00: v lower left
    :param float v01: v upper left
    :param float v10: v lower right
    :param float v11: v upper right
    """
    xd_i, yd_i = 1 - xd, 1 - yd
    u = (u00 * xd_i + u10 * xd) * yd_i + (u01 * xd_i + u11 * xd) * yd
    v = (v00 * xd_i + v10 * xd) * yd_i + (v01 * xd_i + v11 * xd) * yd
    return u, v


@njit(cache=True, fastmath=True)
def advect_t_rk4(
    x_g, y_g, u_g0, v_g0, m_g0, u_g1, v_g1, m_g1, x, y, m, weigths, half_w
):
    # Grid coordinates
    x_ref, y_ref = x_g[0], y_g[0]
    x_step, y_step = x_g[1] - x_ref, y_g[1] - y_ref
    is_circular = abs(x_g[-1] % 360 - (x_g[0] - x_step) % 360) < 1e-5
    nb_x_ = x_g.size
    nb_x = nb_x_ if is_circular else 0
    # cache
    i_cache, j_cache = -1000000, -1000000
    m0, m1 = False, False
    u000, u001, u010, u011 = 0.0, 0.0, 0.0, 0.0
    v000, v001, v010, v011 = 0.0, 0.0, 0.0, 0.0
    u100, u101, u110, u111 = 0.0, 0.0, 0.0, 0.0
    v100, v101, v110, v111 = 0.0, 0.0, 0.0, 0.0
    # On each particle
    for i in prange(x.size):
        # If particle is not valid => continue
        if m[i]:
            continue
        x_, y_ = x[i], y[i]
        # Iterate on whole steps
        for w in weigths:
            # k1, slope at origin
            ii_, jj_, xd, yd = get_grid_indices(
                x_ref, y_ref, x_step, y_step, x_, y_, nb_x
            )
            if ii_ != i_cache or jj_ != j_cache:
                i_cache, j_cache = ii_, jj_
                if not is_circular and (ii_ < 0 or ii_ > nb_x_):
                    m0, m1 = True, True
                else:
                    (m0, u000, u001, u010, u011, v000, v001, v010, v011) = get_uv_quad(
                        ii_, jj_, u_g0, v_g0, m_g0, nb_x
                    )
                    (m1, u100, u101, u110, u111, v100, v101, v110, v111) = get_uv_quad(
                        ii_, jj_, u_g1, v_g1, m_g1, nb_x
                    )
            # The 3 following could be in cache operation but this one must be tested in any case
            if m0 or m1:
                x_, y_ = nan, nan
                m[i] = True
                break
            u0_, v0_ = interp_uv(xd, yd, u000, u001, u010, u011, v000, v001, v010, v011)
            u1_, v1_ = interp_uv(xd, yd, u100, u101, u110, u111, v100, v101, v110, v111)
            u1, v1 = u0_ * w + u1_ * (1 - w), v0_ * w + v1_ * (1 - w)
            # k2, slope at middle with first guess position
            x1, y1 = x_ + u1 * 0.5, y_ + v1 * 0.5
            ii_, jj_, xd, yd = get_grid_indices(
                x_ref, y_ref, x_step, y_step, x1, y1, nb_x
            )
            if ii_ != i_cache or jj_ != j_cache:
                i_cache, j_cache = ii_, jj_
                if not is_circular and (ii_ < 0 or ii_ > nb_x_):
                    m0, m1 = True, True
                else:
                    (m0, u000, u001, u010, u011, v000, v001, v010, v011) = get_uv_quad(
                        ii_, jj_, u_g0, v_g0, m_g0, nb_x
                    )
                    (m1, u100, u101, u110, u111, v100, v101, v110, v111) = get_uv_quad(
                        ii_, jj_, u_g1, v_g1, m_g1, nb_x
                    )
                if m0 or m1:
                    x_, y_ = nan, nan
                    m[i] = True
                    break
            u0_, v0_ = interp_uv(xd, yd, u000, u001, u010, u011, v000, v001, v010, v011)
            u1_, v1_ = interp_uv(xd, yd, u100, u101, u110, u111, v100, v101, v110, v111)
            w_ = w - half_w
            u2, v2 = u0_ * w_ + u1_ * (1 - w_), v0_ * w_ + v1_ * (1 - w_)
            # k3, slope at middle with updated guess position
            x2, y2 = x_ + u2 * 0.5, y_ + v2 * 0.5
            ii_, jj_, xd, yd = get_grid_indices(
                x_ref, y_ref, x_step, y_step, x2, y2, nb_x
            )
            if ii_ != i_cache or jj_ != j_cache:
                i_cache, j_cache = ii_, jj_
                if not is_circular and (ii_ < 0 or ii_ > nb_x_):
                    m0, m1 = True, True
                else:
                    (m0, u000, u001, u010, u011, v000, v001, v010, v011) = get_uv_quad(
                        ii_, jj_, u_g0, v_g0, m_g0, nb_x
                    )
                    (m1, u100, u101, u110, u111, v100, v101, v110, v111) = get_uv_quad(
                        ii_, jj_, u_g1, v_g1, m_g1, nb_x
                    )
                if m0 or m1:
                    x_, y_ = nan, nan
                    m[i] = True
                    break
            u0_, v0_ = interp_uv(xd, yd, u000, u001, u010, u011, v000, v001, v010, v011)
            u1_, v1_ = interp_uv(xd, yd, u100, u101, u110, u111, v100, v101, v110, v111)
            u3, v3 = u0_ * w_ + u1_ * (1 - w_), v0_ * w_ + v1_ * (1 - w_)
            # k4, slope at end with updated guess position
            x3, y3 = x_ + u3, y_ + v3
            ii_, jj_, xd, yd = get_grid_indices(
                x_ref, y_ref, x_step, y_step, x3, y3, nb_x
            )
            if ii_ != i_cache or jj_ != j_cache:
                i_cache, j_cache = ii_, jj_
                if not is_circular and (ii_ < 0 or ii_ > nb_x_):
                    m0, m1 = True, True
                else:
                    (m0, u000, u001, u010, u011, v000, v001, v010, v011) = get_uv_quad(
                        ii_, jj_, u_g0, v_g0, m_g0, nb_x
                    )
                    (m1, u100, u101, u110, u111, v100, v101, v110, v111) = get_uv_quad(
                        ii_, jj_, u_g1, v_g1, m_g1, nb_x
                    )
                if m0 or m1:
                    x_, y_ = nan, nan
                    m[i] = True
                    break
            u0_, v0_ = interp_uv(xd, yd, u000, u001, u010, u011, v000, v001, v010, v011)
            u1_, v1_ = interp_uv(xd, yd, u100, u101, u110, u111, v100, v101, v110, v111)
            w_ -= half_w
            u4, v4 = u0_ * w_ + u1_ * (1 - w_), v0_ * w_ + v1_ * (1 - w_)
            # RK4 compute
            dx = (u1 + 2 * u2 + 2 * u3 + u4) / 6
            dy = (v1 + 2 * v2 + 2 * v3 + v4) / 6
            x_ += dx
            y_ += dy
        x[i], y[i] = x_, y_


@njit(
    [
        "Tuple((f8[:,:],b1[:,:]))(f8[:],f8[:],f8[:,:],b1[:,:],f8,b1,i1)",
        "Tuple((f4[:,:],b1[:,:]))(f8[:],f8[:],f4[:,:],b1[:,:],f8,b1,i1)",
    ],
    cache=True,
    fastmath=True,
)
def compute_stencil(x, y, h, m, earth_radius, vertical=False, stencil_halfwidth=4):
    """
    Compute stencil on RegularGrid

    :param array x: longitude coordinates
    :param array y: latitude coordinates
    :param array h: 2D array to derivate
    :param array m: mask associated to h to know where are invalid data
    :param float earth_radius: Earth radius in m
    :param bool vertical: if True stencil will be vertical (along y)
    :param int stencil_halfwidth: from 1 to 4 to specify maximal kernel usable


    stencil_halfwidth:

        - (1) :

            - (-1, 1, 0)
            - (0, -1, 1)
            - (-1, 0, 1) / 2

        - (2) : (1, -8, 0, 8, 1) / 12
        - (3) : (-1, 9, -45, 0, 45, -9, 1) / 60
        - (4) : (3, -32, 168, -672, 0, 672, -168, 32, 3) / 840
    """
    if vertical:
        # If vertical we transpose matrix and inverse coordinates
        h = h.T
        m = m.T
        x, y = y, x
    shape = h.shape
    nb_x, nb_y = shape
    # Out array
    m_out = empty(shape, dtype=numba_types.bool_)
    grad = empty(shape, dtype=h.dtype)
    # Distance step in degrees
    d_step = x[1] - x[0]
    if vertical:
        is_circular = False
    else:
        # Test if matrix is circular
        is_circular = abs(x[-1] % 360 - (x[0] - d_step) % 360) < 1e-5

    # Compute caracteristic distance, constant when vertical
    d_ = 360 / (d_step * pi * 2 * earth_radius)
    for j in range(nb_y):
        # Buffer of maximal size of stencil (9)
        if is_circular:
            h_3, h_2, h_1, h0 = h[-4, j], h[-3, j], h[-2, j], h[-1, j]
            m_3, m_2, m_1, m0 = m[-4, j], m[-3, j], m[-2, j], m[-1, j]
        else:
            m_3, m_2, m_1, m0 = False, False, False, False
        h1, h2, h3, h4 = h[0, j], h[1, j], h[2, j], h[3, j]
        m1, m2, m3, m4 = m[0, j], m[1, j], m[2, j], m[3, j]
        for i in range(nb_x):
            # Roll value and only last
            h_4, h_3, h_2, h_1, h0, h1, h2, h3 = h_3, h_2, h_1, h0, h1, h2, h3, h4
            m_4, m_3, m_2, m_1, m0, m1, m2, m3 = m_3, m_2, m_1, m0, m1, m2, m3, m4
            i_ = i + 4
            if i_ >= nb_x:
                if is_circular:
                    i_ = i_ % nb_x
                    m4 = m[i_, j]
                    h4 = h[i_, j]
                else:
                    # When we are out
                    m4 = False
            else:
                m4 = m[i_, j]
                h4 = h[i_, j]

            # Current value not defined
            if m0:
                m_out[i, j] = True
                continue
            if not vertical:
                # For each row we compute distance
                d_ = 360 / (d_step * cos(deg2rad(y[j])) * pi * 2 * earth_radius)
            if m1 ^ m_1:
                # unbalanced kernel
                if m_1:
                    grad[i, j] = (h1 - h0) * d_
                    m_out[i, j] = False
                    continue
                if m1:
                    grad[i, j] = (h0 - h_1) * d_
                    m_out[i, j] = False
                    continue
                continue
            if m2 or m_2 or stencil_halfwidth == 1:
                grad[i, j] = (h1 - h_1) / 2 * d_
                m_out[i, j] = False
                continue
            if m3 or m_3 or stencil_halfwidth == 2:
                grad[i, j] = (h_2 - h2 + 8 * (h1 - h_1)) / 12 * d_
                m_out[i, j] = False
                continue
            if m4 or m_4 or stencil_halfwidth == 3:
                grad[i, j] = (h3 - h_3 + 9 * (h_2 - h2) + 45 * (h1 - h_1)) / 60 * d_
                m_out[i, j] = False
                continue
            # If all values of buffer are available
            grad[i, j] = (
                (3 * (h_4 - h4) + 32 * (h3 - h_3) + 168 * (h_2 - h2) + 672 * (h1 - h_1))
                / 840
                * d_
            )
            m_out[i, j] = False
    if vertical:
        return grad.T, m_out.T
    else:
        return grad, m_out
