# -*- coding: utf-8 -*-
"""
"""
import logging
from scipy.interpolate import interp1d
from numpy import concatenate, int32, empty, maximum, where, array, \
    sin, deg2rad, pi, ones, cos, ma, int8, histogram2d, arange, float_, \
    linspace, meshgrid, sinc, errstate, round_, int_, column_stack, interp
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter, convolve
from scipy.ndimage.filters import convolve1d
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree
from matplotlib.figure import Figure
from matplotlib.path import Path as BasePath
from pyproj import Proj
from ..tools import fit_circle_c, distance_vector
from ..observations import EddiesObservations
from ..eddy_feature import Amplitude


@property
def isvalid(self):
    return False not in (self.vertices[0] == self.vertices[-1]
                         ) and len(self.vertices) > 2


BasePath.isvalid = isvalid


def mean_coordinates(self):
    return self.vertices.mean(axis=0)


BasePath.mean_coordinates = mean_coordinates


@property
def lon(self):
    return self.vertices[:, 0]


BasePath.lon = lon


@property
def lat(self):
    return self.vertices[:, 1]


BasePath.lat = lat


# def nearest_grd_indice(self, lon_value, lat_value, grid):
# if not hasattr(self, '_grid_indices'):
# self._grid_indices = nearest(lon_value, lat_value,
# grid.lon[0], grid.lat[:, 0])
# return self._grid_indices
# BasePath.nearest_grd_indice = nearest_grd_indice


def fit_circle_path(self):
    if not hasattr(self, '_circle_params'):
        self._fit_circle_path()
    return self._circle_params


def _fit_circle_path(self):
    lon_mean, lat_mean = self.mean_coordinates()
    # Prepare for shape test and get eddy_radius_e
    # http://www.geo.hunter.cuny.edu/~jochen/gtech201/lectures/
    # lec6concepts/map%20coordinate%20systems/
    # how%20to%20choose%20a%20projection.htm
    proj = Proj('+proj=aeqd +ellps=WGS84 +lat_0=%s +lon_0=%s'
                % (lat_mean, lon_mean))

    c_x, c_y = proj(self.lon, self.lat)
    try:
        centlon_e, centlat_e, eddy_radius_e, aerr = fit_circle_c(c_x, c_y)
        centlon_e, centlat_e = proj(centlon_e, centlat_e, inverse=True)
        centlon_e = (centlon_e - lon_mean + 180) % 360 + lon_mean - 180
        self._circle_params = centlon_e, centlat_e, eddy_radius_e, aerr
    except ZeroDivisionError:
        # Some time, edge is only a dot of few coordinates
        if len(unique(self.lon)) == 1 and len(unique(self.lat)) == 1:
            logging.warning('An edge is only define in one position')
            logging.debug('%d coordinates %s,%s', len(self.lon), self.lon,
                          self.lat)
            self._circle_params = 0, -90, nan, nan


BasePath.fit_circle = fit_circle_path
BasePath._fit_circle_path = _fit_circle_path


def uniform_resample(x_val, y_val, num_fac=2, fixed_size=None):
    """
    Resample contours to have (nearly) equal spacing
       x_val, y_val    : input contour coordinates
       num_fac : factor to increase lengths of output coordinates
    """
    # Get distances
    dist = empty(x_val.shape)
    dist[0] = 0
    distance_vector(
        x_val[:-1], y_val[:-1], x_val[1:], y_val[1:], dist[1:])
    dist.cumsum(out=dist)
    # Get uniform distances
    if fixed_size is None:
        fixed_size = dist.size * num_fac
    d_uniform = linspace(0, dist[-1], num=fixed_size)
    x_new = interp(d_uniform, dist, x_val)
    y_new = interp(d_uniform, dist, y_val)
    return x_new, y_new


class GridDataset(object):
    """
    Class to have basic tool on NetCDF Grid
    """

    __slots__ = (
        '_x_var',
        '_y_var',
        'x_c',
        'y_c',
        'x_bounds',
        'y_bounds',
        'centered',
        'xinterp',
        'yinterp',
        'x_dim',
        'y_dim',
        'coordinates',
        'filename',
        'dimensions',
        'variables_description',
        'global_attrs',
        'vars',
        'interpolators',
    )

    GRAVITY = 9.807
    EARTH_RADIUS = 6378136.3
    N = 1

    def __init__(self, filename, x_name, y_name, centered=None):
        self.dimensions = None
        self.variables_description = None
        self.global_attrs = None
        self.x_c = None
        self.y_c = None
        self.x_bounds = None
        self.y_bounds = None
        self.x_dim = None
        self.y_dim = None
        self.centered = centered
        self.xinterp = None
        self.yinterp = None
        self.filename = filename
        self.coordinates = x_name, y_name
        self.vars = dict()
        self.interpolators = dict()
        logging.warning('We assume the position of grid is the center'
                        ' corner for %s', filename)
        self.load_general_features()
        self.load()

    @property
    def is_centered(self):
        """Give information if pixel is describe with center position or
        a corner
        """
        if self.centered is None:
            return True
        else:
            return self.centered

    def load_general_features(self):
        """Load attrs
        """
        with Dataset(self.filename) as h:
            # Load generals
            self.dimensions = {i: len(v) for i, v in h.dimensions.items()}
            self.variables_description = dict()
            for i, v in h.variables.items():
                args = (i, v.datatype)
                kwargs = dict(
                    dimensions=v.dimensions,
                    zlib=True,
                )
                if hasattr(v, '_FillValue'):
                    kwargs['fill_value'] = v._FillValue,
                attrs = dict()
                for attr in v.ncattrs():
                    if attr in kwargs.keys():
                        continue
                    if attr == '_FillValue':
                        continue
                    attrs[attr] = getattr(v, attr)
                self.variables_description[i] = dict(
                    args=args,
                    kwargs=kwargs,
                    attrs=attrs,
                    infos=dict())
            self.global_attrs = {attr: getattr(h, attr) for attr in h.ncattrs()}

    def write(self, filename):
        """Write dataset output with same format like input
        """
        with Dataset(filename, 'w') as h_out:
            for dimension, size in self.dimensions.items():
                test = False
                for varname, variable in self.variables_description.items():
                    if varname not in self.coordinates and varname not in self.vars.keys():
                        continue
                    if dimension in variable['kwargs']['dimensions']:
                        test = True
                        break
                if test:
                    h_out.createDimension(dimension, size)

            for varname, variable in self.variables_description.items():
                if varname not in self.coordinates and varname not in self.vars.keys():
                    continue
                var = h_out.createVariable(*variable['args'], **variable['kwargs'])
                for key, value in variable['attrs'].items():
                    setattr(var, key, value)

                infos = self.variables_description[varname]['infos']
                if infos.get('transpose', False):
                    var[:] = self.vars[varname].T
                else:
                    var[:] = self.vars[varname]

            for attr, value in self.global_attrs.items():
                setattr(h_out, attr, value)

    def load(self):
        """Load variable (data)
        """
        x_name, y_name = self.coordinates
        with Dataset(self.filename) as h:
            self.x_dim = h.variables[x_name].dimensions
            self.y_dim = h.variables[y_name].dimensions

            self.vars[x_name] = h.variables[x_name][:]
            self.vars[y_name] = h.variables[y_name][:]

            if self.is_centered:
                self.x_c = self.vars[x_name]
                self.y_c = self.vars[y_name]

                self.x_bounds = concatenate((
                    self.x_c, (2 * self.x_c[-1] - self.x_c[-2],)))
                self.y_bounds = concatenate((
                    self.y_c, (2 * self.y_c[-1] - self.y_c[-2],)))
                d_x = self.x_bounds[1:] - self.x_bounds[:-1]
                d_y = self.y_bounds[1:] - self.y_bounds[:-1]
                self.x_bounds[:-1] -= d_x / 2
                self.x_bounds[-1] -= d_x[-1] / 2
                self.y_bounds[:-1] -= d_y / 2
                self.y_bounds[-1] -= d_y[-1] / 2

            else:
                self.x_bounds = self.vars[x_name]
                self.y_bounds = self.vars[y_name]

                if len(self.x_dim) == 1:
                    raise Exception('not test')
                    self.x_c = (self.x_bounds[1:] + self.x_bounds[:-1]) / 2
                    self.y_c = (self.y_bounds[1:] + self.y_bounds[:-1]) / 2
                else:
                    raise Exception('not write')

        self.init_pos_interpolator()

    def is_circular(self):
        """Check grid circularity
        """
        return False

    def grid(self, varname):
        """give grid required
        """
        if varname not in self.vars:
            coordinates_dims = list(self.x_dim)
            coordinates_dims.extend(list(self.y_dim))
            with Dataset(self.filename) as h:
                dims = h.variables[varname].dimensions
                sl = [slice(None) if dim in coordinates_dims else 0 for dim in dims]
                self.vars[varname] = h.variables[varname][sl]
                if len(self.x_dim) == 1:
                    i_x = where(array(dims) == self.x_dim)[0][0]
                    i_y = where(array(dims) == self.y_dim)[0][0]
                    if i_x > i_y:
                        self.variables_description[varname]['infos']['transpose'] = True
                        self.vars[varname] = self.vars[varname].T
        return self.vars[varname]

    def high_filter(self, grid_name, x_cut, y_cut):
        """create a high filter with a low one
        """
        result = self._low_filter(grid_name, x_cut, y_cut)
        self.vars[grid_name] -= result

    def low_filter(self, grid_name, x_cut, y_cut):
        """low filtering
        """
        result = self._low_filter(grid_name, x_cut, y_cut)
        self.vars[grid_name] -= self.vars[grid_name] - result

    @property
    def bounds(self):
        """Give bound
        """
        return self.x_bounds.min(), self.x_bounds.max(), self.y_bounds.min(), self.y_bounds.max()

    def eddy_identification(self, grid_name, step=0.005, shape_error=55, extra_variables=None,
                            extra_array_variables=None, array_sampling=50, pixel_limit=None):
        if extra_variables is None:
            extra_variables = list()
        if extra_array_variables is None:
            extra_array_variables = list()
        if pixel_limit is None:
            pixel_limit = (8, 1000)
        fig = Figure()
        ax = fig.add_subplot(111)
        data = self.grid(grid_name)
        z_min, z_max = data.min(), data.max()

        levels = arange(z_min - z_min % step, z_max - z_max % step + 2 * step, step)

        x, y = self.vars[self.coordinates[0]], self.vars[self.coordinates[1]]
        ## To re write
        if len(x.shape) == 1:
            data = data.T
        ## 
        contours = ax.contour(x, y, data, levels)

        anticyclonic_search = True
        iterator = 1 if anticyclonic_search else -1

        # Loop over each collection
        for coll_ind, coll in enumerate(contours.collections[::iterator]):

            corrected_coll_index = coll_ind
            if iterator == -1:
                corrected_coll_index = - coll_ind - 1

            contour_paths = coll.get_paths()
            nb_paths = len(contour_paths)
            if nb_paths == 0:
                continue
            cvalues = contours.cvalues[corrected_coll_index]
            logging.debug('doing collection %s, contour value %s, %d paths',
                          corrected_coll_index, cvalues, nb_paths)

            # Loop over individual c_s contours (i.e., every eddy in field)
            for cont in contour_paths:
                # Filter for closed contours
                if not cont.isvalid:
                    continue
                centlon_e, centlat_e, eddy_radius_e, aerr = cont.fit_circle()
                # Filter for shape
                if aerr < 0 or aerr > shape_error:
                    continue
                # Get indices of centroid
                # Give only 1D array of lon and lat not 2D data
                i_x, i_y = self.nearest_grd_indice(centlon_e, centlat_e)

                # Check if centroid is on define value
                if hasattr(data, 'mask') and data.mask[i_x, i_y]:
                    continue
                # Test to know cyclone or anticyclone
                acyc_not_cyc = data[i_x, i_y] >= cvalues
                if anticyclonic_search != acyc_not_cyc:
                    continue

                i_x_in, i_y_in = self.get_pixels_in(cont)
                nb_pixel = i_x_in.shape[0]

                # Maybe limit max must be replace with a maximum of surface
                if nb_pixel < pixel_limit[0] or nb_pixel > pixel_limit[1]:
                    continue

                # Resample the contour points for a more even
                # circumferential distribution
                contlon_e, contlat_e = uniform_resample(cont.lon, cont.lat)

                amp = self.get_amplitude(contlon_e, contlat_e, i_x_in, i_y_in, cvalues, data,
                                         anticyclonic_search=anticyclonic_search)

                print(amp.within_amplitude_limits(), amp.amplitude)
                if (not amp.within_amplitude_limits()) or (amp.amplitude == 0):
                    continue

                eddy.reshape_mask_eff(grd)
                # Instantiate Amplitude object
                amp = Amplitude(contlon_e, contlat_e, eddy, grd)

                if anticyclonic_search:
                    reset_centroid = amp.all_pixels_above_h0(
                        contours.levels[corrected_coll_index])

                else:
                    reset_centroid = amp.all_pixels_below_h0(
                        contours.levels[corrected_coll_index])

                if not amp.within_amplitude_limits() or amp.amplitude:
                    continue

                if reset_centroid:
                    centi = reset_centroid[0]
                    centj = reset_centroid[1]
                    centlon_e = grd.lon[centj, centi]
                    centlat_e = grd.lat[centj, centi]

                # Get sum of eke within Ceff
                teke = grd.eke[eddy.slice_j, eddy.slice_i][eddy.mask_eff].sum()

                (uavg, contlon_s, contlat_s, inner_contlon, inner_contlat,
                 any_inner_contours
                 ) = get_uavg(eddy, contours, centlon_e, centlat_e, cont, grd,
                              anticyclonic_search)

                # Use azimuth equal projection for radius
                proj = Proj('+proj=aeqd +ellps=WGS84 +lat_0=%s +lon_0=%s'
                            % (inner_contlat.mean(),
                               inner_contlon.mean()))

                # First, get position based on innermost
                # contour
                c_x, c_y = proj(inner_contlon, inner_contlat)
                centx_s, centy_s, _, _ = fit_circle_c(c_x, c_y)
                centlon_s, centlat_s = proj(centx_s, centy_s,
                                            inverse=True)
                # Second, get speed-based radius based on
                # contour of max uavg
                # (perhaps we should make a new proj here
                # based on contlon_s, contlat_s but I'm not
                # sure it's that important ... Antoine?)
                # A. : I dont think, the difference is tiny
                c_x, c_y = proj(contlon_s, contlat_s)
                _, _, eddy_radius_s, aerr_s = fit_circle_c(c_x, c_y)

                # Instantiate new EddyObservation object
                properties = EddiesObservations(
                    size=1,
                    track_extra_variables=extra_variables,
                    track_array_variables=array_sampling,
                    array_variables=extra_array_variables
                )
                properties.obs['amplitude'] = amp.amplitude
                properties.obs['radius_s'] = eddy_radius_s / 1000
                properties.obs['speed_radius'] = uavg
                properties.obs['radius_e'] = eddy_radius_e / 1000
                properties.obs['eke'] = teke
                if 'shape_error_e' in eddy.track_extra_variables:
                    properties.obs['shape_error_e'] = aerr
                if 'shape_error_s' in eddy.track_extra_variables:
                    properties.obs['shape_error_s'] = aerr_s

                if aerr > 99.9 or aerr_s > 99.9:
                    logging.warning(
                        'Strange shape at this step! shape_error : %f, %f',
                        aerr,
                        aerr_s)
                    continue

                # Update SLA eddy properties

                # See CSS11 section B4
                properties.obs['lon'] = centlon_s
                properties.obs['lat'] = centlat_s
                if 'contour_lon_e' in extra_array_variables:
                    (properties.obs['contour_lon_e'],
                     properties.obs['contour_lat_e']) = uniform_resample(
                        cont.lon, cont.lat,
                        fixed_size=array_sampling)
                if 'contour_lon_s' in extra_array_variables:
                    (properties.obs['contour_lon_s'],
                     properties.obs['contour_lat_s']) = uniform_resample(
                        contlon_s, contlat_s,
                        fixed_size=array_sampling)

                # for AVISO
                eddy.update_eddy_properties(properties)

                # Mask out already found eddies
                eddy.sla[eddy.slice_j, eddy.slice_i][
                    eddy.mask_eff] = eddy.fillval

    @staticmethod
    def _gaussian_filter(data, sigma, mode='reflect'):
        """Standard gaussian filter
        """
        local_data = data.copy()
        local_data[data.mask] = 0

        v = gaussian_filter(local_data, sigma=sigma, mode=mode)
        w = gaussian_filter(float_(-data.mask), sigma=sigma, mode=mode)

        with errstate(invalid='ignore'):
            return ma.array(v / w, mask=w == 0)

    @staticmethod
    def get_amplitude(cont_x, cont_y, i_x_in, i_y_in, contour_height, data, anticyclonic_search=True):
        # Instantiate Amplitude object
        amp = Amplitude(
            contour_x=cont_x,
            contour_y=cont_y,
            i_contour_x=i_x_in,
            i_contour_y=i_y_in,
            contour_height=contour_height,
            data=data)

        if anticyclonic_search:
            reset_centroid = amp.all_pixels_above_h0()

        else:
            reset_centroid = amp.all_pixels_below_h0(
                contours.levels[corrected_coll_index])

        return amp
        # if not amp.within_amplitude_limits() or amp.amplitude:
        # continue


class UnRegularGridDataset(GridDataset):
    """Class which manage unregular grid
    """

    __slots__ = 'index_interp'

    def bbox_indice(self, vertices):
        dist, idx = self.index_interp.query(vertices, k=1)
        i_y = idx % self.x_c.shape[1]
        i_x = int_((idx - i_y) / self.x_c.shape[1])
        return slice(i_x.min() - self.N, i_x.max() + self.N + 1), slice(i_y.min() - self.N, i_y.max() + self.N + 1)

    def get_pixels_in(self, contour):
        slice_x, slice_y = self.bbox_indice(contour.vertices)
        pts = array((self.x_c[slice_x, slice_y].reshape(-1),
                     self.y_c[slice_x, slice_y].reshape(-1))).T
        mask = contour.contains_points(pts).reshape((slice_x.stop - slice_x.start, -1))
        i_x, i_y = where(mask)
        i_x += slice_x.start
        i_y += slice_y.start

        # import matplotlib
        # matplotlib.use('AGG')
        # import matplotlib.pyplot as plt
        # plt.figure()    
        # plt.plot(self.x_c[i_x, i_y], self.y_c[i_x, i_y], 'g.', zorder=-10, markersize=20)
        # plt.plot(self.x_c[slice_x, slice_y], self.y_c[slice_x, slice_y], 'r.', zorder=+10)
        # plt.plot(contour.lon, contour.lat, 'b', zorder=20)
        # plt.savefig('/tmp/toto.png')
        return i_x, i_y

    def nearest_grd_indice(self, x, y):
        dist, idx = self.index_interp.query((x, y), k=1)
        i_y = idx % self.x_c.shape[1]
        i_x = int_((idx - i_y) / self.x_c.shape[1])
        return i_x, i_y

    def compute_pixel_path(self, x0, y0, x1, y1):
        pass

    def init_pos_interpolator(self):
        self.index_interp = cKDTree(
            column_stack((
                self.x_c.reshape(-1),
                self.y_c.reshape(-1)
            )))

    def _low_filter(self, grid_name, x_cut, y_cut, factor=40.):
        data = self.grid(grid_name)
        mean_data = data.mean()
        x = self.grid(self.coordinates[0])
        y = self.grid(self.coordinates[1])
        regrid_x_step = x_cut / factor
        regrid_y_step = y_cut / factor
        x_min, x_max, y_min, y_max = self.bounds
        x_array = arange(x_min, x_max + regrid_x_step, regrid_x_step)
        y_array = arange(y_min, y_max + regrid_y_step, regrid_y_step)
        bins = (x_array, y_array)

        x_flat, y_flat, z_flat = x.reshape((-1,)), y.reshape((-1,)), data.reshape((-1,))
        m = -z_flat.mask
        x_flat, y_flat, z_flat = x_flat[m], y_flat[m], z_flat[m]

        nb_value, bounds_x, bounds_y = histogram2d(
            x_flat, y_flat,
            bins=bins)

        sum_value, _, _ = histogram2d(
            x_flat, y_flat,
            bins=bins,
            weights=z_flat)

        with errstate(invalid='ignore'):
            z_grid = ma.array(sum_value / nb_value, mask=nb_value == 0)
        i_x, i_y = x_cut * 0.125 / regrid_x_step, y_cut * 0.125 / regrid_y_step
        m = nb_value == 0

        z_filtered = self._gaussian_filter(z_grid, (i_x, i_y))

        z_filtered[m] = 0
        x_center = (bounds_x[:-1] + bounds_x[1:]) / 2
        y_center = (bounds_y[:-1] + bounds_y[1:]) / 2
        opts_interpolation = dict(kx=1, ky=1, s=0)
        m_interp = RectBivariateSpline(x_center, y_center, m, **opts_interpolation)
        z_interp = RectBivariateSpline(x_center, y_center, z_filtered, **opts_interpolation).ev(x, y)
        return ma.array(z_interp, mask=m_interp.ev(x, y) > 0.00001)


class RegularGridDataset(GridDataset):
    """Class only for regular grid
    """

    __slots__ = ()

    def init_pos_interpolator(self):
        """Create function to have a quick index interpolator
        """
        self.xinterp = interp1d(self.x_bounds, range(self.x_bounds.shape[0]), assume_sorted=True)
        self.yinterp = interp1d(self.y_bounds, range(self.y_bounds.shape[0]), assume_sorted=True)

    def nearest_grd_indice(self, x, y):
        return int_(round_(self.xinterp(x))), int_(round_(self.yinterp(y)))

    @property
    def xstep(self):
        """Only for regular grid with no step variation
        """
        return self.x_c[1] - self.x_c[0]

    @property
    def ystep(self):
        """Only for regular grid with no step variation
        """
        return self.y_c[1] - self.y_c[0]

    def compute_pixel_path(self, x0, y0, x1, y1):
        """Give a series of index which describe the path between to position
        """
        # First x of grid
        x_ori = self.x_var[0]
        # Float index
        f_x0 = self.xinterp((x0 - x_ori) % 360 + x_ori)
        f_x1 = self.xinterp((x1 - x_ori) % 360 + x_ori)
        f_y0 = self.yinterp(y0)
        f_y1 = self.yinterp(y1)
        # Int index
        i_x0, i_x1 = int32(round(f_x0)), int32(round(f_x1))
        i_y0, i_y1 = int32(round(f_y0)), int32(round(f_y1))

        # Delta index of x
        d_x = i_x1 - i_x0
        nb_x = self.x_var.shape[0] - 1
        d_x = (d_x + nb_x / 2) % nb_x - nb_x / 2
        i_x1 = i_x0 + d_x

        # Delta index of y
        d_y = i_y1 - i_y0

        d_max = maximum(abs(d_x), abs(d_y))

        # Compute number of pixel which we go trought
        nb_value = (abs(d_max) + 1).sum()
        # Create an empty array to store value of pixel across the travel
        # Max Index ~65000
        i_g = empty(nb_value, dtype='u2')
        j_g = empty(nb_value, dtype='u2')

        # Index to determine the position in the global array
        ii = 0
        # Iteration on each travel
        for i, delta in enumerate(d_max):
            # If the travel don't cross multiple pixel
            if delta == 0:
                i_g[ii: ii + delta + 1] = i_x0[i]
                j_g[ii: ii + delta + 1] = i_y0[i]
            # Vertical move
            elif d_x[i] == 0:
                sup = -1 if d_y[i] < 0 else 1
                i_g[ii: ii + delta + 1] = i_x0[i]
                j_g[ii: ii + delta + 1] = arange(i_y0[i], i_y1[i] + sup, sup)
            # Horizontal move
            elif d_y[i] == 0:
                sup = -1 if d_x[i] < 0 else 1
                i_g[ii: ii + delta + 1] = arange(i_x0[i], i_x1[i] + sup, sup) % nb_x
                j_g[ii: ii + delta + 1] = i_y0[i]
            # In case of multiple direction
            else:
                a = (i_x1[i] - i_x0[i]) / float(i_y1[i] - i_y0[i])
                if abs(d_x[i]) >= abs(d_y[i]):
                    sup = -1 if d_x[i] < 0 else 1
                    value = arange(i_x0[i], i_x1[i] + sup, sup)
                    i_g[ii: ii + delta + 1] = value % nb_x
                    j_g[ii: ii + delta + 1] = (value - i_x0[i]) / a + i_y0[i]
                else:
                    sup = -1 if d_y[i] < 0 else 1
                    j_g[ii: ii + delta + 1] = arange(i_y0[i], i_y1[i] + sup, sup)
                    i_g[ii: ii + delta + 1] = (int_(j_g[ii: ii + delta + 1]) - i_y0[i]) * a + i_x0[i]
            ii += delta + 1
        i_g %= nb_x
        return i_g, j_g, d_max

    def clean_land(self):
        """Function to remove all land pixel
        """
        pass

    def is_circular(self):
        """Check if grid is circular
        """
        return abs((self.x_bounds[0] % 360) - (self.x_bounds[-1] % 360)) < 0.0001

    def _low_filter(self, grid_name, x_cut, y_cut):
        """low filtering
        """
        i_x, i_y = x_cut * 0.125 / self.xstep, y_cut * 0.125 / self.xstep
        logging.info(
            'Filtering with this wave : (%s, %s) converted in pixel (%s, %s)',
            x_cut, y_cut, i_x, i_y
        )
        data = self.grid(grid_name).copy()
        data[data.mask] = 0
        return self._gaussian_filter(
            data,
            (i_x, i_y),
            mode='wrap' if self.is_circular() else 'reflect')

    def add_uv(self, grid_height):
        """Compute a u and v grid
        """
        h_dict = self.variables_description[grid_height]
        for variable in ('u', 'v'):
            self.variables_description[variable] = dict(
                infos=h_dict['infos'].copy(),
                attrs=h_dict['attrs'].copy(),
                args=tuple((variable, *h_dict['args'][1:])),
                kwargs=h_dict['kwargs'].copy(),
            )
            self.variables_description[variable]['attrs']['units'] += '/s'
        data = self.grid(grid_height)
        gof = sin(deg2rad(self.y_c)) * ones((self.x_c.shape[0], 1)) * 4. * pi / 86400.
        # gof = sin(deg2rad(self.y_c))* ones((self.x_c.shape[0], 1))  * 4. * pi / (23 * 3600 + 56 *60 +4.1 )
        with errstate(divide='ignore'):
            gof = self.GRAVITY / (gof * ones((self.x_c.shape[0], 1)))

        m_y = array((1, 0, -1))
        m_x = array((1, 0, -1))
        d_hy = convolve(
            data,
            weights=m_y.reshape((-1, 1)).T
        )
        mask = convolve(
            int8(data.mask),
            weights=ones(m_y.shape).reshape((1, -1))
        )
        d_hy = ma.array(d_hy, mask=mask != 0)

        d_y = self.EARTH_RADIUS * 2 * pi / 360 * convolve(self.y_c, m_y)

        self.vars['u'] = - d_hy / d_y * gof
        mode = 'wrap' if self.is_circular() else 'reflect'
        d_hx = convolve(
            data,
            weights=m_x.reshape((-1, 1)),
            mode=mode,
        )
        mask = convolve(
            int8(data.mask),
            weights=ones(m_x.shape).reshape((-1, 1)),
            mode=mode,
        )
        d_hx = ma.array(d_hx, mask=mask != 0)
        d_x_degrees = convolve(self.x_c, m_x, mode=mode).reshape((-1, 1))
        d_x_degrees = (d_x_degrees + 180) % 360 - 180
        d_x = self.EARTH_RADIUS * 2 * pi / 360 * d_x_degrees * cos(deg2rad(self.y_c))
        self.vars['v'] = d_hx / d_x * gof
