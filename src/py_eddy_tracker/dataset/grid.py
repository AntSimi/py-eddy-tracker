# -*- coding: utf-8 -*-
"""
"""
import logging
from scipy.interpolate  import interp1d
from numpy import concatenate, int32, empty, maximum, where, array, \
    sin, deg2rad, pi, ones, cos, ma, int8, histogram2d, arange, float_, \
    linspace, meshgrid, sinc, errstate, float64, uint32
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter, convolve
from scipy.ndimage.filters import convolve1d
from scipy.spatial import cKDTree
from scipy.interpolate import RectBivariateSpline
from matplotlib.figure import Figure
from matplotlib.path import Path as BasePath
from pyproj import Proj
from ..tools import fit_circle_c, distance_vector, winding_number_poly
from ..property_functions import uniform_resample

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


def nearest_grd_indice(self, lon_value, lat_value, grid):
    if not hasattr(self, '_grid_indices'):
        self._grid_indices = grid.nearest_indices(lon_value, lat_value)
    return self._grid_indices
BasePath.nearest_grd_indice = nearest_grd_indice


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
        self._circle_params =  centlon_e, centlat_e, eddy_radius_e, aerr
    except ZeroDivisionError:
        # Some time, edge is only a dot of few coordinates
        if len(unique(self.lon)) == 1 and len(unique(self.lat)) == 1:
            logging.warning('An edge is only define in one position')
            logging.debug('%d coordinates %s,%s', len(self.lon), self.lon,
                          self.lat)
            self._circle_params = 0, -90, nan, nan

BasePath.fit_circle = fit_circle_path
BasePath._fit_circle_path = _fit_circle_path


class GridDataset(object):
    """
    Class to have basic tool on NetCDF Grid
    """

    __slots__ = (
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
        if self.centered is None:
            return True
        else:
            return self.centered

    def load_general_features(self):
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
                    kwargs['fill_value']= v._FillValue,
                attrs = dict()
                for attr in v.ncattrs():
                    if attr in kwargs.keys():
                        continue
                    if attr == '_FillValue':
                        continue
                    attrs[attr] =  getattr(v, attr)
                self.variables_description[i] = dict(
                    args=args,
                    kwargs=kwargs,
                    attrs=attrs,
                    infos=dict())
            self.global_attrs = {attr: getattr(h, attr) for attr in h.ncattrs()}

    def write(self, filename):
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
                var = h_out.createVariable(* variable['args'], ** variable['kwargs'])
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
        x_name, y_name = self.coordinates
        with Dataset(self.filename) as h:
            self.x_dim = h.variables[x_name].dimensions
            self.y_dim = h.variables[y_name].dimensions

            self.vars[x_name] = h.variables[x_name][:]
            self.vars[y_name] = h.variables[y_name][:]

        self.transform_coordinates()

        self.init_pos_interpolator()

    def is_circular(self):
        return False

    def grid(self, varname):
        if varname not in self.vars:
            coordinates_dims = list(self.x_dim)
            coordinates_dims.extend(list(self.y_dim))
            with Dataset(self.filename) as h:
                dims = h.variables[varname].dimensions
                sl = [slice(None) if dim in coordinates_dims else 0 for dim in dims]
                self.vars[varname] = h.variables[varname][sl]
                if len(self.x_dim) ==1 :
                    i_x = where(array(dims) == self.x_dim)[0][0]
                    i_y = where(array(dims) == self.y_dim)[0][0]
                    if i_x > i_y:
                        self.variables_description[varname]['infos']['transpose'] = True
                        self.vars[varname] = self.vars[varname].T
        return self.vars[varname]

    def high_filter(self, grid_name, x_cut, y_cut):
        result = self._low_filter(grid_name, x_cut, y_cut)
        self.vars[grid_name] -= result

    def low_filter(self, grid_name, x_cut, y_cut):
        result = self._low_filter(grid_name, x_cut, y_cut)
        self.vars[grid_name] -= self.vars[grid_name] - result

    @property
    def bounds(self):
        return self.x_bounds.min(), self.x_bounds.max(), \
            self.y_bounds.min(), self.y_bounds.max()

    def contours(self, grid_name, step):
        fig = Figure()
        ax = fig.add_subplot(111)
        data = self.grid(grid_name)
        z_min, z_max = data.min(), data.max()
        
        levels = arange(z_min - z_min % step, z_max - z_max % step + 2 * step, step)
        logging.info('We set %d levels : %s', len(levels), levels)

        x, y = self.vars[self.coordinates[0]], self.vars[self.coordinates[1]]
        if len(x.shape) == 1:
            data = data.T
        return ax.contour(x, y, data, levels)

    def eddy_identification(self, grid_name, step=0.005, shape_error=50):
        contours = self.contours(grid_name, step)
        grid = self.grid(grid_name)
        u = self.grid('u')
        v = self.grid('v')

        anticyclonic_search = True
        iterator = 1 if anticyclonic_search else -1

        # Loop over each collection
        for coll_ind, coll in enumerate(contours.collections[::iterator]):
            corrected_coll_index = iterator * coll_ind - 1

            contour_value = contours.cvalues[corrected_coll_index]
            contour_paths = coll.get_paths()
            nb_paths = len(contour_paths)
            if nb_paths == 0:
                continue
            logging.debug('doing collection %s, contour value %s, %d paths',
                          corrected_coll_index, contour_value, nb_paths)
            # Loop over individual c_s contours (i.e., every eddy in field)
            for contour in contour_paths:
                # Filter for closed contours
                if not contour.isvalid:
                    continue
                lon_e, lat_e, radius_e, err_e = contour.fit_circle()
                # Filter for shape: >50% is not an eddy 
                if err_e < 0 or err_e > shape_error:
                    continue

                # Get indices of centroid
                i_lon, j_lat = contour.nearest_grd_indice(lon_e, lat_e, self)

                # print(i_lon, j_lat)
                sla = grid[i_lon, j_lat]
                # print(sla)
                # print(type(sla))
                # if sla is masked:
                    # continue
                if anticyclonic_search != (sla >= contour_value):
                    continue

                # Instantiate new EddyObservation object
                # properties = EddiesObservations(
                    # size=1,
                    # track_extra_variables=eddy.track_extra_variables,
                    # track_array_variables=eddy.track_array_variables_sampling,
                    # array_variables=eddy.track_array_variables
                    # )

                contour_lon, contour_lat = contour.lon, contour.lat
                # Set indices to bounding box around eddy
                eddy_indices = self.get_pixel_indices_in_contour(contour)
                eddy_sla = grid[eddy_indices]

                # sum(mask) between 8 and 1000, CSS11 criterion 2
                if len(eddy_sla) > 8:
                    # print(eddy_indices)
                    if anticyclonic_search:
                        i_max = eddy_sla.argmax()
                        amp = eddy_sla[i_max] - contour_value
                    else:
                        i_max = eddy_sla.argmin()
                        amp = contour_value - eddy_sla[i_max]

                    teke = (u[eddy_indices] ** 2 + u[eddy_indices] ** 2 ).sum() * 0.5

                    # Get sum of eke within Ceff
                    args = (eddy, contours, centlon_e, centlat_e, contour, grd,
                            anticyclonic_search)

                    if True:
                        (uavg, contlon_s, contlat_s,
                         inner_contlon, inner_contlat,
                         any_inner_contours
                         ) = get_uavg(*args)
                    #~ else:
                        #~ (uavg, contlon_s, contlat_s,
                         #~ inner_contlon, inner_contlat,
                         #~ any_inner_contours, uavg_profile
                         #~ ) = get_uavg(
                            #~ *args, save_all_uavg=True)

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
                    if 'contour_lon_e' in eddy.track_array_variables:
                        (properties.obs['contour_lon_e'],
                         properties.obs['contour_lat_e']) = uniform_resample(
                            contour_lon, contour_lat,
                            fixed_size=eddy.track_array_variables_sampling)
                    if 'contour_lon_s' in eddy.track_array_variables:
                        (properties.obs['contour_lon_s'],
                         properties.obs['contour_lat_s']) = uniform_resample(
                            contlon_s, contlat_s,
                            fixed_size=eddy.track_array_variables_sampling)

                    # for AVISO
                    eddy.update_eddy_properties(properties)

                    # Mask out already found eddies
                    eddy.sla[eddy.slice_j, eddy.slice_i][
                        eddy.mask_eff] = eddy.fillval

    
    def _gaussian_filter(self, data, sigma, mode='reflect'):
        local_data = data.copy()
        has_mask = hasattr(data, 'mask')
        if has_mask:
            local_data[data.mask] = 0

        v = gaussian_filter(local_data, sigma=sigma, mode=mode)
        w = gaussian_filter(
            float_(-data.mask) if has_mask else ones(data.shape),
            sigma=sigma,
            mode=mode)

        with errstate(invalid='ignore'):
            if has_mask:
                return ma.array(v / w, mask= w==0)
            else:
                return v / w

    def create_uv_var(self, ref_var):
        h_dict = self.variables_description[ref_var]
        for variable in ('u', 'v'):
            self.variables_description[variable] = dict(
                infos=h_dict['infos'].copy(),
                attrs=h_dict['attrs'].copy(),
                args=tuple((variable, * h_dict['args'][1:])),
                kwargs=h_dict['kwargs'].copy(),
                )
            if 'units' in self.variables_description[variable]['attrs'].keys():
                self.variables_description[variable]['attrs']['units'] += '/s'


class UnRegularGridDataset(GridDataset):

    __slots__ = ('xy_interp')

    def transform_coordinates(self):
        x_name, y_name = self.coordinates
        if self.is_centered:
            self.x_c = self.vars[x_name]
            self.y_c = self.vars[y_name]

            #
            # depends order of dimensions !!
            self.x_bounds = concatenate((
                self.x_c, (2 * self.x_c[-1] - self.x_c[-2],)))
            self.x_bounds = concatenate((self.x_bounds, self.x_bounds[:,-1:]), axis=1)

            self.y_bounds = concatenate((
                self.y_c, 2 * self.y_c[:, -1:] - self.y_c[:, -2:-1]), axis = 1)
            self.y_bounds = concatenate((self.y_bounds, self.y_bounds[-1:]))
            # depends order of dimensions !!
            #

            d_x = self.x_bounds[1:] - self.x_bounds[:-1]
            d_y = self.y_bounds[1:] - self.y_bounds[:-1]
            self.x_bounds[:-1] -= d_x / 2
            self.x_bounds[-1] -= d_x[-1] / 2
            self.y_bounds[:-1] -= d_y / 2
            self.y_bounds[-1] -= d_y[-1] / 2

        else:
            self.x_bounds = self.vars[x_name]
            self.y_bounds = self.vars[y_name]

            raise Exception('not write')

    def compute_pixel_path(self, x0, y0, x1, y1):
        pass

    def init_pos_interpolator(self):
        self.xy_interp = cKDTree(concatenate((self.x_c.reshape(1,-1), self.y_c.reshape(1,-1))).T)
        # self.xy_interp = cKDTree(concatenate((self.x_c, self.y_c)).reshape(-1,2))

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
            z_grid = ma.array(sum_value / nb_value, mask =nb_value == 0)
        i_x, i_y = x_cut * 0.125 / regrid_x_step, y_cut * 0.125 / regrid_y_step
        m = nb_value == 0

        z_filtered = self._gaussian_filter(z_grid, (i_x, i_y))

        z_filtered[m] = 0
        x_center = (bounds_x[:-1] + bounds_x[1:]) / 2
        y_center = (bounds_y[:-1] + bounds_y[1:]) / 2
        opts_interpolation = dict(kx=1, ky=1, s=0)
        m_interp = RectBivariateSpline(x_center, y_center, m, **opts_interpolation)
        z_interp = RectBivariateSpline(x_center, y_center, z_filtered, **opts_interpolation).ev(x, y)
        return ma.array(z_interp, mask=m_interp.ev(x,y) > 0.00001)

    def add_uv(self, grid_height):
        self.create_uv_var(grid_height)
        data = self.grid(grid_height)
        gof = sin(deg2rad(self.y_c)) * 4. * pi / 86400.
        # gof = sin(deg2rad(self.y_c)) * 4. * pi / (23 * 3600 + 56 *60 +4.1 )
        with errstate(divide='ignore'):
            gof = self.GRAVITY / gof

        m_y = array((1, 0, -1)).reshape((1,3))
        m_x = array((1, 0, -1)).reshape((3,1))
        d_hy = convolve(
            data,
            weights=m_y
            )
        mask = convolve(
            int8(data.mask),
            weights=ones(m_y.shape)
            )
        d_hy = ma.array(d_hy, mask=mask != 0)

        nb_y, nb_x = self.x_c.shape
        shape = (nb_y - 2, nb_x)
        d_y = empty(shape, dtype=float64).reshape(-1)
        distance_vector(
            self.x_c[:-2,:].reshape(-1).astype(float64),
            self.y_c[:-2,:].reshape(-1).astype(float64),
            self.x_c[2:,:].reshape(-1).astype(float64),
            self.y_c[2:,:].reshape(-1).astype(float64),
            d_y,
            )
        d_y_final = empty((nb_y, nb_x), dtype=float_)
        d_y_final[1:-1] = d_y.reshape(shape)
        d_y_final[0] = d_y_final[1] # / 2
        d_y_final[-1] = d_y_final[-2] # / 2
        
        self.vars['u'] = - d_hy / d_y_final * gof

        mode = 'wrap' if self.is_circular() else 'reflect'
        d_hx = convolve(
            data,
            weights=m_x,
            mode=mode,
            )
        mask = convolve(
            int8(data.mask),
            weights=ones(m_x.shape),
            mode=mode,
            )
        d_hx = ma.array(d_hx, mask=mask != 0)

        shape = (nb_y, nb_x - 2)
        d_x = empty(shape, dtype=float64).reshape(-1)
        
        distance_vector(
            self.x_c[:, :-2].reshape(-1).astype(float64),
            self.y_c[:, :-2].reshape(-1).astype(float64),
            self.x_c[:, 2:].reshape(-1).astype(float64),
            self.y_c[:, 2:].reshape(-1).astype(float64),
            d_x,
            )
        d_x_final = empty((nb_y, nb_x), dtype=float_)
        d_x_final[:, 1:-1] = d_x.reshape(shape)
        d_x_final[:,0] = d_x_final[:,1] # / 2
        d_x_final[:,-1] = d_x_final[:,-2] # / 2

        self.vars['v'] = d_hx / d_x_final * gof

    @staticmethod
    def unravel_index(shape, indexes):
        nb_y, nb_x = shape
        i_x = indexes % nb_x
        i_y = (indexes - i_x) / nb_x
        return uint32(i_y), uint32(i_x)
        
    def nearest_indices(self, xs, ys):
        dist, indexes = self.xy_interp.query((xs,ys))
        return self.unravel_index(self.x_c.shape, uint32(indexes))

    def get_bounds(self, lons, lats, offset=1):
        i_lons, i_lats = empty(lons.shape, dtype='u4'), empty(lons.shape, dtype='u4')
        for i in range(len(lons)):
            i_lons[i], i_lats[i] = self.nearest_indices(lons[i], lats[i])
        return (
            slice(i_lons.min() - offset, i_lons.max() + offset + 1),
            slice(i_lats.min() - offset, i_lats.max() + offset + 1))

    def get_pixel_indices_in_contour(self, contour):
        lons, lats = contour.lon, contour.lat
        bbox = self.get_bounds(lons, lats)
        grid_lons, grid_lats = self.x_c[bbox].reshape(-1), self.y_c[bbox].reshape(-1)
        ravel_indices = list()
        for i, grid_lon in enumerate(grid_lons):
            if winding_number_poly(grid_lon, grid_lats[i], contour.vertices) != 0:
                ravel_indices.append(i)
        relative_indices = self.unravel_index(
            (bbox[0].stop - bbox[0].start, bbox[1].stop - bbox[1].start),
            array(ravel_indices, dtype='u4'))
        return relative_indices[0] + bbox[0].start, relative_indices[1] + bbox[1].start


class RegularGridDataset(GridDataset):
    
    __slots__ = ()

    def transform_coordinates(self):
        x_name, y_name = self.coordinates
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

            raise Exception('not test')
            self.x_c = (self.x_bounds[1:] + self.x_bounds[:-1]) / 2
            self.y_c = (self.y_bounds[1:] + self.y_bounds[:-1]) / 2

    def init_pos_interpolator(self):
        self.xinterp = interp1d(self.x_bounds, range(self.x_bounds.shape[0]), assume_sorted=True)
        self.yinterp = interp1d(self.y_bounds, range(self.y_bounds.shape[0]), assume_sorted=True)

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
        pass

    def is_circular(self):
        return abs((self.x_bounds[0] % 360) - (self.x_bounds[-1] % 360)) < 0.0001

    def _low_filter(self, grid_name, x_cut, y_cut):
        i_x, i_y = x_cut * 0.125 / self.xstep, y_cut * 0.125 / self.xstep
        logging.info(
            'Filtering with this wave : (%s, %s) converted in pixel (%s, %s)',
            x_cut, y_cut, i_x, i_y
            )
        data = self.grid(grid_name).copy()
        return self._gaussian_filter(
            data,
            (i_x, i_y),
            mode='wrap' if self.is_circular() else 'reflect')

    def add_uv(self, grid_height):
        self.create_uv_var(grid_height)
        data = self.grid(grid_height)
        gof = sin(deg2rad(self.y_c))* ones((self.x_c.shape[0], 1))  * 4. * pi / 86400.
        # gof = sin(deg2rad(self.y_c))* ones((self.x_c.shape[0], 1))  * 4. * pi / (23 * 3600 + 56 *60 +4.1 )
        with errstate(divide='ignore'):
            gof = self.GRAVITY / (gof * ones((self.x_c.shape[0], 1)))

        m_y = array((1, 0, -1))
        m_x = array((1, 0, -1))
        d_hy = convolve(
            data,
            weights=m_y.reshape((-1,1)).T
            )
        if hasattr(data, 'mask'):
            mask = convolve(
                int8(data.mask),
                weights=ones(m_y.shape).reshape((1,-1))
                )
            d_hy = ma.array(d_hy, mask=mask != 0)
        
        # Bad
        d_y = self.EARTH_RADIUS * 2 * pi / 360 * convolve(self.y_c, m_y)
        ##
        
        self.vars['u'] = - d_hy / d_y * gof
        mode = 'wrap' if self.is_circular() else 'reflect'
        d_hx = convolve(
            data,
            weights=m_x.reshape((-1,1)),
            mode=mode,
            )
        if hasattr(data, 'mask'):
            mask = convolve(
                int8(data.mask),
                weights=ones(m_x.shape).reshape((-1, 1)),
                mode=mode,
                )
            d_hx = ma.array(d_hx, mask=mask != 0)
        # Bad
        d_x_degrees = convolve(self.x_c, m_x, mode=mode).reshape((-1,1))
        d_x_degrees = (d_x_degrees + 180) % 360 - 180
        d_x = self.EARTH_RADIUS * 2 * pi / 360 * d_x_degrees * cos(deg2rad(self.y_c))
        ##
        self.vars['v'] = d_hx / d_x * gof
