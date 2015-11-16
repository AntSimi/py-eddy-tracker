# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import logging


class ColoredFormatter(logging.Formatter):
    COLOR_LEVEL = dict(
        CRITICAL="\037[37;41m",
        ERROR="\033[31;47m",
        WARNING="\033[30;47m",
        INFO="\033[36m",
        DEBUG="\033[34m",
        )

    def __init__(self, message):
        super(ColoredFormatter, self).__init__(message)

    def format(self, record):
        color = self.COLOR_LEVEL.get(record.levelname, '')
        color_reset = '\033[0m'
        model = color + '%s' + color_reset
        record.msg = model % record.msg
        record.funcName = model % record.funcName
        record.module = model % record.module
        record.levelname = model % record.levelname
        return super(ColoredFormatter, self).format(record)


class EddyParser(ArgumentParser):
    """General parser for applications
    """

    FORMAT_LOG = "%(levelname)-8s %(asctime)s %(module)s." \
                     "%(funcName)s :\n\t\t\t\t\t%(message)s"

    def __init__(self, *args, **kwargs):
        super(EddyParser, self).__init__(*args, **kwargs)
        self.add_base_argument()

    def add_base_argument(self):
        """Base arguments
        """
        self.add_argument('-v', '--verbose',
                          dest='logging_level',
                          default='ERROR',
                          help='Levels : DEBUG, INFO, WARNING,'
                          ' ERROR, CRITICAL')
        
    def parse_args(self, *args, **kwargs):
        # set up logging to CONSOLE
        console = logging.StreamHandler()
        console.setFormatter(ColoredFormatter(self.FORMAT_LOG))
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        # Parsing
        opts = super(EddyParser, self).parse_args(*args, **kwargs)
        # set current level
        logging.getLogger().setLevel(getattr(logging, opts.logging_level.upper()))
        return opts


VAR_DESCR = dict(
    time=dict(
        attr_name='time',
        nc_name='j1',
        nc_type='int32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='Julian date',
            units='days',
            description='date of this observation',
            reference=2448623,
            reference_description="Julian date on Jan 1, 1992",
            )
        ),
    ocean_time=dict(
        attr_name=None,
        nc_name='ocean_time',
        nc_type='float64',
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='ROMS ocean_time (seconds)',
            )
        ),
    type_cyc=dict(
        attr_name=None,
        nc_name='cyc',
        nc_type='byte',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='cyclonic',
            units='boolean',
            description='cyclonic -1; anti-cyclonic +1',
            )
        ),
    lon=dict(
        attr_name='lon',
        compute_type='float64',
        nc_name='lon',
        nc_type='float32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='deg. longitude',
            )
        ),
    lat=dict(
        attr_name='lat',
        compute_type='float64',
        nc_name='lat',
        nc_type='float32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='deg. latitude',
            )
        ),
    amplitude=dict(
        attr_name='amplitude',
        nc_name='A',
        nc_type='float32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='amplitude',
            units='cm',
            description='magnitude of the height difference between the '
                        'extremum of SSH within the eddy and the SSH '
                        'around the contour defining the eddy perimeter',
            )
        ),
    radius=dict(
        attr_name=None,
        nc_name='L',
        nc_type='float32',
        nc_dims=('Nobs',),
        scale_factor=1e-3,
        nc_attr=dict(
            long_name='speed radius scale',
            units='km',
            description='radius of a circle whose area is equal to that '
                        'enclosed by the contour of maximum circum-average'
                        ' speed',
            )
        ),
    speed_radius=dict(
        attr_name='speed_radius',
        scale_factor=100,
        nc_name='U',
        nc_type='float32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='maximum circum-averaged speed',
            units='cm/sec',
            description='average speed of the contour defining the radius '
                        'scale L',
            )
        ),
    eke=dict(
        attr_name='eke',
        nc_name='Teke',
        nc_type='float32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='sum EKE within contour Ceff',
            units='m^2/sec^2',
            description='sum of eddy kinetic energy within contour '
                        'defining the effective radius',
            )
        ),
    radius_e=dict(
        attr_name='radius_e',
        scale_factor=1e-3,
        nc_name='radius_e',
        nc_type='float32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='effective radius scale',
            units='km',
            description='effective eddy radius',
            )
        ),
    radius_s=dict(
        attr_name='radius_s',
        scale_factor=1e-3,
        nc_name='L',
        nc_type='float32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='speed radius scale',
            units='km',
            description='speed eddy radius',
            )
        ),
    track=dict(
        attr_name=None,
        nc_name='track',
        nc_type='int32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='track number',
            units='ordinal',
            description='eddy identification number',
            )
        ),
    n=dict(
        attr_name=None,
        nc_name='n',
        nc_type='int16',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='observation number',
            units='ordinal',
            description='observation sequence number (XX day intervals)',
            )
        ),
    contour_e=dict(
        attr_name=None,
        nc_name='contour_e',
        nc_type='f4',
        nc_dims=('contour_points', 'Nobs',),
        nc_attr=dict(
            long_name='positions of effective contour points',
            description='lons/lats of effective contour points; lons (lats) '
                        'in first (last) half of vector',
            )
        ),
    contour_s=dict(
        attr_name=None,
        nc_name='contour_s',
        nc_type='f4',
        nc_dims=('contour_points', 'Nobs',),
        nc_attr=dict(
            long_name='positions of speed-based contour points',
            description='lons/lats of speed-based contour points; lons (lats) '
                        'in first (last) half of vector',
            )
        ),
    uavg_profile=dict(
        attr_name=None,
        nc_name='uavg_profile',
        nc_type='f4',
        nc_dims=('uavg_contour_count', 'Nobs',),
        nc_attr=dict(
            long_name='radial profile of uavg',
            description='all uavg values from effective contour inwards to '
                        'smallest inner contour (pixel == 1)',
            )
        ),
    shape_error=dict(
        attr_name=None,
        nc_name='shape_error',
        nc_type='f2',
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='%',
            )
        ),
    )
