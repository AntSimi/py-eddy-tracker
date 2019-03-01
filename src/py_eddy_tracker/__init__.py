# -*- coding: utf-8 -*-
"""
===========================================================================
This file is part of py-eddy-tracker.

    py-eddy-tracker is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    py-eddy-tracker is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with py-eddy-tracker.  If not, see <http://www.gnu.org/licenses/>.

Copyright (c) 2014-2017 by Antoine Delepoulle and Evan Mason
Email: emason@imedea.uib-csic.es
===========================================================================

__init__.py

Version 3.0.0

===========================================================================

"""
from argparse import ArgumentParser
import logging


class ColoredFormatter(logging.Formatter):
    COLOR_LEVEL = dict(
        CRITICAL="\037[37;41m",
        ERROR="\033[31;47m",
        WARNING="\033[30;47m",
        INFO="\033[36m",
        DEBUG="\033[34m\t",
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
    time_jj=dict(
        attr_name='time',
        nc_name='time',
        nc_type='int32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            standard_name='time',
            units='days since 1950-01-01 00:00:00',
            calendar='proleptic_gregorian',
            axis='T',
            long_name='time of gridded file',
            description='date of this observation',
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
    segment_size=dict(
        attr_name=None,
        nc_name='segment_size',
        nc_type='byte',
        nc_dims=('Nobs',),
        nc_attr=dict()
        ),
    dlon=dict(
        attr_name=None,
        nc_name='dlon',
        nc_type='float64',
        nc_dims=('Nobs',),
        nc_attr=dict()
        ),
    dlat=dict(
        attr_name=None,
        nc_name='dlat',
        nc_type='float64',
        nc_dims=('Nobs',),
        nc_attr=dict()
        ),
    virtual=dict(
        attr_name=None,
        nc_name='virtual',
        nc_type='byte',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='virtual_position',
            units='boolean',
            description='Virtual observation: 0 for real',
            )
        ),
    cost_association=dict(
        attr_name=None,
        nc_name='cost_association',
        nc_type='float32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='cost_value_to_associate_with_next_observation',
            description='Cost value to associate with the next observation',
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
    i=dict(
        attr_name='i',
        nc_name='i',
        nc_type='uint16',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='longitude index in the grid of the detection',
            description='longitude index in the grid of the detection',
            )
        ),
    j=dict(
        attr_name='j',
        nc_name='j',
        nc_type='uint16',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='latitude index in the grid of the detection',
            description='latitude index in the grid of the detection',
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
        nc_name='radius_e',
        nc_type='float32',
        output_type='u2',
        scale_factor=0.05,
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='effective radius scale',
            units='km',
            description='effective eddy radius',
            )
        ),
    radius_s=dict(
        attr_name='radius_s',
        nc_name='L',
        nc_type='float32',
        output_type='u2',
        scale_factor=0.05,
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
        nc_type='uint32',
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
        nc_type='uint16',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='observation number',
            units='ordinal',
            description='observation sequence number (XX day intervals)',
            )
        ),
    contour_lon_e=dict(
        attr_name=None,
        nc_name='contour_lon_e',
        nc_type='f4',
        output_type='i2',
        scale_factor=0.01,
        add_offset=180,
        nc_dims=('Nobs', 'NbSample'),
        nc_attr=dict(
            long_name='Longitude of contour',
            description='lons of contour points',
            )
        ),
    contour_lat_e=dict(
        attr_name=None,
        nc_name='contour_lat_e',
        nc_type='f4',
        output_type='i2',
        scale_factor=0.01,
        nc_dims=('Nobs', 'NbSample'),
        nc_attr=dict(
            long_name='Latitude of contour',
            description='lats of contour points',
            )
        ),
    contour_lon_s=dict(
        attr_name=None,
        nc_name='contour_lon_s',
        nc_type='f4',
        output_type='i2',
        scale_factor=0.01,
        add_offset=180,
        nc_dims=('Nobs', 'NbSample'),
        nc_attr=dict(
            long_name='Longitude of speed-based contour points',
            description='lons  of speed-based contour points',
            )
        ),
    contour_lat_s=dict(
        attr_name=None,
        nc_name='contour_lat_s',
        nc_type='f4',
        output_type='i2',
        scale_factor=0.01,
        nc_dims=('Nobs', 'NbSample'),
        nc_attr=dict(
            long_name='Latitude of speed-based contour points',
            description='lats of speed-based contour points',
            )
        ),
    uavg_profile=dict(
        attr_name=None,
        nc_name='uavg_profile',
        nc_type='f4',
        nc_dims=('Nobs', 'NbSample'),
        nc_attr=dict(
            long_name='radial profile of uavg',
            description='all uavg values from effective contour inwards to '
                        'smallest inner contour (pixel == 1)',
            )
        ),
    shape_error_e=dict(
        attr_name=None,
        nc_name='shape_error_e',
        nc_type='f2',
        output_type='u1',
        scale_factor=0.5,
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='%',
            )
        ),
    shape_error_s=dict(
        attr_name=None,
        nc_name='shape_error_s',
        nc_type='f2',
        output_type='u1',
        scale_factor=0.5,
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='%',
            )
        ),
    height_max_speed_contour=dict(
        attr_name=None,
        nc_name='height_max_speed_contour',
        nc_type='f4',
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='m',
            )
        ),
    height_external_contour=dict(
        attr_name=None,
        nc_name='height_external_contour',
        nc_type='f4',
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='m',
            )
        ),
    height_inner_contour=dict(
        attr_name=None,
        nc_name='height_inner_contour',
        nc_type='f4',
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='m',
            )
        ),
    nb_contour_selected=dict(
        attr_name=None,
        nc_name='nb_contour_selected',
        nc_type='u2',
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='1',
            )
        ),
    )

for key in VAR_DESCR.keys():
    if 'output_type' not in VAR_DESCR[key]:
        VAR_DESCR[key]['output_type'] = VAR_DESCR[key]['nc_type']
    
VAR_DESCR_inv = {VAR_DESCR[key]['nc_name'] : key for key in VAR_DESCR.keys()}
