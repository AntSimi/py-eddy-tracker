# -*- coding: utf-8 -*-
VAR_DESCR = dict(
    time=dict(
        attr_name='new_time_tmp',
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
        attr_name='new_lon_tmp',
        nc_name='lon',
        nc_type='float32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='deg. longitude',
            )
        ),
    lat=dict(
        attr_name='new_lat_tmp',
        nc_name='lat',
        nc_type='float32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            units='deg. latitude',
            )
        ),
    amplitude=dict(
        attr_name='new_amp_tmp',
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
        nc_attr=dict(
            long_name='speed radius scale',
            units='km',
            description='radius of a circle whose area is equal to that '
                        'enclosed by the contour of maximum circum-average'
                        ' speed',
            )
        ),
    speed_radius=dict(
        attr_name='new_uavg_tmp',
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
        attr_name='new_teke_tmp',
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
        attr_name='new_radii_e_tmp',
        nc_name='radius_e',
        nc_type='float32',
        nc_dims=('Nobs',),
        nc_attr=dict(
            long_name='effective radius scale',
            units='km',
            description='effective eddy radius',
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
    )