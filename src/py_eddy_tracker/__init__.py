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

Copyright (c) 2014-2020 by Evan Mason
Email: evanmason@gmail.com
===========================================================================
"""

from argparse import ArgumentParser
import logging
import numpy
import zarr


def start_logger():
    FORMAT_LOG = (
        "%(levelname)-8s %(asctime)s %(module)s.%(funcName)s :\n\t\t\t\t\t%(message)s"
    )
    # set up logging to CONSOLE
    console = logging.StreamHandler()
    console.setFormatter(ColoredFormatter(FORMAT_LOG))
    logger = logging.getLogger("pet")
    # add the handler to the root logger
    logger.addHandler(console)
    return logger


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
        color = self.COLOR_LEVEL.get(record.levelname, "")
        color_reset = "\033[0m"
        model = color + "%s" + color_reset
        record.msg = model % record.msg
        record.funcName = model % record.funcName
        record.module = model % record.module
        record.levelname = model % record.levelname
        return super(ColoredFormatter, self).format(record)


class EddyParser(ArgumentParser):
    """General parser for applications
    """

    def __init__(self, *args, **kwargs):
        super(EddyParser, self).__init__(*args, **kwargs)
        self.add_base_argument()

    def add_base_argument(self):
        """Base arguments
        """
        self.add_argument(
            "-v",
            "--verbose",
            dest="logging_level",
            default="ERROR",
            help="Levels : DEBUG, INFO, WARNING," " ERROR, CRITICAL",
        )

    def parse_args(self, *args, **kwargs):
        logger = start_logger()
        # Parsing
        opts = super(EddyParser, self).parse_args(*args, **kwargs)
        # set current level
        logger.setLevel(getattr(logging, opts.logging_level.upper()))
        return opts


VAR_DESCR = dict(
    time=dict(
        attr_name="time",
        nc_name="time",
        old_nc_name=["j1"],
        nc_type="int32",
        nc_dims=("obs",),
        nc_attr=dict(
            standard_name="time",
            units="days since 1950-01-01 00:00:00",
            calendar="proleptic_gregorian",
            axis="T",
            longname="Time",
            description="Date of this observation",
        ),
    ),
    type_cyc=dict(
        attr_name=None,
        nc_name="cyclonic_type",
        old_nc_name=["cyc"],
        nc_type="byte",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="cyclonic",
            units="boolean",
            description="Cyclonic -1; anti-cyclonic +1",
        ),
    ),
    segment_size=dict(
        attr_name=None,
        nc_name="segment_size",
        nc_type="byte",
        nc_dims=("obs",),
        nc_attr=dict(),
    ),
    dlon=dict(
        attr_name=None,
        nc_name="dlon",
        nc_type="float64",
        nc_dims=("obs",),
        nc_attr=dict(),
    ),
    dlat=dict(
        attr_name=None,
        nc_name="dlat",
        nc_type="float64",
        nc_dims=("obs",),
        nc_attr=dict(),
    ),
    virtual=dict(
        attr_name=None,
        nc_name="observation_flag",
        old_nc_name=["virtual"],
        nc_type="byte",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="virtual_position",
            units="boolean",
            description="Flag indicating if the value is interpolated between two"
            " observations or not (0: observed, 1: interpolated)",
        ),
    ),
    cost_association=dict(
        attr_name=None,
        nc_name="cost_association",
        nc_type="float32",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="cost_value_to_associate_with_next_observation",
            description="Cost value to associate with the next observation",
        ),
    ),
    lon=dict(
        attr_name="lon",
        compute_type="float64",
        nc_name="longitude",
        old_nc_name=["lon", "Lon"],
        nc_type="float32",
        nc_dims=("obs",),
        nc_attr=dict(
            units="degrees_east",
            axis="X",
            description="Observation longitude",
            longname="longitude of measurement",
            standard_name="longitude",
        ),
    ),
    lat=dict(
        attr_name="lat",
        compute_type="float64",
        nc_name="latitude",
        old_nc_name=["lat", "Lat"],
        nc_type="float32",
        nc_dims=("obs",),
        nc_attr=dict(
            units="degrees_north",
            axis="Y",
            longname="latitude of measurement",
            standard_name="latitude",
            description="Observation latitude",
        ),
    ),
    lon_max=dict(
        attr_name="lon_max",
        compute_type="float64",
        nc_name="longitude_max",
        old_nc_name=["lon_max"],
        nc_type="float32",
        nc_dims=("obs",),
        nc_attr=dict(
            units="degrees_east",
            axis="X",
            description="Observation longitude",
            longname="longitude of amplitude max",
            standard_name="longitude",
        ),
    ),
    lat_max=dict(
        attr_name="lat_max",
        compute_type="float64",
        nc_name="latitude_max",
        old_nc_name=["lat_max"],
        nc_type="float32",
        nc_dims=("obs",),
        nc_attr=dict(
            units="degrees_north",
            axis="Y",
            description="Observation latitude",
            longname="latitude of amplitude max",
            standard_name="latitude",
        ),
    ),
    amplitude=dict(
        attr_name="amplitude",
        nc_name="amplitude",
        old_nc_name=["A"],
        nc_type="float32",
        output_type="uint16",
        scale_factor=0.001,
        nc_dims=("obs",),
        nc_attr=dict(
            longname="amplitude",
            units="m",
            description="Magnitude of the height difference between the extremum of ADT within "
            "the eddy and the ADT around the contour defining the eddy perimeter",
        ),
    ),
    speed_average=dict(
        attr_name="speed_average",
        scale_factor=0.0001,
        nc_name="speed_average",
        old_nc_name=["U"],
        nc_type="float32",
        output_type="uint16",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="maximum circum-averaged speed",
            units="m/s",
            description="Average speed of the contour defining the radius scale “speed_radius”",
        ),
    ),
    uavg_profile=dict(
        attr_name=None,
        nc_name="uavg_profile",
        output_type="u2",
        scale_factor=0.0001,
        nc_type="f4",
        # filters=[zarr.Delta('u2')],
        nc_dims=("obs", "NbSample"),
        nc_attr=dict(
            longname="radial profile of uavg",
            units="m/s",
            description="Speed average values from effective contour inwards to smallest inner contour",
        ),
    ),
    i=dict(
        attr_name="i",
        nc_name="i",
        nc_type="uint16",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="longitude index in the grid of the detection",
            description="Longitude index in the grid of the detection",
        ),
    ),
    j=dict(
        attr_name="j",
        nc_name="j",
        nc_type="uint16",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="latitude index in the grid of the detection",
            description="Latitude index in the grid of the detection",
        ),
    ),
    eke=dict(
        attr_name="eke",
        nc_name="Teke",
        nc_type="float32",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="sum EKE within contour Ceff",
            units="m^2/s^2",
            description="Sum of eddy kinetic energy within contour "
            "defining the effective radius",
        ),
    ),
    radius_e=dict(
        attr_name="radius_e",
        nc_name="effective_radius",
        old_nc_name=["radius_e", "Dia"],
        nc_type="float32",
        output_type="u2",
        scale_factor=50.0,
        nc_dims=("obs",),
        nc_attr=dict(
            longname="effective radius scale",
            units="m",
            description="Radius of a circle whose area is equal to that enclosed by the effective contour",
        ),
    ),
    radius_s=dict(
        attr_name="radius_s",
        nc_name="speed_radius",
        old_nc_name=["L", "radius_s"],
        nc_type="float32",
        output_type="u2",
        scale_factor=50.0,
        nc_dims=("obs",),
        nc_attr=dict(
            longname="speed radius scale",
            units="m",
            description="Radius of a circle whose area is equal to that \
                enclosed by the contour of maximum circum-average speed",
        ),
    ),
    track=dict(
        attr_name=None,
        nc_name="track",
        old_nc_name=["Eddy_id"],
        nc_type="uint32",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="track number",
            units="ordinal",
            description="Eddy identification number",
        ),
    ),
    sub_track=dict(
        attr_name=None,
        nc_name="sub_track",
        nc_type="uint32",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="segment_number",
            units="ordinal",
            description="segment number inside a group",
        ),
    ),
    n=dict(
        attr_name=None,
        nc_name="observation_number",
        old_nc_name=["n", "Eddy_tsp"],
        nc_type="uint16",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="observation number",
            units="ordinal",
            description="Observation sequence number, days from eddy first detection",
        ),
    ),
    contour_lon_e=dict(
        attr_name=None,
        nc_name="effective_contour_longitude",
        old_nc_name=["contour_lon_e"],
        nc_type="f4",
        filters=[zarr.Delta("i2")],
        output_type="i2",
        scale_factor=numpy.float32(0.01),
        add_offset=180,
        nc_dims=("obs", "NbSample"),
        nc_attr=dict(
            longname="effective contour longitudes",
            description="Longitudes of effective contour",
            units="degrees_east",
            axis="X",
        ),
    ),
    contour_lat_e=dict(
        attr_name=None,
        nc_name="effective_contour_latitude",
        old_nc_name=["contour_lat_e"],
        nc_type="f4",
        filters=[zarr.Delta("i2")],
        output_type="i2",
        scale_factor=numpy.float32(0.01),
        nc_dims=("obs", "NbSample"),
        nc_attr=dict(
            longname="effective contour latitudes",
            description="Latitudes of effective contour",
            units="degrees_east",
            axis="X",
        ),
    ),
    contour_lon_s=dict(
        attr_name=None,
        nc_name="speed_contour_longitude",
        old_nc_name=["contour_lon_s"],
        nc_type="f4",
        filters=[zarr.Delta("i2")],
        output_type="i2",
        scale_factor=numpy.float32(0.01),
        add_offset=180,
        nc_dims=("obs", "NbSample"),
        nc_attr=dict(
            longname="speed contour longitudes",
            description="Longitudes of speed contour",
            units="degrees_east",
            axis="X",
        ),
    ),
    contour_lat_s=dict(
        attr_name=None,
        nc_name="speed_contour_latitude",
        old_nc_name=["contour_lat_s"],
        nc_type="f4",
        filters=[zarr.Delta("i2")],
        output_type="i2",
        scale_factor=numpy.float32(0.01),
        nc_dims=("obs", "NbSample"),
        nc_attr=dict(
            longname="speed contour latitudes",
            description="Latitudes of speed contour",
            units="degrees_east",
            axis="X",
        ),
    ),
    shape_error_e=dict(
        attr_name=None,
        nc_name="effective_contour_shape_error",
        old_nc_name=["shape_error_e"],
        nc_type="f2",
        output_type="u1",
        scale_factor=0.5,
        nc_dims=("obs",),
        nc_attr=dict(
            units="%",
            description="Error criterion of fit on effective contour",
            longname="effective contour error",
        ),
    ),
    score=dict(
        attr_name=None,
        nc_name="score",
        nc_type="f2",
        output_type="u1",
        scale_factor=0.4,
        nc_dims=("obs",),
        nc_attr=dict(units="%", description="score", longname="score",),
    ),
    index_other=dict(
        attr_name=None,
        nc_name="index_other",
        nc_type="u4",
        nc_dims=("obs",),
        nc_attr=dict(
            units="ordinal",
            description="index in the other dataset",
            longname="index_other",
        ),
    ),
    shape_error_s=dict(
        attr_name=None,
        nc_name="speed_contour_shape_error",
        old_nc_name=["shape_error_s"],
        nc_type="f2",
        output_type="u1",
        scale_factor=0.5,
        nc_dims=("obs",),
        nc_attr=dict(
            units="%",
            description="Error criterion of fit on speed contour",
            longname="speed contour error",
        ),
    ),
    height_max_speed_contour=dict(
        attr_name=None,
        nc_name="speed_contour_height",
        old_nc_name=["height_max_speed_contour"],
        nc_type="f4",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="speed contour height",
            description="ADT filtered height for speed contour",
            units="m",
        ),
    ),
    height_external_contour=dict(
        attr_name=None,
        nc_name="effective_contour_height",
        old_nc_name=["height_external_contour"],
        nc_type="f4",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="effective contour height",
            description="ADT filtered height for effective contour",
            units="m",
        ),
    ),
    height_inner_contour=dict(
        attr_name=None,
        nc_name="inner_contour_height",
        old_nc_name=["height_inner_contour"],
        nc_type="f4",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="inner contour height",
            description="ADT filtered height for inner contour",
            units="m",
        ),
    ),
    chl=dict(
        attr_name=None,
        nc_name="chl",
        old_nc_name=["Chl"],
        nc_type="f4",
        nc_dims=("obs",),
        nc_attr=dict(longname="Log base 10 chlorophyll", units="Log(Chl/[mg/m^3])",),
    ),
    dchl=dict(
        attr_name=None,
        nc_name="dchl",
        old_nc_name=["dChl"],
        nc_type="f4",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="Log base 10 chlorophyll anomaly (Chl minus Chl_bg)",
            units="Log(Chl/[mg/m^3])",
        ),
    ),
    chl_bg=dict(
        attr_name=None,
        nc_name="chl_bg",
        old_nc_name=["Chl_bg"],
        nc_type="f4",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="Log base 10 background chlorophyll", units="Log(Chl/[mg/m^3])",
        ),
    ),
    year=dict(
        attr_name=None,
        nc_name="year",
        old_nc_name=["Year"],
        nc_type="u2",
        nc_dims=("obs",),
        nc_attr=dict(longname="Year", units="year",),
    ),
    month=dict(
        attr_name=None,
        nc_name="month",
        old_nc_name=["Month"],
        nc_type="u1",
        nc_dims=("obs",),
        nc_attr=dict(longname="Month", units="month",),
    ),
    day=dict(
        attr_name=None,
        nc_name="day",
        old_nc_name=["Day"],
        nc_type="u1",
        nc_dims=("obs",),
        nc_attr=dict(longname="Day", units="day",),
    ),
    nb_contour_selected=dict(
        attr_name=None,
        nc_name="num_contours",
        old_nc_name=["nb_contour_selected"],
        nc_type="u2",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="number of contour",
            units="ordinal",
            description="Number of contour selected for this eddy",
        ),
    ),
)

for key in VAR_DESCR.keys():
    if "output_type" not in VAR_DESCR[key]:
        VAR_DESCR[key]["output_type"] = VAR_DESCR[key]["nc_type"]

VAR_DESCR_inv = dict()
for key in VAR_DESCR.keys():
    VAR_DESCR_inv[VAR_DESCR[key]["nc_name"]] = key
    for key_old in VAR_DESCR[key].get("old_nc_name", list()):
        VAR_DESCR_inv[key_old] = key

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
