# -*- coding: utf-8 -*-
"""
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

Copyright (c) 2014-2020 by Evan Mason and Antoine Delepoulle
Email: evanmason@gmail.com

"""

import logging
from argparse import ArgumentParser

import zarr

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


def start_logger():
    FORMAT_LOG = (
        "%(levelname)-8s %(asctime)s %(module)s.%(funcName)s :\n\t\t\t\t\t%(message)s"
    )
    logger = logging.getLogger("pet")
    if len(logger.handlers) == 0:
        # set up logging to CONSOLE
        console = logging.StreamHandler()
        console.setFormatter(ColoredFormatter(FORMAT_LOG))
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
        super().__init__(message)

    def format(self, record):
        color = self.COLOR_LEVEL.get(record.levelname, "")
        color_reset = "\033[0m"
        model = color + "%s" + color_reset
        record.msg = model % record.msg
        record.funcName = model % record.funcName
        record.module = model % record.module
        record.levelname = model % record.levelname
        return super().format(record)


class EddyParser(ArgumentParser):
    """General parser for applications"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_base_argument()

    def add_base_argument(self):
        """Base arguments"""
        self.add_argument(
            "-v",
            "--verbose",
            dest="logging_level",
            default="ERROR",
            help="Levels : DEBUG, INFO, WARNING," " ERROR, CRITICAL",
        )

    def memory_arg(self):
        self.add_argument(
            "--memory",
            action="store_true",
            help="Load file in memory before to read with netCDF library",
        )

    def contour_intern_arg(self):
        self.add_argument(
            "--intern",
            action="store_true",
            help="Use intern contour instead of outter contour",
        )

    def parse_args(self, *args, **kwargs):
        logger = start_logger()
        # Parsing
        opts = super().parse_args(*args, **kwargs)
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
            long_name="Time",
            comment="Date of this observation",
        ),
    ),
    type_cyc=dict(
        attr_name=None,
        nc_name="cyclonic_type",
        old_nc_name=["cyc"],
        nc_type="byte",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Rotating sense of the eddy",
            comment="Cyclonic -1; Anti-cyclonic +1",
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
    distance_next=dict(
        attr_name=None,
        nc_name="distance_next",
        nc_type="float32",
        output_type="uint16",
        scale_factor=50.0,
        nc_dims=("obs",),
        nc_attr=dict(long_name="Distance to next position", units="m"),
    ),
    virtual=dict(
        attr_name=None,
        nc_name="observation_flag",
        old_nc_name=["virtual"],
        nc_type="byte",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Virtual Eddy Position",
            comment="Flag indicating if the value is interpolated between two"
            " observations or not (0: observed eddy, 1: interpolated eddy)",
        ),
    ),
    cost_association=dict(
        attr_name=None,
        nc_name="cost_association",
        nc_type="float32",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Cost association between two eddies",
            comment="Cost value to associate one eddy with the next observation",
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
            comment="Longitude center of the fit circle",
            long_name="Eddy Center Longitude",
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
            long_name="Eddy Center Latitude",
            standard_name="latitude",
            comment="Latitude center of the fit circle",
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
            long_name="Longitude of the SSH maximum",
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
            long_name="Latitude of the SSH maximum",
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
            long_name="Amplitude",
            units="m",
            comment="Magnitude of the height difference between the extremum of SSH within "
            "the eddy and the SSH around the effective contour defining the eddy edge",
        ),
    ),
    speed_area=dict(
        attr_name="speed_area",
        nc_name="speed_area",
        nc_type="float32",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Speed area",
            units="m^2",
            comment="Area enclosed by speed contour in m^2",
        ),
    ),
    effective_area=dict(
        attr_name="effective_area",
        nc_name="effective_area",
        nc_type="float32",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Effective area",
            units="m^2",
            comment="Area enclosed by effective contour in m^2",
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
            long_name="Maximum circum-averaged Speed",
            units="m/s",
            comment="Average speed of the contour defining the radius scale “speed_radius”",
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
            long_name="Radial Speed Profile",
            units="m/s",
            comment="Speed average values from effective contour inwards to smallest contour, evenly spaced points",
        ),
    ),
    i=dict(
        attr_name="i",
        nc_name="i",
        nc_type="uint16",
        nc_dims=("obs",),
        nc_attr=dict(long_name="Longitude index in the grid of the detection"),
    ),
    j=dict(
        attr_name="j",
        nc_name="j",
        nc_type="uint16",
        nc_dims=("obs",),
        nc_attr=dict(long_name="Latitude index in the grid of the detection"),
    ),
    eke=dict(
        attr_name="eke",
        nc_name="Teke",
        nc_type="float32",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="EKE",
            units="m^2/s^2",
            comment="Sum of Eddy Kinetic Energy within the effective contour",
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
            long_name="Effective Radius",
            units="m",
            comment="Radius of a circle whose area is equal to that enclosed by the effective contour",
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
            long_name="Speed Radius",
            units="m",
            comment="Radius of a circle whose area is equal to that "
            "enclosed by the contour of maximum circum-average speed",
        ),
    ),
    track=dict(
        attr_name=None,
        nc_name="track",
        old_nc_name=["Eddy_id"],
        nc_type="uint32",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Trajectory number", comment="Trajectory identification number"
        ),
    ),
    segment=dict(
        attr_name=None,
        nc_name="segment",
        nc_type="uint32",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Segment Number", comment="Segment number inside a group"
        ),
    ),
    previous_obs=dict(
        attr_name=None,
        nc_name="previous_obs",
        nc_type="int32",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Previous obs index",
            comment="Index of previous obs, if there are a spliting",
        ),
    ),
    next_obs=dict(
        attr_name=None,
        nc_name="next_obs",
        nc_type="int32",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Next obs index",
            comment="Index of next obs, if there are a merging",
        ),
    ),
    n=dict(
        attr_name=None,
        nc_name="observation_number",
        old_nc_name=["n", "Eddy_tsp"],
        nc_type="uint16",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Eddy temporal index in a trajectory",
            comment="Observation sequence number, days starting at the eddy first detection",
        ),
    ),
    contour_lon_e=dict(
        attr_name=None,
        nc_name="effective_contour_longitude",
        old_nc_name=["contour_lon_e"],
        nc_type="f4",
        filters=[zarr.Delta("i2")],
        output_type="i2",
        scale_factor=0.01,
        add_offset=180.0,
        nc_dims=("obs", "NbSample"),
        nc_attr=dict(
            long_name="Effective Contour Longitudes",
            comment="Longitudes of the effective contour",
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
        scale_factor=0.01,
        nc_dims=("obs", "NbSample"),
        nc_attr=dict(
            long_name="Effective Contour Latitudes",
            comment="Latitudes of effective contour",
            units="degrees_east",
            axis="X",
        ),
    ),
    num_point_e=dict(
        attr_name=None,
        nc_name="num_point_e",
        nc_type="u2",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="number of point for effective contour",
            units="ordinal",
            description="Number of point for effective contour, if greater than NbSample, there is a resampling",
        ),
    ),
    contour_lon_s=dict(
        attr_name=None,
        nc_name="speed_contour_longitude",
        old_nc_name=["contour_lon_s"],
        nc_type="f4",
        filters=[zarr.Delta("i2")],
        output_type="i2",
        scale_factor=0.01,
        add_offset=180.0,
        nc_dims=("obs", "NbSample"),
        nc_attr=dict(
            long_name="Speed Contour Longitudes",
            comment="Longitudes of speed contour",
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
        scale_factor=0.01,
        nc_dims=("obs", "NbSample"),
        nc_attr=dict(
            long_name="Speed Contour Latitudes",
            comment="Latitudes of speed contour",
            units="degrees_east",
            axis="X",
        ),
    ),
    num_point_s=dict(
        attr_name=None,
        nc_name="num_point_s",
        nc_type="u2",
        nc_dims=("obs",),
        nc_attr=dict(
            longname="number of point for speed contour",
            units="ordinal",
            description="Number of point for speed contour, if greater than NbSample, there is a resampling",
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
            comment="Error criterion between the effective contour and its fit with the circle of same effective radius",
            long_name="Effective Contour Error",
        ),
    ),
    score=dict(
        attr_name=None,
        nc_name="score",
        nc_type="f2",
        output_type="u1",
        scale_factor=0.4,
        nc_dims=("obs",),
        nc_attr=dict(units="%", comment="score", long_name="Score"),
    ),
    index_other=dict(
        attr_name=None,
        nc_name="index_other",
        nc_type="u4",
        nc_dims=("obs",),
        nc_attr=dict(
            # units="ordinal",
            comment="Corresponding index in the other dataset in score computation",
            long_name="Index in the other dataset",
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
            comment="Error criterion between the speed contour and its fit with the circle of same speed radius",
            long_name="Speed Contour Error",
        ),
    ),
    height_max_speed_contour=dict(
        attr_name=None,
        nc_name="speed_contour_height",
        old_nc_name=["height_max_speed_contour"],
        nc_type="f4",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Speed Contour Height",
            comment="SSH filtered height for speed contour",
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
            long_name="Effective Contour Height",
            comment="SSH filtered height for effective contour",
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
            long_name="Inner Contour Height",
            comment="SSH filtered height for the smallest detected contour",
            units="m",
        ),
    ),
    chl=dict(
        attr_name=None,
        nc_name="chl",
        old_nc_name=["Chl"],
        nc_type="f4",
        nc_dims=("obs",),
        nc_attr=dict(long_name="Log base 10 chlorophyll", units="Log(Chl/[mg/m^3])"),
    ),
    dchl=dict(
        attr_name=None,
        nc_name="dchl",
        old_nc_name=["dChl"],
        nc_type="f4",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Log base 10 chlorophyll anomaly (Chl minus Chl_bg)",
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
            long_name="Log base 10 background chlorophyll",
            units="Log(Chl/[mg/m^3])",
        ),
    ),
    year=dict(
        attr_name=None,
        nc_name="year",
        old_nc_name=["Year"],
        nc_type="u2",
        nc_dims=("obs",),
        nc_attr=dict(long_name="Year", units="year"),
    ),
    month=dict(
        attr_name=None,
        nc_name="month",
        old_nc_name=["Month"],
        nc_type="u1",
        nc_dims=("obs",),
        nc_attr=dict(long_name="Month", units="month"),
    ),
    day=dict(
        attr_name=None,
        nc_name="day",
        old_nc_name=["Day"],
        nc_type="u1",
        nc_dims=("obs",),
        nc_attr=dict(long_name="Day", units="day"),
    ),
    nb_contour_selected=dict(
        attr_name=None,
        nc_name="num_contours",
        old_nc_name=["nb_contour_selected"],
        nc_type="u2",
        nc_dims=("obs",),
        nc_attr=dict(
            long_name="Number of contours",
            comment="Number of contours selected for this eddy",
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
