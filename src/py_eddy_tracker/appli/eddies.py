# -*- coding: utf-8 -*-
"""
Applications on detection and tracking files
"""
import argparse

from netCDF4 import Dataset

from .. import EddyParser
from ..observations.observation import EddiesObservations
from ..observations.tracking import TrackEddiesObservations


def eddies_add_circle():
    parser = EddyParser("Add or replace contour with radius parameter")
    parser.add_argument("filename", help="all file to merge")
    parser.add_argument("out", help="output file")
    args = parser.parse_args()
    obs = EddiesObservations.load_file(args.filename)
    if obs.track_array_variables == 0:
        obs.track_array_variables = 50
        obs = obs.add_fields(
            array_fields=(
                "contour_lon_e",
                "contour_lat_e",
                "contour_lon_s",
                "contour_lat_s",
            )
        )
    obs.circle_contour()
    obs.write_file(filename=args.out)


def merge_eddies():
    parser = EddyParser("Merge eddies")
    parser.add_argument("filename", nargs="+", help="all file to merge")
    parser.add_argument("out", help="output file")
    parser.add_argument(
        "--add_rotation_variable", help="add rotation variables", action="store_true"
    )
    parser.add_argument(
        "--include_var", nargs="+", type=str, help="use only listed variable"
    )
    args = parser.parse_args()

    if args.include_var is None:
        with Dataset(args.filename[0]) as h:
            args.include_var = h.variables.keys()

    obs = TrackEddiesObservations.load_file(
        args.filename[0], raw_data=True, include_vars=args.include_var
    )
    if args.add_rotation_variable:
        obs = obs.add_rotation_type()
    for filename in args.filename[1:]:
        other = TrackEddiesObservations.load_file(
            filename, raw_data=True, include_vars=args.include_var
        )
        if args.add_rotation_variable:
            other = other.add_rotation_type()
        obs = obs.merge(other)
    obs.write_file(filename=args.out)


def get_frequency_grid():
    parser = EddyParser("Compute eddy frequency")
    parser.add_argument("observations", help="Input observations to compute frequency")
    parser.add_argument("out", help="Grid output file")
    parser.add_argument(
        "--intern",
        help="Use speed contour instead of effective contour",
        action="store_true",
    )
    parser.add_argument(
        "--xrange", nargs="+", type=float, help="Horizontal range : START,STOP,STEP"
    )
    parser.add_argument(
        "--yrange", nargs="+", type=float, help="Vertical range : START,STOP,STEP"
    )
    args = parser.parse_args()

    if (args.xrange is None or len(args.xrange) not in (3,)) or (
        args.yrange is None or len(args.yrange) not in (3,)
    ):
        raise Exception("Use START/STOP/STEP for --xrange and --yrange")

    var_to_load = ["longitude"]
    var_to_load.extend(EddiesObservations.intern(args.intern, public_label=True))
    e = EddiesObservations.load_file(args.observations, include_vars=var_to_load)

    bins = args.xrange, args.yrange
    g = e.grid_count(bins, intern=args.intern)
    g.write(args.out)


def display_infos():
    parser = EddyParser("Display General inforamtion")
    parser.add_argument(
        "observations", nargs="+", help="Input observations to compute frequency"
    )
    parser.add_argument("--vars", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument(
        "--area",
        nargs=4,
        type=float,
        metavar=("llcrnrlon", "llcrnrlat", "urcrnrlon", "urcrnrlat"),
        help="Bounding box",
    )
    args = parser.parse_args()
    if args.vars:
        vars = args.vars
    else:
        vars = [
            "amplitude",
            "speed_radius",
            "speed_area",
            "effective_radius",
            "effective_area",
            "time",
            "latitude",
            "longitude",
        ]
    filenames = args.observations
    filenames.sort()
    for filename in filenames:
        with Dataset(filename) as h:
            track = "track" in h.variables
        print(f"-- {filename} -- ")
        if track:
            vars_ = vars.copy()
            vars_.extend(("track", "observation_number", "observation_flag"))
            e = TrackEddiesObservations.load_file(filename, include_vars=vars_)
        else:
            e = EddiesObservations.load_file(filename, include_vars=vars)
        if args.area is not None:
            area = dict(
                llcrnrlon=args.area[0],
                llcrnrlat=args.area[1],
                urcrnrlon=args.area[2],
                urcrnrlat=args.area[3],
            )
            e = e.extract_with_area(area)
        print(e)
