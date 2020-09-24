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
from netCDF4 import Dataset
from .. import EddyParser
from ..observations.tracking import TrackEddiesObservations
from ..observations.observation import EddiesObservations


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
    args = parser.parse_args()
    vars = [
        "amplitude",
        "speed_radius",
        "speed_area",
        "effective_radius",
        "effective_area",
        "time",
        "latitude",
    ]
    for filename in args.observations:
        with Dataset(filename) as h:
            track = 'track' in h.variables
        print(f"-- {filename} -- ")
        if track:
            vars_ = vars.copy()
            vars_.extend(('track', 'observation_number', 'observation_flag'))
            e = TrackEddiesObservations.load_file(filename, include_vars=vars_)
        else:
            e = EddiesObservations.load_file(filename, include_vars=vars)
        print(e)
