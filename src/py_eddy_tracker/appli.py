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

from py_eddy_tracker import EddyParser
from py_eddy_tracker.observations.tracking import TrackEddiesObservations
from netCDF4 import Dataset


def merge_eddies():
    parser = EddyParser('Merge eddies')
    parser.add_argument('filename', nargs='+', help='all file to merge')
    parser.add_argument('out', help='output file')
    parser.add_argument('--add_rotation_variable', help='add rotation variables', action='store_true')
    parser.add_argument('--include_var', nargs='+', type=str, help='use only listed variable')
    args = parser.parse_args()

    if args.include_var is None:
        with Dataset(args.filename[0]) as h:
            args.include_var = h.variables.keys()

    obs = TrackEddiesObservations.load_file(args.filename[0], raw_data=True, include_vars=args.include_var)
    if args.add_rotation_variable:
        obs = obs.add_rotation_type()
    for filename in args.filename[1:]:
        other = TrackEddiesObservations.load_file(filename, raw_data=True, include_vars=args.include_var)
        if args.add_rotation_variable:
            other = other.add_rotation_type()
        obs = obs.merge(other)
    obs.write_file(filename=args.out)
