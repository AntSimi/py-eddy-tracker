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
from .. import EddyParser
from ..dataset.grid import RegularGridDataset


def grid_parser():
    parser = EddyParser("Grid filtering")
    parser.add_argument("filename")
    parser.add_argument("grid")
    parser.add_argument("longitude")
    parser.add_argument("latitude")
    parser.add_argument("filename_out")
    parser.add_argument(
        "--cut_wavelength",
        default=500,
        type=float,
        help="Wavelength for mesoscale filter in km",
    )
    parser.add_argument("--filter_order", default=3, type=int)
    parser.add_argument("--low", action="store_true")
    parser.add_argument(
        "--extend",
        default=0,
        type=float,
        help="Keep pixel compute by filtering on mask",
    )
    return parser


def grid_filtering():
    args = grid_parser().parse_args()

    h = RegularGridDataset(args.filename, args.longitude, args.latitude)
    if args.low:
        h.bessel_low_filter(
            args.grid, args.cut_wavelength, order=args.filter_order, extend=args.extend
        )
    else:
        h.bessel_high_filter(
            args.grid, args.cut_wavelength, order=args.filter_order, extend=args.extend
        )
    h.write(args.filename_out)
