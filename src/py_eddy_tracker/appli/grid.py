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
from datetime import datetime
from .. import EddyParser
from ..dataset.grid import RegularGridDataset, UnRegularGridDataset


def filtering_parser():
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
    args = filtering_parser().parse_args()

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


def eddy_id(args=None):
    parser = EddyParser("Eddy Identification")
    parser.add_argument("filename")
    parser.add_argument("datetime")
    parser.add_argument("h")
    parser.add_argument("u", help="If it s None, it will be deduce from h")
    parser.add_argument("v", help="If it s None, it will be deduce from h")
    parser.add_argument("longitude")
    parser.add_argument("latitude")
    parser.add_argument("path_out")
    help = "Wavelength for mesoscale filter in km"
    parser.add_argument("--cut_wavelength", default=500, type=float, help=help)
    parser.add_argument("--filter_order", default=3, type=int)
    help = "Step between 2 isoline in m"
    parser.add_argument("--isoline_step", default=0.002, type=float, help=help)
    help = "Error max accepted to fit circle in percent"
    parser.add_argument("--fit_errmax", default=55, type=float, help=help)
    parser.add_argument("--height_unit", default=None, help="Force height unit")
    parser.add_argument("--speed_unit", default=None, help="Force speed unit")
    parser.add_argument("--unregular", action="store_true", help="if grid is unregular")
    help = "Output will be wrote in zarr"
    parser.add_argument("--zarr", action="store_true", help=help)
    args = parser.parse_args(args) if args else parser.parse_args()

    date = datetime.strptime(args.datetime, "%Y%m%d")
    kwargs = dict(
        step=args.isoline_step,
        shape_error=args.fit_errmax,
        pixel_limit=(5, 2000),
        force_height_unit=args.height_unit,
        force_speed_unit=args.speed_unit,
    )
    a, c = identification(
        args.filename,
        args.longitude,
        args.latitude,
        date,
        args.h,
        args.u,
        args.v,
        unregular=args.unregular,
        cut_wavelength=args.cut_wavelength,
        filter_order=args.filter_order,
        **kwargs
    )
    out_name = date.strftime("%(path)s/%(sign_type)s_%Y%m%d.nc")
    a.write_file(path=args.path_out, filename=out_name, zarr_flag=args.zarr)
    c.write_file(path=args.path_out, filename=out_name, zarr_flag=args.zarr)


def identification(
    filename,
    lon,
    lat,
    date,
    h,
    u="None",
    v="None",
    unregular=False,
    cut_wavelength=500,
    filter_order=1,
    **kwargs
):
    grid_class = UnRegularGridDataset if unregular else RegularGridDataset
    grid = grid_class(filename, lon, lat)
    if u == "None" and v == "None":
        grid.add_uv(h)
        u, v = "u", "v"
    if cut_wavelength != 0:
        grid.bessel_high_filter(h, cut_wavelength, order=filter_order)
    return grid.eddy_identification(h, u, v, date, **kwargs)
