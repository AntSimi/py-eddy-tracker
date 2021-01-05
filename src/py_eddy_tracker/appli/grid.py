# -*- coding: utf-8 -*-
"""
All entry point to manipulate grid
"""
from argparse import Action
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


class DictAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        indexs = None
        if len(values):
            indexs = dict()
            for value in values:
                k, v = value.split("=")
                indexs[k] = int(v)
        setattr(namespace, self.dest, indexs)


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
    help = (
        "Wavelength for mesoscale filter in km to remove low scale if 2 args is given first one will be"
        "used to remove high scale and second value to remove low scale"
    )
    parser.add_argument(
        "--cut_wavelength", default=[500], type=float, help=help, nargs="+"
    )
    parser.add_argument("--filter_order", default=3, type=int)
    help = "Step between 2 isoline in m"
    parser.add_argument("--isoline_step", default=0.002, type=float, help=help)
    help = "Error max accepted to fit circle in percent"
    parser.add_argument("--fit_errmax", default=55, type=float, help=help)
    parser.add_argument(
        "--lat_max", default=85, type=float, help="Maximal latitude filtered"
    )
    parser.add_argument("--height_unit", default=None, help="Force height unit")
    parser.add_argument("--speed_unit", default=None, help="Force speed unit")
    parser.add_argument("--unregular", action="store_true", help="if grid is unregular")
    parser.add_argument(
        "--sampling",
        default=50,
        type=int,
        help="Array size used to build contour, first and last point will be the same",
    )
    parser.add_argument(
        "--sampling_method",
        default="visvalingam",
        type=str,
        choices=("visvalingam", "uniform"),
        help="Method to resample contour",
    )
    help = "Output will be wrote in zarr"
    parser.add_argument("--zarr", action="store_true", help=help)
    help = "Indexs to select grid : --indexs time=2, will select third step along time dimensions"
    parser.add_argument(
        "--indexs",
        nargs="*",
        help=help,
        action=DictAction,
    )
    args = parser.parse_args(args) if args else parser.parse_args()

    cut_wavelength = args.cut_wavelength
    nb_cw = len(cut_wavelength)
    if nb_cw > 2 or nb_cw == 0:
        raise Exception("You must specify 1 or 2 values for cut wavelength.")
    elif nb_cw == 1:
        cut_wavelength = [0, *cut_wavelength]
    inf_bnds, upper_bnds = cut_wavelength

    date = datetime.strptime(args.datetime, "%Y%m%d")
    kwargs = dict(
        step=args.isoline_step,
        shape_error=args.fit_errmax,
        pixel_limit=(5, 2000),
        force_height_unit=args.height_unit,
        force_speed_unit=args.speed_unit,
        nb_step_to_be_mle=0,
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
        cut_wavelength=upper_bnds,
        cut_highwavelength=inf_bnds,
        lat_max=args.lat_max,
        filter_order=args.filter_order,
        indexs=args.indexs,
        sampling=args.sampling,
        sampling_method=args.sampling_method,
        **kwargs,
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
    cut_highwavelength=0,
    lat_max=85,
    filter_order=1,
    indexs=None,
    **kwargs
):
    grid_class = UnRegularGridDataset if unregular else RegularGridDataset
    grid = grid_class(filename, lon, lat, indexs=indexs)
    if u == "None" and v == "None":
        grid.add_uv(h)
        u, v = "u", "v"
    kw_filter = dict(order=filter_order, lat_max=lat_max)
    if cut_highwavelength != 0:
        grid.bessel_low_filter(h, cut_highwavelength, **kw_filter)
    if cut_wavelength != 0:
        grid.bessel_high_filter(h, cut_wavelength, **kw_filter)
    return grid.eddy_identification(h, u, v, date, **kwargs)
