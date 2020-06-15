# -*- coding: utf-8 -*-
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
