# -*- coding: utf-8 -*-
"""
Entry point to create and manipulate observations network
"""

import logging

from .. import EddyParser
from ..observations.network import Network, NetworkObservations
from ..observations.tracking import TrackEddiesObservations

logger = logging.getLogger("pet")


def build_network():
    parser = EddyParser("Merge eddies")
    parser.add_argument(
        "identification_regex", help="Give an expression which will use with glob"
    )
    parser.add_argument("out", help="output file")
    parser.add_argument(
        "--window", "-w", type=int, help="Half time window to search eddy", default=1
    )
    parser.contour_intern_arg()

    parser.memory_arg()
    args = parser.parse_args()

    n = Network(
        args.identification_regex,
        window=args.window,
        intern=args.intern,
        memory=args.memory,
    )
    group = n.group_observations(minimal_area=True)
    n.build_dataset(group).write_file(filename=args.out)


def divide_network():
    parser = EddyParser("Separate path for a same group (network)")
    parser.add_argument("input", help="input network file")
    parser.add_argument("out", help="output file")
    parser.contour_intern_arg()
    parser.add_argument(
        "--window", "-w", type=int, help="Half time window to search eddy", default=1
    )
    args = parser.parse_args()
    contour_name = TrackEddiesObservations.intern(args.intern, public_label=True)
    e = TrackEddiesObservations.load_file(
        args.input,
        include_vars=("time", "track", "latitude", "longitude", *contour_name),
    )
    n = NetworkObservations.from_split_network(
        TrackEddiesObservations.load_file(args.input, raw_data=True),
        e.split_network(intern=args.intern, window=args.window),
    )
    n.write_file(filename=args.out)


def subset_network():
    parser = EddyParser("Subset network")
    parser.add_argument("input", help="input network file")
    parser.add_argument("out", help="output file")
    parser.add_argument(
        "-l",
        "--length",
        nargs=2,
        type=int,
        help="Nb of days that must be covered by the network, first minimum number of day and last maximum number of day,"
        "if value is negative, this bound won't be used",
    )
    parser.add_argument(
        "--remove_dead_end",
        nargs=2,
        type=int,
        help="Remove short dead end, first is for minimal obs number and second for minimal segment time to keep",
    )
    parser.add_argument(
        "--remove_trash", action="store_true", help="Remove trash (network id == 0)",
    )
    parser.add_argument(
        "-p",
        "--period",
        nargs=2,
        type=int,
        help="Start day and end day, if it's a negative value we will add to day min and add to day max,"
        "if 0 it is not used",
    )
    args = parser.parse_args()
    n = NetworkObservations.load_file(args.input, raw_data=True)
    if args.length is not None:
        n = n.longer_than(*args.length)
    if args.remove_dead_end is not None:
        n = n.remove_dead_end(*args.remove_dead_end)
    if args.period is not None:
        n = n.extract_with_period(args.period)
    n.write_file(filename=args.out)
