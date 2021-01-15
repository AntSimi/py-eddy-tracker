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
    parser = EddyParser("Separate path for a same group")
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
        e, e.split_network(intern=args.intern, window=args.window)
    )
    n.write_file(filename=args.out)


def subsample_network():
    parser = EddyParser("Sub sample")
    parser.add_argument("input", help="input network file")
    parser.add_argument("out", help="output file")
