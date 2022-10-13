# -*- coding: utf-8 -*-
"""
Entry point to create and manipulate observations network
"""

import logging

from numpy import in1d, zeros

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

    parser.add_argument(
        "--min-overlap",
        "-p",
        type=float,
        help="minimum overlap area to associate observations",
        default=0.2,
    )
    parser.add_argument(
        "--minimal-area",
        action="store_true",
        help="If True, use intersection/little polygon, else intersection/union",
    )
    parser.add_argument(
        "--hybrid-area",
        action="store_true",
        help="If True, use minimal-area method if overlap is under min overlap, else intersection/union",
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
    group = n.group_observations(
        min_overlap=args.min_overlap, minimal_area=args.minimal_area, hybrid_area=args.hybrid_area
    )
    n.build_dataset(group).write_file(filename=args.out)


def divide_network():
    parser = EddyParser("Separate path for a same group (network)")
    parser.add_argument("input", help="input network file")
    parser.add_argument("out", help="output file")
    parser.contour_intern_arg()
    parser.add_argument(
        "--window", "-w", type=int, help="Half time window to search eddy", default=1
    )
    parser.add_argument(
        "--min-overlap",
        "-p",
        type=float,
        help="minimum overlap area to associate observations",
        default=0.2,
    )
    parser.add_argument(
        "--minimal-area",
        action="store_true",
        help="If True, use intersection/little polygon, else intersection/union",
    )
    parser.add_argument(
        "--hybrid-area",
        action="store_true",
        help="If True, use minimal-area method if overlap is under min overlap, else intersection/union",
    )
    args = parser.parse_args()
    contour_name = TrackEddiesObservations.intern(args.intern, public_label=True)
    e = TrackEddiesObservations.load_file(
        args.input,
        include_vars=("time", "track", "latitude", "longitude", *contour_name),
    )
    n = NetworkObservations.from_split_network(
        TrackEddiesObservations.load_file(args.input, raw_data=True),
        e.split_network(
            intern=args.intern,
            window=args.window,
            min_overlap=args.min_overlap,
            minimal_area=args.minimal_area,
            hybrid_area=args.hybrid_area
        ),
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
        "--remove_trash",
        action="store_true",
        help="Remove trash (network id == 0)",
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


def quick_compare():
    parser = EddyParser(
        """Tool to have a quick comparison between several network:
        - N : network
        - S : segment
        - Obs : observations
        """
    )
    parser.add_argument("ref", help="Identification file of reference")
    parser.add_argument("others", nargs="+", help="Identifications files to compare")
    parser.add_argument(
        "--path_out", default=None, help="Save each group in separate file"
    )
    args = parser.parse_args()

    kw = dict(
        include_vars=[
            "longitude",
            "latitude",
            "time",
            "track",
            "segment",
            "next_obs",
            "previous_obs",
        ]
    )

    if args.path_out is not None:
        kw = dict()

    ref = NetworkObservations.load_file(args.ref, **kw)
    print(
        f"[ref] {args.ref} -> {ref.nb_network} network / {ref.nb_segment} segment / {len(ref)} obs "
        f"-> {ref.network_size(0)} trash obs, "
        f"{len(ref.merging_event())} merging, {len(ref.splitting_event())} spliting"
    )
    others = {
        other: NetworkObservations.load_file(other, **kw) for other in args.others
    }

    # if args.path_out is not None:
    #     groups_ref, groups_other = run_compare(ref, others, **kwargs)
    #     if not exists(args.path_out):
    #         mkdir(args.path_out)
    #     for i, other_ in enumerate(args.others):
    #         dirname_ = f"{args.path_out}/{other_.replace('/', '_')}/"
    #         if not exists(dirname_):
    #             mkdir(dirname_)
    #         for k, v in groups_other[other_].items():
    #             basename_ = f"other_{k}.nc"
    #             others[other_].index(v).write_file(filename=f"{dirname_}/{basename_}")
    #         for k, v in groups_ref[other_].items():
    #             basename_ = f"ref_{k}.nc"
    #             ref.index(v).write_file(filename=f"{dirname_}/{basename_}")
    #     return
    display_compare(ref, others)


def run_compare(ref, others):
    outs = dict()
    for i, (k, other) in enumerate(others.items()):
        out = dict()
        print(
            f"[{i}]   {k} -> {other.nb_network} network / {other.nb_segment} segment / {len(other)} obs "
            f"-> {other.network_size(0)} trash obs, "
            f"{len(other.merging_event())} merging, {len(other.splitting_event())} spliting"
        )
        ref_id, other_id = ref.identify_in(other, size_min=2)
        m = other_id != -1
        ref_id, other_id = ref_id[m], other_id[m]
        out["same N(N)"] = m.sum()
        out["same N(Obs)"] = ref.network_size(ref_id).sum()

        # For network which have same obs
        ref_, other_ = ref.networks(ref_id), other.networks(other_id)
        ref_segu, other_segu = ref_.identify_in(other_, segment=True)
        m = other_segu == -1
        ref_track_no_match, _ = ref_.unique_segment_to_id(ref_segu[m])
        ref_segu, other_segu = ref_segu[~m], other_segu[~m]
        m = ~in1d(ref_id, ref_track_no_match)
        out["same NS(N)"] = m.sum()
        out["same NS(Obs)"] = ref.network_size(ref_id[m]).sum()

        # Check merge/split
        def follow_obs(d, i_follow):
            m = i_follow != -1
            i_follow = i_follow[m]
            t, x, y = (
                zeros(m.size, d.time.dtype),
                zeros(m.size, d.longitude.dtype),
                zeros(m.size, d.latitude.dtype),
            )
            t[m], x[m], y[m] = (
                d.time[i_follow],
                d.longitude[i_follow],
                d.latitude[i_follow],
            )
            return t, x, y

        def next_obs(d, i_seg):
            last_i = d.index_segment_track[1][i_seg] - 1
            return follow_obs(d, d.next_obs[last_i])

        def previous_obs(d, i_seg):
            first_i = d.index_segment_track[0][i_seg]
            return follow_obs(d, d.previous_obs[first_i])

        tref, xref, yref = next_obs(ref_, ref_segu)
        tother, xother, yother = next_obs(other_, other_segu)

        m = (tref == tother) & (xref == xother) & (yref == yother)
        print(m.sum(), m.size, ref_segu.size, ref_track_no_match.size)

        tref, xref, yref = previous_obs(ref_, ref_segu)
        tother, xother, yother = previous_obs(other_, other_segu)

        m = (tref == tother) & (xref == xother) & (yref == yother)
        print(m.sum(), m.size, ref_segu.size, ref_track_no_match.size)

        ref_segu, other_segu = ref.identify_in(other, segment=True)
        m = other_segu != -1
        out["same S(S)"] = m.sum()
        out["same S(Obs)"] = ref.segment_size()[ref_segu[m]].sum()

        outs[k] = out
    return outs


def display_compare(ref, others):
    def display(value, ref=None):
        if ref:
            outs = [f"{v/ref[k] * 100:.1f}% ({v})" for k, v in value.items()]
        else:
            outs = value
        return "".join([f"{v:^18}" for v in outs])

    datas = run_compare(ref, others)
    ref_ = {
        "same N(N)": ref.nb_network,
        "same N(Obs)": len(ref),
        "same NS(N)": ref.nb_network,
        "same NS(Obs)": len(ref),
        "same S(S)": ref.nb_segment,
        "same S(Obs)": len(ref),
    }
    print("     ", display(ref_.keys()))
    for i, (_, v) in enumerate(datas.items()):
        print(f"[{i:2}] ", display(v, ref=ref_))
