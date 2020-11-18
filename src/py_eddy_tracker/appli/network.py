# -*- coding: utf-8 -*-
"""
Entry point to create and manipulate observations network
"""

import logging

from netCDF4 import Dataset
from numpy import arange, empty, zeros
from Polygon import Polygon

from .. import EddyParser
from ..generic import build_index
from ..observations.network import Network
from ..observations.tracking import TrackEddiesObservations
from ..poly import create_vertice_from_2darray, polygon_overlap

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
        "--intern",
        action="store_true",
        help="Use intern contour instead of outter contour",
    )
    args = parser.parse_args()

    n = Network(args.identification_regex, window=args.window, intern=args.intern)
    group = n.group_observations(minimal_area=True)
    n.build_dataset(group).write_file(filename=args.out)


def divide_network():
    parser = EddyParser("Separate path for a same group")
    parser.add_argument("input", help="input network file")
    parser.add_argument("out", help="output file")
    parser.add_argument(
        "--intern",
        action="store_true",
        help="Use intern contour instead of outter contour",
    )
    parser.add_argument(
        "--window", "-w", type=int, help="Half time window to search eddy", default=1
    )
    args = parser.parse_args()
    contour_name = TrackEddiesObservations.intern(args.intern, public_label=True)
    e = TrackEddiesObservations.load_file(
        args.input, include_vars=("time", "track", *contour_name)
    )
    e.split_network(intern=args.intern, window=args.window)
    # split_network(args.input, args.out)


def split_network(input, output):
    """Divide each group in track"""
    sl = slice(None)
    with Dataset(input) as h:
        group = h.variables["track"][sl]
    track_s, track_e, track_ref = build_index(group)
    # nb = track_e - track_s
    # m = nb > 1500
    # print(group[track_s[m]])

    track_id = 12003
    sls = [slice(track_s[track_id - track_ref], track_e[track_id - track_ref], None)]
    for sl in sls:

        print(sl)
        with Dataset(input) as h:
            time = h.variables["time"][sl]
            group = h.variables["track"][sl]
            x = h.variables["effective_contour_longitude"][sl]
            y = h.variables["effective_contour_latitude"][sl]
        print(group[0])
        ids = empty(
            time.shape,
            dtype=[
                ("group", group.dtype),
                ("time", time.dtype),
                ("track", "u2"),
                ("previous_cost", "f4"),
                ("next_cost", "f4"),
                ("previous_observation", "i4"),
                ("next_observation", "i4"),
            ],
        )
        ids["group"] = group
        ids["time"] = time
        # To store id track
        ids["track"] = 0
        ids["previous_cost"] = 0
        ids["next_cost"] = 0
        ids["previous_observation"] = -1
        ids["next_observation"] = -1
        # Cost with previous
        track_start, track_end, track_ref = build_index(group)
        for i0, i1 in zip(track_start, track_end):
            if (i1 - i0) == 0 or group[i0] == Network.NOGROUP:
                continue
            sl_group = slice(i0, i1)
            set_tracks(
                x[sl_group],
                y[sl_group],
                time[sl_group],
                i0,
                ids["track"][sl_group],
                ids["previous_cost"][sl_group],
                ids["next_cost"][sl_group],
                ids["previous_observation"][sl_group],
                ids["next_observation"][sl_group],
                window=5,
            )

        new_i = ids.argsort(order=("group", "track", "time"))
        ids_sort = ids[new_i]
        # To be able to follow indices sorting
        reverse_sort = empty(new_i.shape[0], dtype="u4")
        reverse_sort[new_i] = arange(new_i.shape[0])
        # Redirect indices
        m = ids_sort["next_observation"] != -1
        ids_sort["next_observation"][m] = reverse_sort[ids_sort["next_observation"][m]]
        m = ids_sort["previous_observation"] != -1
        ids_sort["previous_observation"][m] = reverse_sort[
            ids_sort["previous_observation"][m]
        ]
        # print(ids_sort)
        display_network(
            x[new_i],
            y[new_i],
            ids_sort["track"],
            ids_sort["time"],
            ids_sort["next_cost"],
        )


def next_obs(
    i_current, next_cost, previous_cost, polygons, t, t_start, t_end, t_ref, window
):
    t_max = t_end.shape[0] - 1
    t_cur = t[i_current]
    t0, t1 = t_cur + 1 - t_ref, t_cur + window - t_ref
    if t0 > t_max:
        return -1
    t1 = min(t1, t_max)
    for t_step in range(t0, t1 + 1):
        i0, i1 = t_start[t_step], t_end[t_step]
        # No observation at the time step !
        if i0 == i1:
            continue
        sl = slice(i0, i1)
        # Intersection / union, to be able to separte in case of multiple inside
        c = polygon_overlap(polygons[i_current], polygons[sl])
        # We remove low overlap
        if (c > 0.1).sum() > 1:
            print(c)
        c[c < 0.1] = 0
        # We get index of maximal overlap
        i = c.argmax()
        c_i = c[i]
        # No overlap found
        if c_i == 0:
            continue
        target = i0 + i
        # Check if candidate is already used
        c_target = previous_cost[target]
        if (c_target != 0 and c_target < c_i) or c_target == 0:
            previous_cost[target] = c_i
        next_cost[i_current] = c_i
        return target
    return -1


def set_tracks(
    x,
    y,
    t,
    ref_index,
    track,
    previous_cost,
    next_cost,
    previous_observation,
    next_observation,
    window,
):
    # Will split one group in tracks
    t_start, t_end, t_ref = build_index(t)
    nb = x.shape[0]
    used = zeros(nb, dtype="bool")
    current_track = 1
    # build all polygon (need to check if wrap is needed)
    polygons = list()
    for i in range(nb):
        polygons.append(Polygon(create_vertice_from_2darray(x, y, i)))

    for i in range(nb):
        # If observation already in one track, we go to the next one
        if used[i]:
            continue
        build_track(
            i,
            current_track,
            used,
            track,
            previous_observation,
            next_observation,
            ref_index,
            next_cost,
            previous_cost,
            polygons,
            t,
            t_start,
            t_end,
            t_ref,
            window,
        )
        current_track += 1


def build_track(
    first_index,
    track_id,
    used,
    track,
    previous_observation,
    next_observation,
    ref_index,
    next_cost,
    previous_cost,
    *args,
):
    i_next = first_index
    while i_next != -1:
        # Flag
        used[i_next] = True
        # Assign id
        track[i_next] = track_id
        # Search next
        i_next_ = next_obs(i_next, next_cost, previous_cost, *args)
        if i_next_ == -1:
            break
        next_observation[i_next] = i_next_ + ref_index
        if not used[i_next_]:
            previous_observation[i_next_] = i_next + ref_index
        # Target was previously used
        if used[i_next_]:
            if next_cost[i_next] == previous_cost[i_next_]:
                m = track[i_next_:] == track[i_next_]
                track[i_next_:][m] = track_id
                previous_observation[i_next_] = i_next + ref_index
            i_next_ = -1
        i_next = i_next_


def display_network(x, y, tr, t, c):
    tr0, tr1, t_ref = build_index(tr)
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("jet")
    from ..generic import flatten_line_matrix

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, aspect="equal")
    ax.grid()
    ax_time = fig.add_subplot(122)
    ax_time.grid()
    i = 0
    for s, e in zip(tr0, tr1):
        if s == e:
            continue
        sl = slice(s, e)
        color = cmap((tr[s] - tr[tr0[0]]) / (tr[tr0[-1]] - tr[tr0[0]]))
        ax.plot(
            flatten_line_matrix(x[sl]),
            flatten_line_matrix(y[sl]),
            color=color,
            label=f"{tr[s]} - {e-s} obs from {t[s]} to {t[e-1]}",
        )
        i += 1
        ax_time.plot(
            t[sl],
            tr[s].repeat(e - s) + c[sl],
            color=color,
            label=f"{tr[s]} - {e-s} obs",
            lw=0.5,
        )
        ax_time.plot(t[sl], tr[s].repeat(e - s), color=color, lw=1, marker="+")
        ax_time.text(t[s], tr[s] + 0.15, f"{x[s].mean():.2f}, {y[s].mean():.2f}")
        ax_time.axvline(t[s], color=".75", lw=0.5, ls="--", zorder=-10)
        ax_time.text(
            t[e - 1], tr[e - 1] - 0.25, f"{x[e-1].mean():.2f}, {y[e-1].mean():.2f}"
        )
    ax.legend()
    ax_time.legend()
    plt.show()
