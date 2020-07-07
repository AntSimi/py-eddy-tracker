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

from numpy import arange
from matplotlib import pyplot
from netCDF4 import Dataset
from matplotlib.collections import LineCollection
from py_eddy_tracker.poly import create_vertice
from py_eddy_tracker import EddyParser
from py_eddy_tracker.observations.tracking import TrackEddiesObservations


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


def anim():
    parser = EddyParser("Merge eddies")
    parser.add_argument("filename", help="eddy atlas")
    parser.add_argument("id", help="Track id to anim", type=int)
    parser.add_argument(
        "--intern",
        action="store_true",
        help="display intern contour inplace of outter contour",
    )
    parser.add_argument(
        "--keep_step", default=25, help="number maximal of step displayed", type=int
    )
    parser.add_argument("--cmap", help="matplotlib colormap used")
    parser.add_argument(
        "--time_sleep",
        type=float,
        default=0.01,
        help="Sleeping time in second between 2 frame",
    )
    parser.add_argument(
        "--infinity_loop", action="store_true", help="Press Escape key to stop loop"
    )
    args = parser.parse_args()

    atlas = TrackEddiesObservations.load_file(args.filename)
    eddy = atlas.extract_ids([args.id])
    x_name, y_name = eddy.intern(args.intern)
    t0, t1 = eddy.period
    t, x, y = eddy.time, eddy[x_name], eddy[y_name]
    x_min, x_max = x.min(), x.max()
    d_x = x_max - x_min
    x_min -= 0.05 * d_x
    x_max += 0.05 * d_x
    y_min, y_max = y.min(), y.max()
    d_y = y_max - y_min
    y_min -= 0.05 * d_y
    y_max += 0.05 * d_y

    # General value
    cmap = pyplot.get_cmap(args.cmap)
    colors = cmap(arange(args.keep_step + 1) / args.keep_step)

    # plot
    fig = pyplot.figure()
    # manager = plt.get_current_fig_manager()
    # fig.window.showMaximized()
    ax = fig.add_axes((0.05, 0.05, 0.9, 0.9))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.grid()
    # # init mappable
    txt = ax.text(16.6, 36.8, "", zorder=10)
    c = LineCollection([], zorder=1)
    ax.add_collection(c)

    fig.canvas.draw()
    fig.canvas.mpl_connect("key_press_event", keyboard)
    pyplot.show(block=False)
    # save background for future bliting
    bg_cache = fig.canvas.copy_from_bbox(ax.bbox)
    loop = True
    while loop:
        segs = list()
        # display contour every day
        for t_ in range(t0, t1 + 1, 1):
            fig.canvas.restore_region(bg_cache)
            # select contour for this time step
            m = t == t_
            if m.sum():
                segs.append(create_vertice(x[m][0], y[m][0]))
                c.set_paths(segs)
                c.set_color(colors[-len(segs) :])
                txt.set_text(f"{t0} -> {t_} -> {t1}")
                ax.draw_artist(c)
                ax.draw_artist(txt)
                # Remove first segment to keep only T contour
                if len(segs) > args.keep_step:
                    segs.pop(0)
                # paint updated artist
                fig.canvas.blit(ax.bbox)
            fig.canvas.start_event_loop(args.time_sleep)
        if args.infinity_loop:
            fig.canvas.start_event_loop(0.5)
        else:
            loop = False


def keyboard(event):
    if event.key == "escape":
        exit()
