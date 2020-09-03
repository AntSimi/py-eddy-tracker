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

from numpy import arange, empty
from matplotlib import pyplot
from matplotlib.collections import LineCollection
from datetime import datetime
from ..poly import create_vertice
from ..generic import flatten_line_matrix
from .. import EddyParser
from ..observations.tracking import TrackEddiesObservations


class Anim:
    def __init__(self, eddy, intern=False, sleep_event=0.1, **kwargs):
        self.eddy = eddy
        x_name, y_name = eddy.intern(intern)
        self.t, self.x, self.y = eddy.time, eddy[x_name], eddy[y_name]
        self.pause = False
        self.period = self.eddy.period
        self.sleep_event = sleep_event
        self.setup(**kwargs)

    def setup(self, cmap="jet", nb_step=25, figsize=(8, 6)):
        cmap = pyplot.get_cmap(cmap)
        self.colors = cmap(arange(nb_step + 1) / nb_step)
        self.nb_step = nb_step

        x_min, x_max = self.x.min(), self.x.max()
        d_x = x_max - x_min
        x_min -= 0.05 * d_x
        x_max += 0.05 * d_x
        y_min, y_max = self.y.min(), self.y.max()
        d_y = y_max - y_min
        y_min -= 0.05 * d_y
        y_max += 0.05 * d_y

        # plot
        self.fig = pyplot.figure(figsize=figsize)
        t0, t1 = self.period
        self.fig.suptitle(f'{t0} -> {t1}')
        self.ax = self.fig.add_axes((0.05, 0.05, 0.9, 0.9))
        self.ax.set_xlim(x_min, x_max), self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect("equal")
        self.ax.grid()
        # init mappable
        self.txt = self.ax.text(x_min + 0.05 * d_x, y_min + 0.05 * d_y, "", zorder=10)
        self.contour = LineCollection([], zorder=1)
        self.ax.add_collection(self.contour)

        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect("key_press_event", self.keyboard)

    def show(self, infinity_loop=False):
        pyplot.show(block=False)
        # save background for future bliting
        self.fig.canvas.draw()
        self.bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        loop = True
        t0, t1 = self.period
        while loop:
            self.segs = list()
            self.now = t0
            while True:
                dt = self.sleep_event
                if not self.pause:
                    d0 = datetime.now()
                    self.next()
                    dt_draw = (datetime.now() - d0).total_seconds()
                    dt = self.sleep_event - dt_draw
                    if dt < 0:
                        # self.sleep_event = dt_draw * 1.01
                        dt = 1e-10
                self.fig.canvas.start_event_loop(dt)

                if self.now > t1:
                    break
            if infinity_loop:
                self.fig.canvas.start_event_loop(0.5)
            else:
                loop = False

    def next(self):
        self.now += 1
        return self.draw_contour()

    def prev(self):
        self.now -= 1
        return self.draw_contour()

    def draw_contour(self):
        t0, t1 = self.period
        # select contour for this time step
        m = self.t == self.now
        self.ax.figure.canvas.restore_region(self.bg_cache)
        if m.sum():
            self.segs.append(
                create_vertice(
                    flatten_line_matrix(self.x[m]), flatten_line_matrix(self.y[m])
                )
            )
        else:
            self.segs.append(empty((0, 2)))
        self.contour.set_paths(self.segs)
        self.contour.set_color(self.colors[-len(self.segs) :])
        self.contour.set_lw(arange(len(self.segs)) / len(self.segs) * 2.5)
        self.txt.set_text(f"{self.now} - {1/self.sleep_event:.0f} frame/s")
        self.ax.draw_artist(self.contour)
        self.ax.draw_artist(self.txt)
        # Remove first segment to keep only T contour
        if len(self.segs) > self.nb_step:
            self.segs.pop(0)
        # paint updated artist
        self.ax.figure.canvas.blit(self.ax.bbox)

    def keyboard(self, event):
        if event.key == "escape":
            exit()
        elif event.key == " ":
            self.pause = not self.pause
        elif event.key == "+":
            self.sleep_event *= 0.9
        elif event.key == "-":
            self.sleep_event *= 1.1
        elif event.key == "right" and self.pause:
            self.next()
        elif event.key == "left" and self.pause:
            self.segs.pop(-1)
            self.segs.pop(-1)
            self.prev()


def anim():
    parser = EddyParser(
        """Anim eddy, keyboard shortcut : Escape => exit, SpaceBar => pause,
        left arrow => t - 1, right arrow => t + 1, + => speed increase of 10 %, - => speed decrease of 10 %"""
    )
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
    variables = ["time", "track"]
    variables.extend(TrackEddiesObservations.intern(args.intern, public_label=True))

    atlas = TrackEddiesObservations.load_file(args.filename, include_vars=variables)
    eddy = atlas.extract_ids([args.id])
    a = Anim(
        eddy,
        intern=args.intern,
        sleep_event=args.time_sleep,
        cmap=args.cmap,
        nb_step=args.keep_step,
    )
    a.show(infinity_loop=args.infinity_loop)
