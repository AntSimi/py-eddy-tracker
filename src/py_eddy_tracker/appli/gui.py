# -*- coding: utf-8 -*-
"""
Entry point of graphic user interface
"""

from datetime import datetime

from matplotlib import pyplot
from matplotlib.collections import LineCollection
from numpy import arange, empty, where

from .. import EddyParser
from ..generic import flatten_line_matrix
from ..gui import GUI
from ..observations.tracking import TrackEddiesObservations
from ..poly import create_vertice


class Anim:
    def __init__(
        self, eddy, intern=False, sleep_event=0.1, graphic_information=False, **kwargs
    ):
        self.eddy = eddy
        x_name, y_name = eddy.intern(intern)
        self.t, self.x, self.y = eddy.time, eddy[x_name], eddy[y_name]
        self.x_core, self.y_core, self.track = eddy["lon"], eddy["lat"], eddy["track"]
        self.graphic_informations = graphic_information
        self.pause = False
        self.period = self.eddy.period
        self.sleep_event = sleep_event
        self.mappables = list()
        self.setup(**kwargs)

    def setup(self, cmap="jet", nb_step=25, figsize=(8, 6), **kwargs):
        cmap = pyplot.get_cmap(cmap)
        self.colors = cmap(arange(nb_step + 1) / nb_step)
        self.nb_step = nb_step

        x_min, x_max = self.x_core.min() - 2, self.x_core.max() + 2
        d_x = x_max - x_min
        y_min, y_max = self.y_core.min() - 2, self.y_core.max() + 2
        d_y = y_max - y_min
        # plot
        self.fig = pyplot.figure(figsize=figsize, **kwargs)
        t0, t1 = self.period
        self.fig.suptitle(f"{t0} -> {t1}")
        self.ax = self.fig.add_axes((0.05, 0.05, 0.9, 0.9))
        self.ax.set_xlim(x_min, x_max), self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect("equal")
        self.ax.grid()
        # init mappable
        self.txt = self.ax.text(x_min + 0.05 * d_x, y_min + 0.05 * d_y, "", zorder=10)
        self.segs = list()
        self.contour = LineCollection([], zorder=1)
        self.ax.add_collection(self.contour)

        self.fig.canvas.draw()
        self.fig.canvas.mpl_connect("key_press_event", self.keyboard)
        self.fig.canvas.mpl_connect("resize_event", self.reset_bliting)

    def reset_bliting(self, event):
        self.contour.set_visible(False)
        self.txt.set_visible(False)
        for m in self.mappables:
            m.set_visible(False)
        self.fig.canvas.draw()
        self.bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.contour.set_visible(True)
        self.txt.set_visible(True)
        for m in self.mappables:
            m.set_visible(True)

    def show(self, infinity_loop=False):
        pyplot.show(block=False)
        # save background for future bliting
        self.fig.canvas.draw()
        self.bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        loop = True
        t0, t1 = self.period
        while loop:
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

    def func_animation(self, frame):
        while self.mappables:
            self.mappables.pop().remove()
        self.now = frame
        self.update()
        artists = [self.contour, self.txt]
        artists.extend(self.mappables)
        return artists

    def update(self):
        m = self.t == self.now
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
        txt = f"{self.now}"
        if self.graphic_informations:
            txt += f"- {1/self.sleep_event:.0f} frame/s"
        self.txt.set_text(txt)
        for i in where(m)[0]:
            mappable = self.ax.text(
                self.x_core[i], self.y_core[i], self.track[i], fontsize=8
            )
            self.mappables.append(mappable)
            self.ax.draw_artist(mappable)
        self.ax.draw_artist(self.contour)
        self.ax.draw_artist(self.txt)
        # Remove first segment to keep only T contour
        if len(self.segs) > self.nb_step:
            self.segs.pop(0)

    def draw_contour(self):
        # select contour for this time step
        while self.mappables:
            self.mappables.pop().remove()
        self.ax.figure.canvas.restore_region(self.bg_cache)
        self.update()
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
    parser.add_argument("id", help="Track id to anim", type=int, nargs="*")
    parser.add_argument(
        "--intern",
        action="store_true",
        help="display intern contour inplace of outter contour",
    )
    parser.add_argument(
        "--keep_step", default=25, help="number maximal of step displayed", type=int
    )
    parser.add_argument("--cmap", help="matplotlib colormap used")
    parser.add_argument("--all", help="All eddies will be drawed", action="store_true")
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
    variables = ["time", "track", "longitude", "latitude"]
    variables.extend(TrackEddiesObservations.intern(args.intern, public_label=True))

    eddies = TrackEddiesObservations.load_file(args.filename, include_vars=variables)
    if not args.all:
        if len(args.id) == 0:
            raise Exception(
                "You need to specify id to display or ask explicity all with --all option"
            )
        eddies = eddies.extract_ids(args.id)
    a = Anim(
        eddies,
        intern=args.intern,
        sleep_event=args.time_sleep,
        cmap=args.cmap,
        nb_step=args.keep_step,
    )
    a.show(infinity_loop=args.infinity_loop)


def gui_parser():
    parser = EddyParser("Eddy atlas GUI")
    parser.add_argument("atlas", nargs="+")
    parser.add_argument("--med", action="store_true")
    return parser.parse_args()


def guieddy():
    args = gui_parser()
    atlas = {
        dataset: TrackEddiesObservations.load_file(dataset) for dataset in args.atlas
    }
    g = GUI(**atlas)
    if args.med:
        g.med()
    g.show()
