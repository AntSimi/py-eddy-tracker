# -*- coding: utf-8 -*-
"""
Entry point of graphic user interface
"""

import logging
from datetime import datetime, timedelta
from itertools import chain

from matplotlib import pyplot
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from numpy import arange, where

from .. import EddyParser
from ..gui import GUI
from ..observations.tracking import TrackEddiesObservations
from ..poly import create_vertice

logger = logging.getLogger("pet")


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
        self.field_color = None
        self.time_field = False
        self.setup(**kwargs)

    def setup(
        self,
        cmap="jet",
        lut=None,
        field_color="time",
        range_color=(None, None),
        nb_step=25,
        figsize=(8, 6),
        **kwargs,
    ):
        self.field_color = self.eddy[field_color].astype("f4")
        rg = range_color
        if rg[0] is None and rg[1] is None and field_color == "time":
            self.time_field = True
        else:
            rg = (
                self.field_color.min() if rg[0] is None else rg[0],
                self.field_color.max() if rg[1] is None else rg[1],
            )
            self.field_color = (self.field_color - rg[0]) / (rg[1] - rg[0])

        self.colors = pyplot.get_cmap(cmap, lut=lut)
        self.nb_step = nb_step

        x_min, x_max = self.x_core.min() - 2, self.x_core.max() + 2
        d_x = x_max - x_min
        y_min, y_max = self.y_core.min() - 2, self.y_core.max() + 2
        d_y = y_max - y_min
        # plot
        self.fig = pyplot.figure(figsize=figsize, **kwargs)
        t0, t1 = self.period
        self.fig.suptitle(f"{t0} -> {t1}")
        self.ax = self.fig.add_axes((0.05, 0.05, 0.9, 0.9), projection="full_axes")
        self.ax.set_xlim(x_min, x_max), self.ax.set_ylim(y_min, y_max)
        self.ax.set_aspect("equal")
        self.ax.grid()
        # init mappable
        self.txt = self.ax.text(x_min + 0.05 * d_x, y_min + 0.05 * d_y, "", zorder=10)
        self.segs = list()
        self.t_segs = list()
        self.c_segs = list()
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
                if dt == 0:
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
            segs = list()
            t = list()
            c = list()
            for i in where(m)[0]:
                segs.append(create_vertice(self.x[i], self.y[i]))
                c.append(self.field_color[i])
                t.append(self.now)
            self.segs.append(segs)
            self.c_segs.append(c)
            self.t_segs.append(t)
        self.contour.set_paths(chain(*self.segs))
        if self.time_field:
            self.contour.set_color(
                self.colors(
                    [
                        (self.nb_step - self.now + i) / self.nb_step
                        for i in chain(*self.c_segs)
                    ]
                )
            )
        else:
            self.contour.set_color(self.colors(list(chain(*self.c_segs))))
        # linewidth will be link to time delay
        self.contour.set_lw(
            [
                (1 - (self.now - i) / self.nb_step) * 2.5 if i <= self.now else 0
                for i in chain(*self.t_segs)
            ]
        )
        # Update date txt and info
        txt = f"{(timedelta(int(self.now)) + datetime(1950,1,1)).strftime('%Y/%m/%d')}"
        if self.graphic_informations:
            txt += f"- {1/self.sleep_event:.0f} frame/s"
        self.txt.set_text(txt)
        # Update id txt
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
            self.t_segs.pop(0)
            self.c_segs.pop(0)

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
            # we remove 2 step to add 1 so we rewind of only one
            self.segs.pop(-1)
            self.segs.pop(-1)
            self.t_segs.pop(-1)
            self.t_segs.pop(-1)
            self.c_segs.pop(-1)
            self.c_segs.pop(-1)
            self.prev()


def anim():
    parser = EddyParser(
        """Anim eddy, keyboard shortcut : Escape => exit, SpaceBar => pause,
        left arrow => t - 1, right arrow => t + 1, + => speed increase of 10 %, - => speed decrease of 10 %"""
    )
    parser.add_argument("filename", help="eddy atlas")
    parser.add_argument("id", help="Track id to anim", type=int, nargs="*")
    parser.contour_intern_arg()
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
    parser.add_argument(
        "--first_centered",
        action="store_true",
        help="Longitude will be centered on first obs, if there are only one group.",
    )
    parser.add_argument(
        "--field", default="time", help="Field use to color contour instead of time"
    )
    parser.add_argument(
        "--vmin", default=None, type=float, help="Inferior bound to color contour"
    )
    parser.add_argument(
        "--vmax", default=None, type=float, help="Upper bound to color contour"
    )
    parser.add_argument("--mp4", help="Filename to save animation (mp4)")
    args = parser.parse_args()
    variables = ["time", "track", "longitude", "latitude", args.field]
    variables.extend(TrackEddiesObservations.intern(args.intern, public_label=True))

    eddies = TrackEddiesObservations.load_file(
        args.filename, include_vars=set(variables)
    )
    if not args.all:
        if len(args.id) == 0:
            raise Exception(
                "You need to specify id to display or ask explicity all with --all option"
            )
        eddies = eddies.extract_ids(args.id)
        if args.first_centered:
            # TODO: include observatin class
            x0 = eddies.lon[0]
            eddies.lon[:] = (eddies.lon - x0 + 180) % 360 + x0 - 180
            eddies.contour_lon_e[:] = (
                (eddies.contour_lon_e.T - eddies.lon + 180) % 360 + eddies.lon - 180
            ).T

    kw = dict()
    if args.mp4:
        kw["figsize"] = (16, 9)
        kw["dpi"] = 120
    a = Anim(
        eddies,
        intern=args.intern,
        sleep_event=args.time_sleep,
        cmap=args.cmap,
        nb_step=args.keep_step,
        field_color=args.field,
        range_color=(args.vmin, args.vmax),
        graphic_information=logger.getEffectiveLevel() == logging.DEBUG,
        **kw,
    )
    if args.mp4 is None:
        a.show(infinity_loop=args.infinity_loop)
    else:
        kwargs = dict(frames=arange(*a.period), interval=50)
        ani = FuncAnimation(a.fig, a.func_animation, **kwargs)
        ani.save(args.mp4, fps=30, extra_args=["-vcodec", "libx264"])


def gui_parser():
    parser = EddyParser("Eddy atlas GUI")
    parser.add_argument("atlas", nargs="+")
    parser.add_argument("--med", action="store_true")
    parser.add_argument("--nopath", action="store_true", help="Don't draw path")
    return parser.parse_args()


def guieddy():
    args = gui_parser()
    atlas = {
        dataset: TrackEddiesObservations.load_file(dataset) for dataset in args.atlas
    }
    g = GUI(**atlas)
    if args.med:
        g.med()
    g.hide_path(not args.nopath)
    g.show()
