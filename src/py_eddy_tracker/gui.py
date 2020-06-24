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

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.projections import register_projection
import py_eddy_tracker_sample as sample
from .generic import flatten_line_matrix, split_line
from .observations.tracking import TrackEddiesObservations


try:
    from pylook.axes import PlatCarreAxes
except ImportError:
    from matplotlib.axes import Axes

    class PlatCarreAxes(Axes):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_aspect("equal")


class GUIAxes(PlatCarreAxes):
    name = "full_axes"

    def end_pan(self, *args, **kwargs):
        (x0, x1), (y0, y1) = self.get_xlim(), self.get_ylim()
        x, y = (x1 + x0) / 2, (y1 + y0) / 2
        dx, dy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
        r_coord = dx / dy
        # r_axe
        _, _, w_ax, h_ax = self.get_position(original=True).bounds
        w_fig, h_fig = self.figure.get_size_inches()
        r_ax = w_ax / h_ax * w_fig / h_fig
        if r_ax < r_coord:
            y0, y1 = y - dx / r_ax, y + dx / r_ax
            self.set_ylim(y0, y1)
        else:
            x0, x1 = x - dy * r_ax, x + dy * r_ax
            self.set_xlim(x0, x1)
        super().end_pan(*args, **kwargs)


register_projection(GUIAxes)


class A(TrackEddiesObservations):
    pass


def no(*args, **kwargs):
    return False


class GUI:
    __slots__ = (
        "datasets",
        "figure",
        "map",
        "time_ax",
        "param_ax",
        "settings",
        "m",
        "last_event",
    )
    COLORS = ("r", "g", "b", "y", "k")
    KEYTIME = dict(down=-1, up=1, pagedown=-5, pageup=5)

    def __init__(self, **datasets):
        self.datasets = datasets
        self.m = dict()
        self.set_initial_values()
        self.setup()
        self.last_event = datetime.now()
        self.draw()
        self.event()

    def set_initial_values(self):
        t0, t1 = 1e6, 0
        for dataset in self.datasets.values():
            t0_, t1_ = dataset.period
            t0, t1 = min(t0, t0_), max(t1, t1_)

        self.settings = dict(period=(t0, t1), now=t1,)

    @property
    def now(self):
        return self.settings["now"]

    @property
    def period(self):
        return self.settings["period"]

    @property
    def bbox(self):
        return self.map.get_xlim(), self.map.get_ylim()

    def indexs(self, dataset):
        (x0, x1), (y0, y1) = self.bbox
        x, y = dataset.longitude, dataset.latitude
        m = (x0 < x) & (x < x1) & (y0 < y) & (y < y1) & (self.now == dataset.time)
        return np.where(m)[0]

    def med(self):
        self.map.set_xlim(-6, 37)
        self.map.set_ylim(30, 46)

    def setup(self):
        self.figure = plt.figure()
        # map
        self.map = self.figure.add_axes((0, 0.25, 1, 0.75), projection="full_axes")
        self.map.grid()
        self.map.tick_params("x", pad=-12)
        self.map.tick_params("y", pad=-22)
        # time ax
        self.time_ax = self.figure.add_axes((0, 0.15, 1, 0.1), facecolor=".95")
        self.time_ax.can_pan
        self.time_ax.set_xlim(*self.period)
        self.time_ax.press = False
        self.time_ax.can_pan = self.time_ax.can_zoom = no
        for i, dataset in enumerate(self.datasets.values()):
            self.time_ax.hist(
                dataset.time,
                bins=np.arange(self.period[0] - 0.5, self.period[1] + 0.51),
                color=self.COLORS[i],
                histtype="step",
            )
        # param
        self.param_ax = self.figure.add_axes((0, 0, 1, 0.15), facecolor="0.2")

    def draw(self):
        # map
        for i, (name, dataset) in enumerate(self.datasets.items()):
            self.m[name] = dict(
                contour_s=self.map.plot(
                    [], [], color=self.COLORS[i], lw=0.5, label=name
                )[0],
                contour_e=self.map.plot([], [], color=self.COLORS[i], lw=0.5)[0],
                path_previous=self.map.plot([], [], color=self.COLORS[i], lw=0.5)[0],
                path_future=self.map.plot([], [], color=self.COLORS[i], lw=0.2, ls=":")[
                    0
                ],
            )
        self.m["title"] = self.map.set_title("")
        # time_ax
        self.m["time_vline"] = self.time_ax.axvline(0, color="k", lw=1)
        self.m["time_text"] = self.time_ax.text(
            0, 0, "", fontsize=8, bbox=dict(facecolor="w", alpha=0.75)
        )

    def update(self):
        # text = []
        # map
        xs, ys, ns = list(), list(), list()
        for j, (name, dataset) in enumerate(self.datasets.items()):
            i = self.indexs(dataset)
            self.m[name]["contour_s"].set_label(f"{name} {len(i)} eddies")
            if len(i) == 0:
                self.m[name]["contour_s"].set_data([], [])
            else:
                self.m[name]["contour_s"].set_data(
                    flatten_line_matrix(dataset["contour_lon_s"][i]),
                    flatten_line_matrix(dataset["contour_lat_s"][i]),
                )
            # text.append(f"{i.shape[0]}")
            local_path = dataset.extract_ids(dataset["track"][i])
            x, y, t, n, tr = (
                local_path.longitude,
                local_path.latitude,
                local_path.time,
                local_path["n"],
                local_path["track"],
            )
            m = t <= self.now
            if m.sum():
                x_, y_ = split_line(x[m], y[m], tr[m])
                self.m[name]["path_previous"].set_data(x_, y_)
            else:
                self.m[name]["path_previous"].set_data([], [])
            m = t >= self.now
            if m.sum():
                x_, y_ = split_line(x[m], y[m], tr[m])
                self.m[name]["path_future"].set_data(x_, y_)
            else:
                self.m[name]["path_future"].set_data([], [])
            m = t == self.now
            xs.append(x[m]), ys.append(y[m]), ns.append(n[m])

        x, y, n = np.concatenate(xs), np.concatenate(ys), np.concatenate(ns)
        n_min = 0
        if len(n) > 50:
            n_ = n.copy()
            n_.sort()
            n_min = n_[-50]
        for text in self.m.pop("texts", list()):
            text.remove()
        self.m["texts"] = [
            self.map.text(x_, y_, n_) for x_, y_, n_ in zip(x, y, n) if n_ >= n_min
        ]

        self.m["title"].set_text(self.now)
        self.map.legend()
        # time ax
        x, y = self.m["time_vline"].get_data()
        self.m["time_vline"].set_data(self.now, y)
        # self.m["time_text"].set_text("\n".join(text))
        self.m["time_text"].set_position((self.now, 0))
        # force update
        self.map.figure.canvas.draw()

    def event(self):
        self.figure.canvas.mpl_connect("resize_event", self.adjust)
        self.figure.canvas.mpl_connect("scroll_event", self.scroll)
        self.figure.canvas.mpl_connect("button_press_event", self.press)
        self.figure.canvas.mpl_connect("motion_notify_event", self.move)
        self.figure.canvas.mpl_connect("button_release_event", self.release)
        self.figure.canvas.mpl_connect("key_press_event", self.keyboard)

    def keyboard(self, event):
        if event.key in self.KEYTIME:
            self.settings["now"] += self.KEYTIME[event.key]
            self.update()
        elif event.key == "home":
            self.settings["now"] = self.period[0]
            self.update()
        elif event.key == "end":
            self.settings["now"] = self.period[1]
            self.update()

    def press(self, event):
        if event.inaxes == self.time_ax and self.m["time_vline"].contains(event)[0]:
            self.time_ax.press = True
            self.time_ax.bg_cache = self.figure.canvas.copy_from_bbox(self.time_ax.bbox)

    def move(self, event):
        if event.inaxes == self.time_ax and self.time_ax.press:
            x, y = self.m["time_vline"].get_data()
            self.m["time_vline"].set_data(event.xdata, y)
            self.figure.canvas.restore_region(self.time_ax.bg_cache)
            self.time_ax.draw_artist(self.m["time_vline"])
            self.figure.canvas.blit(self.time_ax.bbox)

    def release(self, event):
        if self.time_ax.press:
            self.time_ax.press = False
            self.settings["now"] = int(self.m["time_vline"].get_data()[0])
            self.update()

    def scroll(self, event):
        if event.inaxes != self.map:
            return
        if event.button == "up":
            self.settings["now"] += 1
        if event.button == "down":
            self.settings["now"] -= 1
        self.update()

    def adjust(self, event=None):
        self.map._pan_start = None
        self.map.end_pan()

    def show(self):
        self.update()
        plt.show()


if __name__ == "__main__":

    # a_ = A.load_file(
    # "/home/toto/dev/work/pet/20200611_example_dataset/tracking/Anticyclonic_track_too_short.nc"
    # )
    # c_ = A.load_file(
    #     "/home/toto/dev/work/pet/20200611_example_dataset/tracking/Cyclonic_track_too_short.nc"
    # )
    a = A.load_file(sample.get_path("eddies_med_adt_allsat_dt2018/Anticyclonic.zarr"))
    # c = A.load_file(sample.get_path("eddies_med_adt_allsat_dt2018/Cyclonic.zarr"))
    # g = GUI(Acyc=a, Cyc=c, Acyc_short=a_, Cyc_short=c_)
    g = GUI(Acyc=a)
    # g = GUI(Acyc_short=a_)
    # g = GUI(Acyc_short=a_, Cyc_short=c_)
    g.med()
    g.show()
