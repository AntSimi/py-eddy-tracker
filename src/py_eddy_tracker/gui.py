# -*- coding: utf-8 -*-
"""
GUI class
"""

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections import register_projection

from .generic import flatten_line_matrix, split_line

try:
    from pylook.axes import PlatCarreAxes
except ImportError:
    from matplotlib.axes import Axes

    class PlatCarreAxes(Axes):
        """
        Class to replace missing pylook class
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_aspect("equal")


class GUIAxes(PlatCarreAxes):
    """
    Axes which will use full space available
    """

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
        "d_indexs",
        "m",
        "last_event",
    )
    COLORS = ("r", "g", "b", "y", "k")
    KEYTIME = dict(down=-1, up=1, pagedown=-5, pageup=5)

    def __init__(self, **datasets):
        self.datasets = datasets
        self.d_indexs = dict()
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

        self.settings = dict(period=(t0, t1), now=t1)

    @property
    def now(self):
        return self.settings["now"]

    @now.setter
    def now(self, value):
        self.settings["now"] = value

    @property
    def period(self):
        return self.settings["period"]

    @property
    def bbox(self):
        return self.map.get_xlim(), self.map.get_ylim()

    @bbox.setter
    def bbox(self, values):
        self.map.set_xlim(values[0], values[1])
        self.map.set_ylim(values[2], values[3])

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
        self.map.tick_params("both", pad=-22)
        # self.map.tick_params("y", pad=-22)
        self.map.bg_cache = None
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

    def hide_path(self, state):
        for name in self.datasets:
            self.m[name]["path_previous"].set_visible(state)
            self.m[name]["path_future"].set_visible(state)

    def draw(self):
        self.m["mini_ax"] = self.figure.add_axes((0.3, 0.85, 0.4, 0.15), zorder=80)
        self.m["mini_ax"].grid()
        # map
        for i, (name, dataset) in enumerate(self.datasets.items()):
            kwargs = dict(color=self.COLORS[i])
            self.m[name] = dict(
                contour_s=self.map.plot([], [], lw=1, label=name, **kwargs)[0],
                contour_e=self.map.plot([], [], lw=0.5, ls="-.", **kwargs)[0],
                path_previous=self.map.plot([], [], lw=0.5, **kwargs)[0],
                path_future=self.map.plot([], [], lw=0.2, ls=":", **kwargs)[0],
                mini_line=self.m["mini_ax"].plot([], [], **kwargs, lw=1)[0],
            )
        # time_ax
        self.m["annotate"] = self.map.annotate(
            "",
            (0, 0),
            xycoords="figure pixels",
            zorder=100,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="w", edgecolor="0.5", alpha=0.85),
        )
        self.m["mini_ax"].set_visible(False)
        self.m["annotate"].set_visible(False)

        self.m["time_vline"] = self.time_ax.axvline(0, color="k", lw=1)
        self.m["time_text"] = self.time_ax.text(
            0,
            0,
            "",
            fontsize=8,
            bbox=dict(facecolor="w", alpha=0.75),
            verticalalignment="bottom",
        )

    def update(self):
        time_text = [
            (timedelta(days=int(self.now)) + datetime(1950, 1, 1)).strftime("%d/%m/%Y")
        ]
        # map
        xs, ys, ns = list(), list(), list()
        for j, (name, dataset) in enumerate(self.datasets.items()):
            i = self.indexs(dataset)
            self.d_indexs[name] = i
            self.m[name]["contour_s"].set_label(f"{name} {len(i)} eddies")
            if len(i) == 0:
                self.m[name]["contour_s"].set_data([], [])
                self.m[name]["contour_e"].set_data([], [])
            else:
                if "contour_lon_s" in dataset.elements:
                    self.m[name]["contour_s"].set_data(
                        flatten_line_matrix(dataset["contour_lon_s"][i]),
                        flatten_line_matrix(dataset["contour_lat_s"][i]),
                    )
                if "contour_lon_e" in dataset.elements:
                    self.m[name]["contour_e"].set_data(
                        flatten_line_matrix(dataset["contour_lon_e"][i]),
                        flatten_line_matrix(dataset["contour_lat_e"][i]),
                    )
            time_text.append(f"{i.shape[0]}")
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

        self.map.legend()
        # time ax
        x, y = self.m["time_vline"].get_data()
        self.m["time_vline"].set_data(self.now, y)
        self.m["time_text"].set_text("\n".join(time_text))
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

    def get_infos(self, name, index):
        i = self.d_indexs[name][index]
        d = self.datasets[name]
        now = d.obs[i]
        tr = now["track"]
        nb = d.nb_obs_by_track[tr]
        i_first = d.index_from_track[tr]
        track = d.obs[i_first : i_first + nb]
        nb -= 1
        t0 = timedelta(days=int(track[0]["time"])) + datetime(1950, 1, 1)
        t1 = timedelta(days=int(track[-1]["time"])) + datetime(1950, 1, 1)
        txt = f"--{name}--\n"
        txt += f"    {t0} -> {t1}\n"
        txt += f"    Tracks : {tr}  {now['n']}/{nb} ({now['n'] / nb * 100:.2f} %)\n"
        for label, n, f, u in (
            ("Amp.", "amplitude", 100, "cm"),
            ("S. radius", "radius_s", 1e-3, "km"),
            ("E. radius", "radius_e", 1e-3, "km"),
        ):
            v = track[n] * f
            min_, max_, mean_, std_ = v.min(), v.max(), v.mean(), v.std()
            txt += f"    {label} : {now[n] * f:.1f} {u} ({min_:.1f} <-{mean_:.1f}+-{std_:.1f}-> {max_:.1f})\n"
        return track, txt.strip()

    def move(self, event):
        if event.inaxes == self.time_ax and self.time_ax.press:
            x, y = self.m["time_vline"].get_data()
            self.m["time_vline"].set_data(event.xdata, y)
            self.figure.canvas.restore_region(self.time_ax.bg_cache)
            self.time_ax.draw_artist(self.m["time_vline"])
            self.figure.canvas.blit(self.time_ax.bbox)

        if event.inaxes == self.map:
            touch = dict()
            for name in self.datasets.keys():
                flag, data = self.m[name]["contour_s"].contains(event)
                if flag:
                    # 51 is for contour on 50 point must be rewrote
                    touch[name] = data["ind"][0] // 51
            a = self.m["annotate"]
            ax = self.m["mini_ax"]
            if touch:
                if not a.get_visible():
                    self.map.bg_cache = self.figure.canvas.copy_from_bbox(self.map.bbox)
                    a.set_visible(True)
                    ax.set_visible(True)
                else:
                    self.figure.canvas.restore_region(self.map.bg_cache)
                a.set_x(event.x), a.set_y(event.y)
                txt = list()
                x0_, x1_, y1_ = list(), list(), list()
                for name in self.datasets.keys():
                    if name in touch:
                        track, txt_ = self.get_infos(name, touch[name])
                        txt.append(txt_)
                        x, y = track["time"], track["radius_s"] / 1e3
                        self.m[name]["mini_line"].set_data(x, y)
                        x0_.append(x.min()), x1_.append(x.max()), y1_.append(y.max())
                    else:
                        self.m[name]["mini_line"].set_data([], [])
                ax.set_xlim(min(x0_), max(x1_)), ax.set_ylim(0, max(y1_))
                a.set_text("\n".join(txt))

                self.map.draw_artist(a)
                self.map.draw_artist(ax)
                self.figure.canvas.blit(self.map.bbox)
            if not flag and self.map.bg_cache is not None and a.get_visible():
                a.set_visible(False)
                ax.set_visible(False)
                self.figure.canvas.restore_region(self.map.bg_cache)
                self.map.draw_artist(a)
                self.map.draw_artist(ax)
                self.figure.canvas.blit(self.map.bbox)
                self.map.bg_cache = None

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
