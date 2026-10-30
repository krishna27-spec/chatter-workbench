"""The centre panel: signal lanes (one per channel, or all in one), a State lane, an Output lane and action markers.

Drawing rule (spec §8b, "never smooth"): on every zoom, pan or resize the visible samples are compared with the lane's
width in device pixels. More samples than pixels → MinMaxPyramid.envelope gives each pixel's min and max, drawn as
interleaved points (one vertical stroke per pixel), so no spike can disappear. Fewer → the raw samples, with a dot
on each once they are 4 px or more apart. Antialiasing is off globally (theme.py), so lines stay 1 px sharp.

Regions (sub-project 2b) are named, coloured time stretches drawn as shaded bands across the signal lanes; the
Spectrum panel computes one spectrum per region. The optional Spectrogram lane shows one channel's STFT, reduced to
the lane's pixel columns by maximum (peak hold), so a short burst stays visible when zoomed out.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsPathItem, QLabel, QVBoxLayout, QWidget

from workbench.core.analysis.spectrum import SG_HOP, Spectrogram, column_max
from workbench.core.recording import Recording
from workbench.core.reference.pyramid import MinMaxPyramid

AXIS_W = 72                                   # same left-axis width on every lane, so time lines up vertically
DOT_SPACING_PX = 4
REGION_PRESETS = ("chatter", "stable", "idle")
REGION_COLOURS = {"chatter": "s3", "stable": "s1", "idle": "ink2"}       # theme token per preset name
OTHER_COLOURS = ("acc", "s2", "wf2", "volt")


@dataclass(eq=False)
class Region:
    """A named stretch of time. Its colour is a theme token, so it follows day/night."""
    id: int
    name: str
    t0: float
    t1: float
    colour_key: str
    visible: bool = True


class _Lane:
    """One PlotItem and the channel curves drawn in it."""

    def __init__(self, plot: pg.PlotItem, names: list[str]):
        self.plot = plot
        self.names = names
        self.curves: dict[str, pg.PlotDataItem] = {}
        self.cursors: list[pg.InfiniteLine] = []


class TimePlot(QWidget):
    view_changed = Signal(float, float)
    cursors_changed = Signal(float, float)
    regions_changed = Signal()

    def __init__(self, tokens: dict[str, str], parent=None):
        super().__init__(parent)
        self.tokens = tokens
        self.rec: Recording | None = None
        self._pyr: dict[str, MinMaxPyramid] = {}
        self._visible: list[str] = []
        self._colours: dict[str, str] = {}
        self._overlaid = False
        self._cursors_on = False
        self._auto_y = True                        # each lane's y range fits the samples in view (exact envelope)
        self._cursor_t = [0.0, 0.0]
        self.lanes: list[_Lane] = []
        self.state_plot: pg.PlotItem | None = None
        self.output_plot: pg.PlotItem | None = None
        self.markers: list[pg.InfiniteLine] = []
        self._regions: list[Region] = []
        self._region_items: dict[int, list[pg.LinearRegionItem]] = {}
        self._region_ids = itertools.count(1)
        self._sg: Spectrogram | None = None
        self._sg_t0 = 0.0
        self._sg_label = ""
        self._sg_levels = (0.0, 1.0)
        self.sg_plot: pg.PlotItem | None = None
        self._sg_image: pg.ImageItem | None = None

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setSpacing(2)
        self.glw.scene().sigMouseClicked.connect(lambda ev: self.fit() if ev.double() else None)
        self.readout = QLabel()
        self.readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.readout.setContentsMargins(8, 3, 8, 3)
        self.readout.hide()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.glw, 1)
        lay.addWidget(self.readout)

        self._redraw_timer = QTimer(self, singleShot=True, interval=0)   # many range signals → one redraw
        self._redraw_timer.timeout.connect(self._redraw)
        self.glw.setBackground(tokens["panel"])

    # ------------------------------------------------------------------ public
    def set_recording(self, rec: Recording | None) -> None:
        self.rec = rec
        self._pyr = {c.name: MinMaxPyramid(c.data) for c in rec.channels} if rec else {}
        self._visible = [c.name for c in rec.channels] if rec else []
        self._colours = {}
        self._regions = []
        self._sg = None
        self._build(keep_view=False)

    def set_visible_channels(self, names: list[str]) -> None:
        self._visible = [n for n in names if n in self._pyr]
        self._build()

    def visible_channels(self) -> list[str]:
        return list(self._visible)

    def set_overlaid(self, on: bool) -> None:
        self._overlaid = on
        self._build()

    def set_auto_y(self, on: bool) -> None:
        """On: y follows the visible data. Off: fixed to the whole file's range (compare amplitudes across zooms)."""
        self._auto_y = on
        self._build()

    def y_range(self, lane: int = 0) -> tuple[float, float]:
        lo, hi = self.lanes[lane].plot.getViewBox().viewRange()[1]
        return float(lo), float(hi)

    # ---- regions
    def regions(self) -> list[Region]:
        return list(self._regions)

    def region_colour(self, r: Region) -> str:
        return self.tokens[r.colour_key]

    def add_region(self, name: str | None = None, t0: float | None = None, t1: float | None = None) -> Region:
        """A new region; by default the next unused preset name, over the middle third of the view."""
        if self.rec is None:
            raise ValueError("open a recording first")
        used = {r.name for r in self._regions}
        rid = next(self._region_ids)
        if name is None:
            name = next((n for n in REGION_PRESETS if n not in used), f"region {rid}")
        if t0 is None or t1 is None:
            a, b = self.view_range()
            t0, t1 = a + (b - a) / 3, a + 2 * (b - a) / 3
        key = REGION_COLOURS.get(name, OTHER_COLOURS[len(self._regions) % len(OTHER_COLOURS)])
        r = Region(rid, name, float(min(t0, t1)), float(max(t0, t1)), key)
        self._regions.append(r)
        self._build()
        self.regions_changed.emit()
        return r

    def update_region(self, r: Region, **changes) -> None:
        """Change name, t0, t1 or visible. A preset name also takes that preset's colour."""
        for k, v in changes.items():
            setattr(r, k, v)
        if "name" in changes and changes["name"] in REGION_COLOURS:
            r.colour_key = REGION_COLOURS[changes["name"]]
        self._build()
        self.regions_changed.emit()

    def remove_region(self, r: Region) -> None:
        self._regions = [x for x in self._regions if x is not r]
        self._build()
        self.regions_changed.emit()

    # ---- spectrogram lane
    def set_spectrogram(self, sg: Spectrogram | None, t0: float = 0.0, label: str = "") -> None:
        """Show sg (frame times relative to t0) as a lane; None removes the lane."""
        self._sg, self._sg_t0, self._sg_label = sg, t0, label
        if sg is not None:
            sample = sg.db[:: max(1, len(sg.db) // 2000)]
            self._sg_levels = (float(np.percentile(sample, 5)), float(sample.max()))
        self._build()

    def set_cursors(self, on: bool) -> None:
        if on and not self._cursors_on and self.rec is not None:
            t0, t1 = self.view_range()
            self._cursor_t = [t0 + (t1 - t0) / 3, t0 + 2 * (t1 - t0) / 3]
        self._cursors_on = on
        self._build()

    def colour_of(self, name: str) -> str:
        if name in self._colours:
            return self._colours[name]
        defaults = {"X": "x", "Y": "y", "Z": "z"}
        if name in defaults:
            return self.tokens[defaults[name]]
        cycle = ("x", "y", "z", "volt", "acc")
        names = [c.name for c in self.rec.channels] if self.rec else [name]
        return self.tokens[cycle[names.index(name) % len(cycle)]] if name in names else self.tokens["ink2"]

    def colours(self) -> dict[str, str]:
        return {c.name: self.colour_of(c.name) for c in self.rec.channels} if self.rec else {}

    def update_channel(self, old: str, new: str, colour: str, visible: bool) -> None:
        """Apply a Properties edit (the window has already renamed the channel in the Recording)."""
        if old != new:
            self._pyr[new] = self._pyr.pop(old)
            self._visible = [new if n == old else n for n in self._visible]
            self._colours.pop(old, None)
        self._colours[new] = colour
        if visible and new not in self._visible:
            order = [c.name for c in self.rec.channels]
            self._visible = [n for n in order if n in self._visible or n == new]
        elif not visible:
            self._visible = [n for n in self._visible if n != new]
        self._build()

    def apply_tokens(self, tokens: dict[str, str]) -> None:
        overrides = self._colours
        self.tokens = tokens
        self._colours = overrides
        self.glw.setBackground(tokens["panel"])
        self._build()

    def view_range(self) -> tuple[float, float]:
        if not self.lanes and self.state_plot is None:
            return (0.0, 0.0)
        vb = self._first_plot().getViewBox()
        x0, x1 = vb.viewRange()[0]
        return float(x0), float(x1)

    def set_view(self, t0: float, t1: float) -> None:
        plot = self._first_plot()
        if plot is not None:
            plot.getViewBox().setXRange(t0, t1, padding=0)
            self._redraw()

    def fit(self) -> None:
        if self.rec is not None:
            self.set_view(self.rec.t0, self.rec.t0 + self.rec.n_samples / self.rec.fs)

    def lane_titles(self) -> list[str]:
        return [lane.plot.getAxis("left").labelText for lane in self.lanes]

    def cursor_values(self) -> dict:
        """t_a, t_b, dt and each visible channel's (value at A, value at B): nearest sample, not interpolated."""
        out = {"t_a": self._cursor_t[0], "t_b": self._cursor_t[1], "dt": self._cursor_t[1] - self._cursor_t[0]}
        if self.rec is not None:
            for ch in self.rec.channels:
                if ch.name in self._visible:
                    ia, ib = (min(self.rec.index_of(t), self.rec.n_samples - 1) for t in self._cursor_t)
                    out[ch.name] = (float(ch.data[ia]), float(ch.data[ib]))
        return out

    def move_cursor(self, which: int, t: float) -> None:
        self._cursor_t[which] = t
        for lane in self.lanes:
            if lane.cursors:
                lane.cursors[which].blockSignals(True)
                lane.cursors[which].setValue(t)
                lane.cursors[which].blockSignals(False)
        self._update_readout()

    # ------------------------------------------------------------------ building
    def _first_plot(self) -> pg.PlotItem | None:
        if self.lanes:
            return self.lanes[0].plot
        return self.state_plot or self.output_plot

    def _new_plot(self, row: int, label: str) -> pg.PlotItem:
        p = self.glw.addPlot(row=row, col=0)
        p.setMenuEnabled(False)
        p.hideButtons()
        p.showGrid(x=True, y=True, alpha=0.25)
        p.getViewBox().setMouseEnabled(x=True, y=False)
        p.getAxis("left").setWidth(AXIS_W)
        p.getAxis("left").enableAutoSIPrefix(False)     # show 0.05, not "50 (x0.001)": real values on the axis
        p.setLabel("left", label)
        for ax in ("left", "bottom"):
            p.getAxis(ax).setPen(pg.mkPen(self.tokens["line"]))
            p.getAxis(ax).setTextPen(pg.mkPen(self.tokens["ink2"]))
        return p

    def _build(self, keep_view: bool = True) -> None:
        view = self.view_range() if keep_view and (self.lanes or self.state_plot) else None
        self.glw.clear()
        self.lanes, self.state_plot, self.output_plot, self.markers = [], None, None, []
        self.sg_plot, self._sg_image, self._region_items = None, None, {}
        self.readout.setVisible(self._cursors_on and self.rec is not None)
        rec = self.rec
        if rec is None:
            return
        t_end = rec.t0 + rec.n_samples / rec.fs
        groups = [self._visible] if self._overlaid and self._visible else [[n] for n in self._visible]
        row = 0
        for names in groups:
            p = self._new_plot(row, self._lane_label(names))
            p.setMinimumHeight(70)                    # a lane never shrinks below a readable trace
            lane = _Lane(p, names)
            for n in names:
                c = p.plot(pen=pg.mkPen(self.colour_of(n), width=1))
                c.setDownsampling(auto=False)
                lane.curves[n] = c
            lo = min(float(np.nanmin(self._pyr[n].mins[-1])) for n in names)
            hi = max(float(np.nanmax(self._pyr[n].maxs[-1])) for n in names)
            if not (math.isfinite(lo) and math.isfinite(hi)):
                lo, hi = -1.0, 1.0
            pad = (hi - lo) * 0.05 or 1.0
            p.getViewBox().setYRange(lo - pad, hi + pad, padding=0)
            self.lanes.append(lane)
            row += 1

        if self._sg is not None:
            self.sg_plot = self._new_plot(row, self._sg_label or "Spectrogram")
            self.sg_plot.getAxis("left").enableAutoSIPrefix(True)       # Hz → kHz on this axis is welcome
            self._sg_image = pg.ImageItem()
            self._sg_image.setColorMap(pg.ColorMap([0.0, 0.35, 0.7, 1.0],
                                                   [QColor(self.tokens[k]) for k in ("wf0", "wf1", "wf2", "wf3")]))
            self.sg_plot.addItem(self._sg_image)
            self.sg_plot.showGrid(x=False, y=False)
            self.sg_plot.getViewBox().setYRange(0, float(self._sg.f[-1]), padding=0)
            self.sg_plot.setMinimumHeight(150)
            row += 1

        dec = rec.tables.get("decisions", {})
        if len(dec.get("state", [])) and len(dec.get("t", [])):
            self.state_plot = self._new_plot(row, "State")
            self._draw_states(self.state_plot, np.asarray(dec["t"], float), np.asarray(dec["state"], float))
            row += 1
        out = rec.tables.get("output", {})
        if len(out.get("spindle_V", [])) and len(out.get("t", [])):
            self.output_plot = self._new_plot(row, "Spindle V")
            self._draw_output(self.output_plot, np.asarray(out["t"], float), np.asarray(out["spindle_V"], float))
            row += 1

        plots = [lane.plot for lane in self.lanes] + [p for p in (self.sg_plot, self.state_plot, self.output_plot) if p]
        if not plots:
            return
        first = plots[0]
        for p in plots:
            vb = p.getViewBox()
            vb.setLimits(xMin=rec.t0, xMax=t_end, minXRange=min(10 / rec.fs, t_end - rec.t0))
            if p is not first:
                p.setXLink(first)
            if p is not plots[-1]:                    # time labels only under the bottom lane; grid lines stay
                p.getAxis("bottom").setStyle(showValues=False)
                p.getAxis("bottom").setHeight(0)
        if self.state_plot:
            self.state_plot.setMaximumHeight(34)
            self.state_plot.setMinimumHeight(34)
        if self.output_plot:
            self.output_plot.setMaximumHeight(90)
        if plots[-1].getAxis("bottom").labelText == "":
            plots[-1].setLabel("bottom", "time", units="s")

        self._add_markers(plots)
        self._add_region_items([lane.plot for lane in self.lanes] + ([self.sg_plot] if self.sg_plot else []))
        if self._cursors_on:
            self._add_cursors()
        vb = first.getViewBox()
        vb.sigXRangeChanged.connect(lambda *_: self._redraw_timer.start())
        vb.sigResized.connect(lambda *_: self._redraw_timer.start())
        if view and view[1] > view[0]:
            vb.setXRange(*view, padding=0)
        else:
            vb.setXRange(rec.t0, t_end, padding=0)
        self._redraw()

    def _lane_label(self, names: list[str]) -> str:
        """'X [V]' for one channel; the tick values show the range (they follow the view with Auto Y)."""
        if len(names) != 1:
            return ", ".join(names)
        ch = self.rec.channel(names[0])
        return f"{ch.name} [{ch.unit}]" if ch.unit else ch.name

    def _draw_states(self, p: pg.PlotItem, t: np.ndarray, state: np.ndarray) -> None:
        """One filled rectangle per run of equal state (state 0–3 coloured s0–s3; NaN = calibrating, left empty)."""
        end_step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0256
        ends = np.append(t[1:], t[-1] + end_step)
        code = np.where(np.isnan(state), -1, state).astype(int)
        cut = np.flatnonzero(np.diff(code)) + 1
        starts, stops = np.r_[0, cut], np.r_[cut, len(code)]
        paths = {k: QPainterPath() for k in range(4)}
        for a, b in zip(starts, stops):
            k = code[a]
            if 0 <= k <= 3:
                paths[k].addRect(QRectF(t[a], 0.0, ends[b - 1] - t[a], 1.0))
        for k, path in paths.items():
            item = QGraphicsPathItem(path)
            item.setBrush(QBrush(QColor(self.tokens[f"s{k}"])))
            item.setPen(QPen(Qt.NoPen))
            p.addItem(item)
        p.getViewBox().setYRange(0, 1, padding=0)
        p.getAxis("left").setTicks([[]])
        p.showGrid(x=False, y=False)

    def _draw_output(self, p: pg.PlotItem, t: np.ndarray, v: np.ndarray) -> None:
        ok = np.isfinite(t) & np.isfinite(v)
        t, v = t[ok], v[ok]
        if not len(t):
            return
        x = np.repeat(t, 2)[1:]                       # steps: the value holds until the next decision
        y = np.repeat(v, 2)[:-1]
        p.plot(x, y, pen=pg.mkPen(self.tokens["volt"], width=1))
        lo, hi = float(v.min()), float(v.max())
        pad = (hi - lo) * 0.15 or 0.5
        p.getViewBox().setYRange(lo - pad, hi + pad, padding=0)
        ticks = [lo] if hi == lo else [lo, hi]        # a short lane: label only the levels actually used
        p.getAxis("left").setTicks([[(t, f"{t:.3g}") for t in ticks], []])

    def _add_markers(self, plots: list[pg.PlotItem]) -> None:
        timed = [e for e in self.rec.events if e.t is not None]
        pen = pg.mkPen(self.tokens["ink2"], width=1, style=Qt.DashLine)
        for i, ev in enumerate(timed):
            for j, p in enumerate(plots):
                opts = {}
                if j == 0:                            # text only on the top lane, 3 rows so neighbours don't collide
                    text = ev.text if len(ev.text) <= 32 else ev.text[:31] + "…"
                    opts = dict(label=text, labelOpts={"position": (0.92, 0.78, 0.64)[i % 3],
                                                               "color": self.tokens["ink"], "anchors": [(0, 0.5), (0, 0.5)]})
                line = pg.InfiniteLine(pos=ev.t, angle=90, movable=False, pen=pen, **opts)
                p.addItem(line, ignoreBounds=True)
                self.markers.append(line)

    def _add_region_items(self, plots: list[pg.PlotItem]) -> None:
        for r in self._regions:
            if not r.visible:
                continue
            colour = QColor(self.region_colour(r))
            fill = QColor(colour)
            fill.setAlpha(40)
            items = []
            for j, p in enumerate(plots):
                item = pg.LinearRegionItem(values=(r.t0, r.t1), brush=QBrush(fill), pen=pg.mkPen(colour, width=1))
                item.setZValue(-10)                   # behind the curves
                if j == 0:
                    pg.InfLineLabel(item.lines[0], text=r.name, position=0.97, color=colour, anchors=[(0, 0), (0, 0)])
                item.sigRegionChanged.connect(lambda it, rid=r.id: self._on_region_dragged(rid, it))
                p.addItem(item, ignoreBounds=True)
                items.append(item)
            self._region_items[r.id] = items

    def _on_region_dragged(self, rid: int, item: pg.LinearRegionItem) -> None:
        t0, t1 = item.getRegion()
        r = next(x for x in self._regions if x.id == rid)
        r.t0, r.t1 = float(t0), float(t1)
        for other in self._region_items.get(rid, []):
            if other is not item:
                other.blockSignals(True)
                other.setRegion((t0, t1))
                other.blockSignals(False)
        self.regions_changed.emit()

    def _add_cursors(self) -> None:
        pen = pg.mkPen(self.tokens["acc"], width=1)
        for lane in self.lanes:
            for which in (0, 1):
                line = pg.InfiniteLine(pos=self._cursor_t[which], angle=90, movable=True, pen=pen,
                                       hoverPen=pg.mkPen(self.tokens["acc"], width=2),
                                       label="AB"[which] if lane is self.lanes[0] else None,
                                       labelOpts={"position": 0.05, "color": self.tokens["acc"]})
                line.sigPositionChanged.connect(lambda ln, w=which: self._on_cursor_moved(w, ln.value()))
                lane.plot.addItem(line, ignoreBounds=True)
                lane.cursors.append(line)
        self._update_readout()

    def _on_cursor_moved(self, which: int, t: float) -> None:
        self.move_cursor(which, t)
        self.cursors_changed.emit(*self._cursor_t)

    def _update_readout(self) -> None:
        if not self._cursors_on or self.rec is None:
            return
        v = self.cursor_values()
        parts = [f"A {v['t_a']:.6f} s", f"B {v['t_b']:.6f} s", f"Δt {v['dt'] * 1e3:.3f} ms"]
        parts += [f"{n}: {v[n][0]:.6g} / {v[n][1]:.6g}" for n in self._visible if n in v]
        self.readout.setText("    ".join(parts))

    # ------------------------------------------------------------------ drawing
    def _redraw_spectrogram(self, x0: float, x1: float) -> None:
        if self._sg is None or self._sg_image is None:
            return
        n_px = max(1, int(self.sg_plot.getViewBox().boundingRect().width() * self.devicePixelRatioF()))
        t_cols, cols = column_max(self._sg, x0 - self._sg_t0, x1 - self._sg_t0, n_px)
        if not len(t_cols):
            self._sg_image.clear()
            return
        dt = float(np.median(np.diff(t_cols))) if len(t_cols) > 1 else SG_HOP / self.rec.fs
        df = float(self._sg.f[1] - self._sg.f[0])
        self._sg_image.setImage(cols, autoLevels=False, levels=self._sg_levels)
        self._sg_image.setRect(QRectF(self._sg_t0 + t_cols[0] - dt / 2, -df / 2, dt * len(t_cols),
                                      df * cols.shape[1]))

    def _redraw(self) -> None:
        rec = self.rec
        if rec is None or not self.lanes:
            if rec is not None:
                self._redraw_spectrogram(*self.view_range())
                self.view_changed.emit(*self.view_range())
            return
        x0, x1 = self.view_range()
        n = rec.n_samples
        first = max(0, math.ceil((x0 - rec.t0) * rec.fs - 1e-9))
        last = min(n - 1, math.floor((x1 - rec.t0) * rec.fs + 1e-9))
        if last < first:
            first = last = min(max(first, 0), n - 1)
        count = last - first + 1
        vb = self.lanes[0].plot.getViewBox()
        n_px = max(1, int(vb.boundingRect().width() * self.devicePixelRatioF()))
        for lane in self.lanes:
            lane_lo, lane_hi = math.inf, -math.inf
            for name, curve in lane.curves.items():
                data = rec.channel(name).data
                if count > n_px:
                    lo, hi = self._pyr[name].envelope(first, last + 1, n_px)
                    edges = first + np.arange(n_px + 1) * count // n_px
                    centre = rec.t0 + (edges[:-1] + edges[1:] - 1) / 2 / rec.fs
                    x = np.repeat(centre, 2)
                    y = np.column_stack([lo, hi]).ravel()
                    curve.setData(x, y, connect="finite", symbol=None)
                    seen = y
                else:
                    a, b = max(0, first - 1), min(n, last + 2)        # one extra each side: lines run to the edge
                    t = rec.t0 + np.arange(a, b) / rec.fs
                    dots = n_px / count >= DOT_SPACING_PX
                    colour = self.colour_of(name)
                    curve.setData(t, np.asarray(data[a:b], dtype=np.float64), connect="finite",
                                  symbol="o" if dots else None, symbolSize=3, symbolPen=None,
                                  symbolBrush=colour if dots else None)
                    seen = np.asarray(data[first:last + 1], dtype=np.float64)
                if self._auto_y and len(seen) and np.isfinite(seen).any():
                    lane_lo = min(lane_lo, float(np.nanmin(seen)))
                    lane_hi = max(lane_hi, float(np.nanmax(seen)))
            if self._auto_y and math.isfinite(lane_lo):
                pad = (lane_hi - lane_lo) * 0.05 or max(abs(lane_hi) * 0.05, 1e-9)
                lane.plot.getViewBox().setYRange(lane_lo - pad, lane_hi + pad, padding=0)
        self._redraw_spectrogram(x0, x1)
        self.view_changed.emit(x0, x1)
