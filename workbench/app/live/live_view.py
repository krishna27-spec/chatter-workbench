"""The Live workspace's centre: the Chatter Monitor dashboard's cards, drawn from the engine about 30 times a second.

Status cards (process state, the three models, RMS, dominant frequency, spindle and output, recording), the last N
seconds of every channel (min/max per pixel, as in Analyse), the state history strip, the live spectrum of the last
second with spindle harmonics, the waterfall (a rolling spectrogram, computed only for new frames), and the log of
decisions and actions. Reading the engine never blocks it: the ring buffer and snapshot() copy under a short lock.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
                               QSplitter, QVBoxLayout, QWidget)

from workbench.core.analysis.spectrum import SCALES, SG_HOP, SG_WIN, harmonics, peaks, psd, scaled, to_db

STATE_NAMES = {0: "No cutting", 1: "Stable", 2: "Transition", 3: "CHATTER", None: "Calibrating"}
WINDOWS = (5.0, 10.0, 30.0)


def _card(title: str) -> tuple[QFrame, QLabel, QLabel]:
    f = QFrame()
    f.setFrameShape(QFrame.StyledPanel)
    v = QVBoxLayout(f)
    v.setContentsMargins(8, 4, 8, 6)
    t = QLabel(title.upper())
    t.setStyleSheet("font-size: 10px; letter-spacing: 1px;")
    big = QLabel("–")
    big.setStyleSheet("font-size: 18px; font-weight: 600;")
    small = QLabel("")
    small.setWordWrap(True)
    for w in (t, big, small):
        v.addWidget(w)
    return f, big, small


class LiveView(QWidget):
    def __init__(self, tokens: dict[str, str], parent=None):
        super().__init__(parent)
        self.tokens = tokens
        self.paused = False
        self._frame = 0
        self._n_events = 0
        self._wf = None
        self._wf_key = None
        self._lanes = []
        self._lane_key = None

        cards = QGridLayout()
        self.cards = {}
        for i, (key, title) in enumerate((("state", "Process state"), ("models", "Top-3 models"), ("rms", "RMS"),
                                          ("dom", "Dominant frequency"), ("spindle", "Spindle output"),
                                          ("rec", "Recording"))):
            frame, big, small = _card(title)
            self.cards[key] = (big, small)
            cards.addWidget(frame, 0, i)

        self.pause_btn = QPushButton("Pause view")
        self.pause_btn.setCheckable(True)
        self.window = QComboBox()
        for w in WINDOWS:
            self.window.addItem(f"last {w:g} s", w)
        self.window.setCurrentIndex(1)
        self.channel = QComboBox()
        self.scale = QComboBox()
        self.scale.addItem("dB (PSD)", "db")
        self.scale.addItem("linear PSD", "psd")
        self.scale.addItem("linear amplitude", "amplitude")
        self.recal_btn = QPushButton("Recalibrate")
        self.record_btn = QPushButton("Record")
        self.record_btn.setCheckable(True)
        self.hint = QLabel("Choose the input in Inputs (left) and press Start acquisition.")
        bar = QHBoxLayout()
        for w in (self.pause_btn, self.window, QLabel("Spectrum of"), self.channel, self.scale, self.recal_btn,
                  self.record_btn,
                  self.hint):
            bar.addWidget(w)
        bar.addStretch(1)
        self.pause_btn.toggled.connect(lambda on: setattr(self, "paused", on))

        self.glw = pg.GraphicsLayoutWidget()
        self.spec = pg.PlotWidget()
        self.spec.setLabel("bottom", "frequency", units="Hz")
        self.spec.setLabel("left", "PSD [dB]")
        self.spec.getAxis("left").enableAutoSIPrefix(False)
        self.spec_curve = self.spec.plot(pen=pg.mkPen(tokens["acc"], width=1))
        self._harm_lines = []
        self.wf_plot = pg.PlotWidget()
        self.wf_plot.setLabel("left", "frequency", units="Hz")
        self.wf_plot.setLabel("bottom", "time", units="s")
        self.wf_img = pg.ImageItem()
        self.wf_plot.addItem(self.wf_img)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        bottom = QSplitter(Qt.Horizontal)
        for w in (self.spec, self.wf_plot, self.log):
            bottom.addWidget(w)
        main = QSplitter(Qt.Vertical)
        main.addWidget(self.glw)
        main.addWidget(bottom)
        main.setStretchFactor(0, 3)
        main.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(cards)
        lay.addLayout(bar)
        lay.addWidget(main, 1)
        self.apply_tokens(tokens)

    def apply_tokens(self, tokens: dict[str, str]) -> None:
        self.tokens = tokens
        for w in (self.glw, self.spec, self.wf_plot):
            w.setBackground(tokens["panel"])
        self.spec_curve.setPen(pg.mkPen(tokens["acc"], width=1))
        self.wf_img.setColorMap(pg.ColorMap([0.0, 0.35, 0.7, 1.0], [QColor(tokens[k]) for k in ("wf0", "wf1", "wf2", "wf3")]))
        self._lane_key = None                          # rebuild lanes in the new colours

    # ------------------------------------------------------------------ per engine
    def attach(self, names: list[str]) -> None:
        """A new acquisition: lanes and channel list for its channels."""
        self.channel.clear()
        self.channel.addItems(names)
        if "Y" in names:
            self.channel.setCurrentText("Y")
        self._lane_key = None
        self._wf = None
        self._n_events = 0
        self.log.clear()
        self.hint.setText("")

    def _build_lanes(self, names: list[str]) -> None:
        self.glw.clear()
        self._lanes = []
        colours = {"X": "x", "Y": "y", "Z": "z"}
        for i, n in enumerate(names):
            p = self.glw.addPlot(row=i, col=0)
            p.setMenuEnabled(False)
            p.hideButtons()
            p.setMouseEnabled(x=False, y=False)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setLabel("left", n)
            p.getAxis("left").setWidth(64)
            p.getAxis("left").enableAutoSIPrefix(False)
            if i < len(names) - 1:
                p.getAxis("bottom").setStyle(showValues=False)
            c = p.plot(pen=pg.mkPen(self.tokens[colours.get(n, ("x", "y", "z", "volt")[i % 4])], width=1))
            if self._lanes:
                p.setXLink(self._lanes[0][0])
            self._lanes.append((p, c))
        self.state_plot = self.glw.addPlot(row=len(names), col=0)
        self.state_plot.setMaximumHeight(28)
        self.state_plot.hideAxis("left")
        self.state_plot.hideAxis("bottom")
        self.state_plot.setMouseEnabled(x=False, y=False)
        self.state_img = pg.ImageItem()
        self.state_plot.addItem(self.state_img)
        self.state_plot.setXLink(self._lanes[0][0])
        self._lane_key = tuple(names)

    def refresh(self, engine, rpm: float | None = None, teeth: int = 2) -> None:
        if engine is None or self.paused:
            return
        snap = engine.snapshot()
        fs = engine.fs
        names = engine.source.names
        if self._lane_key != tuple(names):
            self._build_lanes(names)
        span = self.window.currentData()
        x, first = engine.ring.latest(int(span * fs))
        n = x.shape[1]
        if n >= 2:
            t0 = first / fs
            n_px = max(200, int(self.glw.width()))
            for (p, c), row in zip(self._lanes, x):
                if n > n_px:
                    edges = np.arange(n_px + 1) * n // n_px
                    lo = np.minimum.reduceat(row, edges[:-1])
                    hi = np.maximum.reduceat(row, edges[:-1])
                    tt = t0 + (edges[:-1] + edges[1:]) / 2 / fs
                    c.setData(np.repeat(tt, 2), np.column_stack([lo, hi]).ravel())
                else:
                    c.setData(t0 + np.arange(n) / fs, row)
            t_end = (first + n) / fs
            self._lanes[0][0].setXRange(max(0.0, t_end - span), t_end, padding=0)
            self._draw_states(engine, t_end - span, t_end)
        self._frame += 1
        if self._frame % 6 == 0 and n >= 2048:                  # spectrum and dominant f, 5 times a second
            self._spectrum(x, fs, names, rpm, teeth)
        if self._frame % 2 == 0:
            self._waterfall(x, first, fs, names, span)
        self._cards(snap, x, fs, names)
        self._events(engine)

    def _draw_states(self, engine, t0, t1) -> None:
        rows = [d for d in list(engine.decisions) if t0 <= d["t"] <= t1]
        if not rows:
            self.state_img.clear()
            return
        colours = {k: QColor(self.tokens[f"s{k}"]) for k in range(4)}
        img = np.zeros((len(rows), 1, 4), np.uint8)
        for i, d in enumerate(rows):
            s = d["state"]
            if s != "" and s is not None:
                c = colours[int(s)]
                img[i, 0] = (c.red(), c.green(), c.blue(), 255)
        self.state_img.setImage(img, autoLevels=False)
        hop = SG_HOP / engine.fs
        self.state_img.setRect(QRectF(rows[0]["t"] - hop, 0, len(rows) * hop, 1))

    def _spectrum(self, x, fs, names, rpm, teeth) -> None:
        name = self.channel.currentText()
        if name not in names:
            return
        seg = x[names.index(name), -int(fs):]
        try:
            s = psd(seg, fs, nperseg=2048)
        except ValueError:
            return
        scale = self.scale.currentData()
        self.spec.setLabel("left", SCALES[scale].format(u="V"))
        self.spec_curve.setData(s.f[1:], scaled(s, scale)[1:])
        for line in self._harm_lines:
            self.spec.removeItem(line)
        self._harm_lines = []
        if rpm and s.f[-1] / (teeth * rpm / 60.0) <= 400:      # same cap as the Spectrum panel
            h = harmonics(rpm, teeth, s.f[-1])
            for f in h.tooth_lines:
                line = pg.InfiniteLine(f, angle=90, pen=pg.mkPen(self.tokens["ink2"], width=1, style=Qt.DashLine))
                self.spec.addItem(line, ignoreBounds=True)
                self._harm_lines.append(line)
        top = peaks(s.f, s.P, n=1)
        big, small = self.cards["dom"]
        big.setText(f"{top[0].f:.0f} Hz" if top else "–")
        small.setText(f"{name}, last 1 s" + (f", tooth passing {teeth * rpm / 60:.1f} Hz" if rpm else ""))

    def _waterfall(self, x, first, fs, names, span) -> None:
        name = self.channel.currentText()
        if name not in names:
            return
        key = (name, span, fs)
        n_keep = int(span * fs / SG_HOP)
        if self._wf is None or self._wf_key != key:
            self._wf = np.full((n_keep, SG_WIN // 2 + 1), np.nan, np.float32)
            self._wf_key = key
            self._wf_next = first                                  # sample index of the next frame start
        row = x[names.index(name)]
        start = max(self._wf_next, first)
        k = (first + x.shape[1] - SG_WIN - start) // SG_HOP + 1
        if k <= 0:
            return
        k = min(k, n_keep)
        start = max(start, first + x.shape[1] - SG_WIN - (k - 1) * SG_HOP)
        idx = (start - first) + np.arange(k)[:, None] * SG_HOP + np.arange(SG_WIN)
        seg = row[idx].astype(np.float64)
        w = np.hanning(SG_WIN + 1)[:-1]
        seg = (seg - seg.mean(1, keepdims=True)) * w
        P = np.abs(np.fft.rfft(seg, axis=1)) ** 2 / (fs * np.sum(w ** 2))
        P[:, 1:-1] *= 2
        self._wf = np.concatenate([self._wf[k:], to_db(P).astype(np.float32)])
        self._wf_next = start + k * SG_HOP
        finite = self._wf[np.isfinite(self._wf[:, 0])]
        if len(finite):
            lo, hi = float(np.percentile(finite, 5)), float(finite.max())
            self.wf_img.setImage(np.nan_to_num(self._wf, nan=lo), autoLevels=False, levels=(lo, hi))
            t_end = self._wf_next / fs
            self.wf_img.setRect(QRectF(t_end - n_keep * SG_HOP / fs, 0, n_keep * SG_HOP / fs, fs / 2))

    def _cards(self, snap, x, fs, names) -> None:
        state = snap.get("state")
        big, small = self.cards["state"]
        if not snap.get("running"):
            big.setText("Stopped")
            small.setText(snap.get("error") or "")
        elif snap.get("loading_models"):
            big.setText("Loading models")
            small.setText("about 0.3 s")
        elif snap.get("calibrating", True) and state is None:
            big.setText("Calibrating")
            small.setText("first 2 s: spindle on, tool in the air")
        else:
            big.setText(STATE_NAMES.get(state, "–"))
            colour = self.tokens["ink2"] if state in (None, 0) else self.tokens[f"s{state}"]   # s0 grey is a fill colour
            big.setStyleSheet(f"font-size: 18px; font-weight: 600; color: {colour};")
            small.setText(f"vote: {snap.get('vote', '')}" + (f"   ({snap['dropped']} blocks not analysed)"
                                                            if snap.get("dropped") else ""))
        big, small = self.cards["models"]
        probs = snap.get("probs") or {}
        labels = snap.get("labels") or {}
        big.setText(snap.get("vote") or "–")
        small.setText("\n".join(f"{m}: {labels.get(m, '')}  P(chatter) {p[-1]:.2f}" for m, p in probs.items())
                      or "no models (needs 3 input pins)")
        big, small = self.cards["rms"]
        if x.shape[1]:
            r = np.sqrt(np.mean(x[:, -int(0.5 * fs):].astype(np.float64) ** 2, axis=1))
            big.setText(f"{r[min(1, len(r) - 1)]:.4g}")
            small.setText("  ".join(f"{n} {v:.4g}" for n, v in zip(names, r)) + "   (last 0.5 s)")
        big, small = self.cards["spindle"]
        st = snap.get("output_state")
        if st is None:
            big.setText("no output")
            small.setText("watch only")
        else:
            big.setText(f"{snap['rpm']:.0f} rpm")
            v = "  ".join(f"{x:.3f} V" for x in snap.get("volts", []))
            small.setText(f"{st.upper()}  {v}\ncontroller: {snap.get('controller') or 'constant speed'}")
        big, small = self.cards["rec"]
        if snap.get("recording"):
            big.setText(f"● {snap['rec_seconds']:.1f} s")
            small.setText(snap["recording"])
        else:
            big.setText("not recording")
            small.setText("")

    def _events(self, engine) -> None:
        for e in list(engine.events):
            if e["n"] > self._n_events:
                self.log.appendPlainText(f"{e['t']:8.2f} s  {e['who']:<10s} {e['what']}")
                self._n_events = e["n"]
