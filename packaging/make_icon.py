"""Draw the app icon (256 px PNG for Linux, multi-size .ico for Windows) with Qt, so no image tool is needed.
Dark tile, a stable trace that breaks into a red chatter burst: the app's subject in one glance."""
import math
import sys
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen

OUT = Path(__file__).parent


def draw(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    s = size / 256
    p.setBrush(QColor("#1C1F23"))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(8 * s, 8 * s, 240 * s, 240 * s, 28 * s, 28 * s)
    p.setPen(QPen(QColor("#31363C"), max(1.0, 2 * s)))
    p.drawLine(QPointF(28 * s, 128 * s), QPointF(228 * s, 128 * s))
    for colour, x0, x1, amp, cyc in (("#4DA3FF", 28, 128, 14, 5), ("#FF6B5E", 128, 228, 62, 9)):
        path = QPainterPath()
        for i in range(201):
            x = x0 + (x1 - x0) * i / 200
            env = 1.0 if colour == "#4DA3FF" else math.sin(math.pi * i / 200) ** 0.6
            y = 128 - amp * env * math.sin(2 * math.pi * cyc * i / 200)
            (path.moveTo if i == 0 else path.lineTo)(QPointF(x * s, y * s))
        p.setPen(QPen(QColor(colour), max(1.5, 7 * s), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
    p.end()
    return img


if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    draw(256).save(str(OUT / "icon.png"))
    ok = draw(256).save(str(OUT / "icon.ico"), "ICO")
    print("icon.png", (OUT / "icon.png").stat().st_size, "bytes; icon.ico", "ok" if ok else "FAILED (Qt has no ICO writer)")
