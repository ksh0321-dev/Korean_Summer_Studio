"""마이크 레벨 미터 (dBFS). 위쪽 CLIP 불 = 클리핑, 아래 '무음' = 신호 없음."""

import math
import time

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import QWidget

FLOOR_DB = -60.0
SILENCE_DB = -55.0
CLIP_PEAK = 0.99
CLIP_HOLD_S = 1.5
DECAY_DB_PER_TICK = 1.5  # 30ms마다 떨어지는 양


class LevelMeter(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(72)
        self._db = FLOOR_DB
        self._clip_until = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._decay)
        self._timer.start(30)

    def set_peak(self, peak: float) -> None:
        self._db = max(self._db, 20 * math.log10(max(peak, 1e-6)))
        if peak >= CLIP_PEAK:
            self._clip_until = time.monotonic() + CLIP_HOLD_S

    def _decay(self) -> None:
        self._db = max(FLOOR_DB, self._db - DECAY_DB_PER_TICK)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        bar = QRectF(self.rect()).adjusted(18, 30, -18, -30)
        painter.fillRect(bar, QColor("#111114"))

        ratio = min(1.0, max(0.0, (self._db - FLOOR_DB) / -FLOOR_DB))
        gradient = QLinearGradient(bar.bottomLeft(), bar.topLeft())
        gradient.setColorAt(0.0, QColor("#2ecc71"))
        gradient.setColorAt(0.78, QColor("#2ecc71"))  # -12 dB
        gradient.setColorAt(0.82, QColor("#f1c40f"))
        gradient.setColorAt(0.93, QColor("#f1c40f"))  # -3 dB
        gradient.setColorAt(0.96, QColor("#e74c3c"))
        filled = QRectF(bar.left(), bar.bottom() - bar.height() * ratio, bar.width(), bar.height() * ratio)
        painter.fillRect(filled, gradient)

        clipping = time.monotonic() < self._clip_until
        clip_box = QRectF(bar.left(), 4, bar.width(), 20)
        painter.fillRect(clip_box, QColor("#e74c3c" if clipping else "#2a2a31"))
        painter.setPen(QColor("#ffffff" if clipping else "#6f6f78"))
        painter.drawText(clip_box, Qt.AlignmentFlag.AlignCenter, "CLIP")

        painter.setPen(QColor("#9a9aa3"))
        label = "무음" if self._db <= SILENCE_DB else f"{self._db:.0f} dB"
        painter.drawText(QRectF(0, bar.bottom() + 6, self.width(), 20), Qt.AlignmentFlag.AlignCenter, label)
