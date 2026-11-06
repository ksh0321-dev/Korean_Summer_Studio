"""박수 싱크 보정 창: 박수마다 '손이 처음 닿은 장면'을 고르면 보정값을 계산한다.

움직임이 가장 큰 프레임을 미리 골라 두지만 1~2프레임 빗나갈 수 있어서 사람이 확인·수정한다.
"""

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ks_studio.core.sync import INCONSISTENT_MS, contact_time, estimate_delay_ms
from ks_studio.services.sync import THUMB, Clap

MIN_CLAPS = 2


def _icon(rgb: np.ndarray) -> QIcon:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QIcon(QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()))


class CalibrationDialog(QDialog):
    def __init__(self, claps: list[Clap], camera_name: str, mic_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("박수 싱크 보정")
        self.resize(1280, 820)
        self._claps = claps
        self._lists: list[QListWidget] = []
        self.delay_ms = 0

        intro = QLabel(
            f"{camera_name} + {mic_name}\n"
            "박수마다 손이 처음 닿은 장면을 클릭해 고르세요. 움직임이 가장 큰 장면을 미리 골라 두었습니다 (선택 표시)."
        )
        intro.setWordWrap(True)
        rows = QWidget()
        rows_layout = QVBoxLayout(rows)
        for number, clap in enumerate(claps, 1):
            rows_layout.addWidget(QLabel(f"박수 {number}  (소리 {clap.sound_time:.3f}초)"))
            strip = QListWidget()
            strip.setViewMode(QListView.ViewMode.IconMode)
            strip.setFlow(QListView.Flow.LeftToRight)
            strip.setWrapping(False)
            strip.setIconSize(QSize(*THUMB))
            strip.setFixedHeight(THUMB[1] + 52)
            strip.setMovement(QListView.Movement.Static)
            for frame in clap.frames:
                item = QListWidgetItem(_icon(frame.image), f"{(frame.time - clap.sound_time) * 1000:+.0f}ms")
                strip.addItem(item)
            strip.setCurrentRow(clap.guess)
            strip.scrollToItem(strip.item(clap.guess), QListWidget.ScrollHint.PositionAtCenter)
            strip.currentRowChanged.connect(lambda _row: self._update_result())
            rows_layout.addWidget(strip)
            self._lists.append(strip)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows)

        self.result = QLabel()
        self.result.setWordWrap(True)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("저장")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.result)
        layout.addWidget(self.buttons)
        self._update_result()

    def _update_result(self) -> None:
        save = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        if len(self._claps) < MIN_CLAPS:
            self.result.setText(f"⚠ 박수를 {len(self._claps)}개만 찾았습니다. 손 옆면이 보이게, 크게 {MIN_CLAPS}번 이상 쳐서 다시 측정하세요.")
            save.setEnabled(False)
            return
        offsets = []
        for clap, strip in zip(self._claps, self._lists):
            index = max(0, strip.currentRow())
            touched = contact_time(clap.previous_scene_time(index), clap.frames[index].time)
            offsets.append(touched - clap.sound_time)
        self.delay_ms, spread = estimate_delay_ms(offsets)
        if self.delay_ms >= 0:
            meaning = f"소리가 화면보다 {self.delay_ms}ms 먼저 납니다 → 녹화할 때 오디오를 {self.delay_ms}ms 늦춥니다."
        else:
            meaning = f"소리가 화면보다 {-self.delay_ms}ms 늦습니다 → 녹화할 때 오디오를 {-self.delay_ms}ms 앞당깁니다."
        per_clap = " / ".join(f"{o * 1000:+.0f}" for o in offsets)
        text = f"보정값 {self.delay_ms:+d}ms  (박수별 {per_clap}ms)\n{meaning}"
        if spread > INCONSISTENT_MS:
            text += f"\n⚠ 박수끼리 {spread}ms 차이가 납니다. 고른 장면을 확인하거나 다시 측정하는 것을 권장합니다."
        self.result.setText(text)
        save.setEnabled(True)
