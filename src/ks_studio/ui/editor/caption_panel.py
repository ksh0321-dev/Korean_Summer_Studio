"""컷편집 오른쪽 '자막' 탭: 타임라인 자막 줄 표 · 음성으로 만들기 · 모양 · SRT.

자막은 project.json에 타임라인 시각(초)으로 저장되고 (되돌리기 가능), 영상에는 굽지 않는다 — 내보내기에서 입힌다.
AI 얼굴 보정은 자막 없는 영상에 해야 하므로 자막 단계를 따로 두지 않고 컷편집에 넣었다.
"""

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ks_studio.core.subtitles import TASKS, Subtitles, add_cue, delete_cue, update_cue

COL_START, COL_END, COL_TEXT = range(3)
CURRENT_ROW = QColor(30, 110, 100)


def format_time(seconds: float) -> str:
    minutes, rest = divmod(max(0.0, seconds), 60)
    return f"{int(minutes)}:{rest:05.2f}"


def parse_time(text: str) -> float | None:
    """'1:05.20' 또는 '65.2' → 초. 못 읽으면 None."""
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return int(minutes) * 60 + float(seconds)
        return float(text)
    except ValueError:
        return None


class CaptionPanel(QWidget):
    edited = Signal(object)  # 새 Subtitles (받는 쪽이 프로젝트에 넣는다 — 되돌리기 가능)
    seek_requested = Signal(float)  # 초
    generate_requested = Signal(str)  # task ("translate"·"transcribe")
    srt_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._subs = Subtitles()
        self._now = 0.0
        self._current = -1
        self.task_combo = QComboBox()
        for key, label in TASKS.items():
            self.task_combo.addItem(label, key)
        self.generate_button = QPushButton("🎙 음성으로 자막 만들기")
        self.generate_button.setToolTip("타임라인 소리를 알아들어 자막 줄을 만듭니다 (지금 자막은 바뀝니다 — Ctrl+Z로 되돌리기)")
        self.generate_button.clicked.connect(lambda: self.generate_requested.emit(self.task_combo.currentData()))
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(2, 15)
        self.size_spin.setSuffix(" %")
        self.bottom_spin = QSpinBox()
        self.bottom_spin.setRange(0, 45)
        self.bottom_spin.setSuffix(" %")
        self.size_spin.valueChanged.connect(self._on_style)
        self.bottom_spin.valueChanged.connect(self._on_style)
        look = QFormLayout()
        look.addRow("자막 언어", self.task_combo)
        look.addRow("글자 크기", self.size_spin)
        look.addRow("아래 여백", self.bottom_spin)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["시작", "끝", "자막"])
        self.table.horizontalHeader().setSectionResizeMode(COL_TEXT, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellClicked.connect(lambda row, _col: self._seek_row(row))
        buttons = QHBoxLayout()
        self.add_button = QPushButton("+ 재생 헤드에")
        self.add_button.clicked.connect(self._add)
        self.delete_button = QPushButton("삭제")
        self.delete_button.clicked.connect(self._delete)
        self.start_button = QPushButton("시작=헤드")
        self.start_button.clicked.connect(lambda: self._set_time(COL_START))
        self.end_button = QPushButton("끝=헤드")
        self.end_button.clicked.connect(lambda: self._set_time(COL_END))
        for button in (self.add_button, self.delete_button, self.start_button, self.end_button):
            buttons.addWidget(button)
        self.srt_button = QPushButton("SRT 파일로 저장…")
        self.srt_button.clicked.connect(self.srt_requested.emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addLayout(look)
        layout.addLayout(buttons)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.srt_button)

    # ── 바깥에서 ─────────────────────────────────────────────────

    def set_subtitles(self, subs: Subtitles) -> None:
        if subs == self._subs:
            return
        self._subs = subs
        self.task_combo.setCurrentIndex(max(0, self.task_combo.findData(subs.task)))
        for spin, value in ((self.size_spin, subs.style.size), (self.bottom_spin, subs.style.bottom)):
            spin.blockSignals(True)
            spin.setValue(round(value * 100))
            spin.blockSignals(False)
        self._fill()

    def set_time(self, seconds: float) -> None:
        """재생 헤드 시각 — 지금 보이는 자막 줄을 칠한다."""
        self._now = seconds
        index = self._subs.at(seconds)
        row = -1 if index is None else index
        if row == self._current:
            return
        self.table.blockSignals(True)
        for r, color in ((self._current, QColor(0, 0, 0, 0)), (row, CURRENT_ROW)):
            if 0 <= r < self.table.rowCount():
                for col in range(3):
                    if self.table.item(r, col) is not None:
                        self.table.item(r, col).setBackground(color)
        self.table.blockSignals(False)
        self._current = row

    def select(self, index: int) -> None:
        if 0 <= index < self.table.rowCount():
            self.table.selectRow(index)
            self.table.scrollToItem(self.table.item(index, COL_TEXT))

    # ── 표 ──────────────────────────────────────────────────────

    def _fill(self, select: int | None = None) -> None:
        cues = self._subs.cues
        self.table.blockSignals(True)
        self.table.setRowCount(len(cues))
        for row, cue in enumerate(cues):
            for col, text in ((COL_START, format_time(cue.start)), (COL_END, format_time(cue.end)), (COL_TEXT, cue.text)):
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.blockSignals(False)
        self._current = -1
        self.set_time(self._now)
        if select is not None:
            self.select(select)

    def _emit(self, subs: Subtitles, select: int | None = None) -> None:
        self._subs = subs
        self._fill(select)
        self.edited.emit(subs)

    def _seek_row(self, row: int) -> None:
        if row < len(self._subs.cues):
            self.seek_requested.emit(self._subs.cues[row].start)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        row, col = item.row(), item.column()
        if col == COL_TEXT:
            subs, index = update_cue(self._subs, row, text=item.text())
        else:
            seconds = parse_time(item.text())
            if seconds is None:  # 못 읽는 시간은 되돌린다
                self._fill(row)
                return
            subs, index = update_cue(self._subs, row, **({"start": seconds} if col == COL_START else {"end": seconds}))
        self._emit(subs, index)

    def _selected_rows(self) -> list[int]:
        return sorted({index.row() for index in self.table.selectedIndexes()})

    def _add(self) -> None:
        subs, index = add_cue(self._subs, self._now)
        self._emit(subs, index)
        self.table.editItem(self.table.item(index, COL_TEXT))

    def _delete(self) -> None:
        subs = self._subs
        for row in reversed(self._selected_rows()):
            subs = delete_cue(subs, row)
        self._emit(subs)

    def _set_time(self, col: int) -> None:
        rows = self._selected_rows()
        if rows:
            subs, index = update_cue(self._subs, rows[0], **({"start": self._now} if col == COL_START else {"end": self._now}))
            self._emit(subs, index)

    def _on_style(self) -> None:
        style = replace(self._subs.style, size=self.size_spin.value() / 100, bottom=self.bottom_spin.value() / 100)
        self._emit(replace(self._subs, style=style))

    # ── 음성 인식 진행 ───────────────────────────────────────────

    def show_progress(self, text: str, ratio: float | None = None) -> None:
        self.status.setText(text)
        if ratio is None:
            self.progress.hide()
        else:
            self.progress.show()
            self.progress.setValue(round(ratio * 1000))
        self.generate_button.setEnabled(ratio is None)
