"""자막 단계: 영상 하나를 골라 자막을 만든다 — 음성으로 자동(기본 영어, 한국어는 번역) 만든 뒤 표에서 고친다.

자막은 영상에 굽지 않고 <프로젝트>/subtitles/subtitles.json에 들고 있다가 내보내기 탭에서 입힌다.
다른 단계와 독립: 녹화본·컷편집 결과·AI 보정본·가져온 영상 아무거나.
"""

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ks_studio.core.subtitles import (
    TASKS,
    SubtitleState,
    Subtitles,
    add_cue,
    delete_cue,
    load_subtitles,
    save_subtitles,
    to_srt,
    update_cue,
)
from ks_studio.core.workspace import Workspace
from ks_studio.services.sources import SourceCache
from ks_studio.services.speech import speech_unavailable_reason
from ks_studio.ui.editor.media_bin import VIDEO_FILTER
from ks_studio.ui.subtitle.subtitle_player import SubtitlePlayer
from ks_studio.ui.widgets.layout import stage_layout
from ks_studio.ui.widgets.source_row import SourceRow, add_row, facts
from ks_studio.ui.widgets.status_bar import ERROR, INFO

SAVE_DELAY_MS = 500
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


class SubtitlePage(QWidget):
    message = Signal(str, str)
    indicator = Signal(str, str, str)

    def __init__(self, sources: SourceCache, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sources = sources
        self._workspace: Workspace | None = None
        self._state = SubtitleState()
        self._source: str | None = None
        self._unavailable = speech_unavailable_reason()
        self.worker = None
        if self._unavailable is None:
            from ks_studio.services.speech import SpeechWorker

            self.worker = SpeechWorker(self)
            self.worker.status.connect(self._on_status)
            self.worker.progress.connect(self._on_progress)
            self.worker.done.connect(self._on_done)
            self.worker.failed.connect(self._on_failed)

        # 왼쪽: 영상 목록
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_select)
        import_button = QPushButton("+ 가져오기")
        import_button.clicked.connect(self._import)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel("영상"))
        header.addStretch(1)
        header.addWidget(import_button)
        left_layout.addLayout(header)
        left_layout.addWidget(self.list, 1)

        # 가운데: 영상 + 자막
        self.player = SubtitlePlayer()
        self.player.position_changed.connect(self._on_position)

        # 오른쪽: 자동 만들기 · 모양 · SRT
        auto = QGroupBox("음성으로 자막 만들기")
        auto_layout = QVBoxLayout(auto)
        self.task_combo = QComboBox()
        for key, label in TASKS.items():
            self.task_combo.addItem(label, key)
        self.generate_button = QPushButton("🎙 자막 만들기")
        self.generate_button.clicked.connect(self._generate)
        self.auto_progress = QProgressBar()
        self.auto_progress.setRange(0, 1000)
        self.auto_progress.hide()
        self.auto_status = QLabel(self._unavailable or "말소리를 알아들어 자막 줄을 만듭니다. 만든 뒤 아래 표에서 고치세요.")
        self.auto_status.setWordWrap(True)
        auto_layout.addWidget(QLabel("자막 언어"))
        auto_layout.addWidget(self.task_combo)
        auto_layout.addWidget(self.generate_button)
        auto_layout.addWidget(self.auto_progress)
        auto_layout.addWidget(self.auto_status)
        look = QGroupBox("자막 모양")
        look_form = QFormLayout(look)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(2, 15)
        self.size_spin.setSuffix(" %")
        self.size_spin.setToolTip("글자 크기 = 영상 짧은 변 × %")
        self.bottom_spin = QSpinBox()
        self.bottom_spin.setRange(0, 45)
        self.bottom_spin.setSuffix(" %")
        self.bottom_spin.setToolTip("아래 여백 = 영상 높이 × %")
        self.size_spin.valueChanged.connect(self._on_style)
        self.bottom_spin.valueChanged.connect(self._on_style)
        look_form.addRow("글자 크기", self.size_spin)
        look_form.addRow("아래 여백", self.bottom_spin)
        self.srt_button = QPushButton("SRT 파일로 저장…")
        self.srt_button.clicked.connect(self._save_srt)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(auto)
        right_layout.addWidget(look)
        right_layout.addWidget(self.srt_button)
        right_layout.addStretch(1)

        # 아래: 자막 줄 표
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["시작", "끝", "자막"])
        self.table.horizontalHeader().setSectionResizeMode(COL_TEXT, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellClicked.connect(self._on_cell_clicked)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("+ 지금 위치에 줄 추가")
        self.add_button.clicked.connect(self._add)
        self.delete_button = QPushButton("줄 삭제")
        self.delete_button.clicked.connect(self._delete)
        self.start_button = QPushButton("시작 = 지금")
        self.start_button.clicked.connect(lambda: self._set_time(COL_START))
        self.end_button = QPushButton("끝 = 지금")
        self.end_button.clicked.connect(lambda: self._set_time(COL_END))
        for button in (self.add_button, self.delete_button, self.start_button, self.end_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(QLabel("줄을 누르면 그 자리로 갑니다 · 두 번 눌러 고치기"))
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addLayout(buttons)
        bottom_layout.addWidget(self.table, 1)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self._save)
        self._current_row = -1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(stage_layout(left=left, center=self.player, right=right, bottom=bottom))
        self._update_controls()

    # ── 페이지 수명 ──────────────────────────────────────────────

    def activate(self, workspace: Workspace) -> None:
        if workspace != self._workspace:
            self._save()
            self._workspace = workspace
            self._state = load_subtitles(workspace.subtitles_file)
            self._source = None
            self.player.release()
        self._fill_list()
        self._update_controls()

    def deactivate(self) -> None:
        self.player.player.pause()
        self._save()

    def _save(self) -> None:
        self._save_timer.stop()
        if self._workspace is not None and (self._state.docs or self._state.imported):
            save_subtitles(self._workspace.subtitles_file, self._state)

    # ── 영상 ────────────────────────────────────────────────────

    def _fill_list(self) -> None:
        current = self._source
        self.list.blockSignals(True)
        self.list.clear()
        for source in self._workspace.videos(self._state.imported):
            path = self._workspace.resolve(source)
            notes = [facts(path, self._sources)] if path.exists() else ["⚠ 파일 없음"]
            count = len(self._state.for_source(source).cues)
            if count:
                notes.append(f"자막 {count}줄")
            row = SourceRow(source, path.name, " · ".join(n for n in notes if n), thumbnail=(self._sources, path))
            add_row(self.list, row, source, enabled=path.exists())
        rows = [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
        self.list.setCurrentRow(rows.index(current) if current in rows else -1)
        self.list.blockSignals(False)

    def _import(self) -> None:
        if self._workspace is None:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "자막을 넣을 영상 가져오기", "", VIDEO_FILTER)
        added = [self._workspace.to_source(Path(f)) for f in files]
        if added:
            self._state = replace(self._state, imported=tuple(dict.fromkeys((*self._state.imported, *added))))
            self._save()
            self._source = added[0]
            self._fill_list()
            self._on_select(self.list.currentRow())

    def _on_select(self, row: int) -> None:
        if row < 0 or self._workspace is None:
            return
        self._source = self.list.item(row).data(Qt.ItemDataRole.UserRole)
        subs = self._subs()
        self.task_combo.setCurrentIndex(max(0, self.task_combo.findData(subs.task)))
        for spin, value in ((self.size_spin, subs.style.size), (self.bottom_spin, subs.style.bottom)):
            spin.blockSignals(True)
            spin.setValue(round(value * 100))
            spin.blockSignals(False)
        self.player.open(self._media())
        self.player.set_subtitles(subs)
        self._fill_table()
        self._update_controls()

    def _media(self) -> Path:
        """틀고 알아들을 파일: 편집용 사본이 있으면 사본 (휴대폰 회전·색이 맞다)."""
        return self._sources.media(self._workspace.resolve(self._source))

    def _subs(self) -> Subtitles:
        return self._state.for_source(self._source) if self._source else Subtitles()

    def _set_subs(self, subs: Subtitles, select: int | None = None) -> None:
        self._state = replace(self._state, docs={**self._state.docs, self._source: subs})
        self._save_timer.start()
        self.player.set_subtitles(subs)
        self._fill_table(select)

    # ── 표 ──────────────────────────────────────────────────────

    def _fill_table(self, select: int | None = None) -> None:
        cues = self._subs().cues
        self.table.blockSignals(True)
        self.table.setRowCount(len(cues))
        for row, cue in enumerate(cues):
            for col, text in ((COL_START, format_time(cue.start)), (COL_END, format_time(cue.end)), (COL_TEXT, cue.text)):
                self.table.setItem(row, col, QTableWidgetItem(text))
        self._current_row = -1
        self.table.blockSignals(False)
        if select is not None and 0 <= select < len(cues):
            self.table.selectRow(select)
            self.table.scrollToItem(self.table.item(select, COL_TEXT))
        self._on_position(self.player.seconds())

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        row, col = item.row(), item.column()
        if col == COL_TEXT:
            subs, index = update_cue(self._subs(), row, text=item.text())
        else:
            seconds = parse_time(item.text())
            if seconds is None:  # 못 읽는 시간은 되돌린다
                self._fill_table(row)
                return
            subs, index = update_cue(self._subs(), row, **({"start": seconds} if col == COL_START else {"end": seconds}))
        self._set_subs(subs, index)

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        cues = self._subs().cues
        if row < len(cues):
            self.player.seek(cues[row].start)

    def _selected_rows(self) -> list[int]:
        return sorted({index.row() for index in self.table.selectedIndexes()})

    def _add(self) -> None:
        subs, index = add_cue(self._subs(), self.player.seconds())
        self._set_subs(subs, index)
        self.table.editItem(self.table.item(index, COL_TEXT))

    def _delete(self) -> None:
        subs = self._subs()
        for row in reversed(self._selected_rows()):
            subs = delete_cue(subs, row)
        self._set_subs(subs)

    def _set_time(self, col: int) -> None:
        rows = self._selected_rows()
        if not rows:
            return
        now = self.player.seconds()
        subs, index = update_cue(self._subs(), rows[0], **({"start": now} if col == COL_START else {"end": now}))
        self._set_subs(subs, index)

    def _on_position(self, seconds: float) -> None:
        """지금 보이는 자막 줄을 표에서 칠한다."""
        index = self._subs().at(seconds)
        row = -1 if index is None else index
        if row == self._current_row:
            return
        self.table.blockSignals(True)
        for r, color in ((self._current_row, QColor(0, 0, 0, 0)), (row, CURRENT_ROW)):
            if 0 <= r < self.table.rowCount():
                for col in range(3):
                    if self.table.item(r, col) is not None:
                        self.table.item(r, col).setBackground(color)
        self.table.blockSignals(False)
        self._current_row = row

    # ── 모양 ────────────────────────────────────────────────────

    def _on_style(self) -> None:
        if self._source is None:
            return
        style = replace(self._subs().style, size=self.size_spin.value() / 100, bottom=self.bottom_spin.value() / 100)
        self._set_subs(replace(self._subs(), style=style))

    def _save_srt(self) -> None:
        name = Path(self._source).stem + ".srt"
        path, _ = QFileDialog.getSaveFileName(self, "SRT로 저장", str(self._workspace.exports_dir / name), "자막 (*.srt)")
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(to_srt(self._subs()), encoding="utf-8")
            self.message.emit(f"SRT 저장: {Path(path).name}", INFO)

    # ── 음성으로 만들기 ──────────────────────────────────────────

    def _generate(self) -> None:
        if self._subs().cues:
            answer = QMessageBox.question(self, "자막 만들기", "지금 자막을 지우고 새로 만들까요?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        task = self.task_combo.currentData()
        self._set_subs(replace(self._subs(), task=task))
        self.auto_progress.setValue(0)
        self.auto_progress.show()
        self.worker.request(self._media(), task)
        self._update_controls()

    def _media_matches(self, media: Path) -> bool:
        return self._source is not None and media == self._media()

    def _on_status(self, media: Path, text: str) -> None:
        if self._media_matches(media):
            self.auto_status.setText(text)

    def _on_progress(self, media: Path, ratio: float) -> None:
        if self._media_matches(media):
            self.auto_progress.setValue(round(ratio * 1000))
            self.auto_status.setText(f"말 알아듣는 중… {int(ratio * 100)}%")

    def _on_done(self, media: Path, cues: list) -> None:
        self.auto_progress.hide()
        self._update_controls()
        if not self._media_matches(media):
            return
        self._set_subs(replace(self._subs(), cues=tuple(cues)), 0 if cues else None)
        self.auto_status.setText(f"자막 {len(cues)}줄을 만들었습니다. 표에서 고치세요 (음악만 나오는 곳에 지어낸 줄이 있으면 지우세요)")
        self.message.emit(f"자막 {len(cues)}줄 만듦: {Path(self._source).name}", INFO)
        self._fill_list()

    def _on_failed(self, media: Path, reason: str) -> None:
        self.auto_progress.hide()
        self.auto_status.setText(f"⚠ 실패: {reason}")
        self.message.emit(f"⚠ 자막 만들기 실패: {reason}", ERROR)
        self._update_controls()

    def _update_controls(self) -> None:
        ready = self._source is not None
        busy = self.worker is not None and self.worker.busy_with is not None
        self.generate_button.setEnabled(ready and self.worker is not None and not busy)
        for widget in (self.add_button, self.delete_button, self.start_button, self.end_button, self.size_spin,
                       self.bottom_spin, self.srt_button, self.task_combo, self.table):
            widget.setEnabled(ready)
