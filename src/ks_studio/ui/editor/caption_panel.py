"""컷편집 오른쪽 '자막' 탭: 타임라인 자막 줄 표 · 음성으로 만들기 · 나누기·합치기 · 찾아 바꾸기 · 용어집 · 모양 · SRT.

자막은 project.json에 타임라인 시각(초)으로 저장되고 (되돌리기 가능), 영상에는 굽지 않는다 — 내보내기에서 입힌다.
AI 얼굴 보정은 자막 없는 영상에 해야 하므로 자막 단계를 따로 두지 않고 컷편집에 넣었다.
"""

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFontDatabase
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ks_studio.app.paths import CAPTION_STYLES_DIR, GLOSSARY_PATH
from ks_studio.core.glossary import load_glossary, save_glossary
from ks_studio.core.presets import delete_preset, list_presets, valid_name
from ks_studio.core.subtitles import (
    ALIGNS,
    TASKS,
    CaptionStyle,
    Subtitles,
    add_cue,
    delete_cue,
    load_style,
    merge_cues,
    save_style,
    split_cue,
    update_cue,
)
from ks_studio.ui.editor.caption_dialogs import ColorButton, FindReplaceDialog, GlossaryDialog
from ks_studio.ui.widgets.video_view import CAPTION_BOTTOM_MAX

COL_START, COL_END, COL_TEXT, COL_TRANS = range(4)
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
    generate_requested = Signal(str)  # task (TASKS의 키)
    srt_requested = Signal()
    safe_zone_toggled = Signal(bool)  # 미리보기에 숏폼 안전 영역 보이기

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._subs = Subtitles()
        self._now = 0.0
        self._current = -1
        self.styles_dir = CAPTION_STYLES_DIR
        self.glossary_path = GLOSSARY_PATH
        self._find_dialog: FindReplaceDialog | None = None
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
        # 자동 줄 나누기 (음성으로 만들 때): 숏폼 세로 화면에서 1~2줄이 되게 짧게 나눈다
        self.line_spin = QSpinBox()
        self.line_spin.setRange(6, 40)
        self.line_spin.setSuffix(" 글자")
        self.line_spin.setToolTip("한 줄에 한글 이만큼 (영어는 글자 폭으로 환산). 세로 영상은 16 안팎이 알맞습니다")
        self.lines_spin = QSpinBox()
        self.lines_spin.setRange(1, 3)
        self.lines_spin.setSuffix(" 줄")
        self.lines_spin.setToolTip("자막 한 번에 첫째 줄이 이 줄 수까지 (두 줄 자막의 번역 줄은 따로)")
        self.line_spin.valueChanged.connect(self._on_layout)
        self.lines_spin.valueChanged.connect(self._on_layout)
        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("한 줄"))
        split_row.addWidget(self.line_spin)
        split_row.addWidget(QLabel("· 최대"))
        split_row.addWidget(self.lines_spin)
        split_row.addStretch(1)
        self.glossary_button = QPushButton("용어집…")
        self.glossary_button.setToolTip("자주 나오는 낱말(곶감 → dried persimmon)을 적어 두면 받아쓰기·번역이 그 말로 나옵니다")
        self.glossary_button.clicked.connect(self._edit_glossary)
        task_row = QHBoxLayout()
        task_row.addWidget(self.task_combo, 1)
        task_row.addWidget(self.glossary_button)
        make = QFormLayout()
        make.addRow("자막 언어", task_row)
        make.addRow("자동 줄 나누기", split_row)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["시작", "끝", "자막", "번역"])
        self.table.horizontalHeader().setSectionResizeMode(COL_TEXT, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(COL_TRANS, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellClicked.connect(lambda row, _col: self._seek_row(row))
        self.add_button = QPushButton("+ 재생 헤드에")
        self.add_button.clicked.connect(self._add)
        self.delete_button = QPushButton("삭제")
        self.delete_button.clicked.connect(self._delete)
        self.start_button = QPushButton("시작=헤드")
        self.start_button.clicked.connect(lambda: self._set_time(COL_START))
        self.end_button = QPushButton("끝=헤드")
        self.end_button.clicked.connect(lambda: self._set_time(COL_END))
        self.split_button = QPushButton("헤드에서 나누기")
        self.split_button.setToolTip("재생 헤드 시각에서 지금 자막 줄을 둘로 나눕니다 (단어 시간이 있으면 단어 단위로)")
        self.split_button.clicked.connect(self._split)
        self.merge_button = QPushButton("다음 줄과 합치기")
        self.merge_button.clicked.connect(self._merge)
        self.find_button = QPushButton("찾아 바꾸기…")
        self.find_button.clicked.connect(self._find_replace)
        rows = QVBoxLayout()
        for group in ((self.add_button, self.delete_button, self.start_button, self.end_button),
                      (self.split_button, self.merge_button, self.find_button)):
            row = QHBoxLayout()
            for button in group:
                row.addWidget(button)
            rows.addLayout(row)
        self.srt_button = QPushButton("SRT 파일로 저장…")
        self.srt_button.clicked.connect(self.srt_requested.emit)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addLayout(make)
        layout.addLayout(rows)
        layout.addWidget(self.table, 1)
        layout.addWidget(self._style_section())
        layout.addWidget(self.srt_button)

    # ── 모양 ────────────────────────────────────────────────────

    def _style_section(self) -> QWidget:
        """모양: 프리셋 · 글꼴 · 첫째/둘째 줄 크기와 색 · 테두리 · 배경 상자 · 자리. 접어 둘 수 있다."""
        self.size_spin = QSpinBox()
        self.size_spin.setRange(2, 15)
        self.size_spin.setSuffix(" %")
        self.size_spin.setToolTip("첫째 줄 글자 크기 (영상 짧은 변 대비)")
        self.color_button = ColorButton("첫째 줄 글자 색")
        self.second_spin = QSpinBox()
        self.second_spin.setRange(40, 120)
        self.second_spin.setSuffix(" %")
        self.second_spin.setToolTip("둘째 줄(번역) 글자 크기 (첫째 줄 대비)")
        self.second_color_button = ColorButton("둘째 줄(번역) 글자 색")
        self.font_combo = QComboBox()
        self.font_combo.addItem("기본 (Pretendard)", "")
        for name in sorted(QFontDatabase.families(), key=str.casefold):
            self.font_combo.addItem(name, name)
        self.font_combo.setToolTip("내보낼 때도 이 글꼴로 굽습니다. 다른 PC에서 열면 그 PC에 같은 글꼴이 있어야 합니다")
        self.outline_button = ColorButton("테두리 색")
        self.outline_spin = QSpinBox()
        self.outline_spin.setRange(0, 40)
        self.outline_spin.setSuffix(" %")
        self.outline_spin.setToolTip("테두리 두께 (글자 크기 대비, 0 = 테두리 없음)")
        self.box_check = QCheckBox("배경 상자")
        self.box_check.setToolTip("글 뒤에 반투명 상자 — 밝은 화면에서도 읽힙니다")
        self.box_button = ColorButton("배경 상자 색")
        self.box_spin = QSpinBox()
        self.box_spin.setRange(10, 100)
        self.box_spin.setSuffix(" %")
        self.box_spin.setToolTip("배경 상자 진하기")
        self.shadow_spin = QSpinBox()
        self.shadow_spin.setRange(0, 100)
        self.shadow_spin.setSuffix(" %")
        self.shadow_spin.setToolTip("그림자 진하기 (0 = 없음)")
        self.bold_check = QCheckBox("굵게")
        self.italic_check = QCheckBox("기울임")
        self.align_combo = QComboBox()
        for key, label in ALIGNS.items():
            self.align_combo.addItem(label, key)
        self.align_combo.setToolTip("두 줄 이상일 때 줄 맞춤")
        self.bold_check.toggled.connect(self._on_style)
        self.italic_check.toggled.connect(self._on_style)
        self.align_combo.currentIndexChanged.connect(self._on_style)
        self.bottom_spin = QSpinBox()
        self.bottom_spin.setRange(0, round(CAPTION_BOTTOM_MAX * 100))
        self.bottom_spin.setSuffix(" %")
        self.bottom_spin.setToolTip("아래 여백 (영상 높이 대비). 미리보기에서 자막을 위아래로 끌어도 바뀝니다")
        self.safe_check = QCheckBox("숏폼 안전 영역 보기")
        self.safe_check.setToolTip("세로 영상에서 쇼츠·릴스·틱톡 화면(제목·버튼)이 덮는 곳을 미리보기에 붉게 보입니다")
        self.safe_check.toggled.connect(self.safe_zone_toggled.emit)
        for spin in (self.size_spin, self.second_spin, self.outline_spin, self.box_spin, self.bottom_spin, self.shadow_spin):
            spin.valueChanged.connect(self._on_style)
        for button in (self.color_button, self.second_color_button, self.outline_button, self.box_button):
            button.color_changed.connect(self._on_style)
        self.font_combo.currentIndexChanged.connect(self._on_style)
        self.box_check.toggled.connect(self._on_style)
        self.style_combo = QComboBox()
        self.style_combo.activated.connect(self._load_style)
        save = QPushButton("저장")
        save.clicked.connect(self._save_style)
        self.style_delete = QPushButton("삭제")
        self.style_delete.clicked.connect(self._delete_style)

        def row(*widgets) -> QHBoxLayout:
            line = QHBoxLayout()
            for w in widgets:
                line.addWidget(w) if isinstance(w, QWidget) else line.addWidget(QLabel(w))
            line.addStretch(1)
            return line

        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(0, 0, 0, 0)
        presets = QHBoxLayout()
        presets.addWidget(self.style_combo, 1)
        presets.addWidget(save)
        presets.addWidget(self.style_delete)
        form.addRow("프리셋", presets)
        form.addRow("글꼴", self.font_combo)
        form.addRow("첫째 줄", row(self.size_spin, self.color_button))
        form.addRow("둘째 줄", row(self.second_spin, self.second_color_button))
        form.addRow("글씨", row(self.bold_check, self.italic_check, "줄 맞춤", self.align_combo))
        form.addRow("테두리", row(self.outline_spin, self.outline_button))
        form.addRow("그림자", row(self.shadow_spin))
        form.addRow(self.box_check, row(self.box_spin, self.box_button))
        form.addRow("아래 여백", row(self.bottom_spin, self.safe_check))
        toggle = QToolButton()
        toggle.setText("모양")
        toggle.setCheckable(True)
        toggle.setChecked(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.DownArrow)
        toggle.toggled.connect(lambda on: (body.setVisible(on), toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)))
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toggle)
        layout.addWidget(body)
        self._fill_styles()
        return section

    def _show_style(self, style: CaptionStyle) -> None:
        widgets = (self.size_spin, self.second_spin, self.outline_spin, self.box_spin, self.bottom_spin, self.font_combo,
                   self.box_check, self.color_button, self.second_color_button, self.outline_button, self.box_button,
                   self.shadow_spin, self.bold_check, self.italic_check, self.align_combo)
        for w in widgets:
            w.blockSignals(True)
        self.size_spin.setValue(round(style.size * 100))
        self.second_spin.setValue(round(style.second_size * 100))
        self.outline_spin.setValue(round(style.outline_width * 100))
        self.box_spin.setValue(round(style.box_opacity * 100))
        self.bottom_spin.setValue(round(style.bottom * 100))
        self.shadow_spin.setValue(round(style.shadow * 100))
        self.bold_check.setChecked(style.bold)
        self.italic_check.setChecked(style.italic)
        self.align_combo.setCurrentIndex(max(0, self.align_combo.findData(style.align)))
        index = self.font_combo.findData(style.font)
        if index < 0:  # 이 PC에 없는 글꼴 — 이름은 남겨 둔다 (그리는 건 비슷한 글꼴로)
            self.font_combo.addItem(f"{style.font} (이 PC에 없음)", style.font)
            index = self.font_combo.count() - 1
        self.font_combo.setCurrentIndex(index)
        self.box_check.setChecked(style.box)
        self.color_button.set_color(style.color)
        self.second_color_button.set_color(style.second_color)
        self.outline_button.set_color(style.outline)
        self.box_button.set_color(style.box_color)
        for w in widgets:
            w.blockSignals(False)

    def _on_style(self, *_args) -> None:
        style = replace(
            self._subs.style,
            size=self.size_spin.value() / 100, bottom=self.bottom_spin.value() / 100, color=self.color_button.color(),
            second_size=self.second_spin.value() / 100, second_color=self.second_color_button.color(),
            font=self.font_combo.currentData(), outline=self.outline_button.color(), outline_width=self.outline_spin.value() / 100,
            box=self.box_check.isChecked(), box_color=self.box_button.color(), box_opacity=self.box_spin.value() / 100,
            shadow=self.shadow_spin.value() / 100, bold=self.bold_check.isChecked(), italic=self.italic_check.isChecked(),
            align=self.align_combo.currentData(),
        )
        if style != self._subs.style:
            self._emit(replace(self._subs, style=style))

    def showEvent(self, event) -> None:
        """탭을 열 때 모양 목록을 다시 읽는다 (자막·글자 탭이 같은 목록을 쓴다 — 다른 탭에서 저장한 것도 보이게)."""
        super().showEvent(event)
        self._fill_styles(self.style_combo.currentData())

    def _fill_styles(self, current: str | None = None) -> None:
        self.style_combo.clear()
        self.style_combo.addItem("불러올 모양 고르기…", None)
        for name in list_presets(self.styles_dir):
            self.style_combo.addItem(name, name)
        self.style_combo.setCurrentIndex(max(0, self.style_combo.findData(current)) if current else 0)
        self.style_delete.setEnabled(current is not None)

    def _load_style(self, index: int) -> None:
        name = self.style_combo.itemData(index)
        self.style_delete.setEnabled(name is not None)
        if name is None:
            return
        try:
            style = load_style(self.styles_dir, name)
        except (OSError, ValueError):
            self.show_progress(f"⚠ 모양 '{name}'을 읽지 못했습니다")
            return
        self._emit(replace(self._subs, style=style))  # 자리(아래 여백)까지 프리셋 그대로

    def _save_style(self) -> None:
        name, ok = QInputDialog.getText(self, "자막 모양 저장", "이름 (앱 전체에서 같이 씁니다):", text=self.style_combo.currentData() or "")
        if not ok:
            return
        if not valid_name(name):
            self.show_progress('⚠ 이름이 비었거나 쓸 수 없는 글자(\\ / : * ? " < > |)가 있습니다')
            return
        save_style(self.styles_dir, name, self._subs.style)
        self._fill_styles(name.strip())
        self.show_progress(f"모양 '{name.strip()}'을 저장했습니다")

    def _delete_style(self) -> None:
        name = self.style_combo.currentData()
        if name is not None:
            delete_preset(self.styles_dir, name)
            self._fill_styles()

    # ── 나누기 · 합치기 · 찾아 바꾸기 · 용어집 ──────────────────────

    def _target_row(self) -> int | None:
        """고른 줄 (없으면 재생 헤드에 보이는 줄)."""
        rows = self._selected_rows()
        if rows:
            return rows[0]
        return self._subs.at(self._now)

    def _split(self) -> None:
        index = self._subs.at(self._now)
        result = split_cue(self._subs, index, self._now) if index is not None else None
        if result is None:
            self.show_progress("나눌 수 없습니다: 재생 헤드를 자막 줄 안쪽(양 끝에서 0.3초 넘게)에 두세요")
            return
        self._emit(*result)

    def _merge(self) -> None:
        index = self._target_row()
        result = merge_cues(self._subs, index) if index is not None else None
        if result is None:
            self.show_progress("합칠 수 없습니다: 다음 줄이 있는 줄을 고르세요")
            return
        self._emit(*result)

    def _find_replace(self) -> None:
        if self._find_dialog is None:
            self._find_dialog = FindReplaceDialog(lambda: self._subs, self)
            self._find_dialog.replaced.connect(lambda subs, _count: self._emit(subs))
        self._find_dialog.show()
        self._find_dialog.raise_()

    def _edit_glossary(self) -> None:
        dialog = GlossaryDialog(load_glossary(self.glossary_path), self)
        if dialog.exec():
            terms = dialog.terms()
            save_glossary(self.glossary_path, terms)
            self.show_progress(f"용어집 {len(terms)}낱말 — 다음에 음성으로 만들 때 씁니다")

    # ── 바깥에서 ─────────────────────────────────────────────────

    def set_subtitles(self, subs: Subtitles) -> None:
        if subs == self._subs:
            return
        self._subs = subs
        self.task_combo.setCurrentIndex(max(0, self.task_combo.findData(subs.task)))
        for spin, value in ((self.line_spin, subs.line_chars), (self.lines_spin, subs.max_lines)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._show_style(subs.style)
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
                for col in range(self.table.columnCount()):
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
            for col, text in ((COL_START, format_time(cue.start)), (COL_END, format_time(cue.end)), (COL_TEXT, cue.text),
                              (COL_TRANS, cue.translation)):
                self.table.setItem(row, col, QTableWidgetItem(text))
        # 번역 칸은 두 줄 자막이거나 번역이 있는 줄이 있을 때만
        self.table.setColumnHidden(COL_TRANS, self._subs.task != "bilingual" and not any(c.translation for c in cues))
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
        elif col == COL_TRANS:
            subs, index = update_cue(self._subs, row, translation=item.text())
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

    def _on_layout(self) -> None:
        """자동 줄 나누기 설정 — 다음에 음성으로 만들 때 쓴다."""
        self._emit(replace(self._subs, line_chars=self.line_spin.value(), max_lines=self.lines_spin.value()))

    # ── 음성 인식 진행 ───────────────────────────────────────────

    def show_progress(self, text: str, ratio: float | None = None) -> None:
        self.status.setText(text)
        if ratio is None:
            self.progress.hide()
        else:
            self.progress.show()
            self.progress.setValue(round(ratio * 1000))
        self.generate_button.setEnabled(ratio is None)
