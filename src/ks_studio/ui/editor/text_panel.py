"""컷편집 오른쪽 '글자' 탭: 자막과 따로 화면 아무 곳에 넣는 제목·강조 문구 (숏폼 첫 3초 문구 등).

글자마다 시작·끝(타임라인 초)·글·자리·모양. 자리는 미리보기에서 글자를 끌어도 바뀐다. 내보낼 때 굽는다.
모양은 자막과 같은 칸(글꼴·크기·색·테두리·배경 상자)이고, 모양 프리셋도 자막과 같은 목록을 쓴다.
"""

import re
from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ks_studio.app.paths import CAPTION_STYLES_DIR
from ks_studio.core.presets import delete_preset, list_presets, valid_name
from ks_studio.core.project import TextItem
from ks_studio.core.subtitles import ALIGNS, MIN_CUE, STYLE_TEMPLATES, CaptionStyle, load_style, save_style
from ks_studio.ui.editor.caption_dialogs import ColorButton
from ks_studio.ui.editor.caption_panel import format_time, parse_time

NEW_TEXT_SECONDS = 3.0
COL_START, COL_END, COL_TEXT = range(3)
NEWLINE_MARK = " ⏎ "  # 표 칸에서 줄바꿈 표시 (칸 안에선 Enter를 칠 수 없다 — "여러 줄…" 버튼에서)
TEMPLATE = "template:"  # 프리셋 목록에서 기본 제공 모양의 표시


def to_cell(text: str) -> str:
    return text.replace("\n", NEWLINE_MARK)


def from_cell(text: str) -> str:
    return "\n".join(part.strip() for part in re.split(r"\s*⏎\s*", text))


class TextPanel(QWidget):
    edited = Signal(object)  # 새 글자 목록 (tuple[TextItem]) — 받는 쪽이 프로젝트에 넣는다 (되돌리기 가능)
    seek_requested = Signal(float)
    notice = Signal(str)  # 상태 표시줄 알림 (모양 저장 등)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._texts: tuple[TextItem, ...] = ()
        self._now = 0.0
        self.styles_dir = CAPTION_STYLES_DIR  # 자막 모양 프리셋과 같은 목록
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["시작", "끝", "글자"])
        self.table.horizontalHeader().setSectionResizeMode(COL_TEXT, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellClicked.connect(lambda row, _c: self.seek_requested.emit(self._texts[row].start) if row < len(self._texts) else None)
        self.table.itemSelectionChanged.connect(self._show_style)
        add = QPushButton("+ 재생 헤드에")
        add.clicked.connect(self._add)
        delete = QPushButton("삭제")
        delete.clicked.connect(self._delete)
        start = QPushButton("시작=헤드")
        start.clicked.connect(lambda: self._set_time("start"))
        end = QPushButton("끝=헤드")
        end.clicked.connect(lambda: self._set_time("end"))
        self.multiline = QPushButton("여러 줄…")
        self.multiline.setToolTip("고른 글자를 여러 줄로 고칩니다 (Enter로 줄바꿈). 표에선 ⏎로 보입니다")
        self.multiline.clicked.connect(self._edit_multiline)
        buttons = QHBoxLayout()
        for button in (add, delete, start, end, self.multiline):
            buttons.addWidget(button)
        hint = QLabel("미리보기에서 글자를 끌어 옮길 수 있습니다. 글자는 내보낼 때 영상에 구워집니다.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons)
        layout.addWidget(self.table, 1)
        layout.addWidget(self._style_section())
        layout.addWidget(hint)
        self._show_style()

    def _style_section(self) -> QWidget:
        """고른 글자의 모양·자리: 프리셋 · 글꼴 · 크기·색 · 테두리 · 배경 상자 · 자리. 자막 '모양'처럼 접어 둘 수 있다."""
        self.size_spin = QSpinBox()
        self.size_spin.setRange(2, 30)
        self.size_spin.setSuffix(" %")
        self.size_spin.setToolTip("글자 크기 (영상 짧은 변 대비)")
        self.color_button = ColorButton("글자 색")
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
        self.shadow_spin.setToolTip("그림자 진하기 (0 = 없음). 테두리 없이도 밝은 화면에서 글이 읽힙니다")
        self.bold_check = QCheckBox("굵게")
        self.italic_check = QCheckBox("기울임")
        self.align_combo = QComboBox()
        for key, label in ALIGNS.items():
            self.align_combo.addItem(label, key)
        self.align_combo.setToolTip("여러 줄일 때 줄 맞춤")
        self.x_spin, self.y_spin = QSpinBox(), QSpinBox()
        for spin in (self.x_spin, self.y_spin):
            spin.setRange(0, 100)
            spin.setSuffix(" %")
        self.fade_in_spin, self.fade_out_spin = QSpinBox(), QSpinBox()
        for spin, tip in ((self.fade_in_spin, "나타날 때 이 시간 동안 서서히 진해집니다 (0 = 바로)"),
                          (self.fade_out_spin, "사라질 때 이 시간 동안 서서히 옅어집니다 (0 = 바로)")):
            spin.setRange(0, 5000)
            spin.setSingleStep(100)
            spin.setSuffix(" ms")
            spin.setToolTip(tip)
        for spin in (self.size_spin, self.outline_spin, self.box_spin, self.shadow_spin, self.x_spin, self.y_spin,
                     self.fade_in_spin, self.fade_out_spin):
            spin.valueChanged.connect(self._on_style)
        for button in (self.color_button, self.outline_button, self.box_button):
            button.color_changed.connect(self._on_style)
        self.font_combo.currentIndexChanged.connect(self._on_style)
        self.align_combo.currentIndexChanged.connect(self._on_style)
        for check in (self.box_check, self.bold_check, self.italic_check):
            check.toggled.connect(self._on_style)
        self.style_combo = QComboBox()
        self.style_combo.setToolTip("맨 위는 기본 제공 모양, 그 아래는 저장한 모양입니다 (자막과 같은 목록)")
        self.style_combo.activated.connect(self._load_style)
        self.style_save = QPushButton("저장")
        self.style_save.clicked.connect(self._save_style)
        self.style_delete = QPushButton("삭제")
        self.style_delete.clicked.connect(self._delete_style)
        self.apply_all = QPushButton("이 모양을 모든 글자에")
        self.apply_all.setToolTip("고른 글자의 모양(글꼴·크기·색·테두리·배경 상자)을 다른 글자에도 입힙니다. 자리는 그대로")
        self.apply_all.clicked.connect(self._apply_to_all)

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
        presets.addWidget(self.style_save)
        presets.addWidget(self.style_delete)
        form.addRow("프리셋", presets)
        form.addRow("글꼴", self.font_combo)
        form.addRow("크기", row(self.size_spin, self.color_button))
        form.addRow("글씨", row(self.bold_check, self.italic_check, "줄 맞춤", self.align_combo))
        form.addRow("테두리", row(self.outline_spin, self.outline_button))
        form.addRow("그림자", row(self.shadow_spin))
        form.addRow(self.box_check, row(self.box_spin, self.box_button))
        form.addRow("자리", row("가로", self.x_spin, "세로", self.y_spin))
        form.addRow("페이드", row("인", self.fade_in_spin, "아웃", self.fade_out_spin))
        form.addRow("", self.apply_all)
        toggle = QToolButton()
        toggle.setText("모양")
        toggle.setCheckable(True)
        toggle.setChecked(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.DownArrow)
        toggle.toggled.connect(lambda on: (body.setVisible(on), toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)))
        self._style_widgets = (self.size_spin, self.color_button, self.font_combo, self.outline_spin, self.outline_button,
                               self.box_check, self.box_spin, self.box_button, self.x_spin, self.y_spin,
                               self.shadow_spin, self.bold_check, self.italic_check, self.align_combo,
                               self.fade_in_spin, self.fade_out_spin, self.style_combo, self.style_save, self.apply_all,
                               self.multiline)
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toggle)
        layout.addWidget(body)
        self._fill_styles()
        return section

    # ── 바깥에서 ─────────────────────────────────────────────────

    def set_texts(self, texts: tuple[TextItem, ...]) -> None:
        if texts == self._texts:
            return
        selected = self._selected()
        selected_id = self._texts[selected].id if selected is not None and selected < len(self._texts) else None
        self._texts = texts
        self._fill(next((i for i, t in enumerate(texts) if t.id == selected_id), None))

    def select(self, text_id: str) -> None:
        """타임라인 글자 띠에서 누른 글자를 표에서 고른다."""
        row = next((i for i, t in enumerate(self._texts) if t.id == text_id), None)
        if row is not None:
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0))

    def set_time(self, seconds: float) -> None:
        self._now = seconds

    # ── 표 ──────────────────────────────────────────────────────

    def _fill(self, select: int | None = None) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._texts))
        for row, item in enumerate(self._texts):
            for col, text in ((COL_START, format_time(item.start)), (COL_END, format_time(item.end)), (COL_TEXT, to_cell(item.text))):
                self.table.setItem(row, col, QTableWidgetItem(text))
        self.table.blockSignals(False)
        if select is not None:
            self.table.selectRow(select)
        self._show_style()

    def _selected(self) -> int | None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        return rows[0] if rows else None

    def _emit(self, texts, select_id: str | None = None) -> None:
        texts = tuple(sorted(texts, key=lambda t: (t.start, t.end)))
        self._texts = texts
        self._fill(next((i for i, t in enumerate(texts) if t.id == select_id), None))
        self.edited.emit(texts)

    def _replace(self, row: int, **changes) -> None:
        item = replace(self._texts[row], **changes)
        if item.end - item.start < MIN_CUE:
            item = replace(item, end=item.start + MIN_CUE)
        self._emit((*self._texts[:row], item, *self._texts[row + 1 :]), item.id)

    def _on_item_changed(self, cell: QTableWidgetItem) -> None:
        row, col = cell.row(), cell.column()
        if col == COL_TEXT:
            self._replace(row, text=from_cell(cell.text()))
            return
        seconds = parse_time(cell.text())
        if seconds is None:
            self._fill(row)
            return
        self._replace(row, **{"start" if col == COL_START else "end": max(0.0, seconds)})

    def _add(self) -> None:
        item = TextItem(round(self._now, 2), round(self._now + NEW_TEXT_SECONDS, 2), "글자를 입력하세요")
        self._emit((*self._texts, item), item.id)
        row = self._selected()
        if row is not None:
            self.table.editItem(self.table.item(row, COL_TEXT))

    def _delete(self) -> None:
        row = self._selected()
        if row is not None:
            self._emit(self._texts[:row] + self._texts[row + 1 :])

    def _set_time(self, side: str) -> None:
        row = self._selected()
        if row is not None:
            self._replace(row, **{side: round(self._now, 3)})

    # ── 모양 · 자리 ──────────────────────────────────────────────

    def _show_style(self) -> None:
        row = self._selected()
        item = self._texts[row] if row is not None and row < len(self._texts) else None
        for w in self._style_widgets:
            w.setEnabled(item is not None)
            w.blockSignals(True)
        self.apply_all.setEnabled(item is not None and len(self._texts) > 1)
        self.style_delete.setEnabled(self._preset_name() is not None)
        if item is not None:
            style = item.style
            self.size_spin.setValue(round(style.size * 100))
            self.color_button.set_color(style.color)
            index = self.font_combo.findData(style.font)
            if index < 0:  # 이 PC에 없는 글꼴 — 이름은 남겨 둔다 (그리는 건 비슷한 글꼴로)
                self.font_combo.addItem(f"{style.font} (이 PC에 없음)", style.font)
                index = self.font_combo.count() - 1
            self.font_combo.setCurrentIndex(index)
            self.outline_spin.setValue(round(style.outline_width * 100))
            self.outline_button.set_color(style.outline)
            self.box_check.setChecked(style.box)
            self.box_spin.setValue(round(style.box_opacity * 100))
            self.box_button.set_color(style.box_color)
            self.shadow_spin.setValue(round(style.shadow * 100))
            self.bold_check.setChecked(style.bold)
            self.italic_check.setChecked(style.italic)
            self.align_combo.setCurrentIndex(max(0, self.align_combo.findData(style.align)))
            self.x_spin.setValue(round(item.x * 100))
            self.y_spin.setValue(round(item.y * 100))
            self.fade_in_spin.setValue(item.fade_in_ms)
            self.fade_out_spin.setValue(item.fade_out_ms)
        for w in self._style_widgets:
            w.blockSignals(False)

    def _style_from_controls(self, style: CaptionStyle) -> CaptionStyle:
        return replace(
            style, size=self.size_spin.value() / 100, color=self.color_button.color(), font=self.font_combo.currentData(),
            outline=self.outline_button.color(), outline_width=self.outline_spin.value() / 100,
            box=self.box_check.isChecked(), box_color=self.box_button.color(), box_opacity=self.box_spin.value() / 100,
            shadow=self.shadow_spin.value() / 100, bold=self.bold_check.isChecked(), italic=self.italic_check.isChecked(),
            align=self.align_combo.currentData(),
        )

    def _on_style(self, *_args) -> None:
        row = self._selected()
        if row is None:
            return
        item = self._texts[row]
        self._replace(row, style=self._style_from_controls(item.style), x=self.x_spin.value() / 100, y=self.y_spin.value() / 100,
                      fade_in_ms=self.fade_in_spin.value(), fade_out_ms=self.fade_out_spin.value())

    def _edit_multiline(self) -> None:
        row = self._selected()
        if row is None:
            return
        text, ok = QInputDialog.getMultiLineText(self, "글자 고치기", "글 (Enter로 줄바꿈):", self._texts[row].text)
        if ok and text.strip():
            self._replace(row, text=text.strip("\n"))

    def _apply_to_all(self) -> None:
        """고른 글자의 모양을 모든 글자에 (자리·시간·글은 그대로). 한 번의 편집 (Ctrl+Z 한 번)."""
        row = self._selected()
        if row is None:
            return
        style = self._texts[row].style
        self._emit(tuple(replace(t, style=style) for t in self._texts), self._texts[row].id)
        self.notice.emit(f"모양을 글자 {len(self._texts)}개 모두에 입혔습니다")

    # ── 모양 프리셋 (자막과 같은 목록) ──────────────────────────────

    def showEvent(self, event) -> None:
        """탭을 열 때 모양 목록을 다시 읽는다 (자막·글자 탭이 같은 목록을 쓴다 — 다른 탭에서 저장한 것도 보이게)."""
        super().showEvent(event)
        self._fill_styles(self.style_combo.currentData())  # 기본 제공 모양을 골라 둔 채여도 그대로

    def _fill_styles(self, current: str | None = None) -> None:
        self.style_combo.clear()
        self.style_combo.addItem("불러올 모양 고르기…", None)
        for name in STYLE_TEMPLATES:
            self.style_combo.addItem(f"★ {name}", TEMPLATE + name)
        self.style_combo.insertSeparator(self.style_combo.count())
        for name in list_presets(self.styles_dir):
            self.style_combo.addItem(name, name)
        self.style_combo.setCurrentIndex(max(0, self.style_combo.findData(current)) if current else 0)
        self.style_delete.setEnabled(self._preset_name() is not None)

    def _preset_name(self) -> str | None:
        """지금 고른 '저장한 모양' 이름 (기본 제공 모양·아무것도 아니면 None — 지울 수 없다)."""
        data = self.style_combo.currentData()
        return data if data and not data.startswith(TEMPLATE) else None

    def _load_style(self, index: int) -> None:
        name = self.style_combo.itemData(index)
        self.style_delete.setEnabled(self._preset_name() is not None)
        row = self._selected()
        if name is None or row is None:
            return
        if name.startswith(TEMPLATE):
            style = STYLE_TEMPLATES[name.removeprefix(TEMPLATE)]
        else:
            try:
                style = load_style(self.styles_dir, name)
            except (OSError, ValueError):
                self.notice.emit(f"⚠ 모양 '{name}'을 읽지 못했습니다")
                return
        self._replace(row, style=replace(style, bottom=0.0))  # 자리는 글자 것 그대로 (아래 여백은 자막에서만 쓴다)

    def _save_style(self) -> None:
        row = self._selected()
        if row is None:
            return
        name, ok = QInputDialog.getText(self, "모양 저장", "이름 (자막·글자 모두에서 같이 씁니다):", text=self._preset_name() or "")
        if not ok:
            return
        if not valid_name(name):
            self.notice.emit('⚠ 이름이 비었거나 쓸 수 없는 글자(\\ / : * ? " < > |)가 있습니다')
            return
        # 자막에서 불러와도 자막이 화면 맨 아래에 붙지 않게, 아래 여백은 자막 기본값으로 저장한다
        save_style(self.styles_dir, name, replace(self._texts[row].style, bottom=CaptionStyle().bottom))
        self._fill_styles(name.strip())
        self.notice.emit(f"모양 '{name.strip()}'을 저장했습니다")

    def _delete_style(self) -> None:
        name = self._preset_name()
        if name is not None:
            delete_preset(self.styles_dir, name)
            self._fill_styles()

    def move_text(self, text_id: str, x: float, y: float) -> None:
        """미리보기에서 끌어 놓은 자리."""
        row = next((i for i, t in enumerate(self._texts) if t.id == text_id), None)
        if row is not None:
            self.table.selectRow(row)
            self._replace(row, x=round(x, 3), y=round(y, 3))
