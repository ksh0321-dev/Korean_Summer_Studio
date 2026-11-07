"""'자막' 탭의 작은 창들: 찾아 바꾸기 · 용어집, 그리고 색 고르는 버튼."""

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ks_studio.core.glossary import Term
from ks_studio.core.subtitles import FIND_IN, replace_text


class ColorButton(QPushButton):
    """누르면 색을 고르는 버튼 (버튼 바탕이 그 색)."""

    color_changed = Signal(str)  # "#rrggbb"

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._color = "#ffffff"
        self.setFixedWidth(44)
        self.setToolTip(title)
        self.clicked.connect(self._pick)

    def color(self) -> str:
        return self._color

    def set_color(self, color: str) -> None:
        self._color = QColor(color).name()
        self.setStyleSheet(f"QPushButton {{ background: {self._color}; border: 1px solid #888; min-height: 18px; }}")

    def _pick(self) -> None:
        chosen = QColorDialog.getColor(QColor(self._color), self, self._title)
        if chosen.isValid() and chosen.name() != self._color:
            self.set_color(chosen.name())
            self.color_changed.emit(self._color)


class FindReplaceDialog(QDialog):
    """모든 자막 줄에서 찾아 바꾼다. 바꾸면 replaced(새 자막, 바꾼 곳 수)를 보낸다 (되돌리기 한 번)."""

    replaced = Signal(object, int)

    def __init__(self, subs_fn, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("찾아 바꾸기")
        self._subs_fn = subs_fn  # 지금 자막 (창을 띄운 뒤에 바뀌었을 수 있다)
        self.find_edit, self.replace_edit = QLineEdit(), QLineEdit()
        self.where = QComboBox()
        for key, label in FIND_IN.items():
            self.where.addItem(label, key)
        self.case = QCheckBox("대소문자 구별")
        self.result = QLabel()
        form = QFormLayout()
        form.addRow("찾을 말", self.find_edit)
        form.addRow("바꿀 말", self.replace_edit)
        form.addRow("찾을 곳", self.where)
        form.addRow("", self.case)
        run = QPushButton("모두 바꾸기")
        run.setDefault(True)
        run.clicked.connect(self._run)
        close = QPushButton("닫기")
        close.clicked.connect(self.close)
        buttons = QHBoxLayout()
        buttons.addWidget(self.result, 1)
        buttons.addWidget(run)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)

    def _run(self) -> None:
        subs, count = replace_text(self._subs_fn(), self.find_edit.text(), self.replace_edit.text(),
                                   self.where.currentData(), self.case.isChecked())
        self.result.setText(f"{count}곳을 바꿨습니다" if count else "찾지 못했습니다")
        if count:
            self.replaced.emit(subs, count)


class GlossaryDialog(QDialog):
    """용어집 고치기: 한국어 낱말 · 영어. 음성으로 자막을 만들 때 받아쓰기·번역 힌트로 쓴다."""

    def __init__(self, terms, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("용어집")
        self.resize(460, 420)
        note = QLabel("자주 나오는 낱말을 적어 두면 음성으로 자막을 만들 때 그 낱말로 알아듣고, 영어 번역도 그 말로 옮깁니다.\n"
                      "예: 곶감 → dried persimmon · 송편 → songpyeon")
        note.setWordWrap(True)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["한국어", "영어"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        for term in terms:
            self._add_row(term.korean, term.english)
        add = QPushButton("+ 낱말 추가")
        add.clicked.connect(lambda: self._add_row("", "", edit=True))
        remove = QPushButton("삭제")
        remove.clicked.connect(self._remove)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addWidget(self.table, 1)
        layout.addLayout(row)
        layout.addWidget(box)

    def _add_row(self, korean: str, english: str, edit: bool = False) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(korean))
        self.table.setItem(row, 1, QTableWidgetItem(english))
        if edit:
            self.table.editItem(self.table.item(row, 0))

    def _remove(self) -> None:
        for row in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)

    def terms(self) -> tuple[Term, ...]:
        items = ((self.table.item(r, 0), self.table.item(r, 1)) for r in range(self.table.rowCount()))
        terms = (Term(k.text().strip() if k else "", e.text().strip() if e else "") for k, e in items)
        return tuple(t for t in terms if t.korean)
