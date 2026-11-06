"""컷편집 단축키 설정 탭: 동작마다 키를 눌러 바꾼다. 겹치는 키는 알려 주고, 기본값으로 되돌릴 수 있다.

저장은 바꾼 것만 AppSettings.shortcuts에 (이름 → 키). 이름은 도구모음 버튼 글자에서 기호를 뺀 것 (shortcut_name).
"""

import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QKeySequenceEdit,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def shortcut_name(text: str) -> str:
    """'✂ 분할' → '분할', '뒤 자르기 ⇥' → '뒤 자르기' (글자·숫자·공백·/ 만 남긴다)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s/]", "", text)).strip()


def conflicts(keys: dict[str, str]) -> dict[str, list[str]]:
    """같은 키를 쓰는 동작들 (키 → 이름 목록, 둘 이상인 것만). 빈 키는 뺀다."""
    found: dict[str, list[str]] = {}
    for name, key in keys.items():
        if key:
            found.setdefault(QKeySequence(key).toString(), []).append(name)
    return {k: v for k, v in found.items() if len(v) > 1}


class ShortcutTab(QWidget):
    def __init__(self, catalog: list[tuple[str, str, tuple[str, ...]]], overrides: dict[str, list[str]],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._defaults = {name: (keys[0] if keys else "") for name, _tip, keys in catalog}
        self.table = QTableWidget(len(catalog), 3)
        self.table.setHorizontalHeaderLabels(["기능", "단축키 (눌러서 바꾸기 · Del로 지우기)", "기본"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.edits: dict[str, QKeySequenceEdit] = {}
        for row, (name, tip, keys) in enumerate(catalog):
            label = QTableWidgetItem(name)
            label.setToolTip(tip)
            label.setFlags(label.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, label)
            current = overrides.get(name, list(keys))
            edit = QKeySequenceEdit(QKeySequence(current[0] if current else ""))
            edit.setMaximumSequenceLength(1)
            edit.setClearButtonEnabled(True)
            edit.keySequenceChanged.connect(self._check)
            self.table.setCellWidget(row, 1, edit)
            self.edits[name] = edit
            default = QTableWidgetItem(" / ".join(QKeySequence(k).toString(QKeySequence.SequenceFormat.NativeText) for k in keys))
            default.setFlags(default.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, default)
        self.warning = QLabel()
        self.warning.setObjectName("statusMessage")
        self.warning.setProperty("level", "warning")
        reset = QPushButton("모두 기본값으로")
        reset.clicked.connect(self._reset)
        bottom = QHBoxLayout()
        bottom.addWidget(self.warning, 1)
        bottom.addWidget(reset)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("컷편집 화면 어디에서나 동작합니다. 바꾼 것은 이 PC의 개인 설정(user/settings.json)에 저장됩니다."))
        layout.addWidget(self.table, 1)
        layout.addLayout(bottom)
        self._check()

    def keys(self) -> dict[str, str]:
        return {name: edit.keySequence().toString() for name, edit in self.edits.items()}

    def problems(self) -> dict[str, list[str]]:
        return conflicts(self.keys())

    def overrides(self) -> dict[str, list[str]]:
        """기본값과 다른 것만 (기본이 둘인 동작은 첫 키를 그대로 두면 둘째도 그대로)."""
        out = {}
        for name, key in self.keys().items():
            if key != QKeySequence(self._defaults[name]).toString():
                out[name] = [key] if key else []
        return out

    def _check(self) -> None:
        problems = self.problems()
        self.warning.setText("⚠ 겹치는 키: " + " · ".join(f"{k} ({', '.join(v)})" for k, v in problems.items()) if problems else "")

    def _reset(self) -> None:
        for name, edit in self.edits.items():
            edit.setKeySequence(QKeySequence(self._defaults[name]))
