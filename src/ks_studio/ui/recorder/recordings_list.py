from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget

from ks_studio.services.media import duration_seconds


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = round(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


class RecordingsList(QWidget):
    """프로젝트의 녹화본 목록 (마무리 중인 것은 맨 위에 표시). 더블클릭 = 컷편집 타임라인에 넣기."""

    open_requested = Signal(Path)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._folder: Path | None = None
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self._on_double_click)
        open_button = QPushButton("폴더 열기")
        open_button.clicked.connect(self._open_folder)

        header = QHBoxLayout()
        header.addWidget(QLabel("녹화본  (더블클릭 = 컷편집에 넣기)"))
        header.addStretch(1)
        header.addWidget(open_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.list)

    def show_recordings(self, folder: Path, finished: list[Path], pending_names: list[str]) -> None:
        self._folder = folder
        self.list.clear()
        for name in pending_names:
            self.list.addItem(f"⏳ {name}  — 마무리 중…")
        for path in finished:
            size_mb = path.stat().st_size / 1_000_000
            item = QListWidgetItem(f"{path.name}    {_format_duration(duration_seconds(path))}    {size_mb:.0f} MB")
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.list.addItem(item)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self.open_requested.emit(Path(path))

    def _open_folder(self) -> None:
        if self._folder is None:
            return
        self._folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._folder)))
