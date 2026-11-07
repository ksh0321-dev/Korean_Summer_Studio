"""워크스페이스(미디어 보관함): 이 프로젝트의 녹화본 + 가져온 미디어.

더블클릭 = 재생 헤드 위치에 넣기 · 타임라인으로 끌기 = 원하는 위치에 넣기
"+ 가져오기" 또는 탐색기에서 끌어다 놓기 = 미디어 추가 (복사/참조는 ⚙ 설정)
우클릭 = 파일 위치 열기(탐색기) · 워크스페이스에서 빼기
"🗑 빼기" = 목록에서 빼고 그 영상으로 만든 사본·캐시를 지운다 (원본 파일은 그대로 — 확인은 컷편집 페이지가 묻는다)
휴대폰 영상처럼 편집용 사본이 필요한 미디어는 상태를 한 줄 더 보여 주고, 사본이 준비될 때까지 넣을 수 없다.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QMimeData, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ks_studio.services.accel import open_folder
from ks_studio.services.sources import SourceCache
from ks_studio.ui.editor.timeline import MIME_MEDIA, format_frames

VIDEO_FILTER = "영상 (*.mp4 *.mov *.mkv *.m4v *.avi *.webm)"
MEDIA_FILTER = "영상·사진 (*.mp4 *.mov *.mkv *.m4v *.avi *.webm *.jpg *.jpeg *.png *.webp *.bmp)"  # 사진은 5초 영상으로


def show_in_explorer(path: Path) -> None:
    """탐색기(맥은 Finder)로 그 파일이 있는 폴더를 열고 파일을 골라 둔다 (파일이 없으면 폴더만)."""
    if path.exists() or path.parent.exists():
        open_folder(path, select=True)


@dataclass(frozen=True)
class BinItem:
    path: Path
    source: str  # project.json에 적힌 형식
    imported: bool  # False = 녹화본
    copying: bool = False
    note: str = ""  # 편집용 사본 상태 등 덧붙일 한 줄
    blocked: bool = False  # True면 타임라인에 넣을 수 없다 (사본 만드는 중·실패)


class _MediaList(QListWidget):
    def mimeTypes(self) -> list[str]:
        return [MIME_MEDIA]

    def mimeData(self, items) -> QMimeData:
        data = QMimeData()
        data.setData(MIME_MEDIA, str(items[0].data(Qt.ItemDataRole.UserRole)).encode())
        return data


class MediaBin(QWidget):
    import_requested = Signal(list)  # list[Path]
    add_requested = Signal(str)  # 원본 경로
    remove_requested = Signal(str)  # 원본 경로 — 워크스페이스에서 빼기

    def __init__(self, sources: SourceCache, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sources = sources
        self._items: list[BinItem] = []
        self.list = _MediaList()
        self.list.setIconSize(QSize(128, 72))
        self.list.setWordWrap(True)  # 편집용 사본 상태처럼 긴 줄도 잘리지 않게 (가로 스크롤 대신 줄바꿈)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setDragEnabled(True)
        self.list.setDragDropMode(QListWidget.DragDropMode.DragOnly)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.itemDoubleClicked.connect(self._on_double_click)
        self.list.currentRowChanged.connect(lambda _row: self._update_remove_button())
        self.list.customContextMenuRequested.connect(self._on_context_menu)
        self.setAcceptDrops(True)  # 탐색기에서 끌어온 파일 (목록은 받지 않으므로 여기로 온다)
        sources.ready.connect(lambda _path: self._refresh_icons())

        import_button = QPushButton("+ 가져오기")
        import_button.clicked.connect(self._choose_files)
        self.remove_button = QPushButton("🗑 빼기")
        self.remove_button.setToolTip("고른 영상을 목록에서 빼고, 그 영상으로 만든 편집용 사본·캐시를 지웁니다 (원본은 그대로)")
        self.remove_button.clicked.connect(self._remove_selected)
        header = QHBoxLayout()
        header.addWidget(QLabel("워크스페이스"))
        header.addStretch(1)
        header.addWidget(self.remove_button)
        header.addWidget(import_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(header)
        layout.addWidget(self.list)

    def show_items(self, items: list[BinItem], fps: int) -> None:
        self._items = items
        self.list.clear()
        for item in items:
            widget_item = QListWidgetItem(self._describe(item, fps))
            widget_item.setData(Qt.ItemDataRole.UserRole, str(item.path))
            if item.copying or item.blocked or not item.path.exists():
                widget_item.setFlags(widget_item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled)
            self.list.addItem(widget_item)
        self._refresh_icons()
        self._update_remove_button()

    def _selected(self) -> BinItem | None:
        row = self.list.currentRow()
        return self._items[row] if 0 <= row < len(self._items) else None

    def _update_remove_button(self) -> None:
        item = self._selected()
        self.remove_button.setEnabled(item is not None and not item.copying)

    def _remove_selected(self) -> None:
        item = self._selected()
        if item is not None and not item.copying:
            self.remove_requested.emit(str(item.path))

    def _describe(self, item: BinItem, fps: int) -> str:
        name = item.path.name
        if item.copying:
            return f"{name}\n복사 중…"
        if not item.path.exists():
            return f"{name}\n⚠ 파일 없음"
        info = self._sources.info(item.path)
        if info is None:
            return f"{name}\n⚠ 읽을 수 없음"
        parts = [format_frames(round(info.duration), 1, with_frames=False), f"{info.fps:.2f}".rstrip("0").rstrip(".") + "fps"]
        if abs(info.fps - fps) > 0.01:
            parts.append(f"→{fps}fps로 변환")
        if Path(item.source).is_absolute():
            parts.append("참조")
        text = f"{name}\n" + " · ".join(parts)
        return f"{text}\n{item.note}" if item.note else text

    def set_note(self, path: Path, note: str, blocked: bool, fps: int) -> None:
        """목록을 다시 만들지 않고 한 항목의 상태 줄만 바꾼다 (진행률처럼 자주 바뀌는 것)."""
        for row, item in enumerate(self._items):
            if item.path == path:
                self._items[row] = item = replace(item, note=note, blocked=blocked)
                widget_item = self.list.item(row)
                widget_item.setText(self._describe(item, fps))
                flags = widget_item.flags()
                if blocked:
                    flags &= ~Qt.ItemFlag.ItemIsDragEnabled
                elif not item.copying and item.path.exists():
                    flags |= Qt.ItemFlag.ItemIsDragEnabled
                widget_item.setFlags(flags)

    def _refresh_icons(self) -> None:
        for row, item in enumerate(self._items):
            if item.copying or not item.path.exists():
                continue
            thumbs = self._sources.thumbnails(item.path)
            if thumbs is None or len(thumbs.images) == 0:
                continue
            rgb = np.ascontiguousarray(thumbs.images[0])
            h, w, _ = rgb.shape
            image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
            self.list.item(row).setIcon(QIcon(QPixmap.fromImage(image)))

    def _on_double_click(self, widget_item: QListWidgetItem) -> None:
        if widget_item.flags() & Qt.ItemFlag.ItemIsDragEnabled:
            self.add_requested.emit(widget_item.data(Qt.ItemDataRole.UserRole))

    def _on_context_menu(self, pos) -> None:
        widget_item = self.list.itemAt(pos)
        if widget_item is None:
            return
        item = self._items[self.list.row(widget_item)]
        if item.copying:
            return
        menu = QMenu(self)
        menu.addAction("📂 파일 위치 열기", lambda: show_in_explorer(item.path))
        menu.addSeparator()
        menu.addAction("🗑 워크스페이스에서 빼기 (원본은 그대로)", lambda: self.remove_requested.emit(str(item.path)))
        menu.exec(self.list.mapToGlobal(pos))

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "미디어 가져오기", "", MEDIA_FILTER)
        if files:
            self.import_requested.emit([Path(f) for f in files])

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.import_requested.emit(paths)
        event.acceptProposedAction()
