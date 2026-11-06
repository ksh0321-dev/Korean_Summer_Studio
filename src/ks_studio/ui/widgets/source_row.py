"""영상 목록의 한 줄: 썸네일 + 파일 이름 · 찍은 날짜 · 길이 (· 상태) + 메모 칸 (AI 보정·자막·내보내기 탭이 같이 쓴다).

썸네일은 컷편집 워크스페이스와 같은 것 (SourceCache.thumbnails 첫 장). 아직 없으면 만들어지는 대로 채운다.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from ks_studio.services.sources import SourceCache, media_date


def clock(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


def facts(path: Path, sources: SourceCache) -> str:
    """'2026-11-05 11:45 · 0:44'. 모르는 값은 뺀다."""
    parts = []
    when: datetime | None = media_date(path)
    if when is not None:
        parts.append(when.strftime("%Y-%m-%d %H:%M"))
    info = sources.info(path)
    if info is not None and info.fps:
        parts.append(clock(info.duration))
    return " · ".join(parts)


THUMB_SIZE = (128, 72)


class SourceRow(QWidget):
    memo_changed = Signal(str, str)  # 원본, 메모

    def __init__(self, source: str, title: str, details: str, memo: str | None = None,
                 thumbnail: tuple[SourceCache, Path] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source = source
        self.thumb = QLabel()
        self.thumb.setFixedSize(*THUMB_SIZE)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setObjectName("sourceThumb")
        self._thumbnail = thumbnail
        if thumbnail is not None:
            thumbnail[0].ready.connect(self._load_thumbnail)  # 이 줄이 지워지면 연결도 같이 끊긴다
            self._load_thumbnail()
        name = QLabel(title)
        name.setObjectName("sourceName")
        info = QLabel(details)
        info.setObjectName("sourceMeta")
        info.setWordWrap(True)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.addWidget(self.thumb, 0, Qt.AlignmentFlag.AlignTop)
        layout = QVBoxLayout()
        layout.setSpacing(2)
        outer.addLayout(layout, 1)
        layout.addWidget(name)
        layout.addWidget(info)
        self.memo: QLineEdit | None = None
        if memo is not None:
            self.memo = QLineEdit(memo)
            self.memo.setPlaceholderText("메모")
            self.memo.editingFinished.connect(lambda: self.memo_changed.emit(self.source, self.memo.text()))
            layout.addWidget(self.memo)

    def _load_thumbnail(self, _key: str = "") -> None:
        if self._thumbnail is None:  # 이미 채웠다
            return
        sources, path = self._thumbnail
        if not path.exists():
            return
        thumbs = sources.thumbnails(path)  # 없으면 백그라운드에서 만들고 ready로 다시 부른다
        if thumbs is None or len(thumbs.images) == 0:
            return
        rgb = np.ascontiguousarray(thumbs.images[0])
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self.thumb.setPixmap(QPixmap.fromImage(image).scaled(*THUMB_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                                                             Qt.TransformationMode.SmoothTransformation))
        self._thumbnail = None  # 한 번 채우면 끝


def add_row(widget: QListWidget, row: SourceRow, source: str, enabled: bool = True) -> QListWidgetItem:
    """목록에 한 줄을 붙인다. 선택은 item의 UserRole(원본 문자열)로 찾는다."""
    item = QListWidgetItem()
    item.setData(Qt.ItemDataRole.UserRole, source)
    if not enabled:
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
    item.setSizeHint(row.sizeHint())
    widget.addItem(item)
    widget.setItemWidget(item, row)
    return item
