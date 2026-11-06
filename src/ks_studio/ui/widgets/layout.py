"""모든 단계 페이지가 공유하는 패널 배치 도구."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QWidget


def stage_layout(
    center: QWidget,
    left: QWidget | None = None,
    right: QWidget | None = None,
    bottom: QWidget | None = None,
) -> QSplitter:
    """CapCut식 배치: [왼쪽 | 가운데 | 오른쪽] 위, [아래] 밑. 크기는 드래그로 조절."""
    top = QSplitter(Qt.Orientation.Horizontal)
    sizes = []
    if left is not None:
        top.addWidget(left)
        sizes.append(280)
    top.addWidget(center)
    sizes.append(900)
    if right is not None:
        top.addWidget(right)
        sizes.append(300)
    top.setSizes(sizes)
    top.setStretchFactor(top.indexOf(center), 1)

    if bottom is None:
        return top

    outer = QSplitter(Qt.Orientation.Vertical)
    outer.addWidget(top)
    outer.addWidget(bottom)
    outer.setSizes([600, 260])
    outer.setStretchFactor(0, 1)
    return outer
