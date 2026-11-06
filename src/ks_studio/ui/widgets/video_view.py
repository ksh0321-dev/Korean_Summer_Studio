"""카메라 프레임을 그리는 위젯. 미리보기는 좌우반전한다 (녹화본은 원본 유지).

위에 겹쳐 그리는 것: 9:16 세로 가이드, 가운데 큰 글자(카운트다운), 왼쪽 위 배지(● REC 00:12),
편집 화면에서 선택한 클립의 테두리(화면 비율 좌표).
"""

import numpy as np
from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtMultimedia import QVideoFrame, QVideoSink
from PySide6.QtWidgets import QWidget


class VideoView(QWidget):
    def __init__(self, mirrored: bool = True, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sink = QVideoSink(self)
        self.sink.videoFrameChanged.connect(self._on_frame)
        self._mirrored = mirrored
        self._image = QImage()
        self._message = ""
        self._guide = False
        self._overlay = ""
        self._badge = ""
        self._box: QRectF | None = None  # 0~1 화면 비율 좌표

    def set_message(self, text: str) -> None:
        """프레임 대신 안내 문구를 띄운다."""
        self._image = QImage()
        self._message = text
        self.update()

    def set_guide(self, visible: bool) -> None:
        self._guide = visible
        self.update()

    def set_overlay(self, text: str) -> None:
        self._overlay = text
        self.update()

    def set_badge(self, text: str) -> None:
        self._badge = text
        self.update()

    def set_box(self, box: QRectF | None) -> None:
        self._box = box
        self.update()

    def image_rect(self) -> QRect:
        """그림이 실제로 그려지는 영역 (위젯 안에서 비율을 지켜 가운데)."""
        if self._image.isNull():
            return QRect()
        target = QRect(QPoint(0, 0), self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio))
        target.moveCenter(self.rect().center())
        return target

    def show_rgb(self, rgb) -> None:
        """렌더러가 만든 RGB 배열(H, W, 3)을 그린다 (편집 미리보기용). None이면 안내 문구로 돌아간다."""
        if rgb is None:
            self._image = QImage()
            self.update()
            return
        # 폭이 정렬 단위로 안 떨어지면(예: 세로 결과물의 미리보기 404px) PyAV가 줄 끝에 여백을 붙여 연속 메모리가 아니다
        rgb = np.ascontiguousarray(rgb)
        h, w, _ = rgb.shape
        self._rgb = rgb  # QImage가 배열 메모리를 그대로 쓰므로 살아 있게 붙잡아 둔다
        self._image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.update()

    def _on_frame(self, frame: QVideoFrame) -> None:
        self._image = frame.toImage()
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image.isNull():
            painter.setPen(QColor("#9a9aa3"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return

        target = self.image_rect()
        painter.save()
        if self._mirrored:
            painter.translate(self.width(), 0)
            painter.scale(-1, 1)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self._image)
        painter.restore()

        if self._guide:
            self._paint_guide(painter, target)
        if self._box is not None:
            box = QRectF(
                target.left() + self._box.left() * target.width(),
                target.top() + self._box.top() * target.height(),
                self._box.width() * target.width(),
                self._box.height() * target.height(),
            )
            painter.setPen(QPen(QColor("#f1c40f"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(box)
        if self._badge:
            badge = QRect(target.left() + 12, target.top() + 12, 170, 32)
            painter.fillRect(badge, QColor(0, 0, 0, 160))
            painter.setPen(QColor("#ff4d4d"))
            painter.setFont(QFont(self.font().family(), 12, QFont.Weight.Bold))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, self._badge)
        if self._overlay:
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont(self.font().family(), max(24, target.height() // 4), QFont.Weight.Bold))
            painter.drawText(target, Qt.AlignmentFlag.AlignCenter, self._overlay)

    def _paint_guide(self, painter: QPainter, target: QRect) -> None:
        """릴스용 9:16 영역만 밝게 두고 바깥을 어둡게 + 테두리."""
        width = round(target.height() * 9 / 16)
        crop = QRect(0, target.top(), width, target.height())
        crop.moveCenter(target.center())
        shade = QColor(0, 0, 0, 110)
        painter.fillRect(QRect(target.left(), target.top(), crop.left() - target.left(), target.height()), shade)
        painter.fillRect(QRect(crop.right() + 1, target.top(), target.right() - crop.right(), target.height()), shade)
        painter.setPen(QPen(QColor(255, 255, 255, 180), 1, Qt.PenStyle.DashLine))
        painter.drawRect(crop)
