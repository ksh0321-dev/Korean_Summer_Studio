"""카메라 프레임을 그리는 위젯. 미리보기는 좌우반전한다 (녹화본은 원본 유지).

위에 겹쳐 그리는 것: 9:16 세로 가이드, 가운데 큰 글자(카운트다운), 왼쪽 위 배지(● REC 00:12),
편집 화면에서 선택한 클립의 테두리(화면 비율 좌표), 자막(위아래로 끌어 자리를 바꾼다), 숏폼 안전 영역.
"""

from dataclasses import replace

import numpy as np
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtMultimedia import QVideoFrame, QVideoSink
from PySide6.QtWidgets import QWidget

# 세로(9:16) 숏폼에서 앱 화면(제목·버튼·설명)이 덮는 곳 — 화면 비율 (왼쪽, 위, 오른쪽, 아래). 쇼츠·릴스·틱톡을 대략 아우른다
SAFE_MARGINS = (0.06, 0.12, 0.16, 0.22)
CAPTION_BOTTOM_MAX = 0.85


class VideoView(QWidget):
    caption_moved = Signal(float)  # 자막을 끌어 놓은 자리: 아래 여백 (화면 높이 비율)

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
        self._caption: tuple | None = None  # (자막 글, CaptionStyle, 둘째 줄) — 내보낼 때 굽는 것과 같은 모양으로
        self._caption_rect: QRectF | None = None  # 마지막으로 그린 자막 자리 (끌기용)
        self._caption_drag: list | None = None  # [잡은 곳에서 자막 아래끝까지, 지금 아래 여백]
        self._caption_draggable = False
        self._safe = False

    def set_caption_draggable(self, on: bool) -> None:
        """자막을 위아래로 끌어 자리를 바꿀 수 있게 (편집 화면)."""
        self._caption_draggable = on
        self.setMouseTracking(on)

    def set_safe_zone(self, on: bool) -> None:
        """세로 영상에서 숏폼 앱 화면이 덮는 곳을 어둡게 보인다 (자막·글을 그 안쪽에 두게)."""
        self._safe = on
        self.update()

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

    def set_caption(self, text: str, style, second: str = "") -> None:
        caption = (text, style, second) if text or second else None
        if caption != self._caption:
            self._caption = caption
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
        if self._caption is not None:
            from ks_studio.services.captions import paint_caption  # 자막이 있을 때만 (녹화 화면은 필요 없다)

            text, style, second = self._caption
            if self._caption_drag is not None:  # 끄는 중: 놓기 전까지 화면에서만 옮겨 보인다
                style = replace(style, bottom=self._caption_drag[1])
            self._caption_rect = paint_caption(painter, QRectF(target), text, style, self._image.width(), self._image.height(), second)
        else:
            self._caption_rect = None
        if self._safe and self._image.height() > self._image.width():
            self._paint_safe_zone(painter, target)

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

    def _paint_safe_zone(self, painter: QPainter, target: QRect) -> None:
        left, top, right, bottom = SAFE_MARGINS
        w, h = target.width(), target.height()
        safe = QRectF(target.left() + w * left, target.top() + h * top, w * (1 - left - right), h * (1 - top - bottom))
        shade = QColor(255, 80, 80, 45)
        outer = QRectF(target)
        painter.fillRect(QRectF(outer.left(), outer.top(), outer.width(), safe.top() - outer.top()), shade)
        painter.fillRect(QRectF(outer.left(), safe.bottom(), outer.width(), outer.bottom() - safe.bottom()), shade)
        painter.fillRect(QRectF(outer.left(), safe.top(), safe.left() - outer.left(), safe.height()), shade)
        painter.fillRect(QRectF(safe.right(), safe.top(), outer.right() - safe.right(), safe.height()), shade)
        painter.setPen(QPen(QColor(255, 120, 120, 200), 1, Qt.PenStyle.DashLine))
        painter.drawRect(safe)
        painter.setFont(QFont(self.font().family(), 9))
        painter.drawText(safe.adjusted(4, 2, -4, -2), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, "숏폼 안전 영역")

    # ── 자막 끌기 ────────────────────────────────────────────────

    def caption_hit(self, pos) -> bool:
        """pos(위젯 좌표)가 끌 수 있는 자막 위인가."""
        return self._caption_draggable and self._caption_rect is not None and self._caption_rect.contains(pos)

    def _target_height(self) -> float:
        return max(1.0, float(self.image_rect().height()))

    def mousePressEvent(self, event) -> None:
        y = event.position().y()
        if (self._caption_draggable and event.button() == Qt.MouseButton.LeftButton and self._caption_rect is not None
                and self._caption_rect.contains(event.position())):
            self._caption_drag = [self._caption_rect.bottom() - y, self._caption[1].bottom]
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        y = event.position().y()
        if self._caption_drag is not None:
            target = self.image_rect()
            bottom_px = y + self._caption_drag[0]
            self._caption_drag[1] = max(0.0, min(CAPTION_BOTTOM_MAX, (target.bottom() - bottom_px) / self._target_height()))
            self.update()
            return
        if self._caption_draggable:
            over = self._caption_rect is not None and self._caption_rect.contains(event.position())
            self.setCursor(Qt.CursorShape.SizeVerCursor if over else Qt.CursorShape.ArrowCursor)
            self.setToolTip("위아래로 끌어 자막 자리 바꾸기" if over else "")
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._caption_drag is not None:
            bottom = round(self._caption_drag[1], 3)
            self._caption_drag = None
            if self._caption is not None:  # 받는 쪽이 저장할 때까지 놓은 자리에 보이게
                text, style, second = self._caption
                self._caption = (text, replace(style, bottom=bottom), second)
            self.update()
            self.caption_moved.emit(bottom)
            return
        super().mouseReleaseEvent(event)

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
