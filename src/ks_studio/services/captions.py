"""자막 그림: 글 → 테두리 있는 투명 그림(RGBA), 그리고 영상 프레임에 굽기.

자막 탭 미리보기와 내보내기(저해상도 미리보기·최종)가 같은 함수로 그려서 보이는 모습이 같다.
Qt의 QImage·QPainter로 그린다 (OpenCV 글꼴은 한글을 못 쓴다). QImage는 작업 스레드에서 써도 된다.
굽기는 AI 보정과 같이 자막 자리만 RGB로 바꿔 섞고, 자막이 닿지 않은 화소는 원래 YUV 값 그대로 둔다.
"""

from functools import lru_cache

import av
import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen

from ks_studio.app.fonts import family
from ks_studio.core.subtitles import CaptionStyle, Subtitles
from ks_studio.services.yuv import from_planes, planes

MAX_WIDTH = 0.9  # 자막 한 줄 최대 폭 = 영상 폭 × 0.9 (넘으면 줄바꿈)
LINE_GAP = 1.15


def _wrap(text: str, metrics: QFontMetricsF, width: float) -> list[str]:
    lines = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split(" "):
            trial = f"{line} {word}".strip()
            if line and metrics.horizontalAdvance(trial) > width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return [line for line in lines if line.strip()]


def _font(px: float, name: str = "") -> QFont:
    font = QFont(name or family())  # 기본은 화면과 같은 Pretendard (자막으로 구워도 되는 무료 글꼴)
    font.setPixelSize(max(1, round(px)))
    font.setBold(True)
    return font


BOX_PAD = 0.3  # 배경 상자 안쪽 여백 = 글자 크기 × 0.3


@lru_cache(maxsize=64)
def caption_image(text: str, style: CaptionStyle, frame_w: int, frame_h: int, second: str = "") -> tuple[np.ndarray, int, int] | None:
    """(RGBA 그림, 왼쪽 x, 위 y) — 영상 아래 가운데. 글이 없으면 None. 같은 자막은 다시 그리지 않는다.

    second = 둘째 줄 (두 줄 자막의 영어 번역): 첫째 줄 아래에 작게, 다른 색으로.
    style.box면 글 뒤에 반투명 상자 (밝은 화면에서도 읽히게).
    """
    px = max(8.0, min(frame_w, frame_h) * style.size)
    pad = px * BOX_PAD if style.box else 0.0
    blocks = []  # (글꼴, 줄들, 색)
    for words, size, color in ((text, px, style.color), (second, px * style.second_size, style.second_color)):
        if words.strip():
            font = _font(size, style.font)
            lines = _wrap(words.strip(), QFontMetricsF(font), frame_w * MAX_WIDTH - 2 * pad)
            if lines:
                blocks.append((font, lines, color))
    if not blocks:
        return None
    stroke = px * max(0.0, style.outline_width)
    edge = stroke + pad  # 그림 가장자리에서 글까지
    width = max(QFontMetricsF(font).horizontalAdvance(line) for font, lines, _ in blocks for line in lines) + 2 * edge + 4
    height = sum(QFontMetricsF(font).height() * LINE_GAP * len(lines) for font, lines, _ in blocks) + 2 * edge
    w, h = int(width) + 2, int(height) + 2
    image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if style.box:
        box = QColor(style.box_color)
        box.setAlphaF(max(0.0, min(1.0, style.box_opacity)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(box)
        painter.drawRoundedRect(QRectF(0, 0, w, h), px * 0.25, px * 0.25)
    pen = QPen(QColor(style.outline), stroke * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    top = edge
    for font, lines, color in blocks:
        metrics = QFontMetricsF(font)
        line_h = metrics.height() * LINE_GAP
        path = QPainterPath()
        for i, line in enumerate(lines):
            path.addText(QPointF((w - metrics.horizontalAdvance(line)) / 2, top + metrics.ascent() + i * line_h), font, line)
        if stroke > 0:
            painter.strokePath(path, pen)
        painter.fillPath(path, QColor(color))
        top += line_h * len(lines)
    painter.end()
    rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
    array = np.frombuffer(rgba.constBits(), np.uint8).reshape(h, rgba.bytesPerLine())[:, : w * 4].reshape(h, w, 4).copy()
    x = (frame_w - w) // 2
    y = round(frame_h * (1 - style.bottom)) - h
    return array, x, max(0, y)


def paint_caption(painter: QPainter, target: QRectF, text: str, style: CaptionStyle, frame_w: int, frame_h: int,
                  second: str = "") -> QRectF | None:
    """화면용: 영상(frame_w×frame_h)이 target 칸에 그려졌을 때 그 위에 자막을 같은 비율로. 그린 자리(화면 좌표)를 돌려준다."""
    drawn = caption_image(text, style, frame_w, frame_h, second)
    if drawn is None:
        return None
    array, x, y = drawn
    scale = target.width() / frame_w
    image = QImage(array.data, array.shape[1], array.shape[0], array.shape[1] * 4, QImage.Format.Format_RGBA8888)
    rect = QRectF(target.x() + x * scale, target.y() + y * scale, array.shape[1] * scale, array.shape[0] * scale)
    painter.drawImage(rect, image)
    return rect


def burn(frame: av.VideoFrame, overlay: tuple[np.ndarray, int, int]) -> av.VideoFrame:
    """yuv420p 프레임에 자막 그림을 섞는다. 자막 상자 안에서도 투명한 화소는 원래 값 그대로."""
    array, x, y = overlay
    if frame.format.name != "yuv420p":
        frame = frame.reformat(format="yuv420p")
    w, h = frame.width, frame.height
    ah, aw = array.shape[:2]
    x0, y0 = max(0, x) // 2 * 2, max(0, y) // 2 * 2  # 색 정보가 2×2 단위라 짝수 상자
    x1, y1 = min(w, (min(w, x + aw) + 1) // 2 * 2), min(h, (min(h, y + ah) + 1) // 2 * 2)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return frame
    alpha = np.zeros((y1 - y0, x1 - x0), np.float32)
    color = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    iy0, ix0, iy1, ix1 = max(y0, y), max(x0, x), min(y1, y + ah), min(x1, x + aw)  # 상자 안 자막 그림 자리
    alpha[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = array[iy0 - y : iy1 - y, ix0 - x : ix1 - x, 3] / 255
    color[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = array[iy0 - y : iy1 - y, ix0 - x : ix1 - x, :3]
    y_plane, u_plane, v_plane = planes(frame)  # 판을 따로 (높이 절반이 홀수여도 — services.yuv)
    cx0, cy0, cx1, cy1 = x0 // 2, y0 // 2, x1 // 2, y1 // 2
    roi = np.concatenate(
        [y_plane[y0:y1, x0:x1].ravel(), u_plane[cy0:cy1, cx0:cx1].ravel(), v_plane[cy0:cy1, cx0:cx1].ravel()]
    ).reshape((y1 - y0) * 3 // 2, x1 - x0)
    rgb = cv2.cvtColor(roi, cv2.COLOR_YUV2RGB_I420).astype(np.float32)
    mixed = np.clip(rgb + (color - rgb) * alpha[..., None] + 0.5, 0, 255).astype(np.uint8)
    back = cv2.cvtColor(mixed, cv2.COLOR_RGB2YUV_I420).ravel()
    n_y, n_c = (y1 - y0) * (x1 - x0), (cy1 - cy0) * (cx1 - cx0)
    touched = alpha > 0
    touched_c = touched.reshape(cy1 - cy0, 2, cx1 - cx0, 2).any(axis=(1, 3))
    y_plane[y0:y1, x0:x1][touched] = back[:n_y].reshape(y1 - y0, x1 - x0)[touched]
    u_plane[cy0:cy1, cx0:cx1][touched_c] = back[n_y : n_y + n_c].reshape(cy1 - cy0, cx1 - cx0)[touched_c]
    v_plane[cy0:cy1, cx0:cx1][touched_c] = back[n_y + n_c :].reshape(cy1 - cy0, cx1 - cx0)[touched_c]
    return from_planes(y_plane, u_plane, v_plane, frame)


class CaptionBurner:
    """내보내기 프레임마다: 그 시각의 자막을 굽는다 (pts = 출력 프레임 번호)."""

    def __init__(self, subs: Subtitles, size: tuple[int, int], fps: float) -> None:
        self._subs, self._size, self._fps = subs, size, fps

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        index = self._subs.at(frame.pts / self._fps)
        if index is None:
            return frame
        cue = self._subs.cues[index]
        overlay = caption_image(cue.text, self._subs.style, *self._size, cue.translation)
        return burn(frame, overlay) if overlay is not None else frame
