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


def _font(px: float, name: str = "", bold: bool = True, italic: bool = False) -> QFont:
    font = QFont(name or family())  # 기본은 화면과 같은 Pretendard (자막으로 구워도 되는 무료 글꼴)
    font.setPixelSize(max(1, round(px)))
    font.setBold(bold)
    font.setItalic(italic)
    return font


BOX_PAD = 0.3  # 배경 상자 안쪽 여백 = 글자 크기 × 0.3
SHADOW_OFFSET = 0.06  # 그림자: 글자 크기 × 0.06만큼 오른쪽 아래로
SHADOW_BLUR = 0.07  # 그림자 번짐 (가우시안 시그마) = 글자 크기 × 0.07
SHADOW_SPREAD = 0.1  # 번지기 전에 글 모양을 글자 크기 × 0.1 두께로 키운다 (가는 글씨는 그림자가 너무 옅었다)


def _shadow_image(paths: list, pen: QPen | None, w: int, h: int, px: float, strength: float) -> QImage:
    """글(테두리 포함) 모양을 검게 칠해 번지게 한 그림자 (RGBA). 글과 같은 크기의 그림 위 좌표."""
    mask = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    mask.fill(Qt.GlobalColor.transparent)
    painter = QPainter(mask)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.translate(px * SHADOW_OFFSET, px * SHADOW_OFFSET)
    spread = QPen(QColor("#000000"), px * SHADOW_SPREAD, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    for path in paths:
        if pen is not None:
            painter.strokePath(path, QPen(QColor("#000000"), pen.widthF(), Qt.PenStyle.SolidLine,
                                          Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.strokePath(path, spread)
        painter.fillPath(path, QColor("#000000"))
    painter.end()
    rgba = mask.convertToFormat(QImage.Format.Format_RGBA8888)
    alpha = np.frombuffer(rgba.constBits(), np.uint8).reshape(h, rgba.bytesPerLine())[:, : w * 4].reshape(h, w, 4)[..., 3]
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), max(0.5, px * SHADOW_BLUR)) * max(0.0, min(1.0, strength))
    shadow = np.zeros((h, w, 4), np.uint8)
    shadow[..., 3] = np.clip(alpha + 0.5, 0, 255).astype(np.uint8)
    image = QImage(shadow.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return image.copy()  # numpy 배열이 사라져도 남게


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
            font = _font(size, style.font, style.bold, style.italic)
            lines = _wrap(words.strip(), QFontMetricsF(font), frame_w * MAX_WIDTH - 2 * pad)
            if lines:
                blocks.append((font, lines, color))
    if not blocks:
        return None
    stroke = px * max(0.0, style.outline_width)
    edge = stroke + pad  # 그림 가장자리에서 글까지
    width = max(QFontMetricsF(font).horizontalAdvance(line) for font, lines, _ in blocks for line in lines) + 2 * edge + 4
    height = sum(QFontMetricsF(font).height() * LINE_GAP * len(lines) for font, lines, _ in blocks) + 2 * edge
    margin = round(px * (SHADOW_OFFSET + SHADOW_SPREAD + 3 * SHADOW_BLUR)) if style.shadow > 0 else 0  # 그림자가 번질 자리 (사방 같게 — 가운데가 그대로)
    cw, ch = int(width) + 2, int(height) + 2  # 글(과 배경 상자) 칸
    w, h = cw + 2 * margin, ch + 2 * margin
    pen = QPen(QColor(style.outline), stroke * 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
    paths = []  # (글 모양, 색)
    top = margin + edge
    for font, lines, color in blocks:
        metrics = QFontMetricsF(font)
        line_h = metrics.height() * LINE_GAP
        path = QPainterPath()
        for i, line in enumerate(lines):
            advance = metrics.horizontalAdvance(line)
            if style.align == "left":
                left = margin + edge + 2
            elif style.align == "right":
                left = margin + cw - edge - 2 - advance
            else:
                left = (w - advance) / 2
            path.addText(QPointF(left, top + metrics.ascent() + i * line_h), font, line)
        paths.append((path, color))
        top += line_h * len(lines)
    image = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    if style.box:
        box = QColor(style.box_color)
        box.setAlphaF(max(0.0, min(1.0, style.box_opacity)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(box)
        painter.drawRoundedRect(QRectF(margin, margin, cw, ch), px * 0.25, px * 0.25)
    if style.shadow > 0:  # 상자 위, 글 아래
        painter.drawImage(0, 0, _shadow_image([p for p, _ in paths], pen if stroke > 0 else None, w, h, px, style.shadow))
    for path, color in paths:
        if stroke > 0:
            painter.strokePath(path, pen)
        painter.fillPath(path, QColor(color))
    painter.end()
    rgba = image.convertToFormat(QImage.Format.Format_RGBA8888)
    array = np.frombuffer(rgba.constBits(), np.uint8).reshape(h, rgba.bytesPerLine())[:, : w * 4].reshape(h, w, 4).copy()
    x = (frame_w - w) // 2
    y = round(frame_h * (1 - style.bottom)) - h + margin  # 아래 여백은 글 칸 기준 (그림자 자리는 빼고)
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


def fade(drawn: tuple[np.ndarray, int, int], opacity: float) -> tuple[np.ndarray, int, int]:
    """페이드: 그림의 투명도만 opacity배 (1이면 그대로)."""
    if opacity >= 1:
        return drawn
    array, x, y = drawn
    faded = array.copy()
    faded[..., 3] = (faded[..., 3].astype(np.float32) * opacity + 0.5).astype(np.uint8)
    return faded, x, y


def text_overlay(text: str, style: CaptionStyle, x: float, y: float, frame_w: int, frame_h: int):
    """화면 글자(제목·강조 문구): 자막과 같은 그림을, 글 가운데가 (x, y) 화면 비율 자리에 오게. (그림, 왼쪽, 위) 또는 None."""
    drawn = caption_image(text, style, frame_w, frame_h)
    if drawn is None:
        return None
    array = drawn[0]
    h, w = array.shape[:2]
    left = min(max(0, round(x * frame_w - w / 2)), max(0, frame_w - w))
    top = min(max(0, round(y * frame_h - h / 2)), max(0, frame_h - h))
    return array, left, top


def paint_text(painter: QPainter, target: QRectF, text: str, style: CaptionStyle, x: float, y: float,
               frame_w: int, frame_h: int) -> QRectF | None:
    """화면용 글자 (굽는 것과 같은 모양·자리). 그린 자리(화면 좌표)를 돌려준다."""
    drawn = text_overlay(text, style, x, y, frame_w, frame_h)
    if drawn is None:
        return None
    array, left, top = drawn
    scale = target.width() / frame_w
    image = QImage(array.data, array.shape[1], array.shape[0], array.shape[1] * 4, QImage.Format.Format_RGBA8888)
    rect = QRectF(target.x() + left * scale, target.y() + top * scale, array.shape[1] * scale, array.shape[0] * scale)
    painter.drawImage(rect, image)
    return rect


class OverlayBurner:
    """내보내기 프레임마다: 그 시각의 글자들과 자막을 굽는다 (pts = 출력 프레임 번호). 글자 먼저, 자막이 맨 위."""

    def __init__(self, subs: Subtitles | None, texts, size: tuple[int, int], fps: float) -> None:
        self._captions = CaptionBurner(subs, size, fps) if subs is not None and subs.cues else None
        self._texts, self._size, self._fps = tuple(texts), size, fps
        self._drawn: dict[str, object] = {}  # 글자 id → 그린 그림 (보이는 동안 프레임마다 다시 그리지 않게)

    def __call__(self, frame: av.VideoFrame) -> av.VideoFrame:
        t = frame.pts / self._fps
        for item in self._texts:
            if item.start <= t < item.end:
                opacity = item.opacity_at(t)
                if opacity <= 0:
                    continue
                if item.id not in self._drawn:
                    self._drawn[item.id] = text_overlay(item.text, item.style, item.x, item.y, *self._size)
                drawn = self._drawn[item.id]
                if drawn is not None:
                    frame = burn(frame, fade(drawn, opacity))
        return self._captions(frame) if self._captions is not None else frame
