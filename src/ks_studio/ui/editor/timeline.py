"""컷편집 타임라인. QPainter로 직접 그린다.

위에서부터: 눈금자 · …V3 · V2 · V1(메인, 썸네일) · A1(V1 파형, 오프셋만큼 밀려 보임). 빨간 선 = 재생 헤드.
V1은 자석식(빈틈 없음), 위 줄은 원하는 시간에 놓는다. 화면에는 위 줄이 아래 줄을 덮어 보인다.

마우스: 눈금자 클릭/끌기·빈 곳 클릭 = 재생 헤드 이동 · 빈 곳 끌기 = 박스로 여러 클립 선택
        클립 클릭 = 선택 · Ctrl/Shift+클릭 = 선택에 더하기/빼기 · 우클릭 = 클립 메뉴 (편집 화면이 채운다)
        클립 끌기 = 옮기기 (V1 안에서는 순서 바꾸기, 위아래 줄로도 옮길 수 있음. 위 줄에서는 재생 헤드·클립 끝에 달라붙음)
        클립 양 끝 끌기 = 자르기 · 휠 = 좌우 스크롤 · Ctrl+휠 = 확대/축소 · Shift+휠 = 위아래 스크롤
        워크스페이스에서 끌어다 놓기 = 삽입
위 줄에서 다른 클립과 겹치는 자리는 빨갛게 표시되고 놓이지 않는다.
왼쪽 줄 이름 칸: 👁 = 화면 숨기기, 🔊 = 소리 끄기 (줄마다 따로) · [+ 줄] = 맨 위에 줄 추가 · 우클릭 = 줄 추가/삭제
위 줄 클립은 아래쪽 띠에 파형을 그린다. 줄이 화면보다 많으면 줄 영역만 위아래로 스크롤된다 (눈금자·줄 이름은 고정).
스냅(켜고 끌 수 있음): 클립을 옮기거나 끝을 끌어 자를 때 재생 헤드·다른 클립 끝에 달라붙는다.
I/O 구간은 눈금자에 청록색 띠로, 줄 영역에는 옅은 음영으로 표시한다.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QMenu, QScrollBar, QWidget

from ks_studio.core.edits import add_track, move_clip, move_to, trim_clip
from ks_studio.core.project import DEFAULT_VIDEO_TRACKS, MAIN_TRACK, MIN_VIDEO_TRACKS, Clip, Placed, Project
from ks_studio.core.subtitles import MIN_CUE, move_cue, update_cue
from ks_studio.core.workspace import Workspace
from ks_studio.services.sources import SourceCache

MIME_MEDIA = "application/x-ks-media"
TICKS_H, CAPTION_H = 24, 18  # 눈금자 띠 아래에 자막 띠 (자막 블록 = 자막 한 줄)
RULER_H, ROW_H, AUDIO_H, GAP = TICKS_H + CAPTION_H, 56, 44, 4  # RULER_H = 줄이 시작하는 높이 (눈금자 + 자막 띠)
HEADER_W = 72  # 왼쪽 줄 이름 칸 (줄 이름 + 👁·🔊 버튼)
BUTTON_W, BUTTON_H = 30, 24
WAVE_H = 16  # 위 줄 클립 아래쪽 파형 띠
LEFT = HEADER_W + 8  # 타임라인 0프레임의 x
EDGE_PX = 6  # 클립 끝에서 이 거리 안이면 자르기
DRAG_START_PX = 5
SNAP_PX = 8
MIN_PPF, MAX_PPF = 0.02, 24.0  # 프레임당 픽셀
RULER_STEPS = (1, 5, 15, 30, 60, 150, 300, 600, 1800, 3600, 9000, 18000)
DROP_OK, DROP_BAD = QColor(63, 209, 199, 110), QColor(231, 76, 60, 130)
RANGE_BAR, RANGE_SHADE = QColor("#3fd1c7"), QColor(63, 209, 199, 28)
ZOOM_STEP = 1.25
WAVE_ON, WAVE_MUTED = QColor("#2ecc71"), QColor("#5c5c66")
HIDDEN_SHADE = QColor(22, 22, 26, 170)
ADD_BUTTON = QRectF(4, 2, HEADER_W - 8, TICKS_H - 4)  # 왼쪽 위 [+ 줄]
SCROLLBAR_W = 12
AUTOSCROLL_PX = 12  # 끌다가 위아래 끝에서 이 거리 안이면 자동 스크롤


def format_frames(frame: int, fps: int, with_frames: bool = True) -> str:
    seconds, ff = divmod(frame, fps)
    minutes, ss = divmod(seconds, 60)
    return f"{minutes:02d}:{ss:02d}.{ff:02d}" if with_frames else f"{minutes:02d}:{ss:02d}"


def content_height(video_tracks: int) -> int:
    """눈금자 아래 줄 영역 전체 높이."""
    return video_tracks * (ROW_H + GAP) + AUDIO_H + GAP + 8


class TimelineWidget(QWidget):
    seek_requested = Signal(int)
    selection_changed = Signal(object)  # 선택이 바뀜. 값 = 마지막으로 고른 클립 id 또는 None (전체는 .selection)
    edited = Signal(object)  # 새 Project
    media_dropped = Signal(str, int, int)  # 원본 경로, 줄, 위치 (V1이면 끼워 넣을 순서, 위 줄이면 시작 프레임)
    track_toggled = Signal(int, str)  # 줄, "hide" 또는 "mute"
    track_delete_requested = Signal(int)  # 줄 (확인 창은 받는 쪽에서)
    clip_menu_requested = Signal(QPoint)  # 클립·빈 곳 우클릭 (화면 좌표). 메뉴는 편집 화면이 만든다
    caption_clicked = Signal(int)  # 자막 띠에서 누른 자막 줄 번호 (끌어서 옮기기·길이 바꾸기는 edited로)

    def __init__(self, sources: SourceCache, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(RULER_H + content_height(DEFAULT_VIDEO_TRACKS))
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self._sources = sources
        sources.ready.connect(lambda _path: self.update())
        self._project = Project()
        self._workspace: Workspace | None = None
        self.playhead = 0
        self.selected: str | None = None  # 마지막으로 고른 클립 (속성 창·분할·Q/W 대상)
        self.selection: set[str] = set()  # 선택된 클립 전부 (삭제·복사·복제 대상)
        self._box: QRectF | None = None  # 박스 선택 중인 영역
        self._ppf = 1.0
        self._scroll = 0.0  # 화면 왼쪽 끝의 타임라인 프레임
        self._draft: Project | None = None  # 자르는 중 미리보기
        self._drag: dict | None = None
        # 놓을 자리 표시: ("main", 순서) 또는 ("overlay", 줄, 시작, 길이, 놓을 수 있는지)
        self._drop: tuple | None = None
        self._thumb_images: dict[tuple[str, int], QImage] = {}
        # 세로 스크롤 = '아래에서부터 올린 양'(px). 0이면 V1·A1이 바닥에 붙어 보인다 → 맨 위에 줄을 추가해도 V1은 제자리
        self._vscroll = 0
        self._cache: QPixmap | None = None  # 재생 헤드를 뺀 그림. 재생 중에는 이것만 베끼고 빨간 선만 새로 그린다
        self.snapping = True
        self.range_in: int | None = None  # I/O 구간 [range_in, range_out)
        self.range_out: int | None = None
        self._scrollbar = QScrollBar(Qt.Orientation.Vertical, self)
        self._scrollbar.valueChanged.connect(lambda value: self._set_vscroll(self._max_vscroll() - value))

    # ── 바깥에서 부르는 것 ─────────────────────────────────────

    def set_project(self, project: Project, workspace: Workspace, fit: bool = False) -> None:
        self._project, self._workspace = project, workspace
        alive = {p.clip.id for p in project.placed()}
        if not self.selection <= alive or (self.selected is not None and self.selected not in alive):
            kept = self.selection & alive
            self.set_selection(kept, self.selected if self.selected in kept else next(iter(kept), None))
        if fit:
            self.fit()
        self._set_vscroll(self._vscroll)

    def fit(self) -> None:
        span = max(self._project.duration, self._project.fps * 10)
        self._ppf = max(MIN_PPF, min(MAX_PPF, (self.width() - LEFT - 12) / span))
        self._scroll = 0.0
        self.update()

    def set_range(self, start: int | None, end: int | None) -> None:
        self.range_in, self.range_out = start, end
        self.update()

    def zoom(self, factor: float) -> None:
        """재생 헤드를 화면의 같은 자리에 둔 채 확대/축소."""
        x = min(max(self._x(self.playhead), LEFT), self.width())
        anchor = self._frame_at(x)
        self._ppf = max(MIN_PPF, min(MAX_PPF, self._ppf * factor))
        self._scroll = max(0.0, anchor - (x - LEFT) / self._ppf)
        self.update()

    def set_playhead(self, frame: int) -> None:
        self.playhead = frame
        x = self._x(frame)
        if x < LEFT or x > self.width() - 40:  # 재생 헤드가 화면 밖으로 나가면 따라간다
            self._scroll = max(0.0, frame - (self.width() * 0.2) / self._ppf)
            self.update()
        else:
            super().update()  # 내용은 그대로 → 캐시를 베끼고 재생 헤드만 다시

    def update(self, *args) -> None:
        """내용이 바뀌었을 때의 다시 그리기 (캐시를 버린다). 재생 헤드만 움직이면 set_playhead가 캐시를 그대로 쓴다.

        실측: 재생 중 매 프레임 타임라인 전체(파형 수천 개 선)를 그리면 메인 스레드가 GIL을 오래 잡아
        화면 속 화면 재생이 55 → 25fps로 떨어졌다.
        """
        self._cache = None
        super().update(*args)

    # ── 좌표 ────────────────────────────────────────────────────

    def _x(self, frame: float) -> float:
        return LEFT + (frame - self._scroll) * self._ppf

    def _frame_at(self, x: float) -> float:
        return self._scroll + (x - LEFT) / self._ppf

    def _max_vscroll(self) -> int:
        return max(0, content_height(self._project.video_tracks) - (self.height() - RULER_H))

    def _set_vscroll(self, value: float) -> None:
        self._vscroll = int(max(0, min(value, self._max_vscroll())))
        top = self._max_vscroll()
        self._scrollbar.blockSignals(True)
        self._scrollbar.setRange(0, top)
        self._scrollbar.setPageStep(max(1, self.height() - RULER_H))
        self._scrollbar.setValue(top - self._vscroll)
        self._scrollbar.blockSignals(False)
        self._scrollbar.setGeometry(QRect(self.width() - SCROLLBAR_W, RULER_H, SCROLLBAR_W, self.height() - RULER_H))
        self._scrollbar.setVisible(top > 0)
        self.update()

    def _autoscroll(self, y: float) -> None:
        """끄는 중 위아래 끝에 가까우면 그쪽으로 스크롤."""
        if y < RULER_H + AUTOSCROLL_PX:
            self._set_vscroll(self._vscroll + 8)
        elif y > self.height() - AUTOSCROLL_PX:
            self._set_vscroll(self._vscroll - 8)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._set_vscroll(self._vscroll)

    def _row_top(self, track: int) -> float:
        """영상 줄 track(1=V1)의 위쪽 y. 위 줄일수록 화면 위에 있다."""
        hidden_above = self._max_vscroll() - self._vscroll  # 스크롤되어 눈금자 뒤로 숨은 양
        return RULER_H + GAP + (self._project.video_tracks - track) * (ROW_H + GAP) - hidden_above

    def _audio_top(self) -> float:
        return self._row_top(MAIN_TRACK) + ROW_H + GAP

    def _track_at(self, y: float) -> int | None:
        """y에 있는 줄. A1(오디오) 줄은 V1 클립의 일부다. 눈금자면 None."""
        if y < RULER_H:
            return None
        if y >= self._audio_top():
            return MAIN_TRACK
        for track in range(self._project.video_tracks, 0, -1):
            if self._row_top(track) <= y < self._row_top(track) + ROW_H + GAP:
                return track
        return None

    def _drop_track(self, y: float) -> int:
        """끌어 놓을 줄: 줄 밖(위쪽)이면 맨 위 줄."""
        track = self._track_at(max(y, RULER_H + GAP))
        return track if track is not None else self._project.video_tracks

    def _header_button(self, track: int, kind: str) -> QRectF:
        """줄 이름 칸 아래쪽의 👁(hide)·🔊(mute) 버튼 자리."""
        x = 4 if kind == "hide" else 4 + BUTTON_W + 4
        return QRectF(x, self._row_top(track) + ROW_H - BUTTON_H - 4, BUTTON_W, BUTTON_H)

    def _header_hit(self, x: float, y: float) -> tuple[int, str] | None:
        if y < RULER_H:  # 눈금자 뒤로 스크롤된 버튼은 누를 수 없다
            return None
        for track in range(1, self._project.video_tracks + 1):
            for kind in ("hide", "mute"):
                if self._header_button(track, kind).contains(QPointF(x, y)):
                    return track, kind
        return None

    def _hit(self, x: float, y: float) -> tuple[Placed, str] | None:
        track = self._track_at(y)
        if track is None or x < HEADER_W:
            return None
        for placed in self._project.placed():
            if placed.track != track:
                continue
            x0, x1 = self._x(placed.start), self._x(placed.end)
            if x0 <= x <= x1:
                if x - x0 <= EDGE_PX:
                    return placed, "left"
                if x1 - x <= EDGE_PX:
                    return placed, "right"
                return placed, "body"
        return None

    def _main_bounds(self, project: Project) -> list[int]:
        """V1 클립 경계 (0 = 맨 앞, 마지막 = V1 끝)."""
        return [p.start for p in project.placed() if p.track == MAIN_TRACK] + [project.main_duration]

    def _boundary_index(self, x: float) -> int:
        """x에서 가장 가까운 V1 클립 경계의 순서."""
        frame = self._frame_at(x)
        return int(np.argmin([abs(b - frame) for b in self._main_bounds(self._project)]))

    def _snap_shift(self, edges: list[float], ignore: str | None) -> float:
        """edges 중 하나가 재생 헤드·다른 클립 끝에 SNAP_PX 안이면 그만큼 옮길 거리 (스냅이 꺼져 있으면 0)."""
        if not self.snapping:
            return 0
        targets = [0, self.playhead] + [e for p in self._project.placed() if p.clip.id != ignore for e in (p.start, p.end)]
        limit = SNAP_PX / self._ppf
        best = None
        for edge in edges:
            for target in targets:
                if abs(target - edge) <= limit and (best is None or abs(target - edge) < abs(best)):
                    best = target - edge
        return best or 0

    def _snap(self, start: int, length: int, ignore: str | None) -> int:
        """옮기는 클립의 시작이나 끝이 달라붙도록 시작 위치를 고친다."""
        return max(0, start + round(self._snap_shift([start, start + length], ignore)))

    def _drop_at(self, x: float, y: float, length: int, grab: float, ignore: str | None) -> tuple:
        track = self._drop_track(y)
        if track == MAIN_TRACK:
            return ("main", self._boundary_index(x))
        start = self._snap(max(0, round(self._frame_at(x) - grab)), length, ignore)
        return ("overlay", track, start, length, self._project.fits(track, start, length, ignore=ignore))

    # ── 그리기 ──────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        ratio = self.devicePixelRatioF()
        if self._cache is None or self._cache.size() != self.size() * ratio:
            self._cache = QPixmap(self.size() * ratio)
            self._cache.setDevicePixelRatio(ratio)
            content = QPainter(self._cache)
            self._paint_content(content)
            content.end()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
        painter.setClipRect(QRectF(HEADER_W, 0, self.width(), self.height()))
        self._paint_playhead(painter)

    def _paint_content(self, painter: QPainter) -> None:
        painter.fillRect(self.rect(), QColor("#16161a"))
        painter.setClipRect(QRectF(HEADER_W, RULER_H, self.width(), self.height()))  # 줄 영역: 스크롤돼도 눈금자 위로 안 그린다
        for track in range(1, self._project.video_tracks + 1):
            painter.fillRect(QRectF(HEADER_W, self._row_top(track), self.width(), ROW_H), QColor("#1c1c22"))
        painter.fillRect(QRectF(HEADER_W, self._audio_top(), self.width(), AUDIO_H), QColor("#1a1d1b"))
        project = self._draft or self._project
        for placed in project.placed():
            x0, x1 = self._x(placed.start), self._x(placed.end)
            if x1 < 0 or x0 > self.width():
                continue
            self._paint_clip(painter, project, placed, x0, x1)
        self._paint_transitions(painter, project)
        self._paint_drop(painter)
        if self._box is not None:
            painter.setPen(QPen(QColor("#f1c40f"), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(241, 196, 15, 30))
            painter.drawRect(self._box)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        range_x = self._range_x()
        if range_x is not None:
            painter.fillRect(QRectF(range_x[0], RULER_H, range_x[1] - range_x[0], self.height()), RANGE_SHADE)
        painter.setClipRect(QRectF(HEADER_W, 0, self.width(), self.height()))
        self._paint_ruler(painter)
        self._paint_markers(painter)
        self._paint_captions(painter, project)
        if range_x is not None:
            painter.fillRect(QRectF(range_x[0], TICKS_H - 5, max(2.0, range_x[1] - range_x[0]), 5), RANGE_BAR)
        painter.setClipping(False)
        self._paint_headers(painter)

    def _range_x(self) -> tuple[float, float] | None:
        """I/O 구간의 화면 x. 시작만 정했으면 끝은 타임라인 끝, 끝만 정했으면 시작은 0."""
        if self.range_in is None and self.range_out is None:
            return None
        start = self.range_in if self.range_in is not None else 0
        end = self.range_out if self.range_out is not None else max(start + 1, self._project.duration)
        return self._x(start), self._x(end)

    def _paint_headers(self, painter: QPainter) -> None:
        painter.fillRect(QRectF(0, 0, HEADER_W, self.height()), QColor("#111114"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2a2a31"))
        painter.drawRoundedRect(ADD_BUTTON, 4, 4)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(ADD_BUTTON, Qt.AlignmentFlag.AlignCenter, "+ 줄")
        painter.setPen(QColor("#9a9aa3"))
        painter.drawText(QRectF(0, TICKS_H, HEADER_W, CAPTION_H), Qt.AlignmentFlag.AlignCenter, "자막")
        painter.setClipRect(QRectF(0, RULER_H, HEADER_W, self.height()))
        painter.drawText(QRectF(0, self._audio_top(), HEADER_W, AUDIO_H), Qt.AlignmentFlag.AlignCenter, "A1")
        for track in range(1, self._project.video_tracks + 1):
            painter.setPen(QColor("#9a9aa3"))
            painter.drawText(QRectF(8, self._row_top(track) + 2, HEADER_W, 20), Qt.AlignmentFlag.AlignVCenter, f"V{track}")
            for kind, off in (("hide", track in self._project.hidden), ("mute", track in self._project.muted)):
                rect = self._header_button(track, kind)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#5a2a2a" if off else "#2a2a31"))
                painter.drawRoundedRect(rect, 4, 4)
                painter.setPen(QColor("#ffffff" if not off else "#c8a0a0"))
                glyph = {"hide": "👁", "mute": "🔇" if off else "🔊"}[kind]
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, glyph)
                if off and kind == "hide":
                    painter.setPen(QPen(QColor("#ff6b6b"), 2))
                    painter.drawLine(rect.bottomLeft() + QPointF(6, -5), rect.topRight() + QPointF(-6, 5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setClipping(False)

    def _paint_ruler(self, painter: QPainter) -> None:
        fps = self._project.fps
        step = next((s for s in RULER_STEPS if s * self._ppf >= 70), RULER_STEPS[-1])
        first = int(max(0, self._frame_at(HEADER_W)) // step * step)
        painter.setPen(QColor("#6f6f78"))
        frame = first
        while self._x(frame) < self.width():
            x = self._x(frame)
            painter.drawLine(QPointF(x, TICKS_H - 6), QPointF(x, TICKS_H))
            painter.drawText(QPointF(x + 3, TICKS_H - 9), format_frames(frame, fps, with_frames=step < fps))
            frame += step

    def _paint_transitions(self, painter: QPainter, project: Project) -> None:
        """V1 디졸브(앞 클립과 겹친 곳)를 X자 무늬로 표시한다."""
        top = self._row_top(MAIN_TRACK)
        overlaps = project.main_overlaps()
        for placed in project.main_placed():
            frames = overlaps[placed.clip.id][0]
            if not frames:
                continue
            rect = QRectF(self._x(placed.start), top, frames * self._ppf, ROW_H)
            painter.fillRect(rect, QColor(241, 196, 15, 70))
            painter.setPen(QPen(QColor("#f1c40f"), 2))
            painter.drawLine(rect.topLeft(), rect.bottomRight())
            painter.drawLine(rect.bottomLeft(), rect.topRight())
            painter.drawRect(rect)

    def _caption_rect(self, project: Project, cue) -> QRectF:
        fps = project.fps
        x0, x1 = self._x(cue.start * fps), self._x(cue.end * fps)
        return QRectF(x0, TICKS_H + 2, max(3.0, x1 - x0), CAPTION_H - 4)

    def _paint_captions(self, painter: QPainter, project: Project) -> None:
        """자막 띠: 자막 줄마다 블록 (글은 들어가는 만큼). 누르면 그 자막으로 가고, 끌면 옮기고, 양 끝을 끌면 길이를 바꾼다."""
        painter.fillRect(QRectF(HEADER_W, TICKS_H, self.width(), CAPTION_H), QColor("#14141a"))
        font = painter.font()
        font.setPixelSize(11)
        painter.setFont(font)
        for cue in project.subtitles.cues:
            rect = self._caption_rect(project, cue)
            if rect.right() < HEADER_W or rect.left() > self.width():
                continue
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#3d6b8f"))
            painter.drawRoundedRect(rect, 3, 3)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(rect.adjusted(4, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter, cue.text)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _caption_part(self, index: int, x: float) -> str:
        """자막 블록의 어디를 눌렀나: 왼쪽 끝·오른쪽 끝(길이 바꾸기) 또는 몸통(옮기기). 아주 짧은 블록은 몸통만."""
        rect = self._caption_rect(self._project, self._project.subtitles.cues[index])
        if rect.width() > 3 * EDGE_PX:
            if x - rect.left() <= EDGE_PX:
                return "left"
            if rect.right() - x <= EDGE_PX:
                return "right"
        return "body"

    def _caption_dragged(self, x: float) -> Project:
        """자막 블록을 x까지 끈 모양 (재생 헤드·클립 끝에 달라붙는다)."""
        drag, project = self._drag, self._project
        fps, cue = project.fps, project.subtitles.cues[drag["index"]]
        moved = (x - drag["x0"]) / self._ppf  # 타임라인 프레임
        start, end = cue.start * fps, cue.end * fps
        part = drag["part"]
        if part == "body":
            moved += self._snap_shift([start + moved, end + moved], None)
            subs, _ = move_cue(project.subtitles, drag["index"], moved / fps)
        elif part == "left":
            moved += self._snap_shift([start + moved], None)
            subs, _ = update_cue(project.subtitles, drag["index"], start=min((start + moved) / fps, cue.end - MIN_CUE))
        else:
            moved += self._snap_shift([end + moved], None)
            subs, _ = update_cue(project.subtitles, drag["index"], end=max((end + moved) / fps, cue.start + MIN_CUE))
        return replace(project, subtitles=subs)

    def _caption_at(self, x: float, y: float) -> int | None:
        if not TICKS_H <= y < RULER_H:
            return None
        project = self._project
        for i, cue in enumerate(project.subtitles.cues):
            if self._caption_rect(project, cue).adjusted(-2, -4, 2, 4).contains(QPointF(x, y)):
                return i
        return None

    def _paint_markers(self, painter: QPainter) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#f1c40f"))
        for frame in self._project.markers:
            x = self._x(frame)
            painter.drawPolygon(QPolygonF([QPointF(x - 5, 2), QPointF(x + 5, 2), QPointF(x + 5, 9), QPointF(x, 14), QPointF(x - 5, 9)]))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_clip(self, painter: QPainter, project: Project, placed: Placed, x0: float, x1: float) -> None:
        clip, start = placed.clip, placed.start
        path = self._workspace.resolve(clip.source) if self._workspace else Path(clip.source)
        exists = path.exists()
        video = QRectF(x0 + 1, self._row_top(placed.track), max(2.0, x1 - x0 - 2), ROW_H)
        main = placed.track == MAIN_TRACK
        # V1 소리는 A1 줄에, 위 줄 소리는 클립 아래쪽 띠에 그린다
        audio = QRectF(video.left(), self._audio_top(), video.width(), AUDIO_H) if main else video.adjusted(0, ROW_H - WAVE_H, 0, 0)
        picture = video if main else video.adjusted(0, 0, 0, -WAVE_H)
        painter.fillRect(picture, QColor("#24384a" if exists else "#5a2020"))
        painter.fillRect(audio, QColor("#1f2d25" if exists else "#5a2020"))
        if exists:
            self._paint_thumbnails(painter, project, clip, start, path, picture)
            color = WAVE_MUTED if placed.track in project.muted else WAVE_ON
            self._paint_waveform(painter, project, clip, start, path, audio, color, 10 ** (clip.volume_db / 20))
        self._paint_fades(painter, project, clip, audio)
        if placed.track in project.hidden:
            painter.fillRect(picture, HIDDEN_SHADE)
        label = path.name if exists else f"⚠ 파일 없음: {path.name}"
        sound = [f"오디오 {clip.audio_offset_ms:+d}ms"] if clip.audio_offset_ms else []
        if clip.volume_db:
            sound.append(f"볼륨 {clip.volume_db:+.1f}dB")
        offset = " · ".join(sound)
        if main:
            self._paint_label(painter, picture, label)
            if offset:
                self._paint_label(painter, audio, offset)
        else:
            self._paint_label(painter, picture, f"{label}  ·  {offset}" if offset else label)
        outline = video.united(audio)
        if clip.id in self.selection:
            painter.setPen(QPen(QColor("#f1c40f"), 3 if clip.id == self.selected else 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(outline)

    def _paint_drop(self, painter: QPainter) -> None:
        if self._drop is None:
            return
        if self._drop[0] == "main":
            x = self._x(self._main_bounds(self._project)[self._drop[1]])
            top = self._row_top(MAIN_TRACK) - 2
            painter.fillRect(QRectF(x - 2, top, 4, self._audio_top() + AUDIO_H + 2 - top), QColor("#3fd1c7"))
            return
        _, track, start, length, ok = self._drop
        x0, x1 = self._x(start), self._x(start + length)
        painter.fillRect(QRectF(x0, self._row_top(track), max(2.0, x1 - x0), ROW_H), DROP_OK if ok else DROP_BAD)

    def _paint_label(self, painter: QPainter, rect: QRectF, text: str) -> None:
        """썸네일·파형 위에서도 읽히게 어두운 배경을 깐다."""
        metrics = painter.fontMetrics()
        box = QRectF(rect.left() + 4, rect.top() + 3, min(rect.width() - 8, metrics.horizontalAdvance(text) + 8), metrics.height() + 2)
        if box.width() <= 8:
            return
        painter.fillRect(box, QColor(0, 0, 0, 150))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(box.adjusted(4, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

    def _paint_thumbnails(self, painter, project, clip, start, path, rect: QRectF) -> None:
        thumbs = self._sources.thumbnails(path)
        if thumbs is None or len(thumbs.times) == 0:
            return
        tile = rect.height() * 16 / 9
        painter.save()
        painter.setClipRect(rect, Qt.ClipOperation.IntersectClip)
        x = rect.left() + max(0, int((0 - rect.left()) // tile)) * tile
        while x < min(rect.right(), self.width()):
            source_time = project.source_time(clip, self._frame_at(x) - start)
            index = max(0, int(np.searchsorted(thumbs.times, source_time, side="right")) - 1)
            painter.drawImage(QRectF(x, rect.top(), tile, rect.height()), self._thumb_image(path, thumbs, index))
            x += tile
        painter.restore()

    def _thumb_image(self, path, thumbs, index: int) -> QImage:
        key = (str(path), index)
        if key not in self._thumb_images:
            rgb = np.ascontiguousarray(thumbs.images[index])
            h, w, _ = rgb.shape
            self._thumb_images[key] = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        return self._thumb_images[key]

    def _paint_fades(self, painter: QPainter, project: Project, clip: Clip, rect: QRectF) -> None:
        """페이드 인·아웃을 소리 칸 위 어두운 삼각형과 선으로 보여 준다."""
        per_ms = project.fps * self._ppf / 1000
        for ms, left in ((clip.fade_in_ms, True), (clip.fade_out_ms, False)):
            if not ms:
                continue
            width = min(rect.width(), ms * per_ms)
            edge = rect.left() if left else rect.right()
            inner = edge + width if left else edge - width
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 120))
            painter.drawPolygon(QPolygonF([QPointF(edge, rect.top()), QPointF(inner, rect.top()), QPointF(edge, rect.bottom())]))
            painter.setPen(QPen(QColor("#ffffff"), 1))
            painter.drawLine(QPointF(edge, rect.bottom()), QPointF(inner, rect.top()))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_waveform(self, painter, project, clip, start, path, rect: QRectF, color: QColor, gain: float = 1.0) -> None:
        peaks = self._sources.peaks(path)
        if peaks is None or len(peaks) == 0:
            return
        mid, half = rect.center().y(), rect.height() / 2 - 2
        lo, hi = int(max(rect.left(), HEADER_W)), int(min(rect.right(), self.width()))
        if lo >= hi:
            return
        px = np.arange(lo, hi + 1)
        # 오프셋이 +면 소리가 늦게 나므로, 그 위치에서 들리는 건 원본의 더 앞부분이다
        times = project.source_time(clip, self._frame_at(px) - start) - clip.audio_offset_ms / 1000
        index = np.clip((times * 100).astype(int), 0, len(peaks))
        # 말소리는 진폭이 작아 그대로 그리면 평평하다 → 원본 최대값으로 나누고 √로 작은 소리를 키운다
        scale = gain / max(float(peaks.max()), 1e-3)  # 볼륨을 바꾸면 파형 크기도 따라 바뀐다
        painter.setPen(color)
        for x, i0, i1 in zip(px[:-1], index[:-1], index[1:]):
            segment = peaks[i0 : max(i1, i0 + 1)]
            value = min(1.0, float(segment.max()) * scale) ** 0.5 * half if len(segment) else 0.0
            painter.drawLine(QPointF(x, mid - value), QPointF(x, mid + value))

    def _paint_playhead(self, painter: QPainter) -> None:
        x = self._x(self.playhead)
        painter.setPen(QPen(QColor("#ff4d4d"), 2))
        painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        painter.setBrush(QColor("#ff4d4d"))
        painter.drawPolygon(QPolygonF([QPointF(x - 6, 0), QPointF(x + 6, 0), QPointF(x, 8)]))

    # ── 마우스 ──────────────────────────────────────────────────

    def select(self, clip_id: str | None) -> None:
        """클립 하나만 선택 (None이면 선택 해제)."""
        self.set_selection({clip_id} if clip_id else set(), clip_id)

    def set_selection(self, clip_ids: set[str], primary: str | None) -> None:
        changed = primary != self.selected or set(clip_ids) != self.selection
        self.selection, self.selected = set(clip_ids), primary
        if changed:
            self.selection_changed.emit(primary)
        self.update()

    def _clip_rect(self, placed: Placed) -> QRectF:
        """클립이 화면에서 차지하는 영역 (V1은 A1 파형 줄까지)."""
        x0, x1 = self._x(placed.start), self._x(placed.end)
        bottom = self._audio_top() + AUDIO_H if placed.track == MAIN_TRACK else self._row_top(placed.track) + ROW_H
        top = self._row_top(placed.track)
        return QRectF(x0, top, x1 - x0, bottom - top)

    def _seek_to_x(self, x: float) -> None:
        frame = int(round(self._frame_at(x)))
        self.seek_requested.emit(max(0, min(frame, max(0, self._project.duration - 1))))

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x, y = event.position().x(), event.position().y()
        if x < HEADER_W:
            button = self._header_hit(x, y)
            if button is not None:
                self.track_toggled.emit(*button)
            elif ADD_BUTTON.contains(QPointF(x, y)):
                self.edited.emit(add_track(self._project, self._project.video_tracks + 1))
            return
        caption = self._caption_at(x, y)
        if caption is not None:  # 자막 블록: '자막' 탭에서 그 줄을 고르고, 끌면 옮기기·길이 바꾸기 (그냥 떼면 그 자막으로 간다)
            self.caption_clicked.emit(caption)
            self._drag = {"kind": "caption-press", "index": caption, "part": self._caption_part(caption, x), "x0": x, "y0": y}
            return
        hit = self._hit(x, y)
        adding = bool(event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier))
        if hit is None:
            if y < RULER_H:
                self._drag = {"kind": "scrub"}
                self._seek_to_x(x)
            else:  # 빈 곳: 놓을 때까지 안 움직이면 재생 헤드 이동, 끌면 박스 선택
                self._drag = {"kind": "empty", "x0": x, "y0": y, "adding": adding, "before": set(self.selection)}
            return
        placed, part = hit
        if adding:  # 선택에 더하거나 빼기만 하고 끌지 않는다
            clip_id = placed.clip.id
            chosen = self.selection ^ {clip_id}
            self.set_selection(chosen, clip_id if clip_id in chosen else next(iter(chosen), None))
            return
        self.select(placed.clip.id)
        kind = "press" if part == "body" else f"trim-{part}"
        self._drag = {"kind": kind, "placed": placed, "x0": x, "y0": y, "grab": self._frame_at(x) - placed.start}

    def mouseMoveEvent(self, event) -> None:
        x, y = event.position().x(), event.position().y()
        if self._drag is None:
            hit = self._hit(x, y)
            edge = hit is not None and hit[1] != "body"
            caption = self._caption_at(x, y)
            if caption is not None:
                edge = self._caption_part(caption, x) != "body"
            self.setCursor(Qt.CursorShape.SizeHorCursor if edge else Qt.CursorShape.ArrowCursor)
            button = self._header_hit(x, y)
            tips = {"hide": "화면 숨기기/보이기 (소리는 그대로)", "mute": "소리 끄기/켜기 (화면은 그대로)"}
            if button:
                self.setToolTip(f"V{button[0]} {tips[button[1]]}")
            elif ADD_BUTTON.contains(QPointF(x, y)):
                self.setToolTip("맨 위에 영상 줄 추가 (줄 이름 칸 우클릭: 끼워 넣기·삭제)")
            else:
                self.setToolTip("")
            return
        kind = self._drag["kind"]
        if kind == "caption-press" and abs(x - self._drag["x0"]) >= DRAG_START_PX:
            self._drag["kind"] = kind = "caption-drag"
        if kind == "caption-drag":
            self._draft = self._caption_dragged(x)
            self.update()
            return
        if kind == "scrub":
            self._seek_to_x(x)
        elif kind == "empty" and abs(x - self._drag["x0"]) + abs(y - self._drag["y0"]) >= DRAG_START_PX:
            self._drag["kind"] = kind = "box"
        if kind == "box":
            self._box = QRectF(QPointF(self._drag["x0"], self._drag["y0"]), QPointF(x, y)).normalized()
            inside = {p.clip.id for p in self._project.placed() if self._clip_rect(p).intersects(self._box)}
            chosen = self._drag["before"] | inside if self._drag["adding"] else inside
            self.set_selection(chosen, self.selected if self.selected in chosen else next(iter(sorted(chosen)), None))
            return
        if kind == "press" and abs(x - self._drag["x0"]) + abs(y - self._drag["y0"]) >= DRAG_START_PX:
            self._drag["kind"] = kind = "move"
        if kind == "move":
            self._autoscroll(y)
            placed = self._drag["placed"]
            self._drop = self._drop_at(x, y, placed.end - placed.start, self._drag["grab"], placed.clip.id)
            self.update()
        elif kind.startswith("trim"):
            placed = self._drag["placed"]
            clip = placed.clip
            moved = (x - self._drag["x0"]) / self._ppf  # 화면에서 끈 거리 (타임라인 프레임)
            # V1 앞쪽 자르기는 클립 시작이 제자리라 끝이 움직이지 않는다 → 스냅하지 않는다
            if kind == "trim-right":
                moved += self._snap_shift([placed.end + moved], clip.id)
            elif placed.track != MAIN_TRACK:
                moved += self._snap_shift([placed.start + moved], clip.id)
            # 타임라인 프레임 → 원본 프레임 (원본 fps가 다를 수 있다)
            delta = round(moved * clip.fps / self._project.fps)
            trimmed = self._trimmed(clip, delta, kind)
            if trimmed is not self._project:  # 위 줄에서 옆 클립과 겹치면 거부됨 → 마지막으로 된 모양 유지
                self._draft = trimmed
            self.update()

    def _trimmed(self, clip: Clip, delta: int, kind: str) -> Project:
        if kind == "trim-left":
            new_in = max(0, min(clip.in_frame + delta, clip.out_frame - 1))
            return trim_clip(self._project, clip.id, new_in, clip.out_frame)
        info = self._sources.info(self._workspace.resolve(clip.source)) if self._workspace else None
        limit = info.frames if info else clip.out_frame
        new_out = max(clip.in_frame + 1, min(clip.out_frame + delta, limit))
        return trim_clip(self._project, clip.id, clip.in_frame, new_out)

    def mouseReleaseEvent(self, _event) -> None:
        drag, self._drag = self._drag, None
        drop, self._drop = self._drop, None
        if drag is None:
            return
        if drag["kind"] == "caption-press":  # 끌지 않고 뗐다 → 그 자막 시작으로
            self.seek_requested.emit(round(self._project.subtitles.cues[drag["index"]].start * self._project.fps))
        elif drag["kind"] == "caption-drag" and self._draft is not None:
            self.edited.emit(self._draft)
        if drag["kind"] == "empty":  # 끌지 않고 뗐다 → 재생 헤드 이동, 선택 해제
            if not drag["adding"]:
                self.select(None)
            self._seek_to_x(drag["x0"])
        self._box = None
        if drag["kind"] == "move" and drop is not None:
            edited = self._moved(drag["placed"], drop)
            if edited is not self._project:
                self.edited.emit(edited)
        elif drag["kind"].startswith("trim") and self._draft is not None:
            self.edited.emit(self._draft)
        self._draft = None
        self.update()

    def _moved(self, placed: Placed, drop: tuple) -> Project:
        clip_id = placed.clip.id
        if drop[0] == "main":
            if placed.track != MAIN_TRACK:
                return move_to(self._project, clip_id, MAIN_TRACK, drop[1])
            index = self._project.index_of(clip_id)
            target = drop[1] if drop[1] <= index else drop[1] - 1  # 빼낸 뒤 기준 순서
            return move_clip(self._project, clip_id, target) if target != index else self._project
        _, track, start, _, ok = drop
        if not ok or (track, start) == (placed.track, placed.start):
            return self._project
        return move_to(self._project, clip_id, track, start)

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120
        x = event.position().x()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Shift+휠은 시스템에 따라 가로 휠로 바뀌어 들어온다
            steps = (event.angleDelta().y() or event.angleDelta().x()) / 120
            self._set_vscroll(self._vscroll + steps * (ROW_H + GAP))
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            anchor = self._frame_at(x)
            self._ppf = max(MIN_PPF, min(MAX_PPF, self._ppf * ZOOM_STEP**steps))
            self._scroll = max(0.0, anchor - (x - LEFT) / self._ppf)
        else:
            self._scroll = max(0.0, min(self._scroll - steps * 80 / self._ppf, float(self._project.duration)))
        self.update()

    def contextMenuEvent(self, event) -> None:
        """줄 이름 칸 우클릭: 줄 추가/삭제 메뉴. 클립·빈 곳 우클릭: 편집 화면에 클립 메뉴를 요청."""
        if event.pos().x() >= HEADER_W:
            if event.pos().y() < RULER_H:
                return
            hit = self._hit(event.pos().x(), event.pos().y())
            if hit is not None and hit[0].clip.id not in self.selection:
                self.select(hit[0].clip.id)
            self.clip_menu_requested.emit(event.globalPos())
            return
        track = self._track_at(event.pos().y())
        if track is None:
            return
        menu = QMenu(self)
        add = menu.addAction("이 줄 위에 줄 추가")
        delete = menu.addAction(f"V{track} 줄 삭제")
        if track == MAIN_TRACK:
            delete.setText("V1(메인) 줄은 삭제할 수 없습니다")
            delete.setEnabled(False)
        elif self._project.video_tracks <= MIN_VIDEO_TRACKS:
            delete.setText(f"최소 {MIN_VIDEO_TRACKS}줄은 남겨야 합니다")
            delete.setEnabled(False)
        chosen = menu.exec(event.globalPos())
        if chosen is add:
            self.edited.emit(add_track(self._project, track + 1))
        elif chosen is delete:
            self.track_delete_requested.emit(track)

    # ── 워크스페이스에서 끌어다 놓기 ─────────────────────────────

    def _media_length(self, path: str) -> int:
        info = self._sources.info(Path(path))
        if info is None or info.frames == 0:
            return self._project.fps
        return self._project.length(Clip(path, 0, info.frames, fps=info.fps))

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME_MEDIA):
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        path = bytes(event.mimeData().data(MIME_MEDIA)).decode()
        self._autoscroll(event.position().y())
        self._drop = self._drop_at(event.position().x(), event.position().y(), self._media_length(path), 0, None)
        self.update()
        event.acceptProposedAction()

    def dragLeaveEvent(self, _event) -> None:
        self._drop = None
        self.update()

    def dropEvent(self, event) -> None:
        path = bytes(event.mimeData().data(MIME_MEDIA)).decode()
        drop = self._drop_at(event.position().x(), event.position().y(), self._media_length(path), 0, None)
        self._drop = None
        self.update()
        if drop[0] == "main":
            self.media_dropped.emit(path, MAIN_TRACK, drop[1])
        elif drop[4]:
            self.media_dropped.emit(path, drop[1], drop[2])
        event.acceptProposedAction()
