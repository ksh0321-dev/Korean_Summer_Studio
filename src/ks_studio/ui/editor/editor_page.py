"""컷편집 페이지: 워크스페이스 · 미리보기 · 속성 · 도구모음 · 타임라인 (V1 자석식 + 위 줄).

편집할 때마다 새 Project를 History에 쌓고(되돌리기), 마지막 편집 2초 뒤 project.json에 자동 저장한다.
편집 기능은 QAction 하나로 도구모음 버튼과 단축키를 함께 만든다 (버튼 툴팁에 단축키 표시).
버튼 없는 단축키: Space 재생/정지 · ←/→ 1프레임 · Shift+←/→ 1초 · Home/End 처음/끝
미리보기: 선택한 클립에 노란 테두리 · 끌면 위치 이동 · Ctrl+휠 = 크기 (화면 속 화면)
"""

import logging
import shutil
import threading
import time
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QTabWidget,
)

from ks_studio.core.edits import (
    add_media,
    clips_using,
    delete_clips,
    delete_range,
    delete_ranges,
    delete_track,
    duplicate,
    edit_points,
    follow_main,
    insert_clip,
    paste,
    place_overlay,
    remove_from_workspace,
    set_audio_offset,
    set_clip_props,
    set_fps,
    set_linked,
    set_subtitles,
    split,
    toggle_marker,
    toggle_track,
    toggle_transition,
    trim_to,
)
from ks_studio.core.history import History
from ks_studio.core.project import MAIN_TRACK, TIMELINE_FPS_CHOICES, Clip, Project, timeline_fps_for
from ks_studio.core.settings import MEDIA_COPY, AppSettings
from ks_studio.core.workspace import Workspace
from ks_studio.services.cleanup import derived_files, remove_files, size_of
from ks_studio.services.optimize import Optimizer, Plan, inspect
from ks_studio.services.playback import Player
from ks_studio.services.render import canvas_size, layer_rect
from ks_studio.services.sources import SourceCache, optimized_path, probe
from ks_studio.core.subtitles import to_srt
from ks_studio.services.render import TimelineRenderer
from ks_studio.ui.editor.caption_panel import CaptionPanel
from ks_studio.ui.editor.export_dialog import ExportDialog
from ks_studio.ui.settings.shortcuts import shortcut_name
from ks_studio.ui.editor.inspector import Inspector
from ks_studio.ui.editor.media_bin import BinItem, MediaBin
from ks_studio.ui.editor.silence_dialog import SilenceDialog
from ks_studio.ui.editor.timeline import TimelineWidget, format_frames
from ks_studio.ui.widgets.layout import stage_layout
from ks_studio.ui.widgets.status_bar import ERROR, INFO, WARNING
from ks_studio.ui.widgets.video_view import VideoView

log = logging.getLogger(__name__)
AUTOSAVE_MS = 2000
DISSOLVE_MS = 500  # 디졸브 버튼의 기본 길이
CAPTION_KEY = Path("<timeline>")  # 음성 인식 작업 이름 (타임라인 소리)
REMOVE_RETRY_MS, REMOVE_RETRIES = 3000, 5  # 워크스페이스에서 뺄 때 잠깐 열려 있던 파일 다시 지우기


class _Copier(QObject):
    """가져온 미디어를 프로젝트 media/로 복사 (큰 파일도 UI가 멈추지 않게 스레드에서)."""

    done = Signal(Path, Path)  # 원본, 복사본
    failed = Signal(Path, str)

    def copy(self, src: Path, dest: Path) -> None:
        threading.Thread(target=self._run, args=(src, dest), daemon=True, name="media-copy").start()

    def _run(self, src: Path, dest: Path) -> None:
        tmp = dest.with_name(dest.name + ".part")
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, tmp)
            tmp.replace(dest)
        except OSError as e:
            tmp.unlink(missing_ok=True)
            self.failed.emit(src, str(e))
            return
        self.done.emit(src, dest)


def _unique_path(folder: Path, name: str) -> Path:
    candidate = folder / name
    stem, suffix, n = candidate.stem, candidate.suffix, 1
    while candidate.exists():
        candidate = folder / f"{stem}_{n}{suffix}"
        n += 1
    return candidate


class EditorPage(QWidget):
    message = Signal(str, str)  # 상태 표시줄 알림 (글, 수준)
    indicator = Signal(str, str, str)  # 상태 표시줄 오른쪽 표시 (키, 글, 수준)

    def __init__(self, sources: SourceCache, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sources = sources
        self._settings = AppSettings()
        self._workspace: Workspace | None = None
        self._history = History(Project())
        self._dirty = False
        self._copying: dict[Path, Path] = {}  # 복사본 경로 → 원본
        self._clipboard: list = []  # 복사한 클립 (Placed). 붙여넣을 때 새 id를 받는다
        # 편집용 사본: 원본마다 검사 결과(None = 필요 없음)와 진행률·실패
        self.optimizer = Optimizer(self)
        self._plans: dict[Path, Plan | None] = {}
        self._optimize_progress: dict[Path, int] = {}
        self._optimize_failed: dict[Path, str] = {}

        self.bin = MediaBin(sources)
        self.view = VideoView(mirrored=False)  # 편집 화면은 결과물 그대로 보여준다
        self.player = Player(sources, self)
        self.inspector = Inspector()
        self.timeline = TimelineWidget(sources)
        self.play_button = QPushButton("▶ 재생")
        self.time_label = QLabel()
        self.export_button = QPushButton("내보내기 (임시)")
        self.fps_combo = QComboBox()
        for fps in TIMELINE_FPS_CHOICES:
            self.fps_combo.addItem(f"타임라인 {fps}fps", fps)
        self.fps_combo.setToolTip("원본 fps가 달라도 섞어 쓸 수 있습니다 (60→30은 건너뛰고, 30→60은 같은 프레임을 두 번 씀)")
        self.fps_combo.currentIndexChanged.connect(lambda _i: self._change_fps(self.fps_combo.currentData()))
        for widget in (self.play_button, self.export_button, self.fps_combo):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space가 버튼 클릭이 아니라 재생 단축키로 가게
        self._copier = _Copier(self)
        self._autosave = QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(AUTOSAVE_MS)
        self._autosave.timeout.connect(self.save)
        self._active = False
        self._shown = 0  # 재생 중 화면에 낸 프레임 수 (1초마다 재생 fps로 알린다)
        self._shown_since = 0.0
        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(1000)
        self._fps_timer.timeout.connect(self._report_fps)
        self._fps_timer.start()

        self.player.frame_ready.connect(self._on_frame)
        self.player.position_changed.connect(self._on_position)
        self.player.playing_changed.connect(lambda playing: self.play_button.setText("❚❚ 정지" if playing else "▶ 재생"))
        self.player.playing_changed.connect(lambda playing: self.view.set_box(None) if playing else self._update_view_box())
        self.play_button.clicked.connect(self.player.toggle)
        self.export_button.clicked.connect(self._export)
        self.timeline.seek_requested.connect(self.player.seek)
        self.timeline.selection_changed.connect(lambda _id: self._show_selected())
        self.timeline.selection_changed.connect(lambda _id: self._update_clip_actions())
        self.timeline.edited.connect(self._apply)
        self.timeline.track_toggled.connect(self._toggle_track)
        self.timeline.track_delete_requested.connect(self._delete_track)
        self.timeline.clip_menu_requested.connect(self._show_clip_menu)
        self.timeline.media_dropped.connect(lambda path, track, position: self._insert_media(Path(path), track, position))
        self.inspector.offset_changed.connect(lambda clip_id, ms: self._apply(set_audio_offset(self.project, clip_id, ms)))
        self.inspector.prop_changed.connect(self._change_clip_prop)
        self.view.installEventFilter(self)  # 미리보기에서 끌기·Ctrl+휠로 화면 속 화면 조절, 크기 바뀜
        for kind in (QAbstractButton, QAbstractItemView, QComboBox):  # 스페이스(재생)·화살표(프레임 이동)를 가로채지 않게
            for widget in self.findChildren(kind):
                widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._view_resize_timer = QTimer(self)
        self._view_resize_timer.setSingleShot(True)
        self._view_resize_timer.setInterval(150)
        self._view_resize_timer.timeout.connect(
            lambda: self.player.set_view_box(self.view.width() * self.view.devicePixelRatioF(),
                                             self.view.height() * self.view.devicePixelRatioF())
        )
        self._view_drag: dict | None = None
        self.bin.add_requested.connect(lambda path: self.append_media(Path(path)))
        self.bin.import_requested.connect(self._import)
        self.bin.remove_requested.connect(self._remove_from_workspace)
        self._copier.done.connect(self._on_copied)
        self.optimizer.progress.connect(self._on_optimize_progress)
        self.optimizer.finished.connect(self._on_optimized)
        self.optimizer.failed.connect(self._on_optimize_failed)
        self._copier.failed.connect(self._on_copy_failed)

        self._shortcuts: dict[str, tuple] = {}  # 이름 → (동작, 기본 키, 설명)
        toolbar = self._build_toolbar()

        transport = QHBoxLayout()
        transport.addWidget(self.play_button)
        transport.addWidget(self.time_label)
        transport.addStretch(1)
        transport.addWidget(self.fps_combo)
        transport.addWidget(self.export_button)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self.view, 1)
        center_layout.addLayout(transport)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)
        bottom_layout.addWidget(toolbar)
        bottom_layout.addWidget(self.timeline, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.captions = CaptionPanel()
        self.captions.edited.connect(lambda subs: self._apply(set_subtitles(self.project, subs)))
        self.captions.seek_requested.connect(lambda seconds: self.player.seek(round(seconds * self.project.fps)))
        self.captions.generate_requested.connect(self._generate_captions)
        self.captions.srt_requested.connect(self._save_srt)
        self.timeline.caption_clicked.connect(self._on_caption_clicked)
        self._speech = None  # 음성 인식 (처음 쓸 때 연다)
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.inspector, "속성")
        self.right_tabs.addTab(self.captions, "자막")
        layout.addWidget(stage_layout(left=self.bin, center=center, right=self.right_tabs, bottom=bottom))
        self._refresh()

    def _action(self, text: str, tip: str, slot, keys: tuple[str, ...] = (), checkable: bool = False) -> QAction:
        """도구모음 버튼과 단축키를 함께 만드는 동작. 페이지에 붙여 두므로 버튼이 없어도 단축키가 동작한다.

        단축키는 ⚙ 설정의 '단축키' 탭에서 바꾼다 — 이름(버튼 글자에서 기호를 뺀 것)으로 저장한다.
        """
        action = QAction(text, self)
        name = shortcut_name(text)
        self._shortcuts[name] = (action, tuple(keys), tip)
        self._set_keys(name, list(keys))
        action.setCheckable(checkable)
        if checkable:
            action.toggled.connect(slot)
        else:
            action.triggered.connect(lambda _checked=False: slot())
        self.addAction(action)
        return action

    def _set_keys(self, name: str, keys: list[str]) -> None:
        action, _defaults, tip = self._shortcuts[name]
        action.setShortcuts([QKeySequence(k) for k in keys])
        shown = " / ".join(QKeySequence(k).toString(QKeySequence.SequenceFormat.NativeText) for k in keys)
        action.setToolTip(f"{tip}  [{shown}]" if keys else tip)

    def shortcut_catalog(self) -> list[tuple[str, str, tuple[str, ...]]]:
        """설정 창에 보일 단축키 목록: (이름, 설명, 기본 키)."""
        return [(name, tip, defaults) for name, (_action, defaults, tip) in self._shortcuts.items()]

    def apply_shortcuts(self, overrides: dict[str, list[str]]) -> None:
        for name, (_action, defaults, _tip) in self._shortcuts.items():
            self._set_keys(name, list(overrides.get(name, defaults)))

    def _build_toolbar(self) -> QWidget:
        """타임라인 위 도구모음: [편집] | [구간] | [이동] ··· [보기]."""
        a = self._action
        self.undo_action = a("↶ 되돌리기", "되돌리기", self.undo, ("Ctrl+Z",))
        self.redo_action = a("↷ 다시 하기", "다시 하기", self.redo, ("Ctrl+Shift+Z", "Ctrl+Y"))
        self.split_action = split_action = a("✂ 분할", "재생 헤드에서 나누기 (선택한 클립, 선택이 없으면 재생 헤드에 걸린 모든 줄)", self._split, ("Ctrl+B",))
        self.trim_left_action = trim_left = a("⇤ 앞 자르기", "재생 헤드 앞부분 잘라내기 (선택한 클립, 선택이 없으면 V1). V1은 빈자리가 닫힘", lambda: self._trim("left"), ("Q",))
        self.trim_right_action = trim_right = a("뒤 자르기 ⇥", "재생 헤드 뒷부분 잘라내기 (선택한 클립, 선택이 없으면 V1)", lambda: self._trim("right"), ("W",))
        self.delete_action = delete = a("🗑 삭제", "선택한 클립 삭제 (V1은 빈자리가 닫힘)", self._delete_selected, ("Del",))
        self.duplicate_action = a("⧉ 복제", "선택한 클립 바로 뒤에 복제", self._duplicate, ("Ctrl+D",))
        self.copy_action = a("복사", "선택한 클립 복사", self._copy, ("Ctrl+C",))
        self.paste_action = a("붙여넣기", "재생 헤드 위치에 붙여넣기 (V1은 편집점에, 위 줄은 간격 유지)", self._paste, ("Ctrl+V",))
        a("모두 선택", "모든 클립 선택", self._select_all, ("Ctrl+A",))
        mark_in = a("[ 구간 시작", "재생 헤드를 구간 시작으로", self._mark_in, ("I",))
        mark_out = a("구간 끝 ]", "재생 헤드까지를 구간 끝으로", self._mark_out, ("O",))
        self.delete_range_action = a("구간 지우기", "모든 줄에서 구간을 잘라내고 뒤를 당기기", self._delete_range, ("Shift+Del",))
        self.clear_range_action = a("구간 해제", "구간 표시 지우기", lambda: self._set_range(None, None), ("Ctrl+Shift+X",))
        dissolve = a("◇ 디졸브", f"재생 헤드에서 가장 가까운 V1 클립 경계에 디졸브 넣기/빼기 ({DISSOLVE_MS}ms, 속성 창에서 길이 조절)", self._toggle_dissolve, ("T",))
        marker = a("◆ 마커", "재생 헤드에 마커 놓기/지우기 (↑/↓로 마커에도 멈춤)", lambda: self._apply(toggle_marker(self.project, self.player.position)), ("M",))
        silence = a("🔇 무음 자르기…", "V1에서 말이 없는 구간을 찾아 한 번에 잘라내기 (기준을 정하고 목록을 확인한 뒤)", self._cut_silence)
        prev_cut = a("⏮ 이전 컷", "이전 편집점(클립 시작·끝)으로", lambda: self._jump_edit(-1), ("Up",))
        next_cut = a("다음 컷 ⏭", "다음 편집점(클립 시작·끝)으로", lambda: self._jump_edit(1), ("Down",))
        snap = a("🧲 스냅", "클립을 옮기거나 자를 때 재생 헤드·클립 끝에 달라붙기", self._set_snapping, ("S",), checkable=True)
        snap.setChecked(True)
        self.link_action = a(
            "🔗 연동",
            "켜면 V1 클립을 지우거나 옮기고 자를 때, 그 위 줄 클립이 같은 장면을 따라 움직입니다",
            lambda on: self._apply(set_linked(self.project, on)),
            checkable=True,
        )
        zoom_out = a("－", "축소", lambda: self.timeline.zoom(1 / 1.25), ("-",))
        zoom_in = a("＋", "확대", lambda: self.timeline.zoom(1.25), ("=",))
        fit = a("⤢ 맞춤", "타임라인 전체를 화면에 맞추기", self.timeline.fit, ("\\",))
        # 버튼 없이 단축키만
        a("재생/정지", "재생/정지", self.player.toggle, ("Space",))
        a("1프레임 뒤로", "1프레임 뒤로", lambda: self.player.step(-1), ("Left",))
        a("1프레임 앞으로", "1프레임 앞으로", lambda: self.player.step(1), ("Right",))
        a("1초 뒤로", "1초 뒤로", lambda: self.player.step(-self.project.fps), ("Shift+Left",))
        a("1초 앞으로", "1초 앞으로", lambda: self.player.step(self.project.fps), ("Shift+Right",))
        a("처음으로", "처음으로", lambda: self.player.seek(0), ("Home",))
        a("끝으로", "끝으로", lambda: self.player.seek(self.project.duration - 1), ("End",))

        groups = [
            [self.undo_action, self.redo_action],
            [split_action, trim_left, trim_right, delete, self.duplicate_action],
            [mark_in, mark_out, self.delete_range_action, self.clear_range_action, silence],
            [dissolve, marker],
            [prev_cut, next_cut],
            None,  # 나머지는 오른쪽으로
            [snap, self.link_action],
            [zoom_out, zoom_in, fit],
        ]
        bar = QWidget()
        bar.setObjectName("editToolbar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        for i, group in enumerate(groups):
            if group is None:
                row.addStretch(1)
                continue
            if i > 0 and groups[i - 1] is not None:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.VLine)
                line.setObjectName("toolbarSeparator")
                row.addWidget(line)
            for action in group:
                button = QToolButton()
                button.setDefaultAction(action)
                button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # Space가 버튼이 아니라 재생으로 가게
                row.addWidget(button)
        return bar

    @property
    def project(self) -> Project:
        return self._history.present

    # ── 페이지 수명 ──────────────────────────────────────────────

    def activate(self, settings: AppSettings, workspace: Workspace) -> None:
        self._settings = settings
        self.apply_shortcuts(settings.shortcuts)
        self._active = True
        if workspace != self._workspace:
            self.save()
            self._workspace = workspace
            self._sources.set_cache_dir(workspace.cache_dir)
            self._history = History(Project.load(workspace.project_file))
            self._set_range(None, None)
            self._dirty = False
            self._refresh(fit=True)
            self.player.seek(0)
        else:
            self._refresh_bin()

    def deactivate(self) -> None:
        self._active = False
        self.player.pause()
        self.save()
        self.indicator.emit("fps", "", INFO)

    def save(self) -> None:
        self._autosave.stop()
        if self._dirty and self._workspace is not None:
            self.project.save(self._workspace.project_file)
            self._dirty = False

    # ── 편집 ────────────────────────────────────────────────────

    def _apply(self, project: Project) -> None:
        """모든 편집이 여기를 지난다. 연동이 켜져 있으면 V1 편집에 위 줄이 따라가고, 둘이 한 번의 되돌리기로 묶인다."""
        edited, project = project, follow_main(self.project, project)
        if project.video_tracks > edited.video_tracks:
            self.message.emit(f"연동: 따라가던 클립이 겹쳐서 V{project.video_tracks} 줄을 새로 만들었습니다", INFO)
        if self._history.push(project):
            self._changed()

    def undo(self) -> None:
        if self._history.undo():
            self._changed()

    def redo(self) -> None:
        if self._history.redo():
            self._changed()

    def _changed(self) -> None:
        self._dirty = True
        self._autosave.start()
        self._refresh()

    def _refresh(self, fit: bool = False) -> None:
        if self._workspace is not None:
            self.timeline.set_project(self.project, self._workspace, fit=fit)
            self.player.set_project(self.project, self._workspace)
        self.captions.set_subtitles(self.project.subtitles)
        self._show_caption(self.player.position)
        self.export_button.setEnabled(self.project.duration > 0)
        self.fps_combo.blockSignals(True)
        self.fps_combo.setCurrentIndex(max(0, self.fps_combo.findData(self.project.fps)))
        self.fps_combo.blockSignals(False)
        self.link_action.blockSignals(True)
        self.link_action.setChecked(self.project.linked)
        self.link_action.blockSignals(False)
        self.undo_action.setEnabled(self._history.can_undo)
        self.redo_action.setEnabled(self._history.can_redo)
        self._update_range_actions()
        self._update_clip_actions()
        self._show_selected()
        self._on_position(self.player.position)
        self._refresh_bin()

    def _show_selected(self) -> None:
        placed = self.project.find(self.timeline.selected) if self.timeline.selected else None
        self.inspector.show_clip(placed.clip if placed else None, self.project.fps)
        self._update_view_box()

    def _split(self) -> None:
        """선택한 클립이 재생 헤드에 걸려 있으면 그것만, 아니면 재생 헤드에 걸린 모든 줄을 나눈다."""
        frame = self.player.position
        placed = self.project.find(self.timeline.selected) if self.timeline.selected else None
        only = placed.clip.id if placed and placed.start < frame < placed.end else None
        self._apply(split(self.project, frame, only))

    def _clip_at_playhead(self):
        """Q/W 대상: 선택한 클립이 재생 헤드에 걸려 있으면 그것, 아니면 재생 헤드 아래 V1 클립."""
        frame = self.player.position
        selected = self.project.find(self.timeline.selected) if self.timeline.selected else None
        if selected and selected.start < frame < selected.end:
            return selected
        return next((p for p in self.project.placed() if p.track == MAIN_TRACK and p.start <= frame < p.end), None)

    def _trim(self, side: str) -> None:
        target = self._clip_at_playhead()
        edited = trim_to(self.project, self.player.position, side, target.clip.id) if target else self.project
        if edited is self.project:
            self.message.emit("재생 헤드가 클립 한가운데에 있어야 자를 수 있습니다", INFO)
            return
        self._apply(edited)
        if side == "left" and target.track == MAIN_TRACK:  # 앞이 당겨졌으니 잘린 자리(클립 시작)로
            self.player.seek(target.start)

    def _set_range(self, start: int | None, end: int | None) -> None:
        self.timeline.set_range(start, end)
        self._update_range_actions()

    def _update_range_actions(self) -> None:
        marked = self.timeline.range_in is not None or self.timeline.range_out is not None
        self.delete_range_action.setEnabled(marked)
        self.clear_range_action.setEnabled(marked)

    def _mark_in(self) -> None:
        start, end = self.player.position, self.timeline.range_out
        self._set_range(start, end if end is not None and end > start else None)

    def _mark_out(self) -> None:
        start, end = self.timeline.range_in, self.player.position + 1  # 재생 헤드 프레임까지 포함
        self._set_range(start if start is not None and start < end else None, end)

    def _delete_range(self) -> None:
        start, end = self.timeline.range_in, self.timeline.range_out
        if start is None and end is None:
            return
        start = start if start is not None else 0
        end = end if end is not None else self.project.duration
        self._apply(delete_range(self.project, start, end))
        self.message.emit(f"구간 지우기: {(end - start) / self.project.fps:.2f}초를 잘라냈습니다", INFO)
        self._set_range(None, None)
        self.player.seek(start)

    def _cut_silence(self) -> None:
        if self._workspace is None or not self.project.clips:
            self.message.emit("V1에 클립이 있어야 무음을 찾을 수 있습니다", INFO)
            return
        self.player.pause()
        dialog = SilenceDialog(self.project, self._workspace, self._sources, self)
        accepted = dialog.exec() == SilenceDialog.DialogCode.Accepted
        ranges = dialog.ranges
        dialog.deleteLater()
        if not accepted or not ranges:
            return
        total = sum(b - a for a, b in ranges) / self.project.fps
        self._apply(delete_ranges(self.project, ranges))
        self.message.emit(f"무음 자르기: {len(ranges)}곳, {total:.1f}초를 잘라냈습니다 (Ctrl+Z로 되돌리기)", INFO)

    def _toggle_dissolve(self) -> None:
        edited, clip_id = toggle_transition(self.project, self.player.position, DISSOLVE_MS)
        if clip_id is None:
            self.message.emit("디졸브는 V1에 클립이 두 개 이상 있어야 넣을 수 있습니다", INFO)
            return
        self._apply(edited)
        added = self.project.find(clip_id).clip.transition_ms > 0
        self.message.emit(f"디졸브 {'넣음' if added else '뺌'} ({DISSOLVE_MS}ms, 겹친 만큼 전체가 짧아짐)" if added else "디졸브를 뺐습니다", INFO)

    def _change_clip_prop(self, name: str, value: float) -> None:
        """속성 창의 볼륨·페이드·화면·디졸브 → 선택한 클립 전부."""
        ids = self.timeline.selection or ({self.timeline.selected} if self.timeline.selected else set())
        if ids:
            self._apply(set_clip_props(self.project, ids, **{name: int(value) if name.endswith("_ms") else value}))

    def _jump_edit(self, direction: int) -> None:
        frame = self.player.position
        points = [p for p in edit_points(self.project) if (p < frame if direction < 0 else p > frame)]
        if points:
            self.player.seek(max(points) if direction < 0 else min(points))

    def _set_snapping(self, on: bool) -> None:
        self.timeline.snapping = on

    def _change_fps(self, fps: int) -> None:
        """타임라인 fps를 바꿔도 재생 헤드는 같은 시각에 둔다."""
        old = self.project.fps
        if fps == old:
            return
        position = round(self.player.position * fps / old)
        self._set_range(None, None)  # 구간은 타임라인 프레임이라 fps가 바뀌면 의미가 달라진다
        self._apply(set_fps(self.project, fps))
        self.player.seek(position)

    def _toggle_track(self, track: int, kind: str) -> None:
        """👁·🔊 버튼. 소리를 켰다 껐다 비교할 수 있게, 재생 중이면 그 자리에서 이어서 재생한다."""
        playing = self.player.playing
        self._apply(toggle_track(self.project, track, kind))
        if playing:
            self.player.play()

    def _delete_track(self, track: int) -> None:
        """클립이 있는 줄이면 같이 지워진다고 확인을 받는다. 빈 줄은 바로 지운다."""
        count = sum(1 for p in self.project.placed() if p.track == track)
        if count:
            answer = QMessageBox.question(
                self,
                "줄 삭제",
                f"V{track} 줄에 있는 클립 {count}개도 같이 지워집니다.\n삭제할까요? (Ctrl+Z로 되돌릴 수 있습니다)",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._apply(delete_track(self.project, track))

    def _delete_selected(self) -> None:
        if self.timeline.selection:
            self._apply(delete_clips(self.project, self.timeline.selection))

    def _copy(self) -> None:
        self._clipboard = [p for p in self.project.placed() if p.clip.id in self.timeline.selection]
        self._update_clip_actions()

    def _paste(self) -> None:
        count = self._apply_and_select_new(paste(self.project, self._clipboard, self.player.position))
        self.message.emit(f"붙여넣기: 클립 {count}개", INFO)

    def _duplicate(self) -> None:
        count = self._apply_and_select_new(duplicate(self.project, self.timeline.selection))
        self.message.emit(f"복제: 클립 {count}개", INFO)

    def _apply_and_select_new(self, edited: Project) -> int:
        """붙여넣기·복제 뒤에는 새로 생긴 클립을 선택해 둔다 (바로 옮기거나 지울 수 있게). 새 클립 수를 돌려준다."""
        before = {p.clip.id for p in self.project.placed()}
        self._apply(edited)
        new = {p.clip.id for p in self.project.placed()} - before
        if new:
            self.timeline.set_selection(new, next(iter(sorted(new))))
        return len(new)

    def _select_all(self) -> None:
        ids = {p.clip.id for p in self.project.placed()}
        self.timeline.set_selection(ids, self.timeline.selected if self.timeline.selected in ids else next(iter(sorted(ids)), None))

    def _update_clip_actions(self) -> None:
        selected = bool(self.timeline.selection)
        for action in (self.delete_action, self.duplicate_action, self.copy_action):
            action.setEnabled(selected)
        self.paste_action.setEnabled(bool(self._clipboard))

    def _show_clip_menu(self, pos) -> None:
        """클립·빈 곳 우클릭 메뉴. 도구모음과 같은 동작을 그대로 쓴다."""
        self._update_clip_actions()
        menu = QMenu(self)
        for action in (self.copy_action, self.paste_action, self.duplicate_action, None,
                       self.split_action, self.trim_left_action, self.trim_right_action, None, self.delete_action):
            if action is None:
                menu.addSeparator()
            else:
                menu.addAction(action)
        menu.exec(pos)

    def append_media(self, path: Path) -> None:
        """재생 헤드가 있는 클립 경계에 넣는다 (클립 한가운데면 그 클립 뒤)."""
        found = self.project.locate(self.player.position)
        if found is None:
            index = len(self.project.clips)
        else:
            index = found[0] if found[1] == 0 else found[0] + 1
        self._insert_media(path, MAIN_TRACK, index)

    def _insert_media(self, path: Path, track: int, position: int) -> None:
        """track이 V1이면 position = 끼워 넣을 순서, 위 줄이면 시작 프레임."""
        if self._workspace is None:
            return
        if self._optimize_state(path)[1]:
            self.message.emit(f"{path.name}: 편집용 사본이 준비되면 넣을 수 있습니다", WARNING)
            return
        info = self._sources.info(path)
        if info is None or info.frames == 0:
            QMessageBox.warning(self, "넣을 수 없음", f"영상을 읽을 수 없습니다:\n{path}")
            return
        project = self.project
        if project.duration == 0:  # 빈 타임라인이면 첫 원본의 fps(30/60)를 따른다
            project = set_fps(project, timeline_fps_for(info.fps))
        clip = Clip(self._workspace.to_source(path), 0, info.frames, fps=info.fps)
        if track == MAIN_TRACK:
            edited = insert_clip(project, position, clip)
        else:
            edited = place_overlay(project, clip, track, position)
            if edited is project:  # 끌어 놓는 사이 fps가 바뀌어 길이가 달라진 경우 등
                return
        self._apply(edited)
        self.timeline.select(clip.id)
        self.player.seek(self.project.find(clip.id).start)

    # ── 미디어 가져오기 ──────────────────────────────────────────

    def _import(self, paths: list[Path]) -> None:
        if self._workspace is None:
            return
        rejected = []
        for path in paths:
            if probe(path) is None:
                rejected.append(path.name)
                continue
            inside = self._workspace.to_source(path) != str(path.resolve())
            if inside or self._settings.media_import != MEDIA_COPY:
                self._apply(add_media(self.project, self._workspace.to_source(path)))
                continue
            dest = _unique_path(self._workspace.media_dir, path.name)
            self._copying[dest] = path
            self._copier.copy(path, dest)
        self._refresh_bin()
        if rejected:
            QMessageBox.warning(self, "가져올 수 없음", "영상이 아니거나 읽을 수 없는 파일:\n" + "\n".join(rejected))

    def _on_copied(self, _src: Path, dest: Path) -> None:
        self._copying.pop(dest, None)
        self._apply(add_media(self.project, self._workspace.to_source(dest)))
        self._refresh_bin()
        self.message.emit(f"가져오기 완료: {dest.name}", INFO)

    def _on_copy_failed(self, src: Path, message: str) -> None:
        for dest, original in list(self._copying.items()):
            if original == src:
                self._copying.pop(dest)
        self._refresh_bin()
        self.message.emit(f"⚠ 가져오기 복사 실패: {src.name} — {message}", ERROR)
        QMessageBox.warning(self, "복사 실패", f"{src.name}: {message}")

    # ── 편집용 사본 ──────────────────────────────────────────────

    def _optimize_state(self, path: Path) -> tuple[str, bool]:
        """(워크스페이스에 보일 상태 줄, 타임라인에 못 넣는지). 사본이 필요한데 없으면 만들기를 요청한다."""
        if self._workspace is None or not path.exists():
            return "", False
        if path not in self._plans:
            self._plans[path] = inspect(path)  # 파일 머리만 읽어 금방 끝난다
        plan = self._plans[path]
        if plan is None:
            return "", False
        reasons = "·".join(plan.reasons)
        if optimized_path(self._workspace.cache_dir, path).exists():
            return f"편집용 사본 ✓ ({reasons})", False
        if path in self._optimize_failed:
            return f"⚠ 편집용 사본 실패: {self._optimize_failed[path]}", True
        if not self.optimizer.is_busy(path):
            self.optimizer.request(path, optimized_path(self._workspace.cache_dir, path), plan)
        percent = self._optimize_progress.get(path, 0)
        return f"편집용 사본 만드는 중 {percent}% ({reasons})", True

    def _on_optimize_progress(self, path: Path, ratio: float) -> None:
        percent = int(ratio * 100)
        if self._optimize_progress.get(path) == percent:
            return
        self._optimize_progress[path] = percent
        note, blocked = self._optimize_state(path)
        self.bin.set_note(path, note, blocked, self.project.fps)

    def _on_optimized(self, path: Path, _output: Path) -> None:
        self._optimize_progress.pop(path, None)
        plan = self._plans.get(path)
        self.message.emit(f"편집용 사본 준비: {path.name} ({'·'.join(plan.reasons) if plan else ''})", INFO)
        self._refresh()  # 썸네일·파형·정지 화면을 사본으로 다시

    def _on_optimize_failed(self, path: Path, message: str) -> None:
        self._optimize_progress.pop(path, None)
        self._optimize_failed[path] = message
        self.message.emit(f"⚠ 편집용 사본 실패: {path.name} — {message}", ERROR)
        self._refresh_bin()

    def _refresh_bin(self) -> None:
        if self._workspace is None:
            return
        ws = self._workspace
        recordings = [p for p in ws.recordings() if ws.to_source(p) not in self.project.removed]
        items = [BinItem(p, ws.to_source(p), imported=False) for p in recordings]  # 녹화본은 처음부터 편집하기 좋은 형식
        items += [BinItem(p, ws.to_source(p), imported=False, note="AI 보정본") for p in ws.beautified()]
        for source in self.project.media:
            path = ws.resolve(source)
            note, blocked = self._optimize_state(path)
            items.append(BinItem(path, source, imported=True, note=note, blocked=blocked))
        items += [BinItem(dest, ws.to_source(dest), imported=True, copying=True) for dest in self._copying]
        self.bin.show_items(items, self.project.fps)

    def _remove_from_workspace(self, path_text: str) -> None:
        """워크스페이스에서 빼기: 목록에서 빼고, 그 영상을 쓰는 클립과 그 영상으로 만든 사본·캐시를 지운다.

        원본은 지우지 않는다. 단 AI 보정본은 그 자체가 만든 파일이라 지운다 (AI 보정 탭에서 다시 만들 수 있다).
        목록·클립은 Ctrl+Z로 되돌릴 수 있고, 지운 사본·캐시는 필요하면 다시 만들어진다.
        """
        ws, path = self._workspace, Path(path_text)
        if self.optimizer.is_busy(path):
            QMessageBox.information(self, "워크스페이스에서 빼기", "편집용 사본을 만드는 중이라 지금은 뺄 수 없습니다.\n끝난 뒤 다시 해 주세요.")
            return
        source = ws.to_source(path)
        made_by_ai = path.resolve().parent == ws.ai_dir.resolve()
        recording = path.resolve().parent == ws.recordings_dir.resolve()
        derived = derived_files(ws.cache_dir, path)
        files = derived + ([path] if made_by_ai and path.exists() else [])
        clips = len(clips_using(self.project, source))
        lines = [f"'{path.name}'을(를) 워크스페이스에서 뺍니다.", ""]
        lines.append("• AI 보정본 파일을 지웁니다 (AI 보정 탭에서 다시 만들 수 있습니다)" if made_by_ai else "• 원본 파일은 그대로 둡니다")
        if derived:
            lines.append(f"• 이 영상으로 만든 편집용 사본·캐시 {size_of(derived) / 1e6:.0f}MB를 지웁니다")
        if clips:
            lines.append(f"• 타임라인에서 이 영상을 쓰는 클립 {clips}개도 지웁니다 (Ctrl+Z로 되돌릴 수 있습니다)")
        if QMessageBox.question(self, "워크스페이스에서 빼기", "\n".join(lines)) != QMessageBox.StandardButton.Yes:
            return
        self._apply(remove_from_workspace(self.project, source, recording=recording))
        freed, failed = remove_files(files)
        self._refresh_bin()
        self.message.emit(f"워크스페이스에서 뺐습니다: {path.name}" + (f" · {freed / 1e6:.0f}MB 지움" if freed else ""), INFO)
        if failed:
            self._retry_remove(failed, REMOVE_RETRIES)

    def _retry_remove(self, files: list[Path], tries: int) -> None:
        """썸네일·파형을 만드느라 잠깐 열려 있던 파일은 조금 뒤 다시 지운다. 끝내 못 지우면 다음에 프로젝트를 열 때 지운다."""
        def again() -> None:
            _freed, failed = remove_files(files)
            if failed and tries > 1:
                self._retry_remove(failed, tries - 1)

        QTimer.singleShot(REMOVE_RETRY_MS, again)

    # ── 재생·내보내기 ────────────────────────────────────────────

    def _report_fps(self) -> None:
        """재생 중이면 실제로 화면에 낸 fps, 멈춰 있으면 타임라인 fps."""
        if not self._active:
            return
        now = time.monotonic()
        shown, elapsed = self._shown, now - self._shown_since
        self._shown, self._shown_since = 0, now
        target = self.project.fps
        if not self.player.playing or elapsed <= 0:
            self.indicator.emit("fps", f"타임라인 {target}fps", INFO)
            return
        fps = shown / elapsed
        if fps < 0.9 * target:
            self.indicator.emit("fps", f"재생 {fps:.0f}/{target}fps ⚠ 미리보기가 밀림 (내보내기에는 영향 없음)", WARNING)
        else:
            self.indicator.emit("fps", f"재생 {fps:.0f}/{target}fps", INFO)

    def _on_frame(self, rgb) -> None:
        if self.player.playing:
            self._shown += 1
        if rgb is None:
            self.view.set_message("워크스페이스에서 영상을 더블클릭하거나 타임라인으로 끌어 놓으세요")
        else:
            self.view.show_rgb(rgb)

    # ── 미리보기에서 화면 속 화면 조절 ───────────────────────────

    def _screen_target(self):
        """미리보기에서 조절할 클립: 선택한 클립이 재생 헤드에 보이는 중이면 그것."""
        placed = self.project.find(self.timeline.selected) if self.timeline.selected else None
        frame = self.player.position
        if placed is None or not placed.start <= frame < placed.end or placed.track in self.project.hidden:
            return None
        return placed

    def _update_view_box(self, project: Project | None = None) -> None:
        """선택한 클립이 화면 어디에 그려지는지 노란 테두리로 (0~1 비율)."""
        project = project or self.project
        target = self._screen_target()
        if target is None or self._workspace is None:
            self.view.set_box(None)
            return
        clip = project.find(target.clip.id).clip
        info = self._sources.info(self._workspace.resolve(clip.source))
        if info is None:
            self.view.set_box(None)
            return
        canvas = canvas_size(project, self._workspace, self._sources)
        x, y, w, h = layer_rect(info.width, info.height, canvas, clip)
        self.view.set_box(QRectF(x / canvas[0], y / canvas[1], w / canvas[0], h / canvas[1]))

    def eventFilter(self, watched, event) -> bool:
        if watched is not self.view:
            return False
        kind = event.type()
        if kind == QEvent.Type.Resize:  # 미리보기를 칸의 실제 화소 크기로 (창 크기를 바꾸는 동안은 잠깐 모아서)
            self._view_resize_timer.start()
            return False
        if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            target = self._screen_target()
            if target is not None:
                self.player.pause()
                clip = target.clip
                self._view_drag = {"id": clip.id, "start": event.position(), "pos": (clip.pos_x, clip.pos_y), "draft": None}
                return True
        elif kind == QEvent.Type.MouseMove and self._view_drag is not None:
            drag, area = self._view_drag, self.view.image_rect()
            if area.width() > 0:
                moved = event.position() - drag["start"]
                x = min(1.5, max(-0.5, drag["pos"][0] + moved.x() / area.width()))
                y = min(1.5, max(-0.5, drag["pos"][1] + moved.y() / area.height()))
                drag["draft"] = set_clip_props(self.project, {drag["id"]}, pos_x=round(x, 4), pos_y=round(y, 4))
                self.player.set_project(drag["draft"], self._workspace)  # 끄는 동안은 화면만 다시 그린다
                self._update_view_box(drag["draft"])
            return True
        elif kind == QEvent.Type.MouseButtonRelease and self._view_drag is not None:
            drag, self._view_drag = self._view_drag, None
            if drag["draft"] is not None:
                self._apply(drag["draft"])  # 손을 뗄 때 한 번만 편집으로 남긴다 (되돌리기 한 번)
            return True
        elif kind == QEvent.Type.Wheel and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            target = self._screen_target()
            if target is not None:
                steps = event.angleDelta().y() / 120
                scale = min(4.0, max(0.05, target.clip.scale * 1.1**steps))
                self._apply(set_clip_props(self.project, {target.clip.id}, scale=round(scale, 4)))
                return True
        return False

    def _show_caption(self, frame: int) -> None:
        """미리보기에 그 시각의 자막 (내보낼 때 굽는 것과 같은 모양) · '자막' 탭에서 그 줄을 칠한다."""
        subs, seconds = self.project.subtitles, frame / self.project.fps
        index = subs.at(seconds)
        self.view.set_caption(subs.cues[index].text if index is not None else "", subs.style)
        self.captions.set_time(seconds)

    # ── 자막 ────────────────────────────────────────────────────

    def _on_caption_clicked(self, index: int) -> None:
        self.right_tabs.setCurrentWidget(self.captions)
        self.captions.select(index)

    def _generate_captions(self, task: str) -> None:
        """타임라인 소리 전체를 알아들어 자막을 만든다 (지금 자막은 바뀐다 — Ctrl+Z로 되돌릴 수 있다)."""
        from ks_studio.services.speech import speech_unavailable_reason

        reason = speech_unavailable_reason()
        if reason is not None:
            self.captions.show_progress(reason)
            return
        if self.project.duration == 0 or self._workspace is None:
            self.captions.show_progress("타임라인에 영상을 먼저 넣으세요")
            return
        if self._speech is None:
            from ks_studio.services.speech import SpeechWorker

            self._speech = SpeechWorker(self)
            self._speech.status.connect(lambda _key, text: self.captions.show_progress(text, 0.0))
            self._speech.progress.connect(lambda _key, r: self.captions.show_progress(f"말 알아듣는 중… {int(r * 100)}%", r))
            self._speech.done.connect(self._on_captions_made)
            self._speech.failed.connect(lambda _key, reason: self.captions.show_progress(f"⚠ 실패: {reason}"))
        project, workspace = self.project, self._workspace
        renderer = TimelineRenderer(project, workspace, self._sources, (64, 36), "rgb24")
        self._caption_task = task
        self.captions.show_progress("타임라인 소리 모으는 중…", 0.0)
        self._speech.request(CAPTION_KEY, task, audio_fn=renderer.audio_track)

    def _on_captions_made(self, _key, cues: list) -> None:
        subs = replace(self.project.subtitles, cues=tuple(cues), task=self._caption_task)
        self._apply(set_subtitles(self.project, subs))
        self.captions.show_progress(f"자막 {len(cues)}줄을 만들었습니다. 표에서 고치세요 (음악만 나오는 곳에 지어낸 줄이 있으면 지우세요)")
        self.right_tabs.setCurrentWidget(self.captions)

    def _save_srt(self) -> None:
        if self._workspace is None:
            return
        default = str(self._workspace.exports_dir / f"{self._workspace.root.name}.srt")
        path, _ = QFileDialog.getSaveFileName(self, "SRT로 저장", default, "자막 (*.srt)")
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(to_srt(self.project.subtitles), encoding="utf-8")
            self.message.emit(f"SRT 저장: {Path(path).name}", INFO)

    def _on_position(self, frame: int) -> None:
        self._show_caption(frame)
        fps = self.project.fps
        self.timeline.set_playhead(frame)
        self.time_label.setText(f"{format_frames(frame, fps)} / {format_frames(self.project.duration, fps)}")
        if self._view_drag is None and not self.player.playing:  # 끄는 중엔 초안 위치로, 재생 중엔 그리지 않는다 (프레임마다 계산하면 재생이 밀린다)
            self._update_view_box()

    def _export(self) -> None:
        if self._workspace is None or self.project.duration == 0:
            return
        self.player.pause()
        self.save()
        size = canvas_size(self.project, self._workspace, self._sources)
        name = f"{self._workspace.root.name}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        output = self._workspace.exports_dir / name
        dialog = ExportDialog(self.project, self._workspace, self._sources, output, size, self)
        dialog.exec()
        dialog.deleteLater()
        if output.exists():
            self.message.emit(f"내보내기 완료: {output.name}", INFO)
