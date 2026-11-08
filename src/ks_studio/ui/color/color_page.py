"""'색 보정' 단계: 원본 영상마다 빛 · 색 · 색상별(HSL) · LUT을 정하고, 원본과 보정본을 나란히 보며 맞춘다.

원본 파일은 그대로 두고 값만 color/settings.json에 저장한다. 컷편집 미리보기·내보내기·지금 화면 저장에서 입힌다
(그 영상의 AI 보정본에도 같은 색이 따라간다 — core.grade.ColorState.grade_for).
AI 패키지(OpenCV 등)가 없어도 돈다 (numpy·PyAV만).
"""

import shutil
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ks_studio.app.paths import COLOR_LOOKS_DIR
from ks_studio.core.grade import (
    BUILTIN_LOOKS,
    COLOR,
    HSL_COLORS,
    LIGHT,
    ColorGrade,
    ColorState,
    apply_image,
    auto_grade,
    build_lut,
    load_color_state,
    load_look,
    read_cube,
    save_color_state,
    save_look,
)
from ks_studio.core.presets import delete_preset, list_presets, valid_name
from ks_studio.core.project import Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.optimize import Optimizer, inspect
from ks_studio.services.sources import SourceCache, optimized_path, probe
from ks_studio.ui.beautify.effect_control import EffectControl
from ks_studio.ui.editor.media_bin import VIDEO_FILTER
from ks_studio.ui.widgets.latest_worker import LatestWorker
from ks_studio.ui.widgets.layout import stage_layout
from ks_studio.ui.widgets.source_row import SourceRow, add_row, facts
from ks_studio.ui.widgets.status_bar import INFO, WARNING

FRAME_LONG_SIDE = 1600  # 비교할 장면은 이 크기로 줄여서 들고 있는다 (4K 그대로면 값을 바꿀 때마다 느리다)
SCRUB_STEPS_PER_SECOND = 10
SAVE_DELAY_MS = 500
REDRAW_DELAY_MS = 40
GAP = 6
MODES = (("side", "나란히"), ("split", "반반"), ("after", "보정만"))
BUILTIN = "builtin:"
AUTO_NOTE = ("Beta 기능: 지금 장면의 평균 색이 회색, 가운데 밝기가 45%가 되게 색온도·색조·노출만 맞춥니다.\n"
             "나무 벽·노을처럼 한 가지 색이 많은 장면은 원래 분위기를 지워 버릴 수 있어요 — 결과를 보고 슬라이더로 다듬으세요.")
HSL_SWATCH = {  # 색상별 단추 색
    "red": "#e53935", "orange": "#fb8c00", "yellow": "#fdd835", "green": "#43a047",
    "aqua": "#26c6da", "blue": "#1e88e5", "purple": "#8e24aa", "magenta": "#d81b60",
}


def grab_frame(path: Path, seconds: float, long_side: int = FRAME_LONG_SIDE) -> np.ndarray | None:
    """영상의 그 시각 장면 (RGB, 긴 변 long_side 이하). 못 읽으면 None."""
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            container.seek(int(max(0.0, seconds) / stream.time_base), stream=stream, backward=True)
            chosen = None
            for frame in container.decode(stream):
                chosen = frame
                if frame.time is not None and frame.time >= seconds - 0.02:
                    break
            if chosen is None:
                return None
            scale = min(1.0, long_side / max(chosen.width, chosen.height))
            w, h = max(2, round(chosen.width * scale) // 2 * 2), max(2, round(chosen.height * scale) // 2 * 2)
            return chosen.reformat(width=w, height=h, format="rgb24").to_ndarray()
    except (av.FFmpegError, OSError, IndexError):
        return None


def _resize(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if (rgb.shape[1], rgb.shape[0]) == (width, height):
        return rgb
    return av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(width=width, height=height).to_ndarray(format="rgb24")


def compare_image(rgb: np.ndarray, table: np.ndarray, mode: str, box: tuple[int, int]) -> np.ndarray:
    """비교 그림: side = 원본 | 보정 나란히, split = 왼쪽 반 원본·오른쪽 반 보정, after = 보정만. box(가로, 세로) 안에 들어가게."""
    halves = 2 if mode == "side" else 1
    room_w = max(2, (box[0] - GAP * (halves - 1)) // halves)
    scale = min(room_w / rgb.shape[1], max(2, box[1]) / rgb.shape[0], 1.0)
    w, h = max(2, round(rgb.shape[1] * scale)), max(2, round(rgb.shape[0] * scale))
    small = _resize(rgb, w, h)
    if mode == "side":
        out = np.full((h, w * 2 + GAP, 3), 24, np.uint8)
        out[:, :w] = small
        out[:, w + GAP :] = apply_image(small, table)
        return out
    if mode == "split":
        out = small.copy()
        out[:, w // 2 :] = apply_image(small[:, w // 2 :], table)
        out[:, max(0, w // 2 - 1) : w // 2 + 1] = 255
        return out
    return apply_image(small, table)


def _pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _swatch(color: str) -> QIcon:
    pixmap = QPixmap(14, 14)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


class ColorPage(QWidget):
    message = Signal(str, str)  # 상태 표시줄 알림
    indicator = Signal(str, str, str)

    def __init__(self, sources: SourceCache, optimizer: Optimizer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sources = sources
        self._optimizer = optimizer  # 컷편집과 같은 편집용 사본 작업자
        self._workspace: Workspace | None = None
        self._state = ColorState()
        self._source: str | None = None
        self._frame: np.ndarray | None = None  # 지금 비교하는 장면 (줄인 RGB)
        self._duration = 0.0
        self._frame_token = self._draw_token = -1
        self._user_luts: dict[str, np.ndarray | None] = {}
        self.looks_dir = COLOR_LOOKS_DIR
        self._frames = LatestWorker("color-frame", self)
        self._frames.ready.connect(self._on_frame)
        self._drawer = LatestWorker("color-preview", self)
        self._drawer.ready.connect(self._on_drawn)
        optimizer.finished.connect(lambda path, _out: self._on_optimized(path))

        # 왼쪽: 원본 목록
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.currentRowChanged.connect(self._on_select)
        import_button = QPushButton("+ 가져오기")
        import_button.clicked.connect(self._import)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel("원본"))
        header.addStretch(1)
        header.addWidget(import_button)
        left_layout.addLayout(header)
        left_layout.addWidget(self.list, 1)
        hint = QLabel("원본 파일은 그대로 두고, 컷편집 미리보기·내보내기·AI 보정본에 이 색을 입힙니다.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        left_layout.addWidget(hint)

        # 가운데: 안내 / 비교 화면
        self.center = QStackedWidget()
        self.notice = QLabel("왼쪽에서 색을 맞출 영상을 고르세요")
        self.notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.notice.setWordWrap(True)
        self.center.addWidget(self.notice)
        compare = QWidget()
        compare_layout = QVBoxLayout(compare)
        compare_layout.setContentsMargins(0, 0, 0, 0)
        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        for index, (key, label) in enumerate(MODES):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("mode", key)
            self.mode_group.addButton(button, index)
            mode_row.addWidget(button)
        self.mode_group.button(0).setChecked(True)
        self.mode_group.idClicked.connect(lambda _i: self._redraw_now())
        self.caption = QLabel("왼쪽 원본 · 오른쪽 보정")
        mode_row.addSpacing(12)
        mode_row.addWidget(self.caption)
        mode_row.addStretch(1)
        compare_layout.addLayout(mode_row)
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)  # 그림이 칸 크기를 정하지 않게
        self.view.setMinimumSize(200, 120)
        compare_layout.addWidget(self.view, 1)
        scrub_row = QHBoxLayout()
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setToolTip("비교할 장면 고르기")
        self.scrub.valueChanged.connect(self._on_scrub)
        self.time_label = QLabel("0:00.0")
        scrub_row.addWidget(self.scrub, 1)
        scrub_row.addWidget(self.time_label)
        compare_layout.addLayout(scrub_row)
        self.center.addWidget(compare)

        right = self._controls()
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self._save)
        self._redraw_timer = QTimer(self)  # 슬라이더를 끄는 동안 너무 자주 그리지 않게
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(REDRAW_DELAY_MS)
        self._redraw_timer.timeout.connect(self._redraw_now)
        self._scrub_timer = QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(80)
        self._scrub_timer.timeout.connect(self._request_frame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(stage_layout(left=left, center=self.center, right=right))
        self._show_values()

    # ── 오른쪽: 조절 ──────────────────────────────────────────────

    def _controls(self) -> QWidget:
        self.sliders: dict[str, EffectControl] = {}

        def slider(name: str, minimum: int = -100) -> EffectControl:
            control = EffectControl(minimum)
            control.valueChanged.connect(self._on_changed)
            self.sliders[name] = control
            return control

        self.look_combo = QComboBox()
        self.look_combo.setToolTip("★ 기본 제공 모양 · 그 아래는 저장한 모양 (앱 전체 공통)")
        self.look_combo.activated.connect(self._load_look)
        self.look_save = QPushButton("저장")
        self.look_save.clicked.connect(self._save_look)
        self.look_delete = QPushButton("삭제")
        self.look_delete.clicked.connect(self._delete_look)
        looks = QHBoxLayout()
        looks.addWidget(self.look_combo, 1)
        looks.addWidget(self.look_save)
        looks.addWidget(self.look_delete)
        self.auto_button = QPushButton("자동 맞춤 [Beta]")
        self.auto_button.setToolTip(AUTO_NOTE)
        self.auto_button.clicked.connect(self._auto)
        self.reset_button = QPushButton("처음으로")
        self.reset_button.setToolTip("이 영상의 색 보정을 모두 지웁니다")
        self.reset_button.clicked.connect(self._reset)
        self.copy_button = QPushButton("모든 영상에 같게")
        self.copy_button.setToolTip("이 색 보정을 목록의 다른 영상에도 똑같이 입힙니다 (같은 장소·조명에서 찍었을 때)")
        self.copy_button.clicked.connect(self._copy_to_all)
        actions = QHBoxLayout()
        for button in (self.auto_button, self.reset_button, self.copy_button):
            actions.addWidget(button)

        light = QGroupBox("빛")
        light_form = QFormLayout(light)
        for name, label in LIGHT.items():
            light_form.addRow(label, slider(name))
        color = QGroupBox("색")
        color_form = QFormLayout(color)
        for name, label in COLOR.items():
            color_form.addRow(label, slider(name))
        color_form.addRow("바랜 느낌", slider("fade", 0))

        hsl = QGroupBox("색상별 (HSL)")
        hsl_layout = QVBoxLayout(hsl)
        chips = QHBoxLayout()
        self.hsl_group = QButtonGroup(self)
        for index, (key, (label, _angle)) in enumerate(HSL_COLORS.items()):
            chip = QToolButton()
            chip.setCheckable(True)
            chip.setIcon(_swatch(HSL_SWATCH[key]))
            chip.setToolTip(label)
            self.hsl_group.addButton(chip, index)
            chips.addWidget(chip)
        chips.addStretch(1)
        self.hsl_group.button(0).setChecked(True)
        self.hsl_group.idClicked.connect(lambda _i: self._show_values())
        self.hsl_name = QLabel()
        hsl_layout.addLayout(chips)
        hsl_form = QFormLayout()
        hsl_form.addRow(self.hsl_name)
        hsl_form.addRow("색상", slider("hsl_hue"))
        hsl_form.addRow("채도", slider("hsl_sat"))
        hsl_form.addRow("밝기", slider("hsl_lum"))
        hsl_layout.addLayout(hsl_form)

        lut = QGroupBox("LUT (.cube)")
        lut_form = QFormLayout(lut)
        self.lut_label = QLabel("없음")
        self.lut_label.setWordWrap(True)
        lut_load = QPushButton("불러오기…")
        lut_load.clicked.connect(self._choose_lut)
        self.lut_clear = QPushButton("빼기")
        self.lut_clear.clicked.connect(lambda: self._set_grade(replace(self._grade(), lut="", lut_strength=1.0)))
        lut_row = QHBoxLayout()
        lut_row.addWidget(self.lut_label, 1)
        lut_row.addWidget(lut_load)
        lut_row.addWidget(self.lut_clear)
        lut_form.addRow(lut_row)
        lut_form.addRow("강도", slider("lut_strength", 0))

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addLayout(looks)
        body_layout.addLayout(actions)
        for group in (light, color, hsl, lut):
            body_layout.addWidget(group)
        body_layout.addStretch(1)
        self._enable_widgets = (body,)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        scroll.setMinimumWidth(body.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width())
        self._fill_looks()
        return scroll

    # ── 페이지 수명 ──────────────────────────────────────────────

    def activate(self, workspace: Workspace) -> None:
        if workspace != self._workspace:
            self._save()
            self._workspace = workspace
            self._state = load_color_state(workspace.color_settings_file)
            self._source, self._frame = None, None
            self._user_luts.clear()
            self.center.setCurrentIndex(0)
        self._fill_list()
        self._fill_looks(self._look_name())
        self._show_values()

    def deactivate(self) -> None:
        self._save()

    def _save(self) -> None:
        self._save_timer.stop()
        if self._workspace is not None:
            self._workspace.color_dir.mkdir(parents=True, exist_ok=True)
            save_color_state(self._workspace.color_settings_file, self._state)

    # ── 원본 ────────────────────────────────────────────────────

    def _sources_list(self) -> list[str]:
        """녹화본 + 컷편집에서 가져온 미디어 + 이 탭에서 가져온 것 (중복 없이, 컷편집에서 뺀 녹화본은 빼고)."""
        ws = self._workspace
        project = Project.load(ws.project_file)
        found = [ws.to_source(p) for p in ws.recordings()] + list(project.media) + list(self._state.imported)
        return [s for s in dict.fromkeys(found) if s not in project.removed]

    def _fill_list(self) -> None:
        current = self._source
        self.list.blockSignals(True)
        self.list.clear()
        for source in self._sources_list():
            path = self._workspace.resolve(source)
            notes = [facts(path, self._sources)] if path.exists() else ["⚠ 파일 없음"]
            if self._state.grade_for(source) is not None:
                notes.append("🎨 색 보정 있음")
            add_row(self.list, SourceRow(source, path.name, " · ".join(n for n in notes if n), thumbnail=(self._sources, path)),
                    source, enabled=path.exists())
        rows = [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
        self.list.setCurrentRow(rows.index(current) if current in rows else -1)
        self.list.blockSignals(False)

    def _import(self) -> None:
        if self._workspace is None:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "색을 맞출 영상 가져오기", "", VIDEO_FILTER)
        added = [self._workspace.to_source(Path(f)) for f in files]
        if added:
            self._state = replace(self._state, imported=tuple(dict.fromkeys((*self._state.imported, *added))))
            self._save()
            self._source = added[0]
            self._fill_list()
            self._on_select(self.list.currentRow())

    def _media(self) -> Path | None:
        """실제로 읽을 파일. 휴대폰 영상처럼 편집용 사본이 필요한데 아직 없으면 None (만들기를 요청)."""
        path = self._workspace.resolve(self._source)
        plan = inspect(path)
        if plan is None:
            return path
        copy = optimized_path(self._workspace.cache_dir, path)
        if copy.exists():
            return copy
        if not self._optimizer.is_busy(path):
            self._optimizer.request(path, copy, plan)
        self.notice.setText(f"휴대폰 영상이라 편집용 사본부터 만드는 중… ({'·'.join(plan.reasons)})")
        self.center.setCurrentIndex(0)
        return None

    def _on_select(self, row: int) -> None:
        if row < 0 or self._workspace is None:
            return
        self._source = self.list.item(row).data(Qt.ItemDataRole.UserRole)
        self._frame = None
        self._show_values()
        media = self._media()
        if media is None:
            return
        info = probe(media)
        self._duration = info.duration if info else 0.0
        self.scrub.blockSignals(True)
        self.scrub.setRange(0, max(0, int(self._duration * SCRUB_STEPS_PER_SECOND) - 1))
        self.scrub.setValue(min(self.scrub.maximum(), int(self._duration * SCRUB_STEPS_PER_SECOND / 3)))  # 처음 장면은 어두운 때가 많다
        self.scrub.blockSignals(False)
        self._request_frame()

    def _on_optimized(self, path: Path) -> None:
        if self._source and self._workspace and self._workspace.resolve(self._source) == path:
            self._on_select(self.list.currentRow())

    def _on_scrub(self, _value: int) -> None:
        self._update_time()
        self._scrub_timer.start()

    def _update_time(self) -> None:
        seconds = self.scrub.value() / SCRUB_STEPS_PER_SECOND
        self.time_label.setText(f"{int(seconds // 60)}:{seconds % 60:04.1f}")

    def _request_frame(self) -> None:
        if self._workspace is None or self._source is None:
            return
        media = self._media()
        if media is None:
            return
        self._update_time()
        seconds = self.scrub.value() / SCRUB_STEPS_PER_SECOND
        self._frame_token = self._frames.request(lambda: grab_frame(media, seconds))

    def _on_frame(self, token: int, rgb) -> None:
        if token != self._frame_token:
            return
        if rgb is None:
            self.notice.setText("이 영상의 장면을 읽을 수 없습니다")
            self.center.setCurrentIndex(0)
            return
        self._frame = rgb
        self.center.setCurrentIndex(1)
        self._redraw_now()

    # ── 값 ──────────────────────────────────────────────────────

    def _grade(self) -> ColorGrade:
        return self._state.grades.get(self._source, ColorGrade()) if self._source else ColorGrade()

    def _hsl_index(self) -> int:
        return max(0, self.hsl_group.checkedId())

    def _show_values(self) -> None:
        """지금 영상의 값을 조절 칸에 (신호 없이)."""
        grade = self._grade()
        index = self._hsl_index()
        values = {name: getattr(grade, name) for name in (*LIGHT, *COLOR, "fade", "lut_strength")}
        values.update(hsl_hue=grade.hsl[index][0], hsl_sat=grade.hsl[index][1], hsl_lum=grade.hsl[index][2])
        for name, control in self.sliders.items():
            control.blockSignals(True)
            control.setValue(round(values[name] * 100))
            control.blockSignals(False)
        key = list(HSL_COLORS)[index]
        self.hsl_name.setText(f"고른 색: {HSL_COLORS[key][0]}")
        self.lut_label.setText(Path(grade.lut).name if grade.lut else "없음")
        self.lut_clear.setEnabled(bool(grade.lut))
        self.sliders["lut_strength"].setEnabled(bool(grade.lut))
        for widget in self._enable_widgets:
            widget.setEnabled(self._source is not None)

    def _on_changed(self, *_args) -> None:
        if self._source is None:
            return
        grade = self._grade()
        values = {name: self.sliders[name].value() / 100 for name in (*LIGHT, *COLOR, "fade", "lut_strength")}
        hsl = list(grade.hsl)
        hsl[self._hsl_index()] = tuple(self.sliders[name].value() / 100 for name in ("hsl_hue", "hsl_sat", "hsl_lum"))
        self._set_grade(replace(grade, **values, hsl=tuple(hsl)), show=False)

    def _set_grade(self, grade: ColorGrade, show: bool = True, source: str | None = None) -> None:
        source = source or self._source
        if source is None:
            return
        had = self._state.grade_for(source) is not None
        self._state = replace(self._state, grades={**self._state.grades, source: grade})
        if show:
            self._show_values()
        self._save_timer.start()
        self._redraw_timer.start()
        if had != (self._state.grade_for(source) is not None):
            self._fill_list()  # '색 보정 있음' 표시

    # ── 그리기 ──────────────────────────────────────────────────

    def _user_lut(self, grade: ColorGrade) -> np.ndarray | None:
        if not grade.lut or grade.lut_strength <= 0:
            return None
        if grade.lut not in self._user_luts:
            try:
                self._user_luts[grade.lut] = read_cube(self._workspace.resolve(grade.lut))
            except (OSError, ValueError) as e:
                self.message.emit(f"⚠ LUT을 읽을 수 없습니다: {e}", WARNING)
                self._user_luts[grade.lut] = None
        return self._user_luts[grade.lut]

    def _mode(self) -> str:
        return self.mode_group.checkedButton().property("mode")

    def _redraw_now(self) -> None:
        if self._frame is None:
            return
        frame, grade, mode = self._frame, self._grade(), self._mode()
        user = self._user_lut(grade)
        box = (self.view.width(), self.view.height())
        self.caption.setText({"side": "왼쪽 원본 · 오른쪽 보정", "split": "왼쪽 반 원본 · 오른쪽 반 보정", "after": "보정만"}[mode])
        self._draw_token = self._drawer.request(lambda: compare_image(frame, build_lut(grade, user), mode, box))

    def _on_drawn(self, token: int, image) -> None:
        if token == self._draw_token:
            self.view.setPixmap(_pixmap(image))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._redraw_timer.start()

    # ── 단추 ────────────────────────────────────────────────────

    def _auto(self) -> None:
        if self._frame is not None:
            self._set_grade(auto_grade(self._frame, self._grade()))
            self.message.emit("[Beta] 지금 장면에 맞춰 색온도·색조·노출을 맞췄습니다 — 분위기가 사라졌으면 슬라이더로 다듬거나 '처음으로'", INFO)

    def _reset(self) -> None:
        self._set_grade(ColorGrade())

    def _copy_to_all(self) -> None:
        if self._source is None:
            return
        others = [s for s in self._sources_list() if s != self._source]
        if not others:
            return
        answer = QMessageBox.question(self, "모든 영상에 같게", f"이 색 보정을 다른 영상 {len(others)}개에도 똑같이 입힐까요?\n"
                                                              "(그 영상들에 정해 둔 색 보정은 바뀝니다)")
        if answer != QMessageBox.StandardButton.Yes:
            return
        grade = self._grade()
        self._state = replace(self._state, grades={**self._state.grades, **{s: grade for s in others}})
        self._save()
        self._fill_list()
        self.message.emit(f"색 보정을 영상 {len(others) + 1}개에 똑같이 입혔습니다", INFO)

    def _choose_lut(self) -> None:
        """.cube를 골라 프로젝트 luts/ 폴더로 복사하고 (폴더째 옮겨도 맞게) 이 영상에 붙인다."""
        if self._workspace is None or self._source is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "LUT 불러오기", "", "LUT (*.cube)")
        if not path:
            return
        source = Path(path)
        try:
            read_cube(source)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "LUT", f"이 파일은 쓸 수 없습니다:\n{e}")
            return
        if self._workspace.to_source(source) == str(source.resolve()):  # 프로젝트 밖 → 복사
            dest = self._workspace.root / "luts" / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            source = dest
        lut = self._workspace.to_source(source)
        self._user_luts.pop(lut, None)
        self._set_grade(replace(self._grade(), lut=lut, lut_strength=1.0))

    # ── 모양 프리셋 ──────────────────────────────────────────────

    def _fill_looks(self, current: str | None = None) -> None:
        self.look_combo.clear()
        self.look_combo.addItem("불러올 모양 고르기…", None)
        for name in BUILTIN_LOOKS:
            self.look_combo.addItem(f"★ {name}", BUILTIN + name)
        self.look_combo.insertSeparator(self.look_combo.count())
        for name in list_presets(self.looks_dir):
            self.look_combo.addItem(name, name)
        self.look_combo.setCurrentIndex(max(0, self.look_combo.findData(current)) if current else 0)
        self.look_delete.setEnabled(self._look_name() is not None)

    def _look_name(self) -> str | None:
        """지금 고른 '저장한 모양' 이름 (기본 제공 모양이면 None — 지울 수 없다)."""
        data = self.look_combo.currentData()
        return data if data and not data.startswith(BUILTIN) else None

    def _load_look(self, index: int) -> None:
        name = self.look_combo.itemData(index)
        self.look_delete.setEnabled(self._look_name() is not None)
        if name is None or self._source is None:
            return
        if name.startswith(BUILTIN):
            look = BUILTIN_LOOKS[name.removeprefix(BUILTIN)]
        else:
            try:
                look = load_look(self.looks_dir, name)
            except (OSError, ValueError):
                self.message.emit(f"⚠ 모양 '{name}'을 읽지 못했습니다", WARNING)
                return
        current = self._grade()
        self._set_grade(replace(look, lut=current.lut, lut_strength=current.lut_strength))  # LUT은 그대로

    def _save_look(self) -> None:
        if self._source is None:
            return
        name, ok = QInputDialog.getText(self, "색 보정 모양 저장", "이름 (앱 전체에서 같이 씁니다):", text=self._look_name() or "")
        if not ok:
            return
        if not valid_name(name):
            self.message.emit('⚠ 이름이 비었거나 쓸 수 없는 글자(\\ / : * ? " < > |)가 있습니다', WARNING)
            return
        save_look(self.looks_dir, name, self._grade())
        self._fill_looks(name.strip())
        self.message.emit(f"색 보정 모양 '{name.strip()}'을 저장했습니다", INFO)

    def _delete_look(self) -> None:
        name = self._look_name()
        if name is not None:
            delete_preset(self.looks_dir, name)
            self._fill_looks()
