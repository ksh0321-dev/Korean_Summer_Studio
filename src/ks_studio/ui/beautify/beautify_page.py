"""AI 보정 단계: 영상 하나를 골라 얼굴 보정(피부·얼굴형·메이크업 17가지)을 입힌 보정본 파일을 만든다. 다른 단계와 독립.

흐름: 원본 고르기 → 보정할 사람 고르기(처음엔 가장 많이 나온 사람) → 그 사람이 잘 보이는 샘플 컷 4장에
     전/후 비교 (슬라이더로 바로 바뀜) → [이 설정으로 OK]
     → 아무 장면이나 스크럽하는 전/후 미리보기 (느린 PC는 끌 수 있음) → [보정본 만들기] → <프로젝트>/ai/<이름>_보정.mp4
보정본은 컷편집 워크스페이스에 녹화본처럼 나타난다. 원본마다 설정은 ai/settings.json에 남는다.
"""

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ks_studio.app.paths import PRESETS_DIR
from ks_studio.core.beauty import EYESHADOW_COLORS, AiState, BeautySettings, load_ai_state, save_ai_state
from ks_studio.core.presets import NAME_MAX, apply_preset, delete_preset, list_presets, load_preset, save_preset, valid_name
from ks_studio.core.project import Project
from ks_studio.core.workspace import Workspace
from ks_studio.core.people import match
from ks_studio.services.face import ai_unavailable_reason, people, pick_samples
from ks_studio.services.optimize import Optimizer, inspect
from ks_studio.services.sources import SourceCache, cache_name, optimized_path
from ks_studio.ui.beautify.effect_control import EffectControl
from ks_studio.ui.editor.media_bin import VIDEO_FILTER
from ks_studio.ui.widgets.clip_player import ClipPlayer
from ks_studio.ui.widgets.source_row import SourceRow, add_row, facts
from ks_studio.ui.widgets.layout import stage_layout
from ks_studio.ui.widgets.status_bar import ERROR, INFO

SAMPLE_SIZE = 200  # 샘플 컷 한 칸(원본·적용 범위·적용 후 중 하나) 크기
PERSON_ICON = 96  # 보정할 사람 얼굴 썸네일
SAVE_DELAY_MS = 500
LEFT_MIN_WIDTH = 400  # 효과 이름 + 슬라이더 + 숫자 칸이 들어갈 폭 (단계 공통 기본 280은 슬라이더가 너무 짧았다)
VIEW_MESSAGE, VIEW_SAMPLES, VIEW_PREVIEW, VIEW_CLIP = range(4)
EFFECT_TABS = (  # (탭, ((BeautySettings 칸, 이름), ...)) — 턱 길이만 -100~100
    ("피부", (("smooth", "매끈하게"), ("blemish", "잡티 제거"), ("brighten", "톤 밝게"), ("dark_circle", "다크서클"))),
    ("얼굴형", (
        ("slim", "얼굴 갸름"), ("cheekbone", "광대 축소"), ("chin", "턱 길이 (−짧게 · +길게)"),
        ("eyes", "눈 키우기"), ("nose_slim", "콧볼 축소"), ("nose_length", "코 길이 줄이기"), ("smile", "웃는 입"),
        ("face_small", "얼굴 작게 (실험)"),  # 배경은 그대로지만 얼굴 가장자리에 선이 보일 수 있다 — 분할 모델로 다듬을 예정
    )),
    ("메이크업", (
        ("blush", "블러셔"), ("contour", "윤곽 (쉐딩·하이라이트)"), ("eyeshadow", "아이섀도"),
        ("lip", "립 컬러"), ("teeth", "치아 미백"), ("eye_clear", "눈 또렷하게"),
    )),
)
EFFECT_NAMES = {field: name for _, rows in EFFECT_TABS for field, name in rows}
BIPOLAR = {"chin"}


def _pixmap(rgb: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())


class BeautifyPage(QWidget):
    message = Signal(str, str)  # 상태 표시줄 알림
    indicator = Signal(str, str, str)
    _model_progress = Signal(float)  # 얼굴 모델 내려받기 (작업 스레드 → 화면)
    _model_done = Signal(str)  # 실패 이유, 성공이면 ""

    def __init__(self, sources: SourceCache, optimizer: Optimizer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sources = sources
        self._optimizer = optimizer  # 컷편집과 같은 편집용 사본 작업자 (같은 파일을 두 번 만들지 않게)
        self._workspace: Workspace | None = None
        self._state = AiState({})
        self._source: str | None = None
        self._samples: list = []
        self._probes: list = []  # 샘플 후보 장면들 (사람을 바꾸면 여기서 다시 고른다)
        self._people: list = []
        self._preview: tuple | None = None  # (프레임, 원래 RGB, 랜드마크)
        self._unavailable = ai_unavailable_reason()
        self.worker = None
        self.renderer = None
        if self._unavailable is None:
            from ks_studio.services.beautify import AiWorker  # AI 패키지가 있을 때만 (cv2·insightface)

            self.worker = AiWorker(self)
            self.worker.samples_ready.connect(self._on_samples)
            self.worker.preview_ready.connect(self._on_preview)
            self.worker.progress.connect(self._on_progress)
            self.worker.rendered.connect(self._on_rendered)
            self.worker.clip_ready.connect(self._on_clip)
            from ks_studio.ui.widgets.latest_worker import LatestWorker

            self.renderer = LatestWorker("ai-preview", self)  # 샘플 컷·장면 미리보기 그림은 작업 스레드에서 (UI가 멈추지 않게)
            self.renderer.ready.connect(self._on_drawn)
            self.worker.failed.connect(self._on_failed)
        self._sample_token = self._preview_token = -1
        optimizer.finished.connect(lambda path, _out: self._on_optimized(path))
        optimizer.progress.connect(self._on_optimize_progress)

        # 왼쪽: 원본 목록 + 효과
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.currentRowChanged.connect(self._on_select)
        import_button = QPushButton("+ 가져오기")
        import_button.clicked.connect(self._import)
        self.sliders: dict[str, EffectControl] = {}
        effects = QGroupBox("효과 (선택한 영상에)")
        tabs = QTabWidget()
        for title, rows in EFFECT_TABS:
            page = QWidget()
            form = QFormLayout(page)
            for field, name in rows:
                control = EffectControl(-100 if field in BIPOLAR else 0)
                control.valueChanged.connect(self._on_effect_changed)
                self.sliders[field] = control
                form.addRow(name, control)
                if field == "eyeshadow":
                    self.shadow_color = QComboBox()
                    for key, (label, _rgb) in EYESHADOW_COLORS.items():
                        self.shadow_color.addItem(label, key)
                    self.shadow_color.currentIndexChanged.connect(self._on_effect_changed)
                    form.addRow("아이섀도 색", self.shadow_color)
            tabs.addTab(page, title)
        self.presets_dir = PRESETS_DIR
        self.preset_combo = QComboBox()
        self.preset_combo.activated.connect(self._on_preset_chosen)
        self.preset_save = QPushButton("저장")
        self.preset_save.clicked.connect(self._save_preset)
        self.preset_delete = QPushButton("삭제")
        self.preset_delete.clicked.connect(self._delete_preset)
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("프리셋"))
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.preset_save)
        preset_row.addWidget(self.preset_delete)
        effects_layout = QVBoxLayout(effects)
        effects_layout.addWidget(tabs)
        effects_layout.addLayout(preset_row)
        self.preview_check = QCheckBox("OK 뒤 미리보기 켜기 (느린 PC는 끄기)")
        self.preview_check.toggled.connect(self._on_preview_toggled)
        left = QWidget()
        left.setMinimumWidth(LEFT_MIN_WIDTH)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        # 얼굴 모델: 없으면 눌러서 내려받는다 (저절로 받지 않는다 — 약 290MB). 있으면 '완료'로 보인다
        self.model_button = QPushButton()
        self.model_button.clicked.connect(self._download_model)
        self._model_progress.connect(lambda r: self.model_button.setText(f"얼굴 모델 내려받는 중… {int(r * 100)}%"))
        self._model_done.connect(self._on_model_done)
        self._downloading = False
        left_layout.addWidget(self.model_button)
        header = QHBoxLayout()
        header.addWidget(QLabel("원본"))
        header.addStretch(1)
        header.addWidget(import_button)
        left_layout.addLayout(header)
        left_layout.addWidget(self.list, 1)
        left_layout.addWidget(effects)
        left_layout.addWidget(self.preview_check)

        # 가운데: 안내 / 샘플 컷 / 미리보기
        self.center = QStackedWidget()
        self.notice = QLabel()
        self.notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.notice.setWordWrap(True)
        self.scan_progress = QProgressBar()  # 샘플 장면 고르는 동안만
        self.scan_progress.setRange(0, 1000)
        self.scan_progress.setMaximumWidth(480)
        self.scan_progress.hide()
        notice_page = QWidget()
        notice_layout = QVBoxLayout(notice_page)
        notice_layout.addStretch(1)
        notice_layout.addWidget(self.notice)
        notice_layout.addWidget(self.scan_progress, 0, Qt.AlignmentFlag.AlignHCenter)
        notice_layout.addStretch(1)
        self.center.addWidget(notice_page)
        self._scan_started = 0.0
        samples = QWidget()
        samples_layout = QVBoxLayout(samples)
        samples_layout.addWidget(QLabel("보정할 사람 — 클릭해서 바꾸기 (처음엔 영상에 가장 많이 나온 사람)"))
        self.people_list = QListWidget()
        self.people_list.setViewMode(QListView.ViewMode.IconMode)
        self.people_list.setFlow(QListView.Flow.LeftToRight)
        self.people_list.setWrapping(False)
        self.people_list.setMovement(QListView.Movement.Static)
        self.people_list.setIconSize(QSize(PERSON_ICON, PERSON_ICON))
        self.people_list.setFixedHeight(PERSON_ICON + 48)
        self.people_list.currentRowChanged.connect(self._on_person)
        samples_layout.addWidget(self.people_list)
        samples_layout.addWidget(QLabel("샘플 컷 — 원본 · 적용 범위 · 적용 후. 슬라이더를 움직이면 바로 바뀝니다."))
        grid = QGridLayout()
        self.sample_labels = []
        for i in range(4):
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(label, i // 2, i % 2)
            self.sample_labels.append(label)
        samples_layout.addLayout(grid, 1)
        buttons = QHBoxLayout()
        self.resample_button = QPushButton("샘플 다시 고르기")
        self.resample_button.clicked.connect(lambda: self._request_samples(refresh=True))
        self.approve_button = QPushButton("이 설정으로 OK →")
        self.approve_button.setObjectName("recordButton")
        self.approve_button.clicked.connect(self._approve)
        buttons.addStretch(1)
        buttons.addWidget(self.resample_button)
        buttons.addWidget(self.approve_button)
        samples_layout.addLayout(buttons)
        self.center.addWidget(samples)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.addWidget(QLabel("미리보기 — 왼쪽 보정 전 · 오른쪽 보정 후"))
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # 그림 크기가 칸 크기를 정하지 않게 (세로 영상이 창을 아래로 늘렸다) — 칸에 맞춰 그림을 줄인다
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview_label.setMinimumSize(200, 120)
        preview_layout.addWidget(self.preview_label, 1)
        scrub = QHBoxLayout()
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.valueChanged.connect(self._request_preview)
        self.landmarks_check = QCheckBox("얼굴 점 표시")
        self.landmarks_check.toggled.connect(lambda _on: self._show_preview())
        back = QPushButton("샘플로 돌아가기")
        back.clicked.connect(self._back_to_samples)
        self.clip_button = QPushButton("▶ 여기서부터 1분 영상으로 보기")
        self.clip_button.setToolTip("슬라이더 위치부터 약 1분을 보정해서 원본과 나란히 소리와 함께 틀어 줍니다")
        self.clip_button.clicked.connect(self._request_clip)
        scrub.addWidget(self.scrub, 1)
        scrub.addWidget(self.landmarks_check)
        scrub.addWidget(self.clip_button)
        scrub.addWidget(back)
        preview_layout.addLayout(scrub)
        self.center.addWidget(preview)
        clip_page = QWidget()
        clip_layout = QVBoxLayout(clip_page)
        clip_layout.addWidget(QLabel("미리보기 영상 — 왼쪽 원본 · 오른쪽 보정 (한 영상이라 늘 같은 순간)"))
        self.clip_player = ClipPlayer()
        clip_layout.addWidget(self.clip_player, 1)
        clip_back = QPushButton("장면 미리보기로 돌아가기")
        clip_back.clicked.connect(self._leave_clip)
        clip_row = QHBoxLayout()
        clip_row.addStretch(1)
        clip_row.addWidget(clip_back)
        clip_layout.addLayout(clip_row)
        self.center.addWidget(clip_page)

        # 아래: 보정본 만들기
        self.status = QLabel()
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.render_button = QPushButton("보정본 만들기")
        self.render_button.clicked.connect(self._render)
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.addWidget(self.status, 1)
        bottom_layout.addWidget(self.progress)
        bottom_layout.addWidget(self.render_button)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(SAVE_DELAY_MS)
        self._save_timer.timeout.connect(self._save)
        self._effect_timer = QTimer(self)  # 슬라이더를 끄는 동안 너무 자주 다시 그리지 않게
        self._effect_timer.setSingleShot(True)
        self._effect_timer.setInterval(60)
        self._effect_timer.timeout.connect(self._redraw)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        splitter = stage_layout(left=left, center=self.center, bottom=bottom)
        splitter.setSizes([900, 60])
        layout.addWidget(splitter)
        self._set_notice(self._unavailable or "왼쪽에서 보정할 영상을 고르세요")
        self._update_controls()

    # ── 페이지 수명 ──────────────────────────────────────────────

    def activate(self, workspace: Workspace) -> None:
        if workspace != self._workspace:
            self._workspace = workspace
            self._state = load_ai_state(workspace.ai_settings_file)
            self._source = None
        self.preview_check.blockSignals(True)
        self.preview_check.setChecked(self._state.preview)
        self.preview_check.blockSignals(False)
        self._fill_list()
        self._fill_presets()
        self._update_model_button()

    # ── 얼굴 모델 ────────────────────────────────────────────────

    def _update_model_button(self) -> None:
        if self._downloading:
            return
        if self._unavailable is not None:  # AI 패키지가 없다 (안내는 가운데 화면에)
            self.model_button.setText("AI 보정 패키지 없음")
            self.model_button.setEnabled(False)
            return
        from ks_studio.services.face import FACE_MODEL_MB, face_models_ready

        ready = face_models_ready()
        self.model_button.setText("✓ 모델 다운로드 완료" if ready else f"⬇ 얼굴 모델 내려받기 (약 {FACE_MODEL_MB}MB)")
        self.model_button.setEnabled(not ready)
        self.model_button.setToolTip("" if ready else "AI 보정에 쓰는 얼굴 모델을 models 폴더에 내려받습니다 (인터넷 필요, 처음 한 번)")

    def _download_model(self) -> None:
        import threading

        from ks_studio.services.face import download_face_models

        self._downloading = True
        self.model_button.setEnabled(False)
        self.model_button.setText("얼굴 모델 내려받는 중…")

        def work() -> None:
            try:
                download_face_models(self._model_progress.emit)
            except Exception as e:  # 인터넷 끊김 등 — 화면에 알리고 다시 누를 수 있게
                self._model_done.emit(str(e) or type(e).__name__)
                return
            self._model_done.emit("")

        threading.Thread(target=work, daemon=True, name="face-model-download").start()

    def _on_model_done(self, error: str) -> None:
        self._downloading = False
        self._update_model_button()
        if error:
            self.message.emit(f"⚠ 얼굴 모델을 내려받지 못했습니다: {error} — 인터넷 연결을 확인하고 다시 눌러 주세요", ERROR)
            return
        self.message.emit("얼굴 모델 다운로드 완료", INFO)
        if self._source is not None:  # 고른 영상이 있으면 바로 샘플 장면 고르기
            self._request_samples()

    def deactivate(self) -> None:
        self.clip_player.player.pause()
        self._save()

    def _save(self) -> None:
        self._save_timer.stop()
        if self._workspace is not None:
            self._workspace.ai_dir.mkdir(parents=True, exist_ok=True)
            save_ai_state(self._workspace.ai_settings_file, self._state)

    # ── 원본 ────────────────────────────────────────────────────

    def _sources_list(self) -> list[str]:
        """녹화본 + 컷편집에서 가져온 미디어 + AI 탭에서 가져온 것 (중복 없이)."""
        ws = self._workspace
        project = Project.load(ws.project_file)
        found = [ws.to_source(p) for p in ws.recordings()]
        found += list(project.media)
        found += list(self._state.imported)
        return [s for s in dict.fromkeys(found) if s not in project.removed]  # 컷편집 워크스페이스에서 뺀 녹화본은 숨긴다

    def _fill_list(self) -> None:
        current = self._source
        self.list.blockSignals(True)
        self.list.clear()
        for source in self._sources_list():
            path = self._workspace.resolve(source)
            settings = self._state.for_source(source)
            notes = [facts(path, self._sources)] if path.exists() else ["⚠ 파일 없음"]
            if self._output_for(source).exists():
                notes.append("보정본 있음")
            elif settings.approved:
                notes.append("설정 OK")
            row = SourceRow(source, path.name, " · ".join(n for n in notes if n), self._state.memos.get(source, ""),
                            thumbnail=(self._sources, path))
            row.memo_changed.connect(self._on_memo)
            add_row(self.list, row, source, enabled=path.exists())
        rows = [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
        self.list.setCurrentRow(rows.index(current) if current in rows else -1)
        self.list.blockSignals(False)

    def _on_memo(self, source: str, memo: str) -> None:
        self._state = replace(self._state, memos={**self._state.memos, source: memo.strip()})
        self._save_timer.start()

    def _import(self) -> None:
        if self._workspace is None:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "보정할 영상 가져오기", "", VIDEO_FILTER)
        added = [self._workspace.to_source(Path(f)) for f in files]
        if added:
            self._state = replace(self._state, imported=tuple(dict.fromkeys((*self._state.imported, *added))))
            self._save()
            self._source = added[0]
            self._fill_list()
            self._on_select(self.list.currentRow())

    def _media(self) -> Path | None:
        """지금 원본의 실제로 읽을 파일. 휴대폰 영상처럼 편집용 사본이 필요한데 아직 없으면 None (만들기를 요청)."""
        path = self._workspace.resolve(self._source)
        plan = inspect(path)
        if plan is None:
            return path
        copy = optimized_path(self._workspace.cache_dir, path)
        if copy.exists():
            return copy
        if not self._optimizer.is_busy(path):
            self._optimizer.request(path, copy, plan)
        self._set_notice(f"휴대폰 영상이라 편집용 사본부터 만드는 중… ({'·'.join(plan.reasons)})")
        return None

    def _output_for(self, source: str) -> Path:
        return self._workspace.ai_output(source)

    def _on_select(self, row: int) -> None:
        if row < 0 or self._workspace is None:
            return
        self._source = self.list.item(row).data(Qt.ItemDataRole.UserRole)
        self._show_values()
        self._samples, self._preview, self._probes, self._people = [], None, [], []
        self.people_list.clear()
        if self._unavailable:
            return
        self._request_samples()

    def _on_optimized(self, path: Path) -> None:
        if self._source and self._workspace and self._workspace.resolve(self._source) == path:
            self._request_samples()

    def _on_optimize_progress(self, path: Path, ratio: float) -> None:
        if self._source and self._workspace and self._workspace.resolve(self._source) == path:
            self._set_notice(f"휴대폰 영상이라 편집용 사본부터 만드는 중… {int(ratio * 100)}%")

    # ── 샘플 컷 ─────────────────────────────────────────────────

    def _request_samples(self, refresh: bool = False) -> None:
        """샘플 후보 장면: 한 번 찾은 영상은 프로젝트 캐시에서 바로. [샘플 다시 고르기]는 새로 찾는다."""
        from ks_studio.services.face import samples_cache

        from ks_studio.services.face import face_models_ready

        if not face_models_ready():
            self._set_notice("얼굴 모델이 아직 없습니다 — 왼쪽 위 [⬇ 얼굴 모델 내려받기]를 눌러 주세요 (처음 한 번)")
            return
        media = self._media()
        if media is None:
            return
        self._set_notice("얼굴이 나오는 장면을 고르는 중…")
        self._scan_started = time.monotonic()
        self.scan_progress.setValue(0)
        self.scan_progress.show()
        self.worker.request_samples(media, samples_cache(self._workspace.cache_dir, cache_name(media)), refresh)

    def _on_samples(self, media: Path, probes: list) -> None:
        if self._media_matches(media):
            self._probes, self._people = probes, people(probes)
            if not self._people:
                self._set_notice("이 영상에서 정면 얼굴을 찾지 못했습니다")
                return
            if self._settings().target is None:  # 처음엔 가장 많이 나온 사람
                self._set_settings(target=tuple(float(x) for x in self._people[0].embedding))
            self._fill_people()
            self._samples = pick_samples(self._probes, self._settings().target_vector)
            if not self._samples:
                self._set_notice("고른 사람이 나오는 장면을 찾지 못했습니다 — 다른 사람을 고르거나 샘플을 다시 고르세요")
                return
            settings = self._state.for_source(self._source)
            if settings.approved and self._state.preview:
                self._enter_preview()
            else:
                self.center.setCurrentIndex(VIEW_SAMPLES)
                self._redraw()
            self._update_controls()

    def _fill_people(self) -> None:
        """후보 장면에 나온 사람들을 얼굴 썸네일로. 지금 보정할 사람을 선택해 둔다."""
        self.people_list.blockSignals(True)
        self.people_list.clear()
        for person in self._people:
            probe = self._probes[person.probe]
            from ks_studio.ui.beautify.preview_render import face_crop

            item = QListWidgetItem(QIcon(_pixmap(face_crop(probe.rgb, person.face.landmarks, PERSON_ICON))), f"{person.appearances}장면")
            self.people_list.addItem(item)
        target = self._settings().target_vector
        index = match([p.embedding for p in self._people], target) if target is not None else None
        self.people_list.setCurrentRow(index if index is not None else -1)
        self.people_list.blockSignals(False)

    def _on_person(self, row: int) -> None:
        """다른 사람을 고르면 보정 대상이 바뀌므로 OK를 풀고 샘플부터 다시 확인한다."""
        if row < 0 or row >= len(self._people):
            return
        self._set_settings(target=tuple(float(x) for x in self._people[row].embedding), approved=False)
        self._samples, self._preview = pick_samples(self._probes, self._settings().target_vector), None
        self.center.setCurrentIndex(VIEW_SAMPLES)
        self._redraw()
        self._fill_list()
        self._update_controls()

    def _set_settings(self, **changes) -> None:
        settings = replace(self._settings(), **changes)
        self._state = replace(self._state, settings={**self._state.settings, self._source: settings})
        self._save_timer.start()

    def _media_matches(self, media: Path) -> bool:
        return self._source is not None and self._workspace is not None and media == self._current_media()

    def _current_media(self) -> Path | None:
        path = self._workspace.resolve(self._source)
        copy = optimized_path(self._workspace.cache_dir, path)
        return copy if copy.exists() else path

    def _settings(self) -> BeautySettings:
        return self._state.for_source(self._source) if self._source else BeautySettings()

    def _show_values(self) -> None:
        """지금 영상의 효과 값을 슬라이더·숫자 칸에 (알림 없이)."""
        settings = self._settings()
        for field, control in self.sliders.items():
            control.blockSignals(True)
            control.setValue(round(getattr(settings, field) * 100))
            control.blockSignals(False)
        self.shadow_color.blockSignals(True)
        self.shadow_color.setCurrentIndex(max(0, self.shadow_color.findData(settings.eyeshadow_color)))
        self.shadow_color.blockSignals(False)

    # ── 프리셋 ──────────────────────────────────────────────────

    def _fill_presets(self, current: str | None = None) -> None:
        self.preset_combo.clear()
        self.preset_combo.addItem("불러올 프리셋 고르기…")
        self.preset_combo.addItems(list_presets(self.presets_dir))
        index = self.preset_combo.findText(current) if current else 0
        self.preset_combo.setCurrentIndex(max(0, index))
        self._update_controls()

    def _preset_name(self) -> str | None:
        return self.preset_combo.currentText() if self.preset_combo.currentIndex() > 0 else None

    def _on_preset_chosen(self, index: int) -> None:
        """프리셋을 입히면 효과가 한꺼번에 바뀌므로 OK를 풀고 샘플 컷부터 다시 본다."""
        name = self._preset_name()
        if name is None or self._source is None:
            return
        try:
            values = load_preset(self.presets_dir, name)
        except (OSError, ValueError) as e:
            self.message.emit(f"프리셋을 읽지 못했습니다: {name} ({e})", ERROR)
            self._fill_presets()
            return
        settings = apply_preset(self._settings(), values)
        self._state = replace(self._state, settings={**self._state.settings, self._source: settings})
        self._save_timer.start()
        self._show_values()
        if self._samples:
            self.center.setCurrentIndex(VIEW_SAMPLES)
        self._redraw()
        self._fill_list()
        self._update_controls()
        self.message.emit(f"프리셋 불러옴: {name} — 샘플 컷을 확인하고 OK를 다시 눌러 주세요", INFO)

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "프리셋 저장", f"프리셋 이름 (최대 {NAME_MAX}자):", text=self._preset_name() or "")
        name = name.strip()
        if not ok or not name:
            return
        if not valid_name(name):
            QMessageBox.warning(self, "프리셋 저장", '이름에 \\ / : * ? " < > | 는 쓸 수 없습니다.')
            return
        if name in list_presets(self.presets_dir):
            answer = QMessageBox.question(self, "프리셋 저장", f"'{name}' 프리셋을 지금 값으로 덮어쓸까요?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        save_preset(self.presets_dir, name, self._settings())
        self._fill_presets(name)
        self.message.emit(f"프리셋 저장: {name}", INFO)

    def _delete_preset(self) -> None:
        name = self._preset_name()
        if name is None:
            return
        if QMessageBox.question(self, "프리셋 삭제", f"'{name}' 프리셋을 지울까요?") != QMessageBox.StandardButton.Yes:
            return
        delete_preset(self.presets_dir, name)
        self._fill_presets()
        self.message.emit(f"프리셋 삭제: {name}", INFO)

    def _on_effect_changed(self) -> None:
        if self._source is None:
            return
        self._set_settings(
            **{field: slider.value() / 100 for field, slider in self.sliders.items()},
            eyeshadow_color=self.shadow_color.currentData(),
        )
        self._effect_timer.start()

    def _redraw(self) -> None:
        """샘플 컷(또는 장면 미리보기)을 다시 그려 달라고 작업 스레드에 맡긴다. 결과는 _on_drawn."""
        if self.renderer is None:
            return
        from ks_studio.ui.beautify.preview_render import sample_strip

        settings = self._settings()
        if self.center.currentIndex() == VIEW_SAMPLES:
            samples = list(self._samples)
            self._sample_token = self.renderer.request(
                lambda: [sample_strip(s.rgb, s.landmarks, settings, SAMPLE_SIZE) for s in samples]
            )
        elif self.center.currentIndex() == VIEW_PREVIEW:
            self._show_preview()

    def _on_drawn(self, token: int, result) -> None:
        if token == self._sample_token:
            for label, image in zip(self.sample_labels, result):
                label.setPixmap(_pixmap(image))
            for label in self.sample_labels[len(result) :]:
                label.clear()
        elif token == self._preview_token:
            self.preview_label.setPixmap(_pixmap(result))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.center.currentIndex() == VIEW_PREVIEW:
            self._effect_timer.start()  # 창 크기에 맞춰 미리보기를 다시 (잠깐 모아서)

    def _approve(self) -> None:
        self._set_settings(approved=True)
        self._save()
        on = [f"{EFFECT_NAMES[field].split(' (')[0]} {slider.value()}%" for field, slider in self.sliders.items() if slider.value()]
        self.message.emit(f"AI 보정 설정 OK: {Path(self._source).name} — {' · '.join(on) or '효과 없음'}", INFO)
        if self._state.preview:
            self._enter_preview()
        else:
            self.status.setText("미리보기를 끈 상태입니다 — [보정본 만들기]로 바로 진행하세요")
        self._fill_list()
        self._update_controls()

    # ── 미리보기 ────────────────────────────────────────────────

    def _on_preview_toggled(self, on: bool) -> None:
        self._state = replace(self._state, preview=on)
        self._save_timer.start()
        if not on and self.center.currentIndex() == VIEW_PREVIEW:
            self._back_to_samples()
        elif on and self._source and self._settings().approved and self._samples:
            self._enter_preview()

    def _enter_preview(self) -> None:
        import av

        media = self._current_media()
        with av.open(str(media)) as container:
            frames = container.streams.video[0].frames or 1
        self.center.setCurrentIndex(VIEW_PREVIEW)
        self.scrub.blockSignals(True)
        self.scrub.setRange(0, max(0, frames - 1))
        self.scrub.setValue(self._samples[0].frame if self._samples else 0)
        self.scrub.blockSignals(False)
        self._request_preview()

    def _back_to_samples(self) -> None:
        self.center.setCurrentIndex(VIEW_SAMPLES)
        self._redraw()

    def _request_preview(self) -> None:
        from ks_studio.services.beautify import track_path_for

        media = self._current_media()
        settings = self._settings()
        track = track_path_for(self._workspace.cache_dir, media, settings)
        self.worker.request_preview(media, self.scrub.value(), track, settings.target_vector)

    def _on_preview(self, media: Path, frame: int, rgb, landmarks) -> None:
        if self._media_matches(media) and frame == self.scrub.value():
            self._preview = (frame, rgb, landmarks)
            self._show_preview()

    def _show_preview(self) -> None:
        """장면 미리보기: 지금 칸 크기(가로·세로)에 들어가게 줄여서, 작업 스레드에서 보정해 그린다."""
        if self._preview is None or self.renderer is None:
            return
        from ks_studio.ui.beautify.preview_render import preview_pair

        _, rgb, landmarks = self._preview
        settings, points = self._settings(), self.landmarks_check.isChecked()
        box = (self.preview_label.width(), self.preview_label.height())
        self._preview_token = self.renderer.request(lambda: preview_pair(rgb, landmarks, settings, box, points))
        self.status.setText("이 장면에는 보정할 사람이 없습니다 (그대로 둡니다)" if landmarks is None else "")

    # ── 보정본 만들기 ────────────────────────────────────────────

    def _render(self) -> None:
        from ks_studio.services.beautify import track_path_for

        media = self._current_media()
        output = self._output_for(self._source)
        settings = self._settings()
        source = self._workspace.resolve(self._source)
        # 휴대폰 영상: 사본을 다시 압축하지 않고 원본에서 한 번에 (화질 보존)
        original = (source, inspect(source)) if media != source else None
        self.worker.request_render(media, output, track_path_for(self._workspace.cache_dir, media, settings), settings, original)
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("영상 전체 얼굴 분석 중… (처음 한 번만, 강도를 바꿔 다시 만들 때는 건너뜀)")
        self._update_controls()

    # ── 1분 미리보기 영상 ────────────────────────────────────────

    def _clip_path(self) -> Path:
        return self._workspace.cache_dir / "ai_clips" / f"{cache_name(self._current_media())}.mp4"

    def _request_clip(self) -> None:
        from ks_studio.services.beautify import track_path_for

        media = self._current_media()
        self.clip_player.release()  # 같은 파일을 덮어쓰므로 놓아 준다 (Windows)
        self.clip_button.setEnabled(False)
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("1분 미리보기 영상을 만드는 중…")
        track = track_path_for(self._workspace.cache_dir, media, self._settings())
        self.worker.request_clip(media, self._clip_path(), self.scrub.value(), track, self._settings())

    def _on_clip(self, media: Path, output: Path) -> None:
        self.clip_button.setEnabled(True)
        self.progress.hide()
        self.status.setText("")
        if not self._media_matches(media):
            return
        self.center.setCurrentIndex(VIEW_CLIP)
        self.clip_player.open(output)

    def _leave_clip(self) -> None:
        self.clip_player.player.pause()
        self.center.setCurrentIndex(VIEW_PREVIEW)

    def _on_progress(self, kind: str, media: Path, ratio: float) -> None:
        if kind == "samples":
            if self._media_matches(media):
                self.scan_progress.setValue(round(ratio * 1000))
                elapsed = time.monotonic() - self._scan_started
                remaining = elapsed * (1 - ratio) / ratio if ratio > 0 else 0
                self.notice.setText(f"얼굴이 나오는 장면을 고르는 중… {int(ratio * 100)}% · 약 {max(1, round(remaining))}초 남음")
            return
        self.progress.setValue(round(ratio * 1000))
        label = {"analyze": "영상 전체 얼굴 분석 중… ", "clip": "1분 미리보기 영상 만드는 중… "}.get(kind, "보정본 만드는 중… ")
        self.status.setText(label + f"{int(ratio * 100)}%")

    def _on_rendered(self, media: Path, output: Path) -> None:
        self.progress.hide()
        self.status.setText(f"완성: {output.name} — 컷편집 워크스페이스에 나타납니다")
        self.message.emit(f"AI 보정본 완성: {output.name}", INFO)
        self._fill_list()
        self._update_controls()

    def _on_failed(self, kind: str, media: Path, reason: str) -> None:
        self.progress.hide()
        self.message.emit(f"⚠ AI 보정 실패 ({kind}): {reason}", ERROR)
        if kind == "samples":
            self._set_notice(f"⚠ 얼굴 찾기 실패: {reason}")
        if kind == "clip":
            self.clip_button.setEnabled(True)
        self._update_controls()

    # ── 표시 ────────────────────────────────────────────────────

    def _set_notice(self, text: str) -> None:
        self.notice.setText(text)
        self.scan_progress.hide()
        self.center.setCurrentIndex(VIEW_MESSAGE)

    def _update_controls(self) -> None:
        ready = self.worker is not None and self._source is not None and bool(self._samples)
        rendering = self.worker is not None and self.worker.busy_with is not None
        for widget in (*self.sliders.values(), self.shadow_color, self.resample_button, self.approve_button,
                       self.preset_combo, self.preset_save):
            widget.setEnabled(ready)
        self.preset_delete.setEnabled(ready and self._preset_name() is not None)
        self.render_button.setEnabled(ready and self._settings().approved and not rendering)
        self.render_button.setToolTip("" if self._settings().approved else "샘플 컷을 보고 [이 설정으로 OK]를 먼저 누르세요")
