"""내보내기 단계 (영상 편집의 마지막): 컷편집 타임라인(자막을 여기서 굽는다) 또는 영상 파일 하나 → 최종 MP4.

흐름: 녹화 → AI 보정(자막 없는 원본) → 컷편집(보정본 + 자막 줄) → 내보내기에서 합친다.
AI 보정만 한 영상·녹화본도 파일로 바로 내보낼 수 있다 (자막은 컷편집 타임라인에만 있다).
[저해상도 미리보기]: 짧은 변 800px로 빠르게 만들어 바로 튼다 — 최종 파일을 만들어 확인하는 수고를 던다.
최종 파일: <프로젝트>/exports/final/<이름>_<비율>_<해상도>.mp4
"""

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ks_studio.core.project import Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.accel import open_folder
from ks_studio.services.deliver import CODECS, FITS, PRESETS, RATIOS, RESOLUTIONS, ExportWorker, OutputFormat
from ks_studio.services.sources import SourceCache
from ks_studio.ui.editor.media_bin import VIDEO_FILTER
from ks_studio.ui.widgets.clip_player import ClipPlayer
from ks_studio.ui.widgets.layout import stage_layout
from ks_studio.ui.widgets.source_row import SourceRow, add_row, clock, facts

TIMELINE = "<timeline>"  # 목록 맨 위 '컷편집 타임라인'의 자리 표시
from ks_studio.ui.widgets.status_bar import ERROR, INFO


def _combo(options: dict, current) -> QComboBox:
    combo = QComboBox()
    for key, label in options.items():
        combo.addItem(label, key)
    combo.setCurrentIndex(max(0, combo.findData(current)))
    return combo


class ExportPage(QWidget):
    message = Signal(str, str)
    indicator = Signal(str, str, str)

    def __init__(self, sources: SourceCache, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sources = sources
        self._workspace: Workspace | None = None
        self._source: str | None = None
        self._imported: tuple[str, ...] = ()
        self._default_preset = ""  # 설정 '기본값': 프로젝트를 처음 열 때 고를 프리셋
        self._project = Project()  # 컷편집 타임라인 (탭을 열 때 project.json에서)
        self.worker = ExportWorker(self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.cancelled.connect(self._on_cancelled)

        # 왼쪽: 내보낼 영상
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_select)
        import_button = QPushButton("+ 가져오기")
        import_button.clicked.connect(self._import)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel("내보낼 영상"))
        header.addStretch(1)
        header.addWidget(import_button)
        left_layout.addLayout(header)
        left_layout.addWidget(self.list, 1)

        # 가운데: 원본 / 미리보기 재생
        self.view_label = QLabel("왼쪽에서 내보낼 영상을 고르세요")
        self.player = ClipPlayer()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self.view_label)
        center_layout.addWidget(self.player, 1)

        # 오른쪽: 출력 형식 · 자막 · 미리보기 · 내보내기
        fmt = OutputFormat()
        output = QGroupBox("출력 형식")
        form = QFormLayout(output)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("직접 고르기", None)
        for key, (label, _values) in PRESETS.items():
            self.preset_combo.addItem(label, key)
        self.preset_combo.setToolTip("고르면 비율·맞추기·해상도·형식·소리 크기 맞추기를 한 번에 맞춥니다")
        self.preset_combo.activated.connect(self._apply_preset)
        form.addRow("프리셋", self.preset_combo)
        self.ratio_combo = _combo({k: v[0] for k, v in RATIOS.items()}, fmt.ratio)
        self.fit_combo = _combo(FITS, fmt.fit)
        self.size_combo = _combo(RESOLUTIONS, fmt.short)
        self.codec_combo = _combo(CODECS, fmt.codec)
        self.ratio_combo.currentIndexChanged.connect(self._update_controls)
        form.addRow("비율", self.ratio_combo)
        form.addRow("맞추기", self.fit_combo)
        form.addRow("해상도", self.size_combo)
        form.addRow("형식", self.codec_combo)
        # 소리 다듬기 (컷편집 타임라인을 내보낼 때)
        self.loudness_check = QCheckBox("소리 크기 맞추기 (유튜브 기준 -14 LUFS)")
        self.loudness_check.setToolTip("작게 녹음된 영상은 키우고 너무 큰 영상은 줄여, 플랫폼이 다시 줄이거나 키우지 않게 맞춥니다")
        self.denoise_check = QCheckBox("잡음 줄이기 (바람·웅웅 소리)")
        self.denoise_check.setToolTip("고르게 깔린 잡음을 줄입니다. 너무 시끄러운 곳의 소리는 다 지우지 못합니다")
        for check in (self.loudness_check, self.denoise_check):
            check.toggled.connect(lambda _on: self.preset_combo.setCurrentIndex(0))
            form.addRow("", check)
        for combo in (self.ratio_combo, self.fit_combo, self.size_combo, self.codec_combo):
            combo.activated.connect(lambda _i: self.preset_combo.setCurrentIndex(0))  # 손으로 바꾸면 '직접 고르기'
        self.subtitle_check = QCheckBox("자막 넣기")
        self.subtitle_check.setChecked(True)
        self.preview_button = QPushButton("👁 저해상도 미리보기 (800p)")
        self.preview_button.setToolTip("최종과 같은 비율·자막으로 짧은 변 800px 영상을 빠르게 만들어 틀어 줍니다")
        self.preview_button.clicked.connect(lambda: self._start("preview"))
        self.export_button = QPushButton("내보내기")
        self.export_button.setObjectName("recordButton")
        self.export_button.clicked.connect(lambda: self._start("final"))
        self.cancel_button = QPushButton("취소")
        self.cancel_button.clicked.connect(self.worker.cancel)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.open_folder = QPushButton("내보낸 폴더 열기")
        self.open_folder.clicked.connect(self._open_folder)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(output)
        right_layout.addWidget(self.subtitle_check)
        right_layout.addSpacing(8)
        right_layout.addWidget(self.preview_button)
        right_layout.addWidget(self.export_button)
        right_layout.addWidget(self.progress)
        right_layout.addWidget(self.cancel_button)
        right_layout.addWidget(self.status)
        right_layout.addWidget(self.open_folder)
        right_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(stage_layout(left=left, center=center, right=right))
        self._update_controls()

    # ── 페이지 수명 ──────────────────────────────────────────────

    def activate(self, workspace: Workspace) -> None:
        if workspace != self._workspace:
            self._workspace = workspace
            self._source = None
            self._imported = ()
            self.player.release()
            index = self.preset_combo.findData(self._default_preset) if self._default_preset else -1
            if index > 0:
                self.preset_combo.setCurrentIndex(index)
                self._apply_preset(index)
        self._project = Project.load(workspace.project_file)  # 컷편집에서 고쳤을 수 있다 (탭을 바꾸면 저장된다)
        self._fill_list()
        self._update_controls()

    def deactivate(self) -> None:
        self.player.player.pause()

    def set_default_preset(self, key: str) -> None:
        self._default_preset = key

    # ── 영상 ────────────────────────────────────────────────────

    def _fill_list(self) -> None:
        current = self._source
        self.list.blockSignals(True)
        self.list.clear()
        if self._project.duration:
            fps, cues = self._project.fps, len(self._project.subtitles.cues)
            details = f"{clock(self._project.duration / fps)} · 클립 {len(self._project.placed())}개" + (f" · 자막 {cues}줄" if cues else "")
            add_row(self.list, SourceRow(TIMELINE, "🎬 컷편집 타임라인", details), TIMELINE)
        for source in self._workspace.videos(self._imported):
            path = self._workspace.resolve(source)
            notes = [facts(path, self._sources)] if path.exists() else ["⚠ 파일 없음"]
            row = SourceRow(source, path.name, " · ".join(n for n in notes if n), thumbnail=(self._sources, path))
            add_row(self.list, row, source, enabled=path.exists())
        rows = [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())]
        self.list.setCurrentRow(rows.index(current) if current in rows else -1)
        self.list.blockSignals(False)

    def _import(self) -> None:
        if self._workspace is None:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "내보낼 영상 가져오기", "", VIDEO_FILTER)
        added = [self._workspace.to_source(Path(f)) for f in files]
        if added:
            self._imported = tuple(dict.fromkeys((*self._imported, *added)))
            self._source = added[0]
            self._fill_list()
            self._on_select(self.list.currentRow())

    def _on_select(self, row: int) -> None:
        if row < 0 or self._workspace is None:
            return
        self._source = self.list.item(row).data(Qt.ItemDataRole.UserRole)
        self.player.release()
        if self._source == TIMELINE:
            self.view_label.setText("컷편집 타임라인 — [저해상도 미리보기]로 자막까지 합친 결과를 먼저 보세요")
        else:
            self.player.open(self._sources.media(self._path()), play=False)
            self.view_label.setText(f"원본: {self._path().name}")
        self._update_controls()

    def _path(self) -> Path:
        return self._workspace.resolve(self._source)


    def _format(self) -> OutputFormat:
        return OutputFormat(self.ratio_combo.currentData(), self.fit_combo.currentData(),
                            self.size_combo.currentData(), self.codec_combo.currentData(),
                            self.loudness_check.isChecked(), self.denoise_check.isChecked())

    def _apply_preset(self, index: int) -> None:
        key = self.preset_combo.itemData(index)
        if key is None:
            return
        ratio, fit, short, codec, loudness = PRESETS[key][1]
        for combo, value in ((self.ratio_combo, ratio), (self.fit_combo, fit), (self.size_combo, short), (self.codec_combo, codec)):
            combo.setCurrentIndex(max(0, combo.findData(value)))
        self.loudness_check.blockSignals(True)
        self.loudness_check.setChecked(loudness)
        self.loudness_check.blockSignals(False)
        self._update_controls()

    # ── 미리보기 · 내보내기 ──────────────────────────────────────

    def _output(self, kind: str) -> Path:
        if kind == "preview":
            return self._workspace.cache_dir / "export_preview.mp4"
        stem = self._workspace.root.name if self._source == TIMELINE else self._path().stem
        return self._workspace.final_dir / f"{stem}_{self._format().tag}.mp4"

    def _start(self, kind: str) -> None:
        self.player.release()  # 같은 미리보기 파일을 덮어쓰므로 놓는다 (Windows)
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("저해상도 미리보기 만드는 중…" if kind == "preview" else f"내보내는 중… → {self._output(kind).name}")
        if self._source == TIMELINE:
            self.worker.request_timeline(kind, self._project, self._workspace, self._sources, self._output(kind),
                                         self._format(), self.subtitle_check.isChecked())
        else:
            self.worker.request(kind, self._path(), self._output(kind), self._format(), None)
        self._update_controls()

    def _on_progress(self, kind: str, ratio: float) -> None:
        self.progress.setValue(round(ratio * 1000))
        label = "저해상도 미리보기" if kind == "preview" else "내보내기"
        self.status.setText(f"{label} {int(ratio * 100)}%")

    def _on_finished(self, kind: str, output: Path) -> None:
        self.progress.hide()
        self._update_controls()
        if kind == "preview":
            self.status.setText("저해상도 미리보기 — 최종 파일과 같은 비율·자막입니다")
            name = "컷편집 타임라인" if self._source == TIMELINE else self._path().name
            self.view_label.setText(f"저해상도 미리보기 (짧은 변 800px): {name}")
            self.player.open(output)
        else:
            size_mb = output.stat().st_size / 1e6
            self.status.setText(f"완성: {output.name} ({size_mb:.0f}MB)")
            self.message.emit(f"내보내기 완성: {output.name}", INFO)
            self.view_label.setText(f"내보낸 파일: {output.name}")
            self.player.open(output, play=False)

    def _on_failed(self, kind: str, reason: str) -> None:
        self.progress.hide()
        self.status.setText(f"⚠ 실패: {reason}")
        self.message.emit(f"⚠ 내보내기 실패: {reason}", ERROR)
        self._update_controls()

    def _on_cancelled(self, kind: str) -> None:
        self.progress.hide()
        self.status.setText("취소했습니다")
        self._update_controls()

    def _open_folder(self) -> None:
        folder = self._workspace.final_dir
        folder.mkdir(parents=True, exist_ok=True)
        open_folder(folder)  # 탐색기 · Finder

    def _update_controls(self) -> None:
        ready = self._source is not None
        busy = self.worker.busy is not None
        self.fit_combo.setEnabled(self.ratio_combo.currentData() != "original")
        cues = len(self._project.subtitles.cues) if self._source == TIMELINE else 0
        if self._source == TIMELINE:
            self.subtitle_check.setText(f"자막 넣기 ({cues}줄)" if cues else "자막 없음 — 컷편집 '자막' 탭에서 만들 수 있습니다")
        else:
            self.subtitle_check.setText("자막은 컷편집 타임라인을 내보낼 때 들어갑니다")
        self.subtitle_check.setEnabled(cues > 0)
        timeline = self._source == TIMELINE
        for check in (self.loudness_check, self.denoise_check):  # 소리 다듬기는 컷편집 타임라인에만
            check.setEnabled(timeline)
            check.setToolTip(check.toolTip() if timeline else "컷편집 타임라인을 내보낼 때 쓸 수 있습니다")
        for widget in (self.preview_button, self.export_button):
            widget.setEnabled(ready and not busy)
        self.cancel_button.setVisible(busy)
        self.open_folder.setEnabled(self._workspace is not None)
