"""내보내기 단계 (영상 편집의 마지막): 영상 하나 → 고른 비율·해상도·형식의 최종 MP4, 자막은 구워서.

어느 단계를 거쳤든 바로 내보낸다 — 자막만 만든 영상, AI 보정만 한 영상, 녹화본 그대로도.
AI 보정본에 자막이 없으면 그 원본 영상의 자막을 쓴다 (보정본은 원본과 프레임이 같다).
[저해상도 미리보기]: 짧은 변 800px로 빠르게 만들어 바로 튼다 — 최종 파일을 만들어 확인하는 수고를 던다.
최종 파일: <프로젝트>/exports/final/<이름>_<비율>_<해상도>.mp4
"""

import os
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

from ks_studio.core.beauty import load_ai_state
from ks_studio.core.subtitles import Subtitles, load_subtitles
from ks_studio.core.workspace import Workspace
from ks_studio.services.deliver import CODECS, FITS, RATIOS, RESOLUTIONS, ExportWorker, OutputFormat
from ks_studio.services.sources import SourceCache
from ks_studio.ui.editor.media_bin import VIDEO_FILTER
from ks_studio.ui.widgets.clip_player import ClipPlayer
from ks_studio.ui.widgets.layout import stage_layout
from ks_studio.ui.widgets.source_row import SourceRow, add_row, facts
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
        self._subtitles: dict = {}
        self._ai_sources: tuple[str, ...] = ()
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
        self.ratio_combo = _combo({k: v[0] for k, v in RATIOS.items()}, fmt.ratio)
        self.fit_combo = _combo(FITS, fmt.fit)
        self.size_combo = _combo(RESOLUTIONS, fmt.short)
        self.codec_combo = _combo(CODECS, fmt.codec)
        self.ratio_combo.currentIndexChanged.connect(self._update_controls)
        form.addRow("비율", self.ratio_combo)
        form.addRow("맞추기", self.fit_combo)
        form.addRow("해상도", self.size_combo)
        form.addRow("형식", self.codec_combo)
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
        self._subtitles = load_subtitles(workspace.subtitles_file).docs  # 자막 탭에서 고쳤을 수 있다
        self._ai_sources = tuple(load_ai_state(workspace.ai_settings_file).settings)
        self._fill_list()
        self._update_controls()

    def deactivate(self) -> None:
        self.player.player.pause()

    # ── 영상 ────────────────────────────────────────────────────

    def _fill_list(self) -> None:
        current = self._source
        self.list.blockSignals(True)
        self.list.clear()
        for source in self._workspace.videos(self._imported):
            path = self._workspace.resolve(source)
            notes = [facts(path, self._sources)] if path.exists() else ["⚠ 파일 없음"]
            subs, _ = self._subtitles_for(source)
            if subs is not None:
                notes.append(f"자막 {len(subs.cues)}줄")
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
        self.player.open(self._sources.media(self._path()), play=False)
        self.view_label.setText(f"원본: {self._path().name}")
        self._update_controls()

    def _path(self) -> Path:
        return self._workspace.resolve(self._source)

    def _subtitles_for(self, source: str) -> tuple[Subtitles | None, str]:
        """(구울 자막, 설명). 그 영상의 자막, 없으면 AI 보정본의 원본 영상 자막."""
        subs = self._subtitles.get(source)
        if subs is not None and subs.cues:
            return subs, f"자막 넣기 ({len(subs.cues)}줄)"
        path = self._workspace.resolve(source)
        for original in self._ai_sources:
            if self._workspace.ai_output(original) == path:
                subs = self._subtitles.get(original)
                if subs is not None and subs.cues:
                    return subs, f"자막 넣기 (원본 {Path(original).name}의 {len(subs.cues)}줄)"
        return None, "자막 없음 — 자막 탭에서 만들 수 있습니다"

    def _format(self) -> OutputFormat:
        return OutputFormat(self.ratio_combo.currentData(), self.fit_combo.currentData(),
                            self.size_combo.currentData(), self.codec_combo.currentData())

    # ── 미리보기 · 내보내기 ──────────────────────────────────────

    def _output(self, kind: str) -> Path:
        if kind == "preview":
            return self._workspace.cache_dir / "export_preview.mp4"
        return self._workspace.final_dir / f"{self._path().stem}_{self._format().tag}.mp4"

    def _start(self, kind: str) -> None:
        subs, _ = self._subtitles_for(self._source)
        subs = subs if self.subtitle_check.isChecked() else None
        self.player.release()  # 같은 미리보기 파일을 덮어쓰므로 놓는다 (Windows)
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("저해상도 미리보기 만드는 중…" if kind == "preview" else f"내보내는 중… → {self._output(kind).name}")
        self.worker.request(kind, self._path(), self._output(kind), self._format(), subs)
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
            self.view_label.setText(f"저해상도 미리보기 (짧은 변 800px): {self._path().name}")
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
        os.startfile(folder)  # Windows 탐색기

    def _update_controls(self) -> None:
        ready = self._source is not None
        busy = self.worker.busy is not None
        self.fit_combo.setEnabled(self.ratio_combo.currentData() != "original")
        subs, label = self._subtitles_for(self._source) if ready else (None, "자막 넣기")
        self.subtitle_check.setText(label)
        self.subtitle_check.setEnabled(subs is not None)
        for widget in (self.preview_button, self.export_button):
            widget.setEnabled(ready and not busy)
        self.cancel_button.setVisible(busy)
        self.open_folder.setEnabled(self._workspace is not None)
