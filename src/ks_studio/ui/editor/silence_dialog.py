"""무음 구간 자르기 창: 기준을 정하면 V1에서 조용한 구간을 찾아 보여 주고, '자르기'를 누르면 한 번에 잘라낸다.

찾기만 할 뿐 프로젝트는 바꾸지 않는다. 잘라내기는 편집 화면이 ranges로 한다 (되돌리기 한 번에 전부 돌아온다).
"""

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QListWidget,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ks_studio.core.project import MAIN_TRACK, Project
from ks_studio.core.silence import silent_spans, timeline_ranges
from ks_studio.core.workspace import Workspace
from ks_studio.services.sources import SourceCache, compute_peaks
from ks_studio.ui.editor.timeline import format_frames

DEFAULTS = {"threshold_db": -40, "min_ms": 600, "pad_ms": 150}


class SilenceDialog(QDialog):
    def __init__(self, project: Project, workspace: Workspace, sources: SourceCache, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("무음 구간 자르기")
        self.resize(460, 520)
        self._project = project
        self.ranges: list[tuple[int, int]] = []
        # V1 클립의 원본마다 10ms 피크 (같은 원본은 한 번만 계산)
        self._peaks: dict[str, object] = {}
        for placed in project.placed():
            if placed.track == MAIN_TRACK and placed.clip.source not in self._peaks:
                path = workspace.resolve(placed.clip.source)
                self._peaks[placed.clip.source] = compute_peaks(sources.audio(path)) if path.exists() else []

        self.threshold = self._spin(-70, -10, DEFAULTS["threshold_db"], " dB", 1)
        self.min_length = self._spin(100, 5000, DEFAULTS["min_ms"], " ms", 50)
        self.padding = self._spin(0, 1000, DEFAULTS["pad_ms"], " ms", 25)
        form = QFormLayout()
        form.addRow("이보다 조용하면 무음", self.threshold)
        form.addRow("이만큼 이어지면 자르기", self.min_length)
        form.addRow("앞뒤로 남길 여유", self.padding)
        hint = QLabel("V1(메인) 소리를 기준으로 찾습니다. 위 줄 클립은 연동 규칙대로 같이 정리됩니다.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        self.summary = QLabel()
        self.list = QListWidget()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("자르기")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("취소")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(self.summary)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.buttons)
        self._update()

    def _spin(self, low: int, high: int, value: int, suffix: str, step: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(low, high)
        box.setValue(value)
        box.setSuffix(suffix)
        box.setSingleStep(step)
        box.valueChanged.connect(lambda _v: self._update())
        return box

    def _update(self) -> None:
        threshold, min_ms, pad_ms = self.threshold.value(), self.min_length.value(), self.padding.value()
        spans = {
            p.clip.id: silent_spans(self._peaks[p.clip.source], threshold, min_ms, pad_ms)
            for p in self._project.placed()
            if p.track == MAIN_TRACK
        }
        self.ranges = timeline_ranges(self._project, spans)
        fps = self._project.fps
        total = sum(b - a for a, b in self.ranges)
        self.summary.setText(f"무음 {len(self.ranges)}곳 · 합계 {total / fps:.1f}초 (전체 {self._project.duration / fps:.1f}초)")
        self.list.clear()
        for a, b in self.ranges:
            self.list.addItem(f"{format_frames(a, fps)} ~ {format_frames(b, fps)}   ({(b - a) / fps:.2f}초)")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.ranges))
