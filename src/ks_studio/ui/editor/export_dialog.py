"""임시 내보내기 진행 창: 남은 시간 표시, 취소 가능 (쓰다 만 파일은 지운다)."""

import logging
import math
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from ks_studio.core.project import Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.export import export_project
from ks_studio.services.sources import SourceCache

log = logging.getLogger(__name__)
SPEED_WINDOW_S = 5


class _Signals(QObject):
    progress = Signal(int, int)
    finished = Signal(bool, str)  # 성공 여부, 오류 메시지


class ExportDialog(QDialog):
    def __init__(
        self,
        project: Project,
        workspace: Workspace,
        sources: SourceCache,
        output: Path,
        size: tuple[int, int],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("내보내기 (임시)")
        self.setMinimumWidth(460)
        self._output = output
        self._cancel = threading.Event()
        self._running = True
        self._samples: list[tuple[float, int]] = []

        self.label = QLabel(f"{output.name}\n\n남은 시간 계산 중…")
        self.bar = QProgressBar()
        self.bar.setRange(0, max(1, project.duration))
        self.open_button = QPushButton("폴더 열기")
        self.open_button.setVisible(False)
        self.open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(output.parent))))
        self.close_button = QPushButton("취소")
        self.close_button.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.bar)
        layout.addLayout(buttons)

        self._signals = _Signals(self)
        self._signals.progress.connect(self._on_progress)
        self._signals.finished.connect(self._on_finished)
        threading.Thread(
            target=self._run, args=(project, workspace, sources, output, size), daemon=True, name="export"
        ).start()

    def _run(self, project, workspace, sources, output, size) -> None:
        try:
            ok = export_project(project, workspace, sources, output, size, self._signals.progress.emit, self._cancel)
        except Exception as e:
            log.exception("내보내기 실패")
            self._signals.finished.emit(False, str(e))
            return
        self._signals.finished.emit(ok, "")

    def _on_progress(self, done: int, total: int) -> None:
        self.bar.setValue(done)
        now = time.monotonic()
        self._samples = [s for s in self._samples if now - s[0] <= SPEED_WINDOW_S] + [(now, done)]
        start, start_done = self._samples[0]
        speed = (done - start_done) / (now - start) if now > start else 0.0
        eta = f"{math.ceil((total - done) / speed):02d}초 남음" if speed > 0 else "남은 시간 계산 중…"
        self.label.setText(f"{self._output.name}\n\n{eta}")

    def _on_finished(self, ok: bool, error: str) -> None:
        self._running = False
        if self._cancel.is_set():
            super().reject()
            return
        self.bar.setValue(self.bar.maximum())
        self.label.setText(f"완료: {self._output}" if ok else f"⚠ 내보내기 실패: {error}")
        self.open_button.setVisible(ok)
        self.close_button.setText("닫기")

    def reject(self) -> None:
        if self._running:  # 취소 요청 → 스레드가 멈추고 파일을 지우면 _on_finished에서 닫힌다
            self._cancel.set()
            self.close_button.setEnabled(False)
            self.label.setText("취소하는 중…")
            return
        super().reject()
