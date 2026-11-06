"""앱 종료 대기 창: 마무리 중인 녹화가 끝날 때까지 닫을 수 없다. 남은 시간을 보여준다."""

import math
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout, QWidget

from ks_studio.ui.recorder.recorder_page import RecorderPage

SPEED_WINDOW_S = 5  # 최근 이 시간 동안의 처리 속도로 남은 시간을 계산


class CloseWaitDialog(QDialog):
    def __init__(self, page: RecorderPage, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("녹화 마무리 중")
        # 닫기(X) 버튼 없음 — 작업이 끝나야만 닫힌다
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        self.setMinimumWidth(420)
        self._page = page
        self._samples: list[tuple[float, float]] = []  # (시각, 지금까지 마무리한 미디어 초)

        self.label = QLabel()
        bar = QProgressBar()
        bar.setRange(0, 0)  # 끝을 모르는 진행 표시
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(bar)

        page.idle.connect(self.accept)
        timer = QTimer(self)
        timer.timeout.connect(self._update)
        timer.start(500)
        self._update()

    def reject(self) -> None:
        """Esc로도 닫히지 않게 한다."""

    def _update(self) -> None:
        processed, remaining = self._page.finalize_progress()
        now = time.monotonic()
        self._samples = [s for s in self._samples if now - s[0] <= SPEED_WINDOW_S] + [(now, processed)]
        start, start_processed = self._samples[0]
        speed = (processed - start_processed) / (now - start) if now > start else 0.0
        if remaining is None or speed <= 0:
            eta = "남은 시간 계산 중…"
        else:
            eta = f"{math.ceil(remaining / speed):02d}초 남음"
        self.label.setText(f"녹화를 MP4로 마무리하고 있습니다. 끝나면 자동으로 닫힙니다.\n\n{eta}")
