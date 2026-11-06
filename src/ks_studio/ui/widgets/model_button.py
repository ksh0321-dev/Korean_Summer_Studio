"""모델 하나를 내려받는 작은 버튼: '⬇ 얼굴 290MB' → '얼굴 37%' → '✓ 얼굴'. 받기는 작업 스레드에서."""

import threading

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QPushButton, QWidget


class ModelButton(QPushButton):
    finished = Signal(str)  # 실패 이유, 성공이면 ""
    _progress = Signal(float)  # 작업 스레드 → 화면
    _done = Signal(str)

    def __init__(self, name: str, size_mb: int, ready, download, purpose: str, parent: QWidget | None = None) -> None:
        """ready() → 받았나, download(on_progress) → 받기 (실패하면 예외). purpose = 툴팁에 쓸 '무엇에 쓰나'."""
        super().__init__(parent)
        self.name, self._size_mb, self._ready, self._download, self._purpose = name, size_mb, ready, download, purpose
        self._downloading = False
        self._missing: str | None = None
        self.clicked.connect(self._start)
        self._progress.connect(lambda r: self.setText(f"{self.name} {int(r * 100)}%"))
        self._done.connect(self._on_done)
        self.refresh()

    def set_missing(self, reason: str | None) -> None:
        """그 모델을 쓰는 패키지가 없으면 이유를 주고 막는다."""
        self._missing = reason
        self.refresh()

    def refresh(self) -> None:
        if self._downloading:
            return
        if self._missing:
            self.setText(f"{self.name}: 패키지 없음")
            self.setEnabled(False)
            self.setToolTip(self._missing)
            return
        ready = self._ready()
        size = f"{self._size_mb / 1000:.1f}GB" if self._size_mb >= 1000 else f"{self._size_mb}MB"
        self.setText(f"✓ {self.name}" if ready else f"⬇ {self.name} {size}")
        self.setEnabled(not ready)
        state = "받아 둠" if ready else "눌러서 models 폴더에 내려받기 (인터넷 필요, 처음 한 번)"
        self.setToolTip(f"{self.name} 모델 — {self._purpose}\n{state}")

    def _start(self) -> None:
        self._downloading = True
        self.setEnabled(False)
        self.setText(f"{self.name} 0%")

        def work() -> None:
            try:
                self._download(self._progress.emit)
            except Exception as e:  # 인터넷 끊김 등 — 화면에 알리고 다시 누를 수 있게
                self._done.emit(str(e) or type(e).__name__)
                return
            self._done.emit("")

        threading.Thread(target=work, daemon=True, name=f"model-download-{self.name}").start()

    def _on_done(self, error: str) -> None:
        self._downloading = False
        self.refresh()
        self.finished.emit(error)
