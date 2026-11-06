"""앱 맨 아래 상태 표시줄. 왼쪽 = 알림 메시지, 오른쪽 = 늘 보이는 표시 (녹화 중 · 싱크 보정 · fps).

알림 수준: info(6초 뒤 사라짐) · warning(15초) · error(다음 알림이 올 때까지 남음).
지나간 알림은 메시지에 마우스를 올리면 최근 것부터 볼 수 있다.
경고 이상 로그(logging)도 여기로 모은다. 로그는 다른 스레드에서도 오므로 Qt 시그널로 넘긴다.
"""

import logging
import time
from collections import deque

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QStatusBar, QWidget

INFO, WARNING, ERROR, LIVE = "info", "warning", "error", "live"  # live = 녹화 중처럼 지금 진행 중인 것 (빨강)
TIMEOUT_MS = {INFO: 6000, WARNING: 15000, ERROR: 0, LIVE: 0}
HISTORY = 20
INDICATORS = ("recording", "sync", "fps")  # 오른쪽 표시 순서


def _set_level(label: QLabel, level: str) -> None:
    label.setProperty("level", level)
    label.style().polish(label)


class StatusBar(QStatusBar):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizeGripEnabled(False)
        self._message = QLabel()
        self._message.setObjectName("statusMessage")
        self._message.setTextFormat(Qt.TextFormat.PlainText)
        self._message.setMinimumWidth(1)  # 긴 메시지가 오른쪽 표시를 밀어내지 않게
        self.addWidget(self._message, 1)
        self._history: deque[str] = deque(maxlen=HISTORY)
        self._clear = QTimer(self)
        self._clear.setSingleShot(True)
        self._clear.timeout.connect(self._message.clear)
        self._indicators: dict[str, QLabel] = {}
        for key in INDICATORS:
            label = QLabel()
            label.setObjectName("statusIndicator")
            label.hide()
            self.addPermanentWidget(label)
            self._indicators[key] = label

    def show_message(self, text: str, level: str = INFO) -> None:
        self._history.appendleft(f"{time.strftime('%H:%M:%S')}  {text}")
        self._message.setText(text)
        self._message.setToolTip("최근 알림 (최신순)\n" + "\n".join(self._history))
        _set_level(self._message, level)
        if TIMEOUT_MS[level]:
            self._clear.start(TIMEOUT_MS[level])
        else:
            self._clear.stop()

    def set_indicator(self, key: str, text: str, level: str = INFO) -> None:
        """오른쪽 표시. 빈 글자면 숨긴다."""
        label = self._indicators[key]
        label.setText(text)
        label.setVisible(bool(text))
        _set_level(label, level)


class LogBridge(QObject):
    """ks_studio 로거의 경고·오류를 상태 표시줄로 보낸다 (어느 스레드에서 로그를 남겨도 안전)."""

    message = Signal(str, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.handler = _SignalHandler(self.message)
        self.handler.setLevel(logging.WARNING)


class _SignalHandler(logging.Handler):
    def __init__(self, signal) -> None:
        super().__init__()
        self._signal = signal

    def emit(self, record: logging.LogRecord) -> None:
        level = ERROR if record.levelno >= logging.ERROR else WARNING
        self._signal.emit(record.getMessage(), level)
