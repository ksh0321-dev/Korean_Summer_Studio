"""화면용 계산을 작업 스레드에서: 밀린 요청은 버리고 가장 최근 것만 돌린다 (UI 스레드가 멈추지 않게).

슬라이더를 끄는 동안의 AI 샘플 컷, 자막 탭 재생 프레임 변환처럼 '최신 결과만 의미 있는' 일에 쓴다.
ready(번호, 결과)는 UI 스레드로 간다 — 받는 쪽은 자기가 마지막으로 낸 번호인지 보고 쓴다.
"""

import logging
import threading

from PySide6.QtCore import QObject, Signal

log = logging.getLogger(__name__)


class LatestWorker(QObject):
    ready = Signal(int, object)

    def __init__(self, name: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._job = None
        self._token = 0
        self.busy = False  # 맡은 일이 있다 (기다리는 중이거나 계산 중) — 재생 프레임은 바쁘면 다른 작업자에게
        threading.Thread(target=self._worker, daemon=True, name=name).start()

    def request(self, compute) -> int:
        with self._lock:
            self._token += 1
            self._job = (self._token, compute)
            self.busy = True  # 맡기는 순간부터 (스레드가 집어 가기 전에 또 맡기면 앞 일이 버려진다)
        self._wake.set()
        return self._token

    def _worker(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            token, compute = job
            try:
                result = compute()
            except Exception:
                log.exception("화면용 계산 실패")
                result = None
            with self._lock:
                self.busy = self._job is not None
            if result is not None:
                self.ready.emit(token, result)
