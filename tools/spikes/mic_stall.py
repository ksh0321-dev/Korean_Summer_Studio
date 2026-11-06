"""실험: 메인 스레드가 몇 초 멈추면 마이크 입력(QAudioSource)이 끊기는지.

사용법: python mic_stall.py <멈출 초>
"""

import sys
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ks_studio.app.paths import SETTINGS_PATH
from ks_studio.core.settings import AppSettings
from ks_studio.services.capture import MicInput
from ks_studio.services.devices import find_microphone

stall = float(sys.argv[1])
app = QApplication([])
mic = MicInput()
received = []
mic.samples.connect(lambda chunk, first: received.append((time.monotonic(), len(chunk), first)))
mic.start(find_microphone(AppSettings.load(SETTINGS_PATH).microphone))
t0 = time.monotonic()
print(f"입력 버퍼 {mic._source.bufferSize() / 4 / mic.channels / mic.sample_rate:.2f}s", flush=True)


def count(tag):
    _, n, first = received[-1]
    position = (first + n) / mic.sample_rate  # 건너뛴 번호까지 포함한 '오디오 시각'
    print(
        f"[{tag}] 경과 {time.monotonic() - t0:.2f}s · 오디오 위치 {position:.2f}s · 실제로 받은 {sum(r[1] for r in received) / mic.sample_rate:.2f}s",
        flush=True,
    )


QTimer.singleShot(2000, lambda: count("멈추기 전"))
QTimer.singleShot(2100, lambda: time.sleep(stall))  # 메인 스레드를 막는다 (박수 분석처럼)
QTimer.singleShot(int((2.2 + stall) * 1000), lambda: count("멈춘 직후"))
QTimer.singleShot(int((5.2 + stall) * 1000), lambda: (count("3초 더 지나서"), app.quit()))
app.exec()
