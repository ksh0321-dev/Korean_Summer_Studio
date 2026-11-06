"""실험: QAudioSink.processedUSecs()가 '쓴 양'인지 '장치로 넘어간 양'인지. 미리보기 A/V 정렬 공식이 이걸로 정해진다."""

import time

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtMultimedia import QAudioFormat, QAudioSink, QMediaDevices
from PySide6.QtWidgets import QApplication

RATE, BPS = 48000, 4
app = QApplication([])
fmt = QAudioFormat()
fmt.setSampleRate(RATE)
fmt.setChannelCount(1)
fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
sink = QAudioSink(QMediaDevices.defaultAudioOutput(), fmt)
sink.setBufferSize(int(RATE * 0.2) * BPS)
io = sink.start()
written = 0
t0 = time.perf_counter()
rows = []
silence = np.zeros(1600, np.float32).tobytes()


def tick() -> None:
    global written
    while sink.bytesFree() >= len(silence):
        io.write(silence)
        written += len(silence)
    buffered = sink.bufferSize() - sink.bytesFree()
    rows.append((time.perf_counter() - t0, written / BPS / RATE, sink.processedUSecs() / 1e6, buffered / BPS / RATE))


timer = QTimer()
timer.timeout.connect(tick)
timer.start(10)


def report() -> None:
    for wall, w, p, b in rows[::50]:
        print(f"wall {wall:5.2f}s  written {w:5.2f}s  processed {p:5.2f}s  buffered {b:4.2f}s  written-buffered {w - b:5.2f}s")
    app.quit()


QTimer.singleShot(3000, report)
app.exec()
