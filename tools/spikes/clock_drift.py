"""실험: 카메라(startTime)·마이크(받은 샘플 수) 시계가 PC 시계(time.monotonic)와 같은 속도로 가는지.

각 시계마다 '(PC 시각 - 장치 시각)'의 30초 구간 최솟값을 출력한다. 이 값이 시간에 따라 흐르면 드리프트다.
사용법: python clock_drift.py <초>
"""

import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtMultimedia import QVideoSink
from PySide6.QtWidgets import QApplication

from ks_studio.app.paths import SETTINGS_PATH
from ks_studio.core.settings import AppSettings
from ks_studio.services.capture import CameraPreview, MicInput
from ks_studio.services.devices import best_format, find_camera, find_microphone

seconds = int(sys.argv[1])
app = QApplication([])
settings = AppSettings.load(SETTINGS_PATH)
camera_device = find_camera(settings.camera)
sink = QVideoSink()
camera = CameraPreview(sink)
mic = MicInput()

video_rows, audio_rows = [], []
sink.videoFrameChanged.connect(
    lambda f: video_rows.append((time.monotonic(), f.startTime() / 1e6)), Qt.ConnectionType.DirectConnection
)
mic.samples.connect(lambda chunk, first: audio_rows.append((time.monotonic(), (first + len(chunk)) / mic.sample_rate)))
camera.start(camera_device, best_format(camera_device, 30))
mic.start(find_microphone(settings.microphone))
QTimer.singleShot(seconds * 1000, app.quit)
app.exec()


def trend(rows, label):
    t0 = rows[0][0]
    base = None
    print(f"[{label}]")
    for start in range(0, seconds, 30):
        window = [pc - dev for pc, dev in rows if start <= pc - t0 < start + 30]
        if not window:
            continue
        floor = min(window) * 1000
        base = floor if base is None else base
        print(f"  {start:4d}s~  (PC - 장치) 바닥 {floor - base:+8.2f}ms")
    pc_span = rows[-1][0] - rows[0][0]
    dev_span = rows[-1][1] - rows[0][1]
    print(f"  전체: PC {pc_span:.3f}s 동안 장치 {dev_span:.3f}s → {(dev_span / pc_span - 1) * 1e6:+.0f} ppm")


trend(video_rows, "카메라 startTime")
trend(audio_rows, f"마이크 샘플 수 ({mic.sample_rate}Hz)")
