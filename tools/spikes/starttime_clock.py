"""실험: QVideoFrame.startTime이 time.monotonic과 같은 시계인지 (차이의 바닥값이 시간에 따라 흐르는지).

사용법: python starttime_clock.py <초>
"""

import sys
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtMultimedia import QVideoSink
from PySide6.QtWidgets import QApplication

from ks_studio.app.paths import SETTINGS_PATH
from ks_studio.core.settings import AppSettings
from ks_studio.services.capture import CameraPreview
from ks_studio.services.devices import best_format, find_camera

seconds = int(sys.argv[1])
app = QApplication([])
device = find_camera(AppSettings.load(SETTINGS_PATH).camera)
sink = QVideoSink()
camera = CameraPreview(sink)
rows = []
sink.videoFrameChanged.connect(
    lambda f: rows.append((time.monotonic(), f.startTime() / 1e6)), Qt.ConnectionType.DirectConnection
)
camera.start(device, best_format(device, 30))
QTimer.singleShot(seconds * 1000, app.quit)
app.exec()

t0 = rows[0][0]
for sec in range(0, seconds, max(1, seconds // 10)):
    diffs = [(a - s) * 1000 for a, s in rows if sec <= a - t0 < sec + 1]
    print(f"{sec:3d}s  (arrival - startTime) floor {min(diffs) - (rows[0][0] - rows[0][1]) * 1000:7.2f}ms  spread {max(diffs) - min(diffs):5.1f}ms")
