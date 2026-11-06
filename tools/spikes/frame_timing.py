"""실험: 프레임 도착 간격(카메라 스레드) vs QVideoFrame.startTime 간격. 사용법: python frame_timing.py"""

import statistics
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtMultimedia import QVideoSink
from PySide6.QtWidgets import QApplication

from ks_studio.app.paths import SETTINGS_PATH
from ks_studio.core.settings import AppSettings
from ks_studio.services.capture import CameraPreview
from ks_studio.services.devices import best_format, find_camera

app = QApplication([])
device = find_camera(AppSettings.load(SETTINGS_PATH).camera)
sink = QVideoSink()
camera = CameraPreview(sink)
rows = []
sink.videoFrameChanged.connect(
    lambda f: rows.append((time.monotonic(), f.startTime())), Qt.ConnectionType.DirectConnection
)
camera.start(device, best_format(device, 30))
QTimer.singleShot(8000, app.quit)
app.exec()

rows = rows[30:]  # 시작 직후 1초는 버림
arrival = [round((b[0] - a[0]) * 1000, 1) for a, b in zip(rows, rows[1:])]
start = [round((b[1] - a[1]) / 1000, 2) for a, b in zip(rows, rows[1:])]
lat = [r[0] * 1e6 - r[1] for r in rows]
print(f"frames {len(rows)}")
print(f"arrival delta ms: mean {statistics.mean(arrival):.2f} stdev {statistics.pstdev(arrival):.2f} min {min(arrival)} max {max(arrival)}")
print("arrival hist:", sorted({d: arrival.count(d) for d in set(arrival)}.items(), key=lambda kv: -kv[1])[:10])
print(f"startTime delta ms: mean {statistics.mean(start):.2f} stdev {statistics.pstdev(start):.2f} min {min(start)} max {max(start)}")
print("startTime hist:", sorted({d: start.count(d) for d in set(start)}.items(), key=lambda kv: -kv[1])[:10])
print(f"arrival - startTime (us, 상대값): spread {max(lat) - min(lat):.0f}  first {rows[0][1]}")
