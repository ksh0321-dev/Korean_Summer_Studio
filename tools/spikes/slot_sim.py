"""실험: 카메라 도착 시각으로 FrameSlotter를 재생해 복제/버림 위치를 출력.

사용법: python slot_sim.py <도착시각.txt>          (파일이 없으면 22초 동안 카메라에서 모아 저장)
"""

import sys
import time
from pathlib import Path

from ks_studio.core.cfr import FrameSlotter

path = Path(sys.argv[1])
if not path.exists():
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
    collected = []
    sink.videoFrameChanged.connect(lambda f: collected.append(f.startTime() / 1e6), Qt.ConnectionType.DirectConnection)
    camera.start(device, best_format(device, 30))
    QTimer.singleShot(int(sys.argv[2]) * 1000 if len(sys.argv) > 2 else 22000, app.quit)
    app.exec()
    path.write_text("\n".join(map(repr, collected)))

arrivals = [float(x) for x in path.read_text().split()]
span = arrivals[-1] - arrivals[0]
print(f"frames {len(arrivals)} over {span:.2f}s -> camera {(len(arrivals) - 1) / span:.3f} fps")
slotter = FrameSlotter(30, settle_frames=60)
events = []
for i, t in enumerate(arrivals):
    slots = slotter.place(t)
    if len(slots) != 1:
        events.append(f"#{i} t={t - arrivals[0]:.3f} -> {slots}")
print(f"dup/drop events {len(events)}, dropped {slotter.dropped}")
print("\n".join(events[:20]))

# 흔들림을 뺀 '바닥' 지연 추세: 1초 구간마다 (도착 - i/30)의 최솟값
base = arrivals[0]
for sec in range(int(span)):
    window = [(t - base - i / 30) * 1000 for i, t in enumerate(arrivals) if sec <= t - base < sec + 1]
    print(f"{sec:2d}s  floor {min(window):7.1f}ms  max {max(window):7.1f}ms")
