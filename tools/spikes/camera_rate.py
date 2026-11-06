"""실험: 카메라가 실제로 몇 fps를 주는지, 그 시각으로 FrameSlotter가 복제·버림을 얼마나 하는지.

사용법: python camera_rate.py <카메라 ID 일부> <fps> <초> [jpeg]
"""

import statistics
import sys
import time
from collections import Counter

from PySide6.QtCore import Qt, QTimer
from PySide6.QtMultimedia import QMediaDevices, QVideoSink
from PySide6.QtWidgets import QApplication

from ks_studio.core.cfr import FrameSlotter
from ks_studio.services.capture import CameraPreview
from ks_studio.services.devices import best_format

needle, fps, seconds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
app = QApplication([])
device = next(d for d in QMediaDevices.videoInputs() if needle in bytes(d.id()).decode())
fmt = best_format(device, fps)
if len(sys.argv) > 4:  # JPEG 포맷 강제
    fmt = next(
        f for f in device.videoFormats()
        if f.resolution().width() == 1920 and round(f.maxFrameRate()) == fps and f.pixelFormat().name == "Format_Jpeg"
    )
print("카메라:", device.description(), "| 포맷:", fmt.resolution().width(), fmt.resolution().height(), fmt.maxFrameRate(), fmt.pixelFormat().name)
sink = QVideoSink()
camera = CameraPreview(sink)
stamps = []
sink.videoFrameChanged.connect(lambda f: stamps.append((f.startTime() / 1e6, time.monotonic())), Qt.ConnectionType.DirectConnection)
camera.start(device, fmt)
QTimer.singleShot(seconds * 1000, app.quit)
app.exec()

stamps = stamps[fps * 2 :]  # 시작 2초 제외
times = [s for s, _ in stamps]
deltas = [(b - a) * 1000 for a, b in zip(times, times[1:])]
span = times[-1] - times[0]
print(f"받은 프레임 {len(times)}개 / {span:.2f}s → 실제 {(len(times) - 1) / span:.2f}fps")
print(f"간격: 평균 {statistics.mean(deltas):.2f}ms 표준편차 {statistics.pstdev(deltas):.2f} 최소 {min(deltas):.1f} 최대 {max(deltas):.1f}")
print("간격 분포(상위):", Counter(round(d) for d in deltas).most_common(8))
slotter = FrameSlotter(fps, settle_frames=fps * 2)
placed = [slotter.place(t) for t in times]
dups = sum(len(p) - 1 for p in placed if p)
print(f"FrameSlotter({fps}): 복제 {dups} · 버림 {slotter.dropped} · 칸 {slotter.next_slot}")
