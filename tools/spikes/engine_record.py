"""검증: B 엔진으로 N초 녹화 → 마무리(MP4). 사용법: python engine_record.py <출력폴더> <초>"""

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtMultimedia import QVideoSink
from PySide6.QtWidgets import QApplication

from ks_studio.app.paths import SETTINGS_PATH
from ks_studio.core.settings import AppSettings
from ks_studio.services.capture import CameraPreview, MicInput
from ks_studio.services.devices import best_format, find_camera, find_microphone
from ks_studio.services.media import finalize_recording
from ks_studio.services.recording import Recorder

logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
out_dir, seconds = Path(sys.argv[1]), float(sys.argv[2])
app = QApplication([])
settings = AppSettings.load(SETTINGS_PATH)
camera_device = find_camera(settings.camera)
fmt = best_format(camera_device, 30)

sink = QVideoSink()
camera = CameraPreview(sink)
mic = MicInput()
recorder = Recorder(sink, mic)
camera.start(camera_device, fmt)
recorder.reset(30)
mic.start(find_microphone(settings.microphone))
print("mic:", mic.sample_rate, "Hz", mic.channels, "ch")

partial = out_dir / "engine.mkv"


def on_stopped(path: Path) -> None:
    finalize_recording(path, out_dir / "engine.mp4")
    print("finalized")
    app.quit()


recorder.stopped.connect(on_stopped)
recorder.failed.connect(lambda m: (print("FAILED", m), app.quit()))
QTimer.singleShot(2000, lambda: recorder.start(partial, 30, (1920, 1080)))
QTimer.singleShot(int(2000 + seconds * 1000), recorder.stop)
app.exec()
