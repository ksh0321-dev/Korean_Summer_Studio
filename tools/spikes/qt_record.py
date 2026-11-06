"""실험: Qt QMediaRecorder로 N초 녹화. 사용법: python qt_record.py <출력파일(.mkv/.mp4)> <초>"""

import sys

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtMultimedia import QAudioInput, QCamera, QMediaCaptureSession, QMediaFormat, QMediaRecorder
from PySide6.QtWidgets import QApplication

from ks_studio.app.paths import SETTINGS_PATH
from ks_studio.core.settings import AppSettings
from ks_studio.services.devices import best_format, find_camera, find_microphone

out, seconds = sys.argv[1], float(sys.argv[2])
app = QApplication([])
settings = AppSettings.load(SETTINGS_PATH)
cam_device, mic_device = find_camera(settings.camera), find_microphone(settings.microphone)
print("camera:", cam_device.description(), "| mic:", mic_device.description())

session = QMediaCaptureSession()
camera = QCamera(cam_device)
camera.setCameraFormat(best_format(cam_device))
session.setCamera(camera)
audio_input = QAudioInput(mic_device)  # 변수에 안 담으면 GC로 사라져 오디오가 빠진다
session.setAudioInput(audio_input)
recorder = QMediaRecorder()
session.setRecorder(recorder)

fmt = QMediaFormat(QMediaFormat.FileFormat.Matroska if out.endswith(".mkv") else QMediaFormat.FileFormat.MPEG4)
fmt.setVideoCodec(QMediaFormat.VideoCodec.H264)
fmt.setAudioCodec(QMediaFormat.AudioCodec.AAC)
recorder.setMediaFormat(fmt)
recorder.setVideoResolution(1920, 1080)
recorder.setVideoFrameRate(30)
recorder.setEncodingMode(QMediaRecorder.EncodingMode.AverageBitRateEncoding)
recorder.setVideoBitRate(25_000_000)
recorder.setAudioSampleRate(48000)
recorder.setAudioChannelCount(1)
recorder.setAudioBitRate(192_000)
recorder.setOutputLocation(QUrl.fromLocalFile(out))

recorder.errorOccurred.connect(lambda e, msg: print("ERROR:", e, msg, flush=True))
recorder.recorderStateChanged.connect(
    lambda s: (print("state:", s.name, flush=True), s == QMediaRecorder.RecorderState.StoppedState and app.quit())
)


def start() -> None:
    print("format:", recorder.mediaFormat().fileFormat().name, recorder.mediaFormat().videoCodec().name, flush=True)
    recorder.record()
    QTimer.singleShot(int(seconds * 1000), recorder.stop)


camera.start()
QTimer.singleShot(1500, start)  # 카메라 노출이 안정될 시간
app.exec()
print("saved:", recorder.actualLocation().toLocalFile())
