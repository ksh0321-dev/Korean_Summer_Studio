"""카메라 미리보기와 마이크 입력. 녹화 엔진(recording.py)이 같은 프레임·샘플을 받아 쓴다."""

import logging
import math
import time
from collections import deque

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtMultimedia import (
    QAudioDevice,
    QAudioFormat,
    QAudioSource,
    QCamera,
    QCameraDevice,
    QCameraFormat,
    QMediaCaptureSession,
    QVideoSink,
)

log = logging.getLogger(__name__)

INPUT_BUFFER_S = 5  # 메인 스레드가 이만큼 멈춰도 마이크 샘플을 잃지 않게
GAP_S = 0.2  # 도착이 이보다 더 밀리면 샘플을 잃은 것으로 본다 (정상 묶음 지연보다 충분히 크게)


class CameraPreview(QObject):
    error = Signal(str)

    def __init__(self, sink: QVideoSink, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._session = QMediaCaptureSession(self)
        self._session.setVideoSink(sink)
        self._camera: QCamera | None = None

    def start(self, device: QCameraDevice, fmt: QCameraFormat | None) -> None:
        self.stop()
        self._camera = QCamera(device, self)
        if fmt is not None:
            self._camera.setCameraFormat(fmt)
        self._camera.errorOccurred.connect(lambda _error, message: self.error.emit(message))
        self._session.setCamera(self._camera)
        self._camera.start()

    def stop(self) -> None:
        if self._camera is None:
            return
        self._camera.stop()
        self._session.setCamera(None)
        self._camera.deleteLater()
        self._camera = None


class MicInput(QObject):
    """마이크 입력.

    level   = 최근 묶음의 피크 (0.0~1.0, 1.0 = 클리핑)
    samples = (float32 배열 [샘플 수, 채널 수], 첫 샘플 번호). 번호는 실제 시간을 따라간다:
              샘플을 잃으면 그만큼 번호를 건너뛰고, 녹화는 그 구간을 무음으로 채운다 → 이후 싱크 유지.
    clock_origin = 샘플 0번이 들어온 시각(time.monotonic 기준) 추정값.

    실측: 메인 스레드가 2.5초 멈추면 Qt 기본 입력 버퍼(약 0.45초)가 넘쳐 소리 2.15초가 사라졌다.
    """

    level = Signal(float)
    samples = Signal(object, int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._source: QAudioSource | None = None
        self._io = None
        self._received = 0
        # 마이크 시계와 PC 시계는 미세하게 속도가 달라서, 최근 묶음들만으로 기준점을 추정한다
        self._recent_origins: deque[float] = deque(maxlen=200)
        self.sample_rate = 0
        self.channels = 0

    @property
    def clock_origin(self) -> float:
        # 도착은 늦어질 수만 있으므로, (도착 시각 - 받은 길이)의 최솟값이 실제에 가장 가깝다
        return min(self._recent_origins, default=math.inf)

    def start(self, device: QAudioDevice) -> None:
        self.stop()
        fmt = device.preferredFormat()
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
        self.sample_rate, self.channels = fmt.sampleRate(), fmt.channelCount()
        self._received = 0
        self._recent_origins.clear()
        self._source = QAudioSource(device, fmt, self)
        self._source.setBufferSize(int(self.sample_rate * INPUT_BUFFER_S) * 4 * self.channels)
        self._io = self._source.start()
        self._io.readyRead.connect(self._read)

    def stop(self) -> None:
        if self._source is None:
            return
        self._source.stop()
        self._source.deleteLater()
        self._source = None
        self._io = None

    def _read(self) -> None:
        arrival = time.monotonic()
        data = self._io.readAll().data()
        frame_bytes = 4 * self.channels
        chunk = np.frombuffer(data[: len(data) // frame_bytes * frame_bytes], np.float32).reshape(-1, self.channels)
        if not len(chunk):
            return
        # (도착 시각 - 받은 길이)는 정상이면 기준점 이상이다. 기준점보다 GAP_S 넘게 늦으면 샘플을 잃은 것
        missing = (arrival - self.clock_origin) * self.sample_rate - (self._received + len(chunk))
        if missing > GAP_S * self.sample_rate:
            log.warning("마이크 샘플 %.2f초 분량을 잃음 (입력 버퍼 넘침) → 그 구간은 무음으로 채움", missing / self.sample_rate)
            self._received += int(missing)
        first = self._received
        self._received += len(chunk)
        self._recent_origins.append(arrival - self._received / self.sample_rate)
        self.level.emit(float(np.abs(chunk).max()))
        self.samples.emit(chunk, first)
