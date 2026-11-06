"""OS가 인식한 카메라·마이크 목록, 연결/해제 감지, 저장된 장치 찾기."""

from PySide6.QtCore import QObject, Signal
from PySide6.QtMultimedia import QAudioDevice, QCameraDevice, QCameraFormat, QMediaDevices, QVideoFrameFormat

from ks_studio.core.settings import DeviceRef, match_device


def device_ref(device: QCameraDevice | QAudioDevice) -> DeviceRef:
    return DeviceRef(id=bytes(device.id()).decode(), name=device.description())


def camera_refs() -> list[DeviceRef]:
    return [device_ref(d) for d in QMediaDevices.videoInputs()]


def microphone_refs() -> list[DeviceRef]:
    return [device_ref(d) for d in QMediaDevices.audioInputs()]


def find_camera(ref: DeviceRef | None) -> QCameraDevice | None:
    return _find(QMediaDevices.videoInputs(), ref)


def find_microphone(ref: DeviceRef | None) -> QAudioDevice | None:
    return _find(QMediaDevices.audioInputs(), ref)


def _find(devices: list, ref: DeviceRef | None):
    if ref is None:
        return None
    index = match_device([device_ref(d) for d in devices], ref)
    return None if index is None else devices[index]


RECORD_FPS = (30, 60)


def _formats_1080p(device: QCameraDevice) -> list[QCameraFormat]:
    return [f for f in device.videoFormats() if f.resolution().width() == 1920 and f.resolution().height() == 1080]


def supported_fps(device: QCameraDevice) -> list[int]:
    """녹화용 fps(30/60) 중 이 카메라가 1080p로 낼 수 있는 값."""
    rates = {round(f.maxFrameRate()) for f in _formats_1080p(device)}
    return [fps for fps in RECORD_FPS if fps in rates]


def best_format(device: QCameraDevice, fps: int | None = None) -> QCameraFormat | None:
    """1080p 포맷 중 fps가 맞는(없으면 가장 높은) 것. fps가 같으면 NV12 우선 (Qt가 JPEG을 풀 필요 없음)."""
    candidates = [f for f in _formats_1080p(device) if fps is None or round(f.maxFrameRate()) == fps]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda f: (f.maxFrameRate(), f.pixelFormat() == QVideoFrameFormat.PixelFormat.Format_NV12),
    )


class DeviceMonitor(QObject):
    """장치가 꽂히거나 빠지면 신호를 보낸다."""

    cameras_changed = Signal()
    microphones_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._media = QMediaDevices(self)
        self._media.videoInputsChanged.connect(self.cameras_changed)
        self._media.audioInputsChanged.connect(self.microphones_changed)
