"""앱 설정 (선택한 카메라·마이크). 순수 파이썬 — Qt import 금지."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ks_studio.core.jsonio import write_json_atomic

MEDIA_COPY = "copy"  # 가져온 미디어를 프로젝트 폴더(media/)로 복사
MEDIA_REFERENCE = "reference"  # 원본 경로만 기록
RECENT_LIMIT = 8


@dataclass(frozen=True)
class DeviceRef:
    """OS 장치 ID + 사람이 읽는 이름. 카메라 ID는 USB 포트를 바꾸면 달라진다."""

    id: str
    name: str


def match_device(devices: list[DeviceRef], saved: DeviceRef) -> int | None:
    """저장된 장치의 인덱스. ID가 우선이고, 없으면 같은 이름(포트만 바뀐 같은 장치)."""
    for i, device in enumerate(devices):
        if device.id == saved.id:
            return i
    for i, device in enumerate(devices):
        if device.name == saved.name:
            return i
    return None


@dataclass(frozen=True)
class SyncCalibration:
    """카메라·마이크 조합의 고정 지연. 영상이 소리보다 늦은 만큼 녹화 마무리 때 오디오를 늦춘다.

    실측: JOYTRON 웹캠 0~40ms, 휴대폰 웹캠 약 105ms (사람은 소리가 45ms 넘게 먼저 나오면 알아챈다).
    """

    camera: DeviceRef
    microphone: DeviceRef
    audio_delay_ms: int  # +면 오디오를 늦춘다
    measured_at: str  # 측정한 날짜 (YYYY-MM-DD)


def find_calibration(
    calibrations: list[SyncCalibration], camera: DeviceRef | None, microphone: DeviceRef | None
) -> SyncCalibration | None:
    if camera is None or microphone is None:
        return None
    for calibration in calibrations:
        if match_device([camera], calibration.camera) == 0 and match_device([microphone], calibration.microphone) == 0:
            return calibration
    return None


def with_calibration(calibrations: list[SyncCalibration], new: SyncCalibration) -> list[SyncCalibration]:
    """같은 조합의 예전 값은 버리고 새 값으로."""
    old = find_calibration(calibrations, new.camera, new.microphone)
    return [c for c in calibrations if c is not old] + [new]


@dataclass
class AppSettings:
    camera: DeviceRef | None = None
    microphone: DeviceRef | None = None
    workspace_dir: str | None = None  # 지금 연 프로젝트 폴더. None이면 프로그램 안 기본 폴더
    fps: int = 30
    show_vertical_guide: bool = True
    media_import: str = MEDIA_COPY
    recent_projects: list[str] = field(default_factory=list)  # 최근에 연 프로젝트 폴더, 최신 순
    sync_calibrations: list[SyncCalibration] = field(default_factory=list)
    shortcuts: dict[str, list[str]] = field(default_factory=dict)  # 컷편집 단축키 중 바꾼 것만: 동작 이름 → 키 (빈 목록 = 없음)
    # 성능
    preview_quality: str = "1080"  # 재생 중 미리보기 최대 크기: PREVIEW_QUALITIES 키
    software_encoding: bool = False  # 내보내기·AI 보정·최적화를 CPU 인코더로만 (그래픽 드라이버 문제가 있을 때)
    # 새 프로젝트 기본값
    default_caption_style: str = ""  # 새 프로젝트의 자막 모양 프리셋 이름 ("" = 기본 모양)
    default_export_preset: str = ""  # 내보내기 탭을 처음 열 때 고를 프리셋 키 ("" = 직접 고르기)
    # 프로그램
    check_updates: bool = True  # 켤 때 새 버전이 있는지 GitHub에서 확인

    @property
    def sync_calibration(self) -> SyncCalibration | None:
        """지금 고른 카메라·마이크 조합의 보정값 (없으면 보정 필요)."""
        return find_calibration(self.sync_calibrations, self.camera, self.microphone)

    @classmethod
    def load(cls, path: Path) -> "AppSettings":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        data["camera"] = _device_ref(data.get("camera"))
        data["microphone"] = _device_ref(data.get("microphone"))
        data["sync_calibrations"] = [
            SyncCalibration(
                camera=DeviceRef(**c["camera"]),
                microphone=DeviceRef(**c["microphone"]),
                audio_delay_ms=c["audio_delay_ms"],
                measured_at=c["measured_at"],
            )
            for c in data.get("sync_calibrations", [])
        ]
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})  # 새 버전에서 만든 설정을 옛 버전이 열어도 된다

    def save(self, path: Path) -> None:
        write_json_atomic(path, asdict(self))


PREVIEW_QUALITIES = {  # 재생 중 미리보기 최대 크기 (멈춘 장면은 늘 화면 칸 크기 그대로)
    "1080": ("1080p (기본)", (1920, 1080)),
    "720": ("720p (가벼움 — 재생이 끊기면)", (1280, 720)),
    "540": ("540p (아주 가벼움)", (960, 540)),
    "full": ("제한 없음 (4K 모니터·빠른 PC)", (7680, 4320)),
}


def with_recent_project(recent: list[str], folder: str) -> list[str]:
    """folder를 맨 앞으로 (중복 제거, 최대 RECENT_LIMIT개)."""
    return [folder] + [p for p in recent if p != folder][: RECENT_LIMIT - 1]


def _device_ref(data: dict | None) -> DeviceRef | None:
    return DeviceRef(**data) if data else None
