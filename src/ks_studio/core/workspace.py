"""프로젝트 폴더 구조. 순수 파이썬 — Qt import 금지.

프로젝트 = 폴더 하나. 폴더를 옮기면 프로젝트가 통째로 따라간다 (참조로 가져온 미디어는 예외).

<root>/project.json                        컷편집 결정 (클립 목록, 가져온 미디어)
<root>/recordings/20261103_142530.mp4      완성된 녹화본
<root>/recordings/.partial/<이름>.mkv       녹화 중 / 비정상 종료로 남은 파일
<root>/media/                              복사해서 가져온 미디어
<root>/cache/                              파형·썸네일 캐시 (지워도 다시 만든다)
<root>/exports/                            컷편집에서 내보낸 영상 (자막·내보내기 탭의 재료)
<root>/exports/final/                      내보내기 탭의 최종 영상
<root>/ai/<이름>_보정.mp4                    AI 보정본 (워크스페이스에 녹화본처럼 보인다)
<root>/ai/settings.json                    AI 보정 설정 (원본마다)
<root>/subtitles/subtitles.json            자막 (영상마다)
<root>/.test/test.mkv, test.mp4            5초 테스트 녹화 (매번 덮어씀)
<root>/.test/sync_calibration.mp4          박수 싱크 보정용 녹화 (매번 덮어씀)
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def recording_name(now: datetime) -> str:
    return now.strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def project_file(self) -> Path:
        return self.root / "project.json"

    @property
    def recordings_dir(self) -> Path:
        return self.root / "recordings"

    @property
    def partial_dir(self) -> Path:
        return self.recordings_dir / ".partial"

    @property
    def media_dir(self) -> Path:
        return self.root / "media"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def ai_dir(self) -> Path:
        return self.root / "ai"

    @property
    def ai_settings_file(self) -> Path:
        return self.ai_dir / "settings.json"

    def beautified(self) -> list[Path]:
        """AI 보정본, 최신 순."""
        return sorted(self.ai_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)

    def ai_output(self, source: str) -> Path:
        """원본의 AI 보정본 자리."""
        return self.ai_dir / f"{Path(source).stem}_보정.mp4"

    @property
    def subtitles_file(self) -> Path:
        return self.root / "subtitles" / "subtitles.json"

    @property
    def final_dir(self) -> Path:
        return self.exports_dir / "final"

    def cut_exports(self) -> list[Path]:
        """컷편집에서 내보낸 영상, 최신 순."""
        return sorted(self.exports_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)

    def videos(self, extra: tuple[str, ...] = ()) -> list[str]:
        """자막·내보내기 탭에서 고를 수 있는 영상: 녹화본 · 컷편집 결과 · AI 보정본 · 컷편집에 가져온 미디어 · 탭에서 가져온 것."""
        from ks_studio.core.project import Project  # project.json 읽기 (순환 import 없음)

        project = Project.load(self.project_file)
        found = [self.to_source(p) for p in (*self.recordings(), *self.cut_exports(), *self.beautified())]
        found += list(project.media) + list(extra)
        return [s for s in dict.fromkeys(found) if s not in project.removed]

    @property
    def test_partial(self) -> Path:
        return self.root / ".test" / "test.mkv"

    @property
    def test_output(self) -> Path:
        return self.root / ".test" / "test.mp4"

    @property
    def calibration_partial(self) -> Path:
        return self.root / ".test" / "sync_calibration.mkv"

    @property
    def calibration_output(self) -> Path:
        return self.root / ".test" / "sync_calibration.mp4"

    def partial_path(self, name: str) -> Path:
        return self.partial_dir / f"{name}.mkv"

    def final_path(self, name: str) -> Path:
        return self.recordings_dir / f"{name}.mp4"

    def recordings(self) -> list[Path]:
        """완성된 녹화본, 최신 순."""
        return sorted(self.recordings_dir.glob("*.mp4"), reverse=True)

    def leftover_partials(self) -> list[Path]:
        """마무리되지 못한 녹화 (앱 시작 시점에 남아 있으면 비정상 종료의 흔적)."""
        return sorted(self.partial_dir.glob("*.mkv"))

    def to_source(self, path: Path) -> str:
        """project.json에 적을 경로: 프로젝트 안이면 상대 경로, 밖이면 절대 경로."""
        path = path.resolve()
        try:
            return path.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)

    def resolve(self, source: str) -> Path:
        path = Path(source)
        return path if path.is_absolute() else self.root / path
