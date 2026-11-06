"""컷편집 프로젝트 모델. 순수 파이썬 — Qt import 금지.

- 시간은 전부 프레임 번호(정수). 원본이 정확한 CFR이라 프레임 단위로 딱 떨어지고 소수점 오차가 없다.
  · 타임라인 위치 = 타임라인 fps(30/60) 기준 프레임
  · 클립의 in/out = 원본 fps 기준 프레임 (원본 fps는 29.97처럼 소수일 수 있어 그대로 저장)
  · fps가 다르면 렌더할 때 변환한다: 60→30은 건너뛰고, 30→60은 같은 프레임을 두 번 쓴다. 오디오는 시간 기준이라 그대로
- 영상 줄: V1(메인)은 자석식 — 클립 사이에 빈 틈이 없고 순서만 가진다 (clips).
  V2 이상은 자유 배치 — 클립마다 시작 위치를 가지고 빈틈을 허용한다 (overlays). 같은 줄 안에서는 겹치지 않는다.
  화면에는 그 시각 가장 위 줄의 클립이 보인다 (V3 > V2 > V1). 아무것도 없으면 검은 화면.
  소리는 그 시각에 걸린 모든 줄의 클립을 섞는다. 줄마다 화면 숨김(hidden)과 소리 끄기(muted)는 따로 켜고 끈다.
- 클립마다 크기·위치·투명도(화면 속 화면)가 있다. 기본값이면 화면을 다 덮고, 아니면 아래 줄 위에 겹쳐 그린다.
- V1 디졸브(캡컷 방식): 뒤 클립의 transition_ms만큼 앞 클립 끝과 겹친다 → 전체 길이가 그만큼 짧아진다.
- 마커: 타임라인 프레임 번호 목록.
- 모든 객체는 불변(frozen). 편집은 새 Project를 만든다 (core.edits) → 되돌리기는 스냅샷만 쌓으면 된다.
"""

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ks_studio.core.jsonio import write_json_atomic

PROJECT_FPS = 30
TIMELINE_FPS_CHOICES = (30, 60)
FILE_VERSION = 8  # 2: 위 줄 · 3: 줄 음소거·숨김 · 4: 영상 줄 개수 · 5: 연동 · 6: 볼륨·페이드 · 7: 화면 속 화면·디졸브·마커 · 8: 뺀 녹화본
MAIN_TRACK = 1
DEFAULT_VIDEO_TRACKS = 3  # 새 프로젝트의 영상 줄 수 (V1~V3)
MIN_VIDEO_TRACKS = 2  # 이보다 적게는 지울 수 없다


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def timeline_fps_for(source_fps: float) -> int:
    """빈 타임라인에 처음 넣는 원본에 맞출 타임라인 fps (30 또는 60)."""
    return 60 if source_fps > 45 else 30


@dataclass(frozen=True)
class Clip:
    source: str  # Workspace.to_source 형식 (프로젝트 안이면 상대 경로)
    in_frame: int  # 원본에서 쓰는 구간 [in_frame, out_frame), 원본 fps 기준
    out_frame: int
    audio_offset_ms: int = 0  # +면 소리를 늦춘다
    id: str = field(default_factory=new_id)
    fps: float = PROJECT_FPS  # 원본 fps (이 필드가 없던 예전 파일은 30fps 녹화본뿐이었다)
    volume_db: float = 0.0  # 클립 소리 크기 (0 = 원본 그대로)
    fade_in_ms: int = 0  # 클립 시작에서 소리가 0 → 원래 크기로 커지는 시간 (fps와 상관없게 ms로)
    fade_out_ms: int = 0  # 클립 끝에서 소리가 작아지는 시간
    scale: float = 1.0  # 화면에 맞춘 크기 대비 배율
    pos_x: float = 0.5  # 클립 가운데의 화면 위치 (0 = 왼쪽 끝, 1 = 오른쪽 끝)
    pos_y: float = 0.5
    opacity: float = 1.0
    transition_ms: int = 0  # V1에서 앞 클립과 디졸브로 겹치는 시간 (이 클립이 앞 클립 위로 서서히 나타난다)

    @property
    def covers(self) -> bool:
        """화면을 다 덮는지 (기본 크기·위치·투명도). 덮으면 그 아래 줄은 그릴 필요가 없다."""
        return self.scale == 1.0 and self.pos_x == 0.5 and self.pos_y == 0.5 and self.opacity == 1.0


@dataclass(frozen=True)
class Placement:
    """V2 이상 줄에 놓인 클립. 자석식이 아니라 시작 위치를 직접 가진다."""

    clip: Clip
    track: int  # 2, 3, ...
    start: int  # 타임라인 프레임


@dataclass(frozen=True)
class Placed:
    """줄 구분 없이 본 '타임라인 위의 클립 하나'."""

    track: int
    clip: Clip
    start: int
    end: int


@dataclass(frozen=True)
class Layer:
    """바탕 위에 겹쳐 그리는 클립 하나. dissolve면 [dissolve_start, +dissolve_frames) 동안 서서히 나타난다."""

    clip: Clip
    clip_start: int
    dissolve_frames: int = 0


@dataclass(frozen=True)
class Segment:
    """화면 구성이 같은 구간 [start, end). clip = 화면을 다 덮는 바탕 (None이면 검은 화면), above = 그 위에 겹치는 클립들 (아래→위)."""

    start: int
    end: int
    clip: Clip | None
    clip_start: int  # 그 클립이 타임라인에서 시작하는 위치 (클립 안 위치 = t - clip_start)
    above: tuple[Layer, ...] = ()


@dataclass(frozen=True)
class Project:
    fps: int = PROJECT_FPS  # 타임라인(출력) fps
    media: tuple[str, ...] = ()  # 가져온 미디어. 녹화본은 recordings/에서 자동으로 보이므로 넣지 않는다
    clips: tuple[Clip, ...] = ()  # V1 (자석식)
    overlays: tuple[Placement, ...] = ()  # V2 이상
    muted: tuple[int, ...] = ()  # 소리를 끈 줄 번호 (화면은 그대로)
    hidden: tuple[int, ...] = ()  # 화면을 숨긴 줄 번호 (소리는 그대로)
    video_tracks: int = DEFAULT_VIDEO_TRACKS  # 영상 줄 개수 (V1 포함). 위 줄 클립은 항상 이 번호 이하에 있다
    linked: bool = True  # 연동: V1이 바뀌면 위 줄 클립이 아래 V1 장면을 따라간다 (core.edits.follow_main)
    markers: tuple[int, ...] = ()  # 타임라인 프레임
    removed: tuple[str, ...] = ()  # 워크스페이스 목록에서 뺀 녹화본 (파일은 그대로 — 원본이므로)

    def length(self, clip: Clip) -> int:
        """클립이 타임라인에서 차지하는 프레임 수 (최소 1)."""
        return max(1, round((clip.out_frame - clip.in_frame) * self.fps / clip.fps))

    def source_frame(self, clip: Clip, local: int) -> int:
        """클립 안 타임라인 위치 local에 보여줄 원본 프레임 번호 = 시각이 가장 가까운 프레임 (딱 중간이면 앞 프레임).

        내림을 쓰면 원본 fps가 타임라인보다 아주 조금 낮을 때(갤럭시 29.996fps) 항상 한 프레임 옛 장면이 나온다 (실측 -32ms).
        """
        return min(clip.out_frame - 1, clip.in_frame + math.floor(local * clip.fps / self.fps + 0.5 - 1e-6))

    def source_time(self, clip: Clip, local: float) -> float:
        """클립 안 타임라인 위치 local의 원본 시각(초)."""
        return clip.in_frame / clip.fps + local / self.fps

    def overlap_frames(self, previous_length: int, previous_overlap: int, clip: Clip) -> int:
        """V1 디졸브로 앞 클립과 겹치는 프레임 수. 두 클립을 다 먹거나, 앞 클립의 앞 디졸브와 겹치지 않게 줄인다."""
        if not clip.transition_ms:
            return 0
        wanted = round(clip.transition_ms * self.fps / 1000)
        return max(0, min(wanted, self.length(clip) - 1, previous_length - previous_overlap - 1))

    def main_placed(self) -> list[Placed]:
        """V1 클립 위치. 순서대로 이어 붙이되, 디졸브가 있으면 그만큼 앞 클립과 겹친다."""
        out: list[Placed] = []
        overlap = 0
        for clip in self.clips:
            length = self.length(clip)
            if out:
                previous = out[-1]
                overlap = self.overlap_frames(previous.end - previous.start, overlap, clip)
                start = previous.end - overlap
            else:
                start = overlap = 0
            out.append(Placed(MAIN_TRACK, clip, start, start + length))
        return out

    def main_overlaps(self) -> dict[str, tuple[int, int]]:
        """V1 클립 id → (앞 클립과 겹치는 프레임, 뒤 클립과 겹치는 프레임). 디졸브 동안 소리를 교차시킬 때 쓴다."""
        main = self.main_placed()
        result = {}
        for i, placed in enumerate(main):
            before = main[i - 1].end - placed.start if i > 0 else 0
            after = placed.end - main[i + 1].start if i + 1 < len(main) else 0
            result[placed.clip.id] = (max(0, before), max(0, after))
        return result

    @property
    def main_duration(self) -> int:
        main = self.main_placed()
        return main[-1].end if main else 0

    @property
    def duration(self) -> int:
        """V1 끝과 위 줄 클립 끝 중 늦은 쪽."""
        return max([self.main_duration, *(o.start + self.length(o.clip) for o in self.overlays)])

    def placed(self) -> list[Placed]:
        """모든 줄의 클립. V1은 순서대로 이어 붙인 위치를 계산해 넣는다."""
        out = self.main_placed()
        out += [Placed(o.track, o.clip, o.start, o.start + self.length(o.clip)) for o in self.overlays]
        return out

    def find(self, clip_id: str) -> Placed | None:
        return next((p for p in self.placed() if p.clip.id == clip_id), None)

    def fits(self, track: int, start: int, length: int, ignore: str | None = None) -> bool:
        """위 줄 track의 [start, start+length)가 비어 있는지 (ignore = 옮기는 중인 자기 자신)."""
        if not MAIN_TRACK < track <= self.video_tracks or start < 0:
            return False
        return all(
            o.clip.id == ignore or o.track != track or o.start + self.length(o.clip) <= start or start + length <= o.start
            for o in self.overlays
        )

    def visible_segments(self) -> list[Segment]:
        """타임라인을 '이 구간엔 이 클립이 보인다'는 겹치지 않는 구간으로 펼친다 (위 줄이 아래 줄을 덮는다).

        숨긴 줄은 없는 것처럼 건너뛰어 아래 줄이 보인다. 길이는 숨긴 줄까지 포함한 전체 길이 그대로.
        화면을 다 덮지 않는 클립(작게·옆으로·반투명, 디졸브로 나타나는 중)은 above로, 그 아래 덮는 클립이 바탕이 된다.
        """
        overlaps = self.main_overlaps()
        visible = [p for p in self.placed() if p.track not in self.hidden]
        bounds = {0, self.duration, *(p.start for p in visible), *(p.end for p in visible)}
        bounds |= {p.start + overlaps[p.clip.id][0] for p in visible if p.track == MAIN_TRACK}  # 디졸브가 끝나는 곳
        bounds = sorted(bounds)
        segments: list[Segment] = []
        for lo, hi in zip(bounds, bounds[1:]):
            # 아래→위: 줄 번호, 같은 V1이면 뒤 클립(디졸브로 나타나는 쪽)이 위
            here = sorted((p for p in visible if p.start <= lo < p.end), key=lambda p: (p.track, p.start))
            layers = []
            for p in here:
                fading = overlaps[p.clip.id][0] if p.track == MAIN_TRACK and lo < p.start + overlaps[p.clip.id][0] else 0
                layers.append((p, fading))
            base_index = max((i for i, (p, fading) in enumerate(layers) if p.clip.covers and not fading), default=-1)
            base = layers[base_index][0] if base_index >= 0 else None
            above = tuple(Layer(p.clip, p.start, fading) for p, fading in layers[base_index + 1 :])
            clip, clip_start = (base.clip, base.start) if base else (None, lo)
            last = segments[-1] if segments else None
            if last and last.clip == clip and last.above == above and (clip is None or last.clip_start == clip_start):
                segments[-1] = Segment(last.start, hi, clip, last.clip_start, above)
            else:
                segments.append(Segment(lo, hi, clip, clip_start, above))
        return segments

    def index_of(self, clip_id: str) -> int:
        return next(i for i, c in enumerate(self.clips) if c.id == clip_id)

    def locate(self, frame: int) -> tuple[int, int] | None:
        """타임라인 프레임 → (V1 클립 번호, 클립 안 위치). 디졸브로 겹친 곳은 앞 클립. V1 끝 이후면 None."""
        for i, placed in enumerate(self.main_placed()):
            if placed.start <= frame < placed.end:
                return i, frame - placed.start
        return None

    def save(self, path: Path) -> None:
        data = asdict(self)
        write_json_atomic(path, {"version": FILE_VERSION, **data})

    @classmethod
    def load(cls, path: Path) -> "Project":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            fps=data["fps"],
            media=tuple(data["media"]),
            clips=tuple(Clip(**c) for c in data["clips"]),
            overlays=tuple(
                Placement(Clip(**o["clip"]), o["track"], o["start"]) for o in data.get("overlays", [])  # 버전 1엔 없다
            ),
            muted=tuple(data.get("muted", [])),  # 버전 2까지는 없다
            hidden=tuple(data.get("hidden", [])),
            # 버전 3까지는 3줄 고정이었다
            video_tracks=data.get("video_tracks", max([DEFAULT_VIDEO_TRACKS, *(o["track"] for o in data.get("overlays", []))])),
            linked=data.get("linked", True),  # 버전 4까지는 없다
            markers=tuple(data.get("markers", [])),  # 버전 6까지는 없다
            removed=tuple(data.get("removed", [])),  # 버전 7까지는 없다
        )
