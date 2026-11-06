"""자막: 영상 하나의 자막 줄(시작·끝·글) 목록과 모양. 순수 파이썬 — Qt·영상 처리 없음.

저장: 컷편집 project.json (타임라인 시각, 초). 영상 파일에 굽지 않고 들고 있다가 내보낼 때 입힌다.
외국인 대상 콘텐츠라 기본은 영어 자막 (한국어로 말한 것도 영어로 옮긴다).
"""

from dataclasses import asdict, dataclass, replace

MIN_CUE = 0.3  # 자막 한 줄의 가장 짧은 길이 (초)
NEW_CUE = 2.0  # 새로 넣는 자막 줄 길이
TASKS = {"translate": "영어로 (한국어는 번역)", "transcribe": "말한 그대로"}


@dataclass(frozen=True)
class Cue:
    start: float  # 초
    end: float
    text: str


@dataclass(frozen=True)
class CaptionStyle:
    size: float = 0.055  # 글자 크기 = 영상 짧은 변 × size
    bottom: float = 0.08  # 아래 여백 = 영상 높이 × bottom
    color: str = "#ffffff"
    outline: str = "#000000"


@dataclass(frozen=True)
class Subtitles:
    cues: tuple[Cue, ...] = ()
    style: CaptionStyle = CaptionStyle()
    task: str = "translate"  # 음성으로 만들 때: "translate"(영어로) / "transcribe"(말한 그대로)

    def at(self, t: float) -> int | None:
        """t초에 보이는 자막 줄 번호 (겹치면 나중에 시작한 줄)."""
        found = None
        for i, cue in enumerate(self.cues):
            if cue.start <= t < cue.end:
                found = i
        return found

    def to_dict(self) -> dict:
        return {"cues": [asdict(c) for c in self.cues], "style": asdict(self.style), "task": self.task}

    @classmethod
    def from_dict(cls, data: dict) -> "Subtitles":
        cues = tuple(Cue(float(c["start"]), float(c["end"]), str(c["text"])) for c in data.get("cues", []))
        style_data = {k: v for k, v in data.get("style", {}).items() if k in CaptionStyle.__dataclass_fields__}
        task = data.get("task", "translate")
        return cls(cues, CaptionStyle(**style_data), task if task in TASKS else "translate")


def _sorted(cues) -> tuple[Cue, ...]:
    return tuple(sorted(cues, key=lambda c: (c.start, c.end)))


def add_cue(subs: Subtitles, at: float, text: str = "") -> tuple[Subtitles, int]:
    """at초에 새 줄. 다음 줄이 가까우면 그 앞에서 끝낸다 (단 MIN_CUE보다 짧게는 안 한다). (새 자막, 새 줄 번호)."""
    at = max(0.0, at)
    following = [c.start for c in subs.cues if c.start > at]
    end = min([at + NEW_CUE, *following])
    cue = Cue(at, max(end, at + MIN_CUE), text)
    cues = _sorted((*subs.cues, cue))
    return replace(subs, cues=cues), cues.index(cue)


def update_cue(subs: Subtitles, index: int, start: float | None = None, end: float | None = None, text: str | None = None) -> tuple[Subtitles, int]:
    """한 줄을 고친다. 끝은 시작보다 MIN_CUE 이상 뒤. 시작이 바뀌면 순서를 다시 맞춘다. (새 자막, 그 줄의 새 번호)."""
    old = subs.cues[index]
    new_start = max(0.0, old.start if start is None else start)
    new_end = old.end if end is None else end
    cue = Cue(new_start, max(new_end, new_start + MIN_CUE), old.text if text is None else text)
    rest = subs.cues[:index] + subs.cues[index + 1 :]
    cues = _sorted((*rest, cue))
    return replace(subs, cues=cues), cues.index(cue)


def delete_cue(subs: Subtitles, index: int) -> Subtitles:
    return replace(subs, cues=subs.cues[:index] + subs.cues[index + 1 :])


def srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(subs: Subtitles) -> str:
    """SRT 파일 내용 (인스타그램·유튜브에 따로 올릴 때). 빈 줄은 뺀다."""
    blocks = [c for c in subs.cues if c.text.strip()]
    return "\n".join(f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{c.text.strip()}\n" for i, c in enumerate(blocks, 1))
