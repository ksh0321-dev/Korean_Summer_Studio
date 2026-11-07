"""자막: 영상 하나의 자막 줄(시작·끝·글·번역·단어 시간) 목록과 모양. 순수 파이썬 — Qt·영상 처리 없음.

저장: 컷편집 project.json (타임라인 시각, 초). 영상 파일에 굽지 않고 들고 있다가 내보낼 때 입힌다.
외국인 대상 한국어 교육 콘텐츠라 기본은 두 줄 (한국어 + 영어 번역).
"""

from dataclasses import asdict, dataclass, field, replace

MIN_CUE = 0.3  # 자막 한 줄의 가장 짧은 길이 (초)
NEW_CUE = 2.0  # 새로 넣는 자막 줄 길이
TASKS = {"bilingual": "한국어 + 영어 (두 줄)", "translate": "영어로 (한국어는 번역)", "transcribe": "말한 그대로"}
PAUSE = 0.35  # 단어 사이가 이만큼 비면 말이 끊긴 곳 — 자막 줄을 여기서 나눈다
SHORT_CUE = 1.0  # 이보다 짧은 줄은 (글자 수가 넘치지 않으면) 이웃 줄과 합친다 — 너무 빨리 지나가 못 읽는다
MERGED_MAX = 4.0  # 합친 줄은 이 길이(초)까지만 — 짧은 말이 이어지면 문장 네 개가 7초 동안 한 화면에 있었다 (실측)
LONG_CUE = 5.0  # 이보다 긴 줄은 단어 사이가 가장 많이 빈 곳에서 나눈다 — 용어집 힌트를 주면 문장부호가 빠져 9.5초짜리 줄이 나왔다
HOLD = 0.3  # 마지막 단어 뒤로 자막을 이만큼 더 둔다 (다음 줄 시작 전까지만)
SENTENCE_END = ".?!。？！"
BREAKS = SENTENCE_END + ",、，"


@dataclass(frozen=True)
class Word:
    start: float  # 초 (타임라인)
    end: float
    text: str  # 앞 띄어쓰기 포함 (Whisper 그대로 — 이어 붙이면 문장)


@dataclass(frozen=True)
class Cue:
    start: float  # 초
    end: float
    text: str  # 첫째 줄 (말한 말 — 두 줄 자막이면 한국어)
    translation: str = ""  # 둘째 줄 (영어 번역, 두 줄 자막일 때)
    words: tuple[Word, ...] = field(default=(), compare=False)  # 첫째 줄의 단어별 시간 (글을 고치면 버린다)


@dataclass(frozen=True)
class CaptionStyle:
    size: float = 0.055  # 글자 크기 = 영상 짧은 변 × size
    bottom: float = 0.08  # 아래 여백 = 영상 높이 × bottom (미리보기에서 자막을 끌어 바꾼다)
    color: str = "#ffffff"
    outline: str = "#000000"
    second_size: float = 0.75  # 둘째 줄(번역) 글자 크기 = 첫째 줄 × second_size
    second_color: str = "#ffe066"  # 둘째 줄 색 (첫째 줄과 구별되게)
    font: str = ""  # 글꼴 이름 (빈 값 = 앱 글꼴 Pretendard — 같이 배포하는 무료 글꼴)
    outline_width: float = 0.16  # 테두리 두께 = 글자 크기 × outline_width (0 = 테두리 없음)
    box: bool = False  # 글 뒤 배경 상자 (밝은 화면에서도 읽히게)
    box_color: str = "#000000"
    box_opacity: float = 0.55
    shadow: float = 0.0  # 그림자 진하기 0~1 (0 = 없음) — 테두리 없이도 밝은 화면에서 읽히게
    bold: bool = True
    italic: bool = False
    align: str = "center"  # 여러 줄일 때 줄 맞춤: ALIGNS의 키

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CaptionStyle":
        """모르는 칸은 버리고, 타입이 다른 값(손으로 고친 파일)은 기본값으로."""
        values = {}
        for name, f in cls.__dataclass_fields__.items():
            value, default = data.get(name), f.default
            if isinstance(default, bool):
                if isinstance(value, bool):
                    values[name] = value
            elif isinstance(default, float):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values[name] = float(value)
            elif isinstance(value, str):
                values[name] = value
        return cls(**values)


ALIGNS = {"left": "왼쪽", "center": "가운데", "right": "오른쪽"}
_TITLE = CaptionStyle(size=0.08, bottom=0.0)  # 화면 글자 기본 (project.TextItem과 같은 크기)
STYLE_TEMPLATES = {  # 기본 제공 모양 (글자 탭 프리셋 목록 맨 위) — 이름 → 모양
    "유튜브 제목 (노랑 · 굵은 테두리)": replace(_TITLE, size=0.1, color="#ffe000", outline_width=0.24, shadow=0.6),
    "강조 상자 (빨강 상자 · 흰 글자)": replace(_TITLE, size=0.07, outline_width=0.0, box=True, box_color="#e5322d", box_opacity=1.0),
    "깔끔한 흰 글자 (그림자)": replace(_TITLE, size=0.06, outline_width=0.0, shadow=0.8, bold=False),
    "반투명 띠 (검은 상자)": replace(_TITLE, size=0.05, outline_width=0.0, box=True, box_opacity=0.7),
    "형광 (하늘색 · 파란 테두리)": replace(_TITLE, size=0.08, color="#7df9ff", outline="#0040ff", outline_width=0.12, shadow=0.5),
}


@dataclass(frozen=True)
class Subtitles:
    cues: tuple[Cue, ...] = ()
    style: CaptionStyle = CaptionStyle()
    task: str = "bilingual"  # 음성으로 만들 때: TASKS의 키
    line_chars: int = 16  # 자동 줄 나누기: 한 줄에 한글 이만큼 (영어는 글자 폭으로 환산)
    max_lines: int = 2  # 자동 줄 나누기: 자막 한 번에 첫째 줄이 이 줄 수까지

    def at(self, t: float) -> int | None:
        """t초에 보이는 자막 줄 번호 (겹치면 나중에 시작한 줄)."""
        found = None
        for i, cue in enumerate(self.cues):
            if cue.start <= t < cue.end:
                found = i
        return found

    def to_dict(self) -> dict:
        return asdict(self)  # project.json은 Project 전체를 asdict로 쓴다 — 같은 모양

    @classmethod
    def from_dict(cls, data: dict) -> "Subtitles":
        def word(w) -> Word:
            return Word(float(w["start"]), float(w["end"]), str(w["text"]))

        cues = tuple(
            Cue(float(c["start"]), float(c["end"]), str(c["text"]), str(c.get("translation", "")),
                tuple(word(w) for w in c.get("words", ())))
            for c in data.get("cues", [])
        )
        task = data.get("task", cls.task)  # 자막을 저장한 예전 파일엔 늘 task가 있다 (없으면 자막이 없던 파일)
        return cls(cues, CaptionStyle.from_dict(data.get("style", {})), task if task in TASKS else cls.task,
                   int(data.get("line_chars", 16)), int(data.get("max_lines", 2)))


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


def update_cue(subs: Subtitles, index: int, start: float | None = None, end: float | None = None, text: str | None = None,
               translation: str | None = None) -> tuple[Subtitles, int]:
    """한 줄을 고친다. 끝은 시작보다 MIN_CUE 이상 뒤. 시작이 바뀌면 순서를 다시 맞춘다. (새 자막, 그 줄의 새 번호).

    첫째 줄 글을 바꾸면 단어별 시간은 버린다 (더는 글과 맞지 않는다).
    """
    old = subs.cues[index]
    new_start = max(0.0, old.start if start is None else start)
    new_end = old.end if end is None else end
    changed_text = text is not None and text != old.text
    cue = Cue(new_start, max(new_end, new_start + MIN_CUE), text if changed_text else old.text,
              old.translation if translation is None else translation, () if changed_text else old.words)
    rest = subs.cues[:index] + subs.cues[index + 1 :]
    cues = _sorted((*rest, cue))
    return replace(subs, cues=cues), cues.index(cue)


def delete_cue(subs: Subtitles, index: int) -> Subtitles:
    return replace(subs, cues=subs.cues[:index] + subs.cues[index + 1 :])


def move_cue(subs: Subtitles, index: int, delta: float) -> tuple[Subtitles, int]:
    """한 줄을 통째로 delta초 옮긴다 (단어 시간도 같이, 0초 앞으로는 못 간다). (새 자막, 그 줄의 새 번호)."""
    cue = subs.cues[index]
    delta = max(delta, -cue.start)
    moved = map_times(cue, lambda t: t + delta)
    cues = _sorted((*subs.cues[:index], *subs.cues[index + 1 :], moved))
    return replace(subs, cues=cues), cues.index(moved)


def _cut_text(text: str, ratio: float) -> tuple[str, str]:
    """글을 ratio(0~1) 자리에서 가까운 띄어쓰기로 둘로 (띄어쓰기가 없으면 그 글자에서)."""
    text = text.strip()
    if not text:
        return "", ""
    at = round(len(text) * ratio)
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if spaces:
        at = min(spaces, key=lambda i: abs(i - at))
    return text[:at].strip(), text[at:].strip()


def split_cue(subs: Subtitles, index: int, at: float) -> tuple[Subtitles, int] | None:
    """at초에서 한 줄을 둘로 (두 줄 모두 MIN_CUE 이상이어야). 단어 시간이 있으면 단어 단위로, 없으면 시간 비율로 글을 나눈다.

    번역은 글자 비율로 나눈다 — 초벌이라 고쳐 써야 할 수 있다. (새 자막, 뒤 줄 번호) 또는 나눌 수 없으면 None.
    """
    cue = subs.cues[index]
    if not (cue.start + MIN_CUE <= at <= cue.end - MIN_CUE):
        return None
    ratio = (at - cue.start) / (cue.end - cue.start)
    before = tuple(w for w in cue.words if (w.start + w.end) / 2 < at)
    after = cue.words[len(before) :]
    if before and after:
        first_text, second_text = "".join(w.text for w in before).strip(), "".join(w.text for w in after).strip()
        ratio = len(first_text) / max(1, len(first_text) + len(second_text))
    else:
        first_text, second_text = _cut_text(cue.text, ratio)
        before = after = ()
    first_tr, second_tr = _cut_text(cue.translation, ratio)
    first = Cue(cue.start, round(at, 3), first_text, first_tr, before)
    second = Cue(round(at, 3), cue.end, second_text, second_tr, after)
    cues = _sorted((*subs.cues[:index], *subs.cues[index + 1 :], first, second))
    return replace(subs, cues=cues), cues.index(second)


def merge_cues(subs: Subtitles, index: int) -> tuple[Subtitles, int] | None:
    """index 줄과 바로 다음 줄을 하나로 (글·번역·단어를 잇는다). (새 자막, 합친 줄 번호) 또는 다음 줄이 없으면 None."""
    if not 0 <= index < len(subs.cues) - 1:
        return None
    a, b = subs.cues[index], subs.cues[index + 1]
    join = lambda x, y: " ".join(part for part in (x.strip(), y.strip()) if part)  # noqa: E731
    words = a.words + b.words if a.words and b.words else ()
    merged = Cue(a.start, max(a.end, b.end), join(a.text, b.text), join(a.translation, b.translation), words)
    cues = (*subs.cues[:index], merged, *subs.cues[index + 2 :])
    return replace(subs, cues=cues), index


FIND_IN = {"both": "자막·번역", "text": "자막만", "translation": "번역만"}


def replace_text(subs: Subtitles, find: str, new: str, where: str = "both", case: bool = False) -> tuple[Subtitles, int]:
    """모든 줄에서 find를 new로 바꾼다. (새 자막, 바꾼 곳 수).

    단어 하나 안에서 바뀌면 단어 시간도 고쳐 둔다 (강조에 쓴다). 여러 단어에 걸치면 단어 시간은 버린다.
    """
    if not find:
        return subs, 0
    import re

    pattern = re.compile(re.escape(find), 0 if case else re.IGNORECASE)
    count = 0
    cues = []
    for cue in subs.cues:
        text, translation, words = cue.text, cue.translation, cue.words
        if where in ("both", "text"):
            text, n = pattern.subn(lambda _m: new, cue.text)
            count += n
            if n:
                fixed = tuple(Word(w.start, w.end, pattern.sub(lambda _m: new, w.text)) for w in words)
                words = fixed if "".join(w.text for w in fixed).strip() == text.strip() else ()
        if where in ("both", "translation"):
            translation, n = pattern.subn(lambda _m: new, cue.translation)
            count += n
        cues.append(Cue(cue.start, cue.end, text, translation, words))
    return replace(subs, cues=tuple(cues)), count


def srt_time(seconds: float) -> str:
    ms = max(0, round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(subs: Subtitles) -> str:
    """SRT 파일 내용 (인스타그램·유튜브에 따로 올릴 때). 빈 줄은 뺀다. 번역이 있으면 둘째 줄로."""
    blocks = [c for c in subs.cues if c.text.strip() or c.translation.strip()]

    def body(c: Cue) -> str:
        return "\n".join(line for line in (c.text.strip(), c.translation.strip()) if line)

    return "\n".join(f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{body(c)}\n" for i, c in enumerate(blocks, 1))


def map_times(cue: Cue, move) -> Cue | None:
    """편집으로 시각이 옮겨질 때 (move: 옛 시각 → 새 시각). 단어도 같이 옮기고, 없어진 단어는 뺀다. 줄이 너무 짧아지면 None."""
    start, end = move(cue.start), move(cue.end)
    if end - start < MIN_CUE:
        return None
    words = tuple(Word(round(move(w.start), 3), round(move(w.end), 3), w.text) for w in cue.words)
    return Cue(round(start, 3), round(end, 3), cue.text, cue.translation, tuple(w for w in words if w.end > w.start))


# ── 자동 줄 나누기 ───────────────────────────────────────────────


def text_width(text: str) -> float:
    """글 폭 (한글 한 자 = 1). 영문·숫자는 약 0.55, 띄어쓰기·문장부호는 0.3 — 자막 굵은 글꼴 기준 대략."""
    width = 0.0
    for ch in text:
        if ch.isspace() or ch in BREAKS or ch in "'\"()-":
            width += 0.3
        elif ord(ch) < 0x1100:  # 라틴 글자·숫자
            width += 0.55
        else:  # 한글·한자·가나
            width += 1.0
    return width


def _cue(words: list[Word], next_start: float | None) -> Cue:
    end = words[-1].end + HOLD
    if next_start is not None:
        end = min(end, next_start)
    end = max(end, words[0].start + MIN_CUE)
    return Cue(round(words[0].start, 3), round(end, 3), "".join(w.text for w in words).strip(), "", tuple(words))


def _split_long(words: list[Word]) -> list[list[Word]]:
    """LONG_CUE보다 긴 덩어리를 단어 사이가 가장 많이 빈 곳에서 (짧아질 때까지) 둘로."""
    if len(words) < 2 or words[-1].end - words[0].start <= LONG_CUE:
        return [words]
    cut = max(range(1, len(words)), key=lambda i: words[i].start - words[i - 1].end)
    return _split_long(words[:cut]) + _split_long(words[cut:])


def split_words(words: list[Word], line_chars: int = 16, max_lines: int = 2) -> list[Cue]:
    """단어별 시간 → 짧은 자막 줄들. 숏폼 세로 화면에서 1~2줄이 되게.

    1. 말이 끊긴 곳(PAUSE)과 문장 끝(.?!)에서 나눈다 — 뜻이 이어지는 덩어리
    2. 그래도 길면(한 줄 line_chars × max_lines 넘게) 단어 경계에서 나눈다 — 쉼표 뒤를 먼저, 없으면 가득 찬 곳.
       LONG_CUE초보다 오래 보이는 덩어리도 단어 사이가 가장 많이 빈 곳에서 나눈다
    3. 너무 짧은 덩어리(SHORT_CUE 미만)는 글자 수가 넘치지 않으면 이웃과 합친다
    """
    limit = line_chars * max_lines
    groups: list[list[Word]] = []
    current: list[Word] = []
    for i, word in enumerate(words):
        current.append(word)
        gap = words[i + 1].start - word.end if i + 1 < len(words) else None
        if gap is None or gap >= PAUSE or word.text.strip()[-1:] in SENTENCE_END:
            groups.append(current)
            current = []
    pieces: list[list[Word]] = []
    for group in groups:
        while text_width("".join(w.text for w in group).strip()) > limit and len(group) > 1:
            fill, comma, width = 1, None, 0.0
            for j, word in enumerate(group[:-1]):  # 들어가는 데까지, 그 안에 쉼표 뒤가 있으면 거기 (너무 앞쪽만 아니면)
                width += text_width(word.text)
                if width > limit:
                    break
                fill = j + 1
                if word.text.strip()[-1:] in BREAKS and width >= limit * 0.4:
                    comma = j + 1
            cut = comma or fill
            pieces.append(group[:cut])
            group = group[cut:]
        pieces.append(group)
    pieces = [part for piece in pieces for part in _split_long(piece)]
    merged: list[list[Word]] = []
    for piece in pieces:
        if merged:
            last = merged[-1]
            short = piece[-1].end - piece[0].start < SHORT_CUE or last[-1].end - last[0].start < SHORT_CUE
            close = piece[0].start - last[-1].end < PAUSE * 2
            fits = text_width("".join(w.text for w in last + piece).strip()) <= limit and piece[-1].end - last[0].start <= MERGED_MAX
            if short and close and fits:
                merged[-1] = last + piece
                continue
        merged.append(piece)
    return [_cue(piece, merged[i + 1][0].start if i + 1 < len(merged) else None) for i, piece in enumerate(merged)]


# ── 모양 프리셋 (user/caption_styles/<이름>.json, 앱 전체 공통) ────────────


def save_style(folder, name: str, style: CaptionStyle) -> None:
    from ks_studio.core.jsonio import write_json_atomic

    write_json_atomic(folder / f"{name.strip()}.json", {"version": 1, "style": style.to_dict()})


def load_style(folder, name: str) -> CaptionStyle:
    import json

    return CaptionStyle.from_dict(json.loads((folder / f"{name}.json").read_text(encoding="utf-8")).get("style", {}))
