"""하이라이트 분할: 점수 곡선 → 하이라이트 구간 → V1 클립을 자를 자리. 순수 계산 — Qt·파일 접근 없음.

1) find_highlights: 점수가 기준을 넘는 구간 (들어갈 때 기준은 높게, 나올 때는 조금 낮게 — 경계가 들쭉날쭉하지 않게).
   2초 안쪽으로 붙은 것은 합치고, 너무 짧은 것은 버린다. 기준은 그 영상 전체 점수에서 정한다 (민감도로 조절).
2) cut_points: 하이라이트 바깥으로 약 1초 떨어진 곳에서 자른다. ±0.5초 안에서 가장 조용한 곳(말이 끊긴 곳)에 맞춘다.
   예: 하이라이트 6~15초 → 5초·16초 언저리에서 잘라 [0~5] [5~16] [16~20].
3) split_project: 그 자리를 타임라인 프레임으로 바꿔 V1 클립을 나눈다 (위 줄은 그대로). 한 번의 편집 (Ctrl+Z 한 번).

시각은 모두 원본 영상 기준(초) — 클립을 옮기거나 다듬어도 점수는 그 장면을 따라간다.
"""

import math
from dataclasses import dataclass

import numpy as np

from ks_studio.core.edits import split
from ks_studio.core.highlight import AUDIO_STEP, BIN, Scores, Signals, score
from ks_studio.core.project import MAIN_TRACK, Clip, Project

DEFAULT_SENSITIVITY = 50  # 0 = 아주 뚜렷한 곳만, 100 = 조금만 두드러져도
DEFAULT_PAD = 1.0  # 하이라이트 앞뒤로 남길 여유 (초)
DEFAULT_MIN_LENGTH = 3.0  # 이보다 짧은 하이라이트는 버린다 (초)
SNAP = 0.5  # 자를 자리를 이 범위(±초) 안에서 가장 조용한 곳으로
QUIET_TIE_DB = 1.0  # 가장 조용한 값과 이만큼 차이 안이면 같은 조용함 → 원래 자리에 가까운 쪽
MERGE_GAP = 2.0  # 하이라이트 사이가 이보다 짧으면 하나로
MIN_PIECE = 1.0  # 앞뒤에 이보다 짧은 조각이 남으면 자르지 않는다 (하이라이트에 붙인다)
MIN_BETWEEN = 2.0  # 두 하이라이트 사이에 남을 조각이 이보다 짧으면 둘을 한 컷으로 (실측: 1.1초짜리 조각이 생겼다)
MIN_RISE = 12.0  # 하이라이트 기준이 그 영상의 보통 점수보다 적어도 이만큼은 높아야 (표시 점수 기준)
EXIT_SHARE = 0.6  # 나올 때 기준 = 보통 + (들어갈 때 기준 - 보통) × 0.6
PEAK_RISE = (0.45, 0.15)  # 들어갈 기준은 보통 + (가장 센 순간 - 보통) × 이 비율 이상 — 민감도 0일 때 0.45 ~ 100일 때 0.15
# (대부분 조용한 영상은 백분위 기준이 너무 낮아 점수 곡선의 꼬리까지 하이라이트가 됐다 — 실측 2.5초 앞에서 잘림)
DISPLAY_BASE = 20  # 표시 점수: 그 영상의 평소(20백분위) = 0, 가장 센 순간(99백분위) = 100
DISPLAY_RANGE_MIN = 25.0  # 다만 평소~가장 센 순간 폭이 원래 점수로 이보다 좁으면(고른 영상) 이 폭으로 나눈다 — 잡음을 100으로 키우지 않게
HIGHLIGHT_SHARE = 0.5  # 클립의 이만큼 이상이 하이라이트면 '하이라이트 컷'으로 표시


@dataclass(frozen=True)
class Options:
    sensitivity: float = DEFAULT_SENSITIVITY
    pad: float = DEFAULT_PAD
    min_length: float = DEFAULT_MIN_LENGTH


@dataclass(frozen=True)
class Highlight:
    start: float  # 원본 초 (여유를 붙이기 전, 하이라이트 자체)
    end: float
    score: float  # 구간 평균 (표시 점수)
    reasons: tuple[str, ...]


def relative(scores: Scores) -> Scores:
    """표시 점수: 그 영상의 평소 = 0, 가장 센 순간 = 100. 원래 점수 폭이 좁은(고른) 영상은 덜 키운다.

    실측: 원래 점수는 촬영본마다 최고가 45~56점이고 평소에도 15점쯤 깔려 있어서, 컷 뱃지가 하이라이트 47 · 평범한 컷 37처럼
    차이가 잘 안 보였다.
    """
    if len(scores.values) == 0:
        return scores
    base, top = (float(v) for v in np.percentile(scores.values, (DISPLAY_BASE, 99)))
    values = np.clip((scores.values - base) * (100 / max(top - base, DISPLAY_RANGE_MIN)), 0, 100).astype(np.float32)
    return Scores(values, scores.parts, scores.groups)


def thresholds(values: np.ndarray, sensitivity: float) -> tuple[float, float]:
    """(들어갈 때 기준, 나올 때 기준). 민감도 50이면 그 영상의 상위 약 32%부터, 100이면 상위 50%, 0이면 상위 15%.
    또 보통에서 가장 센 순간까지 오르는 폭의 30%(민감도 50)는 넘어야 한다."""
    if len(values) == 0:
        return math.inf, math.inf
    s = float(np.clip(sensitivity, 0, 100)) / 100
    middle, top = float(np.median(values)), float(np.percentile(values, 99))
    rise = PEAK_RISE[0] + (PEAK_RISE[1] - PEAK_RISE[0]) * s
    enter = max(float(np.percentile(values, 85 - 35 * s)), middle + MIN_RISE, middle + rise * (top - middle))
    return enter, middle + EXIT_SHARE * (enter - middle)


def find_highlights(scores: Scores, start: float, end: float, options: Options = Options()) -> list[Highlight]:
    """원본 [start, end)초 안의 하이라이트. scores는 그 영상 전체의 표시 점수 (기준은 전체에서 정한다)."""
    values = scores.values
    enter, leave = thresholds(values, options.sensitivity)
    first, last = max(0, int(start / BIN)), min(len(values), math.ceil(end / BIN))
    spans: list[list[float]] = []
    inside = False
    for i in range(first, last):
        v = float(values[i])
        if not inside and v >= enter:
            inside = True
            j = i  # 시작도 끝과 같은 (나올 때) 기준까지 앞으로 넓힌다 — 앞뒤 경계를 같은 방식으로
            while j > first and float(values[j - 1]) >= leave:
                j -= 1
            spans.append([max(start, j * BIN), min(end, (i + 1) * BIN)])
        elif inside and v >= leave:
            spans[-1][1] = min(end, (i + 1) * BIN)
        else:
            inside = False
    merged: list[list[float]] = []
    for span in spans:
        if merged and span[0] - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    return [Highlight(a, b, scores.mean(a, b), tuple(scores.reasons(a, b))) for a, b in merged if b - a >= options.min_length]


def quiet_point(loudness: np.ndarray, target: float, lo: float, hi: float, audio_offset: float = 0.0) -> float:
    """target(영상 초) ±SNAP 안에서 가장 조용한 곳 — 말이 끊긴 곳. [lo, hi] 밖으로는 안 간다. 소리가 없으면 target."""
    a, b = max(lo, target - SNAP), min(hi, target + SNAP)
    if len(loudness) == 0 or b <= a:
        return min(max(target, lo), hi)
    times = np.arange(math.ceil(a / AUDIO_STEP), math.floor(b / AUDIO_STEP) + 1) * AUDIO_STEP
    index = np.round((times - audio_offset) / AUDIO_STEP).astype(int)  # 영상 t초에 들리는 소리 = 원본 오디오 t - offset초
    ok = (index >= 0) & (index < len(loudness))
    if not ok.any():
        return min(max(target, lo), hi)
    times, levels = times[ok], loudness[index[ok]]
    quiet = levels <= levels.min() + QUIET_TIE_DB
    candidates = times[quiet]
    return float(candidates[np.argmin(np.abs(candidates - target))])


def cut_points(highlights: list[Highlight], start: float, end: float, loudness: np.ndarray | None = None,
               audio_offset: float = 0.0, pad: float = DEFAULT_PAD) -> list[float]:
    """원본 [start, end)초 클립을 자를 자리 (원본 초, 오름차순). 앞뒤 조각이 MIN_PIECE보다 짧아지면 자르지 않는다."""
    loudness = np.zeros(0) if loudness is None else loudness
    lo, hi = start + MIN_PIECE, end - MIN_PIECE
    joined: list[list[float]] = []  # (여유 붙인 시작, 끝)
    for h in highlights:
        if joined and h.start - pad - joined[-1][1] < MIN_BETWEEN:
            joined[-1][1] = h.end + pad
        else:
            joined.append([h.start - pad, h.end + pad])
    cuts: list[float] = []
    for pair in joined:
        for target in pair:
            if lo <= target <= hi:
                cuts.append(quiet_point(loudness, target, lo, hi, audio_offset))
    out: list[float] = []
    for t in sorted(cuts):
        if not out or t - out[-1] >= MIN_PIECE:
            out.append(t)
    return out


# ── 타임라인 ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClipScore:
    """V1 클립 하나의 점수 (점수 줄·뱃지·초록 띠)."""

    scores: Scores  # 그 원본 전체의 표시 점수 (클립의 소리 맞춤 반영)
    mean: float  # 클립 구간 평균
    highlights: tuple[Highlight, ...]  # 클립 안 하이라이트
    is_highlight: bool  # 클립의 절반 이상이 하이라이트
    enter: float  # 하이라이트 기준 점수 (점수 줄에 점선)
    reasons: tuple[str, ...]


class HighlightBook:
    """원본별 재료(Signals)를 들고, 클립마다 점수·하이라이트를 계산해 준다 (같은 값이면 다시 계산하지 않는다).

    재료의 열쇠는 project.json에 적힌 클립 원본 경로(Clip.source) 그대로.
    """

    def __init__(self, signals: dict[str, Signals] | None = None, options: Options = Options()) -> None:
        self.signals: dict[str, Signals] = dict(signals or {})
        self.options = options
        self._scores: dict[tuple[str, int], Scores] = {}
        self._clips: dict[tuple, ClipScore] = {}

    def add(self, source: str, signals: Signals) -> None:
        self.signals[source] = signals
        self._scores = {k: v for k, v in self._scores.items() if k[0] != source}
        self._clips = {k: v for k, v in self._clips.items() if k[0] != source}

    def set_options(self, options: Options) -> None:
        if options != self.options:
            self.options = options
            self._clips.clear()

    def has(self, source: str) -> bool:
        return source in self.signals

    def scores(self, clip: Clip) -> Scores | None:
        signals = self.signals.get(clip.source)
        if signals is None:
            return None
        key = (clip.source, clip.audio_offset_ms)
        if key not in self._scores:
            self._scores[key] = relative(score(signals, clip.audio_offset_ms / 1000))
        return self._scores[key]

    def clip(self, clip: Clip) -> ClipScore | None:
        scores = self.scores(clip)
        if scores is None:
            return None
        key = (clip.source, clip.audio_offset_ms, clip.in_frame, clip.out_frame, clip.fps)
        if key not in self._clips:
            start, end = clip.in_frame / clip.fps, clip.out_frame / clip.fps
            found = find_highlights(scores, start, end, self.options)
            covered = sum(h.end - h.start for h in found)
            enter, _ = thresholds(scores.values, self.options.sensitivity)
            self._clips[key] = ClipScore(scores, scores.mean(start, end), tuple(found),
                                         covered >= HIGHLIGHT_SHARE * (end - start), enter, tuple(scores.reasons(start, end)))
        return self._clips[key]

    def cuts(self, clip: Clip) -> list[float]:
        """그 클립을 자를 자리 (원본 초)."""
        info = self.clip(clip)
        if info is None:
            return []
        start, end = clip.in_frame / clip.fps, clip.out_frame / clip.fps
        return cut_points(list(info.highlights), start, end, self.signals[clip.source].loudness,
                          clip.audio_offset_ms / 1000, self.options.pad)


@dataclass(frozen=True)
class SplitResult:
    project: Project
    highlights: int  # 찾은 하이라이트 수
    cuts: int  # 자른 곳 수
    clips: int  # 살펴본 V1 클립 수


def split_project(project: Project, book: HighlightBook) -> SplitResult:
    """V1 클립을 하이라이트 앞뒤에서 나눈다. 재료가 없는 클립은 그대로."""
    frames: list[tuple[int, str]] = []
    found = looked = 0
    for placed in project.placed():
        if placed.track != MAIN_TRACK or book.clip(placed.clip) is None:
            continue
        looked += 1
        clip = placed.clip
        found += len(book.clip(clip).highlights)
        for t in book.cuts(clip):
            frame = placed.start + round(project.local_of(clip, t))
            if placed.start < frame < placed.end:
                frames.append((frame, clip.id))
    edited = project
    for frame, clip_id in sorted(frames, reverse=True):  # 뒤에서부터: 앞 조각이 원래 id를 지켜서 같은 클립의 앞쪽 자리도 찾는다
        edited = split(edited, frame, clip_id)
    cut = len(edited.clips) - len(project.clips)
    return SplitResult(edited, found, cut, looked)


# ── 나눈 뒤에 이어 할 일 (✨ 옆 ▾ 메뉴) ─────────────────────────────────────────────


@dataclass(frozen=True)
class ClipPick:
    """V1에서 점수가 있는 클립 중 고른 것."""

    ids: frozenset[str]
    frames: int  # 고른 클립 길이 합 (타임라인 프레임)
    unscored: int  # 점수가 없는 V1 클립 수 (분석하지 않은 원본 — 건드리지 않는다)


def _pick(project: Project, book: HighlightBook, chosen) -> ClipPick:
    ids, frames, unscored = set(), 0, 0
    for placed in project.placed():
        if placed.track != MAIN_TRACK:
            continue
        info = book.clip(placed.clip)
        if info is None:
            unscored += 1
        elif chosen(info):
            ids.add(placed.clip.id)
            frames += placed.end - placed.start
    return ClipPick(frozenset(ids), frames, unscored)


def highlight_clips(project: Project, book: HighlightBook) -> ClipPick:
    """하이라이트 컷 (초록 띠 — 클립의 절반 이상이 하이라이트)."""
    return _pick(project, book, lambda info: info.is_highlight)


def other_clips(project: Project, book: HighlightBook) -> ClipPick:
    """하이라이트가 아닌 컷 (점수가 있는 것만)."""
    return _pick(project, book, lambda info: not info.is_highlight)


def low_clips(project: Project, book: HighlightBook, below: float) -> ClipPick:
    """컷 평균 점수가 below보다 낮은 컷 (점수가 있는 것만)."""
    return _pick(project, book, lambda info: info.mean < below)


def clip_means(project: Project, book: HighlightBook) -> list[float]:
    """V1 컷마다 평균 점수 (점수가 있는 것만, 타임라인 순서)."""
    out = []
    for placed in project.placed():
        if placed.track == MAIN_TRACK:
            info = book.clip(placed.clip)
            if info is not None:
                out.append(info.mean)
    return out
