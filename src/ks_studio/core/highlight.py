"""하이라이트 점수: 소리·표정·움직임 → 0.5초마다 0~100점. 순수 계산 — Qt·파일 접근 없음.

재료(Signals)는 services.highlight가 영상에서 뽑는다:
  - 소리 세기: 0.1초마다 dB (원본 오디오 시각)
  - 움직임: 0.2초(1초에 5장)마다 앞 장과의 밝기 차이
  - 표정: 0.2초마다 가장 큰 얼굴의 웃음(입 폭·입꼬리) · 입 벌림 · 눈썹 올라감 · 얼마나 잘 보이나(정면·크기)

점수는 '그 영상 안에서 얼마나 두드러지나'로 매긴다: 항목마다 그 영상의 보통(중앙값)에서 얼마나 올라갔는지.
원래 목소리가 큰 사람이면 큰 목소리가 보통이 된다. 다만 너무 고른 영상이 잡음만으로 높은 점수를 받지 않게
만점이 되려면 넘어야 하는 최소 폭(FEATURES의 넷째 값)을 둔다. 이 값들은 실제 촬영본으로 맞춰 갈 자리다.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from ks_studio.core.beauty import BROW_TOPS, CHIN, EYE_CENTERS, EYE_OUTER, LIPS_INNER, LIPS_OUTER, MOUTH_CORNERS, eye_distance

BIN = 0.5  # 점수 한 칸 (초)
AUDIO_STEP = 0.1  # 소리 세기 한 칸 (초)
SAMPLE_STEP = 0.2  # 영상은 1초에 5장만 본다
SILENT_DB = -60.0  # 이보다 조용하면 다 같은 '조용함' (숨소리·바닥 잡음 차이로 점수가 흔들리지 않게)
JUMP_BEFORE = 3.0  # '소리가 확 커짐' = 앞 3초 보통 세기보다 얼마나 커졌나
SMOOTH_SIGMA = 0.5  # 점수를 이만큼(초) 부드럽게 — 튐을 누그러뜨린다 (1초였을 땐 경계가 앞뒤로 1.5초씩 번졌다. 쪼개짐은 분할 쪽 합치기가 막는다)
SCENE_CUT = 0.25  # 앞 장과 밝기 차이가 이보다 크면 장면 전환 → 움직임으로 치지 않는다
FACE_COLUMNS = ("smile", "mouth", "brow", "presence")

# 항목: (이름, 묶음, 묶음 안 비중, 최소 폭, 이유로 보여줄 말)
FEATURES = (
    ("level", "sound", 0.6, 10.0, "목소리 커짐"),  # dB
    ("jump", "sound", 0.4, 10.0, "소리가 확 커짐"),  # dB (웃음·감탄·박수)
    ("smile", "face", 0.45, 0.08, "웃는 얼굴"),  # 입 폭·입꼬리 (눈 사이 거리 대비)
    ("mouth", "face", 0.3, 0.03, "입 크게 벌림"),  # 안쪽 입술 넓이 (눈 사이 거리² 대비)
    ("brow", "face", 0.25, 0.04, "눈썹 올라감"),  # 눈 ~ 눈썹 거리 (눈 사이 거리 대비)
    ("motion", "motion", 1.0, 0.03, "움직임 큼"),  # 0~1 밝기 차이
)
TOP = 98  # 그 영상에서 이 백분위(가장 센 2%)만큼 올라가면 1
PEAK_SHARE = 0.5  # 묶음 값 = 가장 센 항목 × 0.5 + 비중 평균 × 0.5 (웃음 하나만 뚜렷해도 표정 점수가 충분히 오르게). 전체 점수도 묶음끼리 같은 식
GROUPS = {"sound": "소리", "face": "표정", "motion": "움직임"}
DEFAULT_WEIGHTS = {"sound": 0.45, "face": 0.4, "motion": 0.15}
REASON_MIN = 0.35  # 구간 평균이 이만큼 두드러진 항목만 이유로
REASONS_SHOWN = 3


@dataclass(frozen=True)
class Signals:
    """영상 하나에서 뽑은 재료. 시각은 모두 원본 기준 (소리는 원본 오디오, 나머지는 원본 영상)."""

    duration: float  # 영상 길이 (초)
    loudness: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))  # AUDIO_STEP마다 dB (소리가 없으면 빈 배열)
    motion: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))  # SAMPLE_STEP마다 0~1 (첫 장·장면 전환은 NaN)
    face: np.ndarray | None = None  # (SAMPLE_STEP마다, FACE_COLUMNS) — 얼굴이 없는 장은 NaN. 표정을 안 봤으면 None


@dataclass(frozen=True)
class Scores:
    values: np.ndarray  # BIN마다 0~100
    parts: dict[str, np.ndarray]  # 항목 이름 → BIN마다 0~1 (얼마나 두드러졌나, 이유 표시용)
    groups: tuple[str, ...]  # 점수에 들어간 묶음 (표정을 못 봤으면 빠진다)

    def at(self, seconds: float) -> float:
        if len(self.values) == 0:
            return 0.0
        return float(self.values[min(len(self.values) - 1, max(0, int(seconds / BIN)))])

    def mean(self, start: float, end: float) -> float:
        a, b = self._slice(start, end)
        return float(self.values[a:b].mean()) if b > a else 0.0

    def reasons(self, start: float, end: float) -> list[str]:
        """그 구간에서 두드러진 것 (센 것부터, 많아야 3개). 예: ['웃는 얼굴', '소리가 확 커짐']."""
        a, b = self._slice(start, end)
        if b <= a:
            return []
        labels = {name: label for name, _, _, _, label in FEATURES}
        strong = [(float(part[a:b].mean()), name) for name, part in self.parts.items()]
        strong = sorted((s for s in strong if s[0] >= REASON_MIN), reverse=True)
        return [labels[name] for _, name in strong[:REASONS_SHOWN]]

    def _slice(self, start: float, end: float) -> tuple[int, int]:
        n = len(self.values)
        return max(0, int(start / BIN)), min(n, max(int(start / BIN) + 1, math.ceil(end / BIN)))


# ── 표정: 얼굴 106점 → 숫자 ──────────────────────────────────────────────────────


def _axis(lm: np.ndarray) -> np.ndarray:
    """얼굴 세로 방향 (두 눈 가운데 → 턱 끝) 단위 벡터. 고개가 기울어도 따라간다."""
    axis = lm[CHIN] - (lm[EYE_OUTER[0]] + lm[EYE_OUTER[1]]) / 2
    return axis / (np.linalg.norm(axis) + 1e-9)


def _area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2


def expression(lm: np.ndarray) -> tuple[float, float, float]:
    """(웃음, 입 벌림, 눈썹 올라감) — 모두 눈 사이 거리로 나눠 얼굴 크기와 상관없게.

    웃음 = 입 폭 + 입꼬리가 입술 가운데보다 얼마나 위에 있나. 눈썹 = 눈 가운데와 눈썹 위 끝의 세로 거리.
    사람마다 원래 값이 달라서 점수는 그 영상 안의 보통과 견준다 (score).
    """
    eyes = eye_distance(lm)
    if eyes < 1e-6:
        return math.nan, math.nan, math.nan
    axis = _axis(lm)
    corners = lm[list(MOUTH_CORNERS)]
    width = float(np.linalg.norm(corners[1] - corners[0])) / eyes
    lift = float(np.dot(lm[list(LIPS_OUTER)].mean(axis=0) - corners.mean(axis=0), axis)) / eyes
    mouth = _area(lm[list(LIPS_INNER)]) / eyes**2
    brow = float(np.dot(lm[list(EYE_CENTERS)].mean(axis=0) - lm[list(BROW_TOPS)].mean(axis=0), axis)) / eyes
    return width + lift, mouth, brow


# ── 점수 ─────────────────────────────────────────────────────────────────────────


def bin_count(duration: float) -> int:
    return max(0, math.ceil(duration / BIN - 1e-9))


def _to_bins(values: np.ndarray, step: float, n: int, how: str = "mean", shift: float = 0.0) -> np.ndarray:
    """step 간격 값들을 BIN 칸으로 (칸 안 평균 또는 최대). 값이 없는 칸은 NaN. shift(초)만큼 뒤로 민다."""
    out = np.full(n, np.nan)
    if len(values) == 0 or n == 0:
        return out
    index = np.floor((np.arange(len(values)) * step + shift) / BIN + 1e-9).astype(int)
    ok = (index >= 0) & (index < n) & ~np.isnan(values)
    index, values = index[ok], values[ok]
    if how == "max":
        np.fmax.at(out, index, values)
        return out
    sums = np.bincount(index, values, minlength=n)
    counts = np.bincount(index, minlength=n)
    with np.errstate(invalid="ignore"):
        out = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return out


def _jump(loudness: np.ndarray) -> np.ndarray:
    """0.1초마다: 앞 3초 보통(중앙값) 세기보다 얼마나 커졌나 (dB, 0 이상)."""
    window = round(JUMP_BEFORE / AUDIO_STEP)
    if len(loudness) == 0:
        return loudness
    padded = np.concatenate([np.full(window, loudness[0]), loudness])
    before = np.median(np.lib.stride_tricks.sliding_window_view(padded[:-1], window), axis=1)
    return np.maximum(0.0, loudness - before)


def stand_out(values: np.ndarray, floor: float) -> np.ndarray:
    """그 영상 안에서 얼마나 두드러지나 0~1: 보통(중앙값)이면 0, 그 영상에서 가장 센 순간들(TOP 백분위)이면 1.
    다만 1이 되려면 적어도 floor만큼은 올라가야 한다 — 고른 영상의 잡음이 만점이 되지 않게. NaN은 0.

    실측 (촬영본 4개): 고개 각도만 바뀌어도 웃음 값이 흔들려서, '보통 흔들림의 몇 배'로 재면 진짜 웃음도 0.3~0.6에 머물렀다."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.zeros(len(values))
    middle = float(np.median(valid))
    spread = max(float(np.percentile(valid, TOP)) - middle, floor)
    out = np.clip((values - middle) / spread, 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0)


def _smooth(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    n = len(values)
    if n < 2 or sigma_bins <= 0:
        return values
    radius = max(1, round(3 * sigma_bins))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_bins) ** 2)
    weight = np.convolve(np.ones(n), kernel)[radius : radius + n]  # 끝에서도 평균이 줄지 않게
    return np.convolve(values, kernel)[radius : radius + n] / weight


def score(signals: Signals, audio_offset: float = 0.0, weights: dict[str, float] | None = None) -> Scores:
    """0.5초마다 0~100점.

    audio_offset = 클립의 소리 맞춤(초, Clip.audio_offset_ms/1000): 영상 시각 t에 들리는 소리는 원본 오디오의 t - offset.
    weights = 묶음별 비중 (소리·표정·움직임). 재료가 없는 묶음(소리 없는 영상, 표정 안 봄)은 빼고 나머지로 나눈다.
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    n = bin_count(signals.duration)
    raw: dict[str, np.ndarray] = {}
    if len(signals.loudness):
        loud = np.maximum(signals.loudness.astype(np.float64), SILENT_DB)
        shift = audio_offset  # 원본 오디오 a초의 소리는 영상 a + offset초에 들린다
        raw["level"] = _to_bins(loud, AUDIO_STEP, n, shift=shift)
        raw["jump"] = _to_bins(_jump(loud), AUDIO_STEP, n, "max", shift=shift)
    if len(signals.motion):
        raw["motion"] = _to_bins(signals.motion.astype(np.float64), SAMPLE_STEP, n, "max")
    face_seen = signals.face is not None and len(signals.face) and not np.all(np.isnan(signals.face[:, 3]))
    if face_seen:
        presence = np.nan_to_num(_to_bins(signals.face[:, 3].astype(np.float64), SAMPLE_STEP, n), nan=0.0)
        for column, name in enumerate(FACE_COLUMNS[:3]):
            raw[name] = _to_bins(signals.face[:, column].astype(np.float64), SAMPLE_STEP, n)

    parts: dict[str, np.ndarray] = {}
    group_sum = {g: np.zeros(n) for g in GROUPS}
    group_peak = {g: np.zeros(n) for g in GROUPS}
    group_share = dict.fromkeys(GROUPS, 0.0)
    for name, group, share, floor, _label in FEATURES:
        if name not in raw:
            continue
        part = stand_out(raw[name], floor)
        if group == "face":
            part = part * presence  # 옆얼굴·작은 얼굴의 표정은 덜 믿는다, 얼굴 없는 칸은 0
        parts[name] = part
        group_sum[group] += share * part
        group_peak[group] = np.maximum(group_peak[group], part)
        group_share[group] += share
    used = tuple(g for g in GROUPS if group_share[g] > 0 and weights.get(g, 0) > 0)
    total_weight = sum(weights[g] for g in used)
    mean, peak = np.zeros(n), np.zeros(n)
    top_weight = max((weights[g] for g in used), default=1.0)
    for g in used:
        mixed = PEAK_SHARE * group_peak[g] + (1 - PEAK_SHARE) * group_sum[g] / group_share[g]
        mean += weights[g] / total_weight * mixed
        peak = np.maximum(peak, mixed * weights[g] / top_weight)
    # 묶음끼리도 같은 방식: 소리·표정·움직임은 서로 조금씩 다른 순간에 튄다 — 평균만 내면 진짜 하이라이트도 30~40점이었다
    values = PEAK_SHARE * peak + (1 - PEAK_SHARE) * mean
    values = np.clip(_smooth(values, SMOOTH_SIGMA / BIN), 0.0, 1.0) * 100
    return Scores(values.astype(np.float32), {k: v.astype(np.float32) for k, v in parts.items()}, used)
