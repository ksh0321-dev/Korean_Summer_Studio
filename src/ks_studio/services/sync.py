"""박수 싱크 보정용 분석: 소리에서 박수를 찾고, 박수 앞뒤 프레임을 뽑는다.

손이 닿는 장면은 사람이 고르되, 앞 프레임과 차이(움직임)가 가장 큰 프레임을 미리 골라 둔다
(실측: JOYTRON·휴대폰 웹캠 모두 손이 닿는 프레임에서 움직임이 가장 컸다).
"""

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from ks_studio.services.sources import AUDIO_RATE, decode_audio

REFRACTORY_S = 0.5  # 박수 사이 최소 간격
WINDOW_BEFORE_S, WINDOW_AFTER_S = 0.2, 0.35  # 박수 소리 기준으로 보여줄 범위
GUESS_BEFORE_S, GUESS_AFTER_S = 0.1, 0.3  # 이 범위에서 움직임이 가장 큰 프레임을 미리 고른다
DUPLICATE_MOTION = 0.3  # 이보다 작으면 엔진이 복제한 칸 (같은 장면)
THUMB = (216, 216)  # 화면 가운데(박수 치는 곳)를 정사각형으로 잘라 크게 보여준다


@dataclass(frozen=True)
class ClapFrame:
    time: float
    image: np.ndarray  # RGB 축소본
    motion: float  # 직전 프레임과의 평균 차이


@dataclass(frozen=True)
class Clap:
    sound_time: float
    frames: list[ClapFrame]
    guess: int  # 미리 골라 둔 '손이 처음 닿은' 프레임 번호

    def previous_scene_time(self, index: int) -> float:
        """index 직전의, 복제가 아닌 다른 장면 프레임 시각."""
        for i in range(index - 1, -1, -1):
            if self.frames[i + 1].motion >= DUPLICATE_MOTION:
                return self.frames[i].time
        return self.frames[index].time


def clap_onsets(samples: np.ndarray, rate: int) -> list[float]:
    """5ms 블록 최대값이 '배경의 20배 또는 전체 최대의 20%'를 넘는 곳 → 그 앞 10ms 안에서 소리가 서는 첫 샘플."""
    env = np.abs(samples)
    block = rate // 200
    blocks = env[: len(env) // block * block].reshape(-1, block).max(axis=1)
    if len(blocks) == 0:
        return []
    threshold = max(0.2 * blocks.max(), 20 * np.median(blocks))
    onsets, last = [], -np.inf
    for i in np.flatnonzero(blocks >= threshold):
        start = i * block
        if (start - last) / rate < REFRACTORY_S:
            continue
        lo = max(0, start - rate // 100)
        window = env[lo : start + 2 * block]
        first = lo + int(np.argmax(window >= 0.25 * window.max()))
        onsets.append(first / rate)
        last = start
    return onsets


def frames_around(path: Path, onset: float) -> list[ClapFrame]:
    lo, hi = onset - WINDOW_BEFORE_S, onset + WINDOW_AFTER_S
    out, previous = [], None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int((lo - 1.0) / stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t > hi:
                break
            gray = frame.to_ndarray(format="gray")[::8, ::8].astype(np.int16)
            motion = float(np.abs(gray - previous).mean()) if previous is not None else 0.0
            previous = gray
            if t >= lo:
                size = THUMB[1]
                wide = frame.reformat(width=round(size * frame.width / frame.height), height=size, format="rgb24").to_ndarray()
                x0 = (wide.shape[1] - size) // 2
                out.append(ClapFrame(t, np.ascontiguousarray(wide[:, x0 : x0 + size]), motion))
    return out


def find_claps(path: Path) -> list[Clap]:
    audio = decode_audio(path).astype(np.float32) / 32768
    claps = []
    for onset in clap_onsets(audio, AUDIO_RATE):
        frames = frames_around(path, onset)
        if not frames:
            continue
        candidates = [i for i, f in enumerate(frames) if onset - GUESS_BEFORE_S <= f.time <= onset + GUESS_AFTER_S]
        guess = max(candidates or range(len(frames)), key=lambda i: frames[i].motion)
        claps.append(Clap(onset, frames, guess))
    return claps
