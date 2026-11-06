"""고정 프레임레이트(CFR) 칸 배치. 순수 파이썬 — Qt import 금지.

실측 사실 (JOYTRON HD20):
- Windows가 프레임을 약 15.6ms 타이머 틱에 맞춰 늦게 건넨다 → 도착 간격 31ms/47ms로 흔들림 (가끔 +30ms)
- 카메라 실제 속도는 29.946fps → 30fps 칸과 조금씩 벌어진다

그래서 프레임은 기본적으로 다음 칸에 하나씩 넣고, 최근 몇 프레임 중 '가장 덜 늦은' 지각값(=흔들림을 뺀
실제 어긋남)이 LIMIT 칸을 넘을 때만 복제·버림으로 맞춘다. 흔들림엔 반응하지 않고 속도 차이·누락에만 반응한다.
단, 프레임 하나가 한 칸 이상 늦거나 이르면 바로 맞춘다. 흔들림(틱 한 개 = 0.47칸 미만)으로는 생길 수 없고,
어두워서 카메라가 fps를 낮추는 순간 생긴다 (실측: 15fps로 떨어진 구간에서 영상이 최대 0.3초 앞당겨졌다).
칸 기준점(origin)은 처음 settle_frames 동안 '가장 일찍 도착한 위상'으로 당겨 맞춘 뒤 고정한다.
"""

import math
from collections import deque

# 어긋남이 이 칸 수를 넘으면 맞춘다. 0.5보다 커야 맞춘 직후(±0.4)에 반대쪽으로 다시 튀지 않는다.
LIMIT = 0.6


class FrameSlotter:
    def __init__(self, fps: int, settle_frames: int) -> None:
        self._fps = fps
        self._settle_frames = settle_frames
        # 최근 0.5초 프레임들의 '놓인 칸 대비 지각값' (칸 단위). 0.5초는 실측 40초 데이터에서
        # 이론값(복제 2·버림 0)과 일치한 최소 구간
        self._residuals: deque[float] = deque(maxlen=max(1, fps // 2))
        self._placed = 0
        self.origin: float | None = None
        self.next_slot = 0
        self.dropped = 0

    def place(self, arrival: float) -> list[int]:
        """이 프레임으로 채울 칸 번호들. 마지막 칸이 이 프레임 자리이고 앞은 복제할 칸. 빈 목록이면 버림."""
        if self.origin is None:
            self.origin = arrival
        self._placed += 1
        if self._placed <= self._settle_frames:
            self.origin = min(self.origin, arrival - self.next_slot / self._fps)
            return self._advance(0)

        lateness = (arrival - self.origin) * self._fps - self.next_slot  # 다음 칸에 넣을 때의 지각값
        if lateness <= -1:  # 한 칸 이상 이름 = 같은 칸에 또 온 프레임
            self.dropped += 1
            return []
        drift = min([*self._residuals, lateness])
        extra = 0
        if drift <= -LIMIT:  # 카메라가 칸보다 빠름 → 하나 버림
            self._residuals.clear()
            self.dropped += 1
            return []
        if drift >= LIMIT:  # 카메라가 칸보다 느림 → 복제로 따라잡음
            extra = math.floor(drift + 1 - LIMIT)
            self._residuals.clear()
        extra = max(extra, math.floor(lateness))  # 한 칸 이상 늦음 = 빠진 프레임 → 바로 채움
        self._residuals.append(lateness - extra)
        return self._advance(extra)

    def slot_time(self, slot: int) -> float:
        return self.origin + slot / self._fps

    def _advance(self, extra: int) -> list[int]:
        slots = list(range(self.next_slot, self.next_slot + extra + 1))
        self.next_slot += extra + 1
        return slots
