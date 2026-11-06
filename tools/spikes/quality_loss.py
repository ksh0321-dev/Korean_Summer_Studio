"""실험: 보정본·편집용 사본을 거칠 때 화질이 얼마나 떨어지는지 (효과 0으로 다시 압축만 했을 때의 PSNR).

PSNR(dB)은 원본과 얼마나 같은지: 50 이상이면 눈으로 구분 불가, 40~45는 확대하면 보임, 35 아래는 티가 난다.
사용법: python quality_loss.py <작업폴더> <영상>
"""

import sys
import time
from pathlib import Path

import av
import numpy as np

from ks_studio.core.beauty import BeautySettings
from ks_studio.services.beautify import render_beautified
from ks_studio.services.optimize import inspect, make_optimized

FRAMES = 60


def frames(path: Path, count: int = FRAMES) -> list[np.ndarray]:
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24").astype(np.float64) for _, f in zip(range(count), c.decode(video=0))]


def psnr(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    mse = np.mean([np.mean((x - y) ** 2) for x, y in zip(a, b)])
    return 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")


work, src = Path(sys.argv[1]), Path(sys.argv[2])
work.mkdir(parents=True, exist_ok=True)
plan = inspect(src)
reference = frames(src)
chain = src
if plan is not None:  # 휴대폰 영상: 편집용 사본(1번째 압축) — 회전·색 변환이 들어가므로 원본과는 비교 못 한다
    chain = work / "copy.mp4"
    make_optimized(src, chain, plan)
    reference = frames(chain)
    print(f"{src.name}: 편집용 사본 이유 {plan.reasons} → 사본 기준으로 비교", flush=True)
track = np.full((10_000, 106, 2), np.nan)  # 얼굴 없음 = 효과 0, 다시 압축만
started = time.monotonic()
render_beautified(chain, work / "ai.mp4", track, BeautySettings())
print(f"보정본(효과 0) 재압축: PSNR {psnr(reference, frames(work / 'ai.mp4')):.1f} dB · {time.monotonic() - started:.1f}s · "
      f"크기 {chain.stat().st_size / 1e6:.0f}MB → {(work / 'ai.mp4').stat().st_size / 1e6:.0f}MB", flush=True)
