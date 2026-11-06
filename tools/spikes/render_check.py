"""검증: 렌더러가 원본의 정확한 프레임·오디오를 고르는지, 미리보기 해상도로 얼마나 빠른지.

사용법: python render_check.py <녹화.mp4>
"""

import sys
import time
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QCoreApplication

from ks_studio.core.project import Clip, Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.render import TimelineRenderer
from ks_studio.services.sources import SourceCache

source = Path(sys.argv[1]).resolve()
app = QCoreApplication([])
ws = Workspace(source.parent.parent)
sources = SourceCache()
info = sources.info(source)
print("info:", info)

# 원본을 직접 풀어서 정답 프레임 확보
truth = {}
with av.open(str(source)) as c:
    s = c.streams.video[0]
    for f in c.decode(s):
        k = round(f.pts * s.time_base * 30)
        if k in (10, 19, 100, 129):
            truth[k] = f.to_ndarray(format="rgb24")

project = Project(clips=(Clip(ws.to_source(source), 100, 130, id="A"), Clip(ws.to_source(source), 10, 20, audio_offset_ms=50, id="B")))
renderer = TimelineRenderer(project, ws, sources, (info.width, info.height), "rgb24")
t0 = time.perf_counter()
audio = sources.audio(source)
print(f"오디오 전체 디코딩 {time.perf_counter() - t0:.2f}s, {len(audio) / 48000:.2f}s 분량")

rendered = list(renderer.frames(0))
print("렌더 프레임 수:", len(rendered), "(기대 40)")
checks = {0: 100, 29: 129, 30: 10, 39: 19}
for t, k in checks.items():
    same = np.array_equal(rendered[t][1].to_ndarray(), truth[k])
    print(f"  타임라인 {t:2d} = 원본 {k:3d} 프레임? {same}")
n = 1600
expected_a = audio[100 * n : 101 * n] / 32768
expected_b = audio[10 * n - 2400 : 11 * n - 2400] / 32768  # +50ms = 2400샘플 늦춤 → 원본에서 그만큼 앞
print("  오디오 A 첫 프레임 일치:", np.allclose(rendered[0][2], expected_a))
print("  오디오 B(+50ms) 첫 프레임 일치:", np.allclose(rendered[30][2], expected_b))

# 미리보기 속도: 전체 녹화를 960x540 RGB로
preview = TimelineRenderer(Project(clips=(Clip(ws.to_source(source), 0, info.frames),)), ws, sources, (960, 540), "rgb24")
t0 = time.perf_counter()
count = sum(1 for _ in preview.frames(0))
elapsed = time.perf_counter() - t0
print(f"미리보기 렌더 {count}프레임 {elapsed:.2f}s → {count / elapsed:.0f}fps")

# 찾아가기 비용: 키프레임 사이 한가운데 프레임 하나 꺼내기
t0 = time.perf_counter()
for target in (15, 115, 215, 315):
    next(preview.frames(target, with_audio=False))
print(f"찾아가기 1회 평균 {(time.perf_counter() - t0) / 4 * 1000:.0f}ms")
