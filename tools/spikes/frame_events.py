"""녹화 파일에서 프레임 단위 사건을 찾는다: 복제 칸(앞 프레임과 거의 같음)과 움직임이 툭 튀는 지점(내용 건너뜀 후보).

사용법: python frame_events.py <녹화.mp4>
"""

import sys

import av
import numpy as np

DUP_THRESHOLD = 0.3

diffs = []
with av.open(sys.argv[1]) as container:
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    prev = None
    for frame in container.decode(stream):
        y = frame.to_ndarray(format="gray")[::8, ::8].astype(np.int16)
        diffs.append(np.abs(y - prev).mean() if prev is not None else 0.0)
        prev = y

d = np.array(diffs)
fps = 30
dups = np.flatnonzero(d < DUP_THRESHOLD)[1:]
print(f"frames {len(d)}, 복제 칸 {len(dups)}개")

# 1초 단위로 복제 칸이 3개 이상인 구간 (카메라가 fps를 낮춘 곳)
per_sec = np.bincount(dups // fps, minlength=len(d) // fps + 1)
busy = [(s, c) for s, c in enumerate(per_sec) if c >= 3]
print("복제 칸이 몰린 초:", busy[:40], "..." if len(busy) > 40 else "")

# 움직임 튐: 앞뒤 1초 중앙값의 6배를 넘는 차이
spikes = []
for i in range(fps, len(d) - fps):
    local = np.median(np.r_[d[i - fps : i - 1], d[i + 2 : i + fps]])
    if d[i] > 6 * max(local, 0.5):
        spikes.append((i / fps, round(float(d[i]), 1), round(float(local), 1)))
print(f"움직임 튐 {len(spikes)}개 (시각s, 차이, 주변 중앙값):")
for s in spikes[:60]:
    print("  ", s)
