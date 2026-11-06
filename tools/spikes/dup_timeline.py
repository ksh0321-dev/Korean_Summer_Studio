"""녹화 파일에서 '앞 프레임과 똑같은 프레임'(= 엔진이 복제한 칸) 비율을 구간별로 출력. 카메라가 fps를 낮춘 구간을 찾는다.

사용법: python dup_timeline.py <녹화.mp4> [구간초=30]
"""

import sys

import av
import numpy as np

path = sys.argv[1]
bucket = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
counts: dict[int, list[int]] = {}
with av.open(path) as container:
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    prev = None
    for frame in container.decode(stream):
        y = frame.to_ndarray(format="gray")[::8, ::8]
        key = int(float(frame.pts * stream.time_base) // bucket)
        total_dup = counts.setdefault(key, [0, 0])
        total_dup[0] += 1
        # 복제 칸도 다시 인코딩되므로 비트가 같지는 않다 → 평균 차이가 아주 작으면 복제로 본다
        if prev is not None and np.abs(y.astype(np.int16) - prev).mean() < 0.3:
            total_dup[1] += 1
        prev = y.astype(np.int16)

for key, (total, dup) in sorted(counts.items()):
    real_fps = (total - dup) / bucket
    print(f"{key * bucket:6.0f}s~  복제 {dup:4d}/{total:4d} ({dup / total:5.1%})  → 실제 카메라 약 {real_fps:5.1f}fps")
