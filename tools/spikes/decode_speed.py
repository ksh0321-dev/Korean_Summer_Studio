"""실험: 편집용 사본·미리보기 단계별 속도 — CPU 디코딩 vs GPU(NVDEC) 디코딩, 색 변환 필터, 축소, 인코딩.

사용법: python decode_speed.py <영상> [프레임 수]
"""

import sys
import time
from fractions import Fraction
from pathlib import Path

import av
from av.codec.hwaccel import HWAccel

from ks_studio.services.optimize import inspect, video_encoder

path = Path(sys.argv[1])
count = int(sys.argv[2]) if len(sys.argv) > 2 else 240


def decode(hw: bool):
    frames = []
    kwargs = {"hwaccel": HWAccel(device_type="cuda", allow_software_fallback=False)} if hw else {}
    started = time.perf_counter()
    with av.open(str(path), **kwargs) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for f in c.decode(s):
            frames.append(f)
            if len(frames) >= count:
                break
    return frames, time.perf_counter() - started


def per_frame(label, seconds):
    print(f"  {label}: {seconds / count * 1000:6.1f} ms/프레임 ({count / seconds:5.1f} fps)", flush=True)


plan = inspect(path)
print(f"{path.name}: 사본 이유 {plan.reasons if plan else '없음'}", flush=True)
cpu_frames, t = decode(False)
per_frame(f"CPU 디코딩 ({cpu_frames[0].format.name})", t)
try:
    gpu_frames, t = decode(True)
    per_frame(f"GPU 디코딩 NVDEC ({gpu_frames[0].format.name})", t)
except Exception as e:
    print("  GPU 디코딩 실패:", e)
    gpu_frames = cpu_frames

frame = gpu_frames[0]
t = time.perf_counter()
for f in gpu_frames[:60]:
    f.reformat(width=960, height=540, format="rgb24")
print(f"  미리보기 축소 960x540: {(time.perf_counter() - t) / 60 * 1000:.1f} ms/프레임", flush=True)

if plan:
    graph = av.filter.Graph()
    with av.open(str(path)) as c:
        nodes = [graph.add_buffer(template=c.streams.video[0])]
    for name, args in (*plan.filters, ("fps", str(plan.fps)), ("format", "yuv420p")):
        nodes.append(graph.add(name, args))
    nodes.append(graph.add("buffersink"))
    graph.link_nodes(*nodes).configure()
    out = []
    t = time.perf_counter()
    for f in cpu_frames:
        graph.push(f)
        while True:
            try:
                out.append(graph.pull())
            except (av.BlockingIOError, av.EOFError):
                break
    per_frame(f"CPU 필터 {[n for n, _ in plan.filters] + ['fps', 'format']}", time.perf_counter() - t)
    name, options = video_encoder(plan.size, plan.fps)
    with av.open(str(Path(sys.argv[0]).with_name("_speed_out.mp4")), "w") as dst:
        v = dst.add_stream(name, rate=plan.fps, options=options)
        v.width, v.height = plan.size
        v.pix_fmt = "yuv420p"
        t = time.perf_counter()
        for i, f in enumerate(out):
            f.pts, f.time_base = i, Fraction(1, plan.fps)
            dst.mux(v.encode(f))
        dst.mux(v.encode())
        per_frame(f"인코딩 {name}", time.perf_counter() - t)
    Path(sys.argv[0]).with_name("_speed_out.mp4").unlink(missing_ok=True)
