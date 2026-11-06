"""검증: fps가 다른 원본(60 / 30 / 29.97)을 섞은 타임라인이 정확한 원본 프레임·오디오를 쓰는지.

프레임마다 자기 번호를 2진 블록으로 새기고, 오디오엔 원본 1.000초에 클릭을 넣은 합성 영상을 만든다.
사용법: python mixed_fps_check.py <작업폴더>
"""

import sys
import threading
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QCoreApplication

from ks_studio.core.project import Clip, Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.export import export_project
from ks_studio.services.render import TimelineRenderer
from ks_studio.services.sources import AUDIO_RATE, SourceCache

W, H, BITS = 320, 180, 10
BLOCK = W // BITS


def make_source(path: Path, rate: Fraction, seconds: float) -> int:
    frames = round(seconds * rate)
    with av.open(str(path), "w") as out:
        v = out.add_stream("libx264", rate=rate, options={"crf": "0"})
        v.width, v.height, v.pix_fmt = W, H, "yuv420p"
        a = out.add_stream("aac", rate=AUDIO_RATE, layout="mono")
        for i in range(frames):
            img = np.zeros((H, W, 3), np.uint8)
            for b in range(BITS):
                if i >> b & 1:
                    img[:, b * BLOCK : (b + 1) * BLOCK] = 255
            f = av.VideoFrame.from_ndarray(img, format="rgb24")
            f.pts = i
            out.mux(v.encode(f))
        out.mux(v.encode())
        sound = np.zeros(round(seconds * AUDIO_RATE), np.float32)
        sound[AUDIO_RATE : AUDIO_RATE + 48] = 0.9  # 원본 1.000초에 1ms 클릭
        for s in range(0, len(sound), 1024):
            af = av.AudioFrame.from_ndarray(sound[s : s + 1024].reshape(1, -1), format="flt", layout="mono")
            af.sample_rate, af.pts = AUDIO_RATE, s
            out.mux(a.encode(af))
        out.mux(a.encode())
    return frames


def read_index(frame: av.VideoFrame) -> int:
    gray = frame.to_ndarray(format="gray")
    return sum(1 << b for b in range(BITS) if gray[H // 2, b * BLOCK + BLOCK // 2] > 128)


def check(label: str, project: Project, ws: Workspace, sources: SourceCache) -> None:
    renderer = TimelineRenderer(project, ws, sources, (W, H), "yuv420p")
    rendered = list(renderer.frames(0))
    wrong, start = 0, 0
    for clip in project.clips:
        for local in range(project.length(clip)):
            if read_index(rendered[start + local][1]) != project.source_frame(clip, local):
                wrong += 1
        start += project.length(clip)
    audio = np.concatenate([r[2] for r in rendered])
    click_at = np.flatnonzero(np.abs(audio) > 0.3)
    click = f"{click_at[0] / AUDIO_RATE:.4f}s" if len(click_at) else "없음"
    print(f"[{label}] 타임라인 {project.fps}fps, {len(rendered)}프레임 (기대 {project.duration}), 원본 프레임 틀림 {wrong}개, 첫 클릭 {click}")


app = QCoreApplication([])
work = Path(sys.argv[1])
(work / "media").mkdir(parents=True, exist_ok=True)
ws, sources = Workspace(work), SourceCache()
rates = {"60": Fraction(60), "30": Fraction(30), "2997": Fraction(30000, 1001)}
info = {}
for name, rate in rates.items():
    path = work / "media" / f"src{name}.mp4"
    make_source(path, rate, 3.0)
    info[name] = sources.info(path)
    print(f"원본 {name}: {info[name]}")


def clip(name: str, first: int = 0, last: int | None = None) -> Clip:
    i = info[name]
    return Clip(f"media/src{name}.mp4", first, i.frames if last is None else last, fps=i.fps)


check("60fps 원본 → 30fps 타임라인 (클릭 기대 1.0000s)", Project(fps=30, clips=(clip("60"),)), ws, sources)
check("30fps 원본 → 60fps 타임라인 (클릭 기대 1.0000s)", Project(fps=60, clips=(clip("30"),)), ws, sources)
check("29.97 원본 → 30fps 타임라인 (클릭 기대 1.0000s)", Project(fps=30, clips=(clip("2997"),)), ws, sources)
# 섞기: 30fps 원본 0.5~1.5초 뒤에 60fps 원본 0.5~1.5초 → 60fps 원본의 클릭은 타임라인 1.5초
mixed = Project(fps=60, clips=(clip("30", 15, 45), clip("60", 30, 90)))
check("30+60 섞기 → 60fps 타임라인 (클릭 기대 0.5000s, 1.5000s 중 첫 번째)", mixed, ws, sources)

out = work / "export60.mp4"
ok = export_project(mixed, ws, sources, out, (W, H), lambda d, t: None, threading.Event())
with av.open(str(out)) as c:
    v = c.streams.video[0]
    pts = [float(p.pts * v.time_base) for p in c.demux(v) if p.pts is not None]
    deltas = {round((b - a) * 1000, 2) for a, b in zip(sorted(pts), sorted(pts)[1:])}
    print(f"[내보내기 60fps] ok={ok} 프레임 {len(pts)} (기대 {mixed.duration}), 간격 {deltas} ms, avg_rate {v.average_rate}")
