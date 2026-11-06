"""실험: 휴대폰 원본(회전·HLG HDR·P3·풀 레인지·VFR) → 편집용 사본(SDR BT.709 8bit, 회전 적용, CFR) 3초 변환.

zscale이 없는 PyAV FFmpeg에서 transpose · colorspace · fps 필터만으로 되는지, 결과 장면을 PNG로 확인한다.
사용법: python optimize_spike.py <출력폴더> <영상>...
"""

import sys
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

SECONDS = 3


def chain(frame: av.VideoFrame, cc, target_fps: int) -> list[tuple[str, str]]:
    filters = []
    rotation = round(getattr(frame, "rotation", 0) or 0) % 360  # 표시 행렬 각도 (반시계)
    if rotation == 90:
        filters.append(("transpose", "cclock"))
    elif rotation == 270:
        filters.append(("transpose", "clock"))
    elif rotation == 180:
        filters += [("hflip", ""), ("vflip", "")]
    prim, trc, space, rng = (str(cc.color_primaries), str(cc.color_trc), str(cc.colorspace), str(cc.color_range))
    print(f"  rotation {rotation} · prim {prim} trc {trc} space {space} range {rng} · {frame.format.name}")
    iprim = {"9": "bt2020", "12": "smpte432", "1": "bt709"}.get(prim, "bt709")
    itrc = {"18": "bt2020-10", "14": "bt2020-10", "1": "bt709", "13": "iec61966-2-1"}.get(trc, "bt709")  # HLG는 SDR 감마로 근사
    ispace = {"9": "bt2020ncl", "1": "bt709", "6": "smpte170m", "5": "bt470bg"}.get(space, "bt709")
    irange = "pc" if rng == "2" or frame.format.name.startswith("yuvj") else "tv"
    options = f"all=bt709:iprimaries={iprim}:itrc={itrc}:ispace={ispace}:irange={irange}:range=tv:format=yuv420p"
    filters.append(("colorspace", options))
    filters.append(("fps", str(target_fps)))
    filters.append(("format", "yuv420p"))
    return filters


def run(path: Path, out_dir: Path) -> None:
    print(f"\n=== {path.name}")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        first = next(container.decode(stream))
        target = 60 if float(stream.average_rate) > 45 else 30
        filters = chain(first, stream.codec_context, target)
    started = time.monotonic()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        graph = av.filter.Graph()
        nodes = [graph.add_buffer(template=stream)]
        for name, args in filters:
            nodes.append(graph.add(name, args))
        nodes.append(graph.add("buffersink"))
        graph.link_nodes(*nodes).configure()
        frames = []
        for frame in container.decode(stream):
            if frame.time > SECONDS:
                break
            graph.push(frame)
            while True:
                try:
                    frames.append(graph.pull())
                except (av.BlockingIOError, av.EOFError):
                    break
    elapsed = time.monotonic() - started
    pts = [f.pts for f in frames]
    print(f"  필터 {filters}")
    print(f"  결과 {len(frames)}프레임 {frames[0].width}x{frames[0].height} {frames[0].format.name} · {SECONDS}초 변환 {elapsed:.1f}s"
          f" · pts 간격 {sorted(set(np.diff(pts)))[:4]} (time_base {frames[0].time_base})")
    mid = frames[len(frames) // 2]
    Image.fromarray(mid.to_ndarray(format="rgb24")).save(out_dir / f"{path.stem}_converted.png")
    with av.open(str(path)) as container:  # 비교용: 아무 처리 없이 RGB로만
        raw = next(container.decode(video=0))
        Image.fromarray(raw.to_ndarray(format="rgb24")).save(out_dir / f"{path.stem}_raw.png")


out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for arg in sys.argv[2:]:
    run(Path(arg), out)
