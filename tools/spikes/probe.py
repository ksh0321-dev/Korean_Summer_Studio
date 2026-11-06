"""실험: 녹화 파일의 코덱·프레임 간격(CFR 여부)·A/V 시작 차이 측정. 사용법: python probe.py <파일>"""

import statistics
import sys

import av

with av.open(sys.argv[1]) as container:
    print("format", container.format.name, "| duration", container.duration and container.duration / 1e6, "s", "| bitrate", container.bit_rate)
    for s in container.streams:
        cc = s.codec_context
        extra = f"{cc.width}x{cc.height} avg_rate={s.average_rate} base_rate={s.base_rate}" if s.type == "video" else f"{cc.sample_rate}Hz ch={cc.channels}"
        start = float(s.start_time * s.time_base) if s.start_time is not None else None
        print(s.type, cc.name, extra, "| start", start, "| metadata", dict(s.metadata))

    video = container.streams.video[0]
    pts = sorted(float(p.pts * video.time_base) for p in container.demux(video) if p.pts is not None)

deltas = [round((b - a) * 1000, 2) for a, b in zip(pts, pts[1:])]
span = pts[-1] - pts[0]
print(f"frames {len(pts)} | span {span:.3f}s | expected@30 {round(span * 30) + 1}")
print(f"frame delta ms: mean {statistics.mean(deltas):.2f} min {min(deltas)} max {max(deltas)} stdev {statistics.pstdev(deltas):.2f}")
counts = {d: deltas.count(d) for d in set(deltas)}
print("delta histogram (top):", sorted(counts.items(), key=lambda kv: -kv[1])[:8])
