"""휴대폰 영상 검토: VFR 정도, 오디오/영상 시작 차이, 회전, HDR, 우리 렌더러가 고르는 프레임의 시간 오차.

사용법: python phone_check.py <영상>...
"""

import statistics
import sys
from collections import Counter
from pathlib import Path

import av

from ks_studio.core.project import Clip, Project, timeline_fps_for
from ks_studio.services.sources import probe


def analyze(path: Path) -> None:
    print(f"\n=== {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    with av.open(str(path)) as c:
        print(f"컨테이너 {c.format.name} · 길이 {c.duration / 1e6:.3f}s · 메타 {dict(c.metadata)}")
        v = c.streams.video[0]
        cc = v.codec_context
        print(
            f"영상 {cc.name} {cc.profile or ''} {cc.width}x{cc.height} {cc.pix_fmt} · avg {v.average_rate} base {v.base_rate} "
            f"· time_base {v.time_base} · start {float(v.start_time * v.time_base) if v.start_time is not None else None:.4f}s "
            f"· frames {v.frames}"
        )
        print(f"  색: primaries={cc.color_primaries} trc={cc.color_trc} space={cc.colorspace} range={cc.color_range} · 메타 {dict(v.metadata)}")
        for a in c.streams.audio:
            print(
                f"오디오 {a.codec_context.name} {a.codec_context.sample_rate}Hz {a.codec_context.channels}ch · "
                f"start {float(a.start_time * a.time_base):.4f}s"
            )
        pts = sorted(p.pts for p in c.demux(v) if p.pts is not None)
        tb = v.time_base
        average_rate = float(v.average_rate)
    with av.open(str(path)) as c:
        first = next(c.decode(video=0))
        rotation = getattr(first, "rotation", None)
        print(f"  첫 프레임: format {first.format.name} · rotation {rotation}")

    times = [float(p * tb) for p in pts]
    deltas = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    nominal = 1000 / average_rate
    hist = Counter(round(d, 1) for d in deltas).most_common(6)
    irregular = sum(1 for d in deltas if abs(d - nominal) > nominal * 0.25)
    print(
        f"프레임 간격: 평균 {statistics.mean(deltas):.3f}ms (공칭 {nominal:.3f}) · 표준편차 {statistics.pstdev(deltas):.3f} · "
        f"최소 {min(deltas):.2f} 최대 {max(deltas):.2f} · 공칭에서 25%↑ 벗어남 {irregular}/{len(deltas)}"
    )
    print(f"  간격 분포(상위): {hist}")

    # 우리 렌더러의 프레임 선택을 그대로 흉내 낸다: 원본 프레임 번호 = round((pts - 0) * fps)
    info = probe(path)
    fps = info.fps
    timeline = timeline_fps_for(fps)
    project = Project(fps=timeline, clips=(Clip("x", 0, info.frames, fps=fps),))
    clip = project.clips[0]
    index_of = [round(t * fps) for t in times]
    errors, picked = [], 0
    for local in range(project.length(clip)):
        want = project.source_frame(clip, local)
        while picked + 1 < len(index_of) and index_of[picked + 1] <= want:
            picked += 1
        shown = times[picked]
        ideal_time = local / timeline + times[0]  # 이 타임라인 순간의 실제 원본 시각
        errors.append((shown - ideal_time) * 1000)
    late = [e for e in errors if e < -nominal]  # 한 프레임 넘게 '옛 장면'이 보이는 경우
    early = [e for e in errors if e > nominal / 2]  # 아직 오지 않은 장면이 보이는 경우
    print(
        f"우리 매핑 → 타임라인 {timeline}fps · probe frames {info.frames} (시간 격자 {round((times[-1] - times[0]) * fps) + 1}) · "
        f"클립 길이 {project.length(clip) / timeline:.3f}s vs 실제 {times[-1] - times[0] + nominal / 1000:.3f}s"
    )
    print(
        f"  보여주는 장면의 시간 오차: 평균 {statistics.mean(errors):+.1f}ms · 최소 {min(errors):+.1f} · 최대 {max(errors):+.1f} · "
        f"한 프레임↑ 늦음 {len(late)} · 반 프레임↑ 이름 {len(early)}"
    )


for arg in sys.argv[1:]:
    analyze(Path(arg))
