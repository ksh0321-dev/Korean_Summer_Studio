"""임시 내보내기: 타임라인을 그대로 MP4로 (H.264 NVENC + AAC). AI 보정·마무리 단계가 생기기 전까지 쓰는 용도."""

import os
import threading
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import av

from ks_studio.core.project import Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.render import TimelineRenderer
from ks_studio.services.sources import AUDIO_RATE, SourceCache

VIDEO_OPTIONS = {"preset": "p5", "rc": "vbr", "bf": "0"}
VIDEO_MBPS_AT_30FPS = (16, 20)  # (평균, 최대). fps에 비례해 올린다
AUDIO_BITRATE = 192_000


def export_project(
    project: Project,
    workspace: Workspace,
    sources: SourceCache,
    output: Path,
    size: tuple[int, int],
    on_progress: Callable[[int, int], None],
    cancel: threading.Event,
) -> bool:
    """끝까지 쓰면 True, 취소하면 False (쓰다 만 파일은 지운다)."""
    fps = project.fps
    renderer = TimelineRenderer(project, workspace, sources, size, "yuv420p")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".part")
    with av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart"}) as container:
        average, peak = (round(mbps * fps / 30) for mbps in VIDEO_MBPS_AT_30FPS)
        options = {**VIDEO_OPTIONS, "b": f"{average}M", "maxrate": f"{peak}M", "g": str(fps)}
        video = container.add_stream("h264_nvenc", rate=fps, options=options)
        video.width, video.height = size
        video.pix_fmt = "yuv420p"
        video.time_base = Fraction(1, fps)
        audio = container.add_stream("aac", rate=AUDIO_RATE, layout="mono")
        audio.bit_rate = AUDIO_BITRATE

        for t, frame, samples in renderer.frames(0):
            if cancel.is_set():
                break
            frame.pts, frame.time_base = t, Fraction(1, fps)  # 원본 시간 단위가 남아 있으면 엉뚱하게 환산된다
            container.mux(video.encode(frame))
            chunk = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
            chunk.sample_rate, chunk.pts, chunk.time_base = AUDIO_RATE, t * renderer.samples_per_frame, Fraction(1, AUDIO_RATE)
            container.mux(audio.encode(chunk))
            if t % fps == 0:
                on_progress(t, project.duration)
        else:
            container.mux(video.encode())
            container.mux(audio.encode())
    if cancel.is_set():
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, output)
    return True
