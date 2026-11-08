"""최종 내보내기: 컷편집 타임라인(자막은 여기서 굽는다) 또는 영상 파일 하나 → 고른 비율·해상도·형식의 배포용 MP4.

아무 영상이나 바로: 녹화본·컷편집 결과·AI 보정본·휴대폰 영상. 휴대폰 영상은 편집용 사본이 아니라 원본에서
회전·색 변환·비율·자막을 한 번에 하고 한 번만 압축한다 (편집용 사본과 같은 길 — optimize.make_optimized).
저해상도 미리보기: 짧은 변 800px·빠른 인코더로 같은 결과를 먼저 본다 (내보내고 확인하기는 번거롭다).
"""

import logging
import math
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import os
from fractions import Fraction

import av
import numpy as np
from PySide6.QtCore import QObject, Signal

from ks_studio.core.subtitles import Subtitles
from ks_studio.services.audio_post import AudioPost
from ks_studio.services.captions import CaptionBurner, OverlayBurner
from ks_studio.core.project import Project
from ks_studio.core.workspace import Workspace
from ks_studio.services.accel import hardware_encoders
from ks_studio.services.optimize import Cancelled, Plan, add_video_stream, inspect, make_optimized
from ks_studio.services.render import TimelineRenderer, canvas_size
from ks_studio.services.sources import AUDIO_RATE, SourceCache

log = logging.getLogger(__name__)

RATIOS = {  # 칸 값 → (이름, 가로:세로 또는 None = 원본 비율)
    "original": ("원본 비율", None),
    "9:16": ("9:16 세로 (릴스·쇼츠·틱톡)", (9, 16)),
    "4:5": ("4:5 세로 (인스타 피드)", (4, 5)),
    "1:1": ("1:1 정사각형", (1, 1)),
    "16:9": ("16:9 가로 (유튜브)", (16, 9)),
}
FITS = {"crop": "가운데 자르기 (화면 꽉 차게)", "pad": "검은 여백 (자르지 않게)"}
# 내보내기 프리셋: 고르면 비율·맞추기·해상도·형식·소리 크기 맞추기를 한 번에 (고른 뒤 하나씩 바꿀 수도 있다)
PRESETS = {
    "shorts": ("쇼츠 · 릴스 · 틱톡 (9:16 · 1080p)", ("9:16", "crop", 1080, "h264", True)),
    "feed": ("인스타 피드 (4:5 · 1080p)", ("4:5", "crop", 1080, "h264", True)),
    "square": ("정사각형 (1:1 · 1080p)", ("1:1", "crop", 1080, "h264", True)),
    "youtube": ("유튜브 (16:9 · 1080p)", ("16:9", "pad", 1080, "h264", True)),
    "youtube4k": ("유튜브 4K (16:9 · 2160p · H.265)", ("16:9", "pad", 2160, "hevc", True)),
    "original": ("원본 그대로 (편집·보관용)", ("original", "crop", 0, "h264", False)),
}
RESOLUTIONS = {1080: "1080p", 720: "720p", 1440: "1440p", 2160: "2160p (4K)", 0: "원본 그대로"}  # 짧은 변
CODECS = {"h264": "MP4 · H.264 (어디서나 재생)", "hevc": "MP4 · H.265 (같은 화질에 더 작은 파일)"}
PREVIEW_SHORT = 800  # 저해상도 미리보기의 짧은 변
# 배포용: 화질 목표(cq/crf)로 — 인스타그램·유튜브가 다시 압축하므로 원본은 넉넉하게. GPU가 없으면 CPU 인코더
ENCODERS = {
    "h264": (
        ("h264_nvenc", {"preset": "p6", "rc": "vbr", "cq": "19", "b": "0", "maxrate": "60M", "bufsize": "120M"}),
        *hardware_encoders("h264", "30M"),
        ("libx264", {"preset": "medium", "crf": "18"}),
    ),
    "hevc": (
        ("hevc_nvenc", {"preset": "p6", "rc": "vbr", "cq": "22", "b": "0", "maxrate": "50M", "bufsize": "100M"}),
        *hardware_encoders("hevc", "20M"),
        ("libx265", {"preset": "fast", "crf": "22"}),
    ),
}
PREVIEW_ENCODERS = (
    ("h264_nvenc", {"preset": "p1", "rc": "vbr", "cq": "28", "b": "0", "maxrate": "20M", "bufsize": "40M", "bf": "0"}),
    *hardware_encoders("h264", "6M"),
    ("libx264", {"preset": "ultrafast", "crf": "28"}),
)


@dataclass(frozen=True)
class OutputFormat:
    ratio: str = "original"
    fit: str = "crop"
    short: int = 1080  # 짧은 변 px (0 = 원본 그대로)
    codec: str = "h264"
    loudness: bool = False  # 소리 크기 맞추기 (유튜브 기준 -14 LUFS) — 컷편집 타임라인 내보내기에서
    denoise: bool = False  # 잡음 줄이기 (바람·웅웅 소리)

    @property
    def tag(self) -> str:
        """파일 이름에 붙일 표시 (예: 9x16_1080p)."""
        ratio = "원본비율" if self.ratio == "original" else self.ratio.replace(":", "x")
        return f"{ratio}_{self.short}p" if self.short else ratio


def _even(value: float) -> int:
    return max(2, round(value / 2) * 2)


def target_size(source: tuple[int, int], fmt: OutputFormat, short: int | None = None) -> tuple[int, int]:
    """출력 가로·세로. short를 주면 그 짧은 변으로 (미리보기). 원본보다 크게 키우지는 않는다."""
    w, h = source
    rw, rh = RATIOS[fmt.ratio][1] or (w, h)
    side = short or fmt.short or min(w, h)
    if fmt.ratio == "original" or not fmt.short:
        side = min(side, min(w, h))  # 원본 비율이나 '원본 그대로'는 키우지 않는다
    if rw <= rh:
        return _even(side), _even(side * rh / rw)
    return _even(side * rw / rh), _even(side)


def fit_filters(source: tuple[int, int], target: tuple[int, int], fit: str) -> tuple[tuple[str, str], ...]:
    """source를 target 칸에: crop = 칸을 꽉 채우고 넘치는 가운데 바깥을 자름, pad = 다 보이게 줄이고 검은 여백."""
    (w, h), (tw, th) = source, target
    if abs(w / h - tw / th) < 0.01:
        return (("scale", f"{tw}:{th}"),)
    if fit == "crop":
        scale = max(tw / w, th / h)
        sw, sh = max(tw, _even(math.ceil(w * scale))), max(th, _even(math.ceil(h * scale)))
        return (("scale", f"{sw}:{sh}"), ("crop", f"{tw}:{th}:{(sw - tw) // 2}:{(sh - th) // 2}"))
    scale = min(tw / w, th / h)
    sw, sh = min(tw, _even(w * scale)), min(th, _even(h * scale))
    return (("scale", f"{sw}:{sh}"), ("pad", f"{tw}:{th}:{(tw - sw) // 2}:{(th - sh) // 2}:black"))


def export_plan(path: Path, fmt: OutputFormat, short: int | None = None) -> Plan:
    """원본에서 바로 가는 변환: (휴대폰이면 회전·색 변환) + 비율·크기 필터."""
    base = inspect(path)
    if base is not None:
        source, filters, fps = base.size, base.filters, base.fps
    else:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            source = (stream.codec_context.width, stream.codec_context.height)
            fps = max(1, round(float(stream.average_rate or stream.base_rate or 30)))
        filters = ()
    size = target_size(source, fmt)
    if short is not None:  # 미리보기: 최종과 같은 모양을 짧은 변 short까지만 (최종이 더 작으면 최종 크기)
        size = target_size(source, fmt, min(short, min(size)))
    return Plan((), (*filters, *fit_filters(source, size, fmt.fit)), fps, size)


def export_video(path: Path, output: Path, fmt: OutputFormat, subs: Subtitles | None, on_progress=None,
                 cancel: threading.Event | None = None, preview: bool = False, grade: Path | None = None) -> Plan:
    """path → output. 자막이 있으면 굽는다. grade = 색 보정 .cube (색 보정 탭). preview면 짧은 변 PREVIEW_SHORT·빠른 인코더.
    멈추면 Cancelled."""
    plan = export_plan(path, fmt, PREVIEW_SHORT if preview else None)
    burner = CaptionBurner(subs, plan.size, plan.fps) if subs is not None and subs.cues else None
    process = burner
    if grade is not None:
        from ks_studio.services.grade import LutFilter

        lut = LutFilter(grade)
        process = (lambda f: burner(lut.frame(f))) if burner is not None else lut.frame
    encoders = PREVIEW_ENCODERS if preview else ENCODERS[fmt.codec]
    make_optimized(path, output, plan, on_progress, process=process, encoders=encoders, cancel=cancel)
    return plan


AUDIO_BITRATE = 192_000


def export_timeline(project: Project, workspace: Workspace, sources: SourceCache, output: Path, fmt: OutputFormat,
                    on_progress=None, cancel: threading.Event | None = None, preview: bool = False,
                    with_subtitles: bool = True) -> tuple[int, int]:
    """컷편집 타임라인 → output. 타임라인을 그리면서 바로 비율·크기를 맞추고 자막을 굽는다 (중간 파일 없이 압축 한 번).

    자막은 project.subtitles (타임라인 시각). 돌려주는 값 = 출력 크기. 멈추면 Cancelled (쓰다 만 파일은 지운다).
    """
    fps = project.fps
    canvas = canvas_size(project, workspace, sources)
    size = target_size(canvas, fmt)
    if preview:
        size = target_size(canvas, fmt, min(PREVIEW_SHORT, min(size)))
    # 그리는 크기: 출력을 채우는 만큼만 (자르기면 긴 쪽 기준, 여백이면 짧은 쪽 기준) — 원본보다 크게는 안 그린다
    pick = max if fmt.fit == "crop" else min
    scale = min(1.0, pick(size[0] / canvas[0], size[1] / canvas[1]))
    drawn = (_even(canvas[0] * scale), _even(canvas[1] * scale))
    renderer = TimelineRenderer(project, workspace, sources, drawn, "yuv420p")
    subs = project.subtitles if with_subtitles and project.subtitles.cues else None
    burner = OverlayBurner(subs, project.texts, size, fps) if subs is not None or project.texts else None  # 글자 + 자막
    post = AudioPost(fmt.denoise, fmt.loudness) if fmt.loudness or fmt.denoise else None
    written = [0]  # 인코더에 넘긴 소리 샘플 수 (다듬은 소리는 늦게·다른 크기로 나온다)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".part")
    try:
        with av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart"}) as dst:
            video = add_video_stream(dst, size, fps, PREVIEW_ENCODERS if preview else ENCODERS[fmt.codec])
            audio = dst.add_stream("aac", rate=AUDIO_RATE, layout="mono")
            audio.bit_rate = AUDIO_BITRATE
            graph = av.filter.Graph()
            nodes = [graph.add_buffer(width=drawn[0], height=drawn[1], format="yuv420p", time_base=Fraction(1, fps))]
            for name, args in (*fit_filters(drawn, size, fmt.fit), ("format", "yuv420p")):
                nodes.append(graph.add(name, args))
            nodes.append(graph.add("buffersink"))
            graph.link_nodes(*nodes).configure()

            def drain() -> None:
                while True:
                    try:
                        frame = graph.pull()
                    except (av.BlockingIOError, av.EOFError):
                        return
                    pts = frame.pts
                    if burner is not None:
                        frame = burner(frame)
                    frame.pts, frame.time_base = pts, Fraction(1, fps)
                    dst.mux(video.encode(frame))

            def encode_audio(samples: np.ndarray) -> None:
                chunk = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples.reshape(1, -1), np.float32), format="flt", layout="mono")
                chunk.sample_rate, chunk.pts, chunk.time_base = AUDIO_RATE, written[0], Fraction(1, AUDIO_RATE)
                written[0] += len(samples)
                dst.mux(audio.encode(chunk))

            total = max(1, project.duration)
            for t, frame, samples in renderer.frames(0):
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                frame.pts, frame.time_base = t, Fraction(1, fps)
                graph.push(frame)
                drain()
                for part in (post.push(samples) if post is not None else [samples]):
                    encode_audio(part)
                if on_progress and t % 15 == 0:
                    on_progress(min(1.0, t / total))
            graph.push(None)
            drain()
            for part in (post.flush() if post is not None else []):
                encode_audio(part)
            dst.mux(video.encode(None))
            dst.mux(audio.encode(None))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, output)
    return size


class ExportWorker(QObject):
    """내보내기를 스레드 하나에서 (GPU 인코더를 나눠 쓰므로 한 번에 하나). kind = "preview" 또는 "final"."""

    progress = Signal(str, float)
    finished = Signal(str, Path)
    failed = Signal(str, str)
    cancelled = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs: queue.Queue = queue.Queue()
        self._cancel = threading.Event()
        self.busy: str | None = None
        threading.Thread(target=self._worker, daemon=True, name="export").start()

    def request(self, kind: str, path: Path, output: Path, fmt: OutputFormat, subs: Subtitles | None,
                grade: Path | None = None) -> None:
        """영상 파일 하나를 내보낸다 (grade = 색 보정 .cube)."""
        self.request_job(kind, output, lambda progress, cancel: export_video(path, output, fmt, subs, progress, cancel,
                                                                            kind == "preview", grade))

    def request_timeline(self, kind: str, project: Project, workspace: Workspace, sources: SourceCache,
                         output: Path, fmt: OutputFormat, with_subtitles: bool) -> None:
        """컷편집 타임라인을 내보낸다 (자막을 굽는다)."""
        self.request_job(kind, output, lambda progress, cancel: export_timeline(
            project, workspace, sources, output, fmt, progress, cancel, kind == "preview", with_subtitles))

    def request_job(self, kind: str, output: Path, run) -> None:
        self.busy = kind
        self._cancel.clear()
        self._jobs.put((kind, output, run))

    def cancel(self) -> None:
        self._cancel.set()

    def _worker(self) -> None:
        while True:
            kind, output, run = self._jobs.get()
            try:
                run(lambda r, k=kind: self.progress.emit(k, r), self._cancel)
            except Cancelled:
                self.busy = None
                self.cancelled.emit(kind)
                continue
            except Exception as e:
                log.exception("내보내기 실패: %s", output)
                self.busy = None
                self.failed.emit(kind, str(e))
                continue
            self.busy = None
            self.finished.emit(kind, output)
