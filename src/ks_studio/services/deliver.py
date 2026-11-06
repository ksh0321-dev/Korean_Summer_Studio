"""최종 내보내기: 영상 파일 하나 → 고른 비율·해상도·형식의 배포용 MP4 (+ 자막 굽기). 영상 편집의 마지막 단계.

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

import av
from PySide6.QtCore import QObject, Signal

from ks_studio.core.subtitles import Subtitles
from ks_studio.services.captions import CaptionBurner
from ks_studio.services.optimize import Cancelled, Plan, inspect, make_optimized

log = logging.getLogger(__name__)

RATIOS = {  # 칸 값 → (이름, 가로:세로 또는 None = 원본 비율)
    "original": ("원본 비율", None),
    "9:16": ("9:16 세로 (릴스·쇼츠·틱톡)", (9, 16)),
    "4:5": ("4:5 세로 (인스타 피드)", (4, 5)),
    "1:1": ("1:1 정사각형", (1, 1)),
    "16:9": ("16:9 가로 (유튜브)", (16, 9)),
}
FITS = {"crop": "가운데 자르기 (화면 꽉 차게)", "pad": "검은 여백 (자르지 않게)"}
RESOLUTIONS = {1080: "1080p", 720: "720p", 1440: "1440p", 2160: "2160p (4K)", 0: "원본 그대로"}  # 짧은 변
CODECS = {"h264": "MP4 · H.264 (어디서나 재생)", "hevc": "MP4 · H.265 (같은 화질에 더 작은 파일)"}
PREVIEW_SHORT = 800  # 저해상도 미리보기의 짧은 변
# 배포용: 화질 목표(cq/crf)로 — 인스타그램·유튜브가 다시 압축하므로 원본은 넉넉하게. GPU가 없으면 CPU 인코더
ENCODERS = {
    "h264": (
        ("h264_nvenc", {"preset": "p6", "rc": "vbr", "cq": "19", "b": "0", "maxrate": "60M", "bufsize": "120M"}),
        ("libx264", {"preset": "medium", "crf": "18"}),
    ),
    "hevc": (
        ("hevc_nvenc", {"preset": "p6", "rc": "vbr", "cq": "22", "b": "0", "maxrate": "50M", "bufsize": "100M"}),
        ("libx265", {"preset": "fast", "crf": "22"}),
    ),
}
PREVIEW_ENCODERS = (
    ("h264_nvenc", {"preset": "p1", "rc": "vbr", "cq": "28", "b": "0", "maxrate": "20M", "bufsize": "40M", "bf": "0"}),
    ("libx264", {"preset": "ultrafast", "crf": "28"}),
)


@dataclass(frozen=True)
class OutputFormat:
    ratio: str = "original"
    fit: str = "crop"
    short: int = 1080  # 짧은 변 px (0 = 원본 그대로)
    codec: str = "h264"

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
                 cancel: threading.Event | None = None, preview: bool = False) -> Plan:
    """path → output. 자막이 있으면 굽는다. preview면 짧은 변 PREVIEW_SHORT·빠른 인코더. 멈추면 Cancelled."""
    plan = export_plan(path, fmt, PREVIEW_SHORT if preview else None)
    burner = CaptionBurner(subs, plan.size, plan.fps) if subs is not None and subs.cues else None
    encoders = PREVIEW_ENCODERS if preview else ENCODERS[fmt.codec]
    make_optimized(path, output, plan, on_progress, process=burner, encoders=encoders, cancel=cancel)
    return plan


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

    def request(self, kind: str, path: Path, output: Path, fmt: OutputFormat, subs: Subtitles | None) -> None:
        self.busy = kind
        self._cancel.clear()
        self._jobs.put((kind, path, output, fmt, subs))

    def cancel(self) -> None:
        self._cancel.set()

    def _worker(self) -> None:
        while True:
            kind, path, output, fmt, subs = self._jobs.get()
            try:
                export_video(path, output, fmt, subs, lambda r: self.progress.emit(kind, r), self._cancel, kind == "preview")
            except Cancelled:
                self.busy = None
                self.cancelled.emit(kind)
                continue
            except Exception as e:
                log.exception("내보내기 실패: %s", path)
                self.busy = None
                self.failed.emit(kind, str(e))
                continue
            self.busy = None
            self.finished.emit(kind, output)
