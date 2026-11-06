"""편집용 사본 (DaVinci Resolve의 '최적화 미디어'와 같은 개념).

휴대폰 원본은 그대로 편집하기 곤란하다: 세로 회전 정보, HDR(HLG)·넓은 색역(P3/BT.2020)·풀 레인지·10bit, 가변 프레임(VFR).
가져올 때 백그라운드에서 '편집하기 좋은 사본'을 만들고(위치: sources.optimized_path), 미리보기·내보내기는 사본을 읽는다 (SourceCache.media).
project.json에는 원본 경로가 그대로 남고, 사본은 프로젝트 cache/optimized/에 있어 지워도 다시 만들 수 있다.

사본: H.264 8bit 4:2:0, BT.709 SDR, 회전 적용, 30/60fps 고정(CFR), AAC 48kHz. 위치 정보 등 메타데이터는 옮기지 않는다.
디코딩·회전·색 변환·fps 고정은 CPU, 인코딩은 GPU(NVENC, 없으면 CPU).
- 디코딩은 따로 스레드에서 해서 필터·인코딩과 겹쳐 돈다 (단계를 차례로 하면 세 단계 시간을 다 더한 만큼 걸렸다)
- 필터는 스레드 FILTER_THREADS개 (기본 1개였다 — 실측 갤럭시 HLG 프레임당 15.9 → 8.6ms)
- GPU 디코딩(NVDEC)은 쓰지 않는다: 디코딩 자체는 3배 빨랐지만 색 변환이 CPU라 프레임을 내려받고 형식을 바꾸는
  비용이 더 커서 전체는 오히려 느렸다 (실측 갤럭시 13초 영상 5.0 → 8.7초). 이 FFmpeg엔 HDR 색 변환 GPU 필터가 없다
10bit만 다르고 색이 BT.709 SDR인 영상은 사본을 만들지 않는다 — 렌더러가 그대로 읽을 수 있다
(실측: 20분 4K 10bit 영상 하나로 8분·13GB를 썼다).
HLG는 SDR 화면과 호환되게 설계된 곡선이라 BT.2020 SDR 감마로 근사해도 자연스럽다 (실측: 갤럭시 HLG).
PQ(HDR10)는 같은 근사를 쓰면 어둡게 나올 수 있다 → 이유에 '근사'라고 표시한다.
"""

import logging
import os
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QObject, Signal

from ks_studio.services.sources import AUDIO_RATE

log = logging.getLogger(__name__)

DECODE_QUEUE = 16  # 디코딩 스레드가 앞서 풀어 둘 프레임 수
FILTER_THREADS = 4  # 실측: 1 → 15.9ms, 4 → 8.6ms, 8 → 9.9ms (디코딩과 코어를 나눠 쓴다)
VFR_SAMPLE_PACKETS = 400
VFR_TOLERANCE = 0.25  # 공칭 프레임 간격에서 이만큼(비율) 넘게 벗어난 간격을 '불규칙'으로 센다
VFR_IRREGULAR_RATIO = 0.01
# 화질 우선 (내보내기도 사본을 읽는다). NVIDIA GPU가 없거나 NVENC가 못 여는 크기면 CPU(libx264)로
# 비트레이트 목표(b)와 상한(maxrate)을 풀지 않으면 NVENC가 기본값에 묶여 cq를 낮춰도 화질이 그대로였다
# (실측: cq 18·14·10 모두 PSNR 42.7~42.9dB). 풀면 cq 16 48.2 · 14 49.8 · 12 51.4dB (원본 대비 크기 0.9·1.0·1.3배).
# 편집용 사본·AI 보정본은 중간 파일이라 화질 우선 → 51dB = 눈으로 구분 불가
ENCODERS = (
    ("h264_nvenc", {"preset": "p5", "rc": "vbr", "cq": "12", "b": "0", "maxrate": "400M", "bufsize": "800M", "bf": "0"}),
    ("libx264", {"preset": "veryfast", "crf": "12", "bf": "0"}),
)
AUDIO_BITRATE = 256_000

# FFmpeg 색 번호 → colorspace 필터 이름 (모르는 값·미지정은 BT.709로 본다)
PRIMARIES = {1: "bt709", 5: "bt470bg", 6: "smpte170m", 9: "bt2020", 11: "smpte431", 12: "smpte432"}
TRANSFERS = {1: "bt709", 6: "smpte170m", 13: "iec61966-2-1", 14: "bt2020-10", 15: "bt2020-12", 16: "bt2020-10", 18: "bt2020-10"}
MATRICES = {1: "bt709", 5: "bt470bg", 6: "smpte170m", 9: "bt2020ncl"}
HDR = {16: "PQ", 18: "HLG"}
WIDE_GAMUT = {9: "BT.2020", 11: "P3", 12: "P3"}


class Cancelled(Exception):
    """make_optimized를 cancel 신호로 멈췄다 (쓰다 만 파일은 지웠다)."""


@dataclass(frozen=True)
class Plan:
    reasons: tuple[str, ...]  # 사람이 읽는 이유 ("세로 회전", "HDR(HLG)→SDR", ...)
    filters: tuple[tuple[str, str], ...]  # (필터 이름, 인자). fps·format은 make_optimized가 붙인다
    fps: int  # 사본 fps (30 또는 60)
    size: tuple[int, int]  # 사본 가로·세로 (회전 적용 후)


def inspect(path: Path) -> Plan | None:
    """사본이 필요하면 Plan, 그대로 써도 되면 None. 읽을 수 없어도 None (가져오기에서 걸러진다)."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            pts = sorted(p.pts for p in islice(container.demux(stream), VFR_SAMPLE_PACKETS) if p.pts is not None)
            time_base, rate = stream.time_base, float(stream.average_rate or stream.base_rate or 30)
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            cc = stream.codec_context
            first = next(container.decode(stream))
    except (av.FFmpegError, StopIteration, OSError):
        return None

    reasons, filters = [], []
    rotation = round(getattr(first, "rotation", 0) or 0) % 360  # 표시 행렬 각도 (반시계 방향)
    width, height = cc.width, cc.height
    if rotation in (90, 270):
        filters.append(("transpose", "cclock" if rotation == 90 else "clock"))
        width, height = height, width
        reasons.append("세로 회전")
    elif rotation == 180:
        filters += [("hflip", ""), ("vflip", "")]
        reasons.append("뒤집힘")

    primaries, transfer, matrix = int(cc.color_primaries or 2), int(cc.color_trc or 2), int(cc.colorspace or 2)
    full_range = int(cc.color_range or 0) == 2 or first.format.name.startswith("yuvj")
    deep = max(c.bits for c in first.format.components) > 8
    if transfer in HDR:
        reasons.append(f"HDR({HDR[transfer]})→SDR" + (" 근사" if transfer == 16 else ""))
    elif primaries in WIDE_GAMUT:
        reasons.append(f"{WIDE_GAMUT[primaries]} 색역→BT.709")
    if full_range:
        reasons.append("풀 레인지")
    if transfer in HDR or primaries in WIDE_GAMUT or full_range:
        options = (
            f"all=bt709:iprimaries={PRIMARIES.get(primaries, 'bt709')}:itrc={TRANSFERS.get(transfer, 'bt709')}"
            f":ispace={MATRICES.get(matrix, 'bt709')}:irange={'pc' if full_range else 'tv'}:range=tv:format=yuv420p"
        )
        filters.append(("colorspace", options))

    if _is_vfr(pts, time_base, rate):
        reasons.append("가변 프레임(VFR)")
    if not reasons:
        return None
    if deep:  # 다른 이유로 사본을 만들 때만 같이 8bit로 (format 필터가 맡는다)
        reasons.append("10bit")
    return Plan(tuple(reasons), tuple(filters), 60 if rate > 45 else 30, (width, height))


def _is_vfr(pts: list[int], time_base, rate: float) -> bool:
    if len(pts) < 3:
        return False
    deltas = np.diff(np.array(pts, dtype=np.float64)) * float(time_base)
    nominal = 1 / rate
    irregular = np.count_nonzero(np.abs(deltas - nominal) > nominal * VFR_TOLERANCE)
    return irregular > max(2, VFR_IRREGULAR_RATIO * len(deltas))


def make_optimized(
    path: Path,
    output: Path,
    plan: Plan,
    on_progress: Callable[[float], None] | None = None,
    process: Callable[[av.VideoFrame], av.VideoFrame] | None = None,
    encoders: tuple = ENCODERS,
    cancel: threading.Event | None = None,
) -> None:
    """사본을 만든다. 실패하면 쓰다 만 파일을 지우고 예외를 그대로 올린다.

    영상 첫 프레임을 0초로 맞추고, 오디오는 그 기준으로 앞에 무음을 넣거나 앞을 버려 싱크를 유지한다.
    process가 있으면 변환을 마친 프레임마다 거친다 (pts = 사본의 프레임 번호). AI 보정이 휴대폰 원본에서 회전·색 변환과
    보정을 한 번의 압축으로 끝낼 때 쓴다 — 편집용 사본을 다시 압축하면 화질이 한 번 더 떨어진다.
    내보내기 탭도 같은 길을 쓴다: Plan에 자르기·크기 필터를, process에 자막 굽기를, encoders에 배포용 인코더를 준다.
    cancel이 켜지면 Cancelled를 올린다.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".part")
    try:
        _transcode(path, tmp, plan, on_progress, process, encoders, cancel)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, output)


def _transcode(path, tmp, plan: Plan, on_progress, process=None, encoders=ENCODERS, cancel=None) -> None:
    with av.open(str(path)) as probe:  # 스트림 정보만
        v_info = probe.streams.video[0]
        time_base = v_info.time_base
        channels = probe.streams.audio[0].codec_context.channels if probe.streams.audio else 0
        total = probe.duration / 1_000_000 if probe.duration else None

    frames: queue.Queue = queue.Queue(maxsize=DECODE_QUEUE)
    stop = threading.Event()
    reader = threading.Thread(target=_decode, args=(path, frames, stop), daemon=True, name="optimize-decode")
    with av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart"}) as dst:
        name, options = video_encoder(plan.size, plan.fps, encoders)
        v_out = dst.add_stream(name, rate=plan.fps, options={**options, "g": str(max(1, plan.fps // 2))})
        if "hevc" in name or "265" in name:
            v_out.codec_context.codec_tag = "hvc1"  # 아이폰·맥이 MP4 H.265를 틀려면 이 표시여야 한다
        v_out.width, v_out.height = plan.size
        v_out.pix_fmt = "yuv420p"
        v_out.time_base = Fraction(1, plan.fps)
        audio = _AudioCopy(dst, channels) if channels else None
        graph = None
        origin = None  # 영상 첫 프레임 시각 (초). 사본에서는 이 순간이 0초
        reader.start()
        try:
            while (item := frames.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                kind, frame = item
                if kind == "audio":
                    if audio is not None:
                        audio.add(frame, origin)
                    continue
                if graph is None:
                    graph = _graph(frame, time_base, plan)
                    origin = frame.time
                frame.pts -= round(origin / time_base)
                graph.push(frame)
                _drain(graph, v_out, dst, plan.fps, process)
                if on_progress and total:
                    on_progress(min(1.0, (frame.time or 0) / total))
        finally:
            stop.set()
            reader.join()
        if graph is not None:
            graph.push(None)
            _drain(graph, v_out, dst, plan.fps, process)
        dst.mux(v_out.encode(None))
        if audio is not None:
            audio.finish()


def _decode(path: Path, frames: queue.Queue, stop: threading.Event) -> None:
    """영상·소리 프레임을 차례로 큐에 넣는다. 끝나면 None, 실패하면 예외를 넣는다."""

    def put(item) -> bool:
        while not stop.is_set():
            try:
                frames.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    try:
        with av.open(str(path)) as src:
            video = src.streams.video[0]
            video.thread_type = "AUTO"
            streams = [video, *src.streams.audio[:1]]
            for packet in src.demux(*streams):
                kind = "video" if packet.stream is video else "audio"
                for frame in packet.decode():  # 끝 표시 빈 패킷도 디코딩해야 남은 프레임이 나온다
                    if not put((kind, frame)):
                        return
        put(None)
    except BaseException as e:  # 받는 쪽에서 다시 올린다
        put(e)


def _graph(first: av.VideoFrame, time_base, plan: Plan) -> av.filter.Graph:
    graph = av.filter.Graph()
    graph.threads = FILTER_THREADS
    nodes = [graph.add_buffer(width=first.width, height=first.height, format=first.format.name, time_base=time_base)]
    for name, args in (*plan.filters, ("fps", str(plan.fps)), ("format", "yuv420p")):
        nodes.append(graph.add(name, args))
    nodes.append(graph.add("buffersink"))
    graph.link_nodes(*nodes).configure()
    return graph


def video_encoder(size: tuple[int, int], fps: float, encoders: tuple = ENCODERS) -> tuple[str, dict[str, str]]:
    """먼저 열어 보고 되는 인코더를 차례로 고른다 (GPU → CPU). 기본은 고화질 H.264 (AI 보정본도 같이 쓴다)."""
    for name, options in encoders:
        try:
            ctx = av.CodecContext.create(name, "w")
            ctx.width, ctx.height = size
            ctx.pix_fmt = "yuv420p"
            ctx.time_base = Fraction(1, round(fps))
            ctx.options = dict(options)
            ctx.open()
            return name, options
        except av.FFmpegError:
            log.info("%s를 쓸 수 없어 다음 인코더로", name)
    raise RuntimeError(f"영상 인코더를 열 수 없습니다 ({', '.join(name for name, _ in encoders)})")


def _drain(graph: av.filter.Graph, v_out, dst, fps: int, process=None) -> None:
    while True:
        try:
            frame = graph.pull()
        except (av.BlockingIOError, av.EOFError):
            return
        frame.time_base = Fraction(1, fps)
        if process is not None:
            pts = frame.pts
            frame = process(frame)
            frame.pts, frame.time_base = pts, Fraction(1, fps)
        dst.mux(v_out.encode(frame))


class _AudioCopy:
    """오디오를 48kHz AAC로. 영상 첫 프레임(origin)보다 먼저 시작한 소리는 버리고, 늦게 시작하면 앞에 무음을 넣는다."""

    def __init__(self, dst, channels: int) -> None:
        layout = "stereo" if channels >= 2 else "mono"
        self._dst = dst
        self._out = dst.add_stream("aac", rate=AUDIO_RATE, layout=layout)
        self._out.bit_rate = AUDIO_BITRATE
        self._resampler = av.AudioResampler(format="fltp", layout=layout, rate=AUDIO_RATE)
        self._layout = layout
        self._pts = 0
        self._pending: list[av.AudioFrame] = []  # 영상 첫 프레임 전에 온 소리 (origin을 알 때까지 보관)
        self._aligned = False
        self._skip = 0  # 앞에서 버릴 샘플 수

    def add(self, frame: av.AudioFrame, origin: float | None) -> None:
        self._pending.append(frame)
        if origin is None:
            return
        for pending in self._pending:
            if not self._aligned:
                self._align(pending, origin)
            for resampled in self._resampler.resample(pending):
                self._encode(resampled)
        self._pending.clear()

    def _align(self, frame: av.AudioFrame, origin: float) -> None:
        self._aligned = True
        offset = round(((frame.time or 0) - origin) * AUDIO_RATE)
        if offset > 0:  # 소리가 늦게 시작 → 앞에 무음
            channels = 2 if self._layout == "stereo" else 1
            self._encode(self._frame(np.zeros((channels, offset), np.float32)))
        else:
            self._skip = -offset  # 소리가 먼저 시작 → 앞을 버림

    def _frame(self, samples: np.ndarray) -> av.AudioFrame:
        frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples, dtype=np.float32), format="fltp", layout=self._layout)
        frame.sample_rate = AUDIO_RATE
        return frame

    def _encode(self, frame: av.AudioFrame) -> None:
        if self._skip:
            samples = frame.to_ndarray()
            if samples.shape[1] <= self._skip:
                self._skip -= samples.shape[1]
                return
            frame, self._skip = self._frame(samples[:, self._skip :]), 0
        frame.pts, frame.time_base = self._pts, Fraction(1, AUDIO_RATE)
        self._pts += frame.samples
        self._dst.mux(self._out.encode(frame))

    def finish(self) -> None:
        for resampled in self._resampler.resample(None):
            self._encode(resampled)
        self._dst.mux(self._out.encode(None))


class Optimizer(QObject):
    """편집용 사본을 백그라운드 스레드 하나에서 차례로 만든다 (GPU 인코더를 녹화·내보내기와 나눠 쓰므로 한 번에 하나)."""

    progress = Signal(Path, float)  # 원본, 0~1
    finished = Signal(Path, Path)  # 원본, 사본
    failed = Signal(Path, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._jobs: queue.Queue = queue.Queue()
        self._queued: set[Path] = set()
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True, name="optimize").start()

    def request(self, path: Path, output: Path, plan: Plan) -> None:
        with self._lock:
            if path in self._queued:
                return
            self._queued.add(path)
        self._jobs.put((path, output, plan))

    def is_busy(self, path: Path) -> bool:
        with self._lock:
            return path in self._queued

    def _worker(self) -> None:
        while True:
            path, output, plan = self._jobs.get()
            try:
                make_optimized(path, output, plan, lambda ratio: self.progress.emit(path, ratio))
            except Exception as e:  # 원본 손상·GPU 인코더 오류 등
                log.exception("편집용 사본 만들기 실패: %s", path)
                self._done(path)
                self.failed.emit(path, str(e))
                continue
            self._done(path)
            self.finished.emit(path, output)

    def _done(self, path: Path) -> None:
        with self._lock:
            self._queued.discard(path)
