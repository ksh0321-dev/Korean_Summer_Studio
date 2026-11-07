"""B 녹화 엔진: 카메라 프레임과 마이크 샘플을 따로 받아 하나의 MKV(H.264 NVENC + PCM)로 쓴다.

- 영상: 카메라가 켜진 순간부터 1/fps 칸에 배치 (core.cfr) → 고정 프레임레이트(CFR) 보장.
  녹화는 그 칸 중 '녹화 후 첫 프레임 칸'을 0번으로 삼는다.
  프레임 시각은 Qt가 찍은 startTime을 쓴다. 우리 콜백에서 찍으면 GIL 등으로 16~31ms 흔들린다 (실측).
  startTime은 PC 시계와 같은 속도로 가므로(60초 실측 1ms 이내), 차이의 최솟값으로 PC 시계로 바꾼다.
- 오디오: 샘플 번호로 끊김 없이 이어 붙이고, 영상 0번 칸 시각에 맞춰 앞을 자르거나 무음으로 채운다.
- MKV라서 강제 종료돼도 직전 클러스터(약 1초 전)까지 살아남는다 → media.finalize_recording이 MP4로 만든다.
- 인코딩·파일 쓰기는 별도 스레드 하나가 전담 (PyAV 컨테이너는 스레드 하나에서만 써야 함).
"""

import logging
import queue
import threading
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import QVideoFrame, QVideoFrameFormat, QVideoSink

from ks_studio.core.cfr import FrameSlotter
from ks_studio.services.capture import MicInput
from ks_studio.services.media import AUDIO_DELAY_TAG
from ks_studio.services.accel import hardware_encoders
from ks_studio.services.optimize import video_encoder

log = logging.getLogger(__name__)

VIDEO_OPTIONS = {"preset": "p5", "rc": "vbr", "bf": "0"}
VIDEO_MBPS_AT_30FPS = (25, 30)  # (평균, 최대). 60fps는 프레임이 2배라 비트레이트도 fps에 비례해 올린다
AUDIO_LAYOUTS = {1: "mono", 2: "stereo"}
SETTLE_SECONDS = 2  # 칸 기준점을 맞추는 시간 (미리보기 중에 끝난다)


def _to_nv12(frame: QVideoFrame) -> av.VideoFrame:
    """카메라 프레임을 복사해 NV12 PyAV 프레임으로. NV12가 아니면 RGB를 거쳐 변환 (느림)."""
    if frame.pixelFormat() == QVideoFrameFormat.PixelFormat.Format_NV12 and frame.map(QVideoFrame.MapMode.ReadOnly):
        try:
            w, h = frame.width(), frame.height()
            y = np.frombuffer(frame.bits(0), np.uint8).reshape(-1, frame.bytesPerLine(0))[:h, :w]
            uv = np.frombuffer(frame.bits(1), np.uint8).reshape(-1, frame.bytesPerLine(1))[: h // 2, :w]
            return av.VideoFrame.from_numpy_buffer(np.concatenate([y, uv]), format="nv12")
        finally:
            frame.unmap()
    image = frame.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    rgb = np.frombuffer(image.constBits(), np.uint8).reshape(h, image.bytesPerLine())[:, : w * 3].reshape(h, w, 3)
    return av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="nv12")


class Recorder(QObject):
    stopped = Signal(Path)  # MKV를 정상적으로 닫음 (쓰기 스레드에서 보냄)
    failed = Signal(str)

    def __init__(self, sink: QVideoSink, mic: MicInput, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._mic = mic
        self._slotter: FrameSlotter | None = None
        self._clock_offsets: deque[float] = deque(maxlen=60)  # PC 시계 - startTime (최근 2초)
        self._queue: queue.Queue | None = None
        self._start_slot: int | None = None
        self._thread: threading.Thread | None = None
        # 카메라 스레드에서 바로 받는다 → UI가 바빠도 도착 시각이 밀리지 않는다
        sink.videoFrameChanged.connect(self._on_frame, Qt.ConnectionType.DirectConnection)
        mic.samples.connect(self._on_samples)

    @property
    def is_recording(self) -> bool:
        return self._queue is not None

    def reset(self, fps: int) -> None:
        """카메라를 (다시) 켤 때 호출. 칸 배치를 처음부터 맞춘다."""
        self._slotter = FrameSlotter(fps, settle_frames=SETTLE_SECONDS * fps)
        self._clock_offsets.clear()

    def start(self, path: Path, fps: int, size: tuple[int, int], audio_delay_ms: int = 0) -> None:
        """audio_delay_ms는 MKV에 기록만 하고, 마무리(media.finalize_recording)할 때 적용한다 → 복구할 때도 같은 값."""
        self._start_slot = None
        q = queue.Queue()
        self._thread = threading.Thread(
            target=self._write,
            args=(q, path, fps, size, self._mic.clock_origin, self._mic.sample_rate, self._mic.channels, audio_delay_ms),
            daemon=True,
        )
        self._thread.start()
        self._queue = q

    def stop(self) -> None:
        q, self._queue = self._queue, None
        if q is not None:
            q.put(None)

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _on_frame(self, frame: QVideoFrame) -> None:
        slotter = self._slotter
        if slotter is None or not frame.isValid():
            return
        arrival = time.monotonic()
        stamp = frame.startTime() / 1_000_000 if frame.startTime() >= 0 else arrival
        self._clock_offsets.append(arrival - stamp)
        slots = slotter.place(stamp)
        q = self._queue
        if q is None or not slots:
            return
        if self._start_slot is None:  # 녹화 후 첫 프레임: 이 칸이 0번
            self._start_slot = slots[-1]
            slots = slots[-1:]
            # 도착은 늦어질 수만 있으므로 차이의 최솟값이 실제 환산값에 가장 가깝다
            q.put(("start", slotter.slot_time(self._start_slot) + min(self._clock_offsets), None))
        q.put(("video", [s - self._start_slot for s in slots], (_to_nv12(frame), slotter.dropped)))

    def _on_samples(self, samples: np.ndarray, first_index: int) -> None:
        q = self._queue
        if q is not None:
            q.put(("audio", first_index, samples))

    def _write(self, q: queue.Queue, path: Path, *args) -> None:
        try:
            stats = _write_mkv(q, path, *args)
        except Exception as e:
            log.exception("녹화 쓰기 실패: %s", path)
            self._queue = None
            self.failed.emit(str(e))
            return
        log.info("녹화 닫음 %s: 프레임 %d · 복제 %d · 버림 %d · 오디오 %.2fs", path.name, *stats)
        self.stopped.emit(path)


def _write_mkv(q, path, fps, size, mic_origin, sample_rate, channels, audio_delay_ms):
    container = av.open(str(path), "w", format="matroska", options={"cluster_time_limit": "1000", "flush_packets": "1"})
    container.metadata[AUDIO_DELAY_TAG] = str(audio_delay_ms)
    try:
        average, peak = (round(mbps * fps / 30) for mbps in VIDEO_MBPS_AT_30FPS)
        options = {**VIDEO_OPTIONS, "b": f"{average}M", "maxrate": f"{peak}M", "g": str(fps)}
        # NVIDIA(NVENC) → 맥(VideoToolbox) → CPU(libx264) 중 되는 것 (NVIDIA가 없는 PC·맥에서도 돈다)
        candidates = (("h264_nvenc", options), *hardware_encoders("h264", f"{average}M"),
                      ("libx264", {"preset": "veryfast", "crf": "18", "g": str(fps)}))
        name, chosen = video_encoder(size, fps, candidates, software_ok=False)  # 녹화는 늘 GPU 먼저 (실시간)
        video = container.add_stream(name, rate=fps, options=chosen)
        video.width, video.height = size
        video.pix_fmt = "nv12"
        video.time_base = Fraction(1, fps)
        audio = container.add_stream("pcm_s16le", rate=sample_rate, layout=AUDIO_LAYOUTS[channels])

        audio_start: int | None = None
        early_audio: list[tuple[int, np.ndarray]] = []  # 영상 0번 칸이 정해지기 전에 온 오디오
        last = None
        frames = duplicated = audio_written = 0
        dropped_first = dropped_last = None
        while (item := q.get()) is not None:
            kind, a, b = item
            if kind == "start":
                audio_start = round((a - mic_origin) * sample_rate)
                for first_index, samples in early_audio:
                    audio_written = _write_audio(container, audio, samples, first_index - audio_start, audio_written, sample_rate, channels)
                early_audio.clear()
            elif kind == "video":
                frame, dropped_last = b
                dropped_first = dropped_last if dropped_first is None else dropped_first
                filler = frame if last is None else last
                for slot in a[:-1]:
                    _encode(container, video, filler, slot)
                _encode(container, video, frame, a[-1])
                frames += 1
                duplicated += len(a) - 1
                last = frame
            elif audio_start is None:
                early_audio.append((a, b))
            else:
                audio_written = _write_audio(container, audio, b, a - audio_start, audio_written, sample_rate, channels)
        container.mux(video.encode())
        container.mux(audio.encode())
    finally:
        container.close()
    dropped = (dropped_last - dropped_first) if dropped_first is not None else 0
    return frames, duplicated, dropped, audio_written / sample_rate


def _encode(container, stream, frame: av.VideoFrame, pts: int) -> None:
    frame.pts = pts
    container.mux(stream.encode(frame))


def _write_audio(container, stream, samples: np.ndarray, rel: int, written: int, sample_rate: int, channels: int) -> int:
    """rel = 영상 0번 칸 기준 첫 샘플 위치. 그 전 샘플은 자르고, 빈 구간은 무음으로 채운다. 쓴 끝 위치를 돌려준다."""
    if rel + len(samples) <= written:
        return written
    if rel < written:
        samples = samples[written - rel :]
    elif rel > written:
        samples = np.concatenate([np.zeros((rel - written, channels), np.float32), samples])
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout=AUDIO_LAYOUTS[channels])
    frame.sample_rate = sample_rate
    frame.pts = written
    container.mux(stream.encode(frame))
    return written + len(samples)
