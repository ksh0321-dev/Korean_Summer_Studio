"""편집에 쓰는 원본 미디어: 정보·오디오·파형·썸네일. 무거운 계산은 한 번만 하고 프로젝트 cache/에 저장한다.

- info: 바로 계산 (파일 머리만 읽음)
- audio: 원본 오디오 전체를 48kHz 모노 int16으로 (12분 ≈ 70MB). 렌더 스레드가 기다려서 받는다
- peaks / thumbnails: 화면용. 없으면 백그라운드에서 만들고 ready 신호를 보낸다
- 편집용 사본(services.optimize)이 있으면 원본 대신 사본을 읽는다 (media). 캐시도 사본 기준이라 사본이 생기면 저절로 새로 만든다
"""

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QObject, Signal

from ks_studio.services.foreground import yield_to_playback
from ks_studio.services.yuv import to_rgb

log = logging.getLogger(__name__)

AUDIO_RATE = 48000
PEAK_WINDOW = AUDIO_RATE // 100  # 파형: 10ms마다 최대 진폭 하나
THUMB_SIZE = (128, 72)
OPTIMIZED_DIR = "optimized"
OPTIMIZED_TAG = "_h265"  # 사본 형식이 바뀌면 이 표시를 바꾼다 → 예전 사본은 쓰지 않는 파일이 되어 정리되고 새로 만들어진다


@dataclass(frozen=True)
class MediaInfo:
    frames: int
    fps: float
    width: int
    height: int
    has_audio: bool

    @property
    def duration(self) -> float:
        return self.frames / self.fps


@dataclass(frozen=True)
class Thumbnails:
    times: np.ndarray  # 각 썸네일의 원본 시각(초), 오름차순
    images: np.ndarray  # (N, 72, 128, 3) RGB


def probe(path: Path) -> MediaInfo | None:
    """영상 스트림이 없거나 읽을 수 없으면 None."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return None
            video = container.streams.video[0]
            fps = float(video.average_rate or video.base_rate)
            if video.frames:
                frames = video.frames
            elif video.duration:
                frames = round(float(video.duration * video.time_base) * fps)
            else:
                frames = round(container.duration / 1_000_000 * fps)
            ctx = video.codec_context
            return MediaInfo(frames, fps, ctx.width, ctx.height, bool(container.streams.audio))
    except (av.FFmpegError, OSError):
        return None


def media_date(path: Path) -> datetime | None:
    """찍은 날짜(현지 시각): 파일 안 creation_time(휴대폰 영상), 없으면 파일 수정 시각(녹화본은 녹화를 마친 때)."""
    try:
        with av.open(str(path)) as container:
            tag = container.metadata.get("creation_time") or next(
                (s.metadata.get("creation_time") for s in container.streams if s.metadata.get("creation_time")), None
            )
        if tag:
            return datetime.fromisoformat(tag.replace("Z", "+00:00")).astimezone().replace(tzinfo=None)
    except (av.FFmpegError, OSError, ValueError):
        pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def decode_audio(path: Path) -> np.ndarray:
    """원본 오디오 전체 → 48kHz 모노 int16. 오디오가 없으면 빈 배열."""
    chunks = []
    resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_RATE)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.zeros(0, np.int16)
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
    chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    return np.concatenate(chunks) if chunks else np.zeros(0, np.int16)


def compute_peaks(audio: np.ndarray) -> np.ndarray:
    """10ms 구간마다 최대 진폭 (0.0~1.0)."""
    n = len(audio) // PEAK_WINDOW
    if n == 0:
        return np.zeros(0, np.float32)
    blocks = np.abs(audio[: n * PEAK_WINDOW].astype(np.int32)).reshape(n, PEAK_WINDOW)
    return (blocks.max(axis=1) / 32768).astype(np.float32)


def fit_rgb(frame: av.VideoFrame, width: int, height: int) -> np.ndarray:
    """비율을 지켜 (width, height) 안 가운데에 놓고 남는 곳은 검게 (세로 영상을 가로 칸에 넣을 때 늘이지 않는다)."""
    scale = min(width / frame.width, height / frame.height)
    fw, fh = max(2, round(frame.width * scale) // 2 * 2), max(2, round(frame.height * scale) // 2 * 2)
    canvas = np.zeros((height, width, 3), np.uint8)
    y0, x0 = (height - fh) // 2, (width - fw) // 2
    canvas[y0 : y0 + fh, x0 : x0 + fw] = to_rgb(frame, fw, fh)  # 정확한 반올림 (services.yuv)
    return canvas


def compute_thumbnails(path: Path) -> Thumbnails:
    """키프레임만 풀어서 빠르게 만든다 (우리 녹화본은 1초마다 키프레임)."""
    times, images = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        for frame in container.decode(stream):
            yield_to_playback()  # 재생 중이면 잠깐 멈춘다 (영상 전체를 풀어서 무겁다)
            times.append(float(frame.pts * stream.time_base))
            images.append(fit_rgb(frame, *THUMB_SIZE))
    if not images:
        return Thumbnails(np.zeros(0), np.zeros((0, THUMB_SIZE[1], THUMB_SIZE[0], 3), np.uint8))
    return Thumbnails(np.array(times), np.stack(images))


def cache_name(path: Path) -> str:
    """캐시 파일 이름: 원본 이름 + 경로·크기·수정 시각 해시 → 원본이 바뀌면 자동으로 새로 만든다."""
    stat = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    return f"{path.stem}_{key}"


def optimized_path(cache_dir: Path, path: Path) -> Path:
    """원본 path의 편집용 사본 위치 (프로젝트 cache/optimized/)."""
    return cache_dir / OPTIMIZED_DIR / f"{cache_name(path)}{OPTIMIZED_TAG}.mp4"


class SourceCache(QObject):
    ready = Signal(str)  # 이 원본(경로 문자열)의 파형이나 썸네일이 준비됨

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cache_dir: Path | None = None
        self._lock = threading.Lock()
        self._path_locks: dict[str, threading.Lock] = {}
        self._info: dict[str, MediaInfo | None] = {}
        self._audio: dict[str, np.ndarray] = {}
        self._peaks: dict[str, np.ndarray] = {}
        self._thumbs: dict[str, Thumbnails] = {}
        self._scheduled: set[tuple[str, str]] = set()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sources")

    def set_cache_dir(self, cache_dir: Path) -> None:
        """프로젝트가 바뀌면 호출. 메모리 캐시는 경로 기준이라 그대로 둔다."""
        self._cache_dir = cache_dir

    def media(self, path: Path) -> Path:
        """실제로 읽을 파일: 편집용 사본이 있으면 사본, 없으면 원본."""
        if self._cache_dir is None or not path.exists():
            return path
        optimized = optimized_path(self._cache_dir, path)
        return optimized if optimized.exists() else path

    def info(self, path: Path) -> MediaInfo | None:
        path = self.media(path)
        key = str(path)
        with self._lock:
            if key in self._info:
                return self._info[key]
        info = probe(path)
        with self._lock:
            self._info[key] = info
        return info

    def audio(self, path: Path) -> np.ndarray:
        """기다려서 받는다 (렌더 스레드용). 처음엔 수 초 걸릴 수 있다."""
        path = self.media(path)
        key = str(path)
        with self._path_lock(key):
            if key not in self._audio:
                self._audio[key] = self._load_or_compute(path, "audio.npy", lambda: decode_audio(path))
            return self._audio[key]

    def peaks(self, path: Path) -> np.ndarray | None:
        path = self.media(path)
        return self._get_or_schedule(path, "peaks", self._peaks, lambda: compute_peaks(self.audio(path)))

    def thumbnails(self, path: Path) -> Thumbnails | None:
        path = self.media(path)
        return self._get_or_schedule(path, "thumbs", self._thumbs, lambda: self._load_thumbnails(path))

    def _get_or_schedule(self, path: Path, kind: str, store: dict, compute):
        key = str(path)
        with self._lock:
            if key in store:
                return store[key]
            if (kind, key) in self._scheduled or not path.exists():
                return None
            self._scheduled.add((kind, key))

        def job() -> None:
            try:
                value = self._load_or_compute(path, f"{kind}.npy", compute) if kind == "peaks" else compute()
            except Exception:
                log.exception("%s 만들기 실패: %s", kind, path)
                return
            with self._lock:
                store[key] = value
            self.ready.emit(key)

        self._pool.submit(job)
        return None

    def _load_thumbnails(self, path: Path) -> Thumbnails:
        cache = self._cache_file(path, "thumbs2.npz")  # 2: 비율을 지키는 썸네일 (예전 것은 늘어나 있다)
        if cache is not None and cache.exists():
            data = np.load(cache)
            return Thumbnails(data["times"], data["images"])
        thumbs = compute_thumbnails(path)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, times=thumbs.times, images=thumbs.images)
        return thumbs

    def _load_or_compute(self, path: Path, suffix: str, compute) -> np.ndarray:
        cache = self._cache_file(path, suffix)
        if cache is not None and cache.exists():
            return np.load(cache)
        value = compute()
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, value)
        return value

    def _cache_file(self, path: Path, suffix: str) -> Path | None:
        if self._cache_dir is None or not path.exists():
            return None
        return self._cache_dir / f"{cache_name(path)}.{suffix}"

    def _path_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._path_locks.setdefault(key, threading.Lock())
