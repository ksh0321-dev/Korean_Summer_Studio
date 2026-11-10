"""하이라이트 재료 뽑기: 영상 하나 → core.highlight.Signals (소리 세기 · 움직임 · 표정). 결과는 캐시에 남긴다.

- 소리는 원본 오디오 전체를 0.1초씩 (몇 초면 끝).
- 영상은 처음부터 차례로 풀면서 1초에 5장만 쓴다: 움직임은 160×90 회색, 표정은 긴 변 1280 이하 RGB로 얼굴 106점.
- AI 패키지·얼굴 모델이 없으면 표정은 건너뛴다 (소리·움직임만으로 점수).
- 캐시: cache/<원본 키>.highlight1.npz — 원본이 바뀌면 키가 바뀌어 새로 분석하고, 예전 것은 프로젝트 정리가 지운다.
  표정 없이 분석했던 것은 나중에 얼굴 모델이 생기면 다시 분석한다.
"""

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

import av
import numpy as np
from PySide6.QtCore import QObject, Signal

from ks_studio.core.beauty import frontal_weight, size_weight
from ks_studio.core.highlight import AUDIO_STEP, FACE_COLUMNS, SAMPLE_STEP, SCENE_CUT, Signals, expression
from ks_studio.services.foreground import yield_to_playback
from ks_studio.services.sources import AUDIO_RATE, SourceCache, cache_name, decode_audio

log = logging.getLogger(__name__)

CACHE_SUFFIX = "highlight1.npz"  # 재료 뽑는 법을 바꾸면 숫자를 올린다 (예전 캐시는 정리 때 지워진다)
MOTION_SIZE = (160, 90)
FACE_LONG_SIDE = 1280  # 얼굴 찾기는 어차피 640으로 줄여서 본다 — 크게 풀 필요가 없다


class Cancelled(Exception):
    """분석을 멈췄다."""


def cache_file(cache_dir: Path, media: Path) -> Path:
    return cache_dir / f"{cache_name(media)}.{CACHE_SUFFIX}"


def loudness(audio: np.ndarray) -> np.ndarray:
    """48kHz 모노 int16 → 0.1초마다 세기 (dB, 가장 큰 소리가 0)."""
    step = round(AUDIO_RATE * AUDIO_STEP)
    n = len(audio) // step
    if n == 0:
        return np.zeros(0, np.float32)
    blocks = audio[: n * step].astype(np.float32).reshape(n, step) / 32768
    rms = np.sqrt(np.mean(blocks**2, axis=1))
    return (20 * np.log10(np.maximum(rms, 1e-5))).astype(np.float32)


def _face_row(detector, rgb: np.ndarray) -> list[float]:
    faces = detector.faces(rgb)
    if not faces:
        return [np.nan] * len(FACE_COLUMNS)
    face = max(faces, key=lambda f: f.area)  # 화면에서 가장 큰 얼굴 = 주인공으로 본다
    lm = face.landmarks
    presence = frontal_weight(lm) * size_weight(lm, rgb.shape[1])
    return [*expression(lm), presence]


def _video_signals(path: Path, detector, on_progress, cancel) -> tuple[float, np.ndarray, np.ndarray | None]:
    """(길이, 움직임, 표정) — SAMPLE_STEP마다."""
    motion, faces = [], ([] if detector is not None else None)
    duration = 0.0
    with av.open(str(path)) as container:
        if not container.streams.video:
            return float(container.duration or 0) / 1_000_000, np.zeros(0, np.float32), None
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = float(stream.duration * stream.time_base) if stream.duration else float(container.duration or 0) / 1_000_000
        width, height = stream.codec_context.width, stream.codec_context.height
        scale = min(1.0, FACE_LONG_SIDE / max(width, height, 1))
        face_size = (max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2))
        next_time, previous = 0.0, None
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            duration = max(duration, frame.time)
            if frame.time + 1e-6 < next_time:
                continue
            slots = 0
            while next_time <= frame.time + 1e-6:  # 프레임이 듬성듬성하면(가변 fps) 빈 칸은 같은 장으로 채운다
                next_time += SAMPLE_STEP
                slots += 1
            yield_to_playback()  # 재생 중이면 잠깐 멈춘다
            if cancel is not None and cancel():
                raise Cancelled
            gray = frame.reformat(*MOTION_SIZE, format="gray").to_ndarray().astype(np.int16)
            if previous is None:
                moved = np.nan
            else:
                diff = float(np.abs(gray - previous).mean()) / 255
                moved = np.nan if diff > SCENE_CUT else diff  # 장면 전환은 움직임이 아니다
            previous = gray
            motion += [moved] * slots
            if faces is not None:
                faces += [_face_row(detector, frame.reformat(*face_size, format="rgb24").to_ndarray())] * slots
            if on_progress and total > 0 and len(motion) % 10 == 0:
                on_progress(min(1.0, frame.time / total))
        if duration > 0:
            duration += 1 / float(stream.average_rate or 30)  # 마지막 프레임이 보이는 시간까지
    face = np.array(faces, np.float32).reshape(-1, len(FACE_COLUMNS)) if faces is not None else None
    return duration, np.array(motion, np.float32), face


def analyze(path: Path, detector=None, audio: np.ndarray | None = None,
            on_progress: Callable[[float], None] | None = None, cancel: Callable[[], bool] | None = None) -> Signals:
    """영상 하나의 재료. detector(services.face.FaceDetector)가 없으면 표정은 빼고. audio를 주면 다시 풀지 않는다.

    on_progress(0~1). cancel()이 True를 돌려주면 Cancelled.
    """
    if audio is None:
        audio = decode_audio(path)
    loud = loudness(audio)
    if on_progress:
        on_progress(0.0)
    duration, motion, face = _video_signals(path, detector, on_progress, cancel)
    if duration <= 0:
        duration = len(audio) / AUDIO_RATE
    return Signals(duration, loud, motion, face)


def save(path: Path, signals: Signals) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name.removesuffix(".npz") + ".part.npz")
    extra = {} if signals.face is None else {"face": signals.face}
    np.savez_compressed(tmp, duration=np.float64(signals.duration), loudness=signals.loudness, motion=signals.motion, **extra)
    os.replace(tmp, path)


def load(path: Path) -> Signals | None:
    try:
        with np.load(path) as data:
            face = data["face"] if "face" in data.files else None
            return Signals(float(data["duration"]), data["loudness"], data["motion"], face)
    except (OSError, ValueError, KeyError) as e:
        log.warning("하이라이트 캐시를 못 읽음 (다시 분석) %s: %s", path, e)
        return None


def cached(cache_dir: Path, media: Path, want_face: bool) -> Signals | None:
    """캐시에 쓸 만한 결과가 있으면. 표정을 원하는데 표정 없이 분석한 것이면 None (다시 분석)."""
    file = cache_file(cache_dir, media)
    if not file.exists():
        return None
    signals = load(file)
    if signals is None or (want_face and signals.face is None):
        return None
    return signals


def load_or_analyze(cache_dir: Path, media: Path, detector=None, audio: np.ndarray | None = None,
                    on_progress=None, cancel=None) -> Signals:
    signals = cached(cache_dir, media, detector is not None)
    if signals is not None:
        return signals
    signals = analyze(media, detector, audio, on_progress, cancel)
    try:
        save(cache_file(cache_dir, media), signals)
    except OSError as e:
        log.warning("하이라이트 캐시를 못 씀 %s: %s", media, e)
    return signals


def faces_available() -> bool:
    """표정을 볼 수 있나 (AI 패키지와 얼굴 모델이 있다). 없으면 소리·움직임만으로."""
    from ks_studio.services.face import ai_unavailable_reason, face_models_ready

    return ai_unavailable_reason() is None and face_models_ready()


def face_detector_or_none():
    """표정을 볼 수 있으면 얼굴 찾기, 없으면 None."""
    if not faces_available():
        return None
    from ks_studio.services.face import FaceDetector

    return FaceDetector()


class HighlightWorker(QObject):
    """V1 원본들을 뒤에서 차례로 분석한다 (캐시가 있으면 바로). 한 번에 한 작업.

    얼굴 찾기 모델은 처음 분석할 때 한 번 열어 두고 다시 쓴다 (여는 데 몇 초).
    """

    progress = Signal(int, int, float, str)  # 몇 번째(1부터), 전체 수, 그 영상 진행 0~1, 영상 이름
    done = Signal(object)  # {Clip.source: Signals}
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, sources: SourceCache, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._sources = sources
        self._detector = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cache_dir: Path, jobs: list[tuple[str, Path]]) -> None:
        """jobs = [(Clip.source, 원본 경로)]. 읽는 파일은 편집용 사본이 있으면 사본."""
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(cache_dir, list(jobs)), daemon=True, name="highlight")
        self._thread.start()

    def cancel(self) -> None:
        self._stop.set()

    def _run(self, cache_dir: Path, jobs: list[tuple[str, Path]]) -> None:
        found: dict = {}
        want_face = faces_available()
        try:
            for number, (key, path) in enumerate(jobs, 1):
                media = self._sources.media(path)
                self.progress.emit(number, len(jobs), 0.0, path.name)
                signals = cached(cache_dir, media, want_face)
                if signals is None:
                    if want_face and self._detector is None:
                        self._detector = face_detector_or_none()
                    report = lambda r, n=number, name=path.name: self.progress.emit(n, len(jobs), r, name)  # noqa: E731
                    signals = analyze(media, self._detector if want_face else None, self._sources.audio(path),
                                      report, self._stop.is_set)
                    try:
                        save(cache_file(cache_dir, media), signals)
                    except OSError as e:
                        log.warning("하이라이트 캐시를 못 씀 %s: %s", media, e)
                found[key] = signals
        except Cancelled:
            self.cancelled.emit()
            return
        except Exception as e:  # 깨진 영상 등 — 앱은 계속
            log.exception("하이라이트 분석 실패")
            self.failed.emit(str(e) or type(e).__name__)
            return
        self.done.emit(found)
